import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter

# 基础模块：用于构建PointSequential等容器
class PointModule(nn.Module):
    r"""PointModule
    所有点云模块的基类，占位定义，用于 PointSequential 中串联调用。
    """
    def __init__(self):
        super().__init__()

class PointSequential(PointModule):
    r"""A sequential container for PointModules.
    Modules will be added in the order of construction args or via add().
    """
    def __init__(self, *modules, **named_modules):
        super().__init__()
        self.layers = nn.ModuleList()
        # 支持通过位置参数或命名参数添加子模块
        for module in modules:
            self.add(module)
        for name, module in named_modules.items():
            self.add(module, name=name)

    def add(self, module, name=None):
        # 添加子模块到顺序容器
        if name is not None:
            setattr(self, name, module)
        else:
            # 若未提供名称，用索引作为属性名
            setattr(self, str(len(self.layers)), module)
        self.layers.append(module)

    def forward(self, input):
        # 顺序执行所有子模块（适配点云和spconv模块）
        for module in self.layers:
            if isinstance(module, PointModule):
                input = module(input)
            # 如果是spconv模块
            elif spconv.modules.is_spconv_module(module):
                # 将Point中的稀疏tensor取出进行卷积
                input.sparse_conv_feat = module(input.sparse_conv_feat)
            else:
                # 普通nn.Module（非PointModule），直接作用于 point.feat 特征
                input.feat = module(input.feat)
        return input

# 带序列化的注意力模块
class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        patch_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.order_index = order_index
        # 使用 PyTorch 实现多头自注意力
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        # 是否使用Flash Attention（需要FlashAttn库支持），此处保留接口
        self.enable_flash = enable_flash
        self.enable_rpe = enable_rpe
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        # 相对位置编码（若 enable_rpe=True 时使用），此处略

    def get_padding_and_inverse(self, point):
        # 计算注意力所需的序列padding信息
        # 通过序列化后的顺序(Order)获取padding序列和inverse映射，用于多头注意力计算对齐
        code_lengths = point["serialized_depth"]  # scalar, depth of serialization
        # patch_size为单次注意力序列长度上限
        max_seq_len = self.patch_size
        # 计算每个注意力块有效序列长度（<=patch_size）
        seq_lens = torch.clamp(point["num_points_per_cluster"], max=max_seq_len)
        # 构造prefix sum获取padding index
        prefix_sum = torch.cat([torch.zeros(1, device=seq_lens.device, dtype=torch.long), torch.cumsum(seq_lens, 0)])
        pad = torch.arange(max_seq_len, device=seq_lens.device).repeat(seq_lens.shape[0], 1)
        pad = pad < seq_lens.unsqueeze(1)  # mask shape [num_clusters, max_seq_len]
        pad = pad.flatten()
        # 计算每点在注意力计算中的排序（order）和逆序索引
        inverse = point["serialized_inverse"][self.order_index]
        return pad, inverse, prefix_sum

    def forward(self, point):
        # 获取patch大小K和头数H
        K = self.patch_size
        H = self.num_heads
        C = self.channels
        # 准备序列化后的point特征
        pad, inverse, _ = self.get_padding_and_inverse(point)
        order = point["serialized_order"][self.order_index][pad]  # 经过pad的排序索引
        inverse_idx = inverse  # 用于还原顺序的索引映射

        # 提取并重排QKV
        qkv = self.qkv(point.feat)[order]  # 选择有效点并重排顺序
        if not self.enable_flash:
            # 传统注意力计算: reshape并split Q,K,V
            q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            if self.upcast_attention:
                q = q.float(); k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            # 使用Flash Attention加速
            # 将qkv打包为FlashAttn所需形状并调用
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                point.cu_seqlens,
                max_seqlen=K,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        # 恢复原始顺序
        feat = feat[inverse_idx]
        # 输出经过线性投影和dropout的结果
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point

# 简单两层MLP，用于PointModule
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# Transformer Block：包含序列化注意力和FFN
class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # 局部CPE卷积增强（Conv + Linear + Norm）
        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.norm1 = PointSequential(norm_layer(channels))
        self.ls1 = PointSequential(LayerScale(channels, init_values=layer_scale) if layer_scale is not None else nn.Identity())
        self.attn = SerializedAttention(
            channels=channels, patch_size=patch_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop,
            order_index=order_index, enable_rpe=enable_rpe, enable_flash=enable_flash,
            upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.ls2 = PointSequential(LayerScale(channels, init_values=layer_scale) if layer_scale is not None else nn.Identity())
        self.mlp = PointSequential(
            MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio), out_channels=channels, act_layer=act_layer, drop=proj_drop)
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point):
        # 残差连接 + 前置归一化 + 多头注意力
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.ls1(self.attn(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)
        # FFN部分
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.ls2(self.mlp(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        # 更新SparseConvTensor的特征以供后续spconv层使用
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

# 三维格网池化层：将点划分网格并聚合
class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        # 降采样前对特征进行线性投影减少信息损失
        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point):
        # 计算当前点的grid坐标
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError("Point must contain 'grid_coord' or ('coord' and 'grid_size')")
        # 按stride缩小坐标，实现体素聚类
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        # 将batch索引融合进grid_coord高位（使用位操作避免不同batch点混淆）
        grid_coord = grid_coord | (point.batch.view(-1, 1) << 48)
        # 聚类：按体素合并点
        grid_coord, cluster, counts = torch.unique(grid_coord, sorted=True, return_inverse=True, return_counts=True, dim=0)
        # cluster为每个点对应的体素索引序列，counts为每个体素包含点数
        # 对cluster排序以获取每个体素对应的point索引范围
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]  # 每个聚类在排序后序列中的起始索引
        # 对每个体素聚合特征和坐标（根据reduce规则）
        feat_agg = torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce)
        coord_agg = torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean")
        batch_agg = point.batch[head_indices]
        # 将结果组装为字典，创建新的Point对象
        point_dict = {"feat": feat_agg, "coord": coord_agg, "grid_coord": grid_coord, "batch": batch_agg}
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(point.origin_coord[indices], idx_ptr, reduce="mean")
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(point.color[indices], idx_ptr, reduce="mean")
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride
        # 记录父子关系以支持反向Unpool（traceable模式）
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        # 构建新的Point对象
        new_point = Point(point_dict)
        # 应用Norm和激活（如果定义）
        if hasattr(self, "norm"):
            new_point = self.norm(new_point)
        if hasattr(self, "act"):
            new_point = self.act(new_point)
        # 对新点集进行序列化和稀疏化准备，以便后续Block使用
        new_point.serialization(order=new_point.order, shuffle_orders=self.shuffle_orders)
        new_point.sparsify()
        return new_point

class GridUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=None, act_layer=None, traceable=False):
        super().__init__()
        # 上采样层：将低层特征与skip连接并恢复到高分辨率
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels)); self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer()); self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys() and "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")  # 获取池化前的Point
        inverse = point.pooling_inverse      # 获取对应索引
        feat = point.feat
        # 将下层特征投影并累加到上层skip特征
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
        return parent

# 初始特征嵌入模块：将输入原始点特征投影至嵌入空间
class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None, mask_token=False):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")
        # mask_token 用于掩盖无效点，可选实现（目前未启用）
        # if mask_token:
        #     self.mask_token = nn.Parameter(torch.zeros(1, embed_channels))
        # else:
        #     self.mask_token = None

    def forward(self, point):
        point = self.stem(point)
        # 如果需要mask无效点特征，可以在此处使用mask_token替换，但默认不启用
        # if "mask" in point.keys():
        #     point.feat = torch.where(point.mask.unsqueeze(-1), self.mask_token.to(point.feat.dtype), point.feat)
        return point

# Fourier编码辅助函数：用于可选的位置编码（enable_fourier_encode）
def fourier_encode_vector(vec, num_bands=10, sample_rate=60):
    """Fourier encode a vector of shape [N, D] to [N, (2*num_bands+1)*D]."""
    n, d = vec.shape
    # 采样频率
    samples = torch.linspace(1, sample_rate / 2, num_bands, device=vec.device) * torch.pi
    sines = torch.sin(samples[None, :, None] * vec[:, None, :])
    cosines = torch.cos(samples[None, :, None] * vec[:, None, :])
    encoding = torch.stack([sines, cosines], dim=2).reshape(n, 2 * num_bands, d)
    encoding = torch.cat([vec[:, None, :], encoding], dim=1)
    return encoding.flatten(1)

# Sonata 主模块：点云编码网络（编码模式）
class Sonata(PointModule, nn.Module):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        mask_token=False,
        enc_mode=True,
        enable_fourier_encode=False,
        num_bins=1280,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders

        # 验证各stage参数长度一致
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths) == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)

        ln_layer = nn.LayerNorm  # 层归一化
        act_layer = nn.GELU      # 激活函数

        # 输入嵌入层
        self.embedding = Embedding(in_channels=in_channels, embed_channels=enc_channels[0],
                                   norm_layer=ln_layer, act_layer=act_layer, mask_token=mask_token)

        # 编码器层级组合
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            stage_drop_paths = enc_drop_path[sum(enc_depths[:s]): sum(enc_depths[: s + 1])]
            stage_seq = PointSequential()
            if s > 0:
                stage_seq.add(GridPooling(in_channels=enc_channels[s-1], out_channels=enc_channels[s], stride=stride[s-1],
                                           norm_layer=ln_layer, act_layer=act_layer), name="down")
            for i in range(enc_depths[s]):
                stage_seq.add(Block(channels=enc_channels[s], num_heads=enc_num_head[s],
                                    patch_size=enc_patch_size[s], mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=proj_drop,
                                    drop_path=stage_drop_paths[i] if i < len(stage_drop_paths) else 0.0,
                                    layer_scale=layer_scale, norm_layer=ln_layer, act_layer=act_layer,
                                    pre_norm=pre_norm, order_index=i % len(self.order),
                                    cpe_indice_key=f"stage{s}", enable_rpe=enable_rpe, enable_flash=enable_flash,
                                    upcast_attention=upcast_attention, upcast_softmax=upcast_softmax),
                            name=f"block{i}")
            if len(stage_seq.layers) != 0:
                self.enc.add(module=stage_seq, name=f"enc{s}")

        # 可选：增加傅里叶坐标编码
        self.enable_fourier_encode = enable_fourier_encode
        down_convs = len(enc_channels) - 1  # 总下采样次数
        res_reduction = 2 ** down_convs    # 网格分辨率缩减倍数
        self.reduced_grid_size = int(num_bins / res_reduction)
        self.input_proj = nn.Linear(enc_channels[-1] + 63, enc_channels[-1]) if self.enable_fourier_encode else None
        # 63 =  (2*num_bands+1)*3, 对应3D坐标的固定傅里叶编码长度

    def forward(self, data_dict):
        # 将输入数据(dict)包装为Point对象
        point = Point(data_dict)
        # 1. 初始嵌入
        point = self.embedding(point)
        # 2. 点云坐标序列化和稀疏张量初始化
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        # 3. 逐层编码
        point = self.enc(point)
        # 提取最终层输出的稀疏卷积特征作为上下文
        context = point["sparse_conv_feat"].features  # shape [N_points_final, C_final]
        # 如启用傅里叶编码，则将归一化坐标的编码拼接后再线性投影
        if self.enable_fourier_encode:
            coords = point["grid_coord"]        # final voxel coords
            coords_normalised = coords / (self.reduced_grid_size - 1)
            encoded_coords = fourier_encode_vector(coords_normalised)
            context = torch.cat([context, encoded_coords], dim=-1)
            context = self.input_proj(context)
        return context

# 附属模块：用于TinyDropPath
class DropPath(nn.Module):
    """Droppath操作: 以概率p将整条路径上的输出随机置零。"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        # 产生随机mask
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# LayerScale层: 学习每层残差权重的可训练系数
class LayerScale(nn.Module):
    def __init__(self, channels, init_values=1e-5):
        super().__init__()
        init_values = init_values if init_values is not None else 0.0
        self.gamma = nn.Parameter(init_values * torch.ones(channels))

    def forward(self, x):
        return x * self.gamma

# Point数据类：封装点及属性，支持字典like访问
class Point(dict):
    """Wrap dictionary to allow attribute access and custom methods for point data."""
    def __init__(self, data):
        super().__init__(data)
        # 将自身也作为属性容器
        for k, v in data.items():
            self[k] = v
    def __getattr__(self, item):
        # 使得可以通过属性访问字典项
        if item in self:
            return self[item]
        raise AttributeError(f"'Point' object has no attribute '{item}'")
    def __setattr__(self, key, value):
        self[key] = value
