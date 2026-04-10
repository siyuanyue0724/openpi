
# PICF-JEPA Core v0.4.8 完整方法说明（Camera-Conditioned Geometry Revision, MOVEON）

**Posterior-Innovation Contact Field JEPA — Unified Token-Field, Predictive-Consensus, Single-Stage End-to-End Revision with Real-Signal Future Heads, Global Predictive-Innovation Module, Soft Anchor Lifecycle, and Language-Late Conditioning**

**Version status**
- Spec version: `v0.4.8`
- Status: `MOVEON`
- Relative to `v0.4.7`: closes remaining mixed-fragment regressions and completes notation around visual reference frames, null projection branches, predictive-token pooling, support-gated routing, and sparse projective supervision

---

主线：

**unified multimodal token field → current observation anchors → persistent posterior anchors → global posterior self-attention → global three-head future prediction → explicit innovation token → downstream readout**

---

这里最关键的澄清有六条：

- **V-JEPA 2 / 2.1 本身不吃语言。**
  HF `VJEPA2Model` 的 `forward` 接口只接收视频张量、mask 与 `skip_predictor` 等参数；
  `skip_predictor=True` 代表只取 encoder outputs，而不是语言条件预测器。
  因此，本版不把语言直接塞进 V-JEPA encoder，而是把语言仅放在 **posterior 之后的 global predictive heads 与 downstream selector** 中使用。

- **本版仍然不在运行时做显式 imagination rollout 作为 core 必需项。**
  训练时会让当前 step 的 global posterior state 预测下一时刻的目标；
  但推理时只缓存上一步对当前时刻的预测，
  再把其与当前真实观测形成 innovation，
  不做长链递归 frame generation 作为在线必需路径。

- **本版坚持“真实未来信号优先”。**
  触觉头与点云头默认预测真实未来信号，
  而不是只预测 latent。
  视觉头默认采用双头：
  一头预测下一时刻的 V-JEPA latent target，
  另一头可选预测轻量真实视觉目标。
  这样可以保留视觉 predictive embedding 的优势，
  同时降低“只在 latent 上自回归，最终丢失真实可复现能力”的风险。

- **当前物理 posterior 仍是 language-free。**
  当前物理 belief state 由上一时刻 prior 与当前多模态证据共同决定；
  语言不进入当前 posterior 的 measurement construction 与 fusion。
  语言只影响：
  1) global future heads 对“该期待什么”的预测，
  2) downstream 对“该关注哪些 posterior / innovation 成分”的选择。

- **point-visual 的主力不再是强 `L_{pv}`，而是 camera-conditioned projection / ray geometry。**
  `L_{pv}` 的角色应从“主要对齐机制”降级为弱辅助；
  当前 point-led anchor 架构更适合把几何关系尽早写进 token 与 attention，
  而不是只在训练末端用 embedding matching 逼模型自己隐式学出投影关系。
  当前代码已经落地 token-level `PE_{proj}` / `PE_{ray}`，
  并把 `FusionTransformer` 升级成支持 point↔visual relative projective bias 与 attention-derived focus supervision 的版本。

- **innovation 必须显式构造，而不能从 LSTM forget gate 之类的内部门值里猜。**
  本版中的 innovation 是：
  “上一时刻对当前时刻的预测” 与 “当前真实观测目标”
  的显式差异，
  再被编码成 innovation token。
  Forget gate / input gate 仅是内部控制量，
  不是可校准、可监督、可解释的 prediction error。

因此，本稿的正确理解应为：

**current physical posterior is language-free; all current-sensor tokens enter one unified hidden space; point-visual coupling is geometry-first through camera-conditioned projection/ray conditioning; global posterior self-attention forms a predictive state; language enters only after current posterior is fixed; future prediction uses real-signal heads for tactile and point-cloud targets by default; explicit innovation is formed by comparing previous-step prediction with current-step true targets and is then used together with posterior for control.**

---

# 0. 总述

PICF-JEPA Core v0.4.8 的目标不是在 v0.3.11 的
“point-addressed expert + local query adapter + staged training”
框架上继续局部修补，
而是把系统重写成一条**整体一致、单阶段端到端训练**的主线。

这条主线有四个统一原理：

1. **统一 token 化**
   视觉、点云、触觉、proprio / action context
   先全部 token 化，
   再进入同一隐空间并做共享自注意力。

2. **双层 anchor 机制**
   当前帧通过 observation anchors
   形成“这一帧世界里有哪些概念”的表征；
   跨时间通过 persistent posterior anchors
   维护连续的世界 belief state。

3. **Gaussian posterior 继续保留，但不再由多个硬 expert 链条拼装。**
   每个 persistent anchor 从统一 token 场读出当前证据，
   由一组小型 vote heads 形成 measurement proposals，
   再和 carried prior 通过 information-form Gaussian fusion
   形成当前 posterior。
   Gaussian 在本版里承担：
   - uncertainty calibration
   - continuous voting
   - predictive comparison
   三重角色。

4. **global predictive-innovation module**
   当前 posterior anchors 在 self-attention 后形成全局 posterior context；
   训练时，
   从这个 global context 接三个 future heads，
   直接预测下一时刻的重要目标；
   推理时，
   上一步缓存的预测与当前真实目标比较形成 innovation token，
   innovation token 与当前 posterior tokens 共同进入 action head。

本版的核心收益是：

- 保留 v0.3 系列“current posterior language-free”的物理洁净性；
- 保留显式几何状态 `(x,S,a)`；
- 用统一 token 场替代“很多小模块各管一个模态”的碎片式结构；
- 用 global posterior 统一承载未来预期，
  避免 future heads 被拆成一堆局部小分支；
- 把 innovation 从“日志概念”升级为 action-consumable token；
- 默认单阶段端到端训练，
  不再依赖 Stage 0 / 1 / 1.25 / 1.75 的多段 curriculum；
- 对 tactile 与 point-cloud 采用真实信号监督，
  使模型在操作后仍保有对真实后果的可复现能力。
- 把 point↔visual 几何关系提前写进 token / attention，
  降低 purely matching-based 对齐在 projection ambiguity 与 depth-induced scale inconsistency 下的脆弱性。

## 0.0a 当前实现状态（2026-04-09 严格复核）

截至 2026-04-09，本版在当前仓库中的已部署项包括：

- token-level `PE_{proj}` / `PE_{ray}`
- patch-unit continuous patch-grid projective compatibility
- sparse radius-neighborhood `\mathcal E_t^{pv}` candidate mask
- point↔visual relative projective attention bias `b_{t,m,u}^{proj}`
- attention-derived `L_{focus}^{pv}`
- low-support routing support gates `\omega^{route}`
- point-tactile `L_{pt}`
- predictive-context pooling `\widetilde G_t^{pred}\rightarrow g_t^{pred}`
- carried-prior 中的 `\alpha_{t-1,i}` 输入签名
- trainer 侧 real `PaliGemma` language-late semantic side path
- `Sonata / V-JEPA 2.1 / AnyTouch2 / PaliGemma` 的 lower-lr cotrain 参数组

当前代码里的明确工程近似包括：

- `G_{t,m,u}^{proj}` 仍先做 dense compatibility，再与 sparse radius neighborhood 相交形成 `\mathcal E_t^{pv}`
- `L_{pt}` 当前用 tactile sensor world pose 到局部点集的最近点近似构造 point↔tactile 正例
- innovation residual 当前默认仍走 deterministic normalization，而不是 future-head covariance whitening

因此，本稿中的“已实现”口径以下述范围为准：

- `src/openpi/picf/core/`
- `scripts/picf_core_train_smoke.py`
- `src/openpi/picf/README.md`
- `docs/calvin_readme.txt`

当前这条实现口径还已在两套运行入口下复核：

- 直接 `.venv` Python
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync ...`

这轮 follow-through 还额外确认了两条关键不变量：

- 改 `semantic_override` 不会改变 current posterior：`posterior_mu_diff = 0.0`、`posterior_sigma_diff = 0.0`
- 训练侧 `action_future` teacher forcing 确实会改变 `g_t^{pred}`：同一帧上 `future_action_diff_teacher_vs_policy = 0.2605`
- 多卡 DDP + gradient accumulation 时，semantic side path 仍保持 language-late；
  当前工程实现会优先把 trainable `PaliGemma` 切到 non-reentrant gradient checkpointing，
  并在 `accum_steps>1` 时进一步保守地关闭 semantic gradient checkpointing，
  以规避 repeated backward 下的 `mark ready twice`，不改变本版数学定义
- 当前 trainer 里的 `PaliGemma` 侧路已保留完整 semantic token stream，
  不是单个 pooled semantic summary token
- 保留下来的语义 token 包括：
  - 文本有效 token
  - 图像有效 token
- 它们在 current posterior 固定之后，
  由 world/control stream 通过 posterior-late 异宽 cross-attention 读取
- 同时仍记录一个 `semantic_summary` 作为聚合诊断量，
  但它不直接参与 downstream 主融合

## 0.0b 推荐主路线与宽度审计（2026-04-10 严格复核）

这一节现在既记录**当前已落地实现**，
也记录基于当前仓库实现、`pi0.5` 本地代码、以及近期 VLA / VLM 融合论文做出的**推荐主路线**。

结论先写在前面：

- **主路线应继续向 `pi0.5 / knowledge-insulating` 靠拢**
- **但不应把 anchor 退化成普通 prompt token**
- **当前代码已经不再停留在“full semantic tokens → shared `hidden_dim=256`”的统一小隐空间方案**
- 对本版 PICF，更合理的长期结构是：
  - `PaliGemma` 保留高宽度 semantic stream
  - `current posterior` 继续保持 language-free
  - `anchor/posterior` 保留独立 world-state stream
  - 在 posterior 之后，用 world<-semantic 的多层 gated cross-attention 做深融合

当前已落地的默认参数是：

- `hidden_dim = 256`
- `posterior_hidden_dim = 256`
- `latent_dim = 112`
- `innovation_dim = 256`
- `control_dim = 256`
- `semantic_dim = 2048`
- `semantic_cross_dim = 512`
- `future_hidden_dim = 256`
- `predictive_semantic_reads = 2`
- `control_semantic_reads = 2`

当前训练脚本的可视化导出 contract 也已经固定：

- `diagnostic_interval = 500`
- rank0 每 `500` step 保存一个真实 CALVIN window 的诊断目录
- 目录里包含：
  - `gt_static_t*.png`
  - `pred_physical_t*.png`
  - `pred_semantic_t*.png`
  - `gt_window_static.gif`
  - `pred_physical_window_static.gif`
  - `pred_semantic_window_static.gif`
  - `compare_grid.png`
- 这些 `pred_*` 图像来自 PICF visual future head 的 `visual_real` 分支
- 当前 `visual_real_grid = 4`，因此它们是 **4x4 coarse RGB future prediction 的上采样诊断图**
- 它们用于检查：
  - physical future cache 是否在学当前 world stream 的短时预测
  - semantic-conditioned readout 是否只做条件调制而不是直接污染 posterior
- 它们**不等价于** CALVIN evaluator 的 policy rollout video

按当前实现直接统计，异宽融合块本身的参数量为：

- `control_fusion_params = 4,476,930`
- `predictive_fusion_params = 4,476,930`
- `total_fusion_params = 8,953,860`

这里的 `total_fusion_params` 已包含：

- world-side self-attention blocks
- world<-semantic gated cross-attention blocks
- 对应的 LayerNorm / FF / gate 参数

这说明：

- 让 semantic 保持 `2048` 并不等价于“把整个 PICF 全部抬到 `2048`”
- 当前实现只把 **semantic 读取接口** 做成宽流
- world-state 主干仍保持紧凑

这轮本地验证结果：

- `python -m py_compile src/openpi/picf/core/pipeline.py src/openpi/picf/core/pipeline_test.py scripts/picf_core_train.py scripts/picf_core_train_smoke.py`：通过
- `pytest -q src/openpi/picf/paligemma/wrapper_test.py src/openpi/picf/core/pipeline_test.py src/openpi/picf/core/training_test.py scripts/picf_core_train_test.py`：`60 passed`
- `python scripts/picf_core_train_smoke.py --calvin-root /tmp/openpi_picf_smoke_data/task_ABCD_D --device cpu`：通过
- CPU smoke 当前输出：
  - `loss_total = 0.8595`
  - `loss_visual_latent = 0.7113`
  - `loss_visual_real = 0.5978`
  - `loss_point_real = 0.7382`
  - `action_grad_norm = 4.6336`
  - `point_grad_norm = 0.1792`
- 新增红线回归：
  - 改 semantic 不改 `physical_prediction_cache`
  - 改 previous semantic 不改下一步 innovation
  - semantic future auxiliary loss 仍让 predictive cross-attn 保持在图中
- 2026-04-10 额外硬化：
  - `L_pt` 现在只在显式接触或 tactile history pseudo-contact gate 激活时产生正例
  - `Sonata` 中所有高风险高级索引路径都已补运行时 bounds check，若再出现非法索引，将优先报出明确的 Python `RuntimeError`，而不是只有模糊的 CUDA device assert
- 验证级别说明：
  - 以上结论来自代码路径审计、回归测试、CPU smoke、以及云机双卡 smoke
  - 它支持“当前工程实现满足此处定义的数学 contract”的判断
  - 但它**不是**机器校验的形式化证明；当前仓库没有 Coq / Lean / TLA+ / model-checking 工件
- 当前这条主线的更硬规格，已经收敛到：
  - [`PICF_FORMAL_CONTRACT.md`](/home/siyuanyue/Documents/openpi/PICF_FORMAL_CONTRACT.md)
  - 这份文件负责定义允许边、禁止边、状态转移顺序和验证范围
  - 一键验证脚本：
    - `python scripts/verify_picf_contract.py`

当前已落地的 downstream 融合口径是：

- `current posterior` 继续保持 language-free
- `PaliGemma` 保留 full text/image token stream，宽度保持 `2048`
- control 分支 world tokens:
  - `posterior.tokens`
  - `innovation_token`
  - `proprio_token`
- predictive 分支 world tokens:
  - `posterior.tokens`
  - `posterior.global_post`
  - `proprio_token`
  - `action_cond_token`
- control 分支先只在 world stream 内部做 self-attention，
  然后 world queries 再去 cross-attend `semantic_tokens`
- predictive 分支也先只在 world stream 内部做 self-attention，
  先导出：
  - `physical_global_pred`
  - `physical_prediction_cache`
- 只有在这份物理 cache 固定之后，
  predictive world stream 才再去 cross-attend `semantic_tokens`
  并导出：
  - `global_pred`
  - `prediction_cache`
- 下一步 innovation **只允许**读取 `previous.predictive.physical_prediction_cache`
- predictive semantic memory 默认还会使用：
  - `predictive_semantic_dropout_prob = 0.1`
  来压制 language shortcut
- 因而当前数学关系是：
  - semantic 与 anchor/posterior 在 downstream 平级协作
  - 但不要求所有流预先同宽
  - semantic 不写回 posterior cache / carried prior / physical innovation base

### (R1) 旧代码里为什么会出现 semantic 投影

这是**旧单宽实现**的历史原因，不是当前实现的要求。

旧版本 downstream 使用的是统一 `hidden_dim` 的 shared attention，
因此如果采用：

**`posterior.tokens || semantic.tokens || innovation || proprio` → shared self-attention**

那么进入该层的 token 必须先同宽。

所以旧实现里出现 `semantic -> hidden_dim` 投影，
根因是**旧 attention 结构要求统一 `d_model`**，
而不是方法论要求“语义必须被压成小宽度”。

### (R2) 当前实现已经不再做 `2048 → 256` 的 semantic 主压缩

当前代码里：

- `semantic_tokens` 保持 `semantic_dim = 2048`
- world/control stream 保持 `hidden_dim = 256`
- 融合通过 posterior-late 的异宽 gated cross-attention 完成
- `semantic_summary` 只保留为聚合记录与诊断量

因此，当前真正存在的宽度问题不再是：

**“semantic 会不会被先压到 256 再去融合？”**

而是：

**“world working space = 256` 是否已经足够承载 posterior / innovation / proprio / action-cond 的联合读写？”**

### (R3) 回到代码后，对 `hidden_dim = 256` 的判断

当前 `world-state` 流并不是普通 control token。
例如 posterior token 在投到 `hidden_dim` 前，实际携带的是：

- `h_post`
- `mu_post`
- `logvar_post`
- `geometry_pe(x, a, S)`
- `alpha`
- `contact_prob`

按当前默认配置，相关维数为：

- `_geometry_pe = 78`
- `posterior token_in = 560`
- `post_write in = 304`

也就是说，`256` 确实是压缩后的 world working space。

但在当前结构下，这个压缩是发生在 world tokenization 内部，
不是 semantic 主干被硬砍。

### (R4) 256 / 384 / 512 的当前 core 成本对比

按当前代码做的本地参数审计，已初始化 PICF core 参数量约为：

- `hidden_dim = 256`：`20,606,731`
- `hidden_dim = 384`：`40,072,843`
- `hidden_dim = 512`：`65,830,411`

几个关键模块的变化也很明显：

- `posterior_self`
  - `256`: `1,579,520`
  - `384`: `3,548,928`
  - `512`: `6,304,768`
- `predictive_world`
  - `256`: `1,579,520`
  - `384`: `3,548,928`
  - `512`: `6,304,768`
- `control_world`
  - `256`: `1,579,520`
  - `384`: `3,548,928`
  - `512`: `6,304,768`
- `predictive_semantic_reads`
  - `256`: `2,897,410`
  - `384`: `4,538,114`
  - `512`: `6,309,890`

所以：

- `384` 是合理扩容档
- 但不是“免费提升”
- `256 -> 384` 已接近 **2x**
- `256 -> 512` 已接近 **3.2x**

### (R5) 当前默认推荐

当前实现已经满足：

- `semantic_dim = 2048`
- `world hidden_dim = 256`
- posterior-late hetero cross-attention
- no writeback to posterior / carried prior / physical innovation base

在这个前提下，本版默认推荐是：

- **保留 `hidden_dim = 256` 作为当前默认**
- **保留 `semantic_dim = 2048`**
- **把 `384` 只作为后续 scaling / ablation 档**

原因是：

- 当前主结构风险已经不在“semantic 被压成 256”
- 当前 world token 数较小
- 现有红线测试与 CPU smoke 都已通过
- 而 `384 / 512` 会显著提高 core 成本

### (R6) 后续如果出现真实 bottleneck，再怎么升

若后续长训或评估表现出：

- posterior/world 读写容量不足
- innovation / uncertainty 表征过窄
- action / future head 对 world state 的条件化不充分

则优先级应为：

1. 先把 `hidden_dim` 从 `256` 升到 `384`
2. 再根据显存和收益决定是否升到 `512`
3. 不建议在没有证据时把 `384/512` 直接写成默认

### (R7) 与近期论文的对比性结论

当前推荐并不是“盲目追随 `pi0.5`”，
而是吸收下列路线的优点：

- `pi0.5 / knowledge insulating`
  - 强在保留 VLM backbone 的 semantic knowledge
- `Flamingo`
  - 强在把 cross-attention 插入主干内部，而不是只在顶层硬拼
- `DeepVision-VLA`
  - 强在深层持续注入视觉 expert 特征，而不是只做一次浅层融合
- `DreamVLA`
  - 强在 block-wise structured attention，可借来抑制 semantic / geometry / dynamic 之间的脏串扰

因此，本版未来不应退化成：

- 只有一个 pooled semantic token
- 或者简单 MLP 拼接到 action head
- 或者所有 token 强行压到很小的共享空间

而应走向：

**语义能力保留优先 + world-state posterior 独立 + 深层受控融合**

---

## 0.1 十条硬约束

### (H1) current posterior 是当前时刻唯一物理 belief state

当前时刻的物理状态只能由：

- carried / transported current prior
- 当前 unified token field 中读出的 measurement evidence

共同决定。

因此，下列量**不得**直接进入 current posterior 定义：

- language token / semantic token
- task prompt token
- global future prediction heads 的输出
- innovation token
- downstream action / progress token
- recycle / activity regularizer 的日志量

---

### (H2) 当前所有感知信息先统一 token 化，再进入共享隐空间；其中 point↔visual 默认走几何先验优先

本版要求下列输入全部 token 化：

- point / geometry tokens
- V-JEPA visual tokens
- AnyTouch tactile tokens
- proprio / action context tokens

它们先进入统一隐空间，
再通过共享 self-attention 形成当前感官场。
不得先把三个模态分成彼此隔离的大 expert 路线后再硬拼。

另外，对 point↔visual 关系，本版默认优先使用：

- calibrated projection / ray encoding
- camera-conditioned token conditioning
- optional relative projective attention bias

而不是把强 `L_{pv}` 当作主力机制。
`L_{pv}` 只允许作为弱辅助、warmup 或 anchor-consistency regularizer 存在。

---

### (H3) 语言不进入 current posterior，只进入 posterior 之后的 global predictive stage

语言来自 PaliGemma 或同类 semantic side encoder，
但只允许进入：

- global future prediction heads
- task relevance / selector
- downstream action / progress routing

语言不得进入：

- 当前 observation token field 的构造
- 当前 posterior anchor 的 evidence binding
- 当前几何 readout
- 当前 Gaussian fusion

---

### (H4) tactile 与 point-cloud future heads 默认预测真实信号

本版默认要求：

- tactile future head 预测真实未来 tactile signal
- point-cloud future head 预测真实未来几何/点云相关目标

其中“真实未来点云目标”可采用
深度 / occupancy / TSDF / range image / local scene flow / resampled point set
之一或其组合，
但不得默认退化成只预测 latent。

---

### (H5) 视觉头默认 dual-head

视觉 future prediction 默认至少包含：

- 一个 V-JEPA latent target head
- 一个可选轻量真实视觉 target head

这是因为：
V-JEPA latent target 提供强 predictive representation；
真实视觉 target 则约束模型不要在纯 latent 递归里逐渐失去真实可复现能力。

---

### (H6) innovation 必须显式定义

定义 innovation 的核心形式为：

```math
\epsilon_t = y_t - \hat y_t^{-}
```

其中：

- `\hat y_t^{-}` 是上一步对当前时刻的预测缓存
- `y_t` 是当前真实目标

innovation token 由显式残差编码得到。
不得把 LSTM forget gate / input gate 的门值
直接当作 innovation 本体。

---

### (H7) persistent anchor 的生命周期默认采用软机制，而非硬状态机

本版默认不再把 anchor 生命周期建立在：

- birth hard trigger
- retired hard state
- merge hard overwrite
- dormant / active / retired 多段状态机

之上。

替代为：

- soft activity
- soft competition
- dustbin / null channel
- recycle gate

这意味着 anchor 的形成、持续、衰退、回收
默认由连续量控制，
而不是由离散 if/else 控制。

---

### (H8) 除“真实缺失”和“安全限制”外，原则上不再引入新 hard gates

允许保留的 hard rule 只包括：

- 传感器真实缺失或时间戳失效时的 mask
- 行为输出的安全 clip / force / velocity limit

其余尽量软化，包括：

- contact importance
- modality weighting
- anchor activity
- recycle decision
- innovation weighting

---

### (H9) 默认训练范式为单阶段端到端

允许冻结大 backbone 或采用小学习率 / LoRA；
但本版默认不再采用多段式 stage curriculum。

所有主损失默认从一开始就在同一训练图中生效，
仅允许通过 warmup 调整权重，
而不允许按阶段切换训练图结构。

---

### (H10) 推理时不要求显式 rollout，但要使用上一步预测形成当前 innovation

运行时不做未来长链生成作为 core 必需项；
但必须缓存：

- 上一步全局 future heads 对当前时刻的预测

以便在当前时刻构造：

- innovation token
- uncertainty-aware surprise
- action-relevant discrepancy cue

---

## 0.2 本版与 v0.3.11 的根本差异

相较 v0.3.11，
本版有九个结构性变化：

1. `point-addressed expert bank`
   → `unified multimodal token field`

2. `current candidate slots + persistent slots + writeback`
   → `observation anchors + persistent posterior anchors + soft recycle`

3. `point / visual / tactile canonical experts`
   → `unified evidence binding + Gaussian vote heads`

4. `posterior-carried local query-gated update adapter`
   → `global predictive-innovation module`

5. `language-conditioned expectation only`
   → `language-late global future heads`

6. `innovation memory only affects next-step constructor / trust`
   → `explicit innovation token also enters action head`

7. `staged training`
   → `single-stage end-to-end training`

8. `tactile / point mostly latent-space update`
   → `tactile / point real-signal future prediction by default`

9. `predictive comparison mostly posterior-vs-expectation`
   → `posterior-vs-prior + global future-vs-real + innovation token`

---

## 0.3 记号约定

### 核心时序状态

- `h_{t,i}, c_{t,i}`：
  第 `i` 个 persistent posterior anchor 的 recurrent hidden / cell state

- `q_{t,i}^{post}(z)`：
  第 `i` 个 persistent posterior anchor 的 Gaussian posterior

- `p_{t,i}^{curr-}(z)`：
  第 `i` 个 persistent posterior anchor 在 recycle 之后用于当前步 fusion 的最终 current prior

- `\mu_{t,i}, \Sigma_{t,i}`：
  `q_{t,i}^{post}(z)` 的均值与协方差

- `\mu_{t,i}^{-}, \Sigma_{t,i}^{-}`：
  recycle 之前的 raw carried prior 均值与协方差

- `\bar\mu_{t,i}^{-}, \bar\Sigma_{t,i}^{-}`：
  recycle 之后真正进入当前步 Gaussian fusion 的 final current prior 均值与协方差

### 当前统一 token 场

- `p_{t,m}`：
  第 `m` 个 point / geometry token

- `v_{t,u}`：
  第 `u` 个 visual token（来自 V-JEPA current dense map）

- `\tau_{t,k}`：
  第 `k` 个 tactile token（来自 AnyTouch）

- `r_t^{ctx}`：
  proprio / action / force / timing context tokens

- `U_t`：
  当前统一 token 场原始 tokens 的并集

- `\tilde U_t`：
  当前统一 token 场经过共享 self-attention 后的 fused tokens

- `g_{t,m}^{p}, g_{t,u}^{v}, g_{t,k}^{t}`：
  用于 alignment loss 的 modality-specific projection embeddings；
  默认分别由 `Proj_p(p_{t,m})`、`Proj_v(v_{t,u})`、`Proj_t(\tau_{t,k})` 得到，并在使用前做 L2 normalization

- `\pi_t(x_{t,m})`：
  点 `x_{t,m}` 在当前相机像平面中的投影坐标

- `\tilde\pi_t(x_{t,m})`：
  点 `x_{t,m}` 在当前 visual token grid 上的连续 patch-grid 坐标；默认以 **patch** 为单位，与 `\tilde c_{t,u}^{grid}` 处在同一坐标系中

- `z_t(x_{t,m})`：
  点 `x_{t,m}` 在当前相机坐标系中的正向深度（实现中默认先做 `clip(z_{min},z_{max})`，再进入对数域）

- `\chi_{t,m}^{vis}`：
  点 `m` 对当前相机是否可见、是否通过边界/深度一致性检查的 visibility indicator；默认记作 `\chi_{t,m}^{vis}=\chi_t^{vis}(x_{t,m})`

- `\chi_t^{depth}(x)`：
  可选的 depth-validity / occlusion-consistency 指示量；仅在投影已确认落入图像边界时才评估。
  若当前样本不启用该检查，则默认恒为 1

- `d_t^{img}(\cdot)`：
  当前参考视觉帧（即 `t^{vis}` 对应帧）的深度图双线性采样算子；仅在样本确有可靠深度图时定义并启用

- `c_{t,u}^{pix}, \tilde c_{t,u}^{grid}`：
  visual token `u` 在当前实现所采用的 token-grid 参考坐标；其数值约定与 legacy `_scale_to_grid()` 一致，
  默认一单位对应一个 patch stride。
  注意：当前代码把它当作与 `\tilde\pi_t(x)` 同构的连续 token-grid address，
  而不是额外再去定义一套“严格 pooled patch 几何中心”的独立坐标

- `o_{cam,t}^{B}`：
  当前相机光心在全局坐标系 `F^B` 中的位置

- `d_{t,u}^{ray,C}, d_{t,u}^{ray,B}`：
  visual patch `u` 对应的 camera-frame / world-frame ray direction

- `\mathcal E_t^{pv}`：
  point↔visual 的稀疏 projective candidate edge set；所有几何辅助监督默认只在该集合上求和，而不把集合外 pair 当作显式负例

- `e_{null}^{proj,coarse}, e_{null}^{proj,fine}`：
  point token 在 projection 不可用时使用的 learned null projection embeddings；其维度分别与 2D Fourier `FF_{coarse}(\cdot;1)`、`FF_{fine}(\cdot;1)` 的输出一致

- `t^{vis}`：
  当前 visual reference frame 的时间索引；除显式说明外，当前步所有 point↔visual 几何量都默认相对于该帧定义

- `b_{t,m,u}^{proj}`：
  point token `m` 与 visual token `u` 之间的 relative projective bias

### observation anchors

- `o_{t,n}^{obs,0}`：
  第 `n` 个 observation anchor 的初始 seed query

- `o_{t,n}^{obs}`：
  第 `n` 个 current observation anchor token

- `A_{t,n,m}^{obs,p}`：
  observation anchor `n` 对 point token `m` 的 point-only 几何读出权重

- `\mathcal P_t, \mathcal V_t`：
  当前步 point / visual token 的 index sets

- `\widetilde A_{t,n,m}^{route,p}, \widetilde A_{t,n,u}^{route,v}`：
  从 unified attention 直接继承、尚未做 within-modality 归一化的 point / visual routing masses

- `\bar A_{t,n,m}^{route,p}, \bar A_{t,n,u}^{route,v}`：
  对 fixed point token / fixed visual token 沿 anchor 维归一后的 routing responsibilities

- `s_{t,m}^{route,p}, s_{t,u}^{route,v}`：
  point / visual token 在所有 observation anchors 上获得的总 routing support mass

- `\omega_{t,m}^{route,p}, \omega_{t,u}^{route,v}`：
  由 routing support mass 映射得到的 soft support gates；用于抑制“总支持度极低但归一化后看起来像同锚”的假阳性

- `\tau_{route}^{p}, \tau_{route}^{v}`：
  point / visual routing support gate 的平滑阈值超参

- `R_{t,m,u}^{anc}`：
  point `m` 与 visual patch `u` 被吸入同一 observation anchor 的 routing consistency score

- `x^{obs}_{t,n}, S^{obs}_{t,n}, a^{obs}_{t,n}`：
  current observation anchor 的几何 readout

### persistent anchors 与绑定

- `B_{t,i,n}`：
  persistent anchor `i` 对 current observation anchor `n` 的 soft binding 权重

- `\alpha_{t,i}`：
  persistent anchor `i` 的 soft activity / existence

- `\rho_{t,i}^{rec}`：
  persistent anchor `i` 的 recycle gate

- `u_{t,i}^{supp}`：
  persistent anchor `i` 当前获得的 support mass

- `r_t^{res}`：
  dustbin residual summary

- `\mu_t^{res}, \Sigma_t^{res}`：
  从 residual summary 解码得到的 recycle proposal statistics

- `\bar h^-_{t,i}, \bar c^-_{t,i}, \bar\mu^-_{t,i}, \bar\Sigma^-_{t,i}`：
  recycle 之后供当前步使用的 prior states

### future prediction

- `g_t^{post}`：
  当前 posterior anchors 经 self-attention 后形成的 global posterior context

- `s_t^{txt}`：
  语言摘要（来自 PaliGemma 或同类 side path）

- `\widetilde G_t^{pred}`：
  predictive-stage self-attention 输出的短 token set

- `g_t^{pred}`：
  将 `\widetilde G_t^{pred}` 池化后得到的 global predictive context

- `a_t^{\star}`：
  future heads 使用的动作条件；
  训练时为 teacher-forced executed action，
  推理时为当前动作头输出

- `\hat z_{t+1}^{v}`：
  下一时刻视觉 latent 预测

- `\hat y_{t+1}^{v,real}`：
  下一时刻可选真实视觉目标预测

- `\hat y_{t+1}^{t}`：
  下一时刻真实 tactile target 预测

- `\hat y_{t+1}^{p}`：
  下一时刻真实 point / geometry target 预测

### innovation

- `\hat z_t^{v,-}, \hat y_t^{v,real,-}, \hat y_t^{t,-}, \hat y_t^{p,-}`：
  上一步缓存下来、用于与当前真实目标比较的预测

- `\epsilon_t^{v}, \epsilon_t^{v,real}, \epsilon_t^{t}, \epsilon_t^{p}`：
  各预测头对应的当前 innovation residual

- `e_t^{innov}`：
  innovation token

- `m_t^{v},m_t^{v,real},m_t^{t},m_t^{p}`：
  各 innovation 分支的 availability masks

### 几何与接触

- `x_{t,i}`：
  persistent anchor 的几何中心

- `S_{t,i}\in\mathbb R^{3\times 3}`：
  persistent anchor 的 second-moment / spread tensor

- `a_{t,i}\in\mathbb R_+^3`：
  principal spread extent

- `p_{t,i}^{cnt}`：
  anchor-level contact probability

### 高斯委员会

- `q_{t,i}^{(r)}(z)`：
  persistent anchor `i` 的第 `r` 个 measurement vote

- `\hat\mu_{t,i}^{(r)}, R_{t,i}^{(r)}`：
  vote `r` 的均值与协方差

- `\beta_{t,i}^{(r)}`：
  vote `r` 的可信度权重

### 下游

- `token_{t,i}^{post}`：
  persistent posterior anchor token

- `u_t`：
  action / progress head 使用的最终 pooled state

- `\hat a_t`：
  当前动作输出

---

## 0.4 Operator Contract

除显式说明外，
本版所有内部实现、阈值存储与比较一律使用 SI 基本单位：

- 长度：米 `m`
- 时间：秒 `s`
- 力：牛顿 `N`
- 角度：弧度 `rad`

---

### 0.4.0 坐标系

本版默认 **只使用一个稳定全局坐标系**：

- `F^B`：stable global frame

所有进入 core 的 3D 量都先被变换到 `F^B`，包括：

- 点云坐标
- 相机位姿
- tactile sensor pose
- end-effector / wrist pose
- observation anchor 几何
- persistent anchor 几何

因此：

```math
x_{t,i},\ x_{t,m},\ g_{t,k}^{sens},\ g_t^{c}
\in F^B
```

如果 runtime 需要做局部 crop，
也只是 **在全局坐标系中截取一个空间子集**，
而不是再引入一个新的局部工作坐标系。

采用单一全局坐标系的原因是：
当前系统默认部署在固定桌面场景，
全局参考系已经足够稳定，
额外再引入新的局部工作坐标系只会增加实现负担与记号复杂度，
而不会带来实质收益。

### 0.4.1 Fourier features

对任意 `x=[x_1,x_2,x_3]^\top\in\mathbb R^3`
与尺度 `\ell>0`：

```math
FF_B(x;\ell)=
\bigoplus_{d=1}^{3}
\bigoplus_{k=0}^{B-1}
\Big[
\sin(2^k\pi x_d/\ell),
\cos(2^k\pi x_d/\ell)
\Big]
```

定义：

```math
FF_{coarse}(x;\ell)=FF_{B_{coarse}}(x;\ell),\qquad
FF_{fine}(x;\ell)=FF_{B_{fine}}(x;\ell/4)
```

默认：

- `B_{coarse}=4`
- `B_{fine}=8`

若输入是 2D 坐标（例如 `\tilde\pi_t(x)`），
则对其两个分量按完全同构的方式逐维展开；
也就是说，本版在 2D patch-grid 坐标上使用的是
`FF_B` 的 2D 特例，
而不是重新定义另一套独立的 trig encoding。

---

### 0.4.2 geometry position encoding

```math
PE_{geo}(x,S)=
\Big[
FF_{coarse}(x;R_{ws}),
FF_{fine}(x;R_{crop}),
\log eig(S+\epsilon_S I),
vec(triu(S/(tr(S)+\epsilon_S)))
\Big]
```

注意：
这里 `x` 始终是在 **全局坐标系 `F^B`** 中表达，
只是同时使用工作区尺度 `R_{ws}` 与局部操作尺度 `R_{crop}`
去构造多尺度 Fourier features；
并不存在“局部坐标系编码”和“全局坐标系编码”两套并行状态。

### 0.4.2b point-token position encoding

point token 只需要位置地址时，
不再伪造一个 `S=\epsilon_S I` 去调用 `PE_{geo}`，
而是显式使用 point-only 位置编码：

```math
PE_{pt}(x)=
\Big[
FF_{coarse}(x;R_{ws}),
FF_{fine}(x;R_{crop})
\Big]
```

它只承担全局 3D 位置地址作用，
不引入额外手工形状统计。

### 0.4.2c camera-conditioned projection / ray encoding

对当前相机内参 `K_t` 与位姿 `T_{cam\rightarrow F^B,t}`，
定义：

```math
T_{F^B\rightarrow cam,t}=T_{cam\rightarrow F^B,t}^{-1}
```

对点 `x\in F^B`，其相机坐标为：

```math
x^C = T_{F^B\rightarrow cam,t}\,x
```

令：

```math
z_t(x)=clip(x_3^C, z_{min}, z_{max})
```

当 `x_3^C>0` 时，其像平面投影定义为：

```math
\pi_t(x)=
\begin{bmatrix}
f_x x^C_1/x^C_3 + c_x\\
f_y x^C_2/x^C_3 + c_y
\end{bmatrix}
```

再定义可见性指示量：

```math
\chi_t^{vis}(x)=
\begin{cases}
0, & x_3^C\le 0,\\
0, & x_3^C>0\ \text{且}\ \pi_t(x)\notin[0,W_0)\times[0,H_0),\\
\chi_t^{depth}(x), & x_3^C>0\ \text{且}\ \pi_t(x)\in[0,W_0)\times[0,H_0)
\end{cases}
```

其中：若当前样本不启用 depth-validity / occlusion-consistency 检查，则默认 `\chi_t^{depth}(x)=1`。
这样 `\chi_t^{depth}(x)` 只会在投影已确认位于图像内时才被访问，
避免对越界投影再去评估深度一致性。

并把它映射到与 visual token token-grid address 相同的连续 patch-grid 坐标系，
记为 `\tilde\pi_t(x)`；默认以 **patch** 为单位，因而相邻 grid index 的间距为 1。
当前实现的具体数值约定与 legacy `_scale_to_grid()` 保持一致。
当 `\chi_t^{vis}(x)=0` 时，
`PE_{proj}` 分支不再依赖 `\tilde\pi_t(x)` 的数值；
工程实现上可直接填一个固定 dummy grid coordinate，
或在进入 projection branch 之前直接短路跳过投影坐标的数值读取，
再由 null projection branch 覆盖掉该无效输入。

对 visual token `u`，记其当前 token-grid 参考像素坐标为：

```math
c_{t,u}^{pix}=
[c_{t,u,x}^{pix},c_{t,u,y}^{pix}]^\top
```

其在同一连续 patch-grid 坐标系下的表示记为 `\tilde c_{t,u}^{grid}`；同样以 **patch** 为单位，
并与 legacy `_scale_to_grid()` 的 lattice 约定保持一致。

由此定义 camera-frame ray：

```math
\bar d_{t,u}^{ray,C}=K_t^{-1}[c_{t,u,x}^{pix},c_{t,u,y}^{pix},1]^\top,\qquad
d_{t,u}^{ray,C}=\bar d_{t,u}^{ray,C}/\|\bar d_{t,u}^{ray,C}\|_2
```

再映射到全局坐标系：

```math
d_{t,u}^{ray,B}=R_{cam\rightarrow F^B,t}\,d_{t,u}^{ray,C}
```

令 `e_{null}^{proj,coarse}` 与 `e_{null}^{proj,fine}` 为 learned null projection embeddings。
则 point token 的 camera-conditioned 投影编码定义为：

```math
PE_{proj}(x;K_t,T_{cam\rightarrow F^B,t})=
\Big[
\chi_t^{vis}(x)\,FF_{coarse}(\tilde\pi_t(x);1)
+
\big(1-\chi_t^{vis}(x)\big)e_{null}^{proj,coarse},
\chi_t^{vis}(x)\,FF_{fine}(\tilde\pi_t(x);1)
+
\big(1-\chi_t^{vis}(x)\big)e_{null}^{proj,fine},
\chi_t^{vis}(x)\log z_t(x),
\chi_t^{vis}(x),
\log(f_x/W_0),\log(f_y/H_0)
\Big]
```

对应地，visual token 的 ray 编码定义为：

```math
PE_{ray}(u;K_t,T_{cam\rightarrow F^B,t})=
\Big[
\tilde c_{t,u}^{grid},
d_{t,u}^{ray,B},
o_{cam,t}^{B},
\log(f_x/W_0),\log(f_y/H_0)
\Big]
```

这里 `o_{cam,t}^{B}` 是当前相机在全局坐标系中的光心。

这两类编码的角色分别是：

- `PE_{proj}`：告诉 point token “从当前相机看，你落在什么 patch / 什么深度”；若当前点不可见，则显式走 null branch，而不是把无效投影当作正常数值送入网络
- `PE_{ray}`：告诉 visual token “你对应哪条世界射线，而不只是 2D patch index”

这正是本版 point↔visual 对齐的主 inductive bias。

---

### 0.4.3 tactile pose encoding

```math
rot6d(R)=
\big[
R_{:,1}^\top,
R_{:,2}^\top
\big]
```

```math
pose6d(T)=\log(T)
```

`pose6d` / `rot6d`
只用于 tactile token addressing 与 diagnostics，
不作为 formal object state。

---

### 0.4.4 anchor geometry PE

```math
PE_{anc}(x,a,S)=
MLP_{ancPE}
\Big[
FF_{coarse}(x;R_{ws}),
FF_{fine}(x;R_{crop}),
\log a,
\log eig(S+\epsilon_S I)
\Big]
```

---

# 1. 输入、输出与部署前提

## 1.1 输入

- RGB / video：
  `I_{t-H_v+1:t}`

- 点云 / depth-derived point cloud：
  `P_t`

- per-point color：
  `C_t`（**必需**；本版默认点云输入契约为 `xyzrgb`）

- tactile history：
  `T_{t-L_{tac}+1:t}`

- contact pose / EE pose：
  `g_t^c`

- force / wrench related signal：
  `f_t`

- 可选 indentation：
  `d_t^{indent}`

- 可选 tactile activation / pressure proxy：
  `p_t^{tac}`

- executed action：
  `a_t`

- proprio：
  `p_t`

- instruction / text：
  `l_t`

注意：
本版默认 **点云必须带 `xyzrgb`，但不要求 normal**。
如果数据源无法稳定提供 RGB，
则该数据流不满足本版输入契约，
应在数据层或感知栈外部先解决，
而不是在本版 core 中走任何 `RGB fallback`。

---

## 1.2 输出

核心输出为：

```math
X_t=
\Big\{
q_{t,i}^{post}(z),
x_{t,i},
S_{t,i},
a_{t,i},
\alpha_{t,i},
token_{t,i}^{post}
\Big\}_{i=1}^{K}
```

同时生成：

- `g_t^{post}`：
  posterior anchor 自注意力后的 global posterior context

- `g_t^{pred}`：
  融合语言与动作后的 predictive context

- `\hat z_{t+1}^{v}, \hat y_{t+1}^{v,real}, \hat y_{t+1}^{t}, \hat y_{t+1}^{p}`：
  future heads 的预测

- `e_t^{innov}`：
  当前 innovation token

- `u_t`：
  最终 pooled state

- `\hat a_t`：
  动作输出

---

## 1.3 非目标

本版**不**尝试在 core 内实现：

- runtime 下的长链显式 imagination rollout
- 必须依赖的 raw-frame autoregressive generation
- point-cloud normals 作为 formal state
- 语言直接参与 current posterior fusion
- 大量 hard lifecycle state machine
- 完整的 multi-hypothesis tree search 作为默认在线路径

---

## 1.4 部署前提

进入闭环部署前，
必须满足：

1. camera intrinsics / extrinsics 已知；
2. 点云与 RGB 已在输入侧稳定融合为 `xyzrgb`；
3. tactile sensor 到 wrist / hand frame 外参已知；
4. `F^B` 已定义，且所有 3D 量都能先转换到 `F^B`；
5. proprio / FK / executed action 日志存在；
6. 跨模态时间戳存在，且满足：
   ```math
   \delta_{sync}^{max}\le 20\text{ ms}
   ```
7. 本版 **不提供 point-cloud RGB fallback**；
   若 `xyzrgb` 不能稳定生成，
   则训练/部署前提不成立；
8. 若绑定 SpatialLM / SONATA 变体，
   其具体输入字段、batching、patching 与 tokenizer 细节必须以真实代码为准；
   本总纲不再假设 `normal` 是必需输入；
9. 若未来真实点云目标采用 occupancy / TSDF / range-image / scene-flow，
   则该目标的生成链必须可稳定复现。

## 1.5 episode / rollout initialization

首个控制步 `t=0`
必须显式初始化 persistent anchors，
不能依赖“空 cache 隐式推断”。

默认初始化：

```math
\mu_{0,i}=0_{D_z},\qquad
\Sigma_{0,i}=diag(\sigma_{reset}^2\mathbf 1_{D_z})
```

```math
h_{0,i}=0,\qquad c_{0,i}=0
```

```math
x_{0,i}=0_3,\qquad
S_{0,i}=diag((a_{min}/2)\odot(a_{min}/2))+\epsilon_S I,\qquad
a_{0,i}=a_{min}
```

```math
\alpha_{0,i}=\alpha_{init}
```

其中 `\alpha_{init}` 默认为小正值，
而不是严格 0，
这样 persistent anchors 从一开始就作为“低活性可重用槽位”存在。

同时清空预测缓存：

```math
\hat z_0^{v,-}=0,\qquad
\hat y_0^{v,real,-}=0,\qquad
\hat y_0^{t,-}=0,\qquad
\hat y_0^{p,-}=0
```

并定义：

```math
e_0^{innov}=0
```

因此首个控制步：

- 不做 innovation comparison
- 不要求已有上一步 future prediction
- 直接由当前 unified token field 生成 observation anchors，
  再更新 persistent anchors

---

# 2. 隐藏状态：统一 token 场、observation anchors 与 persistent posterior anchors

## 2.1 local crop 与点云子集

本版所有点云都先统一到全局坐标系 `F^B`：

```math
P_t^{glob}=Transform(P_t\rightarrow F^B)
```

然后仅在 **全局坐标系中** 做空间截取。
设当前操作中心 `x_t^{roi}\in F^B`，
则：

```math
P_t^{loc}=P_t^{glob}\cap Ball(x_t^{roi}, R_{crop})
```

外围集合：

```math
P_t^{per}=P_t^{glob}\setminus P_t^{loc}
```

这里 `P_t^{loc}` 只是“用于当前计算的局部点子集”，
不是“定义在局部坐标系里的点云”。

默认：

```math
R_{crop}=0.08\text{ m}
```

## 2.2 点云 wrapper：保留 Sonata 兼容，但 formal state 不写 normal

本版默认点云 backbone 使用你指定的 **SONATA / SpatialLM 改版**，
输入契约是：

```math
(x_{t,m}, c_{t,m}),
\qquad x_{t,m}\in\mathbb R^3,\ c_{t,m}\in\mathbb R^3
```

也就是 **必须是 `xyzrgb`**。

本版明确做出两条约束：

1. **不要求 normal**；
2. **不再在 backbone 前额外手工计算局部几何统计、pseudo-normal、shape stub 等前处理**。

原因是：
当前文档中的 backbone 角色已经是 SpatialLM / SONATA 这一类空间点云表示学习模型，
若再在它前面人为塞入一整套 `kNN + 二阶矩 + pseudo-normal` 的手工统计，
容易与真实实现重复、冲突，
也会带来不必要的部署开销。

因此本版在总纲层面只规定：

- 点云输入必须是 `xyzrgb`
- 具体 tokenization / patching / voxelization / batching / masking
  **以真实代码实现为准**
- 本文不再对 SONATA / SpatialLM 的内部 patch 构造作过硬假设

point token features 统一记为：

```math
\{f_{t,m}^{p}\}_{m=1}^{M_p}=PointEncoder_{SLM}(P_t^{loc}, C_t^{loc})
```

在进入 unified token field 之前，
每个 point feature 还需要被显式投影成 point token：

```math
p_{t,m}
=
W_p
\Big[
 f_{t,m}^{p},
 PE_{pt}(x_{t,m}),
 PE_{proj}(x_{t,m};K_t,T_{cam\rightarrow F^B,t}),
 emb(point)
\Big]
```

其中 `PE_{pt}(x_{t,m})` 只承担该点在全局坐标系 `F^B` 中的位置地址编码；
它不是额外手工 shape descriptor，也不会替代 backbone 自己学到的几何表征。
这里 `PE_{proj}` 中出现的 `K_t,T_{cam\rightarrow F^B,t}` 按 2.3.1 节约定，
默认指当前 visual reference frame 的相机几何。

其中 `PointEncoder_{SLM}` 是对真实 SONATA / SpatialLM 代码的抽象记号。
只要真实实现满足 `xyzrgb` 输入契约，
本文不再规定它内部是否采用额外相对坐标、局部 patch 索引、稀疏卷积或 transformer patching。

## 2.3 V-JEPA 2.1 visual tokens

### 2.3.1 dense visual encoder

对视频片段：

```math
I_{t-H_v+1:t}
```

用 V-JEPA 2.1 encoder 得到当前 dense visual map：

```math
F_t^v = VJEPA2Encoder(I_{t-H_v+1:t})
```

当启用 `PE_{proj}` / `PE_{ray}` 时，
本版默认把用于当前 visual tokens 的几何参考帧固定为片段最后一帧 `I_t`，
记为 `t^{vis}=t`。
除显式说明外，下文凡在 point↔visual 几何项中写 `K_t,T_{cam\rightarrow F^B,t}`，
默认都指这一当前 visual reference frame 的相机几何。
若实现上确实需要对最后两帧做均值，
则必须先对各帧分别构造带各自几何 conditioning 的 visual tokens，
再在 token / feature 层做 pooling；
不得直接用单帧 `K_t,T_t` 去解释跨帧均值后的 dense map。

### 2.3.2 visual patch tokens

设当前 dense map 中有 `U_v` 个 patch / token 位置，
定义第 `u` 个视觉 token：

```math
v_{t,u}
=
W_v
\Big[
f_{t,u}^{v},
PE_{img}(u),
PE_{cam}(T_{cam\rightarrow F^B,t^{vis}}),
PE_{ray}(u;K_{t^{vis}},T_{cam\rightarrow F^B,t^{vis}}),
emb(visual)
\Big]
```

其中：

- `f_{t,u}^{v}` 来自 `F_t^v`
- `PE_{img}(u)` 是 2D patch 位置编码
- `PE_{cam}` 表示相机在全局坐标系 `F^B` 中的位姿编码

### 2.3.3 camera-conditioned point-visual geometry conditioning

本版对 point↔visual 的默认策略不再是：
“先让两边 token 没有显式几何关系地进入网络，
再靠强 `L_{pv}` 在训练后期把 embedding 拉近”。

替代为两层设计：

1. **geometry-first token conditioning**
   - point token 吃 `PE_{proj}`
   - visual token 吃 `PE_{ray}`

2. **relative projective attention bias（推荐升级项）**
   - 当前实现已在 point↔visual 边上加入 `b_{t,m,u}^{proj}`
   - `FusionTransformer` 当前已经支持 pairwise attention bias，
     并能导出平均 attention map 供 `L_{focus}^{pv}` 使用

就当前代码而言，
`src/openpi/picf/core/pipeline.py::_build_token_field()`
已经把 `CameraModel.K`、投影坐标、ray direction、
relative projective bias 和 attention export
接入 point / visual token builder。

其中可用的 bias 形式为：

```math
b_{t,m,u}^{proj}
=
MLP_{proj-bias}
\Big[
\tilde\pi_t(x_{t,m})-\tilde c_{t,u}^{grid},
\log z_t(x_{t,m}),
\langle d_{t,u}^{ray,B},\widehat{x_{t,m}-o_{cam,t}^{B}}\rangle,
\chi_{t,m}^{vis}
\Big]
```

其中 `\tilde c_{t,u}^{grid}` 与 `\tilde\pi_t(x_{t,m})` 定义在同一连续 patch-grid 坐标系中，且默认都以 **patch** 为单位。
当 `\chi_{t,m}^{vis}=0` 时，
`b_{t,m,u}^{proj}` 默认退化为零偏置或 learned null bias；
工程实现上也推荐只对 `(m,u)\in\mathcal E_t^{pv}` 的候选边显式计算该 bias，
其余边设为 0。

它只在 point→visual 与 visual→point 的 cross-modal logits 上起作用，
不改 point→point、visual→visual、tactile 相关边。

这一步的设计依据是：

- RayI2P 指出，直接做 2D-3D matching 会遇到 projection ambiguity 与 scale inconsistency
- PRoPE 进一步表明，camera-conditioned relative/projective conditioning
  比只做 token-level 几何附加项更稳、更容易泛化到变化的 intrinsics 与序列长度

---

## 2.4 AnyTouch tactile tokens

### 2.4.1 tactile backbone

AnyTouch 2 输出：

```math
(F_t^{t,glob},\{F_{t,k}^{t}\}_{k=1}^{K_{sens}})
```

其中：

- `F_t^{t,glob}`：全局 tactile summary
- `F_{t,k}^{t}`：每个 sensor / patch 的特征

### 2.4.2 pose-addressed tactile token

对 sensor `k`，定义：

```math
\tau_{t,k}
=
W_t
\Big[
F_{t,k}^{t},
F_t^{t,glob},
PE_{sens}(g_{t,k}^{sens}),
f_t,
p_t,
emb(tactile)
\Big]
```

这里的 `PE_{sens}` 只是地址编码：
告诉网络这份触觉证据来自手/腕/指的什么位置与朝向；
不是手工把 tactile 先投影到某个世界点。

---

## 2.5 Pi0.5-style context tokens

本版的 context token 不再随意扩张成一长串自定义 side tokens，
而是明确对齐到 **Pi0.5 风格的紧凑 context/prefix conditioning**：

- 少量、语义固定的上下文 token
- 负责提供机器人状态、上一动作、时间可用性等全局条件
- 不在 current token field 中引入大量手写 sector/global heuristics

当前 unified token field 中默认保留一个紧凑的机器人上下文 token：

```math
r_t^{ctx}
=
W_r
\Big[
p_t,\ f_t,\ a_{t-1},\ \Delta t_v,\ \Delta t_p,\ \Delta t_t,\ emb(ctx)
\Big]
```

如真实代码需要，
它可以实现为一个很小的 token 组，
例如：

- robot-state token
- previous-action token
- timing / availability token

但总纲层面不再展开成大量 hand-crafted context summaries。

另外，和你前面坚持的原则一致：
**语言语义不在 current posterior 形成阶段以 context token 方式提前注入**；
语言只在 posterior 之后进入 predictive / selector 阶段。
因此这里所谓“Pi0.5-style context token”，
主要指其**紧凑 prefix-conditioning 组织方式**，
而不是把 semantic subtask token 直接塞进 current physical posterior 路径。

## 2.6 unified multimodal token field

把所有 token 放入统一集合：

```math
U_t=
\{p_{t,m}\}_{m=1}^{M_p}
\cup
\{v_{t,u}\}_{u=1}^{U_v}
\cup
\{\tau_{t,k}\}_{k=1}^{K_{sens}}
\cup
\{r_t^{ctx}\}
```

然后在统一隐空间做共享 self-attention：

```math
\tilde U_t = \mathrm{FusionTransformer}(U_t)
```

跨模态对齐所依赖的主要编码帮助是：

- point tokens：`PE_{pt}(x_{t,m}) + PE_{proj}(x_{t,m};K_t,T_{cam\rightarrow F^B,t})`
- visual tokens：`PE_{img}(u)+PE_{cam}(T_{cam\rightarrow F^B,t^{vis}})+PE_{ray}(u;K_{t^{vis}},T_{cam\rightarrow F^B,t^{vis}})`
- tactile tokens：`PE_{sens}(g_{t,k}^{sens})`，即触觉传感器位姿编码

因此，point / visual / tactile 的对齐不是靠额外手工 angle-binning，
也不是把强 `L_{pv}` 当主力，
而是靠：

- point / visual 的 camera-conditioned geometry conditioning
- 共享 self-attention
- 后续 anchor binding

共同组织当前世界。

这里的设计理念是：

- 点云、视觉、触觉全部 token 化
- 全部进入同一隐空间
- 先做统一自注意力，让 V-JEPA 的时间预测性、
  触觉的局部接触性、点云的几何性在当前感官层先互相可见

这正是本版的 current-sensor 世界表征。

为了让后续弱辅助 `L_{pv}^{weak}` / `L_{pt}` 的符号闭合，
这里顺带定义 alignment 使用的 modality-specific projection embeddings：

```math
g_{t,m}^{p}=Normalize(Proj_p(p_{t,m}))
```

```math
g_{t,u}^{v}=Normalize(Proj_v(v_{t,u}))
```

```math
g_{t,k}^{t}=Normalize(Proj_t(\tau_{t,k}))
```

除显式说明外，弱辅助 alignment loss 默认使用 **pre-fusion modality tokens** 的这些投影，
而不是使用 posterior tokens。

但需要强调：

- 这些投影只承担弱辅助监督
- 它们不再被视为 point↔visual 主对齐机制
- 主机制是 `PE_{proj}` / `PE_{ray}` 与可选 `b_{t,m,u}^{proj}`

---

## 2.7 current observation anchors：由统一 token 场自发形成

### 2.7.1 seed queries

设当前帧使用 `N_obs` 个 observation anchor queries。

其初始 query 不直接从 learnable constant 开始，
而是由 farthest-point-sampled point subset
或 point token seeds 初始化。
设第 `n` 个 seed 对应 point index `m(n)`，
则：

```math
o_{t,n}^{obs,0}
=
MLP_{obs-seed}
\Big[
p_{t,m(n)},
PE_{pt}(x_{t,m(n)}),
emb(obs)
\Big]
```

这样 current observation anchors 的形成
天然带几何中心性，
不会完全脱离世界坐标。

### 2.7.2 observation anchor reading

第 `n` 个 observation anchor query 从统一 token 场读信息：

```math
\alpha_{t,n,j}^{obs}
=
\operatorname{softmax}_{j\in U_t}
\left(
\frac{(W_q o_{t,n}^{obs,0})^\top (W_k \tilde U_{t,j})}{\sqrt d}
\right)
```

```math
m_{t,n}^{obs}
=
\sum_j \alpha_{t,n,j}^{obs} V \tilde U_{t,j}
```

再经一层 GRU / update：

```math
o_{t,n}^{obs,1}
=
GRU_{obs}(o_{t,n}^{obs,0}, m_{t,n}^{obs})
```

可重复 `R_obs` 轮，
默认 `R_obs=2`。

除显式说明外，
后文若再写 `\alpha_{t,n,j}^{obs}`，
默认指 **最后一轮 observation-anchor reading** 的注意力权重。

### 2.7.3 observation anchor self-attention

为让 current observation anchors
共同形成“当前世界概念图”，
定义：

```math
\hat O_t = \mathrm{SelfAttn}(\{o_{t,n}^{obs,R_{obs}}\}_{n=1}^{N_{obs}})
```

其输出记为：

```math
o_{t,n}^{obs}
```

这一步非常重要：
它不是可选装饰，
而是你坚持的那件事：
**当前世界状态先由多模态 token 场中的一组 anchors 共同组织出来。**

---

## 2.8 由 observation anchors 读出当前显式几何

### 2.8.1 point-only ownership for geometry

虽然 observation anchors 是从统一 token 场读出来的，
但几何必须从 point token 子集读出。
定义 point token index set：

```math
\mathcal P_t = \{1,\dots,M_p\}
```

对每个 observation anchor `n`，
只取其对 point tokens 的注意力，
再在 point 维上重归一化：

```math
A_{t,n,m}^{obs,p}
=
\frac{\alpha_{t,n,m}^{obs}}
{\sum_{m'\in\mathcal P_t}\alpha_{t,n,m'}^{obs}+\epsilon_A},
\qquad m\in\mathcal P_t
```

### 2.8.2 center / spread / extent

```math
x_{t,n}^{obs}
=
\sum_{m\in\mathcal P_t}
A_{t,n,m}^{obs,p} x_{t,m}
```

```math
S_{t,n}^{obs}
=
\sum_{m\in\mathcal P_t}
A_{t,n,m}^{obs,p}
(x_{t,m}-x_{t,n}^{obs})(x_{t,m}-x_{t,n}^{obs})^\top
+\epsilon_S I
```

```math
a_{t,n}^{obs}
=
clip\Big(
2\sqrt{sort_{\downarrow}(eig(S_{t,n}^{obs}))+\epsilon_{ext}},
a_{min},a_{max}
\Big)
```

---

## 2.9 persistent posterior anchors

系统维护 `K` 个 persistent anchors，
每个 anchor 的 cache 为：

```math
a_{t,i}^{cache}
=
\{
h_{t,i}, c_{t,i},
\mu_{t,i}, \Sigma_{t,i},
x_{t,i}, S_{t,i}, a_{t,i},
\alpha_{t,i}
\}
```

其中：

- `(h_{t,i},c_{t,i})`：
  recurrent memory

- `(\mu_{t,i},\Sigma_{t,i})`：
  Gaussian posterior

- `(x_{t,i},S_{t,i},a_{t,i})`：
  显式几何状态

- `\alpha_{t,i}`：
  soft activity / existence

本版不再维护 `active / dormant / retired` 的硬状态机；
活动度由 `\alpha_{t,i}` 连续控制。

---

## 2.10 carried prior

上一时刻 persistent anchors
经 recurrent transition 与动作条件更新，
形成当前 carried prior：

```math
(\mu^-_{t,i}, \Sigma^-_{t,i}, h^-_{t,i}, c^-_{t,i}, x^-_{t,i}, S^-_{t,i}, a^-_{t,i}, \alpha^-_{t,i})
=
F_{prior}(h_{t-1,i}, c_{t-1,i}, \mu_{t-1,i}, \Sigma_{t-1,i}, x_{t-1,i}, S_{t-1,i}, a_{t-1,i}, \alpha_{t-1,i}, a_{t-1}, p_t)
```

其中紧随 `x_{t-1,i}` 之后的五个输入依次表示：

- 上一时刻该 anchor 的几何 spread `S_{t-1,i}`
- 上一时刻该 anchor 的 extent `a_{t-1,i}`
- 上一时刻该 anchor 的 activity `\alpha_{t-1,i}`
- 上一控制步 executed action `a_{t-1}`
- 当前 proprio / robot state `p_t`

这里 `F_{prior}` 不是显式 frame predictor；
它只是 carried prior constructor，
代表上一时刻 posterior 在当前时刻应当期待什么。

---

## 2.11 observation anchors ↔ persistent anchors 的软绑定

### 2.11.1 affinity

对 persistent anchor `i`
与 current observation anchor `n`，
定义：

```math
C_{t,i n}^{bind}
=
\lambda_h \cos(W_h h^-_{t,i}, W_o o_{t,n}^{obs})
-
\lambda_x d_M(x_{t,n}^{obs}, x^-_{t,i}; S^-_{t,i})
```

其中：

```math
d_M(x,x';S)=(x-x')^\top(S+\sigma_{bind}^2 I)^{-1}(x-x')
```

### 2.11.2 soft competition with dustbin

用带 dustbin 的 Sinkhorn / slot-competition：

```math
B_t = \mathrm{Sinkhorn}_{dustbin}(C_t^{bind})
```

于是：

- 每个 observation anchor
  主要绑定到某个 persistent anchor 或 dustbin
- 同一 observation anchor 不会被很多 persistent anchors 重复吞掉
- dustbin 吸收 residual / clutter / new unexplained mass

这是本版防止重复 anchor 的第一道核心机制。

---

## 2.12 soft lifecycle：activity 与 recycle

### 2.12.1 support mass

```math
u_{t,i}^{supp}=\sum_n B_{t,i n}
```

### 2.12.2 activity update

```math
\alpha_{t,i}
=
\sigma\Big(
MLP_{act}
[
h^-_{t,i},u_{t,i}^{supp},
\operatorname{tr}(\Sigma^-_{t,i}),
\alpha^-_{t,i}
]
\Big)
```

活动度高：
说明这个 persistent anchor
当前仍然被 observation anchors 支持。

### 2.12.3 recycle gate

定义 dustbin residual summary：

```math
r_t^{res}
=
\sum_n B_{t,\varnothing n}\,o_{t,n}^{obs}
```

并从 residual summary 中解码一个软重置 proposal：

```math
\mu_t^{res}=Head_{\mu,res}(r_t^{res})
```

```math
\Sigma_t^{res}
=
\operatorname{diag}\big(
\operatorname{softplus}(Head_{\Sigma,res}(r_t^{res})) + \epsilon\mathbf 1
\big)
```

recycle gate：

```math
\rho_{t,i}^{rec}
=
\sigma\Big(
MLP_{rec}
[
h^-_{t,i},u_{t,i}^{supp},
\operatorname{tr}(\Sigma^-_{t,i}),
r_t^{res},
\alpha^-_{t,i}
]
\Big)
```

若 `\rho_{t,i}^{rec}` 高，
则说明该 anchor 当前 support 低、uncertainty 高、
且 residual 较强，
可被软重置：

```math
\bar h^-_{t,i}
=
(1-\rho_{t,i}^{rec}) h^-_{t,i}
+
\rho_{t,i}^{rec} W_{rec,h} r_t^{res}
```

```math
\bar c^-_{t,i}
=
(1-\rho_{t,i}^{rec}) c^-_{t,i}
+
\rho_{t,i}^{rec} W_{rec,c} r_t^{res}
```

```math
\bar \mu^-_{t,i}
=
(1-\rho_{t,i}^{rec}) \mu^-_{t,i}
+
\rho_{t,i}^{rec} \mu_t^{res}
```

```math
\bar \Sigma^-_{t,i}
=
(1-\rho_{t,i}^{rec}) \Sigma^-_{t,i}
+
\rho_{t,i}^{rec} \Sigma_t^{res}
```

这里 recycle 只软重置 **recurrent / latent prior**；
几何 prior `x^-_{t,i},S^-_{t,i},a^-_{t,i}` 仍保留到当前 evidence binding 阶段，
再由当前 observation anchors 重新估计。
这样既能在残差驱动下“重生” anchor，
又避免把回收过程写死成硬覆盖。

这就是本版对“形成、衰退、回收”的统一回答：

- 新世界成分出现时，
  dustbin residual 增大；
- 低活动 persistent anchor
  会被 recycle gate 软重置去接收这些残差；
- 无需显式 birth / reuse / retired 状态机。

因此，本版在当前步真正用于 Gaussian fusion 的 final current prior 定义为：

```math
p_{t,i}^{curr-}(z)=\mathcal N(\bar\mu_{t,i}^{-},\bar\Sigma_{t,i}^{-})
```


---

# 3. Wrapper、teacher 目标与统一表示

## 3.1 总原则

- **V-JEPA 2.1**：
  负责 dense visual predictive representation，
  当前 encoder 不接语言；
  可作为视觉 latent teacher。

- **SONATA / SpatialLM 改版**：
  提供可靠点云空间表征；
  本版默认输入契约为 `xyzrgb`，不要求 normal，且不增加手工局部统计前处理；
  其具体 tokenizer / patching / batching 细节以真实代码实现为准。

- **AnyTouch 2**：
  提供动态 tactile representation；
  其真实未来头默认预测真实 tactile signal。

- **PaliGemma**：
  只负责 semantic / language side path；
  不进入 current posterior；
  只在 global future heads 与 downstream selector 中使用。

---

## 3.2 V-JEPA 2.1 teacher 与目标

### 3.2.1 current encoder usage

当前帧特征：

```python
outputs = VJEPA2Model.from_pretrained(... )(
    pixel_values_videos = video,
    skip_predictor = True,
    return_dict = True,
)
```

取当前 encoder outputs 形成 `F_t^v`。

### 3.2.2 future latent teacher

训练时，
下一时刻视觉 latent teacher 定义为：

```math
z_{t+1}^{v}
=
Pool\big(
VJEPA2Encoder(I_{t-H_v+2:t+1})
\big)
```

这里可采用：

- mean pooled patch latent
- spatially pooled dense map
- 或对 selected visual anchors 的 teacher pooling

本版不要求严格复用 V-JEPA 内部 predictor / target-mask 机制，
因为我们的 predictive state 是 global posterior，
不是 V-JEPA 本身的 student encoder。

### 3.2.3 default visual real auxiliary target

为约束真实可复现性，
可选定义轻量真实视觉目标：

- 下一时刻低分辨率 RGB crop
- 下一时刻 depth patch / motion map / dynamic-region target
- 下一时刻 selected visual slots 的 patch reconstruction

默认不要求重建整张图像。

---

## 3.3 SONATA / SpatialLM teacher 与 point targets

当前点特征：

```math
\{f_{t,m}^{p}\}_{m=1}^{M_p}=PointEncoder_{SLM}(P_t^{loc}, C_t^{loc})
```

这里再次强调：
本版假设 `P_t^{loc}, C_t^{loc}` 共同构成 **`xyzrgb` 点云输入**；
不存在 `RGB fallback`。
若真实实现的 SONATA / SpatialLM API 字段名不同，
则以真实代码为准，
但数学总纲默认它吃的是 `xyzrgb`。

下一时刻点相关真实目标默认采用至少一种：

1. **range / depth image target**
2. **occupancy / TSDF target**
3. **local scene-flow target**
4. **resampled point-set target**（可选 Chamfer）

记为统一真实点目标：

```math
y_{t+1}^{p}
```

同时可选保留下一时刻 point latent teacher：

```math
z_{t+1}^{p}
=
Pool\big(PointEncoder_{SLM}(P_{t+1}^{loc}, C_{t+1}^{loc})\big)
```

但默认**真实点目标优先**，
latent 只作辅助项。

## 3.4 AnyTouch 2 teacher 与 tactile targets

当前 tactile 特征：

```math
(F_t^{t,glob},\{F_{t,k}^{t}\})
=
AnyTouch2(T_{t-L_{tac}+1:t})
```

下一时刻真实 tactile 目标默认至少包含：

- tactile image / feature map
- force-aware target
- contact / deformation proxy
- 可选 action-conditioned deformation summary

统一记作：

```math
y_{t+1}^{t}
```

同时可选定义下一时刻 tactile latent teacher：

```math
z_{t+1}^{t}
=
Pool(AnyTouch2(T_{t-L_{tac}+2:t+1}))
```

但和 point-cloud 一样，
默认**真实 tactile 目标优先**，
latent 只作辅助正则。

---

## 3.5 PaliGemma side path

PaliGemma 输出：

- `last_hidden_state`
- `image_hidden_states`

定义：

```math
s_t^{txt}=mean_{\text{text tokens}}(H_{mm})
```

```math
s_t^{img}=mean_{\text{image tokens}}(H_{img})
```

```math
\tilde s_t^{sem}=[s_t^{txt},s_t^{img}]
```

本版规定：

- `s_t^{txt}` 不进入当前 posterior 形成
- `s_t^{txt}` 只进入：
  - global future heads
  - task selector / relevance weighting
  - downstream progress / action routing

---

# 4. 当前 posterior anchors 的形成

## 4.1 persistent anchor 对当前 observation anchors 读证据

对 persistent anchor `i`，
先从 carried prior 产生 query：

```math
q_{t,i}^{anc,0}
=
MLP_{anc-seed}
\Big[
\bar h^-_{t,i},
W_\mu \bar\mu^-_{t,i},
W_\Sigma \log diag(\bar\Sigma^-_{t,i}),
PE_{anc}(x^-_{t,i}, a^-_{t,i}, S^-_{t,i}),
\alpha^-_{t,i}
\Big]
```

然后从 current observation anchors 读取：

```math
\alpha_{t,i,n}^{anc}
=
\operatorname{softmax}_n
\left(
\frac{(W_q q_{t,i}^{anc,0})^\top (W_k o_{t,n}^{obs})}{\sqrt d}
+
\lambda_B \log(B_{t,i n}+\epsilon_A)
\right)
```

```math
m_{t,i}^{anc}
=
\sum_n \alpha_{t,i,n}^{anc} V o_{t,n}^{obs}
```

再更新 anchor evidence token：

```math
e_{t,i}^{anc}
=
GRU_{anc}(q_{t,i}^{anc,0}, m_{t,i}^{anc})
```

---

## 4.2 当前几何状态更新

从 bound observation anchors 中回读当前几何：

```math
x_{t,i}
=
\frac{\sum_n B_{t,i n} x_{t,n}^{obs}}
{\sum_n B_{t,i n}+\epsilon_A}
```

```math
S_{t,i}
=
\frac{
\sum_n B_{t,i n}
\Big[
S_{t,n}^{obs}
+
(x_{t,n}^{obs}-x_{t,i})(x_{t,n}^{obs}-x_{t,i})^\top
\Big]
}{
\sum_n B_{t,i n}+\epsilon_A
}
+\epsilon_S I
```

```math
a_{t,i}
=
clip\Big(
2\sqrt{sort_{\downarrow}(eig(S_{t,i}))+\epsilon_{ext}},
a_{min},a_{max}
\Big)
```

---

## 4.3 contact as learned state，不再靠硬 tactile gate

anchor-level contact probability 从当前 evidence token 读出：

```math
p_{t,i}^{cnt}
=
\sigma\Big(
Head_{cnt}(e_{t,i}^{anc})
\Big)
```

它不再由硬阈值规则定义，
而是作为 posterior / downstream 可以直接使用的连续量。

---

## 4.4 Gaussian measurement committee

为了保留 v0.3 系列“类投票 + 不确定性”的优点，
但避免模态专家链过碎，
本版采用统一的 measurement vote heads。

对每个 persistent anchor `i`，
设有 `R_vote` 个 measurement subheads：

```math
(\hat\mu_{t,i}^{(r)}, R_{t,i}^{(r)}, \gamma_{t,i}^{(r)})
=
Head_{vote}^{(r)}(e_{t,i}^{anc}),
\qquad r=1,\dots,R_{vote}
```

其中：

- `\hat\mu_{t,i}^{(r)}`：第 `r` 张 measurement 票的均值
- `R_{t,i}^{(r)}`：其协方差
- `\gamma_{t,i}^{(r)}`：其 raw confidence logit

对各 vote 计算 agreement-aware reliability：

```math
a_{t,i}^{(r)}
=
-
\frac{1}{R_{vote}-1}
\sum_{s\neq r}
D_{sym}\Big(
\mathcal N(\hat\mu_{t,i}^{(r)},R_{t,i}^{(r)})
\Vert
\mathcal N(\hat\mu_{t,i}^{(s)},R_{t,i}^{(s)})
\Big)
```

```math
\beta_{t,i}^{(r)}
=
\sigma(\gamma_{t,i}^{(r)} + a_{t,i}^{(r)})
```

这样 measurement committee 的每一张票
都必须同时满足：

- 自己的预测可信
- 与其他票不矛盾

---

## 4.5 current posterior 的 information-form fusion

prior：

```math
\Lambda_{t,i}^{-}=(\bar\Sigma_{t,i}^{-})^{-1},\qquad
\eta_{t,i}^{-}=\Lambda_{t,i}^{-}\bar\mu_{t,i}^{-}
```

measurement committee aggregated in information form：

```math
\Lambda_{t,i}^{meas}
=
\sum_{r=1}^{R_{vote}}
\beta_{t,i}^{(r)} (R_{t,i}^{(r)})^{-1}
```

```math
\eta_{t,i}^{meas}
=
\sum_{r=1}^{R_{vote}}
\beta_{t,i}^{(r)} (R_{t,i}^{(r)})^{-1}\hat\mu_{t,i}^{(r)}
```

posterior：

```math
\Lambda_{t,i}
=
\Lambda_{t,i}^{-} + \Lambda_{t,i}^{meas}
```

```math
\eta_{t,i}
=
\eta_{t,i}^{-} + \eta_{t,i}^{meas}
```

```math
\Sigma_{t,i}=\Lambda_{t,i}^{-1},\qquad
\mu_{t,i}=\Sigma_{t,i}\eta_{t,i}
```

于是：

```math
q_{t,i}^{post}(z)=\mathcal N(\mu_{t,i},\Sigma_{t,i})
```

这就是本版的 current posterior：
仍然是 Gaussian，
但 measurement 不再由 point/visual/tactile 三条硬专家链拼出来，
而是由统一 token 场读出的 evidence 经过 committee voting 形成。

---

## 4.6 recurrent memory writeback 与 posterior anchor token

Gaussian posterior 得到后，
还需要把 carried recurrent memory 写回为当前时刻 cache，
否则 `h_{t,i},c_{t,i}` 只会停留在上一步 prior，
而不会真正吸收当前 posterior 的新信息。

因此定义：

```math
(h_{t,i},c_{t,i})
=
\mathrm{LSTM}_{post}
\Big(
[
W_\mu \mu_{t,i},
W_\Sigma \log diag(\Sigma_{t,i}),
PE_{anc}(x_{t,i}, a_{t,i}, S_{t,i}),
W_\alpha \alpha_{t,i},
W_c p_{t,i}^{cnt}
],
(\bar h^-_{t,i},\bar c^-_{t,i})
\Big)
```

这个 writeback 使 persistent anchors 的 recurrent state
在当前步真正完成“先验 → 证据 → 后验 → 记忆写回”的闭环。

随后定义 posterior anchor token：

```math
token_{t,i}^{post}
=
Linear_{post}
\Big[
W_h h_{t,i},
W_\mu \mu_{t,i},
W_\Sigma \log diag(\Sigma_{t,i}),
PE_{anc}(x_{t,i}, a_{t,i}, S_{t,i}),
W_\alpha \alpha_{t,i},
W_c p_{t,i}^{cnt}
\Big]
```

这些 token 将在后续形成：

- global posterior context
- action state
- global future heads 的输入


---

# 5. Global Predictive-Innovation Module

这是本版相较 v0.3.11 的核心升级。

## 5.1 global posterior self-attention

所有 posterior anchor tokens
先进入一层全局 posterior self-attention：

```math
\tilde T_t^{post}
=
\mathrm{SelfAttn}_{post}(\{token_{t,i}^{post}\}_{i=1}^{K})
```

然后池化得到 global posterior context：

```math
g_t^{post}
=
\mathrm{Pool}_{post}(\tilde T_t^{post})
```

这一步的意义是：

- 当前 posterior anchors 之间互相交流
- 形成包含整个当前世界 belief 的全局状态
- 这个全局状态本身就是 future prediction 的根

你前面坚持“decoder 接在整体自注意力后面更好”，
本版完全按这个原则实现。

---

## 5.2 language-late predictive context

语言不进入 current posterior，
但在 posterior 已经固定以后，
它可以与 global posterior context 一起参与 **未来期待** 的构造。

为避免 `action \leftrightarrow future prediction` 的环路，
这里引入一个“用于 future heads 的选定动作”记号：

```math
a_t^{\star}
=
\begin{cases}
sg(a_t^{gt}), & \text{训练时 teacher forcing}\\
\hat a_t, & \text{推理时使用当前已选动作}
\end{cases}
```

于是先定义 predictive token set：

```math
\widetilde G_t^{pred}
=
\mathrm{SelfAttn}_{pred}
\Big[
g_t^{post},
W_l s_t^{txt},
W_{prop} p_t,
W_a a_t^{\star}
\Big]
```

再池化得到 global predictive context：

```math
g_t^{pred}
=
\mathrm{Pool}_{pred}(\widetilde G_t^{pred})
```

这里：

- `s_t^{txt}`：
  来自 PaliGemma 的语言摘要
- `p_t`：
  proprio / force context
- `a_t^{\star}`：
  训练时为 teacher-forced executed action，
  推理时为当前动作头输出 `\hat a_t`

因此语言只作用于：
**“从当前 posterior 去期待下一时刻什么”**，
而不作用于当前物理 posterior 本身。

注意：
具体前向顺序见 10.2 节——
动作头先基于当前 posterior 与 innovation 产生 `\hat a_t`，
然后 future heads 再用 `a_t^{\star}` 构造 `g_t^{pred}`。


---

## 5.3 global three-head future prediction

### 5.3.1 visual head：dual-head default

视觉 future prediction 默认包含两头。

#### (a) V-JEPA latent head

```math
\hat z_{t+1}^{v}
=
D_v^{lat}(g_t^{pred})
```

target 为下一时刻视觉 teacher latent：

```math
z_{t+1}^{v}
=
Pool\big(VJEPA2Encoder(I_{t-H_v+2:t+1})\big)
```

#### (b) optional real-visual head

```math
\hat y_{t+1}^{v,real}
=
D_v^{real}(g_t^{pred})
```

其 target 可选为：

- 下一时刻低分辨率 RGB crop
- dynamic-region target
- depth / motion map
- selected anchor-aligned visual patch target

本版的方法默认是 **dual-head**：
latent 头与轻量真实视觉辅助头同时存在。
若部署阶段出于算力或数据约束临时关闭真实视觉头，
那属于配置裁剪 / 消融，
而不是本版方法的默认设定。

### 5.3.2 tactile real-signal head

```math
\hat y_{t+1}^{t}
=
D_t^{real}(g_t^{pred})
```

target `y_{t+1}^{t}` 默认是真实 tactile signal，
例如：

- tactile image / map
- deformation proxy
- force-aware tactile target
- contact / pressure map

### 5.3.3 point / geometry real-signal head

```math
\hat y_{t+1}^{p}
=
D_p^{real}(g_t^{pred})
```

target `y_{t+1}^{p}` 默认为真实几何结果之一：

- next local depth / range image
- occupancy / TSDF
- local scene flow
- re-sampled point set（可选 Chamfer）

### 5.3.4 optional latent auxiliaries for tactile / point

若需要更强稳定性，
可再附加：

```math
\hat z_{t+1}^{t}=D_t^{lat}(g_t^{pred}),\qquad
\hat z_{t+1}^{p}=D_p^{lat}(g_t^{pred})
```

但它们是辅助项；
本版默认不允许 tactile / point 退化成“只预测 latent”。

---

## 5.4 no-rollout contract

本版强调：

- 训练时：
  使用一步 future supervision

- 推理时：
  仅缓存上一步对当前时刻的预测，
  与当前真实目标比较形成 innovation

因此，
运行时不是：

```math
\hat y_{t+1}\rightarrow\hat y_{t+2}\rightarrow\hat y_{t+3}\rightarrow\cdots
```

这种长链自回归 rollout，

而是：

```math
\hat y_t^{-}\ \text{vs.}\ y_t
```

的一步 innovation comparison。

这避免了“纯 latent 递归最终脱离真实世界”的在线塌缩风险。

---

## 5.5 explicit innovation construction

在时刻 `t`，
先取上一步缓存的预测：

```math
\hat z_t^{v,-},\quad
\hat y_t^{v,real,-},\quad
\hat y_t^{t,-},\quad
\hat y_t^{p,-}
```

当前真实 targets 为：

```math
z_t^{v},\quad
y_t^{v,real},\quad
y_t^{t},\quad
y_t^{p}
```

构造显式 residual：

```math
\epsilon_t^{v}=z_t^{v}-\hat z_t^{v,-}
```

```math
\epsilon_t^{v,real}=y_t^{v,real}-\hat y_t^{v,real,-}
```

```math
\epsilon_t^{t}=y_t^{t}-\hat y_t^{t,-}
```

```math
\epsilon_t^{p}=y_t^{p}-\hat y_t^{p,-}
```

为避免直接把高维 raw residual 生吞给 action head，
先分别编码：

```math
e_t^{v}=Enc_v^{err}[z_t^{v},\hat z_t^{v,-},\epsilon_t^{v}]
```

```math
e_t^{v,real}=Enc_{v,real}^{err}[y_t^{v,real},\hat y_t^{v,real,-},\epsilon_t^{v,real}]
```

```math
e_t^{t}=Enc_t^{err}[y_t^{t},\hat y_t^{t,-},\epsilon_t^{t}]
```

```math
e_t^{p}=Enc_p^{err}[y_t^{p},\hat y_t^{p,-},\epsilon_t^{p}]
```

再融合成 innovation token：

```math
e_t^{innov}
=
MLP_{innov}
\Big[
e_t^{v},
e_t^{v,real},
e_t^{t},
e_t^{p},
m_t^{v},m_t^{v,real},m_t^{t},m_t^{p}
\Big]
```

其中 `m_t^{(\cdot)}` 是对应 future-target 分支在当前步是否可用的 availability mask。
若某一分支 target 当前不可得（例如该模态缺失或该辅助头被关闭），
则对应 residual encoder 输出置零，mask 置 0，
innovation token 仅由可用分支构造。

### 5.5.1 为什么不从 forget gate 里读误差

LSTM 内部：

```math
c_t=f_t\odot c_{t-1}+i_t\odot \tilde c_t
```

其中 `f_t` / `i_t` 是内部控制量，
它们反映网络如何压缩与保留记忆，
不是可监督、可解释、可校准的 prediction error。
因此本版明确规定：

- forget gate 不是 innovation
- input gate 不是 innovation
- innovation 只能由显式 prediction-vs-real difference 构造

---

## 5.6 posterior + innovation 共同进入动作头

定义 joint control tokens：

```math
J_t=
\Big[
\tilde T_t^{post},
e_t^{innov},
W_l s_t^{txt},
W_{prop} p_t
\Big]
```

再做一层 control attention：

```math
\tilde J_t = \mathrm{SelfAttn}_{ctrl}(J_t)
```

池化得到最终 state：

```math
u_t = \mathrm{Pool}_{ctrl}(\tilde J_t)
```

动作头：

```math
\hat a_t = \pi(u_t)
```

因此 action head 不仅读到：

- 当前 posterior state

还读到：

- 当前 innovation（即“哪部分世界结果与上一步预测不一致”）

这就真正实现了你想要的：

**previous posterior carries prediction; current sensors upload truth; their discrepancy becomes an explicit token and is then routed by attention into control.**

---

## 5.7 predictive surprise

除了显式 innovation token，
仍可保留分布级 predictive comparison：

```math
s_{t,i}^{phys}
=
D_{KL}\!\Big(q_{t,i}^{post}(z)\ \Vert\ p_{t,i}^{curr-}(z)\Big)
```

若启用 language-conditioned task expectation：

```math
\hat q_{t,i}^{task}(z)=F_{task}(h_{t-1,i},a_{t-1},s_t^{txt})
```

则定义：

```math
s_{t,i}^{task}
=
D_{KL}\!\Big(q_{t,i}^{post}(z)\ \Vert\ \hat q_{t,i}^{task}(z)\Big)
```

但这些 `surprise` 默认用于：

- diagnostics
- gating / weighting
- optional task selector

而 innovation token 才是动作路径上的一级输入。

---

# 6. 不确定性、Gaussian 与“类投票”

## 6.1 posterior latent

定义 persistent anchor posterior latent：

```math
\mu_{t,i}=[h_{t,i}^{phys},g_{t,i}^{phys},c_{t,i}^{phys}]
```

默认维度：

- `dim(h^{phys}) = 48`
- `dim(g^{phys}) = 32`
- `dim(c^{phys}) = 32`
- `D_z = 112`

---

## 6.2 covariance parameterization

对 block `b\in\{h,g,c\}`：

```math
\sigma_{b,t,i}^2
=
clip(
softplus(\rho_{b,t,i})+\epsilon,
\sigma_{min}^2,
\sigma_{max}^2
)
```

默认：

```math
\epsilon=10^{-4},\qquad
\sigma_{min}^2=10^{-4},\qquad
\sigma_{max}^2=10
```

---

## 6.3 measurement committee 比单 expert 更适合本版

本版当前 evidence 已经是：

- 多模态统一 token 场
- 统一 observation anchors
- persistent anchor reading

因此 measurement 不再自然对应
“point expert / visual expert / tactile expert”三条大专家链。

更合理的形式是：

- 同一个 persistent anchor，
  从同一 evidence token `e_{t,i}^{anc}` 出发，
  用多个小 vote heads 给出多张 measurement 票；
- 每张票代表一个“可能的状态解释”；
- 其协方差和 agreement 共同决定最终投票力度。

这就是本版保留 Gaussian 的方式：
**Gaussian 表示的不只是 uncertainty，
还表示连续可加权的 committee votes。**

---

## 6.4 uncertainty-aware innovation scaling

为避免 innovation token 直接受各分支量纲支配，
本版统一使用“已标准化 residual”记号：

```math
\bar\epsilon_t^{m}
=
Norm_m(\epsilon_t^{m}),
\qquad m\in\{v, v_{real}, t, p\}
```

为避免与前文 `\epsilon_t^{m}` 的原始残差记号冲突，
后文统一用 `\bar\epsilon_t^{m}` 表示“送入 innovation token 的已标准化误差分量”。

其中 `Norm_m(\cdot)` 有两种合法实现：

### (a) uncertainty-aware 标准化（推荐）

若对应 future head 同时参数化预测协方差，
则：

```math
\bar\epsilon_t^{m}
=
(\Sigma_t^{m,pred}+\Sigma_t^{m,obs})^{-1/2}\epsilon_t^{m}
```

这里的 `\Sigma_t^{m,pred}` 与 `\Sigma_t^{m,obs}` 都是“实现中可得到的 prediction / observation 不确定性估计”的抽象记号；
总纲不强行规定它们必须以某一种具体协方差头实现。

### (b) deterministic fallback 标准化

若 future head 只输出 deterministic prediction，
则不强制引入额外 covariance head；
实现上可改用：

- running-variance normalization
- LayerNorm / RMSNorm
- per-branch learned scale

因此，后文统一用 `\bar\epsilon_t^{m}` 表示“已标准化 innovation residual”，
而不再把 uncertainty-aware 版本与 deterministic fallback 写成两套互相竞争的记号。

本节因此是 **数值实现约束**，
不是要求所有 future heads 都必须额外输出协方差。


---

## 6.5 contact 的地位

本版里 contact 既不是硬 gate，
也不只是日志。

它有三种角色：

1. 当前 posterior 的一个连续 readout：
   ```math
   p_{t,i}^{cnt}
   ```

2. tactile real-signal future head 的重要 target 之一

3. innovation token 的一部分，
   因为触觉真实结果与上一步预测的差异，
   往往正对应 contact dynamics 的变化

---

## 6.6 不确定性训练原则

为了避免模型只追求点误差、学会在不确定时“硬猜”，
本版的状态与预测损失优先使用 proper scoring：

- Gaussian NLL
- KL
- BCE + Brier
- calibration regularizer

而不是只看：

- L2
- success rate
- hard threshold accuracy

---

# 7. Tokens、context 与 downstream readout

## 7.1 posterior anchor token

已定义：

```math
token_{t,i}^{post}
=
Linear_{post}
\Big[
W_h h_{t,i},
W_\mu \mu_{t,i},
W_\Sigma \log diag(\Sigma_{t,i}),
PE_{anc}(x_{t,i}, a_{t,i}, S_{t,i}),
W_\alpha \alpha_{t,i},
W_c p_{t,i}^{cnt}
\Big]
```


---

## 7.2 global posterior token

```math
g_t^{post}
=
\mathrm{Pool}_{post}(\mathrm{SelfAttn}_{post}(\{token_{t,i}^{post}\}))
```

---

## 7.3 semantic token

若启用语言 side path：

```math
s_t^{sem}=Linear_{sem}(LN(\tilde s_t^{sem}))
```

但强调：

- `s_t^{sem}` 不参与 current posterior
- 它只参与 `g_t^{pred}` 和 downstream routing

---

## 7.4 innovation token

已定义：

```math
e_t^{innov}
=
MLP_{innov}
\Big[
e_t^{v},
e_t^{v,real},
e_t^{t},
e_t^{p},
m_t^{v},m_t^{v,real},m_t^{t},m_t^{p}
\Big]
```


---

## 7.5 最终 pooled state

```math
u_t
=
\mathrm{Pool}_{ctrl}
\Big(
\mathrm{SelfAttn}_{ctrl}
[
\tilde T_t^{post},
e_t^{innov},
W_l s_t^{txt},
W_{prop} p_t
]
\Big)
```

因此 `u_t` 同时含有：

- 当前 posterior belief
- 当前预测误差
- 语言条件的任务偏置
- proprio / control context

---

## 7.6 默认 action / progress head

默认动作表示：

```math
\hat a_t=[\widehat{\Delta x}_{ee,t},\ \widehat{\Delta r}_{ee,t}^{axis-angle},\ \hat g_t]\in\mathbb R^7
```

默认 head：

- 2-layer Transformer 或 MLP
- 读取 `u_t`
- 不回写 current posterior

如需 progress / termination / stage-head，
也从 `u_t` 读出，
而不是单独绕开 posterior。

---

# 8. Loss 设计

## 8.1 总损失

```math
L
=
\lambda_{act}L_{act}
+\lambda_{state}L_{state}
+\lambda_{pred}L_{pred}
+\lambda_{align}L_{align}
+\lambda_{bind}L_{bind}
+\lambda_{innov}L_{innov}
+\lambda_{unc}L_{unc}
+\lambda_{reg}L_{reg}
```

---

## 8.2 action imitation / control loss

把动作向量分解为：

```math
\hat a_t=[\widehat{\Delta x}_{ee,t},\ \widehat{\Delta r}_{ee,t}^{axis-angle},\ \hat g_t]
```

则：

```math
L_{act}
=
\|\widehat{\Delta x}_{ee,t}-\Delta x_{ee,t}^{gt}\|_1
+\lambda_{rot}\|\widehat{\Delta r}_{ee,t}^{axis-angle}-\Delta r_{ee,t}^{gt,axis-angle}\|_1
+\lambda_g|\hat g_t-g_t^{gt}|
```

若动作分布采用 diffusion / flow matching，
则对应替换为该类 loss。


---

## 8.3 state loss：posterior 几何与接触监督

若有相应可得监督：

```math
L_{state}
=
L_{geom}
+\lambda_{cnt}L_{cnt}
+\lambda_{force}L_{force}
+\lambda_{slip}L_{slip}
```

其中：

- `L_{geom}`：center / extent / occupancy / TSDF
- `L_{cnt}`：contact probability
- `L_{force}`：local force proxy
- `L_{slip}`：slip classification

若没有显式 object annotation，
可采用弱监督几何 target 或 reconstruction-style proxy。

---

## 8.4 future prediction loss

### 8.4.1 visual latent head

```math
L_{pred}^{v,lat}
=
\|\hat z_{t+1}^{v}-sg(z_{t+1}^{v})\|_2^2
```

### 8.4.2 visual real auxiliary head

```math
L_{pred}^{v,real}
=
\|\hat y_{t+1}^{v,real}-y_{t+1}^{v,real}\|_1
```

### 8.4.3 tactile real head

```math
L_{pred}^{t}
=
\mathcal L_{tactile-real}(\hat y_{t+1}^{t}, y_{t+1}^{t})
```

其中可包含：

- image/map L1
- force-aware loss
- contact BCE
- deformation structure loss

### 8.4.4 point / geometry real head

```math
L_{pred}^{p}
=
\mathcal L_{point-real}(\hat y_{t+1}^{p}, y_{t+1}^{p})
```

默认可由以下组合构成：

- depth/range L1
- occupancy BCE
- TSDF regression
- scene-flow loss
- optional Chamfer on re-sampled point set

### 8.4.5 total prediction loss

```math
L_{pred}
=
\lambda_{v,lat}L_{pred}^{v,lat}
+\lambda_{v,real}L_{pred}^{v,real}
+\lambda_t L_{pred}^{t}
+\lambda_p L_{pred}^{p}
```

---

## 8.5 innovation consistency loss

为使 innovation token 真正与 future prediction residual 对齐，
定义：

```math
L_{innov}
=
\left\|
Head_{innov-dec}(e_t^{innov})
-
[\bar\epsilon_t^{v},\bar\epsilon_t^{v,real},\bar\epsilon_t^{t},\bar\epsilon_t^{p}]
\right\|_2^2
```

其中 `\bar\epsilon_t^{m}` 是第 6.4 节定义的“已标准化 innovation residual”。

若某一分支当前被 mask 或在配置中关闭，
则对应分量从监督向量中删除，
并由同一 availability mask 控制 `Head_{innov-dec}` 的监督维度。

这不是必须项，
但推荐在训练早期加入较小权重，
帮助 `e_t^{innov}` 学成“误差摘要”而不是随便的额外 token。


---

## 8.6 geometry-first cross-modal auxiliary losses

### 8.6.1 projective compatibility target

对 point `m` 与 visual patch `u`，
先以 calibrated projection 构造 soft projective compatibility：

```math
G_{t,m,u}^{proj}
=
\begin{cases}
0, & \chi_{t,m}^{vis}=0,\\
\exp\Big(
-\frac{\|\tilde\pi_t(x_{t,m})-\tilde c_{t,u}^{grid}\|_2^2}{2\sigma_{proj}^2}
\Big), & \chi_{t,m}^{vis}=1
\end{cases}
```

也就是说，当 `\chi_{t,m}^{vis}=0` 时，默认**直接定义** `G_{t,m,u}^{proj}=0`，
并沿用 0.4.2c 节的 dummy-grid / null-branch 约定，
即不再把该 pair 的 `\tilde\pi_t(x_{t,m})`、`\pi_t(x_{t,m})` 或 depth residual 当作有效几何量参与后续计算。

若当前步可用 depth image 或可靠深度一致性检查，
则默认采用 **soft depth factor**，
而不是一刀切的硬阈值截断：

```math
G_{t,m,u}^{proj}
\leftarrow
G_{t,m,u}^{proj}
\cdot
\exp\Big(
-\frac{\big(z_t(x_{t,m})-d_t^{img}(\pi_t(x_{t,m}))\big)^2}{2(\tau_{z}^{proj})^2}
\Big)
```

其中 `d_t^{img}(\pi_t(x_{t,m}))` 表示当前深度图在投影位置的双线性采样值。
该 depth factor 只对 `\chi_{t,m}^{vis}=1` 的 pair 评估；
若当前样本没有可靠深度图，
则上式中的 depth factor 置为 1。
若实现里 `\chi_t^{depth}(x)` 本身已经由同一 depth residual 的硬阈值构造，
则应将该硬阈值关闭，
或把 `\chi_t^{depth}(x)` 降级成仅表示“depth 可用 / 粗 occlusion 有效”的 validity mask，
避免对同一深度偏差同时施加 hard cutoff 与 soft Gaussian factor。

为避免把大量“几何上只是远离投影中心的 pair”误当作显式负例，
本版引入稀疏 candidate edge set：

```math
\mathcal E_t^{pv}
=
\big\{
(m,u)\,\big|\,
G_{t,m,u}^{proj}>\tau_{proj}
\big\}
```

`L_{anc}^{pv}` 与 `L_{pv}^{weak}` 默认只在 `\mathcal E_t^{pv}` 上求和；
集合外 pair 被 **忽略**，
而不是被标成硬负例。
工程实现上，`\mathcal E_t^{pv}` 也应通过 `\tilde\pi_t(x_{t,m})` 邻域内的 radius / top-k patch 检索来近似构造，
而不是显式枚举所有 `M_p\times U_v` pair。
这一步正是为了同时降低：

- 遮挡 / patch aggregation 带来的假阴性
- 多个 3D 点投到相近 patch 时的伪一对一约束
- 深度噪声导致的过度排斥

这一步把 legacy `visual_expert` 里已有的投影、边界检查与 depth residual 逻辑
提升为新 core 的几何监督来源。

### 8.6.2 anchor-level point-visual consistency

为避免把用于几何 readout 的 within-point 归一化权重误当作跨模态 routing 概率，
本版把 **geometry ownership** 与 **cross-modal routing** 分开定义。

首先，从 unified attention 直接提取 point / visual 的 raw routing masses：

```math
\widetilde A_{t,n,m}^{route,p}
=
\alpha_{t,n,m}^{obs}\,\mathbf 1[m\in\mathcal P_t]
```

```math
\widetilde A_{t,n,u}^{route,v}
=
\alpha_{t,n,u}^{obs}\,\mathbf 1[u\in\mathcal V_t]
```

其中 `\mathcal P_t` 与 `\mathcal V_t` 分别是 point / visual token index sets。

再沿 anchor 维做 token-wise responsibility normalization：

```math
\bar A_{t,n,m}^{route,p}
=
\frac{\widetilde A_{t,n,m}^{route,p}}
{\sum_{n'=1}^{N_{obs}}\widetilde A_{t,n',m}^{route,p}+\epsilon_A}
```

```math
\bar A_{t,n,u}^{route,v}
=
\frac{\widetilde A_{t,n,u}^{route,v}}
{\sum_{n'=1}^{N_{obs}}\widetilde A_{t,n',u}^{route,v}+\epsilon_A}
```

同时定义 point / visual token 在所有 anchors 上获得的总 routing support mass：

```math
s_{t,m}^{route,p}=\sum_{n=1}^{N_{obs}}\widetilde A_{t,n,m}^{route,p}
```

```math
s_{t,u}^{route,v}=\sum_{n=1}^{N_{obs}}\widetilde A_{t,n,u}^{route,v}
```

并将其映射为 soft support gates：

```math
\omega_{t,m}^{route,p}=\frac{s_{t,m}^{route,p}}{s_{t,m}^{route,p}+\tau_{route}^{p}}
```

```math
\omega_{t,u}^{route,v}=\frac{s_{t,u}^{route,v}}{s_{t,u}^{route,v}+\tau_{route}^{v}}
```

再定义 point `m` 与 visual patch `u` 路由到同一 observation anchor 的一致性分数：

```math
R_{t,m,u}^{anc}
=
\omega_{t,m}^{route,p}\,\omega_{t,u}^{route,v}
\sum_{n=1}^{N_{obs}}
\bar A_{t,n,m}^{route,p}\,
\bar A_{t,n,u}^{route,v}
```

于是有：

```math
0\le R_{t,m,u}^{anc}\le 1
```

这样定义的 `R_{t,m,u}^{anc}` 既反映“两个 token 是否真的被同一 anchor 吸走”，
又显式抑制了“总 routing support 极低，但在沿 anchor 维归一化后看起来像同锚”的假阳性。

因此默认的 point-visual 主辅助 loss 写成：

```math
L_{anc}^{pv}
=
\sum_{(m,u)\in\mathcal E_t^{pv}}
w_{t,m,u}^{proj}\,
BCE\big(R_{t,m,u}^{anc},\ G_{t,m,u}^{proj}\big)
```

其中：

```math
w_{t,m,u}^{proj}=G_{t,m,u}^{proj}
```

若实现上需要显式控制最大样本权重，
可再做 `clip(w_{t,m,u}^{proj},0,w_{max})`；
默认 `w_{max}=1` 时该裁剪不改变数值。

这样 supervision 的目标变成：

- projectively compatible 的 point / visual token
  更容易被吸入同一个 observation anchor

而不是：

- 它们必须在 embedding 空间几乎重合

这更贴合当前 point-led world anchoring 架构。

### 8.6.3 optional focus loss on point↔visual attention

若 `FusionTransformer` 或专门的 point↔visual cross-attention 层
能导出 attention map，
则可加入 RayI2P 风格的 focus loss。

设第一层或平均后的 point↔visual attention slice 为：

```math
H_{t,u,m}^{pv}
```

则定义：

```math
L_{focus}^{pv}
=
-\sum_{u:\exists m,(m,u)\in\mathcal E_t^{pv}}
\log
\frac{
\sum_m H_{t,u,m}^{pv} G_{t,m,u}^{proj} + \epsilon_A
}{
\sum_m H_{t,u,m}^{pv} + \epsilon_A
}
```

实现上，`H_{t,u,m}^{pv}` 应先对 sensor-missing token 与 `\chi_{t,m}^{vis}=0` 的 point token 做 mask。
它的作用不是替代 `PE_{proj}` / `PE_{ray}`，
而是监督网络“把 point↔visual attention 质量集中到 projectively plausible neighborhood 上”。

这正对应 RayI2P 的启示：
即使存在 attention / matching，
也不应把每个 image patch 强行压成一个单点 correspondence，
而应优先保留 ray / projective neighborhood 结构。

### 8.6.4 optional weak bag-level alignment

若训练早期担心视觉分支参与不足，
可以保留一个小权重、短 warmup 的 bag-level contrastive 项。

对 visual patch `u`，
把其 candidate ray-bag 做加权池化：

```math
\bar g_{t,u}^{p|ray}
=
\frac{
\sum_{m:(m,u)\in\mathcal E_t^{pv}} G_{t,m,u}^{proj} g_{t,m}^{p}
}{
\sum_{m:(m,u)\in\mathcal E_t^{pv}} G_{t,m,u}^{proj}+\epsilon_A
}
```

只对拥有非空 candidate bag 的 visual patch 求该项。
其中 `\mathcal N_{neg}^{v}(u)` 表示 patch `u` 的 negative visual patch 集合；
实现上默认从当前 batch 的其他 visual patch 中采样，
并排除与 `u` 在 projective neighborhood 上高度重叠的 patch。
于是弱辅助对齐项为：

```math
L_{pv}^{weak}
=
\sum_{u:\exists m,(m,u)\in\mathcal E_t^{pv}}
-\log
\frac{
\exp(sim(\bar g_{t,u}^{p|ray}, g_{t,u}^{v})/\tau_{pv})
}{
\exp(sim(\bar g_{t,u}^{p|ray}, g_{t,u}^{v})/\tau_{pv})
+\sum_{u'\in\mathcal N_{neg}^{v}(u)}
\exp(sim(\bar g_{t,u}^{p|ray}, g_{t,u'}^{v})/\tau_{pv})
}
```

注意：
这里 supervision 的正例对象是 **ray-bag / projective neighborhood**，
不是单个点与单个 patch 的硬一对一匹配。

### 8.6.5 point-tactile

利用同步 pose / contact 自动生成 point↔tactile 正例。
记 `\mathcal P\mathcal T_t` 为由同步 pose / contact 自动构造的 point↔tactile 正例集合，
`\mathcal N_{neg}^{t}(m)` 为对应 point `m` 的 negative tactile sample 集合。
则：

```math
L_{pt}
=
\sum_{(m,k)\in \mathcal P\mathcal T_t}
-\log
\frac{
\exp(sim(g_{t,m}^{p}, g_{t,k}^{t})/\tau_{pt})
}{
\exp(sim(g_{t,m}^{p}, g_{t,k}^{t})/\tau_{pt})
+\sum_{k'\in \mathcal N_{neg}^{t}(m)}\exp(sim(g_{t,m}^{p},g_{t,k'}^{t})/\tau_{pt})
}
```

### 8.6.6 total

```math
L_{align}
=
\lambda_{anc}^{pv}L_{anc}^{pv}
+\lambda_{focus}^{pv}L_{focus}^{pv}
+\lambda_{pv}^{weak}L_{pv}^{weak}
+\lambda_{pt}L_{pt}
```

默认建议为：

- `L_{anc}^{pv}`：主 auxiliary
- `L_{focus}^{pv}`：若 attention map 可导出则启用
- `L_{pv}^{weak}`：小权重、短 warmup
- `L_{pt}`：维持原本 point-tactile 对齐作用

当前实现默认已经导出 fusion attention map，
因此 `L_{focus}^{pv}` 可直接启用；
若做消融或切回不导出 attention 的实现，
再把 `\lambda_{focus}^{pv}` 置为 `0`。

---

## 8.7 binding / anti-duplication / recycle losses

### 8.7.1 observation-anchor overlap penalty

```math
L_{dup}^{obs}
=
\sum_{n\neq n'}
\frac{
\langle A_{t,n,:}^{obs,p}, A_{t,n',:}^{obs,p}\rangle
}{
\|A_{t,n,:}^{obs,p}\|_1\|A_{t,n',:}^{obs,p}\|_1+\epsilon_A
}
```

### 8.7.2 persistent-binding overlap penalty

```math
L_{dup}^{pers}
=
\sum_{i\neq j}
\frac{
\langle B_{t,i,:}, B_{t,j,:}\rangle
}{
\|B_{t,i,:}\|_1\|B_{t,j,:}\|_1+\epsilon_A
}
```

### 8.7.3 activity sparsity

```math
L_{act-sparse}=\sum_i \alpha_{t,i}
```

### 8.7.4 residual coverage

避免所有 observation anchors 都掉进 dustbin：

```math
L_{cover}
=
-\frac{1}{N_{obs}}\sum_n \log(1-B_{t,\varnothing n}+\epsilon_A)
```

### 8.7.5 recycle smoothness

```math
L_{rec}
=
\sum_i |\rho_{t,i}^{rec}-\rho_{t-1,i}^{rec}|
```

### 8.7.6 total binding loss

```math
L_{bind}
=
\lambda_{dup}^{obs}L_{dup}^{obs}
+\lambda_{dup}^{pers}L_{dup}^{pers}
+\lambda_{act-sparse}L_{act-sparse}
+\lambda_{cover}L_{cover}
+\lambda_{rec}L_{rec}
```

---

## 8.8 uncertainty loss

### 8.8.1 Gaussian NLL

当存在 anchor-aligned state supervision 时，
定义：

```math
L_{unc}^{state}
=
-\sum_i \log q_{t,i}^{post}(y_{t,i}^{state})
```

其中 `y_{t,i}^{state}` 表示当前可用的 anchor-level state target，
可由几何、接触或弱监督世界状态构成。
若某类状态监督在当前数据集上不可得，
则对应项可被掩蔽或替换为等价的 readout-level proper-scoring loss。


### 8.8.2 contact calibration

```math
L_{unc}^{cnt}
=
BCE(p_{t,i}^{cnt}, y_{t,i}^{cnt})
+\lambda_{brier}\,Brier(p_{t,i}^{cnt}, y_{t,i}^{cnt})
```

### 8.8.3 vote regularization

避免 committee votes 全部 collapse 为同一张票，
采用 margin-based diversity regularizer：

```math
L_{vote-div}
=
\sum_i \sum_{r<s}
\max\Big(
0,
\tau_{vote}^{div}
-
D_{sym}\big(q_{t,i}^{(r)} \Vert q_{t,i}^{(s)}\big)
\Big)
```

这样只有当两张票彼此过近时才产生惩罚，
避免使用“负散度”带来的无界目标。


总 uncertainty loss：

```math
L_{unc}
=
\lambda_{nll}L_{unc}^{state}
+\lambda_{cntcal}L_{unc}^{cnt}
+\lambda_{votediv}L_{vote-div}
```

---

## 8.9 regularization

```math
L_{reg}
=
\sum_i
\left(
\|\mu_{t,i}\|_2^2
+
\sum_{b\in\{h,g,c\}} |\log \sigma_{b,t,i}^2|
\right)
+
\|g_t^{pred}\|_2^2
+\|e_t^{innov}\|_2^2
```

---

## 8.10 augmentation protocol

- 对局部点云做小刚体扰动
- 对点颜色做轻量 photometric jitter（不再使用 color drop，也不做 RGB 置零训练）
- 对视觉输入做轻量 motion-consistent crop / brightness jitter
- 对 tactile 图像做 gain / illumination jitter
- 保持未来真实 targets 与 teacher targets 的同步一致
- 不允许 future target 在 student 输入侧泄露

---

# 9. 训练：单阶段端到端

## 9.1 总原则

本版默认采用**单阶段端到端训练**：

- 不再划分 Stage 0 / 1 / 1.25 / 1.5 / 1.75 / 2
- 所有主损失从一开始就在同一训练图中存在
- 仅允许通过 warmup / coefficient ramp
  调节损失比重，
  但不允许改变图结构

这满足你要求的
“像 LSTM 一样，一张图直接训成”。

---

## 9.2 可训练模块

默认可训练部分：

- unified token adapters
- FusionTransformer
- observation anchor queries / GRU
- persistent anchor prior / update
- Gaussian vote heads
- global posterior self-attention
- global predictive heads
- innovation encoders
- action / progress heads

预训练 encoder 默认策略：

- V-JEPA：冻结或极小学习率 / LoRA
- Sonata：冻结主体，仅训练 xyzrgb contract adapter 与轻量投影
- AnyTouch：冻结主体或极小学习率
- PaliGemma：冻结，仅取 side summary 或轻量 LoRA

注意：
这仍然是**单阶段端到端**，
只是参数组学习率不同，
不是多阶段 curriculum。

---

## 9.3 优化设置

默认：

- optimizer：`AdamW`
- betas：`(0.9, 0.95)`
- weight decay：`1e-4`
- grad clip：`1.0`
- lr schedule：`2%` warmup + cosine decay
- mixed precision：允许
- Gaussian / KL / proper-scoring 部分使用 FP32 accumulator

参数组默认学习率：

- unified core：
  `2e-4`

- future heads / innovation heads / action heads：
  `2e-4`

- projection adapters：
  `1.5e-4`

- LoRA on frozen backbones：
  `5e-5`

---

## 9.4 单阶段训练时的权重 warmup

虽然不做多阶段，
但可以做**连续型权重 warmup**：

前期较重：

- `L_act`
- `L_bind`
- `L_{anc}^{pv}`
- `L_{focus}^{pv}`（若启用）

中期逐渐抬高：

- `L_pred`
- `L_unc`
- `L_innov`
- `L_{pt}`

后期稳定：

- `L_state`
- `L_pred`
- `L_unc`

同时让 `L_{pv}^{weak}` 只在前 `10\%\sim20\%` 的训练 steps 内启用（默认 `warmup_{pv}^{weak}=0.15`），
并在该 warmup window 结束前余弦衰减到 0，
避免它长期反客为主。

这是连续权重调度，
不是阶段切换。

---

## 9.5 数据需求与“几乎不需要额外标注”

本版默认只需要 demonstration 数据自带的：

- 视频 / RGB
- 点云 / depth
- tactile history
- force / proprio
- executed action
- 时间戳 / 标定

不需要额外人工标注：

- object mask
- instance ID
- contact hand-label
- future frame annotation

future supervision 完全来自下一时刻真实观测与 teacher encoder target。

---

## 9.6 为什么默认不会走向“纯 latent collapse”

这是本版坚持 tactile / point 预测真实信号的直接原因。

若所有 future heads 都只预测 latent，
训练中容易发生：

- latent 自我对齐越来越好
- 但对真实世界的可复现性逐渐下降
- 多步递归时更容易出现 representation collapse

因此本版规定：

- tactile：真实信号优先
- point：真实几何目标优先
- visual：dual-head default

同时运行时不做长链 rollout，
只做一步 prediction vs. current truth 的 innovation。
这进一步降低了 latent 自递归漂移风险。

---

# 10. 推理、缓存与部署 contract

## 10.1 cache 内容

### persistent anchor cache

```math
anchor\_state_t
=
\{
h_t,\ c_t,\ \mu_t,\ \Sigma_t,\ x_t,\ S_t,\ a_t,\ \alpha_t
\}
```

### future prediction cache

```math
pred\_cache_t
=
\{
\hat z_{t+1}^{v},
\hat y_{t+1}^{v,real},
\hat y_{t+1}^{t},
\hat y_{t+1}^{p}
\}
```

### semantic cache

```math
sem\_cache_t
=
\{
s_t^{txt},\ \tilde s_t^{sem}
\}
```

### runtime metadata

```math
runtime\_meta_t
=
\{
t_v^{last},\ t_p^{last},\ t_t^{last}
\}
```

说明：
与 `xyzrgb` 生成、RGB-点云配准、投影残差相关的诊断量，
应留在上游感知/标定链中维护；
本版 core 的 runtime metadata 只保留真正影响在线时序逻辑的多模态时间戳。

---

## 10.2 每步推理过程

1. 读取当前 `RGB / point / tactile / proprio / action-context`

2. 构造：
   - point tokens
   - visual tokens
   - tactile tokens
   - context tokens

3. 统一到同一隐空间：
   ```math
   U_t \rightarrow \tilde U_t
   ```

4. 由当前 point seeds 形成 observation anchors：
   ```math
   \tilde U_t \rightarrow \{o_{t,n}^{obs}\}
   ```

5. 从 observation anchors 读显式几何：
   `x_{t,n}^{obs}, S_{t,n}^{obs}, a_{t,n}^{obs}`

6. 由上一步 persistent cache 构造 carried prior：
   `p_{t,i}^{curr-}`

7. 做 observation↔persistent 的软绑定：
   `B_t`

8. 计算 recycle / activity：
   `\rho_{t,i}^{rec}, \alpha_{t,i}`

9. persistent anchors 从 bound observations 读证据，
   形成 `e_{t,i}^{anc}`

10. Gaussian measurement committee 给出 measurement proposals

11. information-form fusion 形成当前 posterior：
    `q_{t,i}^{post}`

12. recurrent memory writeback，
    得到当前 `h_{t,i},c_{t,i}`，
    并构造 posterior anchor tokens：
    `token_{t,i}^{post}`

13. posterior anchors 自注意力得到 `g_t^{post}`

14. 若存在上一步预测缓存，
    则用当前真实 targets 形成 innovation residual，
    再编码为 `e_t^{innov}`；
    若是首步，
    则 `e_t^{innov}=0`

15. 读取或刷新当前语言 side summary（若指令未变化则可直接复用 cache）

16. action head 读取：
    `posterior tokens + innovation token + language token + proprio`

17. 输出当前动作 `\hat a_t`

18. 设用于 future heads 的动作条件为：
    - 训练时 `a_t^{\star}=sg(a_t^{gt})`
    - 推理时 `a_t^{\star}=\hat a_t`

19. 将 `g_t^{post}`、语言 side summary、proprio 与 `a_t^{\star}` 融合形成 `g_t^{pred}`

20. 用三头 future module 预测：
    - `\hat z_{t+1}^{v}`
    - `\hat y_{t+1}^{v,real}`
    - `\hat y_{t+1}^{t}`
    - `\hat y_{t+1}^{p}`

21. 更新：
    - persistent anchor cache
    - future prediction cache
    - semantic cache


---

## 10.3 stale / missing-sensor policy

本版只保留三类必要 hard masks：

### 10.3.1 true sensor dropout

若某模态当前帧真的缺失，
对应 token stream 直接置空或 masked out。

### 10.3.2 xyzrgb contract violation

本版 **不提供 point-cloud RGB fallback**。

若 `xyzrgb` 生成失败、RGB 对齐失效、
或输入流只能提供 `xyz` 而不能稳定提供 `rgb`，
则当前样本 / 当前 rollout 不满足本版输入契约。

在这种情况下，
应当在数据层或感知层先修复输入，
而不是在 core 内把 `rgb` 置零继续运行。

### 10.3.3 timing invalidity

若某模态超时：

- 不强行构造伪 token
- 只保留 timing context token 告诉系统“该模态当前 stale”

这和 v0.3.11 的大段 stale expert policy 相比更统一：
本版不再在 runtime 中走多条 stale-if/else expert 分支，
而是让 unified token field 和 timing tokens 自己处理大多数 stale 情况。

---

## 10.4 uncertainty-driven safe fallback

定义：

### (a) posterior uncertainty spike

```math
hold_t^{state}
=
\mathbf 1\Big[
\frac{1}{K}\sum_i \operatorname{tr}(\Sigma_{t,i}) > \tau_{unc}^{state}
\Big]
```

### (b) innovation spike

```math
hold_t^{innov}
=
\mathbf 1[\|e_t^{innov}\|_2 > \tau_{innov}^{hold}]
```

### (c) active anchor collapse

```math
hold_t^{act}
=
\mathbf 1\Big[
\sum_i \alpha_{t,i} < \tau_{\alpha}^{min}
\Big]
```

### (d) true sensor blackout

```math
hold_t^{miss}
=
\mathbf 1[\text{critical sensors unavailable}]
```

最终：

```math
hold_t = hold_t^{miss} \lor hold_t^{state} \lor (hold_t^{innov}\land hold_t^{act})
```

即：

- innovation 单独不会立刻触发 hold
- innovation 只作为放大器
- 真正触发 hold 的仍是
  “高不确定 + 无法稳定解释当前世界”

---

## 10.5 action supervisor

所有动作都经过：

- delta-position clip
- delta-rotation clip
- gripper clip
- velocity / acceleration clip
- force / torque clip

默认：

```math
\|\Delta x_{ee}\|_2 \le 2.5\times 10^{-2}\text{ m}
```

```math
\|\Delta r_{ee}^{axis-angle}\|_2 \le \pi/18\text{ rad}
```

```math
|g_t|\le 1
```

---

# 11. 默认超参与 Tensor Contract

## 11.1 默认超参

### core sizes

- `K = 16`：
  persistent anchors
- `N_obs = 24`：
  current observation anchors
- `D = 256`：
  unified token hidden size
- `D_post = 256`：
  recurrent anchor hidden size
- `D_innov = 256`：
  innovation token size
- `D_u = 256`：
  pooled control state size
- `D_z = 112`：
  posterior latent dimension
- `R_vote = 4`

### target dimensions（实现依赖）

- `D_v`：
  视觉 latent teacher 的 pooled dim（由所选 V-JEPA pooling 决定）
- `D_{v,real}`：
  真实视觉辅助目标维度（由所选目标决定）
- `D_t`：
  真实 tactile 目标维度（由 tactile target 参数化决定）
- `D_p`：
  真实 point / geometry 目标维度（由 occupancy / TSDF / range / scene-flow 等参数化决定）

### recurrent / transformer

- `R_obs = 2`
- `FusionTransformer layers = 4`
- `posterior self-attn layers = 2`
- `control self-attn layers = 2`
- `heads = 8`

### geometry / binding

- `R_crop = 0.08 m`
- `R_ws = 0.5 m`
- `\epsilon_S = 1e-6`
- `\epsilon_A = 1e-6`
- `\epsilon_{ext}=1e-8 m^2`
- `z_{min}=1e-3 m`
- `z_{max}=10 m`
- `\tau_{route}^{p}=0.1`
- `\tau_{route}^{v}=0.1`
- `a_{min}=[0.005,0.005,0.005] m`
- `a_{max}=[2R_{ws},2R_{ws},2R_{ws}]`
- `\lambda_h = 1.0`
- `\lambda_x = 1.0`
- `\lambda_B = 0.5`
- `\sigma_{bind}=5e-3 m`
- `\alpha_{init}=0.05`

### uncertainty / scaling

- `\epsilon = 1e-4`
- `\sigma_{min}^2=1e-4`
- `\sigma_{max}^2=10`
- `\sigma_{reset}=1.0`
- `\tau_{vote}^{div}=0.5`

### safety / timing

- `\delta_{sync}^{max}=20 ms`
- `\tau_{unc}^{state}`：validation percentile calibrated
- `\tau_{innov}^{hold}`：validation percentile calibrated
- `\tau_{\alpha}^{min}=0.5`

### losses

- `\lambda_{act}=1.0`
- `\lambda_{state}=0.5`
- `\lambda_{pred}=0.5`
- `\lambda_{align}=0.1`
- `\lambda_{bind}=0.05`
- `\lambda_{innov}=0.1`
- `\lambda_{unc}=0.05`
- `\lambda_{reg}=1e-4`
- `\lambda_{rot}=1.0`
- `\lambda_g=1.0`
- `\lambda_{cnt}=0.5`
- `\lambda_{force}=0.2`
- `\lambda_{slip}=0.2`
- `\lambda_{dup}^{obs}=0.5`
- `\lambda_{dup}^{pers}=0.5`
- `\lambda_{act-sparse}=0.01`
- `\lambda_{cover}=0.1`
- `\lambda_{rec}=0.01`
- `\lambda_{nll}=1.0`
- `\lambda_{cntcal}=1.0`
- `\lambda_{votediv}=0.05`
- `\lambda_{brier}=0.1`
- `\lambda_{anc}^{pv}=1.0`
- `\lambda_{focus}^{pv}=0.5`
- `\lambda_{pv}^{weak}=0.2`
- `\lambda_{pt}=1.0`
- `\sigma_{proj}=1.5` patch（与 `\tilde\pi_t,\tilde c_{t,u}^{grid}` 的连续 patch-grid 坐标单位一致）
- `\tau_{proj}=0.25`
- `\tau_{z}^{proj}=0.01 m`
- `\tau_{pv}=0.07`
- `\tau_{pt}=0.07`
- `warmup_{pv}^{weak}=0.15`（fraction of total training steps）
- `w_{max}=1.0`

### prediction heads

- `\lambda_{v,lat}=0.2`
- `\lambda_{v,real}=0.1`
- `\lambda_t=0.3`
- `\lambda_p=0.3`


---

## 11.2 Tensor contract

### inputs

```math
I_{t-H_v+1:t}\in\mathbb R^{B\times H_v\times 3\times H_0\times W_0}
```

```math
P_t^{loc}\in\mathbb R^{B\times M_p\times 3}
```

```math
C_t^{loc}\in\mathbb R^{B\times M_p\times 3}
```

```math
T_{t-L_{tac}+1:t}\in\mathbb R^{B\times L_{tac}\times C_{tac}\times H_{tac}\times W_{tac}}
```

### token field

```math
U_t\in\mathbb R^{B\times M_U\times D}
```

```math
\tilde U_t\in\mathbb R^{B\times M_U\times D}
```

### observation anchors

```math
O_t^{obs}\in\mathbb R^{B\times N_{obs}\times D}
```

```math
A_t^{obs,p}\in\mathbb R^{B\times N_{obs}\times M_p}
```

### persistent anchors

```math
H_t,C_t\in\mathbb R^{B\times K\times D_{post}}
```

```math
\mu_t\in\mathbb R^{B\times K\times D_z}
```

```math
\Sigma_t\in\mathbb R^{B\times K\times D_z\times D_z}
```

```math
x_t\in\mathbb R^{B\times K\times 3}
```

```math
S_t\in\mathbb R^{B\times K\times 3\times 3}
```

```math
a_t\in\mathbb R^{B\times K\times 3}
```

```math
\alpha_t\in\mathbb R^{B\times K}
```

### binding

```math
B_t\in\mathbb R^{B\times (K+1)\times N_{obs}}
```

### future heads

```math
\hat z_{t+1}^{v}\in\mathbb R^{B\times D_v}
```

```math
\hat y_{t+1}^{v,real}\in\mathbb R^{B\times D_{v,real}}
```

```math
\hat y_{t+1}^{t}\in\mathbb R^{B\times D_t}
```

```math
\hat y_{t+1}^{p}\in\mathbb R^{B\times D_p}
```

### innovation

```math
e_t^{innov}\in\mathbb R^{B\times D_{innov}}
```

### downstream

```math
u_t\in\mathbb R^{B\times D_u}
```

```math
\hat a_t\in\mathbb R^{B\times 7}
```

---

# 12. 最小评测协议与关键消融

## 12.1 最小评测协议

### geometry / tracking

- center error
- extent error
- persistent anchor duplication rate
- anchor recycle precision
- continuity under occlusion
- projective anchor consistency AUC
- projected point↔visual routing hit rate
- intrinsics perturbation robustness

### contact / tactile

- contact F1 / IoU
- deformation prediction error
- force RMSE
- slip F1

### predictive ability

- one-step visual latent prediction error
- one-step tactile real-signal prediction error
- one-step point / geometry prediction error
- innovation-to-failure correlation

### downstream

- task success
- contact-rich completion rate
- unseen object / unseen geometry generalization

---

## 12.2 关键消融

1. 去掉 unified token field，
   改回 point / visual / tactile 三专家

2. 去掉 observation anchors，
   直接 persistent anchors 读 raw token field

3. 去掉 current posterior self-attention，
   只做 pooled mean

4. 去掉 global three-head future prediction

5. 只保留 visual latent 头，
   去掉 tactile / point real heads

6. tactile 改回 latent-only 预测

7. point 改回 latent-only 预测

8. 去掉 visual real auxiliary head

9. innovation 不进 action head，
   只做日志

10. innovation 从 forget gate 估计，
    不用显式 residual

11. 语言直接进入 current posterior

12. 语言完全不进入 future heads

13. 去掉 recycle gate，
    只保留固定 persistent anchors

14. 去掉 dustbin / null channel

15. 去掉 overlap penalties

16. 去掉 Gaussian committee，
    改成单 measurement head

17. 去掉 uncertainty calibration，
    只保留 L2

18. 推理时做多步 rollout，
    比较其与一步 innovation 方案的收益与稳定性

19. 去掉 `PE_{proj}` / `PE_{ray}`，
    只保留 `PE_{pt}` + `PE_{img}` + `PE_{cam}`

20. 保留 token-level `PE_{proj}` / `PE_{ray}`，
    但去掉 relative `b_{t,m,u}^{proj}`

21. 去掉 `L_{anc}^{pv}`，
    改回强 pairwise `L_{pv}`

22. 保留 `L_{anc}^{pv}`，
    去掉 `L_{focus}^{pv}`

23. 把 `L_{pv}^{weak}` 从 bag-level 改回单点 patch-level contrastive

---

# 13. 实现建议（PyTorch / runtime）

## 13.1 推荐模块划分

- `PointBackboneWrapper`
  - SONATA / SpatialLM implementation bridge
  - xyzrgb contract adapter
  - code-aligned tokenizer / patching wrapper
  - no extra handcrafted local-statistics front-end

- `VisualBackboneWrapper`
  - V-JEPA 2.1 encoder
  - visual tokenizer
  - optional visual real-target projector

- `TactileBackboneWrapper`
  - AnyTouch 2 encoder
  - tactile tokenizer
  - tactile real-target projector

- `UnifiedTokenField`
  - token projection to common hidden size
  - modality / position / timing embeddings
  - `PE_{proj}` / `PE_{ray}` token conditioning
  - optional point↔visual relative projective bias
  - shared FusionTransformer

- `ObservationAnchorModule`
  - FPS / point-seeded observation queries
  - unified cross-attention
  - observation-anchor self-attention
  - point-only geometry readout
  - optional anchor-level projective consistency loss hooks

- `PersistentPosteriorAnchorModule`
  - recurrent prior
  - observation↔persistent binding
  - recycle gate
  - Gaussian committee heads
  - posterior token builder

- `GlobalPredictiveInnovationModule`
  - posterior self-attn
  - language-late predictive context
  - visual latent head
  - optional visual real head
  - tactile real head
  - point real head
  - innovation token encoder

- `DownstreamHeads`
  - action head
  - progress / task heads
  - optional geometry / contact auxiliary heads

- `RuntimeSupervisor`
  - sensor-missing masks
  - uncertainty-driven hold
  - action clipper

---

## 13.2 推荐实现顺序

1. 先离线验证：
   - xyzrgb generation / calibration
   - world-point projection to image / patch grid
   - ray-direction generation from camera intrinsics
   - tactile pose addressing
   - teacher target generation链

2. 先把 legacy `visual_expert` 中已存在的 `load_camera_model()`、
   `_project_world_points()`、`_scale_to_grid()` 提炼成新 core 可复用几何原语
   并把 `src/openpi/picf/core/pipeline.py::_build_token_field()`
   从“`grid + cam_pose`”升级为
   “`grid + cam_pose + PE_{proj}/PE_{ray}`”

   这一步在当前代码中已经完成。

3. 单独跑 unified token field，
   确认三类 token 同维度、同 batch contract 无误，
   并验证 `PE_{proj}` / `PE_{ray}` 在真实 CALVIN 标定下数值稳定

   最低验证要求：

   - 对同一批 CALVIN 点云，
     新 core 的投影结果与 legacy `_project_world_points()` 的像素坐标一致
   - `\tilde\pi_t(x)` 经 patch-grid 量化后，
     与 visual token grid 的邻域覆盖率稳定、非塌缩
   - `d_{t,u}^{ray,B}` 的范数恒为 1，
     且在相机姿态固定时与 `PE_{cam}` 一致变化
   - `G_{t,m,u}^{proj}` 的候选边密度既不接近 0，
     也不接近“全图都候选”，
     否则说明 `\sigma_{proj}`、`\tau_{proj}` 或 depth consistency 设定不合理
   - `\chi_{t,m}^{vis}=0` 的点不会在 `PE_{proj}` 中产生 NaN / Inf，
     而是稳定地走 null projection branch
   - `\mathcal E_t^{pv}` 的平均候选边数在训练日志中可控、
     且不会退化成近似全连接

4. 跑 observation anchors，
   查看 duplication rate、dustbin 使用率、
   以及 projected point↔visual 是否更容易进入同一 anchor

5. 接 persistent anchors，
   验证 recycle 是否平稳

6. 当前实现默认已经启用 bias-capable attention、
   `b_{t,m,u}^{proj}` 与 `L_{focus}^{pv}`；
   若做消融，再切回只用 token-level `PE_{proj}` / `PE_{ray}` + `L_{anc}^{pv}`

7. 接 global three-head future prediction，
   先只看 one-step prediction 是否收敛

8. 最后接 innovation token 与 action head，
   验证 innovation 是否真的帮助决策

---

## 13.3 部署默认建议

- 若只有单 RGB 相机：
  先用单相机版本

- 若 `xyzrgb` 生成/对齐链不稳定：
  说明当前系统不满足本版输入契约；
  应先修复感知链，再部署本版模型

- 若 tactile 频率高于视觉：
  允许 tactile 连续更新 token field，
  但 action 仍按控制步输出

- 若 point-cloud 偶发缺帧：
  只用 timing token + missing mask，
  不再切换到复杂 stale expert 模式

- 若语言 grounding 不可靠：
  可以仅在 action selector 中使用语言，
  而暂时让 future heads 不读语言

- 若视觉真实头训练太难：
  先保留 visual latent 头 + tactile/point real heads，
  再逐步加入 visual real auxiliary head

---

# 14. 参考文献与官方接口（精选）

## 论文

- V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning (2026)
- Sonata: Self-Supervised Learning of Reliable Point Representations (2025)
- AnyTouch 2: General Optical Tactile Representation Learning for Dynamic Tactile Perception (2026)
- Cameras as Relative Positional Encoding / PRoPE (NeurIPS 2025)
- RAYI2P: Learning Rays for Image-to-Point Cloud Registration (ICLR 2026)
- VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model (2026)
- Video Predictive Embedding is Needed for VLA Models (2026)
- DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge (2025)
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (2023)
- Perceiver IO: A General Architecture for Structured Inputs & Outputs (2021)
- ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries (2026)
- Recursive Belief Vision Language Action Models (2026)
- MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning (2025)
- Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (2026)
- Reconstruction-Guided Slot Curriculum (2026)
- Spatially Anchored Tactile Awareness for Robust Dexterous Manipulation (2025/2026)
- OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation (2026)

## 官方文档 / 代码接口

- Hugging Face Transformers — V-JEPA 2 / `VJEPA2Model`
- facebookresearch/sonata Official Repository
- GeWu-Lab/AnyTouch2 Official Repository
- Hugging Face Transformers — PaliGemma Documentation

---

# 15. 本版最终判断

本版不认为：

- 把所有模态继续拆成很多 local expert 分支，
- 再加 staged curriculum，
- 再用大量 hard gate / lifecycle / stale rule

是适合你当前目标的做法。

本版认为更优折中是：

- **用统一 token 场让视觉、点云、触觉先进入同一隐空间**
- **用 calibrated projection / ray conditioning 先把 point↔visual 几何关系写进 token / attention**
- **用 observation anchors 形成当前世界概念**
- **用 persistent posterior anchors 维护跨时间 belief**
- **用 Gaussian committee 保留 uncertainty 与连续投票**
- **用 posterior 之后的全局自注意力承载未来预期**
- **用三个 global heads 预测下一时刻的重要目标**
- **其中 tactile 与 point 默认预测真实信号**
- **用显式 innovation token 把“预测与真实的差”送入动作头**
- **并把语言仅放在 posterior 之后的 predictive / selector 阶段**

因此，本版的关键判断是：

1. 最值得保留的不是 v0.3.11 的局部 query adapter，
   而是 **posterior 的整体预测性与不确定性表达**。

2. 最稳妥的训练方式不是多段 stage，
   而是 **单阶段端到端**，
   再通过 proper scoring、future heads、innovation token
   让模型自己学会：
   “什么该记、什么该预测、什么差异最重要”。

3. 对 point↔visual 关系，
   主力不应是强 pairwise `L_{pv}`，
   而应是 **projection/ray geometry first, weak alignment second**。
   更贴当前架构目标的弱辅助项
   是 `L_{anc}^{pv}` 与可选 `L_{focus}^{pv}`，
   而不是长期把 embedding similarity 当主任务。

4. 为避免纯 latent 递归塌缩，
   **触觉与点云 future heads 应默认预测真实信号**；
   视觉头则推荐 dual-head：
   latent + optional real target。

5. innovation 不应只作为日志，
   也不应从 forget gate 猜；
   它应成为**显式、可监督、可被 action head 消费的一等公民**。

如果未来还要升级，
最值得做的方向不是回到 support 或加更多小模块，
而是：

1. 把 global future heads 扩展成多步但仍 teacher-forced 的短 horizon prediction
2. 把 observation anchors 的形成从 point-seeded 扩展到更强的 OT / set-prediction 版本
3. 把 visual real head 由轻量目标升级成更结构化的 dynamic-region world target
4. 把 point real head 从 occupancy / TSDF 扩展到更强的 action-conditioned local scene-flow

---
