# CALVIN task_ABCD_D：K/E（相机内外参）导出与验证、URDF 对齐与桥接 Δ、训练↔部署等价、ZIP→点云重建（零重渲染）

> **适用数据**：`dataset/task_ABCD_D.zip`（≈650 GB）  
> **目标**：不解压大包、不重渲染；直接从 **Hydra + episode_*.npz + URDF** 构建训练与部署可复现的几何口径：  
> - 相机内参 **K**、外参 **E / W_T_C**（静态与腕相机）  
> - URDF 口径与数据集口径的固定桥接 **Δ**  
> - 点云生成/投影与跨视角一致性验收  
> - 强回归守门线：**训练 = 部署(+Δ)**（数值闭环）

---

## 0. 当前状态与结论（验收签名）

本 README 对应你已经跑通并输出 **PASS** 的那套口径与脚本（“训练=部署(+Δ)”回归线 + 深度一致性抽检）。

### 0.1 URDF q-不变（EE=15, cam=14）

> **含义**：URDF 中“末端执行器 EE link → 腕相机 link(14)”的相对位姿应与关节 q 无关（固定刚体连接）。  
> **验证**：PyBullet 随机关节采样，比较 `E(q)` 与 `E(0)` 的最大误差。

你机器已得到：

- `max_rot = 1.921e-05°`
- `max_trans = 5.545e-08 m`
- `max_fro = 4.752e-07 < 1e-6 ✅`

### 0.2 Δ 常量等价（任意关节）

> **含义**：用一个固定的桥接变换 Δ 使得任意关节下都有  
> `E(q) @ Δ ≡ E_ref`（数据集口径）  
> 其中 `E(q)` 是 URDF 口径的 EE→cam(14)。

你机器已得到：

- `max ||E(q)·Δ − E_ref||_F = 4.770e-07 < 1e-6 ✅`

### 0.3 逐帧几何等价：训练 = 部署(+Δ)

> **含义**：同一个 `robot_obs`（世界系下 EE 位姿）下：
> - 训练口径：`W_T_Cg^train = W_T_EE @ E_ref`
> - 部署口径：`W_T_Cg^deploy = W_T_EE @ E_raw`
> - 若部署侧需要“训练口径”：`W_T_EE @ (E_raw @ Δ) ≡ W_T_EE @ E_ref`

你机器已得到：

- `frames=240, max=2.511e-16, mean=1.790e-16 ≪ 1e-6 ✅`

### 0.4 跨视角深度一致性（典型抽检）

你机器在“强过滤/自适配/双线性”的版本上得到过典型值：

- `median ≈ 0.040 m`
- `p90 ≈ 0.060 m`
- `<5cm ≈ 72~74%`

> 说明：腕相机只有 **84×84**，遮挡/倾角/表面法线差会让 4–5 cm 的中位误差非常常见且合理。

### 0.5 分片与闭环（如果你已导出 gripper_poses-*.parquet）

- 13 片、总行数 `2,406,148`
- duplicates=0、NaN/Inf=0、旋转正交异常=0
- 手眼闭环 ~1e-7（工程可忽略）

### 0.6 重要约束（避免未来踩坑）

- **只用 cam=14 (gripper_cam)**；cam=12 (finger_tip) 已加 **DO NOT USE** 标记。
- 训练侧统一使用 `E_ref`；部署/仿真用 `E_raw`，如历史模块需要训练口径则右乘 `Δ`：
  - `E_raw @ Δ ≡ E_ref`（数值已回归 PASS）
- 欧拉顺序统一 **zyx**，单位 **弧度**。
- `robot_obs` 坐标系已验证为 **world**，不需要额外 base→world。

---

## 1. 推荐放置位置（你问的 “README 放哪里”）

这份 README 面向 “**标定产物（cameras.json）及其验证**”，最佳位置是 **跟标定文件同目录**：

- ✅ **推荐**：`~/datasets/calvin/dataset/task_ABCD_D/calib/README.md`  
  因为这里同时有：
  - `cameras.json`
  - `variants/`
  - `gripper_poses-*.parquet`
  - `vis/`（可选）

如果你还希望在仓库根目录也能一眼看到，可以再放一份软链接或复制：

- 可选：`~/datasets/calvin/README_CALVIN_GEOM.md`（根目录概览）
- 可选：在 `~/Documents/openpi/docs/` 里放一份 `CALVIN_DATASET_GEOM.md` 供训练工程团队参考（不影响数据产物）。

> 我无法直接确认你机器上“是否已经写进去”——因为我看不到你的本地文件系统；  
> 但你可以把我给出的 `README.md` 覆盖写入到上述路径，作为 **单一可信来源（single source of truth）**。

---

## 2. 目录结构与产物一览（最终约定）

以 `~/datasets/calvin` 为根（你之前就是这样布局的）：

```
~/datasets/calvin/
├─ dataset/
│  ├─ task_ABCD_D.zip                         # 650 GB：只流式读取，不解压
│  └─ task_ABCD_D/
│     └─ calib/
│        ├─ cameras.json                      # 主口径（训练用）：E_ref + meta + Δ
│        ├─ variants/
│        │  ├─ cameras_urdf_cam14_equiv.json  # URDF 口径（E_raw，ee=15, cam=14）
│        │  ├─ cameras_urdf_equiv.json        # cam=12 旧版（已 DO NOT USE）
│        │  └─ ...
│        ├─ gripper_poses-000000.parquet      # 腕相机位姿分片（共 13 片）
│        ├─ ... gripper_poses-000012.parquet
│        └─ vis/                              # 点云可视化 HTML（可选）
└─ tools/
   ├─ __init__.py
   ├─ calvin_geom.py                          # 几何通用工具（本 README 含源码）
   └─ verify_train_deploy_equivalence.py      # 强回归守门线（本 README 含源码）
```

---

## 3. 数据集结构（从 ZIP 流式读出的“事实”）

你抽样扫描的 `episode_*.npz`（training/validation 均一致）包含键：

- `rgb_static`：`(200, 200, 3)` `uint8`
- `rgb_gripper`：`(84, 84, 3)` `uint8`
- `depth_static`：`(200, 200)` `float32`
- `depth_gripper`：`(84, 84)` `float32`
- `rgb_tactile`：`(160, 120, 6)` `uint8`
- `depth_tactile`：`(160, 120, 2)` `float32`
- `robot_obs`：`(15,)` `float64`（注意：很多 episode 是单帧，因此是 1D）
- `scene_obs`：`(24,)` `float64`
- `actions / rel_actions`：`(7,)` `float64`

统计（你机器输出）：

- training episodes：`2,307,126`
- validation episodes：`99,022`

> 重要：因为 `robot_obs` 多数是 **单帧 (15,)**，任何“靠时间维度 std 去猜 pos/rpy 列”的脚本会给出 `votes=none`。  
> 我们的几何工具会把 1D reshape 成 `(1, -1)` 并直接取 `[0:6]`（pos+rpy），不依赖时间维度统计，因此可稳定工作。

---

## 4. 口径定义（训练/部署统一数学）

### 4.1 相机模型：针孔 + OpenCV 光学轴

- 像素坐标：`u` 为列（x），`v` 为行（y）
- 相机坐标（OpenCV）：`+x` 右、`+y` 下、`+z` 前（光轴）

深度（米）从像素回投影到相机坐标：

\[
X_c = Z \cdot K^{-1} [u, v, 1]^T
\]

世界坐标：

\[
X_w = W\_T\_C \cdot [X_c, 1]^T
\]

### 4.2 静态相机（static）

- 内参 `K_static`：来自 `cameras.json['static']['K']`
- 分辨率：`H=W=200`
- 外参 `W_T_C_static`：来自 `cameras.json['static']['W_T_C']`（**OpenCV** 轴）

### 4.3 腕相机（gripper）

- 内参 `K_gripper`：来自 `cameras.json['gripper']['K']`
- 分辨率：`H=W=84`
- 手眼外参（训练口径）：
  - `E_ref = cameras.json['gripper']['E_T_C']`（EE→Cam）

腕相机在世界系下的位姿：

\[
W\_T\_{Cg} = W\_T^{EE}(t) \cdot E\_{T}^{C}
\]

其中 `W_T^EE(t)` 来自 `robot_obs`：

- `pos = robot_obs[0:3]`
- `rpy = robot_obs[3:6]`（**弧度**）
- 欧拉顺序：**zyx**（即 `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`）

### 4.4 内参 K 的“已确认数值”

你机器读出并符合预期（FOV 推导一致）：

- static：`fx=fy=1143.005230...`，`cx=cy=100.0`
- gripper：`fx=fy=54.735465...`，`cx=cy=42.0`

---

## 5. URDF 口径桥接 Δ（训练↔部署等价的关键）

### 5.1 定义

- `E_raw`：URDF “零位关节”下的 EE→cam(14)（部署/仿真原生口径）
- `E_ref`：数据集训练口径的 EE→cam（来自 `cameras.json`）
- 固定桥接：

\[
\Delta = E\_{raw}^{-1} \cdot E\_{ref}
\]

并写入：

- `cameras.json["meta"]["urdf_to_dataset_delta_4x4"]`

### 5.2 三种使用模式（**必须记住**）

- **训练（dataset geometry）**：
  - `W_T_Cg = W_T_EE @ E_ref`
- **部署（URDF geometry）**：
  - `W_T_Cg = W_T_EE @ E_raw`
- **部署侧想喂给“假设训练口径”的历史模块（兼容）**：
  - `W_T_Cg = W_T_EE @ (E_raw @ Δ)`  
  - 且 `E_raw @ Δ ≡ E_ref` 已回归 PASS

---

## 6. 环境、依赖与统一环境变量

建议在 `~/datasets/calvin` 执行：

```bash
cd ~/datasets/calvin
pip install -q numpy pybullet pyyaml plotly pyarrow
```

环境变量（便于脚本引用）：

```bash
export ZIP="dataset/task_ABCD_D.zip"
export CJ="dataset/task_ABCD_D/calib/cameras.json"
export URDF="$HOME/datasets/calvin/calvin_env/data/franka_panda/panda_longer_finger.urdf"
```

---

## 7. 一键验收（最重要的三条）

### 7.1 语法快检（tools）

```bash
python -m py_compile tools/calvin_geom.py && echo "[OK] calvin_geom.py ready"
python -m py_compile tools/verify_train_deploy_equivalence.py && echo "[OK] verify_train_deploy_equivalence.py ready"
```

### 7.2 训练=部署(+Δ) 强回归（守门线）

```bash
python -m tools.verify_train_deploy_equivalence
# 期望 [RESULT] PASS
```

### 7.3 跨视角深度一致性抽检（推荐用你已跑通的 robust 脚本）

你已在 `~/datasets/calvin` 里跑过（示例）：

```bash
python calvin_verify_npz_depth.py \
  --zip dataset/task_ABCD_D.zip \
  --cameras dataset/task_ABCD_D/calib/cameras.json \
  --split validation \
  --episodes 40 \
  --rpy-order zyx \
  --t-mode auto \
  --bilinear
```

典型输出应接近：

- `median ~ 0.04 m`
- `p90 ~ 0.06 m`
- `<5cm ~ 65–75%`

> 注意：如果你用的是 **不含边缘过滤 / 不含语义自适配 / 不含双线性** 的“简化脚本”，数值会显著变差（这是预期现象，不代表几何口径错）。

---

## 8. 训练/部署集成方法（给模型/训练代码用）

在训练 DataLoader 或预处理里，你只需要拿到 `W_T_Cg` 并结合 `K` 做投影/点云。

### 8.1 训练侧（推荐：统一用 E_ref）

```python
from tools.calvin_geom import load_cameras, get_W_T_Cg
cams = load_cameras("dataset/task_ABCD_D/calib/cameras.json")

# rob_obs_t: [x,y,z,r,p,y]（弧度，顺序 zyx）
W_T_Cg = get_W_T_Cg(rob_obs_t, cams, mode="train")  # = W_T^EE @ E_ref
```

### 8.2 部署/仿真侧（URDF 口径）

```python
W_T_Cg = get_W_T_Cg(rob_obs_t, cams, mode="deploy_urdf") # = W_T^EE @ E_raw
```

### 8.3 历史模块兼容（部署侧右乘 Δ）

```python
W_T_Cg = get_W_T_Cg(rob_obs_t, cams, mode="deploy_as_train") # = W_T^EE @ (E_raw @ Δ)
```

### 8.4 初始化阶段强断言（强烈建议写进代码）

```bash
python - <<'PY'
import numpy as np
from tools.calvin_geom import load_cameras
cams = load_cameras("dataset/task_ABCD_D/calib/cameras.json")
assert cams["rpy_order"] == "zyx"
assert np.linalg.norm(cams["E_raw_zero"] @ cams["Delta"] - cams["E_ref"]) < 1e-6
print("[OK] loader invariants checked")
PY
```

---

## 9. ZIP→点云重建与可视化（用于调试几何是否真对齐）

### 9.1 Plotly HTML（推荐，WSL/无 OpenGL 环境也稳）

下面脚本可直接从 ZIP 读一个 episode，并生成 HTML：

```bash
python - <<'PY'
import io, zipfile, re, os, json, math, random
import numpy as np, plotly.graph_objects as go
ZIP="dataset/task_ABCD_D.zip"; CJ="dataset/task_ABCD_D/calib/cameras.json"
OUT="dataset/task_ABCD_D/calib/vis"; EPISODES=3; S_PER_CAM=50000; T_MODE="mid"
os.makedirs(OUT, exist_ok=True)

def depth_to_m(d, near=0.01, far=10.0):
    d=np.asarray(d,np.float32)
    return (far*near)/(far-(far-near)*d) if d.size and np.nanmax(d)<=1.5 else d

def Rx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def W(pos,rpy,order="zyx"):
    r,p,y=rpy
    R = Rz(y) @ Ry(p) @ Rx(r) if order=="zyx" else Rx(r) @ Ry(p) @ Rz(y)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=pos; return T

j=json.load(open(CJ))
Ks=np.array(j["static"]["K"],float); Hs,Ws=j["static"]["H"],j["static"]["W"]
Kg=np.array(j["gripper"]["K"],float); Hg,Wg=j["gripper"]["H"],j["gripper"]["W"]
W_T_Cs=np.array(j["static"]["W_T_C"],float); E=np.array(j["gripper"]["E_T_C"],float)
Kinv_s=np.linalg.inv(Ks); Kinv_g=np.linalg.inv(Kg)

random.seed(0)
with zipfile.ZipFile(ZIP,'r') as zf:
    npzs=[n for n in zf.namelist() if re.match(r'^task_ABCD_D/validation(/.*)?/episode_\d+\.npz$', n)]
    random.shuffle(npzs); npzs=npzs[:EPISODES]
    for ref in npzs:
        d=np.load(io.BytesIO(zf.read(ref)), allow_pickle=True)
        rob=np.asarray(d['robot_obs']); rob=rob.reshape(1,-1) if rob.ndim==1 else rob
        t=0 if T_MODE=="start" else (len(rob)-1 if T_MODE=="end" else len(rob)//2)
        W_T_EE=W(rob[t,0:3], rob[t,3:6], "zyx"); W_T_Cg=W_T_EE@E

        ds=depth_to_m(np.asarray(d['depth_static'],np.float32),0.01,10.0)
        dg=depth_to_m(np.asarray(d['depth_gripper'],np.float32),0.01,2.0)

        # static
        idx=np.random.choice(ds.size, size=min(S_PER_CAM, ds.size), replace=False)
        vs,us=idx//ds.shape[1], idx%ds.shape[1]; z=ds[vs,us]; m=z>1e-3; us,vs,z=us[m],vs[m],z[m]
        Xcs=(Kinv_s@np.stack([us,vs,np.ones_like(us)],0))*z
        Xs=(W_T_Cs@np.vstack([Xcs,np.ones((1,Xcs.shape[1]))]))[:3,:].T
        Cs=d['rgb_static'][vs,us,:3].astype(np.float32)/255.0 if 'rgb_static' in d.files else None

        # gripper
        idxg=np.random.choice(dg.size, size=min(S_PER_CAM, dg.size), replace=False)
        vg,ug=idxg//dg.shape[1], idxg%dg.shape[1]; zg=dg[vg,ug]; mg=zg>1e-3; ug,vg,zg=ug[mg],vg[mg],zg[mg]
        Xcg=(Kinv_g@np.stack([ug,vg,np.ones_like(ug)],0))*zg
        Xg=(W_T_Cg@np.vstack([Xcg,np.ones((1,Xcg.shape[1]))]))[:3,:].T
        Cg=d['rgb_gripper'][vg,ug,:3].astype(np.float32)/255.0 if 'rgb_gripper' in d.files else None

        fig=go.Figure()
        def add_cloud(X,C,name):
            mk=dict(size=1)
            if C is not None:
                mk["color"]=['rgb(%d,%d,%d)'%(int(255*c[0]),int(255*c[1]),int(255*c[2])) for c in C]
            fig.add_trace(go.Scatter3d(x=X[:,0],y=X[:,1],z=X[:,2],mode='markers',marker=mk,name=name))
        add_cloud(Xs, Cs, f'{os.path.basename(ref)}-static')
        add_cloud(Xg, Cg, f'{os.path.basename(ref)}-gripper')
        fig.add_trace(go.Scatter3d(x=[0,1],y=[0,0],z=[0,0],mode='lines',line=dict(width=10),name='1 m ruler'))
        fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
                          width=1280,height=800,showlegend=True)
        out=os.path.join(OUT, os.path.basename(ref).replace(".npz",".html"))
        fig.write_html(out, include_plotlyjs='cdn', full_html=True)
        print("[OK] wrote:", out)
PY
```

打开输出目录（WSL→Windows）：

```bash
wslpath -w dataset/task_ABCD_D/calib/vis
```

---

## 10. 完整性复核（可选但推荐，尤其是要交接/上 CI）

### 10.1 shards & rows

```bash
python - <<'PY'
import glob, pyarrow.parquet as pq
paths=sorted(glob.glob('dataset/task_ABCD_D/calib/gripper_poses-*.parquet'))
print("shards=", len(paths), " total_rows=", sum(pq.ParquetFile(p).metadata.num_rows for p in paths))
PY
```

### 10.2 duplicates

```bash
python - <<'PY'
import glob, pyarrow.parquet as pq
paths=sorted(glob.glob('dataset/task_ABCD_D/calib/gripper_poses-*.parquet'))
seen=set(); dup=0
for p in paths:
    pf=pq.ParquetFile(p)
    for b in pf.iter_batches(columns=['episode_path']):
        for v in b.column(0).to_pylist():
            dup += (v in seen); seen.add(v)
print("duplicates=", dup)
PY
```

---

## 11. 常见坑（你已经踩过的都在这里固化）

- **不要解压 650 GB 大包**：全程用 `zipfile.ZipFile` 流式读取即可。
- 欧拉顺序统一 **zyx**，角度单位 **弧度**。
- `robot_obs` 已在 world 系：不要额外乘 base→world。
- 像素索引必须：`np.rint(...).astype(np.int64)` 并 `clip` 边界。
- 深度单位通常是 **米**；若检测到 0..1 的 z-buffer（`max<=1.5`），再做线性化：
  - `(far*near)/(far-(far-near)*d)`
- URDF 相机 link：
  - ✅ cam=14 (gripper_cam) 固定刚体，q-不变
  - ❌ cam=12 (finger_tip) q-variant，**DO NOT USE**
- 阈值选择：PyBullet 浮点误差通常 1e-6 量级；回归用 1e-6 是合理工程线。
- 若未来 URDF 或实机标定改变：
  - 重新计算 `Δ_new = inv(E_raw_new) @ E_ref`
  - 写回 `cameras.json.meta.urdf_to_dataset_delta_4x4`
  - 再跑回归 PASS 即可上线

---

## 12. 核心源码（可审计/可重建）

> 下列两个文件是你已经跑到 PASS 的强回归核心。  
> 如果你未来丢了 tools 文件夹，也能从 README 重建同样的逻辑。

### 12.1 `tools/calvin_geom.py`

```python
import os, json, numpy as np, math

# --- 基础旋转与位姿 ---
def Rx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def Ry(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def Rz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)

def W_T_from_posrpy(pos, rpy, order="zyx"):
    \"\"\"pos: (3,), rpy: (3,) 弧度；支持 'zyx'/'xyz'\"\"\"
    r,p,y = map(float, rpy)
    if order=="zyx":
        R = Rz(y) @ Ry(p) @ Rx(r)
    elif order=="xyz":
        R = Rx(r) @ Ry(p) @ Rz(y)
    else:
        raise ValueError("unsupported rpy order: "+order)
    T = np.eye(4, dtype=float); T[:3,:3]=R; T[:3,3]=np.asarray(pos,float); return T

def _T_from_pb(pos,orn, p):
    R=np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
    X=np.eye(4); X[:3,:3]=R; X[:3,3]=np.asarray(pos,float); return X

# --- 相机加载（含 URDF 零位 E_raw 计算） ---
def load_cameras(cj_path):
    \"\"\"
    读取 cameras.json 与 meta，计算：
      - E_ref:   数据集口径 EE->Cam（4x4）
      - Delta:   URDF->数据集 的固定桥接变换（4x4）
      - E_raw_zero: 按 meta.ee_link_id / meta.gripper_cam_link_id 在 URDF零位下的 EE->Cam（4x4）
      - EE_IDX / CAM_IDX / URDF / rpy_order / K 等
    返回 dict
    \"\"\"
    j = json.load(open(cj_path, "r"))
    meta = j.get("meta", {})
    cams = {}
    cams["E_ref"]  = np.array(j["gripper"]["E_T_C"], float)
    cams["Delta"]  = np.array(meta["urdf_to_dataset_delta_4x4"], float)
    cams["URDF"]   = os.path.expanduser(meta["urdf"])
    cams["EE_IDX"] = int(meta.get("ee_link_id", 15))
    cams["CAM_IDX"]= int(meta.get("urdf_cam_link_id", meta.get("gripper_cam_link_id", 14)))
    cams["rpy_order"] = meta.get("rpy_order", "zyx")
    cams["K_static"]      = np.array(j["static"]["K"], float)
    cams["W_T_C_static"]  = np.array(j["static"]["W_T_C"], float)
    cams["K_gripper"]     = np.array(j["gripper"]["K"], float)
    cams["H_static"]      = int(j["static"]["H"]); cams["W_static"]= int(j["static"]["W"])
    cams["H_gripper"]     = int(j["gripper"]["H"]); cams["W_gripper"]= int(j["gripper"]["W"])

    # 计算 URDF 零位的 E_raw
    import pybullet as p
    assert os.path.isfile(cams["URDF"]), f"URDF not found: {cams['URDF']}"
    p.connect(p.DIRECT)
    bid = p.loadURDF(cams["URDF"], useFixedBase=True)
    for i in range(p.getNumJoints(bid)): p.resetJointState(bid,i,0.0)
    EE = p.getLinkState(bid, cams["EE_IDX"],  computeForwardKinematics=True)
    CM = p.getLinkState(bid, cams["CAM_IDX"], computeForwardKinematics=True)
    cams["E_raw_zero"] = np.linalg.inv(_T_from_pb(EE[4],EE[5],p)) @ _T_from_pb(CM[4],CM[5],p)
    p.disconnect()
    return cams

# --- 统一获得 W_T_Cgripper（训练/部署两口径） ---
def get_W_T_Cg(rob_obs_t6, cams, mode="train"):
    \"\"\"
    rob_obs_t6: [...,0:3 pos, 3:6 rpy]  (弧度, rpy_order 见 cams["rpy_order"])
    mode:
      - "train"           -> W_T_EE @ E_ref
      - "deploy_urdf"     -> W_T_EE @ E_raw_zero          （部署侧吃 URDF 口径）
      - "deploy_as_train" -> W_T_EE @ (E_raw_zero @ Delta)（部署侧右乘Δ，等价训练口径）
    \"\"\"
    rob_obs_t6 = np.asarray(rob_obs_t6, float).reshape(-1)[:6]
    W_T_EE = W_T_from_posrpy(rob_obs_t6[:3], rob_obs_t6[3:6], order=cams.get("rpy_order","zyx"))
    if mode == "train":
        return W_T_EE @ cams["E_ref"]
    elif mode == "deploy_urdf":
        return W_T_EE @ cams["E_raw_zero"]
    elif mode == "deploy_as_train":
        return W_T_EE @ (cams["E_raw_zero"] @ cams["Delta"])
    else:
        raise ValueError("unknown mode: "+str(mode))
```

### 12.2 `tools/verify_train_deploy_equivalence.py`

```python
import os, io, re, json, random, zipfile, numpy as np, pybullet as p
from tools.calvin_geom import load_cameras, get_W_T_Cg

ZIP = os.environ.get("ZIP","dataset/task_ABCD_D.zip")
CJ  = os.environ.get("CJ","dataset/task_ABCD_D/calib/cameras.json")

TOL_QRAW = 1e-6     # q-不变 (Fro) 工程阈值
TOL_EQV  = 1e-6     # Δ 等价 (Fro)
TOL_EP   = 1e-6     # episode 等价 (Fro)

def _T(pos,orn):
    R=np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
    X=np.eye(4); X[:3,:3]=R; X[:3,3]=np.asarray(pos,float); return X

def _rot_trans_err(E, E0):
    R, t  = E[:3,:3],  E[:3,3]
    R0, t0= E0[:3,:3], E0[:3,3]
    Rd = R0.T @ R
    c = max(-1.0, min(1.0, (np.trace(Rd)-1)/2))
    ang_deg = np.degrees(np.arccos(c))
    terr = float(np.linalg.norm(t-t0))
    fro  = float(np.linalg.norm(E-E0))
    return ang_deg, terr, fro

def urdf_q_invariance_and_delta_equiv(cams, trials=40):
    urdf, ee, cam, Delta, E_ref = cams["URDF"], cams["EE_IDX"], cams["CAM_IDX"], cams["Delta"], cams["E_ref"]
    p.connect(p.DIRECT)
    bid=p.loadURDF(urdf, useFixedBase=True)
    nj=p.getNumJoints(bid)
    for j in range(nj): p.resetJointState(bid,j,0.0)
    EE=p.getLinkState(bid,ee,computeForwardKinematics=True)
    CM=p.getLinkState(bid,cam,computeForwardKinematics=True)
    E0=np.linalg.inv(_T(EE[4],EE[5])) @ _T(CM[4],CM[5])

    mx_fro_raw=0.0; mx_ang=0.0; mx_terr=0.0
    mx_fro_ref=0.0
    random.seed(0)
    for _ in range(trials):
        for j in range(nj): p.resetJointState(bid,j, random.uniform(-0.4,0.4))
        EE=p.getLinkState(bid,ee,computeForwardKinematics=True)
        CM=p.getLinkState(bid,cam,computeForwardKinematics=True)
        E =np.linalg.inv(_T(EE[4],EE[5])) @ _T(CM[4],CM[5])
        ang, terr, fro = _rot_trans_err(E, E0)
        mx_ang  = max(mx_ang,  ang)
        mx_terr = max(mx_terr, terr)
        mx_fro_raw = max(mx_fro_raw, fro)
        mx_fro_ref = max(mx_fro_ref, float(np.linalg.norm(E @ Delta - E_ref)))
    p.disconnect()
    return mx_ang, mx_terr, mx_fro_raw, mx_fro_ref

def episodes_equivalence(cams, n_eps=80):
    with zipfile.ZipFile(ZIP,'r') as zf:
        eps=[n for n in zf.namelist() if re.match(r"^task_ABCD_D/(training|validation)(/.*)?/episode_\d+\.npz$", n)]
        random.seed(0); random.shuffle(eps); eps=eps[:n_eps]
        diffs=[]
        for ref in eps:
            d=np.load(io.BytesIO(zf.read(ref)), allow_pickle=True)
            if 'robot_obs' not in d.files: continue
            rob=np.asarray(d['robot_obs'])
            if rob.ndim==1: rob=rob.reshape(1,-1)
            for t in (0, rob.shape[0]//2, rob.shape[0]-1):
                W1=get_W_T_Cg(rob[t,0:6], cams, mode="train")
                W2=get_W_T_Cg(rob[t,0:6], cams, mode="deploy_as_train")
                diffs.append(np.linalg.norm(W1-W2))
        diffs=np.array(diffs,float)
        return (diffs.size, float(diffs.max()) if diffs.size else None, float(diffs.mean()) if diffs.size else None)

def main():
    cams=load_cameras(CJ)
    print("[META] ee=%s cam=%s urdf=%s" % (cams["EE_IDX"], cams["CAM_IDX"], cams["URDF"]))
    ang, terr, fro_raw, fro_ref = urdf_q_invariance_and_delta_equiv(cams, trials=40)
    n, mmax, mmean = episodes_equivalence(cams, n_eps=80)
    print("[URDF q-invariance] max_rot=%.3e deg  max_trans=%.3e m  max_fro=%.3e  (TOL %.1e)" % (ang, terr, fro_raw, 1e-6))
    print("[Δ equivalence   ] max||E(q)@Δ - E_ref||_F = %.3e                     (TOL %.1e)" % (fro_ref, 1e-6))
    print("[Episode equiv   ] frames=%d  max=%.3e  mean=%.3e                (TOL %.1e)" % (n, mmax, mmean, 1e-6))
    ok = (fro_raw < 1e-6) and (fro_ref < 1e-6) and (n>0 and mmax < 1e-6)
    print("[RESULT]", "PASS" if ok else "FAIL")

if __name__=="__main__":
    main()
```

---

## 13. OpenPI 对接（给下一位接手者的“现状说明”）

> 这一节不改变 CALVIN 的标定结果；只是把“你现在 openpi 那边卡住的点”也写清楚，方便交接。

截至你 2025-12-15 的状态输出：

- openpi 分支：`pi0.5_sonata`
- `openpi.training.config / data_loader` import 失败：`ModuleNotFoundError: torch_scatter`
- CALVIN Zip Reader 仍未落地：`openpi.training.calvin_zip_dataset` 不存在

因此：

- ✅ CALVIN 数据与口径已准备好用于训练（K/E/Δ/回归线 PASS）
- ❌ openpi 训练管线还没真正“吃到 CALVIN”，需要：
  1) 安装/解决 `torch_scatter` 依赖（或做可选依赖隔离）
  2) 实现/引入 Calvin Zip Reader（方案 A 的新 Reader）
  3) 在 data config 里把 CALVIN 的键映射到 openpi 所需的 `observation/...` 格式

这部分属于 **openpi 工程**，不在本 README 覆盖的 “CALVIN 标定与几何口径”范畴之内。

---

## 14. 最终建议（你问的“是否可以直接 move on”）

- **对于 CALVIN 标定与几何口径**：✅ 你已经可以放心 move on（回归线 PASS + K 已确认 + Δ 已固化）。
- **对于 openpi 训练打通**：❌ 还不能说“已完全打通”，因为目前 import/Reader 仍阻塞；但这是下一阶段的工作，与 CALVIN 标定结果无关。

---

（完）
