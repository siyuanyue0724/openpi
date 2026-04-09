# CALVIN 工具集（`scripts/calvin/`）

这份 README 覆盖 `scripts/calvin/` 目录下的 **3 个实用脚本**，用于在 **openpi / π₀.₅-sonata** 训练管线接入 CALVIN 前后做数据检查、做小规模子集、跑渲染/几何基准测试。

> 这份文档本身不依赖你“记得之前聊过什么”。  
> 只要你有：  
> - openpi 仓库（`pi0.5_sonata` 分支）  
> - CALVIN 数据（通常是 `task_*/training|validation/episode_*.npz` 形式；或只有 zip 需要先抽取子集）  
> 就能按本文跑通发现/抽样/基准。

---

## 1. 目录结构（建议放置方式）

把本目录整体放到 openpi 仓库内：

```
openpi/
└─ scripts/
   └─ calvin/
      ├─ README.md                 # 本文档（你正在看）
      ├─ calvin_discover.py        # 扫描 task/split/episode
      ├─ calvin_make_subset.py     # 创建小规模子集（symlink/hardlink/copy）
      └─ calvin_render_bench.py    # PyBullet 渲染基准 & 几何 sanity-check
```

> ✅ **建议提交到 GitHub**：这些脚本都是“工具脚本”，不包含数据、不含私密信息，提交后便于复现与交接。  
> ❌ **不要提交数据本体**（650GB zip、提取的 npz、parquet 等）。

---

## 2. 先决条件

### 2.1 Python / 依赖
- `calvin_discover.py`、`calvin_make_subset.py`：仅依赖 Python 标准库
- `calvin_render_bench.py`：需要 `numpy`、`pybullet`

安装（任选你习惯的环境；示例用 conda env）：

```bash
conda activate openpi
python -V

pip install -q numpy pybullet
```

### 2.2 CALVIN 数据的两种形态

#### 形态 A：**已在磁盘上展开为目录**
你已经有如下目录结构（推荐用于这 3 个脚本）：

```
<DATA_ROOT>/
└─ task_ABCD_D/
   ├─ training/
   │  ├─ episode_0000000.npz
   │  ├─ episode_0000001.npz
   │  └─ ...
   └─ validation/
      ├─ episode_0000000.npz
      └─ ...
```

#### 形态 B：**只有一个大 zip（推荐长期存储方式）**
例如：
```
~/datasets/calvin/dataset/task_ABCD_D.zip
```

这 3 个脚本里：
- `discover` / `make_subset` **目前都按“目录”工作**（不直接读 zip）。
- 如果你只有 zip：推荐只解压/抽取一个 **很小的子集** 到磁盘，然后再用本目录脚本做后续调试。

---

## 3. CALVIN episode（`.npz`）里都有什么

你在本机抽检到的单个 episode（training / validation 基本一致）包含：

- `rgb_static`：`(200, 200, 3)` `uint8`
- `rgb_gripper`：`(84, 84, 3)` `uint8`
- `depth_static`：`(200, 200)` `float32`
- `depth_gripper`：`(84, 84)` `float32`
- `actions`：`(7,)` `float64`
- `rel_actions`：`(7,)` `float64`
- `robot_obs`：`(15,)` `float64`（注意：它是 1D，不是 (T,D)；所以“时序”可能在别处或由上游定义）
- `scene_obs`：`(24,)` `float64`
- `rgb_tactile`：`(160, 120, 6)` `uint8`
- `depth_tactile`：`(160, 120, 2)` `float32`

> 这意味着：如果你要做 VLA 训练，至少可以稳定拿到「静态视角 + 腕视角」RGB/Depth + 低维 state/action。

---

## 4. 脚本一：`calvin_discover.py`

### 4.1 用途
扫描一个 CALVIN 数据根目录，列出：
- 有哪些 `task_*`
- 每个 task 里有哪些 split（training/validation/…）
- 每个 split 下 `episode_*.npz` 数量
- split 下是否存在 `.hydra/` 或语言标注文件
  默认会检查：
  - `<split>/auto_lang_ann.npy`
  - `<split>/lang_annotations/auto_lang_ann.npy`
  - `<split>/lang_*/auto_lang_ann.npy`

用于快速确认：你应该把后续脚本的 `--src` 指到哪里。

### 4.2 用法

```bash
python scripts/calvin/calvin_discover.py --root ~/datasets/calvin/dataset
```

输出是一个 JSON，类似：

```json
{
  "task_ABCD_D": {
    "splits": {
      "training": {
        "path": ".../task_ABCD_D/training",
        "episodes": 2307126,
        "has_.hydra": true,
        "has_auto_lang_ann": true
      },
      "validation": {
        "path": ".../task_ABCD_D/validation",
        "episodes": 99022,
        "has_.hydra": true,
        "has_auto_lang_ann": true
      }
    }
  }
}
```

**常见坑**
- 如果你只有 zip 没有展开目录，这个脚本会提示找不到 `task_*` 目录：这是预期行为。
- 当前本机已经有解压好的 `~/datasets/calvin/dataset/task_ABCD_D/`；不要再去解压同级的 `task_ABCD_D.zip`。

---

## 5. 脚本二：`calvin_make_subset.py`

### 5.1 用途
从一个具体的 `task_*` 目录里，按 episode id 选取若干条样本，创建一个小规模子集目录，用于：
- 快速 smoke test / 开发 dataloader
- 在不全量复制数据的情况下验证训练管线是否能跑

支持三种策略：
- `--strategy symlink`（默认）：快、不占空间；要求目标目录与源目录均在同一系统可访问路径
- `--strategy hardlink`：不额外占空间，但要求在同一文件系统
- `--strategy copy`：最稳，但会占空间

### 5.2 用法（示例）

假设你的数据目录是：
`~/datasets/calvin/dataset/task_ABCD_D/{training,validation}/episode_*.npz`

创建一个 100 条训练、20 条验证的子集：

```bash
python scripts/calvin/calvin_make_subset.py \
  --src ~/datasets/calvin/dataset/task_ABCD_D \
  --dst ~/datasets/calvin/subsets/task_ABCD_D_small \
  --train-ids 0:100 \
  --val-ids 0:20 \
  --strategy symlink
```

你会得到：

```
~/datasets/calvin/subsets/task_ABCD_D_small/
├─ training/
│  ├─ episode_0000000.npz -> (symlink to src)
│  ├─ ...
│  └─ episode_0000099.npz
└─ validation/
   ├─ episode_0000000.npz
   └─ ...
```

### 5.3 `--train-ids / --val-ids` 写法

脚本支持 3 种写法：

1) 连续区间：`START:END`（END 不包含）
- `0:100` → 0..99

2) 离散列表：`1,2,5,10`

3) 指向一个文本文件：
- 文件每行一个 id（支持空行、注释行 `#`）
- 例如 `ids.txt` 内容：
  ```
  1
  2
  5
  10
  ```

### 5.4 split 名称不标准怎么办？
默认使用 `training` / `validation`。如果你的 task 用了别的名字（例如 `train` / `val`），可以显式指定：

```bash
python scripts/calvin/calvin_make_subset.py \
  --src <TASK_DIR> \
  --dst <OUT_DIR> \
  --train-split train \
  --val-split val \
  --train-ids 0:100 \
  --val-ids 0:20
```

### 5.5 元数据拷贝规则（脚本当前行为）
脚本会尝试复制：

- `<split>/.hydra/`（整个目录）
- `<split>/auto_lang_ann.npy`
- `<split>/embeddings.npy`

> ⚠️ 但你当前的 `task_ABCD_D.zip` 里语言文件一般在 `training/lang_annotations/auto_lang_ann.npy` 或 `training/lang_*/*`，**不一定在 split 根目录**。  
> 所以：如果你确实要在子集里保留语言 embedding，可能需要你手动把对应 `lang_*` 子目录复制过来（或后续扩展脚本逻辑）。

---

## 6. 脚本三：`calvin_render_bench.py`

### 6.1 用途
这是一个 **PyBullet 渲染基准 + 几何 sanity-check** 脚本，主要用于：
- 测试 `getCameraImage` 的 FPS、平均耗时
- 验证（mask / depth / 点云）是否基本合理
- 对比 `--use_egl`（GPU EGL）和默认软件渲染的差异

它不会直接依赖 CALVIN 的 `episode_*.npz`，但它模拟了 **类似 CALVIN 的“静态相机 + 腕相机”** 两个视角。

### 6.2 用法（示例）

默认输出到当前目录 `render_bench.json`：

```bash
python scripts/calvin/calvin_render_bench.py --n 200 --hw 200 --out render_bench.json
```

开启 EGL（如果你在 Linux/WSL + NVIDIA GPU，并且 PyBullet 支持 EGL）：

```bash
python scripts/calvin/calvin_render_bench.py --n 200 --hw 200 --use_egl --out render_bench_egl.json
```

### 6.3 输出解释（JSON）
脚本会写一个 JSON，包含：
- `render_ms_mean` / `render_ms_p50` / `render_ms_p95`
- `fps`
- `mask_centroid_err_mean_m` / `mask_centroid_err_p90_m`（“通过 mask 反算物体中心”的误差）
- `depth_xyz_err_mean_m` / `depth_xyz_err_p90_m`（深度→点云→中心的误差）
- `depth_valid_ratio_mean`（深度有效像素比例）

> 如果你的渲染环境正常，误差一般会很小（场景是受控的 cube + plane），fps 主要看你的渲染后端（EGL vs CPU）。

### 6.4 常见问题
- `--use_egl` 下如果报错：通常是 EGL 插件/驱动不可用。先不加 `--use_egl` 跑通，或在原生 Ubuntu + NVIDIA Driver 环境再试。
- WSL 的 GUI / OpenGL 兼容问题：本脚本不依赖 GUI，但 EGL 插件可用性取决于安装方式。

---

## 7. “应该上传到 GitHub 还是保留本地？”

建议：
- ✅ **上传到 GitHub（推荐）**：  
  - README + 脚本是“可复现资产”，不会泄露数据  
  - 便于别人复跑/复核、便于 CI / review
- ⚠️ 只保留本地：  
  - 适合你还在快速迭代、不想打扰主仓库  
  - 但更容易“过段时间就找不到版本、忘记约束/参数”

折中方案：
- 先在 GitHub 放 `scripts/calvin/` + README（稳定版）
- 本地额外保留 `scripts/calvin/_wip/`（不提交）用于实验性改动

---

## 8. 和 CALVIN K/E（相机标定）之间的关系（重要说明）

这 3 个脚本本身 **不负责生成** CALVIN 的 `cameras.json`。  
你当前的相机内参 K 已经存在于：

```
~/datasets/calvin/dataset/task_ABCD_D/calib/cameras.json
```

你已经验证过：
- static: `H=W=200`，`fx≈1143.005`，`cx=100`
- gripper: `H=W=84`，`fx≈54.735`，`cx=42`

这部分属于“标定与几何口径”的范畴，不在本目录脚本职责内；但它会直接影响后续 openpi 的深度→点云（Sonata）管线。

---

## 9. 快速 checklist（你要交接给别人时）

1) 你是否有磁盘目录形式的 CALVIN 数据？  
   - 有：直接跑 discover/make_subset  
   - 只有 zip：先抽取一个小子集到磁盘，再跑 make_subset

2) 子集准备好后，是否能用你自己的 dataloader 读到：
   - `rgb_static / rgb_gripper`
   - `depth_static / depth_gripper`
   - `robot_obs / actions`
   - （可选）语言 embedding

3) `calvin_render_bench.py` 是否能跑通（至少 CPU 路线）？

---

> 维护建议：如果你修改了脚本参数/行为，请同步更新本 README，并在 PR 里附一段「示例命令 + 期望输出」用于 review。
