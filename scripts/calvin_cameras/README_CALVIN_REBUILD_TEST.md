# CALVIN K/E 重建与旧版 `cameras.json` 兼容回归

> 目的：在**不重渲染**、**不全量解压 ZIP** 的前提下，基于官方 Hydra、URDF 和现有验证脚本，重建一个放在 `./test/calib` 下的 `cameras.json`，用于和历史正确版本做**安全对比**，不覆盖原始产物。

---

## 这份重写版做了什么

单文件脚本 `calvin_rebuild_cameras_test.py` 会：

1. 直接从 `task_ABCD_D.zip` 读取 Hydra 配置；
2. 重新计算 **static** 相机的：
   - `K`
   - `W_T_C`
   - `extrinsic_opengl_4x4`
   - `extrinsic_opencv_4x4`
3. 基于本地 CALVIN `calvin_env` 里的 URDF 和 PyBullet 重建多个 **gripper raw candidate**：
   - `EE/TCP × cam12/cam14` 风格候选
4. 当提供历史 `cameras.json` 时，进入 **reference regression** 模式：
   - 用历史文件中的 `gripper.E_T_C` 作为 `E_ref`
   - 用新鲜重建的 URDF `E_raw`
   - 重新计算 `Delta = inv(E_raw) @ E_ref`
   - 在 `./test/calib` 下写出一个**功能等价且字段更兼容旧版风格**的 `cameras.json`
5. 自动回归验证：
   - depth consistency（validation / training）
   - train ↔ deploy equivalence
6. 可选导出 `gripper_poses-*.parquet`

---

## 它和你原来那版的关系

这份脚本不是直接照搬 RoboUniView，而是按你现有本地流程重写成**轻量单文件版**：

- 不依赖可视化栈
- 不改原目录
- 输出统一落到 `./test/calib`
- 保留你现有验证链可用的主字段
- 同时尽量补回旧版里那些“冗余但可追溯”的字段

### 这次补回的旧版兼容字段

顶层：

- `hydra_path`
- `conventions`

`static`：

- `W`, `H`, `fov`, `near`, `far`
- `K`
- `extrinsic_opengl_4x4`
- `extrinsic_opencv_4x4`
- `W_T_C`

`gripper`：

- `W`, `H`, `fov`, `near`, `far`
- `K`
- `E_T_C_opengl_4x4`
- `E_T_C_opencv_4x4`
- `E_T_C`
- `end_effector_link_id`
- `gripper_cam_link`
- `urdf_path`
- `ref_frame`

`meta` 保留/重建：

- `axis`
- `coord_convention`
- `rpy_order`
- `urdf`
- `ee_link_id`
- `urdf_cam_link_id`
- `gripper_cam_link_id`
- `tcp_link_id`
- `base_used`
- `etC_source`
- `urdf_to_dataset_delta_4x4`

---

## 一个重要说明

历史 `cameras.json` 里有些字段本身带有**历史残留口径**，例如：

- `gripper.end_effector_link_id`
- `gripper.ref_frame`

它们和 `meta.ee_link_id / meta.urdf_cam_link_id` 可能并不完全一致。

这份重写脚本在 **reference regression** 模式下会优先：

- **保留历史文件里已有的 legacy 辅助字段**
- 同时**重新计算当前真正驱动验证/导出的 canonical 字段**

也就是说：

- 你现在的主流程用的还是：
  - `static.K`
  - `static.W_T_C`
  - `gripper.K`
  - `gripper.E_T_C`
  - `meta.urdf_to_dataset_delta_4x4`
- 旧风格字段主要是为了：
  - 版式兼容
  - 追溯
  - 和历史文件对照方便

如果**没有**提供历史 `cameras.json`，脚本仍能运行，但那些 legacy 辅助字段会按 **best-effort** 方式生成。

---

## 输出目录

默认写到：

```text
./test/calib/
```

典型产物：

```text
test/calib/
  cameras.json
  cameras.backup.json
  rebuild_report.json
  link_table.json
  variants/
    cameras_EE2cam12.json
    cameras_EE2cam14.json
    cameras_TCP2cam12.json
    cameras_TCP2cam14.json
    cameras_urdf_cam14_equiv.json
    cameras_urdf_equiv.json
```

可选导出：

```text
test/calib/gripper_poses-000000.parquet
...
```

---

## 依赖

建议使用你现有环境：

```bash
conda activate calvin-ke
```

脚本依赖：

- Python 3.10+
- `numpy`
- `pyyaml`
- `pybullet`
- 若要导出 parquet：`pyarrow`

---

## 推荐用法

### 按你当前本地路径直接运行

脚本目录：

```text
/home/siyuanyue/Documents/calibtest/calvin_rebuild_cameras_test.py
```

数据目录：

```text
/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip
/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D/calib/cameras.json
```

推荐命令：

```bash
cd ~/Documents/calibtest

python calvin_rebuild_cameras_test.py \
  --zip /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip \
  --repo-root /home/siyuanyue/datasets/calvin \
  --out-root /home/siyuanyue/Documents/calibtest/test \
  --reference-cameras /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D/calib/cameras.json \
  --verify-episodes 40 \
  --candidate-episodes 12 \
  --rpy-order zyx \
  --t-mode auto \
  --bilinear
```

---

## 可选：顺便做一次 parquet 冒烟导出

```bash
python calvin_rebuild_cameras_test.py \
  --zip /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip \
  --repo-root /home/siyuanyue/datasets/calvin \
  --out-root /home/siyuanyue/Documents/calibtest/test \
  --reference-cameras /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D/calib/cameras.json \
  --verify-episodes 40 \
  --candidate-episodes 12 \
  --rpy-order zyx \
  --t-mode auto \
  --bilinear \
  --export-parquet \
  --export-episodes-max 2000
```

---

## 如果想测试“完全不依赖旧文件”

```bash
python calvin_rebuild_cameras_test.py \
  --zip /home/siyuanyue/datasets/calvin/dataset/task_ABCD_D.zip \
  --repo-root /home/siyuanyue/datasets/calvin \
  --out-root /home/siyuanyue/Documents/calibtest/test \
  --no-reference-auto \
  --verify-episodes 40 \
  --candidate-episodes 12 \
  --rpy-order zyx \
  --t-mode auto \
  --bilinear
```

这会进入 **best_effort_without_reference** 模式。

注意：这种模式仍然能生成可用文件，但无法保证 100% 复现你历史版本里的所有 legacy 注释字段。

---

## 运行成功后会看到什么

典型输出类似：

```text
[OK] wrote:
  final      : .../test/calib/cameras.json
  report     : .../test/calib/rebuild_report.json
  variants   : .../test/calib/variants
  link table : .../test/calib/link_table.json

[SELECTED]
  ref_link_id=15  cam_link_id=14  mode=reference_regression
[VAL  ] mean=0.0388  median=0.0403  p90=0.0636  <5cm=72.21%
[TRAIN] mean=0.0438  median=0.0435  p90=0.0655  <5cm=67.78%
[EQV  ] raw_fro=4.112e-07  ref_fro=4.151e-07
[DIFF ] static_W_T_C=4.146e-07  static_K=0.000e+00  gripper_K=0.000e+00  E_ref=0.000e+00  Delta=1.343e-08
```

这组数值已经说明：

- `static.K` 完全一致
- `gripper.K` 完全一致
- `E_ref` 完全一致
- `Delta` 只有 `1e-8` 量级微小浮点差
- `static.W_T_C` 只有 `1e-7` 量级微差
- depth consistency 落在合理区间
- 训练/部署等价校验通过工程级阈值

---

## 验收标准

你现有这条链建议重点看：

### 1. canonical 字段一致性

`rebuild_report.json` 里如果存在 `reference_compare`，关注：

- `static_W_T_C_fro`
- `static_K_fro`
- `gripper_K_fro`
- `E_ref_fro`
- `Delta_fro`

经验上：

- `static_K_fro = 0`
- `gripper_K_fro = 0`
- `E_ref_fro = 0`
- `Delta_fro ≈ 1e-8`
- `static_W_T_C_fro ≈ 1e-7`

都属于正常。

### 2. 深度一致性

validation / training 一般应落在：

- `median ≈ 0.04–0.05 m`
- `p90 ≈ 0.06–0.07 m`

### 3. 训练 / 部署等价

`raw_fro`、`ref_fro` 在 `1e-6` 以内即可认为工程上通过。

---

## 脚本行为细节

### static 的来源

从 ZIP 里的 Hydra 读取：

- `width / height / fov`
- `look_from / look_at / up_vector`

然后构造：

- `K`
- `C_T_W_gl`
- `C_T_W_cv = rowflip(C_T_W_gl)`
- `W_T_C = inv(C_T_W_cv)`

### gripper 的来源

- 用 PyBullet 加载本地 URDF
- 枚举多个 link pair 生成 raw candidate
- 若有 reference cameras：
  - `E_ref` 直接用 reference 的 `gripper.E_T_C`
  - `Delta = inv(E_raw) @ E_ref`
- 若无 reference：
  - 选 depth consistency 最优的 raw candidate 作为 `E_ref`

### 为什么还要保留 legacy 字段

因为你历史文件里不只有“真正被运行链消费的字段”，还带了很多：

- 手工回溯信息
- 旧命名
- 历史口径说明
- OpenGL / OpenCV 辅助矩阵

这些字段对新验证链不是必需项，但对人工审阅和对照很有用。

---

## 和你现有脚本的关系

这份重写版输出的 `cameras.json` 可直接供你现有这些脚本使用：

- `calvin_verify_npz_depth.py`
- `calvin_verify_from_npz.py`
- `calvin_verify_cross_depth.py`
- `calvin_export_ke.py`
- `verify_train_deploy_equivalence.py`

其中真正关键的是这些 canonical 字段：

- `static.K`
- `static.W_T_C`
- `gripper.K`
- `gripper.E_T_C`
- `meta.urdf_to_dataset_delta_4x4`

---

## 建议的对比方式

### 看新文件结构

```bash
python - <<'PY'
import json, pprint
p='/home/siyuanyue/Documents/calibtest/test/calib/cameras.json'
j=json.load(open(p))
print(j.keys())
print()
print('static keys =', j['static'].keys())
print('gripper keys =', j['gripper'].keys())
print('meta keys =', j['meta'].keys())
PY
```

### 和原版做 canonical 差异对比

```bash
python - <<'PY'
import json, numpy as np
old='/home/siyuanyue/datasets/calvin/dataset/task_ABCD_D/calib/cameras.json'
new='/home/siyuanyue/Documents/calibtest/test/calib/cameras.json'
A=json.load(open(old)); B=json.load(open(new))
def fro(x,y): return float(np.linalg.norm(np.array(x,float)-np.array(y,float), ord='fro'))
print('static_W_T_C', fro(A['static']['W_T_C'], B['static']['W_T_C']))
print('static_K    ', fro(A['static']['K'], B['static']['K']))
print('gripper_K   ', fro(A['gripper']['K'], B['gripper']['K']))
print('E_ref       ', fro(A['gripper']['E_T_C'], B['gripper']['E_T_C']))
print('Delta       ', fro(A['meta']['urdf_to_dataset_delta_4x4'], B['meta']['urdf_to_dataset_delta_4x4']))
PY
```

### 看补回的 legacy 字段是否存在

```bash
python - <<'PY'
import json
p='/home/siyuanyue/Documents/calibtest/test/calib/cameras.json'
j=json.load(open(p))
print('top:', 'hydra_path' in j, 'conventions' in j)
print('static extras:', 'extrinsic_opengl_4x4' in j['static'], 'extrinsic_opencv_4x4' in j['static'], 'fov' in j['static'])
print('gripper extras:', 'E_T_C_opengl_4x4' in j['gripper'], 'E_T_C_opencv_4x4' in j['gripper'], 'ref_frame' in j['gripper'])
PY
```

---

## 注意事项

- 这份脚本默认**不覆盖原始** `dataset/task_ABCD_D/calib/cameras.json`
- 建议始终把输出放在：
  - `~/Documents/calibtest/test`
- 若 reference 文件存在，推荐始终传入 `--reference-cameras`
- 若只是为了验证“功能等价”，canonical 字段比 legacy 辅助字段更重要
- 若只是为了保持旧文件观感，这版已经把大部分缺失字段补回来了

---

## 结论

这份重写版的目标不是“神秘复原当年某个已经消失的临时脚本”，而是：

- 把你当年那条轻量流程**重新整理成稳定的单文件工具**
- 保留你现在验证链真正依赖的主字段
- 同时尽量恢复旧版 `cameras.json` 的外观和辅助字段
- 始终在 `./test` 下安全回归，不污染原始正确产物
