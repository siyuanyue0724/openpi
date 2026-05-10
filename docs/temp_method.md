下面这版我会把前面几个争议点重新统一成一套架构，而不是继续堆局部建议。我的核心判断是：

> **你们当前 AQR/PICF 的方向不是错，反而非常接近近期 object-addressable world model / JEPA-VLA / VLA-JEPA 的趋势。真正要改的不是“再加很多模块”，而是把现有 AQR、posterior、typed memory、V-JEPA temporal embedding、PaliGemma semantic/image tokens、PI0.5 action path 统一成一个“对象可寻址的预测式 belief-state 架构”。**

我建议把最终版理解为：

```text
PICF-AQR-OWM:
  Object-addressable Predictive Anchor Router

核心：
  slots / anchors 不是 keypoints，
  而是带地址、内容、几何、不确定性、时序 belief 的对象/任务状态。

PaliGemma:
  提供语言、语义、开放世界 referring 和 image-language evidence。

V-JEPA:
  提供视频预测式、时空 grounding 的 visual world evidence。

point/tactile/proprio/action:
  提供 metric、接触、身体状态和动力学约束。

posterior:
  是闭环 belief filter，不是普通 cache。

KV/cache:
  只能作为有年龄、有不确定性、有真实观测校正的 episodic evidence memory，
  不能替代 posterior。
```

---

# 0. 先修正两个关键点

## 0.1 “last 2 frames”确实有论文来源，但当前代码实现不等价

JEPA-VLA 明确写到：它从 **two most recent frames** 提取 V-JEPA 2 representations，并把这些表示作为额外 conditioning signals 放进 VLA action prediction；它还比较了 early fusion 和 gated fusion，其中 gated cross-attention 是为了不破坏已有大规模 VLA 先验。([arXiv][1])

但你们当前代码里的：

```python
current_map(use_last_two_mean=True)
```

不是 JEPA-VLA 的做法。它是在 V-JEPA 已经编码完 `num_frames=64` 的 clip 以后，把最后两个 **latent temporal slices** 平均成一个 2D map。也就是说：

```text
JEPA-VLA:
  two most recent visual frames
  -> V-JEPA 2 embeddings
  -> extra conditioning tokens / gated cross-attn

当前代码:
  64-frame V-JEPA clip
  -> latent temporal tokens
  -> take last slice, or mean(last two latent slices)
  -> one 2D visual map
```

所以正确结论不是“打开 last-two-mean 就等于复现 JEPA-VLA”，而是：

> **应该让 AQR 显式读最近两个或多个 V-JEPA temporal embeddings，而不是把它们提前平均掉。**

---

## 0.2 同事说“缺显式 memory”不完全准确：PICF 已经有 posterior/RNN memory

你指出得对。你们当前 PICF 里已经有显式 recurrent posterior memory，不是无记忆模型。

我检查你上传的代码后，当前 posterior 至少包括：

```text
src/openpi/picf/core/contracts.py:
  PicfPosteriorAnchorState:
    h, c
    mu, Sigma
    x, S
    alpha
    contact_prob
    support_mass
    binding
    evidence_tokens
    tokens
    global_post
    role_ids
```

在 `pipeline.py` 里也有：

```text
prior_lstm
post_lstm
posterior_self
posterior_pool
```

并且 `_current_prior()` 会用 previous posterior、proprio、previous action 生成 prior；`_posterior_update()` 会用当前 observation anchors、binding、visual/tactile evidence 做 posterior update；`_innovation()` 还会比较当前观测和上一时刻的 physical prediction cache。这已经是一个神经化 belief filter / RNN posterior，而不是简单 feed-forward router。

所以更准确的问题不是：

```text
当前 PICF 缺 memory。
```

而是：

```text
当前 PICF 有 posterior memory，
但缺少显式暴露给 AQR 的 temporal visual memory，
缺少长期 causal evidence cache，
缺少 object-address / content 分离，
缺少 slot-level JEPA world prediction，
以及 PaliGemma / V-JEPA / point / tactile 的一等 typed support 统一。
```

这一区分非常重要。否则会误把一个已经存在的强模块重新发明一遍。

---

# 1. 架构最终版的数学目标：把 AQR 变成 object-addressable belief filter

我建议从 POMDP / belief-state 角度重新定义整个系统。

机器人实际面对的是部分可观测状态：

```text
s_t:
  object states
  robot state
  contact state
  task-relevant latent state
```

观测是多模态的：

```text
o_t =
  RGB / video
  language
  point / depth
  tactile
  proprio
  previous action
```

策略真正需要的不是某一帧 embedding，而是 belief：

```math
b_t(s) = p(s_t = s | o_{\le t}, a_{<t}, language)
```

经典 belief update 是：

```math
b_t(s)
∝ p(o_t | s)
  ∫ p(s | s', a_{t-1}) b_{t-1}(s') ds'
```

你们当前 posterior LSTM + binding + innovation，本质上已经在神经近似这个式子。

因此最终版不应该是“PaliGemma 模块 + V-JEPA 模块 + slot 模块 + cache 模块 + world model 模块”拼起来，而应该统一成：

```text
1. prior:
   根据上一时刻 posterior、action、proprio 预测当前 slot state。

2. measurement:
   当前多模态 typed memory 给出 evidence。

3. assignment:
   anchor/slot 通过竞争式 routing 绑定 evidence。

4. correction:
   用 measurement 修正 prior，产生 posterior。

5. prediction:
   posterior 预测未来 slot state / support / contact / action-relevant latent。

6. action:
   PI0.5 action path 读取 posterior、task anchors、innovation 和 typed supports。
```

这个统一框架和 VLA-JEPA 的 leakage-free latent state prediction、JEPA-VLA 的 V-JEPA predictive embeddings、OA-WAM 的 object-addressable state、object-centric world model 的 slot latent prediction 是同一类思想：**不要把世界压成 holistic global latent，而要组织成可寻址、可预测、可纠错的对象/任务状态。**([arXiv][2])

---

# 2. 状态定义：anchor/slot 应该拆成 address + content + geometry + uncertainty

当前 anchor 已经不只是椭球，但我建议进一步显式化：

```math
S_{t,j}
=
(a_j,\ c_{t,j},\ \mu_{t,j},\ \Sigma_{t,j},\ \alpha_{t,j},\ r_j)
```

其中：

```text
a_j:
  persistent address / identity vector
  表示“这是哪个槽位 / 哪个对象身份 / 哪个可追踪实体”。

c_{t,j}:
  time-varying content
  表示当前外观、语义、任务相关状态、接触状态。

mu_{t,j}, Sigma_{t,j}:
  几何 belief 的均值和不确定性。

alpha_{t,j}:
  visibility / validity / existence probability。

r_j:
  role / type / physical-task class。
```

这里最重要的是 **address-content separation**：

```text
address:
  which object / which persistent slot

content:
  what it currently looks like / where it is / how it changes
```

OA-WAM 这篇 2026 工作也明确指出，holistic world latents 会把对象身份和上下文纠缠在一起；它提出每个 slot 由 identity vector 和 time-varying content vector 组成，并通过 address-only key projection 把 “which object” 和 “what currently” 在张量层面分离。([arXiv][3])

这点对你们尤其关键，因为“从左往右第四根筷子”不是普通 saliency，它需要：

```text
稳定身份:
  这根筷子在 t, t+1, t+2 仍然是同一根。

关系内容:
  它当前相对其他筷子的 rank 是 4。

几何内容:
  它当前在什么位置，哪个部位可接触。

任务内容:
  当前语言正在选择它，而不是其他筷子。
```

如果没有 address-content separation，模型很容易变成：

```text
slot j 今天绑定第四根，下一帧绑定第三根；
或者 content 还在，但 address 漂了。
```

---

# 3. Slot/RoPE/前后置 token：你的想法方向对，但要避免一个误用

你说“不同 slot 的内容可以提供不同的类 RoPE 编码，或者使用前后置 token”，这个方向有价值，但需要严格区分三类编码：

```text
1. identity/address encoding:
   用于区分哪个 slot / 哪个对象身份。

2. type/modality/role encoding:
   用于区分 text、PG-image、V-JEPA、point、tactile、posterior、action 等来源。

3. coordinate/time encoding:
   用于表达时间、图像位置、世界坐标、view、tactile sensor index 等结构。
```

我不建议把“类 RoPE”直接当 slot identity。原因是 RoPE 的数学本质是让 attention score 依赖相对位置：

```math
q_i^\top k_j
\rightarrow
(R_{\theta(i)} q_i)^\top (R_{\theta(j)} k_j)
```

它适合表达：

```text
time difference
image x/y relative position
video t/h/w structure
relative order
```

但 slot identity 不应该有“距离”含义。slot 1 和 slot 4 不应因为 index 距离大就天然更远或更弱相关。slot identity 更适合用：

```text
persistent learned address vector a_j
address-only key projection
slot role embedding
posterior binding history
```

RoPE / MRoPE 应该用于 token 坐标，而不是身份本身。近期 multimodal RoPE 分析也提醒：简单把视觉/视频/text 展平成一维会破坏原生几何，多维 time-height-width RoPE 又有频率分配和 modality ambiguity 问题；保留 text RoPE 兼容性、给视觉使用清晰的多维坐标和 spatial reset，是更稳妥的方向。([arXiv][4])

所以我建议最终编码形式是：

```math
e_{t,i}^{(m)}
=
W_m z_{t,i}^{(m)}
+ E_m
+ E_{\text{view}(i)}
+ E_{\text{valid}(i)}
+ PE_m(\xi_{t,i})
```

其中：

```text
m:
  modality type

xi:
  modality-specific coordinates

对 text:
  1D language position，尽量保持原 LLM RoPE 兼容

对 V-JEPA:
  time, h, w, view

对 point:
  world x, y, z / camera x, y, z / projection coordinate

对 tactile:
  sensor id / taxel coordinate / time

对 posterior slot:
  slot address a_j + role r_j + time
```

前后置 token / prefix tokens 也可以用，但它们更适合做**轻量适配器**，不适合成为所有多模态证据的唯一瓶颈。JEPA-VLA 的 gated fusion 思路正是：不要粗暴把 V-JEPA tokens 拼进预训练 VLA 破坏分布，而是用 gated cross-attention 选择性接入 predictive embeddings。([arXiv][1])

---

# 4. 最终架构：PICF-AQR-OWM

我建议的完整结构如下。

```text
PICF-AQR-OWM

Layer 1:
  Typed Token Memory

Layer 2:
  Object-Addressable Posterior Slots

Layer 3:
  AQR Measurement Routing

Layer 4:
  Bayesian / GRU-style Posterior Correction

Layer 5:
  Causal Evidence Cache

Layer 6:
  JEPA-style Slot World Model

Layer 7:
  Task Anchors / Ordinal Selectors

Layer 8:
  PI0.5 Action Path
```

下面逐层展开。

---

## 4.1 Layer 1：Typed Token Memory，不要 PG→VJ 单向压缩

当前实现的问题不是 PaliGemma/V-JEPA 不能融合，而是 PG image support 主要被压成 V-JEPA grid bias，且 `pg_priors` 最终没有作为独立 branch 保留。

最终版应该保留每种 modality 的一等 memory：

```text
M_text:
  PaliGemma text / semantic tokens

M_pg_img[v]:
  PaliGemma image tokens per view

M_vjepa[t, v, h, w]:
  V-JEPA temporal-spatial tokens

M_point:
  point / geometry tokens

M_tactile:
  tactile / contact tokens

M_proprio:
  robot state tokens

M_action:
  previous action / action chunk tokens

M_post:
  posterior slot tokens

M_cache:
  bounded causal evidence cache
```

然后 AQR slot/query 同时读这些 memory：

```math
q_j^{l+1}
=
q_j^l
+
\sum_m
\gamma_{j,m}^l
\operatorname{Attn}(q_j^l,\ M_t^{(m)})
```

而不是：

```text
PaliGemma image tokens -> V-JEPA bias -> downstream
```

Grounding DINO / MDETR 这类 grounding 工作的共同经验是：语言和视觉应该在 query selection / decoder / feature enhancer 中紧密交互，而不是把语言过早压成一个弱 prior；机器人这边 CLIPort 也说明 semantic “what” 和 spatial “where” 必须共同建模。([arXiv][5])

但这不意味着要全量 dense all-to-all。Perceiver IO 的价值在这里很合适：用少量 latent/queries 读取大输入，计算复杂度线性依赖输入规模，而不是所有 token 二次 attention。([Hugging Face][6])

最终复杂度应保持：

```math
O(K \cdot \sum_m N_m) + O(K^2)
```

而不是：

```math
O((\sum_m N_m)^2)
```

---

## 4.2 Layer 2：Physical slots 和 task anchors 分离

我建议把当前 `physical queries` 和 `task queries` 的分工进一步明确。

```text
physical slots:
  维护场景对象、接触、几何、posterior identity。
  它们尽量 task-neutral，但不是完全语义盲。

task anchors:
  语言/任务条件 selectors。
  它们不一定维护长期身份，而是选择、组合、读取 physical slots。
```

也就是：

```math
S_t = \{S_{t,j}^{phys}\}_{j=1}^{K_p}
```

是持续存在的对象/世界状态；

```math
Q_t^\ell = \{q_{t,k}^{task}\}_{k=1}^{K_t}
```

是当前任务对这些状态的查询。

对“第四根筷子”来说：

```text
physical slots:
  可能分别维护几根筷子、夹爪、杯子、桌面等。

task anchor:
  根据语言“从左往右第四根”选择 physical slots 中 rank=4 的那个，
  再读取它的局部接触/几何状态。
```

这样避免一个问题：如果每个 task query 都直接绑定视觉 token，它容易随任务切换而漂；而 physical slots 提供了跨时间的 object bank。

Object-centric world model 和 Slot Structured World Model 都支持这个方向：slot attention / object-centric encoders 能把相似对象拆成独立 latent，并让动力学模型在对象层面预测交互。([arXiv][7])

---

## 4.3 Layer 3：AQR Measurement Routing

对每个 physical slot `j`，从每个 modality `m` 读取 support：

```math
\ell_{j,i}^{(m)}
=
\frac{
  (W_q [a_j, c^-_{t,j}, r_j])^\top
  R(\xi_{t,i}^{(m)})
  W_k e_{t,i}^{(m)}
}{\sqrt d}
+
b_{\text{geom}}
+
b_{\text{role}}
+
b_{\text{valid}}
```

```math
p_{j,i}^{(m)}
=
\operatorname{Softmax}_i(\ell_{j,i}^{(m)} / \tau_m)
```

然后在同 role / 同 modality / 同候选集合内做竞争：

```math
P^{(m)}
=
\operatorname{SinkhornLike}(\ell^{(m)})
```

这里的关键改动有三点：

```text
1. PaliGemma image support 不再只 remap 成 V-JEPA bias；
   它保留为 p_pg_img。

2. V-JEPA support 不再只有 p_visual(h,w)；
   它至少有 p_vjepa(t,h,w)，最好含 view。

3. point / tactile / posterior support 仍是 typed branch，
   不被塞进一个不透明 fused token。
```

最终 `PicfAnchorPriorGraphState` 应该包含：

```text
pg_image_priors
vjepa_temporal_priors
visual_priors
point_priors
tactile_priors
posterior_priors
cache_priors
slot_address
slot_content
anchor_x
anchor_S
support_uncertainty
```

当前 contract 里已经有 `pg_priors` 字段，但当前 AQR 返回里 `pg_priors=None`。这应该修掉。

---

## 4.4 Layer 4：Posterior Correction 是核心，不能被 KV cache 替代

这是你问得最关键的一点。

当前 posterior 的强项是：

```text
prior:
  根据上一 posterior + previous action + proprio 预测当前状态

binding:
  把 prior slots 和当前 obs anchors 对齐

measurement:
  从 current visual/point/tactile/posterior evidence 读取证据

correction:
  更新 mu/Sigma/h/c/tokens

innovation:
  比较上一预测和当前真实观测
```

这非常有价值。最终版应该强化它，而不是用 Transformer cache 替掉它。

数学上，posterior update 可以写成 precision update：

```math
\Lambda_t^+
=
\Lambda_t^-
+
\Lambda_t^{meas}
```

```math
\eta_t^+
=
\Lambda_t^- \mu_t^-
+
\Lambda_t^{meas} \mu_t^{meas}
```

```math
\mu_t^+
=
(\Lambda_t^+)^{-1} \eta_t^+
```

内容向量可以用 GRU / LSTM gate：

```math
c_t^+
=
c_t^-
+
K_t \odot (\tilde c_t^{meas} - c_t^-)
```

其中 gate `K_t` 应该由：

```text
measurement confidence
support entropy
innovation norm
modality validity
posterior uncertainty
```

共同决定。

这正是你担心的“模型对比当前状态和应该状态”的能力。它应该保留并增强。

---

# 5. KV cache / Transformer cache 应该怎么用？

## 5.1 KV cache 能带来什么

KV cache / Transformer-XL-style recurrence 有两个明显价值：

```text
1. 减少重复计算：
   历史 tokens 不必每步重新编码。

2. 增强长时一致性：
   action head / task selector 可以访问更长历史。
```

Transformer-XL 用 segment-level recurrence 解决固定上下文和 context fragmentation；GTrXL 则显示 gated Transformer-XL 在部分可观测 RL 中可以比 LSTM 更稳定、更强。([arXiv][8])

近期 causal world modeling for robot control 也强调，用 autoregressive history 和 KV cache 可以提高实时推理效率；但同一篇也指出，如果缓存 stale predicted visual content，模型可能继续“相信”幻觉视频而忽视真实反馈，导致 open-loop drift，因此需要基于最新真实观测的 feedback-grounded step 来重新对齐环境。([arXiv][9])

这对你们非常重要。

---

## 5.2 KV cache 不能替代 posterior

我建议严格规定：

```text
posterior:
  authoritative belief state
  用于闭环纠错、当前动作、对象身份、几何不确定性。

KV/evidence cache:
  auxiliary episodic context
  用于长历史检索、语言一致性、过去 evidence 回看、减少计算。
```

不要让 action head 直接把 cache 当真相。否则会出现：

```text
1. stale memory:
   历史里那根筷子的位置已经过时。

2. hallucinated continuity:
   模型继续相信上一段预测，而不看当前观测。

3. posterior bypass:
   action head 直接读历史 hidden states，绕过 innovation/correction。

4. identity inertia:
   slot 明明已经绑定错了，但 cache 强化旧绑定，纠错变慢。

5. training/inference mismatch:
   训练中 teacher-forced 历史干净，部署时 cache 里有模型自己的错误。
```

因此 KV cache 应该是：

```text
bounded
causal
age-aware
uncertainty-aware
source-aware
innovation-gated
posterior-grounded
```

建议每个 cache entry 带 metadata：

```text
source:
  real_observation / posterior / predicted

age:
  how old

slot_address:
  which object/slot

uncertainty:
  how reliable

innovation_at_write:
  whether model was surprised then

validity_mask:
  modality availability
```

cache read gate：

```math
g_{cache}
=
\sigma(
  W[
    q_j,\ 
    age,\ 
    uncertainty,\ 
    innovation_t,\ 
    source
  ]
)
```

如果当前 innovation 很高：

```math
||\nu_t|| > threshold
```

则：

```text
降低旧 cache 权重
优先当前 measurement
必要时 reset 某些 slot-cache links
```

换句话说：

> **KV cache 是长时 evidence retrieval；posterior 是当前物理 belief；innovation 是纠错开关。三者不能混成一个东西。**

---

# 6. last-2 frames 的最终设计：不要 mean，要 typed temporal V-JEPA support

当前 `use_last_two_mean` 的数学问题很简单。

假设：

```math
z_t = h(s_t) + \epsilon_t
z_{t-1} = h(s_{t-1}) + \epsilon_{t-1}
```

平均：

```math
\bar z_t = \frac{1}{2}(z_t + z_{t-1})
```

如果状态几乎不动，噪声独立：

```math
Var(\bar\epsilon_t) = \frac{1}{2} Var(\epsilon_t)
```

所以更稳。

但如果物体或手在移动：

```math
s_t \neq s_{t-1}
```

平均偏差为：

```math
E[\bar z_t - h(s_t)]
\approx
\frac{1}{2}(h(s_{t-1}) - h(s_t))
```

这会带来：

```text
motion smear
contact lag
thin-object boundary blur
错过接触瞬间
```

因此最终版应该是：

```text
V-JEPA recent temporal tokens:
  z_{t-1,h,w}
  z_{t,h,w}

optional:
  delta = z_t - z_{t-1}
  mean = (z_t + z_{t-1}) / 2
```

但不要只给 mean。

AQR 读取：

```math
p_{j}^{vjepa}(\tau,h,w)
=
\operatorname{softmax}_{\tau,h,w}
\left(
  q_j^\top k_{\tau,h,w}^{vjepa}
  +
  b_{\tau,h,w}
\right)
```

其中：

```text
τ ∈ {t-1, t}
```

如果要更贴近 JEPA-VLA：

```text
two most recent frames
  -> V-JEPA 2 embeddings
  -> projection
  -> gated cross-attn into AQR/task/action tokens
```

如果要更贴近 VLA-JEPA：

```text
current/past observations only enter student path
future latent state只作为 target
不把未来信息泄露进当前 action input
```

VLA-JEPA 的核心正是 leakage-free state prediction：target encoder 看 future latent，student pathway 只看 current observation，future 只作为 supervision target。([arXiv][2])

---

# 7. 高分辨率 crop 做不到时，怎么做 fine grounding？

你指出“无法提供更好数据集，所以 high-res local crop 可能做不到”，这个很关键。我修正前面的建议：

```text
如果原始数据只保留 384 输入，
或者训练管线无法访问更高分辨率原图，
那么 high-res crop 不能创造额外信息。
```

但是，仍然可以做三种不需要新数据的 refinement。

---

## 7.1 Latent local refinement

不是 crop 原图，而是在现有 tokens 上做二级局部读取。

第一步 coarse routing：

```math
p_{j,i}^{global}
```

选 top-k evidence：

```math
\Omega_j
=
TopK_i(p_{j,i}^{vjepa})
\cup
TopK_i(p_{j,i}^{point})
\cup
TopK_i(p_{j,i}^{pg})
\cup
TopK_i(p_{j,i}^{posterior})
```

第二步只在 `Ω_j` 内做 finer competition：

```math
p_{j,i}^{local}
=
\operatorname{softmax}_{i \in \Omega_j}
(
q_j^\top k_i + b_{relation}+b_{geometry}
)
```

这不会突破输入分辨率上限，但能减少全局 distractors，让“第四根筷子”这种关系在小候选集合里更容易比较。

---

## 7.2 Point-neighborhood refinement

如果当前数据有 depth/point tokens，就不需要新标注，也不需要高分辨率 RGB。

coarse anchor 选一个局部点云邻域：

```math
\mathcal N_j
=
\{p_i: ||p_i - \mu_j|| < r_j\}
```

然后在点邻域里做：

```text
local point attention
local geometry PCA
thin-object axis estimation
contact feasibility
ordinal ordering
```

这对筷子这类细长物体可能比 RGB token 更有效，因为点云提供 metric separation。

---

## 7.3 Temporal disambiguation

即使单帧里几根筷子接近，动作过程中的 parallax、夹爪接近、遮挡变化、接触反馈可能提供区分。

所以：

```text
不是只问 t 时刻能不能分出第四根；
还要让 posterior 在 t-3:t 的 evidence 中累计身份。
```

这正是 posterior + cache + V-JEPA temporal support 的价值。

如果第 4 根一开始和第 3 根几乎重合，但夹爪靠近过程中视角/遮挡/触觉让它们分开，posterior 应该能更新绑定。

---

# 8. Ordinal / relation 不是可选项：必须显式建模

“从左往右第四根”是 ordinal referring，不是普通 object grounding。

我建议加入一个 task relation head：

```math
r_{j}
=
f_{\theta}(
  q^{task},
  S_{t,j}^{phys},
  \{S_{t,l}^{phys}\}_{l=1}^K,
  language
)
```

对每个候选 slot，计算 task axis：

```math
u_\ell
=
g_{\theta}(language,\ camera/world frame,\ task context)
```

候选排序分数：

```math
s_j = u_\ell^\top \mu_{t,j}
```

soft rank：

```math
rank_j
=
1
+
\sum_{l \ne j}
\sigma
\left(
  \frac{s_l - s_j}{\tau_{rank}}
\right)
```

如果语言里有：

```text
first / second / third / fourth
left / right / front / back
nearest / farthest
```

则加低权重辅助 loss：

```math
L_{rank}
=
Huber(rank_{selected} - r_\ell)
```

或者 pairwise relation loss：

```math
L_{pair}
=
\sum_{l}
CE(
  sign(s_j - s_l),
  relation_\ell(j,l)
)
```

DOrA 这类 order-aware 3D visual grounding 工作也把 referential order 当成独立问题，而不是普通点/框预测。([arXiv][10])

如果没有人工 rank 标签，可以用弱监督：

```text
1. 从语言里解析 ordinal words；
2. 用 point/world/image 坐标估计候选顺序；
3. 用 demo action/contact 反推被操作对象；
4. 只在高置信候选场景使用该 loss；
5. 低权重，不让它压过 action loss。
```

这不需要新数据集，但需要谨慎过滤。

---

# 9. JEPA-style slot world model：统一 temporal、posterior、cache 的关键

当前 `physical_prediction_cache` 和 `_innovation()` 已经有预测/纠错雏形。我建议把它升级成 **slot-level JEPA world model**。

当前做法更像 global prediction cache：

```text
predict visual_latent / tactile / point summary
compare current targets
make innovation token
```

最终版应该做：

```text
for each slot j:
  predict next slot content
  predict next geometry distribution
  predict next support distribution
  predict next contact state
```

形式：

```math
\hat S_{t+1,j}
=
F_\theta(S_{t,j},\ a_t,\ proprio_t,\ language)
```

target 不用 pixel generation，而是下一时刻 posterior / target encoder 的 latent：

```math
S^{target}_{t+1}
=
E^{target}(o_{t+1})
```

JEPA loss：

```math
L_{jepa}
=
\sum_j
d(
  \hat c_{t+1,j},
  stopgrad(c^{target}_{t+1,\pi(j)})
)
```

其中 `π(j)` 是 slot matching，可由 address/binding/Sinkhorn 决定。

同时预测 support：

```math
L_{support-pred}
=
KL(
  stopgrad(p_{t+1,j}^{target})
  ||
  \hat p_{t+1,j}
)
```

这和 VLA-JEPA 的关键原则一致：**future latent 只作为 target，不作为当前输入，避免 leakage**。([arXiv][2])

这样一来，posterior、V-JEPA temporal、KV cache、innovation 不是拼起来的，而是同一个闭环：

```text
posterior:
  当前 belief

world model:
  预测下一 belief

innovation:
  当前 evidence 与预测的差

cache:
  历史 belief/evidence 的可控检索

action:
  根据 belief + innovation + task 生成动作
```

---

# 10. 数据适配：当前 slot 架构其实很适合“各种数据”

你说“为了塑造原生多模态大模型，需要让模型适配各种数据”，这点我同意，而且这正是 typed slots 比 dense all-to-all 更有优势的地方。

不同数据可能有：

```text
只有 RGB + language
RGB + proprio
RGB + point
RGB + tactile
multi-view RGB
human video without robot action
robot demo with action
missing wrist camera
missing tactile
```

如果做一个巨大的统一 dense Transformer，很容易要求所有模态都齐全。

typed slot memory 可以自然处理缺失模态：

```math
M_t^{(m)} =
\emptyset
\quad \Rightarrow \quad
mask_m = 0
```

slot update 变成：

```math
q_j^{l+1}
=
q_j^l
+
\sum_{m \in available}
\gamma_{j,m}^l
\operatorname{Attn}(q_j^l, M_t^{(m)})
```

训练时做 modality dropout：

```text
随机 drop PG image
随机 drop point
随机 drop tactile
随机 drop wrist view
随机 drop V-JEPA temporal branch
```

迫使 posterior 学到：

```text
有某模态时利用它；
没有某模态时提高不确定性；
不要把某个模态当成唯一 truth。
```

数据 scaling law 也显示，robot imitation learning 中环境和对象多样性比单纯增加同一环境 demos 更重要；因此架构应该能吃 heterogenous data，而不是要求一个完美齐全的数据合同。([arXiv][11])

---

# 11. 最终损失函数：少而统一，不要乱加

我建议最终总 loss 是：

```math
L
=
L_{action}
+
\lambda_{jepa} L_{slot-jepa}
+
\lambda_{support} L_{support-pred}
+
\lambda_{bind} L_{binding-consistency}
+
\lambda_{div} L_{slot-diversity}
+
\lambda_{xmod} L_{cross-modal-align}
+
\lambda_{rank} L_{ordinal}
+
\lambda_{innov} L_{innovation-calib}
+
\lambda_{mask} L_{masked-modality}
```

每项必须服务于 belief-state 统一目标。

---

## 11.1 `L_action`

保持 PI0.5 action path，不要替换。

```text
AQR 是 structured world interface；
PI0.5 是 action generator。
```

这点你们 handoff 里已经说得对。

---

## 11.2 `L_slot-jepa`

让 slot state 可预测：

```math
L_{slot-jepa}
=
\sum_j
\|
\hat c_{t+1,j}
-
stopgrad(c_{t+1,\pi(j)})
\|_2^2
```

或者 cosine / VICReg-style latent loss。

---

## 11.3 `L_support-pred`

让 slot 不只是预测 content，还预测未来 where：

```math
L_{support-pred}
=
\sum_{j,m}
KL(
  stopgrad(p_{t+1,j}^{(m)})
  ||
  \hat p_{t+1,j}^{(m)}
)
```

这直接约束 object identity 和 support persistence。

---

## 11.4 `L_binding-consistency`

防止 slot identity 乱跳：

```math
L_{bind}
=
CE(
  B_{t \rightarrow t+1},
  stopgrad(\tilde B_{t \rightarrow t+1})
)
```

其中 binding target 可以来自 posterior matching、geometry proximity、support overlap、action contact。

---

## 11.5 `L_slot-diversity`

防止所有 slots 绑定同一证据：

```math
L_{div}
=
\sum_{j \ne k}
\langle p_j,\ p_k \rangle
```

或 Sinkhorn entropy/overlap penalty。

---

## 11.6 `L_cross-modal-align`

对同一 slot，不同模态的 support 应该一致：

```math
L_{xmod}
=
d(
  \phi_v(p_j^{vjepa}),
  \phi_p(p_j^{point})
)
+
d(
  \phi_{pg}(p_j^{pg}),
  \phi_v(p_j^{vjepa})
)
+
...
```

只在 modality valid 时启用。

---

## 11.7 `L_ordinal`

只在语言里有 clear ordinal / relation 时启用：

```math
L_{rank}
=
Huber(rank_{selected} - r_\ell)
```

低权重，避免弱标签噪声伤害主任务。

---

## 11.8 `L_innovation-calib`

让不确定性和 surprise 匹配：

```math
L_{innov}
=
-
\log
\mathcal N(
  y_t;
  \hat y_t,
  \Sigma_t^{pred}
)
```

或者对 standardized residual 做 calibration：

```math
\nu_t
=
\Sigma^{-1/2}
(y_t - \hat y_t)
```

目标是：

```text
模型预测错时，posterior 应该知道自己错了；
cache 应该降权；
measurement 应该升权。
```

---

# 12. 实现路线：在当前代码上怎么改

我建议不要一次性重写，而是按下面顺序升级。

---

## Phase 1：把 last-two-mean 改成 JEPA-style temporal tokens

当前：

```text
VjepaFeatureMap.current_map(use_last_two_mean)
```

新增：

```python
recent_maps(n: int = 2) -> [n, H, W, C]
```

在 `pipeline._visual_map()` 里不要只返回 `[H,W,C]`，而是构造：

```text
visual_temporal_tokens:
  [T_recent * H * W, C]

visual_time_ids:
  [T_recent * H * W]

visual_xy_ids:
  [T_recent * H * W, 2]
```

AQR visual reader 改成读：

```text
M_vjepa[t,h,w]
```

保留旧模式作为 ablation：

```text
last_only
last_two_mean
last_two_tokens
last_mean_delta
last4_tokens
```

我建议默认从：

```text
last_two_tokens + optional delta token
```

开始。

---

## Phase 2：保留 PaliGemma image support 作为独立 branch

当前 `_aqr_pg_image_support_read()` 主要使用第一个 image-token range/view，并把结果 remap 到 V-JEPA grid bias。

改成：

```text
for each PG image view/range:
  q_j attends to PG image tokens
  output p_pg_img[j, view, token]

store:
  graph.pg_priors = p_pg_img

optional:
  use p_pg_img to bias V-JEPA/point,
  but do not let it disappear.
```

这样 PaliGemma 的细语义不会只变成一张 V-JEPA scalar bias。

---

## Phase 3：把 posterior slots 显式拆成 address/content

当前 posterior 已有 `h/c/tokens/mu/Sigma/x/S/alpha`。

新增或重构：

```text
slot_address a_j:
  persistent learned vector or slowly updated identity vector

slot_content c_tj:
  current hidden/content

slot_geometry:
  mu/Sigma/x/S

slot_validity:
  alpha/support_mass/contact_prob
```

routing 时：

```text
address 用于 identity/binding；
content 用于 measurement/action。
```

不要让 address 在 residual stream 里被随意覆盖。可以参考 OA-WAM 的 address-only key 思路：attention key 的一部分只读 address，从张量结构上防止 identity/content 混淆。([arXiv][3])

---

## Phase 4：加入 bounded evidence KV cache

不要用普通 LLM KV cache 直接替 posterior。新增：

```text
PicfEvidenceCache:
  posterior_slot_tokens[t-H:t]
  topk evidence tokens per slot
  action/proprio tokens
  innovation tokens
  source flags
  age
  uncertainty
  validity mask
```

读取方式：

```math
q_j
\leftarrow
q_j
+
g_{cache}
\operatorname{Attn}(q_j,\ C_{t-H:t-1})
```

但：

```text
posterior update 先读当前 real measurement；
cache 只能作为 bias/context；
innovation high 时 cache 降权；
predicted-only cache entry 权重快速衰减。
```

---

## Phase 5：把 `_innovation()` 升级为 slot-level prediction error

当前 `_innovation()` 很有价值，应保留。

升级：

```text
global prediction cache
  -> slot-wise prediction cache

predict:
  c_{t+1,j}
  mu/Sigma_{t+1,j}
  visual support
  point support
  tactile/contact
```

当前 observation 到来后：

```text
compute slot innovation
update posterior gate
update cache trust
condition action head
```

这会把 world model 和 posterior 统一起来。

---

## Phase 6：加 relation/rank head，但不要重度依赖新标签

实现一个轻量 head：

```text
inputs:
  task query
  language tokens
  physical slots
  slot geometry
  support distributions

outputs:
  candidate score
  relation logits
  soft rank
```

只在可解析语言或高置信场景启用弱 loss。

对于没有 rank 标签的普通 demo：

```text
rank head 只作为 latent feature，
不强制训练或低权重自监督。
```

---

# 13. 关于“当前架构可能已经比较好用”的判断

我同意你的直觉：**当前基于 slot/anchor 的设计比很多 naive VLA 融合更适合走向原生多模态世界模型。**

原因是：

```text
1. slots 给不同模态提供共同对齐坐标；
2. typed memory 可以处理缺失模态；
3. query-to-memory scaling 比 dense all-to-all 更可扩展；
4. posterior 已经是 recurrent belief，不是纯 feed-forward；
5. AQR 可以自然承接 V-JEPA temporal evidence 和 PaliGemma semantic evidence；
6. PI0.5 action path 保持，不破坏已有控制能力。
```

所以我不建议推倒重来。

我建议的升级是：

```text
不要换掉 AQR；
把 AQR 从“anchor support router”
升级成“object-addressable predictive belief router”。
```

这不是画蛇添足，而是把已有部件放进一个数学一致的闭环：

```text
typed evidence
  -> slot assignment
  -> posterior correction
  -> world prediction
  -> innovation
  -> action
  -> next evidence
```

---

# 14. 自我批判：这套方案仍然不能突破信息论上限

需要非常明确：

如果当前输入中几根筷子已经完全不可分：

```math
I(Y;\ Z_t \mid language) \approx 0
```

其中 `Y` 是“第几根筷子”的真实身份，`Z_t` 是所有当前 typed memory，那么任何 architecture 都不能稳定恢复 `Y`。

Data-processing inequality 给出：

```math
I(Y;\ A_t)
\le
I(Y;\ Z_t)
```

也就是说：

```text
anchor / slot / Sinkhorn / cache / world model
都只能利用已有 evidence，
不能凭空创造不存在的信息。
```

这就是为什么我不再把 high-res crop 作为唯一答案；如果没有高分辨率输入，它确实做不到。但 slot-level temporal accumulation、point-neighborhood refinement、multi-view evidence、posterior identity、JEPA prediction 可以尽量降低架构损失：

```text
E_total =
  E_observation
  + E_architecture
  + E_data/optimization
```

当前你们还没有到 architecture upper bound。很多结构性改动仍然会降低 `E_architecture`。但如果 `E_observation` 很大，比如所有模态里几根筷子都不可分，那模型也不能保证成功。

---

# 15. 最终版一句话架构

我会把最终架构定成：

```text
PICF-AQR-OWM

A typed-memory, object-addressable, JEPA-predictive belief-state architecture:

  PaliGemma text/image tokens,
  V-JEPA recent temporal tokens,
  point/tactile/proprio/action tokens,
  and posterior/cache tokens

are routed by learned physical/task anchors into persistent address-content slots.

Each slot performs:
  prior prediction,
  multimodal measurement routing,
  uncertainty-aware posterior correction,
  causal evidence-cache reading,
  and future latent/support prediction.

Task anchors then select/refine these slots using language,
including ordinal/relation heads when needed,
while the PI0.5 action path remains the final action generator.
```

最终裁决如下：

```text
1. 当前 AQR/PICF 不是错路线，而是一个很好的骨架。

2. “缺 memory”这个说法不准确；
   你们已经有 posterior/RNN/innovation，
   但缺 temporal visual evidence memory 和 object-addressable cache。

3. KV cache 可以用，
   但只能作为有年龄/不确定性/真实观测校正的 evidence memory，
   不能替代 posterior belief。

4. last-2 frames 应该按 JEPA-VLA 思路作为 temporal tokens/gated cross-attn 输入，
   而不是简单 last-two mean。

5. PaliGemma image support 应该成为一等 typed support branch，
   不应只被压成 V-JEPA grid bias。

6. 没有 high-res 数据时，不要承诺亚 patch 精度；
   应该做 latent local refinement、point-neighborhood refinement、
   temporal/posterior disambiguation 和 ordinal relation head。

7. 最终的一体性来自 belief-state 数学：
   prediction -> measurement -> assignment -> correction -> innovation -> action。
```

这套方案的优势是：它不是把 JEPA、slot、cache、posterior、rank head 拼在一起，而是让它们分别对应 belief-state 系统里的必要组成部分。

[1]: https://arxiv.org/html/2602.11832v1 "JEPA-VLA: Video Predictive Embedding is Needed for VLA Models"
[2]: https://arxiv.org/html/2602.10098v2 "VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model"
[3]: https://arxiv.org/html/2605.06481v1 "OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation"
[4]: https://arxiv.org/html/2510.23095v3 "Revisiting Multimodal Positional Encoding in Vision–Language Models"
[5]: https://arxiv.org/abs/2303.05499 "[2303.05499] Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
[6]: https://huggingface.co/docs/transformers/model_doc/perceiver "Perceiver · Hugging Face"
[7]: https://arxiv.org/html/2503.06170v1 "Object-Centric World Model for Language-Guided Manipulation"
[8]: https://arxiv.org/abs/1901.02860 "[1901.02860] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
[9]: https://arxiv.org/html/2601.21998v1 "Causal World Modeling for Robot Control"
[10]: https://arxiv.org/abs/2403.16539?utm_source=chatgpt.com "Data-Efficient 3D Visual Grounding via Order-Aware Referring"
[11]: https://arxiv.org/abs/2410.18647 "[2410.18647] Data Scaling Laws in Imitation Learning for Robotic Manipulation"
