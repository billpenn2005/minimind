# 第 05 课 · 多头注意力（GQA + QK-Norm + KV-Cache）

> 难度：★★★☆☆（全教程最核心的一课）｜ 预计用时：3~4 小时 ｜ 前置：04 课 RoPE
> 学完本课你应能回答：**注意力在"看"什么？什么叫因果？KV 缓存在干嘛？为什么只缓存 K 和 V？GQA 省了多少显存？**

---

## 0. 本课目标

- [ ] 理解 Q/K/V、缩放点积注意力、多头、因果掩码的全部几何与代数；
- [ ] 理解 KV-Cache（生成阶段 O(T) 的秘密）与 GQA（8 头 Q 共享 4 头 KV）；
- [ ] 理解 QK-Norm 为什么是 Llama/Qwen 系小模型的稳定性法宝；
- [ ] 手写 `minimind3/attention.py`（含 SDPA 快路径）；
- [ ] 验收 6 项新检查（05.1~05.6，累计 27 项全过）。

---

## 1. 理论：注意力机制

### 1.1 一个直觉：图书馆查资料

你要写一篇关于"猫"的作文（**查询 query**）。你去图书馆，每本书的**书名（键 key）**告诉你哪本书相关，
**书的内容（值 value）**是你要抄的东西。注意力就是：**按"相关度"给每本书加权，然后加权求和书的内容**。

一个 token 的新表示 = 用它的 Q 与所有 K 算相关度 → softmax 成权重 → 对 V 加权求和。

### 1.2 数学：缩放点积注意力（Scaled Dot-Product Attention）

对序列中第 $i$ 个位置：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

逐步拆解（`B` 批大小，`S` 序列长，`d`=head_dim）：

1. **投影**：`x @ W_q`、`x @ W_k`、`x @ W_v`。minimind 的实现里 W_q 输出 `heads*d`、W_k/W_v 输出 `kv_heads*d`；
2. **切头**：`(B, S, heads*d)` → `(B, heads, S, d)`；
3. **打分**：`Q @ K^T`（形状 `(B, heads, S, S)`）——第 i 行第 j 列 = "第 i 个 token 有多关注第 j 个 token"；
4. **缩放**：除以 $\sqrt{d}$（=head_dim 的开方）。为什么？——点积的数值随维度增大而增大，softmax 会被推成
   "one-hot"（梯度消失）。$\sqrt{d}$ 把分数拉回温和区间（直觉：d 个独立单位方差分量的点积期望就是 d）；
5. **softmax 沿行**：归一成概率分布（和为 1）；
6. **聚合**：对 V 加权求和，得到该位置的输出。

数学巨人的一句话：**注意力的本质是"可微的查表"**——所有部件都可通过反向传播学习。

### 1.3 多头：多个"视角"

把 $d_{model}$ 切成 $H$ 个头，每个头独立做上述计算，最后拼起来再过 `W_o`。为什么？
- 每个头可以学到**不同关系**（语法 / 指代 / 语义……）；
- 计算量与单头相同（分头只是重排），但表达能力翻倍。

minimind3 默认：8 个 Q 头、head_dim = 96。

### 1.4 因果掩码：语言模型只能看"过去"

训练时我们一次喂**整段文本**，但第 $t$ 个 token 的预测**只能用前 $t-1$ 个 token**（否则就是"作弊"——
答案就在眼皮底下）。所以打分矩阵要加"下三角"约束：

$$
\text{mask}[i][j] = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

softmax 里 $e^{-\infty} = 0$ → 未来位置贡献为零。实现上（本课答案的做法）：

```python
mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
scores[..., :seq_len].masked_fill_(mask, float("-inf"))   # 只遮"新增的 seq_len 列"
```

**只遮新列**是增量生成的关键细节：`scores` 是 `(current, total)`，历史列代表"现在对过去的注意力"，
必须保留；新增列是"过去对未来的窥探"，必须遮掉。

### 1.5 KV-Cache：生成阶段不再重算历史

训练时一次性处理整个序列（并行）。**生成时**我们一次只产出**一个新 token**，然后把它拼到历史后面。
朴素的实现会把**整段历史重新过一遍注意力**（O(T²)），浪费到令人发指。KV-Cache 的观察：

> 历史 token 的 K、V，一旦算过，**以后不会再变**（它们只依赖自己与更早的 token）。

所以：把每轮的 K、V 存起来，下一轮只算新 token 的 Q、K、V，与缓存的 K、V 合并打分 → 单步代价
从 O(T²) 降到 **O(T)**。本课答案的缓存约定：**RoPE 旋转之后**再拼接（历史 K 已旋转，直接复用）。

> ⚠️ 谁没缓存？Q 不需要缓存——新 token 的 Q 只需要与所有 K 点积，不需要"过去的 Q"。
> 这就是"为什么只缓存 K 和 V"的答案。

### 1.6 GQA：少一半 KV，少一半显存，保九成效果

标准的 MHA（Multi-Head Attention）每个 Q 头配一个 K/V 头。minimind3 用 **GQA（Grouped Query Attention）**：
8 个 Q 头**分组共享** 4 个 K/V 头（每组 2 个 Q 头共享 1 个 KV 头）。效果：
- KV-Cache 显存直接**减半**（KV 头数从 8 → 4）；
- 训练推理都更快；
- 实测质量损失极小（MQA 全共享会损失质量，GQA 折中最优）。

实现要点（本课答案）：`repeat_kv` 用 `expand + reshape` 把 `(B, kv_heads, S, d)` 展开成 `(B, heads, S, d)`——
`expand` 不复制内存（视图），比 `repeat`（真正复制）省资源。展开顺序是**逐对复制**
`[kv0, kv0, kv1, kv1, ...]`（与 minimind 完全一致），测验里的手写参考实现必须跟它对齐。

### 1.7 QK-Norm：稳定性的小魔法

在 Q、K 各自身上的 **head_dim 维度**做一次 RMSNorm（直接复用第 03 课！）。作用：
- 把 Q、K 的尺度压平，注意力的 softmax 更温和（等价于一个"自适应缩放"）；
- 深模型里能显著减少数值爆炸——Gemma、Qwen3 等近年的小模型标配；
- 实现上就是 `RMSNorm(head_dim)` 放在"切头之后、RoPE 之前"。

---

## 2. 阅读参考答案（minimind3/attention.py）

固定骨架（务必对照手写一遍）：

1. `__init__`：`q_proj / k_proj / v_proj / o_proj` 全是 `nn.Linear(..., bias=False)`；
2. `q_norm / k_norm`：`RMSNorm(head_dim)`；
3. `forward(x, attention_mask, past_key_value, use_cache, layer_idx)`：
   - 投影 → 切头 `(B, S, H, d)` → **QK-Norm** → **RoPE**；
   - 拼接 past K/V → `repeat_kv` + `transpose(1,2)`；
   - **快路径**（无 past、序列>1、mask 全 1 时）：`F.scaled_dot_product_attention`（PyTorch 内置融合算子）；
   - **慢路径**（调试/教学用）：手写 scores → 缩放 → 因果 triu 遮罩 → softmax → dropout → `@ v`；
   - `o_proj` + `resid_dropout`，返回 `(output, present_kv)`。

> 为什么保留慢路径？**它是你的"活参考实现"**——验收 05.3 就是拿你的手写参考去对齐 SDPA 快路径，
> 保证两条路在数学上严格等价。

## 3. 手写任务清单

1. `repeat_kv` 模块级函数（expand → reshape，注意放在模块内可直接 import）；
2. `Attention.__init__`（9 个成员：4 个投影 + 2 个 norm + dropout 等）；
3. forward：投影/切头/过 QK-Norm/过 RoPE；
4. KV-Cache 拼接与 repeat_kv（先拼后展，注意维度顺序：`(2, B, kv, S, d)` 的 torch 惯例转置）；
5. 因果掩码：只对 `scores[..., -seq_len:]` 的新列做 triu 遮罩；padding mask 用 `-1e9`（不用 -inf 防 NaN 泄漏）；
6. 快/慢路径二选一（推荐都写，慢路径就是你的调试器）；
7. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 4. 验收解读（verify/05_attention.py）

| 检查 | 验什么 | 直觉 |
|---|---|---|
| 05.1 | 4 个投影的形状；repeat_kv 展开成 8 头且顺序是 `[kv0,kv0,kv1,...]` | 结构正确 |
| 05.2 | **因果性**：用 autograd 证明 token0 的输出**收不到** token1 的梯度 | 掩码真的生效 |
| 05.3 | 你的 forward == 手写参考注意力（手写 RoPE/掩码/softmax）在 1e-4 内 | 实现等价 |
| 05.4 | **KV-Cache 逐块增量 == 全量一次算完**（1e-4） | 缓存正确性（最重要） |
| 05.5 | padding 行不产生 NaN，有效位置与无掩码结果一致 | 掩码数值安全 |
| 05.6 | SDPA 快路径 == 慢路径（1e-4） | 两条路等价 |

> 05.2 是最优雅的因果性验证：给输入开 `requires_grad`，反向传播看梯度——token 0 对 token 1 输入的梯度
> 必须是 0（物理上"看不见"）。这不依赖任何掩码实现细节，纯行为断言。

## 5. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/attention.py` → `Attention` | `model/model_minimind.py` → `Attention` |

几乎逐行镜像（QK-Norm、repeat_kv、`scores[..., -seq_len:]` 遮罩、KV 拼在 RoPE 之后、SDPA 分支）。
教程额外的点：把慢路径单独保留并双向验证（标准实现只有快路径）。

## 6. 常见坑（全部被验收踩过，属真实教训）

- **`repeat` 代替 `expand+reshape`**：内存翻倍，且验收 05.1 的展开顺序断言会失败；
- **KV 拼接发生在 RoPE 之前**：历史 K 会二次旋转，长上下文结果错乱（验收 05.4 立刻抓住）；
- **因果掩码遮了全部矩阵**：生成时会把你"现在的注意力"也抹掉，输出 NaN 或全 0；
- **手写参考在未切头时就做 QK-Norm**：RMSNorm 的 weight 维度 `(24,)` vs `(96,)` 形状崩（这是 05.3 的真实坑）；
- **mask 用 -inf 在 fp16 下**：softmax 溢出 NaN；padding 用 `-1e9` 保平安；
- **`transpose` 只做视图**：后续 `.contiguous()` 漏了就静默出错。

## 7. 小结 & 下一课

- ✅ Q/K/V 投影 → 切头 → QK-Norm → RoPE → 打分/缩放/掩码/softmax → 聚合 → 输出投影；
- ✅ 因果掩码："只看过去"；缩放：防 softmax 硬化；多头：多视角；
- ✅ KV-Cache：生成 O(T²) → O(T)，只存 K、V（Q 不存）；
- ✅ GQA：KV 头减半省显存而效果几乎无损；
- ✅ QK-Norm：压平尺度防爆炸。

**下一课（06-feedforward）**：注意力负责"信息交换"，前馈层负责"信息加工"。你会写当代最强激活
**SwiGLU**，并理解为什么它的中间维度（2432）比 classic MLP 短一大截。