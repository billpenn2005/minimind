# 第 04 课 · RoPE 旋转位置编码

> 难度：★★☆☆☆ ｜ 预计用时：2~3 小时 ｜ 前置：第 03 课 + 会看"向量点积"
> 学完本课你应能回答：**为什么需要位置编码？RoPE 的"旋转"到底转了什么？"相对位置不变性"为什么是它的王牌？**

---

## 0. 本课目标

- [ ] 理解"注意力没顺序感"问题与位置编码的必要性；
- [ ] 从零推导 RoPE 的频率公式与二维旋转矩阵；
- [ ] 手写 `minimind3/rope.py` 的三个函数：`precompute_freqs_cis` / `rotate_half` / `apply_rotary_pos_emb`；
- [ ] 验收 6 项新检查（04.1~04.6，累计 21 项全过）。

---

## 1. 理论：为什么需要位置编码？

### 1.1 注意力的"失忆"问题

第 05 课你会学到，注意力机制本质上就是"根据内容相似度打分"：

$$
\text{score}(q_i, k_j) = q_i \cdot k_j \quad\text{（查询 i 对键 j 的匹配程度）}
$$

问题来了：**点积完全不在乎 j 在 i 前面还是后面、距离多远**。"猫 吃 鱼"和"鱼 吃 猫"在你还没看位置信息时，
对注意力是完全一样的——因为向量内容一模一样。这叫做**排列等变性（permutation equivariance）**：
把输入洗牌，输出只是跟着洗牌。语言可不允许这样。

### 1.2 解决方案家族

| 方案 | 思路 | 代表 |
|---|---|---|
| 绝对位置相加（Additive） | 位置向量加到词向量上 | 原始 Transformer、GPT-2 |
| 可学习位置表（Learned） | 用一张可学习的位置"码表" | BERT、GPT-3 |
| **旋转位置编码（RoPE）** | 把词向量"旋转"一个与位置有关的**角度** | Llama/Qwen/Gemma/PaliGemma，以及我们的 minimind3 |

RoPE 是今天（2025+）LLM 的绝对主流。它的杀手锏：**乘性而非加性**——不改变向量幅度（范数不变），
只会"转"方向；并且相对位置的差异恰好等于两向量间的"转角差"，从而**仅依赖相对位置**。

## 2. 理论：RoPE 的旋转几何（一步步推导）

### 2.1 二维旋转

先看 2 维向量 $x=(x_0, x_1)$。设位置为 $t$，把它顺时针旋转角度 $\theta t$（$\theta$ 是频率）：

$$
R(\theta t)\, x =
\begin{pmatrix}
\cos \theta t & -\sin \theta t \\
\sin \theta t & \phantom{-}\cos \theta t
\end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
=
\begin{pmatrix} x_0 \cos\theta t - x_1 \sin\theta t \\ x_0 \sin\theta t + x_1 \cos\theta t \end{pmatrix}
$$

**旋转保持模长**：$|R x| = |x|$（旋转不拉伸），这是验收 04.2 的数学根据。

**关键性质**：旋转后做点积，等于"旋转前做点积但带相对转角"——旋转矩阵的正交性给出：

$$
\langle R_{\theta t_1} q,\; R_{\theta t_2} k \rangle = \langle q,\; R_{\theta (t_2 - t_1)} k \rangle
$$

即：**注意力分数只取决于位置差 $(t_2-t_1)$**——这就是"相对位置不变性"，验收 04.5 会精确验证它。

### 2.2 从 2 维推广到 d 维

我们的词向量是 $d$ 维（head_dim，如 96）。把 $d$ 维拆成 $d/2$ 对"二维子平面"，每对用**不同频率**旋转：

$$
\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, \dots, d/2-1
$$

频率从高频（base=1e6 → $\theta_0 = 1$ 附近，转得快）到低频（$i$ 大 → 转得慢）**指数递减**。
直觉：高频分量负责"相邻 token"的细粒度区分，低频分量负责"长距离"的大尺度位置。

位置 $t$ 在第 $i$ 个通道上的转角为 $\theta_i \cdot t$。于是预计算的 cos/sin 表：

$$
\cos[t][i] = \cos(\theta_i \, t), \qquad \sin[t][i] = \sin(\theta_i \, t)
$$

表形状为 `[max_pos, d]`，但为了配合"旋转一定发生在**成对通道**上"的事实，通常把 cos/sin 表设计成
**两半相同**（每半形状 `[max_pos, d/2]`），一个令牌只取**上半**：

```
cos 表:  [max_pos, d/2] 与 [max_pos, d/2] 完全相同 → 拼接成 [max_pos, d]
rotate_half(x) = concat(-x[d/2:], x[:d/2])     # 交换两半并给前半取负
```

所以算子层面，旋转 `q*cos + rotate_half(q)*sin` 就等价于 §2.1 的二维旋转在成对通道上的逐对执行。

### 2.3 YaRN 长上下文扩展（本课可选理解）

`precompute_freqs_cis(dim, end, rope_base, rope_scaling)` 的 `rope_scaling` 参数（第 02 课 config 里的
`inference_rope_scaling`）会在**推理**时把频率"拉平"，让 2048 训练长度的模型外推到 32768 上下文。
做法：把 `beta_fast/beta_slow` 换算成维度下标并钳到 `[0, dim/2]`，只对中间区间的通道做线性缩放。
本课只需理解接口与验证直线（04.6：开启缩放后频率表会变）；实现细节借鉴 giant 的 YaRN 写法即可。

---

## 3. 阅读参考答案（minimind3/rope.py）

| 函数 | 职责 | 关键动作 |
|---|---|---|
| `precompute_freqs_cis` | 构建 `[end, dim]` 的 cos/sin 表 | 频率公式 → 与位置向量外积 → 双半拼接 → 应用 YaRN 缩放 |
| `rotate_half` | 成对旋转的"后半换位取负" | `torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)` |
| `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)` | 把 cos/sin 应用到 q/k | `q * cos + rotate_half(q) * sin`，按 `unsqueeze_dim` 插入头维度，返回输入 dtype |

难点只在**形状对齐**：q/k 是 `(B, num_heads, S, head_dim)`，cos/sin 是 `(S, dim)`——需要
`unsqueeze(unsqueeze_dim)` 把 `(S, dim)` 变成可广播的 `(1, 1, S, dim)`。

## 4. 手写任务清单

1. 写 `precompute_freqs_cis`：
   - 频域下半维 `freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))`（步长 2 覆盖半维）；
   - 位置 `t = arange(end)` 与 freqs 外积 `t[:, None] * freqs[None, :]`；
   - 拼接双半得到 `[end, dim]`，再乘 `attention_factor`（YaRN）；
   - `rope_scaling` 非空时，把 `beta_fast/beta_slow` 换算维度下标并线性插值裁剪频率。
2. 写 `rotate_half`；
3. 写 `apply_rotary_pos_emb`（保住输入 dtype：`.to(q.dtype)`）。
4. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 5. 验收解读（verify/04_rope.py）

| 检查 | 验什么 | 直觉 |
|---|---|---|
| 04.1 | 频率表形状 `[end, dim]`；双半表相同；`cos[t,i] == cos(t·base^(-2i/dim))` | 公式合法性 |
| 04.2 | 旋转后 q 的**模长不变** | 旋转的几何本质 |
| 04.3 | `apply_rotary_pos_emb` 结果 == 手写逐通道旋转公式 | 实现等价性 |
| 04.4 | 平移不变性：位置差固定、整体平移 δ，则 `q_t·k_{t+Δ}` 不变 | **相对位置不变性**（RoPE 的杀手锏） |
| 04.5 | 输出 dtype 与输入一致 | fp16 管线兼容 |
| 04.6 | 开启 YaRN 后频率表确实改变 | 接口生效 |

## 6. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/rope.py` → three functions | `model/model_minimind.py` → 同名单函数 |

注意 minimind 的 `precompute_freqs_cis` 是在模型层序里直接构建的（见 08 课的 buffer 设计）；本课把
它独立成 `rope.py` 便于单测——这也是教程"模块化"的一处微调。

## 7. 常见坑

- **半维/全维混淆**：频率公式用 `arange(0, dim, 2)` 走半维，表再拼接——别把 `dim` 当半维用；
- **形状广播错位**：q/k 与 cos/sin 的维度数不同，`unsqueeze` 一下（位置/头维）再相乘；
- **dtype 漂移**：cos/sin 若用不同 dtype，相乘会默默提升精度或报错——`type_as`/`.to()` 收尾；
- **忘记 `attention_factor`**：YaRN 的长距离注意力需要 1.0 基线，少了会偏。

## 8. 小结 & 下一课

- ✅ 注意力"没顺序感"→ 必须注入位置信息；
- ✅ RoPE 用**旋转**代替加法：保模长、乘法注入、相对位置不变；
- ✅ 频率从高频到低频指数递减 = 从局部到全局的"刻度尺"；
- ✅ 方向余弦表 + `rotate_half` 是实现精华。

**下一课（05-attention）**：最核心的一课！把 Query/Key/Value、缩放点积注意力、因果掩码、
**KV 缓存**（生成提速的关键）、**GQA**（minimind 8 个 Q 头共享 4 个 KV 头的省显存设计）、
以及 QK-Norm 稳定性技巧全部组装进 `Attention` 类。