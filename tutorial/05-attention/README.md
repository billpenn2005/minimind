# 第 05 课 · Attention：GQA + 因果掩码 + KV-Cache

## 目标

手写 `minimind3/attention.py`：`Attention`（含 `repeat_kv`），带 GQA、QK-Norm、RoPE、
因果掩码、KV-Cache、SDPA 快路径，行为与 minimind 标准实现一致。

## 理论

### 1. 缩放点积注意力

$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

除以 $\sqrt{d_k}$ 防止点积随维度爆炸导致 softmax 饱和（梯度消失）。

### 2. MHA → GQA

- MHA：每个 head 独立 Q/K/V，参数量大、KV-Cache 大；
- GQA（分组查询注意力）：全部 query 头共享少量 KV 头（KV 头数整除 query 头数）。
  minimind 默认 8 query / 4 KV 头：KV-Cache 减半，质量损失很小。
  「共享」在实现上 = 复制 KV 头 `n_rep = q_heads // kv_heads` 份（`repeat_kv`）。

### 3. QK-Norm

对每个 head 的 q/k 向量做 RMSNorm：把点积的尺度拉到稳定区间，
是近年（Gemma/Qwen3）常用的训练稳定技巧。实现 = 复用第 03 课的 RMSNorm(head_dim)。

### 4. 因果掩码

语言模型只许看过去。对分数矩阵加**上三角 -inf**：

```
scores[i][j] = -inf  (j > i)   →  softmax 后权重为 0
```

实现技巧：只对"本次新增的 seq_len 列"加掩码（`scores[..., -seq_len:]`），
这样带 KV-Cache 续写时历史部分的因果性天然满足。

### 5. KV-Cache

生成第 t 步只需新 token 的 Q，K/V 复用历史的：`K_t = cat(K_prev, K_new)`。
Naive 生成每次重算全部位置是 O(T²) 内存/时间，KV-Cache 降为 O(T) 增量。
verify 05.4 验证"增量拼接结果 == 一次性全量结果"。

### 6. SDPA

`F.scaled_dot_product_attention` 是 torch 内置的融合注意力（flash/mem-efficient 后端）。
minimind 的做法 = 条件判断：无 past、无 mask 时才走 SDPA，否则手写分支兜底。
verify 05.6 保证两分支数值等价。

## 任务清单（先手写再看答案）

1. `repeat_kv`：KV 头按 n_rep 复制（用 expand+reshape，零拷贝）；
2. 四路线性投影，形状分别为 `(q_h·d, h)` / `(kv_h·d, h)` / `(kv_h·d, h)` / `(h, q_h·d)`；
3. 拆分 head、QK-Norm、RoPE、KV-Cache 拼接、repeat_kv、SDPA 或手写 softmax 注意力、o_proj。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：投影形状与 GQA 头扩展、**因果性（梯度法验证）**、与手写参考逐元素一致、
**KV-Cache 增量==全量**、padding 掩码、SDPA 与手写路径等价。

## 对照标准实现

`master:model/model_minimind.py → Attention / repeat_kv`（命名、形状、掩码写法逐行对应）。

## 常见坑

- 形状：q 是 `(B,S,q_h,d)`，先 `transpose(1,2)` 变 `(B,q_h,S,d)` 再算分数；
- KV-Cache 拼接要在 **RoPE 之前还是之后**？minimind 的约定：先拼后旋转——
  每个新 token 用自己的绝对位置旋转，历史 K 已是旋转过的，直接复用；
  因此拼接发生在 apply_rotary_pos_emb **之后**；
- 因果掩码加在 `scores[..., -seq_len:]` 上而不是整个矩阵，否则续写时历史会被错误屏蔽；
- repeat_kv 用 `expand`（视图）+ `reshape`，不要 `repeat`（拷贝）浪费内存。