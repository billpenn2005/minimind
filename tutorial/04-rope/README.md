# 第 04 课 · RoPE 旋转位置编码

> [上一课 03-rmsnorm](../03-rmsnorm/README.md) ← [目录](../../README.md) → [下一课 05-attention](../05-attention/README.md)
>
> 难度：★★☆☆☆ ｜ 预计用时 1~2 小时 ｜ 前置：03 课

## 0. 本课目标

- [ ] 理解"位置编码"解决什么问题、旋转法为何保相对位置；
- [ ] 手写 `minimind3/rope.py`：`precompute_freqs_cis` / `rotate_half` / `apply_rotary_pos_emb`；
- [ ] 验收 22 项累计（本课新增 04.1~04.6）。

## 1. 理论速览

- **问题**：注意力 $\text{score}(q,k)$ 与 token 顺序无关（置换等变）——"猫吃鱼"和"鱼吃猫"打出的分数一样。必须注入位置。
- **家族对比**：加法式（sinusoidal，2017）| 可学习式（训练学位置向量）| **旋转式 RoPE（当代标配）**。
- **旋转几何**：把每 2 个特征当一组平面，按角度 $\theta_t = t \cdot \theta$ 旋转：
  $$R_{\theta}\binom{x_1}{x_2} = \binom{x_1\cos\theta - x_2\sin\theta}{x_1\sin\theta + x_2\cos\theta}$$
- **相对位置不变性（核心定理）**：内积只差相对角：
  $$R_{\theta_{t_1}}q \cdot R_{\theta_{t_2}}k = q \cdot R_{\theta_{t_2-t_1}}k$$
- **频率**：第 $i$ 组的角速度 $\theta_i = \text{rope\_base}^{-2i/d}$（从高频到低频）；高低频分工：低频管长程相对距离。`rope_base=1e6` 时 θ≈1，长文可用。
- **实现约定**：表形状 `[seq, head_dim]`——左右各一半拼一个 `cos` 表（GPT-NeoX 风格双半表）；应用公式 `q·cos + rotate_half(q)·sin`，其中 `rotate_half` 把后一半取负移到前一半。
- **YaRN**（`rope_scaling` 非 None）：低频段（被 beta_fast/beta_slow 界定）按 1/factor 缩放频率做长文外推；本课只需"能用"，不深究。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/rope.py`

**模块级**：`import math`、`import torch`。

### 2.2 函数规格

**`precompute_freqs_cis(dim: int, end: int = 32768, rope_base: float = 1e6, rope_scaling: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]`**

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dim` | `int` | — | head_dim（频率向量长度 = dim//2） |
| `end` | `int` | `32768` | 表行数（最大位置+1） |
| `rope_base` | `float` | `1e6` | 频率基数 |
| `rope_scaling` | `dict \| None` | `None` | YaRN 配置（None 关闭） |

| 返回 | 类型 | 形状 |
|---|---|---|
| `freqs_cos` | `torch.Tensor` | `[end, dim]`（float32） |
| `freqs_sin` | `torch.Tensor` | `[end, dim]`（float32） |

内部步骤（须与参考答案数值一致）：
1. `freqs = 1.0 / rope_base ** (arange(0, dim, 2)[: dim//2].float() / dim)` → `[dim//2]`；
2. 若 `rope_scaling` 且 `end/original_max_position_embeddings > 1`：按 beta 反推维度区间做 ramp 缩放（低频段除 factor）；`attn_factor = rope_scaling.get("attention_factor", 1.0)`；
3. `t = arange(end)`；`freqs = outer(t, freqs)` → `[end, dim//2]`；
4. `freqs_cos = cat([cos(freqs), cos(freqs)], -1) * attn_factor`；sin 同理。

**`rotate_half(x: torch.Tensor) -> torch.Tensor`**：`torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)`——后一半取负放前。

**`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]`**

| 参数 | 形状 | 说明 |
|---|---|---|
| `q` | `(B, S, H, head_dim)` | query，已拆头 |
| `k` | `(B, S, KVH, head_dim)` | key |
| `cos`/`sin` | `(S, head_dim)` | 从 precompute 表按位置切片 |
| `unsqueeze_dim` | `int = 1` | 在 S 维后插 1 以便广播到 head 维 |

返回 `(q_embed, k_embed)`，形状与输入同、dtype 还原为 q/k 的 dtype：
`q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)`。

## 3. 手写步骤

按 §2.2 三分步实现 → `.venv/Scripts/python.exe verify.py 04`。

## 4. 验收解读（verify/04_rope.py）

| 检查（新） | 验什么 |
|---|---|
| 04.1 | 表形状 `[end, dim]`、数值范围 |cos|≤1；相邻行角度差 = θ |
| 04.2 | **保范**：`||apply_rotary_pos_emb(q)|| == ||q||`（旋转不改变长度） |
| 04.3 | **旋转正交**：随机 x 下 `x·Rθx` 与手动 2D 旋转等价（1e-5） |
| 04.4 | **相对位置不变性**：位置 t1/t2 的内积 ≈ 位置 (t2-t1)/0 的内积（1e-5） |
| 04.5 | 频率单调：`theta_i` 随 i 递减；`rope_base` 生效 |
| 04.6 | YaRN：`rope_scaling` 非 None 时表不报错且形状不变 |

## 5. 参考答案

`answers/minimind3/rope.py`（先写后对；重点对"双半表"与 `unsqueeze_dim=1` 这两处默认）。

## 6. 常见坑

- **半维 vs 全维混淆**：频率长度 `dim//2`，表却是 `[end, dim]`（cat 了两个 cos）；
- **`unsqueeze` 位置错**：`cos.unsqueeze(1)` 把 `(S,d)` → `(S,1,d)`，才能对 `(B,S,H,d)` 广播到 head；
- **dtype 漂移**：sin/cos 是 float32，乘完要 `.to(q.dtype)`（fp16 训练时尤其）；
- **忘 `attn_factor`**：YaRN 下输出整体缩放错。

## 7. 对照标准实现

`answers/minimind3/rope.py` 与 minimind `model/model_minimind.py` 的 RoPE 段逐行等价（含 `rope_scaling` ramp 公式）。

## 8. 小结

✅ 位置注入完成：旋转、保范、相对位置不变性全部数值验证通过。
**下一课**：把 RoPE 装进 GQA 注意力——本教程最核心的一课。