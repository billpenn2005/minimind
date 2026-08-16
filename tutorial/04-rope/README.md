# 第 04 课 · RoPE：旋转位置编码

## 目标

手写 `minimind3/rope.py`：`precompute_freqs_cis / rotate_half / apply_rotary_pos_emb`，
行为与 minimind 标准实现一致（包括 YaRN 长文本缩放）。

## 理论

### 1. 为什么需要位置编码

Self-Attention 是**置换等变**的：把输入 token 顺序打乱，输出只是跟着打乱。
但语言是有顺序的，因此必须把"位置信息"注入。老方案（绝对位置）存在外推难的问题，
RoPE 的目标是：**让注意力分数只依赖 token 间的相对距离**。

### 2. 旋转的几何直觉

把第 $i$ 维和第 $i+d/2$ 维看作复平面上的实部/虚部，位置 $m$ 的信息 = 旋转角 $m\theta_i$：

$$\theta_i = \text{base}^{-2i/d}, \quad i = 0, \dots, d/2-1$$

旋转后向量的点积满足：

$$\langle R(q, m), R(k, n)\rangle = \langle R(q, m+\Delta), R(k, n+\Delta)\rangle$$

即**相对位置不变性**——verify 04.4 专门验证它。低频（小 $i$）旋转慢、长距离仍可分；
高频旋转快，负责局部细节。这就是 base（如 1e6）很大的原因：让低频更密，利于长文本。

### 3. 实现形态（GPT-NeoX 风格）

- 预计算：`freqs = outer(t, 1/base^(2i/d))` 得 $[end, d/2]$，拼接两个相同半表得 $[end, d]$；
- 应用：$q' = q\cos + \mathrm{rotate\_half}(q)\sin$，其中
  $\mathrm{rotate\_half}([x_1..x_{d/2}, x_{d/2+1}..x_d]) = [-x_{d/2+1}..-x_d, x_1..x_{d/2}]$。

### 4. YaRN（选读）

长上下文扩展：对维度做"分段频率缩放"——低频（对应 beta_fast 界）保持原频
（维持远距离区分度），高频（对应 beta_slow 界）压缩 $1/\text{factor}$，
中间用 ramp 线性过渡。

## 任务清单（先手写再看答案）

1. `precompute_freqs_cis(dim, end, rope_base, rope_scaling)`：
   频率向量 → （可选 YaRN ramp）→ 外积 → cos/sin 双半表；
2. `rotate_half(x)`：后半取负放前，前半放后；
3. `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)`：
   按公式旋转，返回与输入同 dtype。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：频率表数值公式、旋转保模、公式等价、**点积平移不变性**、dtype 保持、YaRN 生效。

## 对照标准实现

`master:model/model_minimind.py → precompute_freqs_cis / apply_rotary_pos_emb`
（含 YaRN 的 beta_fast/beta_slow 维度换算）。

## 常见坑

- `torch.arange(0, dim, 2)[: dim // 2]` 取的是半维下标 0,2,4,...；
- cos/sin 表与 head_dim 对齐——下一课 Attention 里 q/k 的最后一维就是 head_dim；
- `unsqueeze_dim` 记得用（q 的形状是 B,S,H,D，位置在第 1 维后插入）；