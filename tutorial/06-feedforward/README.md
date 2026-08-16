# 第 06 课 · FeedForward：SwiGLU 前馈网络

## 目标

手写 `minimind3/feed_forward.py` 的 `FeedForward`（SwiGLU 门控前馈），
与 minimind 标准实现一致。

## 理论

### 1. 为什么需要 MLP

Attention 只做"token 之间的信息交换"（混合位置），且是线性映射的组合。
MLP 提供**逐位置的非线性变换**（混合特征维度），是模型表达力的主要来源。
Transformer 块 = Attention（交换位置信息） + MLP（加工特征），交替堆叠。

### 2. 经典 MLP → SwiGLU

经典：`FFN(x) = ReLU(xW1 + b1)W2 + b2`（两个矩阵）。

SwiGLU（Shazeer 2020）：引入第三个投影做"门控"：

$$FFN(x) = W_{down} \cdot \big(\text{SiLU}(W_{gate}x) \odot W_{up}x\big)$$

- SiLU（Swish-1）：$\text{SiLU}(x) = x\cdot\sigma(x)$，光滑、无界、允许负值通过；
- 门控投影决定"这个特征该以多大比例注入"——信息选择能力更强；
- 参数量是经典的两倍多（3 个矩阵），但同等参数量下效果更好，
  因此模型普遍把 `intermediate_size` 缩小（约 2.5~3.17 倍 d_model）。

### 3. minimind 的取法

`intermediate_size = ceil(hidden_size * π / 64) * 64`（第 02 课已实现），
768 → 2432 ≈ 3.17·d_model。

## 任务清单（先手写再看答案）

1. `FeedForward(config, intermediate_size=None)`：gate/up/down 三个无偏置 Linear；
2. `forward(x)`：`down(silu(gate(x)) * up(x))`；
3. 激活函数从 `ACT2FN` 取（与 transformers 生态一致）。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：投影形状、公式数值、SiLU 三种等价写法、中间维度覆盖、梯度与零输入非线性。

## 对照标准实现

`master:model/model_minimind.py → FeedForward`（gate/up/down 命名一致；标准实现同时
承担 MoE 的专家单元，教程里 MoE 留作进阶）。

## 常见坑

- 三个投影都**没有 bias**（现代 LLM 惯例，省参数且稳定）；
- `silu(gate(x)) * up(x)` 的乘法是逐元素 Hadamard 积，不是矩阵乘；
- 不要直接在模块里硬编码激活，从 `ACT2FN[config.hidden_act]` 取，
  后续 MoE/量化等扩展都不需要改这里。