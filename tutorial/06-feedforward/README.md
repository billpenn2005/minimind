# 第 06 课 · SwiGLU 前馈 FeedForward

> [上一课 05-attention](../05-attention/README.md) ← [目录](../../README.md) → [下一课 07-block](../07-block/README.md)
>
> 难度：★★☆☆☆ ｜ 预计用时 1 小时 ｜ 前置：02+05 课

## 0. 本课目标

- [ ] 理解"注意力交换信息，前馈加工信息"的分工；
- [ ] 手写 `minimind3/feed_forward.py` 的 `FeedForward`（SwiGLU）；
- [ ] 验收 33 项累计（本课新增 06.1~06.5）。

## 1. 理论速览

- **分工**：Attention 做 token 间的信息交换（广播/检索），FFN 逐 token 做非线性加工（把表示投影到更高维再压回）——两步交替 = 现代 Transformer 的呼吸。
- **经典 MLP**：`W2·σ(W1·x)`，中间维度 4×hidden（`intermediate_size`）。
- **SiLU**（$\sigma$ 为 sigmoid）：$\text{SiLU}(x) = x \cdot \sigma(x)$。
- **SwiGLU**（LLaMA/Qwen/minimind 标配）：
  $$\text{FFN}(x) = W_{down}\big(\text{SiLU}(W_{gate}\,x) \odot W_{up}\,x\big)$$
  门控投影经 SiLU 后与另一路逐元素相乘，再投影回 hidden。
- **为什么 intermediate≈3.2×hidden 不是 4×**：SwiGLU 三个投影矩阵 vs 经典 MLP 两个 → 同等中间维度参更多；minimind 用 `ceil(hidden·π/64)·64`（≈3.2×并 64 对齐）使总参数与经典 4× 相当。
- **bias=False 的原因**：Pre-Norm 下偏置收益有限；且转换（第 13 课）要求 strict 加载无多余键。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/feed_forward.py`

**模块级 import**：`from torch import nn`、`from transformers.activations import ACT2FN`、`from .config import MiniMindConfig`。

### 2.2 `class FeedForward(nn.Module)`

**`__init__(self, config: MiniMindConfig, intermediate_size: int | None = None)`**：

| 属性 | 类型 | 值 |
|---|---|---|
| `gate_proj` | `nn.Linear` | `(hidden → inter, bias=False)` |
| `down_proj` | `nn.Linear` | `(inter → hidden, bias=False)` |
| `up_proj` | `nn.Linear` | `(hidden → inter, bias=False)` |
| `act_fn` | callable | `ACT2FN[config.hidden_act]`（"silu"） |

其中 `intermediate_size` 参数传入则用之（`intermediate_size or config.intermediate_size`）。

**`forward(self, x: torch.Tensor) -> torch.Tensor`**：
- 参数 `x`：`(B, S, hidden)`；
- 返回：`(B, S, hidden)`；
- 主体一行：`return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`——**乘法是 gate 一路经 SiLU 后与 up 一路逐元素相乘**（不是相加）。

## 3. 手写步骤

实现 §2.2 → `.venv/Scripts/python.exe verify.py 06`。

## 4. 验收解读（verify/06_feedforward.py）

| 检查（新） | 验什么 |
|---|---|
| 06.1 | 投影形状（hidden→inter、inter→hidden），输出形状 `(B,S,hidden)` |
| 06.2 | 与手写 `down(silu(gate(x))*up(x))` 逐元素一致（1e-5） |
| 06.3 | SiLU 等价 `x*sigmoid(x)`；`ACT2FN["silu"]` 与手写一致 |
| 06.4 | `intermediate_size` 参数可覆盖 config 默认 |
| 06.5 | 梯度流经三个投影；零输入 → 输出 0（SiLU(0)=0 非线性） |

## 5. 参考答案

`answers/minimind3/feed_forward.py`（28 行；先写后对）。

## 6. 常见坑

- **gate/up 交换**：形状仍对、数值全错（06.2 抓）；
- **加 bias**：形状对但第 13 课 strict 加载会多键报错；
- **`ACT2FN` 拼错 key**：`silu` 已在 config 固定，直接索引即可。

## 7. 对照标准实现

`answers/minimind3/feed_forward.py` 与 minimind `FeedForward` 等价；minimind 的 `MLPMoe`（MoE 版）列进阶（config 字段已预留，`aux_loss=0`）。

## 8. 小结

✅ 加工车间完成（SwiGLU 门控、无 bias、π 对齐中间维）。
**下一课**：把注意力 + 前馈 + 两个 Norm 装进残差壳——`MiniMindBlock`。