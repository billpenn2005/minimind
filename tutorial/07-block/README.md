# 第 07 课 · Pre-Norm 残差块 MiniMindBlock

> [上一课 06-feedforward](../06-feedforward/README.md) ← [目录](../../README.md) → [下一课 08-model](../08-model/README.md)
>
> 难度：★★☆☆☆ ｜ 预计用时 1 小时 ｜ 前置：05+06 课

## 0. 本课目标

- [ ] 理解残差连接（梯度高速公路）与 Pre-Norm vs Post-Norm；
- [ ] 手写 `minimind3/block.py` 的 `MiniMindBlock`（含 KV 透传约定）；
- [ ] 验收 38 项累计（本课新增 07.1~07.5）。

## 1. 理论速览

- **深层诅咒**：梯度逐层连乘 $\frac{\partial L}{\partial x_0} = \prod_l \frac{\partial x_{l+1}}{\partial x_l}$，深层梯度消失/爆炸。
- **残差**：$x_{l+1} = x_l + f(x_l)$ → 偏导 $\frac{\partial x_{l+1}}{\partial x_l} = 1 + \frac{\partial f}{\partial x_l}$——那个 `+1` 是"高速公路"，即使 $\partial f$ 极小，梯度也能直通底层。
- **Pre-Norm vs Post-Norm**：Post（2017 原始）先算子层再归一化，训练挑剔；Pre（当代标配）入口归一化 `x + Attn(Norm(x))`，天然稳定，ResNet/LLM 同源。
- **双子结构**：注意力与 MLP 各配一个 Norm，各配一条残差。
- **命名约定**：`input_layernorm` / `post_attention_layernorm` 是 transformers/Llama/Qwen 生态通用命名——第 13 课权重转换靠 state_dict 键名对上。
- **KV 透传**：块接收 `past_key_value`、返回 `present_kv`（`use_cache=False` 时 None），把多块缓存编排留给主体（第 08 课）。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/block.py`

**模块级 import**：`from torch import nn`；`from .attention import Attention`、`from .config import MiniMindConfig`、`from .feed_forward import FeedForward`、`from .rms_norm import RMSNorm`。

### 2.2 `class MiniMindBlock(nn.Module)`

**`__init__(self, layer_id: int, config: MiniMindConfig)`**：

| 属性 | 类型 | 值 |
|---|---|---|
| `layer_id` | `int` | 构造参数 1（调试/未来 MoE 用） |
| `self_attn` | `Attention` | `Attention(config)` |
| `input_layernorm` | `RMSNorm` | `RMSNorm(config.hidden_size, eps=config.rms_norm_eps)` |
| `post_attention_layernorm` | `RMSNorm` | 同上 |
| `mlp` | `FeedForward` | `FeedForward(config)` |

**`forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None) -> tuple`**：

| 参数 | 类型 | 默认 |
|---|---|---|
| `hidden_states` | `torch.Tensor (B,S,hidden)` | — |
| `position_embeddings` | `tuple` | — |
| `past_key_value` | `tuple \| None` | `None` |
| `use_cache` | `bool` | `False` |
| `attention_mask` | `torch.Tensor \| None` | `None` |

返回 `(hidden_states, present_key_value)`：前者 `(B,S,hidden)`，后者为 K/V 元组或 `None`。

**主体四步（顺序固定）**：
1. `residual = hidden_states`；
2. `hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask)`；
3. `hidden_states = hidden_states + residual`；
4. `hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))`；return 二者。

## 3. 手写步骤

实现 §2.2 → `.venv/Scripts/python.exe verify.py 07`。

## 4. 验收解读（verify/07_block.py）

| 检查（新） | 验什么 |
|---|---|
| 07.1 | 输出形状、子模块命名（`self_attn/input_layernorm/post_attention_layernorm/mlp`）——13 课转换的键名保证 |
| 07.2 | **残差恒等**：全部参数清零 → `block(x) == x`（1e-6）：f(x)=0 只剩直通，最优雅的残差证明 |
| 07.3 | 手写展开 `x + attn(ln1(x)) + mlp(ln2(residual))` 与块输出一致（1e-5） |
| 07.4 | 梯度同时流经注意力和 MLP 两个分支 |
| 07.5 | `use_cache=False` 返回的 `present_kv is None`；反之有值 |

## 5. 参考答案

`answers/minimind3/block.py`（先写后对；命名必须逐字符一致）。

## 6. 常见坑

- **Post-Norm 写反**：07.3 的展开式不匹配；
- **忘残差 `+`**：07.2（清参恒等）立刻红叉；
- **忘传 `layer_id`**：behavior 可能仍过，但 13 课逐层转换需要它；
- **返回顺序反**：`(hidden, kv)` 别写成 `(kv, hidden)`。

## 7. 对照标准实现

`answers/minimind3/block.py` 与 minimind `MiniMindBlock` 等价；minimind 的 MoE 分支以 `mlp.aux_loss` 字段形式预留（第 08 课汇总）。

## 8. 小结

✅ 残差壳完成：Pre-Norm、双子结构、生态命名、KV 透传。
**下一课**：把 N 个块 + Embedding + 终归一化 + RoPE 表装成完整主体 `MiniMindModel`。