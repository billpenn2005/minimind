# 第 08 课 · 模型主体 MiniMindModel

> [上一课 07-block](../07-block/README.md) ← [目录](../../README.md) → [下一课 09-causal-lm](../09-causal-lm/README.md)
>
> 难度：★★★☆☆ ｜ 预计用时 2 小时 ｜ 前置：07 课

## 0. 本课目标

- [ ] 理解 Embedding 查找表、`Parameter` vs buffer（`persistent=False`）；
- [ ] 手写 `minimind3/model_body.py` 的 `MiniMindModel`（KV 编排层）；
- [ ] 验收 43 项累计（本课新增 08.1~08.5）。

## 1. 理论速览

- **Embedding**：`nn.Embedding(vocab, hidden)` 查表 `(B,S,hidden)`；表本身可学习（第 09 课与 lm_head 绑定）。
- **主体骨架**：`embed → dropout → N×MiniMindBlock → norm`；RoPE 的 cos/sin 表注册为 buffer。
- **Parameter vs buffer vs persistent=False**：

| 类别 | 被优化器更新 | 进 state_dict |
|---|---|---|
| `nn.Parameter`（W_q 等） | ✅ | ✅ |
| 普通 buffer（BN 均值） | ❌ | ✅ |
| **`persistent=False` buffer（RoPE 表）** | ❌ | **❌ 不保存** |

  RoPE 表是"由 config 恒定的常量"：`[32768, head_dim]` float32 ≈ 100MB 大表，保存纯浪费；每次构造现场重算。（`meta` 设备上防御性重算也依赖此性质。）
- **KV 编排**：`start_pos` 从首层缓存的长度推出；新 token 用 `freqs[start_pos : start_pos+seq_len]` 切片（位置编号连续 = 缓存与 RoPE 的衔接点）；逐层透传 `past_key_value`，把每层 `present_kv` 收进 `presents`。
- **`aux_loss`**：MoE 负载平衡损失汇总位；Dense 恒 0（`new_zeros(1).squeeze()` 起加），字段保留对齐接口。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/model_body.py`

**模块级 import**：`torch`、`from torch import nn`、`from .attention import Attention`、`from .block import MiniMindBlock`、`from .config import MiniMindConfig`、`from .feed_forward import FeedForward`、`from .rms_norm import RMSNorm`、`from .rope import precompute_freqs_cis`。

### 2.2 `class MiniMindModel(nn.Module)`

**`__init__(self, config: MiniMindConfig)`**：

| 属性 | 类型 | 值 |
|---|---|---|
| `config` | `MiniMindConfig` | 构造参数 |
| `vocab_size` | `int` | `config.vocab_size` |
| `num_hidden_layers` | `int` | `config.num_hidden_layers` |
| `embed_tokens` | `nn.Embedding` | `(vocab, hidden)` |
| `dropout` | `nn.Dropout` | `config.dropout` |
| `layers` | `nn.ModuleList` | `[MiniMindBlock(l, config) for l in range(num_hidden_layers)]` |
| `norm` | `RMSNorm` | `RMSNorm(hidden, eps=config.rms_norm_eps)` |
| `freqs_cos` | `torch.Tensor`（buffer） | `precompute_freqs_cis(dim=head_dim, end=max_position_embeddings, rope_base=rope_theta, rope_scaling=config.rope_scaling)` 的 cos，**`register_buffer(..., persistent=False)`** |
| `freqs_sin` | 同上 | sin 同款 |

**`forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs) -> tuple`**：

| 参数 | 类型 | 默认 |
|---|---|---|
| `input_ids` | `torch.Tensor` `(B, S)` int64 | — |
| `attention_mask` | `torch.Tensor \| None` | `None` |
| `past_key_values` | `list \| None`（每层一项：K/V 元组或 None） | `None` |
| `use_cache` | `bool` | `False` |
| `**kwargs` | — | 容错（供 HF/上层透传） |

返回 `(hidden_states, presents, aux_loss)`：
- `hidden_states`：`(B, S, hidden)` float32；
- `presents`：`list`，长度 = 层数，每项为 `(xk, xv)` 或 `None`（依据 use_cache）；
- `aux_loss`：标量 `torch.Tensor`（0.0 或 MoE 汇总）。

**内部顺序**：
1. 兼容 `Cache` 对象（`hasattr(past_key_values, "layers")` 则置 None）；`past_key_values = past_key_values or [None]*N`；
2. `start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0`；
3. `hidden = dropout(embed_tokens(input_ids))`；
4. meta 防御：`freqs_cos[0,0]==0` 时重算表并 `.to(hidden.device)`；
5. `position_embeddings = (freqs_cos[start_pos:start_pos+seq], freqs_sin[…])`；
6. 逐层 `layer(hidden, position_embeddings, past_key_value=…, use_cache=…, attention_mask=…)`，收集 `presents`；
7. `hidden = norm(hidden)`；
8. `aux_loss = sum(层.mlp.aux_loss 若存在 else 0)`（以 `hidden_states.new_zeros(1).squeeze()` 起加）。

## 3. 手写步骤

实现 §2.2 → `.venv/Scripts/python.exe verify.py 08`。

## 4. 验收解读（verify/08_model.py）

| 检查（新） | 验什么 |
|---|---|
| 08.1 | `(B,S,hidden)` 形状、float32 |
| 08.2 | `freqs_cos/sin` 不出现在 `state_dict()`（persistent=False 生效） |
| 08.3 | **模型级 KV 增量 == 全量**（1+2+3 分段 vs 一次喂完，1e-4）——编排正确 |
| 08.4 | 每层 `present` 形状 `(B, kv_heads, S, head_dim)`；层数一致 |
| 08.5 | 单次 forward 后所有参数都有梯度（链路连通） |

## 5. 参考答案

`answers/minimind3/model_body.py`（先写后对；重点核对 buffer 注册与 start_pos 派生）。

## 6. 常见坑

- **buffer 忘 `persistent=False`**：state_dict 凭空多巨表（08.2 抓）；
- **`start_pos` 取错**：应从首层 past 的序列维（`past[0][0].shape[1]`，第 2 维）取；
- **RoPE 切片从头切**：要用 `start_pos:start_pos+seq` —— 增量位置必须编号连续；
- **meta 防御缺失**：`from_pretrained` 流程下建 buffer 报 "Cannot copy out of meta tensor"；
- **用 `nn.Sequential` 装层**：拿不到逐层 `presents`。

## 7. 对照标准实现

`answers/minimind3/model_body.py` 与 minimind `MiniMindModel` 对齐（含 `MoeModelOutput` 返回结构；教学简化：直接返回三元组，靠 causal_lm 包装成 output 对象——见下节课）。

## 8. 小结

✅ 主体完成：embed → 块堆叠 → norm，缓存编排 + 非持久 buffer + meta 防御。
**下一课（大一课）**：`MiniMindForCausalLM`——lm_head、权重绑定、shift 损失、你自己的 generate。