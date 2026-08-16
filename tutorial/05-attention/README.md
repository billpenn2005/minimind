# 第 05 课 · GQA 注意力 Attention

> [上一课 04-rope](../04-rope/README.md) ← [目录](../../README.md) → [下一课 06-feedforward](../06-feedforward/README.md)
>
> 难度：★★★☆☆（核心课）｜ 预计用时 2~3 小时 ｜ 前置：03+04 课

## 0. 本课目标

- [ ] 理解缩放点积注意力、多头、因果掩码、KV-Cache、GQA、QK-Norm；
- [ ] 手写 `minimind3/attention.py`：`repeat_kv` + `Attention`；
- [ ] 验收 28 项累计（本课新增 05.1~05.6）。

## 1. 理论速览

- **缩放点积注意力**：$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$；除 $\sqrt{d_k}$ 防点积随维度过大导致 softmax 饱和。
- **多头**：把 hidden 切成 H 个头的"视角"，各自算注意力再拼回（`(B,S,H,d)`）。
- **因果掩码**：训练时 token 只能看自己及以前；`scores[..., -seq_len:]` 上三角填 `-inf`（只遮"新列"，留历史）。
- **KV-Cache**：生成时每步只算新 token 的 Q，与**缓存的历史 K、V** 拼接后打分——复杂度从 O(全量) 降到 O(新)；**只缓存 K、V，Q 不缓存**。
- **GQA**：`num_key_value_heads < num_attention_heads`，共享 KV 头（`repeat_kv` 展开）——KV 缓存内存减半起。
- **QK-Norm**：对每头的 q/k 向量做 RMSNorm（对数值稳定性、低精度训练友好）。
- **SDPA 快路径**：满足条件走 `F.scaled_dot_product_attention`（融合内核）；否则走手写分支（本教程保留作为"活参考"）。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/attention.py`

**模块级 import**：`math`、`torch`、`torch.nn.functional as F`、`from torch import nn`、`from .config import MiniMindConfig`、`from .rms_norm import RMSNorm`、`from .rope import apply_rotary_pos_emb`。

### 2.2 函数 `repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor`

| 参数 | 形状 | 说明 |
|---|---|---|
| `x` | `(B, S, kv_heads, head_dim)` | KV 状态 |
| `n_rep` | `int` | 每 KV 头复制数（= heads//kv_heads） |

返回 `(B, S, kv_heads*n_rep, head_dim)`。**实现必须**：`n_rep==1` 直接返回；否则
`x[:, :, :, None, :].expand(B, S, kv, n_rep, d).reshape(B, S, kv*n_rep, d)`——顺序是 `[kv0,kv0,…,kv1,kv1,…]`（用 `expand+reshape`，**不是** `x.repeat`）。

### 2.3 `class Attention(nn.Module)`

**`__init__(self, config: MiniMindConfig)`** 设置的实例属性：

| 属性 | 类型 | 值 |
|---|---|---|
| `num_key_value_heads` | `int` | `config.num_key_value_heads` |
| `n_local_heads` | `int` | `config.num_attention_heads` |
| `n_local_kv_heads` | `int` | `config.num_key_value_heads` |
| `n_rep` | `int` | `n_local_heads // n_local_kv_heads` |
| `head_dim` | `int` | `config.head_dim` |
| `is_causal` | `bool` | `True` |
| `q_proj` | `nn.Linear` | `(hidden → heads*head_dim, bias=False)` |
| `k_proj` | `nn.Linear` | `(hidden → kv_heads*head_dim, bias=False)` |
| `v_proj` | `nn.Linear` | `(hidden → kv_heads*head_dim, bias=False)` |
| `o_proj` | `nn.Linear` | `(heads*head_dim → hidden, bias=False)` |
| `q_norm` | `RMSNorm` | `RMSNorm(head_dim, eps=config.rms_norm_eps)` |
| `k_norm` | `RMSNorm` | 同上 |
| `attn_dropout` | `nn.Dropout` | `config.dropout` |
| `resid_dropout` | `nn.Dropout` | `config.dropout` |
| `dropout` | `float` | `config.dropout` |
| `flash` | `bool` | `hasattr(F,"scaled_dot_product_attention") and config.flash_attn` |

**`forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None)`**：

| 参数 | 类型 | 形状 | 默认 |
|---|---|---|---|
| `x` | `torch.Tensor` | `(B, S, hidden)` | — |
| `position_embeddings` | `tuple` | `(cos, sin)` 各 `(S, head_dim)` | — |
| `past_key_value` | `tuple[torch.Tensor, torch.Tensor] \| None` | 各 `(B, prev_S, kv_heads, head_dim)` | `None` |
| `use_cache` | `bool` | — | `False` |
| `attention_mask` | `torch.Tensor \| None` | `(B, S)` 或 `(B, 1, S, S)`，1=有效 0=pad | `None` |

返回：`tuple`：`(output, past_kv)`
- `output`：`torch.Tensor (B, S, hidden)`；
- `past_kv`：`(xk, xv)` 形状 `(B, prev_S+len, kv_heads, head_dim)`；`use_cache=False` 时 `None`。

**内部步骤（顺序敏感，勿改）**：
1. `xq,xk,xv = q_proj(x), k_proj(x), v_proj(x)` → `view(B,S,heads/kv_heads,head_dim)`；
2. **QK-Norm 在拆头之后、RoPE 之前**：`xq,xk = q_norm(xq), k_norm(xk)`；
3. `apply_rotary_pos_emb`（**KV 拼接发生在旋转之后**：历史 K 已旋转，新 K 旋转后直接拼）；
4. `past_kv = (xk,xv) if use_cache else None`；
5. `xq.transpose(1,2)`；`xk = repeat_kv(xk, n_rep).transpose(1,2)`；xv 同理；
6. **快路径**（`flash and S>1 and (不 causal 或无 past) and (mask 是 None 或全 1)`）：`F.scaled_dot_product_attention(xq,xk,xv, dropout_p=dropout if training else 0.0, is_causal=True)`；
   否则手写：`scores = xq @ xk.transpose(-2,-1) / sqrt(head_dim)`；因果时 `scores[:,:,:,-seq_len:] += full((seq_len,seq_len),-inf).triu(1)`；mask 时 `scores += (1-mask.unsqueeze(1).unsqueeze(2)) * -1e9`；`attn_dropout(softmax(scores.float()).type_as(xq)) @ xv`；
7. `output.transpose(1,2).reshape(B,S,-1)` → `resid_dropout(o_proj(output))`。

## 3. 手写步骤

按 §2.2/§2.3 逐项实现（顺序最关键：拆头→QK-Norm→RoPE→拼 KV→展开→打分）→ `verify.py 05`。

## 4. 验收解读（verify/05_attention.py）

| 检查（新） | 验什么 |
|---|---|
| 05.1 | 投影形状；`repeat_kv` 展开顺序/形状正确 |
| 05.2 | **因果性（autograd 证明）**：token0 输出对位置 j>0 输入的梯度为 0 |
| 05.3 | 与手写参考注意力（手动旋转/repeat/掩码）输出一致（1e-4） |
| 05.4 | **KV 增量 == 全量**：分块喂入与整段喂入输出一致（1e-4） |
| 05.5 | pad 行不产生 NaN；有效位置与无掩码运行一致 |
| 05.6 | SDPA 快路径 == 手写慢路径（两个 flash 开关相反的实例，1e-4） |

## 5. 参考答案

`answers/minimind3/attention.py`（先写后对；重点核对 `repeat_kv` 表述与快路径条件写法）。

## 6. 常见坑

- **`repeat` vs `expand+reshape`**：前者会生成 `[kv0,kv1,kv0,kv1,…]` 交错序（错），必须 expand+reshape 得 `[kv0,kv0,…]`；
- **QK-Norm 放错位**：在拆头前做会作用于整行 hidden（shape 错），必须在拆头后、RoPE 前；
- **因果掩码全遮**：`scores` 加了 `-inf` 的 triu 是对整列做的，必须只对 `[:,:,:,-seq_len:]` 新列追加；
- **KV 拼接在旋转前**：历史 K 已旋转、新 K 未旋转 → 位置错乱（05.4 抓）；
- **SDPA 快路径误入**：有 past 缓存时仍走 is_causal=True 全量遮罩 → 增量结果错。

## 7. 对照标准实现

`answers/minimind3/attention.py` 与 minimind `Attention` 逐块等价（含 kv cache 约定、快路径条件）；教学保留手写慢分支作对照，minimind 只留 SDPA 分支。

## 8. 小结

✅ 注意力的每个要点（缩放/GQA/因果/KV/QK-Norm/SDPA）均已数值验证。
**下一课**：SwiGLU 前馈——信息处理的"加工车间"。