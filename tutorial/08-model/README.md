# 第 08 课 · MiniMindModel：组装完整的解码器主体

## 目标

手写 `minimind3/model_body.py` 的 `MiniMindModel`：Embedding + N 个 Block + 最终 Norm
+ RoPE 表与 KV-Cache 编排。

## 理论

### 1. Embedding 与权重绑定

`nn.Embedding(vocab, hidden)` 查表得 `[B, S, hidden]`。
词嵌入维度与隐藏维度一致时，可以让 **lm_head（下一步）与 embed 共享权重**
（`tie_word_embeddings=True`）：省 `vocab×hidden` 参数，且"同一个词的表示"
无论作为输入还是输出都一致（有利迁移）。

### 2. buffer 与 parameter

- `parameter`：可学习、进 optimizer；
- `buffer`：不可学习的张量，可随 `state_dict()` 保存/加载；
- RoPE 表属于 buffer，但 `persistent=False` 使其**不进 checkpoint**（生成时即时计算或
  容量大），minimind 的选择就是不保存（省空间）——verify 08.2 检查这一点。

### 3. 主体模型的前向编排

```
tokens ──> Embed ──> Dropout ──> [Block × N] ──> Norm ──> hidden_states
                                  (每层透传 KV-Cache & 转发表位置)
```

- `start_pos`：从已有 KV 长度推当前 token 的绝对位置，RoPE 表按 `[start : start+len]` 切片；
- 输出三件套 `(hidden_states, presents, aux_loss)`——
  对齐 minimind，`aux_loss` 为 MoE 预留（非 MoE 恒 0）。

### 4. 为什么最终归一化在主体里

Pre-Norm 结构下，最后一层输出未经归一化尺度不稳定；在主体末尾统一过一个
RMSNorm 再给 lm_head，是 LLaMA/minimind 的惯例。

## 任务清单（先手写再看答案）

1. `__init__`：embed、dropout、`ModuleList` 层、最终 RMSNorm、注册 RoPE 表；
2. `forward`：start_pos → embed → 逐层（透传 past/use_cache/attention_mask）→ norm；
3. 返回 `(hidden_states, presents, aux_loss)`。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：前向形状、**buffer 非持久**、**模型级 KV-Cache 增量==全量**、past 结构、全参数梯度。

## 对照标准实现

`master:model/model_minimind.py → MiniMindModel`。

## 常见坑

- `register_buffer` 不写 `persistent=False` 会把几十 MB 的 RoPE 表存进每个 checkpoint；
- `past_key_values` 为 None 时也要生成 `[None] * num_layers`，逐层 zip 才不炸；
- 位置切片 `[start_pos : start_pos + seq_length]` 写错（如 `end` 参数）会让增量续写位置错乱；
- `Batch x S` 的输入直接用 `torch.randint(0, vocab, (B, S))` 构造即可，不用 one-hot。