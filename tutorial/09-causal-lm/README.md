# 第 09 课 · MiniMindForCausalLM：输出头、损失与生成

## 目标

手写 `minimind3/causal_lm.py` 的 `MiniMindForCausalLM`（继承 `PreTrainedModel, GenerationMixin`）：
lm_head 投影、权重绑定、next-token 损失、自实现采样生成循环。

## 理论

### 1. 从 hidden → logits → 损失

- `lm_head`：`hidden_size → vocab_size` 线性层，输出每个位置词表上的 logits；
- next-token 目标：位置 $t$ 的 logits 预测位置 $t+1$ 的 token
  → 实现为"错位对齐"：`x = logits[..., :-1, :]`，`y = labels[..., 1:]`；
- `ignore_index=-100`（HF 约定）：标签中 -100 的位置不计入损失（SFT 掩码的基础）。

### 2. 权重绑定（Weight Tying）

`lm_head.weight` 与 `embed_tokens.weight` 为**同一个 Parameter 对象**：
- 省 `vocab × hidden` 参数（以默认配置计约 490 万）；
- `_tied_weights_keys` 告诉 HF 在 `from_pretrained` 时两键按同一权重处理，
  避免保存两份。

### 3. 生成循环（自实现，不依赖 HF generate）

每一步：取 `logits[:, -1]`，按顺序做后处理再采样——

| 超参 | 作用 |
|---|---|
| temperature | logits 除以 T（>1 更随机，<1 更确定） |
| top_k | 只保留分数最高的 k 个候选 |
| top_p (nucleus) | 按概率累计到 p 截断 |
| repetition_penalty | 对已出现过的 token 的分数罚分 |
| do_sample / eos | 采样 or argmax；命中 eos 即停（可多序列并行） |

KV-Cache 下每步只 forward 新 token，`past_len` 记录已处理位置，RoPE 自动续位。

### 4. PreTrainedModel 集成

继承后自动获得 `save_pretrained / from_pretrained / AutoConfig` 关联等能力：
本课验收 09.8 做 save→load→前向一致的 round-trip；AutoClass 注册留到第 13 课。

## 任务清单（先手写再看答案）

1. `__init__`：`config_class`、`_tied_weights_keys`、lm_head（无 bias）、共享 embed 权重、`post_init()`；
2. `forward`：model → lm_head →（`logits_to_keep` 切片）→ 错位 CE（ignore_index=-100）；
3. `generate`：温度 → 重复惩罚 → top-k → top-p →（multinomial/argmax）→ eos 终止 → 拼接返回。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：形状/绑定、**损失数值与手工 CE 一致**、-100 忽略语义、**极小模型可学习**、
贪心确定性、eos 提前停、采样超参、**KV-Cache 开关结果一致**、HF round-trip。

## 对照标准实现

`master:model/model_minimind.py → MiniMindForCausalLM`（含 generate 同名参数）。

## 常见坑

- 绑定要在 `post_init()` **之前**建立（`post_init` 会处理 tied 键的去重初始化）；
- `logits[..., :-1]` 与 `labels[..., 1:]` 必须同时错位，否则序列少一位信号；
- `top_p` 实现里 `mask[..., 1:] = mask[..., :-1].clone()` 的移位（前移一位）容易写错；
- `torch.multinomial` 的输入要 softmax 成概率；
- 生成循环里 `input_ids[:, past_len:]` 每次只喂增量，靠 KV-Cache 续上下文。