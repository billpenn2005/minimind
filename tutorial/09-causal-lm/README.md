# 第 09 课 · 因果语言模型 MiniMindForCausalLM

> [上一课 08-model](../08-model/README.md) ← [目录](../../README.md) → [下一课 10-data](../10-data/README.md)
>
> 难度：★★★☆☆（全程最重：8 项验收）｜ 预计用时 3~4 小时 ｜ 前置：08 课

## 0. 本课目标

- [ ] 理解 lm_head、**权重绑定**、**shift 损失**、generate 采样策略全家桶；
- [ ] 手写 `minimind3/causal_lm.py`；
- [ ] 验收 51 项累计（本课新增 09.1~09.8）。

## 1. 理论速览

- **lm_head**：`logits = h·W_lmᵀ`（`[B,S,vocab]`）→ softmax → 下个 token 概率；`W_lm` 与 embed 表同构。
- **权重绑定**：`model.embed_tokens.weight = model.lm_head.weight`（同一 Parameter）——省 `vocab×hidden`（默认 6400×768≈4.9M，约占总参 8%），效果更好。声明 `_tied_weights_keys` 让 HF 保存/加载自动去重。**必须在 `post_init()` 之前绑定**（否则 post_init 会重新初始化 lm_head 破坏同一性）。
- **shift 损失**：`logits[..., :-1]` 预测 `labels[..., 1:]`，交叉熵 `ignore_index=-100`（padding/非监督位置的隐身标签，第 10 课制造）。
- **继承 `PreTrainedModel + GenerationMixin`**：白送 `save_pretrained/from_pretrained`，并注册为可用的 transformers 模型类（`config_class`）。
- **generate 策略**：贪心（argmax）｜温度 `softmax(logits/T)`（T→0 收敛贪心）｜top-k（保前 k 大）｜top-p（按概率累计到 p 的动态裁剪）｜重复惩罚（出现过的 token 分数 除以/乘 repetition_penalty）｜eos 早停｜KV 增量（每轮只喂 `input_ids[:, past_len:]`）。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/causal_lm.py`

**模块级 import**：`torch`、`torch.nn.functional as F`、`from torch import nn`、`from transformers import GenerationMixin, PreTrainedModel`、`from transformers.modeling_outputs import MoeCausalLMOutputWithPast`、`from .config import MiniMindConfig`、`from .model_body import MiniMindModel`。

### 2.2 `class MiniMindForCausalLM(PreTrainedModel, GenerationMixin)`

| 类属性 | 类型 | 值 |
|---|---|---|
| `config_class` | type | `MiniMindConfig` |
| `_tied_weights_keys` | `set[str]` | `{"lm_head.weight": "model.embed_tokens.weight"}` |

**`__init__(self, config: MiniMindConfig | None = None)`**：
1. `self.config = config or MiniMindConfig()`；`super().__init__(self.config)`；
2. `self.model = MiniMindModel(self.config)`；
3. `self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)`；
4. 若 `config.tie_word_embeddings`：`self.model.embed_tokens.weight = self.lm_head.weight`（**此刻、post_init 之前**）；
5. `self.post_init()`。

**`forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs)`**：

| 参数 | 类型 | 默认 |
|---|---|---|
| `input_ids` | `Tensor (B,S) \| None` | `None` |
| `attention_mask` | `Tensor \| None` | `None` |
| `past_key_values` | `list \| None` | `None` |
| `use_cache` | `bool` | `False` |
| `logits_to_keep` | `int` | `0`（0=保留全部位置） |
| `labels` | `Tensor (B,S) int64 \| None` | `None`（labels 含 -100） |
| `**kwargs` | — | 透传 model |

返回：`MoeCausalLMOutputWithPast`，字段：
- `loss: Tensor \| None`（labels 非 None 时）：`F.cross_entropy(shift_logits.view(-1,vocab), shift_labels.view(-1), ignore_index=-100)`；
- `logits: Tensor (B, S_or_keep, vocab)`；`past_key_values`（presents）；`hidden_states`；`aux_loss: Tensor`。

**内部**：
1. `hidden, past_kv, aux = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)`；
2. `slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep`；`logits = lm_head(hidden[:, slice_indices, :])`；
3. 损失：`x = logits[..., :-1, :].contiguous()`；`y = labels[..., 1:].contiguous()`；CE（**shift 是"logits 去尾、labels 去头"，必须 contiguous**）。

### 2.3 `generate`（自实现，装饰 `@torch.inference_mode()`）

**签名**：

```python
def generate(self, inputs=None, attention_mask=None, max_new_tokens=1024, temperature=0.85,
             top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
             num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs)
```
- `inputs`：`Tensor (B,S) int64`（或经 `kwargs["input_ids"]` 传入）；返回：`Tensor (B, S+new)` int64（`kwargs["return_kv"]=True` 时返回 `{"generated_ids":…, "past_kv":…}` 字典）。

**行为要求（顺序固定）**：
1. `input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)`；mask 同步 repeat；`finished = zeros(B, bool)`；
2. 循环 ≤ max_new_tokens 次：
   a. `past_len = past_key_values[0][0].shape[1] if past_key_values else 0`；
   b. 前向仅喂 `input_ids[:, past_len:]`（增量！）；mask 尾部补 1；
   c. `logits = outputs.logits[:, -1, :] / temperature`；
   d. 重复惩罚（`repetition_penalty != 1.0` 时，对每行已见 token：`score>0 → score/rp`，否则 `score*rp`）；
   e. top-k（`top_k>0`：低于第 k 大的置 -inf）；top-p（`top_p<1.0`：排序累计概率 >p 的置 -inf，**`mask[..., 1:] = mask[..., :-1].clone()` 移位、最低位归 0**）；
   f. `next_token`：`do_sample` → `multinomial(softmax(logits),1)`，否则 `argmax`；`eos_token_id` 下已 finished 行强制 eos；
   g. 拼接、更新 past、`finished |= (next_token==eos)`；全 finished 提前 break；streamer.put；
3. `streamer.end()`（若提供）；按 `return_kv` 返回。

## 3. 手写步骤

按 §2.2→§2.3 实现 → `.venv/Scripts/python.exe verify.py 09`。

## 4. 验收解读（verify/09_causal_lm.py）

| 检查（新） | 验什么 |
|---|---|
| 09.1 | logits 形状；**`lm_head.weight is model.embed_tokens.weight`（同一对象）**；`h@Wᵀ` 一致 |
| 09.2 | 手动 shift CE == 模型 loss（1e-6）；-100 被忽略 |
| 09.3 | 12 步 AdamW 在固定数据上 loss < 初值 95%（真能学） |
| 09.4 | 贪心生成长度/确定性 |
| 09.5 | eos 早停 |
| 09.6 | temperature/top_k/top_p/repetition_penalty 跑通且改变结果 |
| 09.7 | **KV开 == KV关 生成结果一致**（缓存不改变输出） |
| 09.8 | `save_pretrained → from_pretrained` 往返 logits 一致（1e-5） |

## 5. 参考答案

`answers/minimind3/causal_lm.py`（先写后对；重点核对绑定时机、top-p 移位、incremental 喂入）。

## 6. 常见坑

- **绑定在 post_init 后**：09.1 的 `is` 断言抓；
- **shift 忘 contiguous**：view 报错或静默错位；
- **top-p mask 不移位/不 clone**：`mask[...,1:] = mask[...,:-1].clone()` 右值必须独立拷贝；
- **温度后直接采样忘 softmax**：multinomial 要求非负和=1；
- **增量生成又喂全量**：结果对、速度差——增量必须 `input_ids[:, past_len:]`；
- **绑定时两个 weight 形状不同**：`nn.Linear(hidden, vocab)` 权重形状是 `(vocab, hidden)`，`nn.Embedding(vocab, hidden)` 是 `(vocab, hidden)`——一致才能绑定。

## 7. 对照标准实现

`answers/minimind3/causal_lm.py` 对齐 minimind `MiniMindForCausalLM`（绑定、`_tied_weights_keys`、logits_to_keep、generate 参数全集）；教学另留 `generate_transformer`（调父类 GenerationMixin 对照）与更完整单测。

## 8. 小结（里程碑 🎉）

✅ 模型核心全部完成：config→RMSNorm→RoPE→Attention→FFN→Block→Model→ForCausalLM（损失+生成+保存/加载）。
**下一课**：文本 → id——接入分词器、ChatML 模板与 -100 掩码标签。