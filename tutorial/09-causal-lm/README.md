# 第 09 课 · 因果语言模型 MiniMindForCausalLM（lm_head + 损失 + generate）

> 难度：★★★☆☆（全程最重的一课：8 项验收）｜ 预计用时：3~4 小时 ｜ 前置：08 课模型主体
> 学完本课你应能回答：**hidden 怎么变成词表概率？权重绑定省多少参数？shift 损失错位在哪？generate 的采样策略各在干什么？**

---

## 0. 本课目标

- [ ] 理解 lm_head（输出头）与**权重绑定**（tie word embeddings）；
- [ ] 理解 next-token 损失的**错位（shift）**与 `ignore_index=-100`；
- [ ] 继承 `PreTrainedModel + GenerationMixin`，实现保存/加载；
- [ ] 手写带温度 / top-k / top-p / 重复惩罚 / KV 增量 / 早停的 `generate`；
- [ ] 验收 8 项新检查（09.1~09.8，累计 50 项全过）。

---

## 1. 理论

### 1.1 从语义屋到词表概率：lm_head

主体模型输出每个 token 的 768 维"语义向量"。要变成"下一个词是哪个"的概率，需要最后一跳：

$$
\text{logits} = h \cdot W_{lm}^\top \in \mathbb{R}^{\text{vocab}}
\xrightarrow{\text{softmax}}
P(\text{next token})
$$

`W_lm` 形状 `(vocab, hidden)`，叫 **lm_head（语言模型头）**。它与输入侧的另一张表长得一模一样——
这就引出了权重绑定。

### 1.2 权重绑定：一张表，两处用

输入侧需要"id → 向量"的 embed 表（`[vocab, hidden]`）；输出侧需要"hidden → 分数"的 lm_head
（`[vocab, hidden]`，转置后相乘）。**语义上它们是同一回事**（同一个词就是同一个向量），于是：

```python
model.embed_tokens.weight = model.lm_head.weight   # 共享同一个 Parameter！
```

- 参数量立省 `vocab × hidden`：默认配置 6400×768 ≈ **4.9M**（约总参数量 64M 的 8%）；
- 效果还更好（embed 与输出分布自然对齐，经典研究也支持）；
- 兑现方式（与 minimind 相同）：在 `post_init()` **之前**把两个 name 指向同一个 Parameter 对象，
  并声明 `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}` 让 transformers
  保存/加载时自动处理绑定关系。

### 1.3 训练目标：下一个 token 的错位损失

一个句子 `[x1, x2, x3, x4]`，模型在每个位置预测"下一个"：位置 1 的标签是 x2，位置 2 的标签是 x3……
所以**模型输出切掉最后一格**（`logits[..., :-1]`），**标签切掉第一格**（`labels[..., 1:]`）：

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), ignore_index=-100)
```

**`ignore_index=-100`** 是数据侧送来的约定（10 课详述）：标签值为 -100 的位置不参与损失——
padding、user 提问等"不该学"的 token 就靠这个数字隐身。cross_entropy 遇到 -100 直接跳过该行。

> 为什么叫"因果"？生成方向单向（只看过去），训练目标也是单向接龙——这就是 causal LM。

### 1.4 继承 PreTrainedModel + GenerationMixin：生态级能力

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig                      # 让 AutoConfig 知道
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

- `PreTrainedModel` 白送：`save_pretrained(dir)` / `from_pretrained(dir)` / `to()` / `train()`……（13 课深度使用）；
- `GenerationMixin` 白送：`model.generate(...)` 官方生成接口（本教程也自己实现一套"纯手写版"对照理解）。

### 1.5 手写 generate：采样策略全家桶

生成 = 循环"预测分布 → 选 token → 拼上 → 再预测"。质量由"怎么选"决定：

| 策略 | 规则 | 直觉 |
|---|---|---|
| **贪心（greedy）** | 每步取 argmax | 最稳定但容易复读/单调 |
| **温度 temperature** | `logits / T` 后再 softmax；T→0 接近贪心，T→1 原样，T>1 更"乱" | 调节赌注的胆量 |
| **top-k** | 只保留概率最高的 k 个，其余 -inf | 砍掉"明显离谱"的尾巴 |
| **top-p** | 按概率降序累加，直到累计 ≥ p 的集合保留 | 动态 k：分布集中时留少，平坦时留多 |
| **重复惩罚** | 对已生成过的 token 分数除以 `repetition_penalty`（>1 抑制） | 治复读机 |

实现的三个易错点：
1. **top-p 移位**：mask 需要 `mask[..., 1:] = mask[..., :-1].clone()`——因为 softmax 已经"吃掉"一维概率轴，
   你原本的排序索引要跟着分布错位；
2. **温度先 softmax 还是先采样**：`probs = softmax(logits / temperature)`，multinomial 从 probs 采样；
3. **KV 增量**：每轮只把 `input_ids[:, past_len:]` 喂给模型（而不是整段），缓存不断追加——
   与 05/08 课的 `past_key_values` 闭环。

再加**早停**：`eos_token_id` 被生成就标记该样本 finished；全部 finished 就提前退出（不浪费算力）。

---

## 2. 阅读参考答案（minimind3/causal_lm.py）

- `__init__`：`self.model = MiniMindModel(config)`；在 `post_init()` 之前把 `lm_head` 与 `embed_tokens`
  绑定成同一 Parameter；头里 `slots = max(1, config.num_hidden_layers)` 预留 KV 槽；
- `forward`：`logits_to_keep` 时延后切头（只对最后 k 个位置做 lm_head，训练长序列省算力）；
  标签存在时算 shift 损失；返回 `MoeCausalLMOutputWithPast(loss, logits, presents, aux_loss)`；
- **generate**：复刻 minimind 的实现（temperature/top_k/top_p/repetition_penalty/streamer/return_kv），
  不开 tokenizer（纯 id 操作），* 留一个 `generate_transformer`（调用父类 GenerationMixin）作对照。

## 3. 手写任务清单

1. 类头：`config_class` + `_tied_weights_keys`；
2. `__init__`：主体模型 + lm_head，**绑定在 post_init 之前**；
3. `forward`：logits（含 `logits_to_keep` 优化）+ shift 损失 + 输出对象；
4. generate：贪心 → 温度 → top-k → top-p → 重复惩罚 → eos 早停 → KV 增量逐项加上；
5. 保存/加载可直接用父类（能 `save_pretrained` 就成功）；
6. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 4. 验收解读（verify/09_causal_lm.py）

| 检查 | 验什么 |
|---|---|
| 09.1 | logits 形状；`lm_head.weight is model.embed_tokens.weight`（同一对象）；`h@W^T` 一致 |
| 09.2 | 手动 shift 的交叉熵 == 模型 loss（1e-6）；-100 位置确实被忽略 |
| 09.3 | 12 步 AdamW 在固定随机数据上，loss 降到初值 95% 以下（模型真的能学） |
| 09.4 | 贪心生成：长度正确、结果确定可复现 |
| 09.5 | eos 提前停止生效 |
| 09.6 | temperature/top_k/top_p/repetition_penalty 都能跑通且改变结果 |
| 09.7 | **KV 缓存开 vs 关，生成结果完全一致**（缓存只是提速，不改变输出） |
| 09.8 | `save_pretrained` → `from_pretrained` 往返后 logits 一致（1e-5）（AutoClass 注册留到 13 课） |

## 5. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/causal_lm.py` | `model/model_minimind.py` 的 `MiniMindForCausalLM` |

对齐点：权重绑定时机、`_tied_weights_keys`、`logits_to_keep` 切片、自定义 generate 参数全集、
`MoeCausalLMOutputWithPast`。差异点：minimind 用 `add_start_docstrings` 装饰器（教程省掉）；
教程额外提供 `generate_transformer`（父类对照）与更完整的单测。

## 6. 常见坑（真实踩过）

- **绑定在 `post_init()` 之后才做**：`post_init` 里 `_init_weights` 会重新初始化 lm_head，绑定失效
  （验收 09.1 的 `is 同一对象` 直接抓）；
- **shift 忘了 `.contiguous()`**：`view` 在切片视图上会报"size mismatch"或静默错位；
- **top-p 的 mask 不移位**：`mask[..., 1:] = mask[..., :-1]` 的右值必须 `.clone()`（否则共享内存自覆盖）；
- **温度后直接 argmax 忘了 softmax**：multinomial 要求非负且和为 1，`probs` 需显式 `softmax`；
- **增量生成又喂整段**：第 2 轮起 `input_ids[:, past_len:]`，否则缓存形同虚设（结果仍对、只是慢）；
- **`eos` 用 float 判断**：`torch.isfinite` 对 Python float 会类型报错——生成 id 是 int，比较用
  `torch` 或转 `math` 处理。

## 7. 小结 & 下一课（里程碑！）

🎉 **到这里，模型核心全部完成**：config → RMSNorm → RoPE → Attention → FFN → Block → Model →
ForCausalLM(损失 + 生成 + 保存加载)。50 项验收全过，你已经"拥有"了一个能训练、能对话的完整语言模型结构！

**下一课（10-data）**：把文本变成模型能吃的数字——接入 minimind 的 6400 词表 BPE 分词器，
并手写 Pretrain/SFT 数据集类。你将看到 ChatML 模板（`<|im_start|>`）与 **-100 掩码标签**（user 提问
不参与损失，只监督 assistant 回答）如何落地。