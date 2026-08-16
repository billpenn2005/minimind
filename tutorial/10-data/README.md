# 第 10 课 · 分词器与数据集

## 目标

- 接入 minimind 自训练的 BPE 词表（`minimind3/tokenizer/`，随仓库分发零下载）；
- 手写 `minimind3/tokenizer_utils.py`：ChatML 对话模板 + assistant 掩码扫描；
- 手写 `minimind3/datasets.py`：`PretrainDataset` / `SFTDataset`。

## 理论

### 1. BPE 分词

- 字节对编码：从字符到子词合并，平衡词表大小与覆盖率；
- minimind 词表 6400，特殊 token：`0=endoftext`、`1=<|im_start|>`、`2=<|im_end|>`；
- 中文 1 token ≈ 1.5~1.7 字符（BPE 会把常用词/短语合并成整 token）。

### 2. 为什么用 ChatML 模板

把多轮对话渲染成有结构的文本：

```
<|im_start|>user\n什么是猫？<|im_end|>
<|im_start|>assistant\n猫是一种会抓老鼠的动物。<|im_end|>
```

模型学会输出"整段文本"，因此**训练时只监督 assistant 内容**，user/模板部分不产生损失
（标签置 -100）。verify 10.3 专门验证掩码的准确性。

### 3. 掩码扫描的不变量

- 标记片段：`<|im_start|>assistant\n`（先独立编码，再在完整序列中做子串匹配）；
- 有效区间：标记之后 → 下一个 `<|im_end|>`；
- 关键不变量：**独立编码的标记串 == 完整序列中的连续切片**（BPE 上下文无关，成立）。

### 4. 预训练样本

`bos + 文本 + eos`，padding 到定长，padding 区域标签 -100
（预训练对整段监督，但 padding 无效）。

## 任务清单（先手写再看答案）

1. `load_tokenizer()`：从包内目录加载 AutoTokenizer（惰性单例）；
2. `build_chat_prompt(conversations)`：渲染 ChatML 文本；
3. `encode_chat(...)`：文本 → ids（截断）；
4. `generate_labels(input_ids)`：返回长度相同的掩码数组（assistant 为原 id，其余 -100）；
5. `PretrainDataset` / `SFTDataset`：torch Dataset，产出 `(input_ids, labels)`。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：词表/特殊 token、encode/decode round-trip、**掩码方向正确性**（user 无效、
assistant 有效）、PretrainDataset pad 标签、SFTDataset 与合成数据打通（含一次微型前向）。

## 对照标准实现

- 词表：`master:model/tokenizer.json`、`tokenizer_config.json`（原样副本）；
- 数据集：`master:dataset/lm_dataset.py → PretrainDataset/SFTDataset`
  （教程简化为纯 ChatML，去掉了 tools/thinking/reasoning 分支，聚焦核心语义）。

## 常见坑

- `AutoTokenizer` 加载路径用包内 `Path(__file__).parent / "tokenizer"`，别用相对 cwd；
- 掩码扫描的 `start_marker` 要跟 `encode_chat` 同一套模板，否则匹配不到；
- `max_length` 截断要同时作用于 input_ids 与 labels（先截断再 padding）；
- 中文文本直接写进 jsonl 要 `ensure_ascii=False`，否则全部变成 \uXXXX 转义。