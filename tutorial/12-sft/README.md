# 第 12 课 · 全参 SFT：让模型学会对话

## 目标

手写 `minimind3/train_full_sft.py`：在**对话数据**上继续训练，使模型从
"续写文本的机器"变成"能回答问题的助手"。

## 理论

### 1. SFT 是什么

预训练学"语言本身"，SFT 学"用语言完成指令"。数据 = `(user 问题, assistant 回答)`
对；训练目标 = 在**只监督 assistant 片段**的前提下最小化 CE（掩码在第 10 课实现）。

### 2. 为什么必须掩码

不掩码（user 也参与损失）的后果：
- 模型学会"模拟问题"而非"回答问题"；
- 训练分布偏离推理分布（推理时 user 段是输入不是输出）；
- verify 12.2 的检查法：SFT 后**正确回答的 CE 应显著低于错误回答**——
  若掩码错误，模型只是背书，拉不开差距。

### 3. 与预训练的超参差异

| 超参 | 预训练 | SFT |
|---|---|---|
| learning_rate | 5e-4 | 1e-4~5e-4（更大模型更低） |
| 数据构成 | 纯文本 | 对话 |
| 损失掩码 | padding 忽略 | 仅 assistant 有效 |
| 基础权重 | 从头 | 预训练权重（from_weight） |

SFT 学习率低：因为它只做“性格/能力微调”，不希望破坏预训练知识
（灾难性遗忘）。

### 4. 合成 SFT 数据的可学习性

封闭事实表（6 个名词 × 事实）+ 3 种问法模板，300 条样本：
微型模型（hidden=128）在 CPU 上几个 epoch 即可学会映射——这正是
验收 12.2 能够"判定内容学到没有"的基础。

## 任务清单（先手写再看答案）

1. `train_full_sft.py`：复用预训练循环骨架；
2. 替换数据为 `SFTDataset`（掩码随数据一起）；
3. 默认 `--from_weight pretrain`（可 `none` 从头）；
4. 学习率默认 1e-4 量级。

## 验收

```bash
.venv/Scripts/python.exe verify.py            # 全量（约 1 分钟）
.venv/Scripts/python.exe verify.py --fast     # 跳过训练类
```

检查点：**SFT loss 下降**、**正确答案 CE < 错误答案 CE（掩码有效性）**、权重可回载。
前向（12.3）支持 `--from_weight none` 独立跑通。

## 对照标准实现

`master:trainer/train_full_sft.py`（去掉 DDP/wandb/autocast/compile；核心循环一致）。

## 常见坑

- SFT 数据里 `max_seq_len` 至少覆盖"模板 + 问题 + 回答"，截断会把 answer 切没；
- `from_weight` 路径写错会静默 `strict=False` 加载出半成品——验收断言 strict 加载（12.3）；
- 有 GPU 做真实 SFT 时，把 `--learning_rate` 降到 1e-5~2e-5 并加大数据，
  避免把预训练知识冲掉。