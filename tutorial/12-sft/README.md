# 第 12 课 · 全参指令微调（train_full_sft.py）

> 难度：★★☆☆☆ ｜ 预计用时：2~3 小时 ｜ 前置：10 课掩码 + 11 课训练循环
> 学完本课你应能回答：**SFT 和预训练差在哪？为什么"必须"掩码？什么是灾难性遗忘？**

---

## 0. 本课目标

- [ ] 理解 SFT（Supervised Fine-Tuning）的目标与做法；
- [ ] 亲手证明**掩码的必要性**：不掩码 = 复读机（验收 12.2 的内容级证据）；
- [ ] 复用 11 课循环骨架，写出 `train_full_sft.py`（换数据、换损失、换学习率）；
- [ ] 验收 3 项新检查（12.1~12.3，累计 62 项全过）。

---

## 1. 理论：SFT 在干什么？

### 1.1 从"会说话"到"会回答"

- **预训练**：海量文本 → 学会"语言的统计规律"（说什么像人话）；
- **SFT**：精选的"问题-回答"对 → 学会"被问倒时给出合格答案"（说什么有用）。

两者**数学上完全一样**：都是 next-token 交叉熵。差别只在**数据**（对话格式）与**掩码**（只监督回答）。

### 1.2 为什么必须掩码？（本课的精髓）

如果**不掩码**，模型会把"用户的问题"也当成要预测的目标去背。它学会的是：**无论问什么都照抄问题**
（因为"预测下一个 token"的数据里，problem→problem 的转移最常见）。训练结束你问它"什么是猫？"，
它大概率复读"什么是猫？"——**复读机**。

掩码后，user 区标签全 -100（10 课已实现），损失只来自 assistant 区，模型被迫学习"问题→答案"映射。
验收 12.2 用一种**可量化**的方式证明这件事：同一模型跑两遍（掩码 vs 不掩码），从 SFT 数据里取
"正确答案"和一段"错误答案"，对比两者在 assistant 区上的交叉熵——

| 训练方式 | 正确答案 CE | 错误答案 CE | 结论 |
|---|---|---|---|
| 掩码 SFT | **低**（0.93） | 高（1.37） | 模型真学会了事实 |
| 不掩码 SFT | 高 | 高 | 模型什么都没学会（复读机） |

**CE 差距（gap）是"学到了吗"的机器判据**——这也是整个教程最强的验收之一。

### 1.3 灾难性遗忘与学习率

预训练模型已有大量知识；SFT 用新分布再训，容易**把旧知识冲掉**（灾难性遗忘）。对策：
- **更低的学习率**：预训练 `5e-4` → SFT 默认 `1e-4`（真 GPU 全量 SFT 甚至 `1e-5~2e-5`）；
- 少太多 epoch（教程小模型 2~4 个就够：数据封闭、任务小）；
- 混合通用数据（进阶）。

### 1.4 本课实现 = 11 课循环的"换芯"

`train_full_sft.py` 直接复用 11 课的 `_batches` / `_weight_path`（`from minimind3.train_pretrain import ...`），
只改三处：
1. 数据：`SFTDataset`（10 课）而非 PretrainDataset；
2. 掩码损失：模型 forward 直接用 `labels`（-100 来自数据集，09 课的 shift+ignore 已就位）；
3. 初始化：`--from_weight pretrain`（默认）载入预训练权重继续学；`--from_weight none` 从随机开始（验收 12.3）。

注意 `from_weight` 的文件名约定：`{save_dir}/{from_weight}_{hidden_size}.pth`——比如先跑 11 课得到
`out/pretrain_96.pth`，12 课默认就能找到它（找不到直接 assert 报错，保护新手）。

---

## 2. 手写任务清单

1. 复制 11 课的循环骨架（参数表 + epochs 循环 + 累积/clip/退火 + 保存）；
2. 换数据源 `SFTDataset`；`from_weight` 支持 `pretrain` / `none`（strict 加载）；
3. 微调默认参数：`lr=1e-4`、`epochs=2`、`max_seq_len=96`（覆盖模板+问答；太短会截掉回答！）;
4. 保持"每 epoch 尾 checkpoint + 结束存权重（含 config sidecar）"；
5. 验收。

```bash
.venv/Scripts/python.exe verify.py     # 全量：12.1/12.2 会先快速预训练再 SFT（约 1 分钟）
```

## 3. 验收解读（verify/12_sft.py）

| 检查 | 验什么 |
|---|---|
| 12.1 | 预训练 → SFT 两段子进程：SFT loss 下降（`losses[-1] < losses[0]*0.9`，需 >5 条日志）；`full_sft_96.pth` 产出；权重可 strict 加载 |
| 12.2 | **内容级掩码证明**：掩码 SFT（hidden 128 / 2 层 / 4 epoch / lr 5e-4）后，正确答案 CE 显著低于错误答案 CE（gap ≥ 0.1）——**这就是 12 课的核心验收**（实测 gap ≈ 0.44） |
| 12.3 | `--from_weight none` 也能 SFT：strict 加载无缺失/多余键，logits 有限 |

> 12.2 的训练配置（128/2/4epoch/5e-4/acc1）不是随便选的：它是本教程探针实验证出的
> **CPU 上内容级收敛的最小配置**。想自己改小（比如 96/2/2epoch）会看到 CE 差距消失——那正是
> "没学会"的机器证据。学习率/epoch 若改太大（如 lr 8e-4+4epoch）会过拟合封闭数据，gap 也可能变小。

## 4. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/train_full_sft.py` | `trainer/train_full_sft.py` |

minimind 默认 `lr=1e-5`（真实数据/长训练），教程用 `1e-4`（封闭小数据更快收敛）；minimind 支持
DDP / bf16 / swanlab / compile——教程全部剪掉（进阶）。

## 5. 常见坑

- **`max_seq_len` 太短**：模板+提问就把预算吃光，assistant 回答被 `truncation` 从右切掉——
  损失几乎只剩模板，训练"看似正常"却毫无进步（实测教训：12.2 探针阶段就栽在这类配置上）；
- **`from_weight` 路径写错**：严格断言防呆；若用 `strict=False` 会"静默载入一半参数"（更糟）；
- **SFT 丢大 lr**：把预训练的 5e-4 直接搬来会震荡/遗忘；GPU 上全量 SFT 用 1e-5~2e-5；
- **忘了掩码**：模型变复读机，12.2 的 CE gap 直接归零（这也是本课"手写证明"的意义）；
- **`save_weight` 与 `from_weight` 前缀混淆**：`--save_weight full_sft --from_weight pretrain`，
  两个名字别写反。

## 6. 小结 & 下一课

- ✅ SFT = 换数据 + 换掩码 + 降 lr 的预训练；
- ✅ 不掩码 = 复读机（有 CE 证据）；
- ✅ 灾难性遗忘 → 低 lr、少 epoch；
- ✅ 内容级验收：正确答案 CE < 错误答案 CE，gap 即"学到程度"。

**下一课（13-convert）**：把你的 `full_sft_*.pth` 变成**业界标准**的 HF transformers 目录
（config.json + pytorch_model.bin + tokenizer）。你会学到 `auto_map` / remote-code 动态加载 /
round-trip 一致性，以及一个深刻的坑：**.pth 会"失忆"**——非张量超参（rope_theta 等）不随权重保存，
必须靠 sidecar config 拯救。