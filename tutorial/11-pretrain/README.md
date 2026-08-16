# 第 11 课 · 预训练循环（train_pretrain.py）

> 难度：★★★☆☆ ｜ 预计用时：2~3 小时 ｜ 前置：10 课数据集
> 学完本课你应能回答：**训练循环由哪几块组成？cosine 学习率为什么这样退火？梯度累积在干什么？断点续训怎么做到"从原地继续"？**

---

## 0. 本课目标

- [ ] 理解训练循环五大件：数据批次 → 前向 → 反向 → 梯度处理 → 优化器步进；
- [ ] 理解余弦退火学习率（minimind 公式）与它的动机；
- [ ] 理解梯度裁剪与梯度累积；
- [ ] 手写 `minimind3/train_utils.py`（get_lr / setup_seed / 保存加载）与 `train_pretrain.py`；
- [ ] 验收 4 项新检查（11.1~11.4，累计 59 项），其中 11.2/11.3 会**真的训练一个微型模型**。

---

## 1. 理论：一个训练循环长什么样？

```
for epoch in range(epochs):
    for batch in loader:
        loss = model(ids, labels).loss / accumulation_steps     # 前向
        loss.backward()                                          # 反向
        if (step+1) % accumulation_steps == 0:                   # 攒够梯度
            clip_grad_norm_(params, max_norm)                    # 裁剪
            optimizer.step(); optimizer.zero_grad()              # 步进
            lr = get_lr(global_step, total_steps)                # 退火
```

### 1.1 优化器与学习率

- 优化器：**AdamW**（自适应矩估计 + 权重衰减解耦）。LLM 事实标准。minimind 用 `lr=5e-4`、无 weight_decay（小模型）；
- **学习率**是最敏感的旋钮：太大发散，太小龟速。当代做法是**预热 + 退火**，minimind 用纯余弦退火：

$$
\text{lr}(t) = \text{lr}_{\max} \times \left( 0.1 + 0.45 \times \left(1 + \cos\!\frac{\pi t}{T}\right) \right)
$$

首步 = 1.0·lr，中点 = 0.55·lr，末步 = **0.1·lr**。为什么尾巴要压到 0.1 而不归零？
- 归零会挤掉最后阶段"探索性"的步长，且训练曲线在终点会突然失速；
- 保留 10% 基线让模型在最后阶段"稳稳巩固"，这是总结经验（minimind 与多家实现一致）。

`get_lr` 在 `train_utils.py`，验收 11.1 用三个数值点精确核对（t=1 → lr，t=50% → 0.55lr，t=99% → 0.1lr）。

### 1.2 梯度裁剪（Gradient Clipping）

```
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # 默认 1.0
```

把全体梯度的 L2 范数钳到 `grad_clip` 以内（超了就整体缩放）。防止**梯度爆炸**把参数一步推飞（loss 变 NaN）。
代价极小、收益极大，训练脚本里是"必装安全带"。

### 1.3 梯度累积（Gradient Accumulation）

显存不够时，把大 batch 拆成小 batch，**攒**几轮的梯度再 step——数学上等价于大 batch：

```
每小步:        loss = loss / accumulation_steps; loss.backward()   # 不 step
攒到 N 小步:    optimizer.step(); optimizer.zero_grad()
```

除 `accumulation_steps` 是为了让平均梯度的尺度与"原版大 batch"一致（否则梯度被放大 N 倍）。
CPU 小模型也用得上它（比如 8 的 batch × 4 累积 ≈ 32 的等效 batch）。

### 1.4 Checkpoint vs 权重：两样都得存

| 文件 | 内容 | 用途 |
|---|---|---|
| `weights_*.pth`（如 `pretrain_96.pth`） | 纯 model.state_dict（float32） | 推理/微调/转换的"成果" |
| `*_resume.pth` | model + optimizer + epoch + step + seed + … | **断点续训**的"记忆" |

断点续训不是"把权重读回来重新跑"，而是要**无缝接续**：优化器状态（动量/方差）也得带上，
否则前几轮学习会被"冷启动"破坏。`save_checkpoint` 用 `.tmp` + `os.replace` 原子写入防断电半文件。

### 1.5 可复现性：固定一切随机源

`setup_seed(seed)`：`random` / `numpy` / `torch` / `torch.cuda` 全部固定；数据打乱在每个 epoch
用 `setup_seed(seed + epoch)` + `randperm` 保持确定性。验收 11.2 正是靠这个"两次训练结果一致"的
性质才能稳定断言 loss 下降。

### 1.6 本课的两处细节（真实 bug 的教训）

1. **`save_interval=0`**（关闭周期保存）时 `step % 0` 会 ZeroDivisionError——必须写成
   `args.save_interval > 0 and (step % args.save_interval == 0 or step == iters)`（**短路顺序**）；
2. **resume 跳完所有 batch** 后，循环外的"残差梯度 flush"会引用未定义的 `step`——先把
   `step = start_step` 初始化在外。

---

## 2. 手写任务清单

1. `train_utils.py`（约 60 行）：
   - `setup_seed`；`Logger`（带 flush 的 print）；`get_lr`（§1.1 公式）；
   - `save_checkpoint / load_checkpoint / save_weights`（cast 到 float32、cpu，解包 DDP/compile 名——本课无 DDP，但函数名保持兼容）；
2. `train_pretrain.py`：完整 CLI 参数（data_path/save_dir/save_weight/epochs/batch_size/learning_rate/max_seq_len/hidden_size/layers/grad_clip/accumulation_steps/log_interval/save_interval/from_resume/seed/...）；
   - `datasets` 在 `torch` 之前 import（Windows）；
   - 每 epoch：`setup_seed(seed+epoch)` + `randperm` + `_batches(indices, batch_size, skip=...)`（resume 跳过已处理 batch）；
   - 循环：forward/backward/累积/step/clip/get_lr；`log_interval` 打印 `loss X lr Y`（验收解析）；
   - 周期保存权重 + **每 epoch 尾无条件存 resume**；结束再存一份权重（附 config sidecar，13 课用）；
3. 验收。

```bash
.venv/Scripts/python.exe verify.py      # 全量：11.2/11.3 真的训练（约 1 分钟）
```

## 3. 验收解读（verify/11_pretrain.py）

| 检查 | 验什么 |
|---|---|
| 11.1 | get_lr 三点数值（1→lr，中点→0.55lr，末点→0.1lr；单调递减） |
| 11.2 | **端到端预训练**（子进程）：hidden 96 / 2 层 / seq 64 / batch 8 / 累积 2 / 1 epoch；解析日志断言 **loss 下降**；`pretrain_96.pth` 与 `_resume.pth` 存在；权重可 strict 载入 MiniMindForCausalLM 且 logits 有限 |
| 11.3 | 断点续训：`--from_resume 1 --epochs 2` 追加训练成功（产出更新权重） |
| 11.4 | 数据首行 schema = `{"text": ...}` |

> 11.2 的宽松阈值（`losses[-1] < losses[0]*0.95`）是有意为之：给不同机器/浮点实现留余量，
> 但仍能抓住"训练没生效"（loss 不降）的硬伤。

## 4. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/train_utils.py` | `trainer/trainer_utils.py` |
| `minimind3/train_pretrain.py` | `trainer/train_pretrain.py` |

教程**剪掉**：DDP 多卡 / torch.compile / swanlab-wandb / autocast+scaler（CPU 不需要）/ SkipBatchSampler
（用 `_batches(skip=...)` 等价实现）。保留核心：get_lr / 累积 / clip / (epoch,step) 级 resume / 权重保存。
被剪能力全部列入"进阶"。

## 5. 常见坑

- `% 0` 除零（短路顺序）；resume 空 loader 的 `step` 未初始化；`zero_grad` 放错位置（该在真实 step 前，而非每次 forward 后）；`save` 没解包 `module.`（若套过 DDP）；`torch.save` 前忘了 float32（半精度直接存会丢精度 & 13 课 strict 加载报警）；`os.replace` 才原子（直接 `open(w)` 断电会写坏）。

## 6. 小结 & 下一课

- ✅ 循环五大件：前向/反向/裁剪/步进/退火；
- ✅ cosine：1.0 → 0.55 → 0.1（尾巴留 10% 防失速）；
- ✅ 累积 = 显存不足时的大 batch 平替；clip = 防爆炸安全带；
- ✅ checkpoint 存"记忆"（含优化器），权重存"成果"；
- ✅ 每 epoch 固定种子 → 验收可复现。

**下一课（12-sft）**：预训练让模型"会说话"，SFT 让模型"会回答"。你会用**同一个循环**但换成 SFT 数据集
与掩码损失——并亲手验证一个深刻事实：**如果不掩码（连 user 的问题也学），模型会退化成复读机，
问答质量断崖式下跌**。