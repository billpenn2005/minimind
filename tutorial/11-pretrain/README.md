# 第 11 课 · 预训练循环

## 目标

手写 `minimind3/train_pretrain.py`（训练循环）与 `train_utils.py`（调度/checkpoint），
在合成数据上完成一次可验收的端到端预训练。

## 理论

### 1. 预训练目标

海量文本上做 next-token 预测（第 09 课的 CE 损失），学到语言结构与世界知识。
本教程用**封闭词表的合成语料**演示同一机制（少量步即可看到 loss 明显下降）；
真实语料/更大模型 = 把参数调大 + 换数据（可选，通常需 GPU）。

### 2. 训练循环的五件套（本课全部手写）

| 组件 | 作用 |
|---|---|
| 学习率调度 | cosine 退火 `lr*(0.1+0.45*(1+cos(πt/T)))`：1.0lr → 0.55lr(中点) → 0.1lr(终点)，收敛更稳 |
| 梯度裁剪 | `clip_grad_norm_(params, max_norm)` 防爆炸 |
| 梯度累积 | 小 batch 攒够 `accumulation_steps` 再 optimizer.step()，等效大 batch |
| checkpoint | 纯权重（下游用）+ resume（model/optimizer/step，续训） |
| 复现 | `setup_seed` + 可复现合成数据 |

### 3. 优化器选择

AdamW（解耦权重衰减）：LLM 事实标准。minimind 用 `optim.AdamW(lr=5e-4, 默认无 weight_decay)`。

### 4. resume 语义

`step` 是**全局步**（epoch*iters+step）。续训时：跳过前 `start_step` 个 batch（`_batches(skip=...)`），
日志与学习率调度按全局步继续——verify 11.3 专门验证"断点继续"路径。

## 任务清单（先手写再看答案）

1. `get_lr`（cosine）与 `setup_seed`；
2. `save_checkpoint / load_checkpoint / save_weights`（raw_model 解包 + fp32 保存）；
3. 训练循环：前向 → 累积反向 → 裁剪 → step → 日志 → 定期存权重；
4. `main`：args → 配置组装 → 数据 → resume 分支 → 多 epoch 循环。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：**cosine 调度数值**（1.0/0.55/0.1）、**合成数据端到端训练**（loss 下降 + 权重可回载）、
**resume 续训**（第二次运行不报错、断点接续）、数据文件自洽。
验收内训练耗时：几秒（hidden=96 / 2 层 / CPU）。

## 对照标准实现

`master:trainer/train_pretrain.py`：教程去掉 DDP/swanlab/wandb/autocast/torch.compile
（在"进阶"指引中说明），保留调度、裁剪、累积、checkpoint、resume 等核心。

## 常见坑

- `step % save_interval` 遇 `save_interval=0` 会 ZeroDivision（短路顺序要写对）；
- resume 后空 loader（所有 batch 被 skip）时 `step` 变量要预初始化；
- `optimizer.zero_grad(set_to_none=True)` 在累积模式下只该在真正 step 时调用；
- checkpoint 里存 `raw_model.state_dict()`（DDP/compile 包一层解一层）。