# 第 11 课 · 预训练循环

> [上一课 10-data](../10-data/README.md) ← [目录](../../README.md) → [下一课 12-sft](../12-sft/README.md)
>
> 难度：★★★☆☆ ｜ 预计用时 2~3 小时 ｜ 前置：10 课

## 0. 本课目标

- [ ] 理解训练循环五大件：数据批次→前向→反向→梯度处理→步进；
- [ ] 手写 `train_utils.py`（get_lr/setup_seed/存取）与 `train_pretrain.py`；
- [ ] 验收 60 项累计（本课新增 11.1~11.4）；11.2/11.3 含真实 CPU 训练。

## 1. 理论速览

- **循环骨架**：`loss = model(ids, labels).loss / accum` → `backward()` → 攒够 `accum` 次 → `clip_grad_norm_` + `optimizer.step()` + `zero_grad` → 更新 lr。
- **cosine 退火**（minimind 公式）：$\text{lr}(t) = \text{lr}_{max}\left(0.1 + 0.45(1+\cos\frac{\pi t}{T})\right)$——首步 1.0×lr、中点 0.55×lr、末步 **0.1×lr**（不归零：尾部保留探索/巩固）。
- **梯度裁剪**：把全体梯度 L2 范数钳到 grad_clip（防爆炸）；**累积**：小 batch 攒大 batch 的等价替代（除以 accum 保持尺度）。
- **checkpoint vs 权重**：resume 文件 = model+optimizer+epoch+step（"记忆"）；weights = 纯 state_dict float32（"成果"，供 SFT/转换）+ **sidecar `*.config.json`**（第 13 课关键：`.pth` 不含 rope_theta 等非张量超参）。
- **可复现**：`setup_seed(seed)` + 每 epoch `setup_seed(seed+epoch)` + `randperm`。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/train_utils.py`

**模块级**：`import math, os, random`；`import numpy as np`；`import torch`。

| 函数 | 签名 | 返回 | 行为 |
|---|---|---|---|
| `setup_seed` | `(seed: int) -> None` | None | `random/np/torch/torch.cuda`（含 manual_seed_all）全部设种子 |
| `get_lr` | `(current_step: int, total_steps: int, lr: float) -> float` | float | `lr * (0.1 + 0.45*(1+cos(pi*step/total)))` |
| `Logger` | `(content: str) -> None` | None | `print(content, flush=True)` |
| `save_checkpoint` | `(path: str, model, optimizer, epoch: int, step: int, extra: dict \| None = None) -> None` | None | 解包 DDP/`_orig_mod` → `{"model": {k: v.float().cpu()…}, "optimizer": optimizer.state_dict(), "epoch":…, "step":…, **extra}` → `torch.save`（目录自动建） |
| `load_checkpoint` | `(path: str) -> dict` | dict | `torch.load(path, map_location="cpu")` |
| `save_weights` | `(path: str, model, config=None) -> None` | None | 解包 → `torch.save({k: v.float().cpu()…}, path)`；**若 config 非 None**：写 sidecar `path.replace(".pth", ".config.json")` = `json.dump(config.to_dict(), …, indent=2)` |

### 2.2 新建 `minimind3/train_pretrain.py`

**头部**：**先 `import datasets` 再 `import torch`**（Windows DLL 防护）。

**CLI 参数表（argparse，类型/默认从严）**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--data_path` | str | `data/tiny_pretrain.jsonl` | 预训练 jsonl |
| `--save_dir` | str | `out` | 产物目录 |
| `--save_weight` | str | `pretrain` | 权重名前缀 |
| `--epochs` | int | 1 | 轮数 |
| `--batch_size` | int | 8 | 每批样本 |
| `--learning_rate` | float | 5e-4 | AdamW 初始 lr |
| `--max_seq_len` | int | 64 | 序列长 |
| `--hidden_size` | int | 96 | 隐藏维 |
| `--num_hidden_layers` | int | 2 | 层数 |
| `--use_moe` | int | 0（choices 0/1） | MoE 开关（教程恒 0） |
| `--grad_clip` | float | 1.0 | 梯度裁剪范数 |
| `--accumulation_steps` | int | 1 | 累积步数 |
| `--log_interval` | int | 10 | 日志间隔 |
| `--save_interval` | int | 50 | 周期保存权重间隔（0=关） |
| `--device` | str | cpu | 设备 |
| `--seed` | int | 42 | 种子 |
| `--from_weight` | str | `none` | 基础权重名（none=从头） |
| `--from_resume` | int | 0（choices 0/1） | 断点续训 |

**模块级函数**：
- `_weight_path(args) -> str`：`os.path.join(args.save_dir, f"{args.save_weight}_{args.hidden_size}{'_moe' if args.use_moe else ''}.pth")`；
- `_resume_path(args) -> str`：`_weight_path(args).replace(".pth", "_resume.pth")`；
- `_batches(indices: list[int], batch_size: int, skip: int = 0)`：切批后 `batches[skip:]`（生成器/列表皆可）；
- `train_epoch(epoch, loader, iters, args, model, optimizer, start_step=0, lm_config=None) -> list[tuple[int, float]]`：
  - 局部 `recorded` 收集 `(step, loss)`；
  - **`step = start_step` 必须先初始化**（resume 空 loader 时尾部判断要用）；
  - 每步：`lr = get_lr(epoch*iters+step, args.epochs*iters, args.learning_rate)` 写入 `optimizer.param_groups[0]["lr"]` → `loss = model(input_ids, labels=labels).loss` → `(loss/accum).backward()` → `step % accum == 0` 时 clip+step+zero_grad；
  - 记录 `loss.item() * accum`；`step % log_interval == 0 or step == iters` 时打印 `[epoch {e}/{E}] step {s}/{iters} loss {x:.4f} lr {y:.2e} …`（**验收解析此格式**）；
  - 周期保存：`args.save_interval > 0 and (step % args.save_interval == 0 or step == iters)` → `save_weights(_weight_path(args), model, config=lm_config)`；
  - 尾部残梯度：`start_step + 1 <= step and step % accum != 0` → clip+step+zero_grad；
  - 返回 recorded。
- `main()`：
  1. `setup_seed(args.seed)`；`lm_config = MiniMindConfig(hidden_size=…, num_hidden_layers=…, use_moe=bool(…))`；
  2. 建模型；`from_weight != "none"` 时 `load_state_dict(torch.load(join(save_dir, f"{from_weight}_{hidden_size}.pth")), strict=False)`；
  3. `PretrainDataset(data_path, max_length=max_seq_len)`；AdamW；
  4. resume：`_resume_path` 存在时 `load_checkpoint` → `load_state_dict(ckp["model"], strict=False)`（assert 无 missing）→ 恢复 optimizer/epoch/step；
  5. 每 epoch：`setup_seed(seed+epoch)` + `randperm` + `DataLoader(batch_sampler=_batches(indices, bs, skip=start_step 若首轮))` → `train_epoch(...)` → **epoch 尾无条件 `save_checkpoint(_resume_path(...))`**；
  6. 结束 `save_weights(_weight_path(args), model, config=lm_config)`（**必带 config sidecar**）。

## 3. 手写步骤

实现 §2.1 → §2.2 → `.venv/Scripts/python.exe verify.py 11`（全量含 CPU 训练约 1 分钟；快查可 `--fast`）。

## 4. 验收解读（verify/11_pretrain.py）

| 检查（新） | 验什么 |
|---|---|
| 11.1 | `get_lr` 三点数值：`get_lr(1)=lr`、中点=0.55lr、末点=0.1lr 且递减 |
| 11.2 | **端到端预训练子进程**（hidden 96/2 层/seq 64/batch 8/accum 2/epochs 1/seed 0）：解析 `loss` 日志断言下降（宽松阈值 0.95）；`out/pretrain_96.pth` + `_resume.pth` 存在；权重 strict 载入、logits 有限 |
| 11.3 | 断点续训 `--from_resume 1 --epochs 2` 追加成功 |
| 11.4 | 数据首行 schema `{"text": …}` |

## 5. 参考答案

`answers/minimind3/train_utils.py`、`answers/minimind3/train_pretrain.py`（先写后对）。

## 6. 常见坑

- **`save_interval=0` 的 `%0` 除零**：条件顺序必须 `args.save_interval > 0 and (...)`；
- **resume 空 loader**：`step` 未初始化 → UnboundLocalError（预初始化成 `start_step`）；
- **`zero_grad` 频次**：只在真实 step 时清一次（不必每 forward 清）；
- **`save` 前没 float32/cpu**：半精度直接 `torch.save` 丢精度、13 课 strict 加载报警；
- **权重保存的 goroutine**：用 `torch.save` 即可；backup 文件用 `.tmp + os.replace` 原子更稳（教程 checkpoint 直接 save）。

## 7. 对照标准实现

| 你手写 | minimind 标准 |
|---|---|
| `minimind3/train_utils.py` | `trainer/trainer_utils.py` |
| `minimind3/train_pretrain.py` | `trainer/train_pretrain.py` |

教程剪掉 DDP/swanlab/wandb/autocast+scaler/compile（文档列为扩展），核心（get_lr、累积、clip、resume、sidecar）对齐。

## 8. 小结

✅ 预训练循环完成：cosine 1.0→0.55→0.1、累积、裁剪、可续训、可复现——并已真的让 loss 下降。
**下一课**：SFT——换数据、换掩码、降 lr，并亲手证明"不掩码 = 复读机"。