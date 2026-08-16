# 第 12 课 · 全参指令微调（SFT）

> [上一课 11-pretrain](../11-pretrain/README.md) ← [目录](../../README.md) → [下一课 13-convert](../13-convert/README.md)
>
> 难度：★★☆☆☆ ｜ 预计用时 2 小时 ｜ 前置：10+11 课

## 0. 本课目标

- [ ] 理解 SFT 与预训练的差异、掩码必要性、灾难性遗忘；
- [ ] 手写 `minimind3/train_full_sft.py`；
- [ ] 验收 63 项累计（本课新增 12.1~12.3）；含**内容级掩码证明**（CE gap）。

## 1. 理论速览

- **数学相同、数据不同**：SFT 仍是 next-token 交叉熵；差别在对话格式 + 掩码 + 更低 lr。
- **为什么必须掩码**：不掩码时模型学"问题→照抄问题"（数据中最常见的转移），变复读机。掩码后 user 区标签 -100，损失只来自 assistant，模型被迫学"问题→答案"。
- **机器判据（CE gap）**：训练完对比"正确答案 CE" vs "错误答案 CE"，差距越大学得越实（本课探针实测：好 0.93 vs 坏 1.37）。
- **灾难性遗忘**：预训练知识被 SFT 新分布冲掉 → SFT 用更低 lr（GPU 量级 1e-5~2e-5；教程封闭小数据用 1e-4~5e-4）。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/train_full_sft.py`

**头部 import**：先 `import datasets` 再 `import torch`；复用 11 课：`from .train_pretrain import _batches, _weight_path`；`from .causal_lm import MiniMindForCausalLM`、`from .config import MiniMindConfig`、`from .datasets import SFTDataset`、`from .tokenizer_utils import load_tokenizer`、`from .train_utils import Logger, get_lr, save_checkpoint, save_weights, setup_seed`。

**CLI 参数表**（与 11 课同型，仅默认值不同，下表列出差异项与共用项）：

| 参数 | 类型 | 默认 | 说明（相对 11 课） |
|---|---|---|---|
| `--data_path` | str | `data/tiny_sft.jsonl` | SFT 数据 |
| `--save_weight` | str | `full_sft` | 改为 full_sft |
| `--epochs` | int | 2 | |
| `--learning_rate` | float | **1e-4** | 更低 |
| `--max_seq_len` | int | **96** | 覆盖模板+问答 |
| `--from_weight` | str | **`pretrain`** | 默认基于预训练权重（`none`=随机） |
| 其余（batch_size/hidden_size/layers/use_moe/grad_clip/accum/log_interval/save_interval/device/seed/from_resume） | — | 同 11 课 | — |

**`train_epoch(epoch, loader, iters, args, model, optimizer, start_step=0, lm_config=None) -> list[(step, loss)]`**：与 11 课同骨架（lr 公式、accum、clip、日志格式 `[epoch …] step … loss … lr …`、周期权重保存带 `config=lm_config`、尾部残梯度 `step > start_step and step % accum != 0`）。

**`main()`**：
1. `lm_config = MiniMindConfig(hidden_size, num_hidden_layers, use_moe)`；建模型；
2. `from_weight != "none"`：`wpath = join(save_dir, f"{from_weight}_{hidden_size}.pth")`，**`assert os.path.exists(wpath)`**（防静默半加载）→ `load_state_dict(torch.load(wpath), strict=False)`；
3. `tokenizer = load_tokenizer()`；`SFTDataset(data_path, tokenizer, max_length=max_seq_len)`；AdamW（lr=1e-4 默认）；
4. 每 epoch：setup_seed + randperm + `DataLoader(batch_sampler=_batches(indices, bs))` → train_epoch → **epoch 尾 `save_checkpoint(_resume_path 等价路径)`**；
5. 结束 `save_weights(_weight_path(args), model, config=lm_config)`。

## 3. 手写步骤

实现 §2.1（**直接从 11 课复制骨架再改三处**：数据、掩码损失（模型 forward 用 labels 即含 -100）、from_weight）→ `.venv/Scripts/python.exe verify.py 12`（12.1/12.2 先快训预训练再 SFT，约 1~2 分钟；`--fast` 跳过）。

## 4. 验收解读（verify/12_sft.py）

| 检查（新） | 验什么 |
|---|---|
| 12.1 | 预训练→SFT 子进程：SFT loss 下降（`<0.9×初值`，需 >5 条日志）；`full_sft_96.pth` 产出；strict 加载 |
| 12.2 | **内容级掩码证明**：hidden 128/2 层/4 epoch/lr 5e-4/accum 1 训练后，正确回答的 CE 显著低于错误回答（gap ≥ 0.1；实测 ≈0.44） |
| 12.3 | `--from_weight none` 也可 SFT：strict 加载无缺失/多余键；logits 有限 |

> 12.2 的配置是探针校准的 CPU 收敛临界点——改小（如 96/2/2epoch）gap 会消失，那正是"没学会"的证据；过大 lr 会过拟合封闭数据，gap 也可能变小。

## 5. 参考答案

`answers/minimind3/train_full_sft.py`（先写后对；重点核对 from_weight 断言与 resume 路径生成）。

## 6. 常见坑

- **`max_seq_len` 太短**：模板+提问占满，回答被右截断 → 损失几乎只剩模板，训练"看似正常"却没进步（12.2 曾栽在这）；
- **`from_weight` 路径错**：assert 防呆（strict=False 静默半加载更糟）；
- **SFT 用预训练的大 lr**：震荡/遗忘（GPU 上 1e-5~2e-5）；
- **掩码丢了**：复读机，CE gap 归零。

## 7. 对照标准实现

| 你手写 | minimind 标准 |
|---|---|
| `minimind3/train_full_sft.py` | `trainer/train_full_sft.py` |

minimind 默认 lr 1e-5（真实数据长训练）；教程 1e-4（封闭小数据更易收敛）；教程剪掉 DDP/bf16/swanlab/compile。

## 8. 小结

✅ SFT 完成 + 内容级证据：正确回答 CE < 错误回答 CE（gap≈0.44）。
**下一课**：把 `full_sft_*.pth` 变成业界标准 HF 目录（13 课的最大坑：`.pth 会失忆`）。