"""minimind3 预训练脚本（镜像 minimind trainer/train_pretrain.py 的核心，手写）。

在合成/小语料上做 next-token 预训练：
- cosine 学习率调度；
- 梯度裁剪 + 梯度累积；
- checkpoint（纯权重）+ resume（可续训）；
- 默认微型配置保证 CPU 快速演示；可调大做真实预训练（可选，需 GPU/大语料）。

用法示例（CPU）：
    python -m minimind3.train_pretrain --data_path data/tiny_pretrain.jsonl \
        --epochs 1 --batch_size 8 --max_seq_len 64 --hidden_size 96 \
        --num_hidden_layers 2 --accumulation_steps 2 --log_interval 1
"""
import argparse
import os
import sys
import time

# Windows 下 pyarrow/torch DLL 冲突防护（与 minimind 同款，见 issue #771）
import datasets  # noqa: F401

import torch
from torch import optim
from torch.utils.data import DataLoader

from .causal_lm import MiniMindForCausalLM
from .config import MiniMindConfig
from .datasets import PretrainDataset
from .train_utils import Logger, get_lr, load_checkpoint, save_checkpoint, save_weights, setup_seed


def train_epoch(epoch, loader, iters, args, model, optimizer, start_step=0):
    model.train()
    start_time = time.time()
    recorded = []  # (step, loss) 供验收脚本解析
    step = start_step  # 空 loader（全被 skip）时尾部判断需要初始值
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss = model(input_ids, labels=labels).loss
        (loss / args.accumulation_steps).backward()

        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        cur_loss = loss.item() * args.accumulation_steps
        recorded.append((step, cur_loss))
        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - start_time
            eta = elapsed / max(step - start_step, 1) * (iters - step) / 60
            Logger(
                f"[epoch {epoch + 1}/{args.epochs}] step {step}/{iters} "
                f"loss {cur_loss:.4f} lr {lr:.2e} eta {eta:.1f}min"
            )

        if args.save_interval > 0 and (step % args.save_interval == 0 or step == iters):
            save_weights(_weight_path(args), model)
            Logger(f"  weights saved -> {_weight_path(args)}")

        del input_ids, labels

    # 尾部残梯度
    if start_step + 1 <= step and step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return recorded


def _weight_path(args):
    moe = "_moe" if args.use_moe else ""
    return os.path.join(args.save_dir, f"{args.save_weight}_{args.hidden_size}{moe}.pth")


def _resume_path(args):
    return _weight_path(args).replace(".pth", "_resume.pth")


def main():
    ap = argparse.ArgumentParser(description="MiniMind3 Pretraining")
    ap.add_argument("--data_path", type=str, default="data/tiny_pretrain.jsonl")
    ap.add_argument("--save_dir", type=str, default="out")
    ap.add_argument("--save_weight", type=str, default="pretrain")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--max_seq_len", type=int, default=64)
    ap.add_argument("--hidden_size", type=int, default=96)
    ap.add_argument("--num_hidden_layers", type=int, default=2)
    ap.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from_weight", type=str, default="none", help="加载基础权重名（none=从头）")
    ap.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="自动续训")
    args = ap.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 配置：vocab 对齐所接分词器词表
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    model = MiniMindForCausalLM(lm_config).to(args.device)
    if args.from_weight != "none":
        wpath = os.path.join(args.save_dir, f"{args.from_weight}_{args.hidden_size}.pth")
        model.load_state_dict(torch.load(wpath, map_location=args.device), strict=False)
        Logger(f"base weights loaded: {wpath}")

    train_ds = PretrainDataset(args.data_path, max_length=args.max_seq_len)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if args.from_resume == 1 and os.path.exists(_resume_path(args)):
        ckp = load_checkpoint(_resume_path(args))
        checkpoint_unmatch_keys = model.load_state_dict(ckp["model"], strict=False)
        assert not checkpoint_unmatch_keys.missing_keys, f"缺键: {checkpoint_unmatch_keys.missing_keys}"
        optimizer.load_state_dict(ckp["optimizer"])
        start_epoch, start_step = ckp["epoch"], ckp["step"]
        Logger(f"resumed at epoch {start_epoch} step {start_step}")

    total_params = sum(p.numel() for p in model.parameters())
    Logger(f"model params: {total_params / 1e6:.2f}M")

    for epoch in range(start_epoch, args.epochs):
        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        loader = DataLoader(train_ds, batch_sampler=_batches(indices, args.batch_size, skip=start_step if epoch == start_epoch else 0))
        iters = len(loader) + (start_step if epoch == start_epoch else 0)
        records = train_epoch(epoch, loader, iters, args, model, optimizer, start_step=start_step if epoch == start_epoch else 0)
        start_step = 0  # 仅首个恢复轮次带偏移
        # 每个 epoch 结束必存 resume 检查点（与 save_interval 无关，保证可续训）
        save_checkpoint(_resume_path(args), model, optimizer, epoch + 1, records[-1][0] if records else 0)

    # 最后再存一份权重，保证"跑完即有产出"
    save_weights(_weight_path(args), model)
    Logger("done.")


def _batches(indices, batch_size, skip=0):
    """按 batch 切分，skip 个 batch 丢弃（续训偏移用）。"""
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    return batches[skip:]


if __name__ == "__main__":
    main()