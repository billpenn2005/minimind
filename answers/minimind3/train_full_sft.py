"""minimind3 全参 SFT 脚本（镜像 minimind trainer/train_full_sft.py 的核心，手写）。

与预训练唯一本质区别：数据是（user/assistant）对话，**只有 assistant 片段参与损失**
（标签掩码已在第 10 课的 SFTDataset 中完成），学习率一般更低（1e-4 量级）。

用法示例（CPU，基于上一步预训练权重）：
    python -m minimind3.train_full_sft --data_path data/tiny_sft.jsonl \
        --save_dir out --from_weight pretrain --epochs 1 --batch_size 8 \
        --max_seq_len 96 --hidden_size 96 --num_hidden_layers 2 --log_interval 1
"""
import argparse
import os
import sys
import time

import datasets  # noqa: F401  # Windows pyarrow/torch DLL 冲突防护

import torch
from torch import optim
from torch.utils.data import DataLoader

from .causal_lm import MiniMindForCausalLM
from .config import MiniMindConfig
from .datasets import SFTDataset
from .tokenizer_utils import load_tokenizer
from .train_pretrain import _batches, _weight_path  # 复用切批/路径逻辑
from .train_utils import Logger, get_lr, save_checkpoint, save_weights, setup_seed


def train_epoch(epoch, loader, iters, args, model, optimizer, start_step=0, lm_config=None):
    model.train()
    start_time = time.time()
    recorded = []
    step = start_step
    for input_ids, labels in loader:
        step += 1
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
            Logger(f"[epoch {epoch + 1}/{args.epochs}] step {step}/{iters} loss {cur_loss:.4f} lr {lr:.2e} eta {eta:.1f}min")

        if args.save_interval > 0 and (step % args.save_interval == 0 or step == iters):
            save_weights(_weight_path(args), model, config=lm_config)
            Logger(f"  weights saved -> {_weight_path(args)}")

        del input_ids, labels

    if step > start_step and step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return recorded


def main():
    ap = argparse.ArgumentParser(description="MiniMind3 Full SFT")
    ap.add_argument("--data_path", type=str, default="data/tiny_sft.jsonl")
    ap.add_argument("--save_dir", type=str, default="out")
    ap.add_argument("--save_weight", type=str, default="full_sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_seq_len", type=int, default=96)
    ap.add_argument("--hidden_size", type=int, default=96)
    ap.add_argument("--num_hidden_layers", type=int, default=2)
    ap.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from_weight", type=str, default="pretrain", help="基础权重名（none=从头）")
    ap.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(lm_config).to(args.device)
    if args.from_weight != "none":
        wpath = os.path.join(args.save_dir, f"{args.from_weight}_{args.hidden_size}.pth")
        assert os.path.exists(wpath), f"找不到基础权重: {wpath}"
        model.load_state_dict(torch.load(wpath, map_location=args.device), strict=False)
        Logger(f"base weights loaded: {wpath}")

    tokenizer = load_tokenizer()
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        loader = DataLoader(train_ds, batch_sampler=_batches(indices, args.batch_size))
        records = train_epoch(epoch, loader, len(loader), args, model, optimizer, lm_config=lm_config)
        save_checkpoint(_weight_path(args).replace(".pth", "_resume.pth"), model, optimizer, epoch + 1, records[-1][0] if records else 0)

    save_weights(_weight_path(args), model, config=lm_config)
    Logger("done.")


if __name__ == "__main__":
    main()