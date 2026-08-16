"""minimind3 训练公共工具（镜像 minimind trainer/trainer_utils.py 的核心，手写）。"""
import math
import os
import random

import numpy as np
import torch


def setup_seed(seed: int) -> None:
    """固定全部随机源，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """cosine 退火（warmup 融入常数项，镜像 minimind）：lr*(0.1 + 0.45*(1+cos(pi*t/T)))。"""
    ratio = 0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps))
    return lr * ratio


def Logger(content: str) -> None:
    print(content, flush=True)


def save_checkpoint(path: str, model, optimizer, epoch: int, step: int, extra: dict | None = None) -> None:
    """保存可续训检查点：模型 + 优化器状态 + 进度。"""
    raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    raw = getattr(raw, "_orig_mod", raw)
    data = {
        "model": {k: v.float().cpu() for k, v in raw.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        **(extra or {}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(data, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def save_weights(path: str, model, config=None) -> None:
    """保存纯权重（用于下游 SFT / 转换）：name -> tensor。

    若给定 config，同时写一份 sidecar `*.config.json`：权重不自描述尺寸类
    超参（rope_theta/max_position_embeddings 等），sidecar 保证转换时可还原。
    """
    raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    raw = getattr(raw, "_orig_mod", raw)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({k: v.float().cpu() for k, v in raw.state_dict().items()}, path)
    if config is not None:
        import json

        sidecar = path.replace(".pth", ".config.json")
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)