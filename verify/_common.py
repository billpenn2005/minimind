"""verify 公共工具：路径、种子、注册器。"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 微型模型默认配置（验收用，CPU 秒级可跑）
TINY_CONFIG = dict(
    hidden_size=96,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=512,
    intermediate_size=128,
    max_position_embeddings=512,
    dropout=0.0,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> bool:
    """宽松数值比较（cpu / fp32）。"""
    a, b = a.float(), b.float()
    return bool(torch.allclose(a, b, atol=tol, rtol=tol))