"""第 08 课验收：MiniMindModel 主体。"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minimind3.config import MiniMindConfig  # noqa: E402
from minimind3.model_body import MiniMindModel  # noqa: E402
from verify._common import TINY_CONFIG, set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _make():
    cfg = MiniMindConfig(**TINY_CONFIG)
    model = MiniMindModel(cfg)
    model.eval()
    return cfg, model


@register("08.1 前向形状与 dtype")
def check_forward(args):
    cfg, model = _make()
    x = torch.randint(0, cfg.vocab_size, (2, 13))
    hidden, presents, aux = model(x, use_cache=True)
    assert hidden.shape == (2, 13, cfg.hidden_size)
    assert hidden.dtype == torch.float32
    # presents 每层一个
    assert len(presents) == cfg.num_hidden_layers
    assert aux.dim() == 0 or aux.numel() == 1
    assert torch.isfinite(aux)


@register("08.2 RoPE buffer 注册为非持久且形状正确")
def check_buffers(args):
    cfg, model = _make()
    assert hasattr(model, "freqs_cos") and hasattr(model, "freqs_sin")
    assert model.freqs_cos.shape == (cfg.max_position_embeddings, cfg.head_dim)
    assert model.freqs_sin.shape == (cfg.max_position_embeddings, cfg.head_dim)
    sd = model.state_dict()
    assert "freqs_cos" not in sd and "freqs_sin" not in sd, "buffer 不应进入 checkpoint"


@register("08.3 KV-Cache 增量与全量一致（模型级）")
def check_kv_eq(args):
    set_seed(1)
    cfg, model = _make()
    cos, sin = model.freqs_cos, model.freqs_sin  # noqa: F841 (表已注册)

    x = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        h_full, _, _ = model(x)

        # 增量：1+2+3
        past = None
        hs = []
        for start, length in [(0, 1), (1, 2), (3, 3)]:
            o, past, _ = model(x[:, start:start + length], past_key_values=past, use_cache=True)
            hs.append(o)
        h_inc = torch.cat(hs, dim=1)
    assert torch.allclose(h_full, h_inc, atol=1e-4), "模型级 KV-Cache 不一致"


@register("08.4 past_key_values 结构（每层 tuple）")
def check_past_structure(args):
    cfg, model = _make()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    dest = torch.tensor(7)  # 非 None 占位用底
    _, past, _ = model(x, past_key_values=None, use_cache=True)
    assert len(past) == cfg.num_hidden_layers
    for p in past[:1]:
        # 每层 2 元素：K 与 V
        assert p is not None and isinstance(p, tuple) and len(p) == 2
        assert p[0].shape == (1, 4, cfg.num_key_value_heads, cfg.head_dim)
        assert p[1].shape == (1, 4, cfg.num_key_value_heads, cfg.head_dim)
    assert dest is not None  # 仅保证上面右值不误用


@register("08.5 全模型梯度回传")
def check_grad(args):
    set_seed(2)
    cfg, model = _make()
    model.train()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    (model(x)[0] ** 2).sum().backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters())
    assert n_with_grad == n_total, f"仅 {n_with_grad}/{n_total} 参数收到梯度"