"""第 06 课验收：SwiGLU FeedForward。"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minimind3.config import MiniMindConfig  # noqa: E402
from minimind3.feed_forward import FeedForward  # noqa: E402
from verify._common import TINY_CONFIG, set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@register("06.1 投影形状与输出形状")
def check_shapes(args):
    cfg = MiniMindConfig(**TINY_CONFIG)
    ff = FeedForward(cfg)
    h, i = cfg.hidden_size, cfg.intermediate_size
    assert ff.gate_proj.weight.shape == (i, h)
    assert ff.up_proj.weight.shape == (i, h)
    assert ff.down_proj.weight.shape == (h, i)
    x = torch.randn(3, 7, h)
    y = ff(x)
    assert y.shape == x.shape


@register("06.2 公式：down(silu(gate(x))*up(x))")
def check_formula(args):
    from minimind3.rope import rotate_half  # noqa: F401  (仅为演示可复用的旋转工具)

    set_seed(0)
    cfg = MiniMindConfig(**TINY_CONFIG)
    ff = FeedForward(cfg)
    x = torch.randn(2, 4, cfg.hidden_size)
    y = ff(x)
    ref = ff.down_proj(torch.nn.functional.silu(ff.gate_proj(x)) * ff.up_proj(x))
    assert torch.allclose(y, ref, atol=1e-6)


@register("06.3 silu 等价：x*sigmoid(x)，且与 ACT2FN 一致")
def check_silu(args):
    set_seed(1)
    x = torch.randn(1, 6, 16)
    silu = torch.nn.functional.silu(x)
    manual = x * torch.sigmoid(x)
    assert torch.allclose(silu, manual, atol=1e-6)
    from transformers.activations import ACT2FN

    assert torch.allclose(ACT2FN["silu"](x), manual, atol=1e-6)


@register("06.4 中间维度可覆盖（LoRA/精简测试用）")
def check_override(args):
    cfg = MiniMindConfig(**TINY_CONFIG)
    ff = FeedForward(cfg, intermediate_size=32)
    assert ff.gate_proj.weight.shape == (32, cfg.hidden_size)
    assert ff.down_proj.weight.shape == (cfg.hidden_size, 32)


@register("06.5 梯度回传与非线性")
def check_grad_nonlin(args):
    set_seed(2)
    cfg = MiniMindConfig(**TINY_CONFIG)
    ff = FeedForward(cfg)
    x = torch.randn(2, 3, cfg.hidden_size, requires_grad=True)
    (ff(x) ** 2).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    for p in ff.parameters():
        assert p.grad is not None and p.grad.shape == p.shape
    # 非线性：零输入时 SiLU(0)=0 -> 输出为 0
    y0 = ff(torch.zeros(1, 1, cfg.hidden_size))
    assert torch.allclose(y0, torch.zeros_like(y0), atol=1e-6)