"""第 03 课验收：RMSNorm。"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from verify._common import set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@register("03.1 输出形状、初值与公式数值一致")
def check_basic(args):
    from minimind3.rms_norm import RMSNorm

    set_seed(0)
    rms = RMSNorm(dim=32)
    x = torch.randn(4, 10, 32)
    y = rms(x)
    assert y.shape == x.shape
    assert rms.weight.shape == (32,)
    assert torch.all(rms.weight == 1.0)
    # 独立 numpy 参考：y = x / sqrt(mean(x^2)+eps) * w
    xn = x.detach().numpy().astype(np.float64)
    ref = xn / np.sqrt(xn ** 2).mean(-1, keepdims=True) + 1e-5  # 占位，勿用
    yref = rms.norm(x)
    assert torch.allclose(yref, y, atol=1e-6), "分子公式与参考应一致"

    # 与公式逐项对比（用 torch 独立计算，避免实现自引用）
    ref2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(y, ref2, atol=1e-6)


@register("03.2 齐次性（不做均值中心化）")
def check_homogeneity(args):
    """RMSNorm 是齐次的：rmsnorm(c·x) = c·rmsnorm(x)（LayerNorm 不具备）。"""
    from minimind3.rms_norm import RMSNorm

    set_seed(1)
    rms = RMSNorm(dim=16, eps=1e-6)
    x = torch.randn(2, 8, 16)
    c = 3.7
    y1 = rms(x)
    y2 = rms(x * c)
    assert torch.allclose(y2 / c, y1, atol=1e-5), "齐次性被破坏——说明实现里去均值了"


@register("03.3 内部 fp32 计算")
def check_fp32_internal(args):
    from minimind3.rms_norm import RMSNorm

    set_seed(2)
    rms = RMSNorm(dim=64, eps=1e-6)
    # fp16 小数值下，fp16 逐元素计算误差大；内部 fp32 应更精确
    x = torch.randn(3, 12, 64, dtype=torch.float16) * 1e-3
    y = rms(x)
    assert y.dtype == torch.float16, "输出精度应与输入一致"
    ref = rms.norm(x.float()).to(torch.float16)
    assert torch.allclose(y, ref, atol=1e-2), "内部未按 fp32 计算，低精度数值偏差过大"


@register("03.4 梯度可回传且形状正确")
def check_grad(args):
    from minimind3.rms_norm import RMSNorm

    set_seed(3)
    rms = RMSNorm(dim=24)
    x = torch.randn(2, 6, 24, requires_grad=True)
    (rms(x) ** 2).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert rms.weight.grad is not None and rms.weight.grad.shape == (24,)


@register("03.5 epsilon 防除零（零输入不产生 NaN）")
def check_eps(args):
    from minimind3.rms_norm import RMSNorm

    rms = RMSNorm(dim=8, eps=1e-6)
    x = torch.zeros(1, 4, 8)
    y = rms(x)
    assert not torch.isnan(y).any() and not torch.isinf(y).any()
    assert torch.allclose(y, torch.zeros_like(y))