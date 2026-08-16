"""minimind3 RMSNorm（镜像 minimind 实现，手写）。

RMSNorm（Root Mean Square Layer Normalization）只做"除以均方根"的缩放，
不做均值中心化，比 LayerNorm 少了均值统计，计算更省且对 LLM 训练足够稳定。
注意内部用 fp32 计算，避免低精度下的均方根下溢/溢出。
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学缩放，初值 1

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        # 均方根：mean(x^2) 后取平方根倒数（rsqrt 比 1/sqrt 更快更稳）
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 内部 fp32 计算，输出回到输入精度
        return (self.weight * self.norm(x.float())).type_as(x)