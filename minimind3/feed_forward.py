"""minimind3 FeedForward（SwiGLU，镜像 minimind 实现，手写）。

SwiGLU：把输入先经门控投影过 SiLU 激活，再与另一个投影逐元素相乘。
LLaMA/Qwen/minimind 等现代模型的标准 MLP 形态。

    FFN(x) = down( silu(gate(x)) * up(x) )

三个投影都是无偏置线性层：gate/up 把 hidden → intermediate，down 把 intermediate → hidden。
"""
from torch import nn

from transformers.activations import ACT2FN

from .config import MiniMindConfig


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))