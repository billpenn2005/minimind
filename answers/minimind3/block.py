"""minimind3 MiniMindBlock（Pre-Norm 残差解码块，镜像 minimind 实现，手写）。

结构（Pre-LayerNorm 残差）：
    h' = h + Attn(LN1(h))
    h' = h' + FFN(LN2(h'))

残差连接让梯度有"高速公路"直达输入，缓解深层网络梯度消失；
Pre-Norm（先归一再算子层）比 Post-Norm 训练更稳定，是现代 LLM 的标配。
"""
from torch import nn

from .attention import Attention
from .config import MiniMindConfig
from .feed_forward import FeedForward
from .rms_norm import RMSNorm


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = hidden_states + residual

        # Post-Attn 残差（写法：直接 += ，与标准实现一致）
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value