"""minimind3 配置类（镜像 minimind 的 MiniMindConfig，手写实现）。

MiniMindConfig 是 transformers.PretrainedConfig 的子类，负责：
- 集中管理所有超参数与默认值；
- 序列化（保存/加载 config.json）；
- 抽象的"模型蓝图"——任何地方 new MiniMindForCausalLM(config) 都从它出发。

注：model_type 用 "minimind3" 以避免与本仓库标准实现的 "minimind" 混淆。
"""
import math

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind3"

    def __init__(self, **kwargs):
        # ---- 核心结构 ----
        self.hidden_size = kwargs.get("hidden_size", 768)                 # d_model
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 8)       # 层数
        self.dropout = kwargs.get("dropout", 0.0)

        # ---- 词表与特殊 token ----
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # ---- 注意力 ----
        self.flash_attn = kwargs.get("flash_attn", True)                  # 是否走 SDPA 快路径
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)   # GQA：KV 头数
        # head_dim 显式化，便于 flash_attn 兼容（minimind 直接取 hidden//heads）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)

        # ---- MLP ----
        self.hidden_act = kwargs.get("hidden_act", "silu")
        # minimind 的特色取整：intermediate = ceil(hidden*pi/64)*64
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(self.hidden_size * math.pi / 64) * 64
        )

        # ---- 归一化与位置编码 ----
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # ---- 权重绑定 ----
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # ---- 长上下文（可选 YaRN 缩放，默认关闭） ----
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

        # ---- MoE 相关（use_moe=False 时忽略；保留字段以对齐标准实现） ----
        self.use_moe = kwargs.get("use_moe", False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

        super().__init__(**kwargs)

    # ---------------------------------------------------------------- 参数估算
    def estimate_parameter_count(self) -> int:
        """按结构公式估算参数量（仅教学用；实际以模型 .parameters() 为准）。

        每层 = 注意力(Q,K,V,O 投影) + MLP(gate,up,down 投影)；
        其余 = Embedding 表(+未绑定的 lm_head)。
        """
        h = self.hidden_size
        kv = self.num_key_value_heads
        heads = self.num_attention_heads
        d = self.head_dim
        attn = (h * heads * d) + 2 * (h * kv * d) + (heads * d * h)  # q,k,v,o
        inter = self.intermediate_size
        mlp = 2 * (h * inter) + (inter * h)                          # gate,up,down
        per_layer = attn + mlp
        emb = self.vocab_size * h
        head = 0 if self.tie_word_embeddings else self.vocab_size * h
        return per_layer * self.num_hidden_layers + emb + head