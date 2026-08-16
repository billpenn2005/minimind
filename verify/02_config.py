"""第 02 课验收：MiniMindConfig。"""
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from verify._common import TINY_CONFIG, set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@register("02.1 默认值与 minimind 标准实现一致")
def check_defaults(args):
    from minimind3.config import MiniMindConfig

    cfg = MiniMindConfig()
    assert cfg.hidden_size == 768
    assert cfg.num_hidden_layers == 8
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 768 // 8 == 96
    assert cfg.vocab_size == 6400
    assert cfg.bos_token_id == 1 and cfg.eos_token_id == 2
    assert cfg.hidden_act == "silu"
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.rope_theta == 1e6
    assert cfg.tie_word_embeddings is True
    assert cfg.max_position_embeddings == 32768
    # minimind 特色取整：ceil(768*pi/64)*64 == 2432
    assert cfg.intermediate_size == math.ceil(768 * math.pi / 64) * 64 == 2432
    assert cfg.use_moe is False
    assert cfg.rope_scaling is None
    assert cfg.flash_attn is True


@register("02.2 微型配置（验收用）可构造且字段推导正确")
def check_tiny(args):
    from minimind3.config import MiniMindConfig

    cfg = MiniMindConfig(**TINY_CONFIG)
    assert cfg.hidden_size == 96
    assert cfg.head_dim == 96 // 4 == 24
    assert cfg.intermediate_size == TINY_CONFIG["intermediate_size"]
    assert cfg.num_key_value_heads == 2
    assert cfg.vocab_size == 512


@register("02.3 JSON 序列化 round-trip")
def check_json_roundtrip(args):
    import tempfile

    from transformers import PretrainedConfig

    from minimind3.config import MiniMindConfig

    cfg = MiniMindConfig(**TINY_CONFIG)
    d1 = cfg.to_dict()
    # 经 JSON 字符串再加载
    cfg2 = MiniMindConfig.from_dict(cfg.to_diff_dict())
    d2 = cfg2.to_dict()
    for k in d1:
        assert d1[k] == d2[k], f"字段 {k} round-trip 不一致: {d1[k]} != {d2[k]}"

    with tempfile.TemporaryDirectory() as td:
        cfg.save_pretrained(td)
        loaded = PretrainedConfig.from_pretrained(td)
        assert loaded.model_type == "minimind3"
        assert loaded.hidden_size == 96


@register("02.4 参数量估算公式正确")
def check_param_est(args):
    from minimind3.config import MiniMindConfig

    # 计算出的三层结构：head_dim*heads == hidden
    cfg = MiniMindConfig(**TINY_CONFIG)
    n = cfg.estimate_parameter_count()
    h, v = cfg.hidden_size, cfg.vocab_size
    heads, kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    per_layer = (2 * h * heads * d + 2 * h * kv * d) + 3 * h * cfg.intermediate_size
    assert n == per_layer * cfg.num_hidden_layers + v * h
    assert n > 0

    # 不绑定时 lm_head 计入
    cfg2 = MiniMindConfig(**{**TINY_CONFIG, "tie_word_embeddings": False})
    assert cfg2.estimate_parameter_count() == n + v * h


@register("02.5 YaRN 缩放开关与要素")
def check_yarn(args):
    from minimind3.config import MiniMindConfig

    cfg = MiniMindConfig(inference_rope_scaling=True)
    rs = cfg.rope_scaling
    assert rs is not None and rs["type"] == "yarn"
    for k in ("factor", "beta_fast", "beta_slow", "original_max_position_embeddings", "attention_factor"):
        assert k in rs