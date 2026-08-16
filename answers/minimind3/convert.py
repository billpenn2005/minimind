"""minimind3 HF transformers 格式转换（镜像 minimind scripts/convert_model.py，手写）。

两种产物：
1. **AutoClass 注册版**（对齐 minimind）：save_pretrained 时写入 auto_map，
   在同仓库/同环境内可用 `AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)`；
2. **自包含 remote-code 版**：把全部模型源码生成到产物目录的 modeling_minimind3.py，
   auto_map 指向它——该目录可拷贝到任何机器直接加载（不依赖本仓库）。

另有反向转换 transformers -> torch(.pth)。
"""
import json
import os
import re
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_FILES = [
    "config.py",      # 依赖度最低、且类型注解在类体求值需要它先定义
    "rms_norm.py",
    "rope.py",
    "attention.py",
    "feed_forward.py",
    "block.py",
    "model_body.py",
    "causal_lm.py",
]


# ------------------------------------------------------------------ torch -> HF
def infer_config_from_state_dict(state_dict: dict, sidecar_path=None) -> "MiniMindConfig":
    """从权重反推配置。

    优先：训练脚本随权重写入的 sidecar `*.config.json`
    （携带 rope_theta/max_position_embeddings 等"非张量"超参）；
    兜底：从张量形状推断结构尺寸 + 生态默认超参。
    """
    from .config import MiniMindConfig

    if sidecar_path and os.path.exists(sidecar_path):
        with open(sidecar_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return MiniMindConfig(**d)

    vocab = state_dict["model.embed_tokens.weight"].shape[0]
    hidden = state_dict["model.embed_tokens.weight"].shape[1]
    layers = max(int(k.split(".")[2]) for k in state_dict if k.startswith("model.layers.") and k.endswith(".self_attn.q_proj.weight")) + 1
    q_dim = state_dict["model.layers.0.self_attn.q_proj.weight"].shape[0]
    k_dim = state_dict["model.layers.0.self_attn.k_proj.weight"].shape[0]
    # 约定：hidden = heads * head_dim（q 投影输出维度 == hidden）
    # head_dim 用 q_norm.weight 维度直接读出（最可靠信号），退化时用 gcd 兜底
    if "model.layers.0.self_attn.q_norm.weight" in state_dict:
        head_dim = state_dict["model.layers.0.self_attn.q_norm.weight"].shape[0]
    else:
        head_dim = _gcd(hidden, k_dim)
    num_attention_heads = hidden // head_dim
    num_key_value_heads = k_dim // head_dim
    intermediate = state_dict["model.layers.0.mlp.gate_proj.weight"].shape[0]
    return MiniMindConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        vocab_size=vocab,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate,
        # 无 sidecar 时其余超参取生态默认（rope_theta=1e6 等）
    )


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float32, standalone=True):
    """把训练得到的 .pth 权重转成 HF 目录（含 tokenizer，可含自包含 remote-code）。"""
    from .causal_lm import MiniMindForCausalLM
    from .config import MiniMindConfig
    from .tokenizer_utils import load_tokenizer

    state_dict = torch.load(torch_path, map_location="cpu")
    sidecar = torch_path.replace(".pth", ".config.json")
    lm_config = infer_config_from_state_dict(state_dict, sidecar_path=sidecar)
    lm_config.pad_token_id = 0            # 生态惯例：pad = <|endoftext|>

    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = MiniMindForCausalLM(lm_config)
    miss, extra = model.load_state_dict(state_dict, strict=False)
    assert not miss and not extra, f"strict 加载失败: miss={miss[:3]} extra={extra[:3]}"
    model = model.to(dtype)

    os.makedirs(transformers_path, exist_ok=True)
    model.save_pretrained(transformers_path, safe_serialization=False)  # pytorch_model.bin
    load_tokenizer().save_pretrained(transformers_path)

    if standalone:
        _write_standalone_modeling(transformers_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[convert] {torch_path} -> {transformers_path} ({n_params / 1e6:.2f}M params, {dtype})")
    return transformers_path


def _write_standalone_modeling(out_dir: str) -> None:
    """把 minimind3 全部模型源码拼接成产物目录下的 modeling_minimind3.py。

    做法：按依赖顺序收集源码片段，去掉相对导入（`from .x import ...`），
    使所有类落在同一模块中；再把 config.json 的 auto_map 指向该本地模块。
    结果：目录级 remote-code，任何机器 `trust_remote_code=True` 均可加载。
    """
    header = (
        "# Auto-generated standalone modeling file for minimind3 (remote code).\n"
        "# 由 minimind3/convert.py 生成；与仓库内实现等价。\n"
        "from __future__ import annotations  # 注解懒求值，避免定义顺序敏感\n"
    )
    body = [header]
    for name in _SRC_FILES:
        src = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")
        src = re.sub(r"^from \.\w+ import .*$", "", src, flags=re.M)  # 去掉包内相对导入
        body.append(f"# ============ source: minimind3/{name} ============\n{src}\n")
    (Path(out_dir) / "modeling_minimind3.py").write_text("\n".join(body), encoding="utf-8")

    # 把 auto_map 指向本地模块
    cfg_path = Path(out_dir) / "config.json"
    cfg = _load_json(cfg_path)
    cfg["auto_map"] = {
        "AutoConfig": "modeling_minimind3.MiniMindConfig",
        "AutoModelForCausalLM": "modeling_minimind3.MiniMindForCausalLM",
    }
    _dump_json(cfg_path, cfg)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ HF -> torch
def convert_transformers2torch(transformers_path, torch_path):
    """从 HF 目录加载权重并保存为 .pth（与训练脚本产生的格式一致）。"""
    model = _load_from_hf(transformers_path)
    torch.save({k: v.float().cpu() for k, v in model.state_dict().items()}, torch_path)
    print(f"[convert] {transformers_path} -> {torch_path}")
    return torch_path


def _load_from_hf(transformers_path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)


if __name__ == "__main__":
    # 用法示例：把上一个 SFT 权重转成 HF 目录
    pth = "out/full_sft_96.pth"
    hf_dir = "minimind3-hf/full_sft_96"
    convert_torch2transformers(pth, hf_dir)
    m = _load_from_hf(hf_dir)
    print("loaded:", type(m).__name__)