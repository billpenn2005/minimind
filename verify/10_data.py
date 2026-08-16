"""第 10 课验收：分词器封装 + Pretrain/SFT 数据集。"""
import subprocess
import sys
import tempfile
from pathlib import Path

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


@register("10.1 分词器加载与特殊 token")
def check_tokenizer(args):
    from minimind3.tokenizer_utils import IM_END_ID, IM_START_ID, load_tokenizer

    tok = load_tokenizer()
    assert tok.vocab_size == 6400
    assert tok.bos_token_id == IM_START_ID and tok.eos_token_id == IM_END_ID
    # 特殊 token 必须被识别为单个 id
    assert tok("<|im_start|>", add_special_tokens=False).input_ids == [1]
    assert tok("<|im_end|>", add_special_tokens=False).input_ids == [2]


@register("10.2 encode/decode round-trip")
def check_roundtrip(args):
    from minimind3.tokenizer_utils import load_tokenizer

    tok = load_tokenizer()
    text = "猫会跑。"
    ids = tok(text, add_special_tokens=False).input_ids
    assert tok.decode(ids) == text
    assert ids[0] not in (1, 2), "普通文本不应以特殊 token 开头"


@register("10.3 ChatML 模板与掩码：仅 assistant 段有效")
def check_sft_mask(args):
    from minimind3.tokenizer_utils import build_chat_prompt, encode_chat, generate_labels, load_tokenizer

    tok = load_tokenizer()
    convos = [
        {"role": "user", "content": "什么是猫？"},
        {"role": "assistant", "content": "猫是一种会抓老鼠的动物。"},
        {"role": "user", "content": "狗呢？"},
        {"role": "assistant", "content": "狗是人类忠诚的朋友。"},
    ]
    ids = encode_chat(convos, tok, max_length=256)
    labels = generate_labels(ids, tok)
    assert len(ids) == len(labels)

    # 用户内容区域必须 -100（用完整子序列扫描，避免首个 token 重复造成误判）
    user_ids = tok("什么是猫？", add_special_tokens=False).input_ids
    found_user = False
    for k in range(len(ids) - len(user_ids) + 1):
        if ids[k:k + len(user_ids)] == user_ids:
            assert all(l == -100 for l in labels[k:k + len(user_ids)]), "user 片段被标记为有效标签"
            found_user = True
    assert found_user
    # assistant 内容区域必须被标记（用完整子序列扫描）
    asst_ids = tok("猫是一种会抓老鼠的动物。", add_special_tokens=False).input_ids
    found_asst = False
    for k in range(len(ids) - len(asst_ids) + 1):
        if ids[k:k + len(asst_ids)] == asst_ids:
            assert labels[k:k + len(asst_ids)] == asst_ids, "assistant 片段未被标记为有效标签"
            found_asst = True
    assert found_asst
    # 掩码比：有效比例在合理区间（总长度中 assistant 占一部分）
    ratio = sum(1 for l in labels if l != -100) / len(labels)
    assert 0.1 < ratio < 0.9


@register("10.4 PretrainDataset 形状与 pad 标签")
def check_pretrain_ds(args):
    from minimind3.datasets import PretrainDataset
    from minimind3.tokenizer_utils import load_tokenizer

    tok = load_tokenizer()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pretrain.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for t in ("猫跑在草地。", "狗跳在天空。", "鸟飞在水里。"):
                f.write('{"text": "%s"}' % t + "\n")
        ds = PretrainDataset(str(p), tok, max_length=16)
        assert len(ds) == 3
        input_ids, labels = ds[0]
        assert input_ids.shape == (16,) and labels.shape == (16,)
        assert input_ids[0] == tok.bos_token_id
        # 文本结尾后是 pad，pad 标签为 -100
        pad_mask = input_ids == tok.pad_token_id
        assert pad_mask.any(), "样本应有 padding"
        assert (labels[pad_mask] == -100).all()
        # 非 pad 位置标签 == input_ids（预训练全监督）
        assert torch.equal(labels[~pad_mask], input_ids[~pad_mask])
        # 前向可用（极小模型）
        from minimind3.config import MiniMindConfig
        from minimind3.causal_lm import MiniMindForCausalLM

        cfg = MiniMindConfig(vocab_size=6400, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, intermediate_size=64)
        model = MiniMindForCausalLM(cfg).eval()
        out = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
        assert out.loss is not None and torch.isfinite(out.loss)


@register("10.5 SFTDataset 与合成数据打通")
def check_sft_ds(args):
    from minimind3.datasets import SFTDataset
    from minimind3.tokenizer_utils import load_tokenizer

    tok = load_tokenizer()
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools/make_synthetic_data.py"),
             "--out", str(data_dir), "--num", "10", "--seed", "0"],
            check=True, capture_output=True,
        )
        ds = SFTDataset(str(data_dir / "tiny_sft.jsonl"), tok, max_length=64)
        for i in range(len(ds)):
            input_ids, labels = ds[i]
            assert input_ids.shape == (64,)
            n_valid = (labels != -100).sum().item()
            assert n_valid > 0, f"样本 {i} 无有效标签"
            # 有效标签与 input_ids 完全一致（掩码只是位置标记）
            assert torch.equal(labels[labels != -100], input_ids[labels != -100])