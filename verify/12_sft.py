"""第 12 课验收：全参 SFT（掩码损失 + 内容对齐 + 权重产出）。"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@register("12.1 基于预训练权重的 SFT 端到端训练（loss 下降）")
def check_sft_train(args):
    if getattr(args, "fast", False):
        return
    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "out"
        # 先快速预训练出基础权重
        pt = subprocess.run(
            [sys.executable, "-m", "minimind3.train_pretrain",
             "--data_path", str(REPO_ROOT / "data/tiny_pretrain.jsonl"),
             "--save_dir", str(save_dir), "--epochs", "1", "--batch_size", "8",
             "--max_seq_len", "64", "--hidden_size", "96", "--num_hidden_layers", "2",
             "--accumulation_steps", "1", "--log_interval", "9999", "--save_interval", "0", "--seed", "0"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert pt.returncode == 0, pt.stderr[-400:]

        sft = subprocess.run(
            [sys.executable, "-m", "minimind3.train_full_sft",
             "--data_path", str(REPO_ROOT / "data/tiny_sft.jsonl"),
             "--save_dir", str(save_dir), "--from_weight", "pretrain",
             "--epochs", "1", "--batch_size", "8", "--max_seq_len", "96",
             "--hidden_size", "96", "--num_hidden_layers", "2",
             "--accumulation_steps", "2", "--log_interval", "1",
             "--save_interval", "0", "--seed", "0", "--learning_rate", "2e-4"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert sft.returncode == 0, f"SFT 失败:\n{sft.stdout[-600:]}\n{sft.stderr[-600:]}"
        losses = [float(m) for m in re.findall(r"loss ([\d.]+)", sft.stdout)]
        assert len(losses) > 5, f"日志太短:\n{sft.stdout[-600:]}"
        assert losses[-1] < losses[0] * 0.9, f"SFT loss 未下降: {losses[0]:.4f} -> {losses[-1]:.4f}"
        assert (save_dir / "full_sft_96.pth").exists(), "SFT 权重未产出"
        return save_dir


@register("12.2 掩码有效性：SFT 后正确回答的困惑度低于错误回答")
def check_content(args):
    """关键教学点：如果掩码做错（比如 user 也参与损失），SFT 会退化成背书机，
    正确/错误答案的 CE 差距不会拉开。"""
    if getattr(args, "fast", False):
        return
    from minimind3.causal_lm import MiniMindForCausalLM
    from minimind3.config import MiniMindConfig
    from minimind3.tokenizer_utils import encode_chat, load_tokenizer

    hidden, layers = 128, 2
    cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers)
    model = MiniMindForCausalLM(cfg).eval()
    tok = load_tokenizer()

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "out"
        subprocess.run(
            [sys.executable, "-m", "minimind3.train_pretrain",
             "--data_path", str(REPO_ROOT / "data/tiny_pretrain.jsonl"),
             "--save_dir", str(save_dir), "--epochs", "1", "--batch_size", "8",
             "--max_seq_len", "64", "--hidden_size", str(hidden), "--num_hidden_layers", str(layers),
             "--accumulation_steps", "1", "--log_interval", "9999", "--save_interval", "0", "--seed", "0"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        subprocess.run(
            [sys.executable, "-m", "minimind3.train_full_sft",
             "--data_path", str(REPO_ROOT / "data/tiny_sft.jsonl"),
             "--save_dir", str(save_dir), "--from_weight", "pretrain",
             "--epochs", "4", "--batch_size", "8", "--max_seq_len", "96",
             "--hidden_size", str(hidden), "--num_hidden_layers", str(layers),
             "--accumulation_steps", "1", "--log_interval", "9999",
             "--save_interval", "0", "--seed", "0", "--learning_rate", "5e-4"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        sd = torch.load(str(save_dir / f"full_sft_{hidden}.pth"), map_location="cpu")
        model.load_state_dict(sd, strict=False)

    def ce_of(question, answer):
        ids = encode_chat(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tok, max_length=128,
        )
        inp = torch.tensor([ids])
        with torch.no_grad():
            logits = model(inp).logits[0]
        # 错位：logits[t] 对 answer 段内 tokens
        a_ids = tok(answer, add_special_tokens=False).input_ids
        # 找到 assistant 内容的起点：查模板标记
        marker = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        for k in range(len(ids) - 1, -1, -1):
            if ids[k:k + len(marker)] == marker:
                start = k + len(marker)
                break
        target = ids[start:start + len(a_ids)]
        pred_logits = logits[start - 1:start - 1 + len(target)]
        return torch.nn.functional.cross_entropy(pred_logits, torch.tensor(target)).item()

    good = ce_of("什么是猫？", "猫是一种会抓老鼠的动物。")
    bad = ce_of("什么是猫？", "牛吃草，能产奶。")
    assert good < bad - 0.1, f"正确回答 CE({good:.3f}) 应显著低于错误回答 CE({bad:.3f})（掩码/训练无效）"


@register("12.3 权重可加载且前向正常")
def check_load(args):
    if getattr(args, "fast", False):
        return
    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "out"
        subprocess.run(
            [sys.executable, "-m", "minimind3.train_full_sft",
             "--data_path", str(REPO_ROOT / "data/tiny_sft.jsonl"),
             "--save_dir", str(save_dir), "--from_weight", "none",
             "--epochs", "1", "--batch_size", "8", "--max_seq_len", "96",
             "--hidden_size", "96", "--num_hidden_layers", "2",
             "--accumulation_steps", "1", "--log_interval", "9999",
             "--save_interval", "0", "--seed", "0"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        from minimind3.causal_lm import MiniMindForCausalLM
        from minimind3.config import MiniMindConfig

        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=96, num_hidden_layers=2))
        sd = torch.load(str(save_dir / "full_sft_96.pth"), map_location="cpu")
        miss, extra = model.load_state_dict(sd, strict=False)
        assert not extra and not miss
        assert torch.isfinite(model(torch.tensor([[1, 2, 3]])).logits).all()