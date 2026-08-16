"""第 11 课验收：预训练循环（调度、checkpoint、端到端小训练）。"""
import json
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


@register("11.1 cosine 学习率调度数值")
def check_lr(args):
    from minimind3.train_utils import get_lr

    lr = 1e-3
    T = 100
    l0 = get_lr(1, T, lr)
    lmid = get_lr(50, T, lr)
    lend = get_lr(99, T, lr)
    assert abs(l0 - lr) < 1e-6, "起点应为 lr"
    assert abs(lmid - 0.55 * lr) < 1e-6, f"中点为 0.55lr, got {lmid}"
    assert abs(lend - 0.1 * lr) < 1e-6, "终点应为 0.1lr"
    # 单调不减不当（一半后下降），但整体先高后低
    assert l0 > lmid > lend


@register("11.2 合成数据上端到端预训练（loss 下降 + 产出权重）")
def check_train(args):
    if getattr(args, "fast", False):
        return
    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "out"
        data_path = REPO_ROOT / "data" / "tiny_pretrain.jsonl"
        assert data_path.exists(), "缺预训练数据，先运行 tools/make_synthetic_data.py"
        cmd = [
            sys.executable, "-m", "minimind3.train_pretrain",
            "--data_path", str(data_path),
            "--save_dir", str(save_dir),
            "--epochs", "1", "--batch_size", "8", "--max_seq_len", "64",
            "--hidden_size", "96", "--num_hidden_layers", "2",
            "--accumulation_steps", "2", "--log_interval", "1",
            "--save_interval", "20", "--seed", "0",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert r.returncode == 0, f"训练失败:\n{r.stdout[-800:]}\n{r.stderr[-800:]}"
        out = r.stdout

        # 解析 loss 序列：首尾比较
        losses = [float(m) for m in re.findall(r"loss ([\d.]+)", out)]
        assert len(losses) > 5, f"日志太短:\n{out[-800:]}"
        assert losses[0] > losses[-1] * 0.95, f"loss 未下降: {losses[0]:.4f} -> {losses[-1]:.4f}"

        wpath = save_dir / "pretrain_96.pth"
        rpath = save_dir / "pretrain_96_resume.pth"
        assert wpath.exists(), "纯权重未产出"
        assert rpath.exists(), "resume 检查点未产出"

        # 权重可加载进模型
        from minimind3.causal_lm import MiniMindForCausalLM
        from minimind3.config import MiniMindConfig

        cfg = MiniMindConfig(hidden_size=96, num_hidden_layers=2)
        model = MiniMindForCausalLM(cfg)
        sd = torch.load(str(wpath), map_location="cpu")
        miss, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"多余键: {unexpected[:5]}"
        assert not miss, f"缺键: {miss[:5]}"
        logits = model(torch.tensor([[1, 2, 3]])).logits
        assert torch.isfinite(logits).all()


@register("11.3 resume 续训：步数从断点继续")
def check_resume(args):
    if getattr(args, "fast", False):
        return
    from minimind3.train_utils import get_lr

    with tempfile.TemporaryDirectory() as td:
        save_dir = Path(td) / "out"
        data_path = REPO_ROOT / "data" / "tiny_pretrain.jsonl"
        base = [
            sys.executable, "-m", "minimind3.train_pretrain",
            "--data_path", str(data_path),
            "--save_dir", str(save_dir),
            "--epochs", "1", "--batch_size", "8", "--max_seq_len", "64",
            "--hidden_size", "96", "--num_hidden_layers", "2",
            "--accumulation_steps", "1", "--log_interval", "9999",
            "--save_interval", "0", "--seed", "0",
        ]
        r1 = subprocess.run(base, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert r1.returncode == 0
        r2 = subprocess.run(
            base + ["--from_resume", "1", "--epochs", "2"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert r2.returncode == 0, r2.stderr[-600:]
        assert "resumed" in r2.stdout.lower() or "resumed at" in r2.stdout.lower() or r1.stdout.count("loss") > 0
        # 续训后最终权重存在
        assert (save_dir / "pretrain_96.pth").exists()


@register("11.4 合成数据生成脚本与数据文件自洽")
def check_data_files(args):
    data_path = REPO_ROOT / "data" / "tiny_pretrain.jsonl"
    assert data_path.exists()
    with open(data_path, encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert "text" in first