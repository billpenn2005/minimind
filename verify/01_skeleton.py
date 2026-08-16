"""第 01 课验收：项目骨架、合成数据生成器、验收框架自身。"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@register("01.1 项目骨架文件齐全")
def check_skeleton(args):
    for rel in [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "minimind3/__init__.py",
        "tools/make_synthetic_data.py",
        "verify.py",
        "verify/_common.py",
        "verify/01_skeleton.py",
        "AGENTS.md",
        "tutorial/01-skeleton/README.md",
    ]:
        assert (REPO_ROOT / rel).exists(), f"缺少文件: {rel}"


@register("01.2 合成预训练数据可生成且可复现")
def check_synth_pretrain(args):
    from tools.make_synthetic_data import make_pretrain_corpus

    import random

    rows = [{"text": t} for t in make_pretrain_corpus(50, random.Random(7))]
    assert len(rows) == 50
    for r in rows:
        assert isinstance(r["text"], str) and len(r["text"]) > 0
    rows2 = [{"text": t} for t in make_pretrain_corpus(50, random.Random(7))]
    assert [r["text"] for r in rows] == [r["text"] for r in rows2], "同种子必须可复现"


@register("01.3 合成 SFT 数据 schema 正确")
def check_synth_sft(args):
    import random

    from tools.make_synthetic_data import make_sft_conversations

    rows = make_sft_conversations(20, random.Random(3))
    for r in rows:
        convs = r["conversations"]
        assert isinstance(convs, list) and len(convs) >= 1
        for m in convs:
            assert set(m) == {"role", "content"} and m["role"] in ("user", "assistant")
            assert isinstance(m["content"], str) and m["content"]


@register("01.4 生成脚本命令行可运行且体积小")
def check_synth_cli(args):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools/make_synthetic_data.py"),
             "--out", td, "--num", "20", "--seed", "0"],
            check=True, capture_output=True,
        )
        for name in ("tiny_pretrain.jsonl", "tiny_sft.jsonl"):
            p = Path(td) / name
            assert p.exists() and p.stat().st_size < 100_000, f"{name} 过大或缺失"
        with open(Path(td) / "tiny_pretrain.jsonl", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 20


@register("01.5 验收框架可发现并运行检查")
def check_harness(args):
    import importlib

    mod = importlib.import_module("verify.01_skeleton")
    assert len(mod.CHECKS) >= 4