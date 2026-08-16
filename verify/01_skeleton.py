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


@register("01.1 项目骨架文件齐全（含参考答案就位）")
def check_skeleton(args):
    # 分支自带（快照/工具/验收框架）
    for rel in [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "tools/make_synthetic_data.py",
        "verify.py",
        "verify/_common.py",
        "verify/01_skeleton.py",
        "AGENTS.md",
        "tutorial/01-skeleton/README.md",
        # 参考答案：完整实现 + 规范数据（先写后对；绝不外借到工作区）
        "answers/minimind3/__init__.py",
        "answers/data/tiny_pretrain.jsonl",
        "answers/data/tiny_sft.jsonl",
        # 本课产物：包入口（学习者手写）
        "minimind3/__init__.py",
    ]:
        assert (REPO_ROOT / rel).exists(), f"缺少文件: {rel}"


@register("01.x 按课留白：minimind3/ 只允许存在 ≤ 当前课的产物文件（防跳课/防回拷答案）")
def check_blank_start(args):
    # 每课新增的文件（第 01 课只有包入口）
    _FILES_BY_LESSON = {
        2: {"config.py"},
        3: {"rms_norm.py"},
        4: {"rope.py"},
        5: {"attention.py"},
        6: {"feed_forward.py"},
        7: {"block.py"},
        8: {"model_body.py"},
        9: {"causal_lm.py"},
        10: {"tokenizer_utils.py", "datasets.py"},
        11: {"train_utils.py", "train_pretrain.py"},
        12: {"train_full_sft.py"},
        13: {"convert.py"},
        14: {"e2e.py"},
    }
    limit = getattr(args, "lesson", None) or 14
    allowed = {"__init__.py"}
    for i in range(1, limit + 1):
        allowed |= _FILES_BY_LESSON.get(i, set())
    pkg = REPO_ROOT / "minimind3"
    if pkg.exists():
        for p in pkg.iterdir():
            if p.name == "__pycache__":
                continue
            if p.is_dir():
                assert p.name == "tokenizer" and limit >= 10, \
                    f"minimind3/ 出现未知目录 {p.name}（第 {limit} 课不应有）"
            else:
                assert p.name in allowed, \
                    f"minimind3/{p.name} 不属于第 {limit} 课（跳课或回拷答案？）"
    # data/ 第 10 课起才生成
    if limit < 10:
        assert not (REPO_ROOT / "data").exists(), "data/ 应留到第 10 课再生成，现在应不存在"


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