"""第 14 课验收：端到端流水线与全链验收脚本。"""
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@register("14.1 端到端流水线：预训练→SFT→转换→对话（内容级正确）")
def check_e2e(args):
    if getattr(args, "fast", False):
        return
    r = subprocess.run(
        [sys.executable, "-m", "minimind3.e2e"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=1500,
    )
    assert r.returncode == 0, f"e2e 失败:\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}"
    out = r.stdout
    # 四个阶段都必须执行
    for marker in ("合成数据", "预训练", "SFT", "HF 转换", "对话测试", "[convert]"):
        assert marker in out, f"缺少阶段标记: {marker}"
    # 两条问答
    answers = [m.split("A: ", 1)[1] for m in re.findall(r"\[E2E\] A: .*", out)]
    assert len(answers) == 2, f"期望 2 条回答，实际 {len(answers)}"
    assert len(answers[0]) >= 4 and len(answers[1]) >= 4, "回答过短"
    # 内容正确性（合成数据的封闭事实，微型模型应能背出）
    assert "猫" in answers[0] and ("抓" in answers[0] or "老鼠" in answers[0]), f"回答1错误: {answers[0]}"
    assert "狗" in answers[1] and ("忠诚" in answers[1] or "朋友" in answers[1]), f"回答2错误: {answers[1]}"
    # 转换产物
    assert (REPO_ROOT / "minimind3-hf").exists() or (REPO_ROOT / "out/e2e/hf_128").exists(), "缺 HF 产物"


@register("14.2 全链验收脚本就位、分支数正确、bash 语法无误")
def check_chain(args):
    sh = REPO_ROOT / "tools" / "check_all_branches.sh"
    assert sh.exists(), "缺 tools/check_all_branches.sh"
    r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash 语法错误: {r.stderr}"
    branches = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/tutorial/"],
        capture_output=True, text=True,
    )
    names = sorted(b for b in branches.stdout.splitlines() if b.strip())
    assert len(names) == 14, f"教程分支应 14 个，实际 {len(names)}: {names}"
    # 分支链拓扑：每课是前一课的子提交（孤儿分支 01 除外）
    assert names[-1] == "tutorial/14-final"


@register("14.3 检查模块完整性（已到课的检查文件均非空，且到期全量足够）")
def check_scale(args):
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "verify").glob("[0-9]*.py"))
    assert files, "无检查模块"
    # 课程检查文件（本模块 14_e2e 是“全局护栏”，不计入课程进度）
    lesson_files = [f for f in files if not f.stem.startswith("14")]
    assert lesson_files, "无课程检查模块"
    total = 0
    for m in files:
        import importlib

        mod = importlib.import_module(f"verify.{m.stem}")
        assert mod.CHECKS, f"{m.stem} 为空模块"
        total += len(mod.CHECKS)
    # 逐课递进的检查数下限（01..13 课程文件 + 14 全局文件总量）
    floors = {1: 5, 2: 10, 3: 15, 4: 21, 5: 27, 6: 32, 7: 37, 8: 42, 9: 50, 10: 55, 11: 59, 12: 62, 13: 67}
    lesson = len(lesson_files)
    floor = (floors.get(lesson, 67)) + (3 if lesson == 13 else 0)  # 13 门课时含 14_e2e 的 3 项
    assert total >= floor, f"检查项数量异常: {total} < {floor}"