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


@register("14.2 单分支自洽：参考答案就位、起点留白、按课过滤可用")
def check_chain(args):
    # 参考答案存在（完整实现 + 规范数据）
    assert (REPO_ROOT / "answers/minimind3/__init__.py").exists()
    assert (REPO_ROOT / "answers/data/tiny_pretrain.jsonl").exists()
    assert (REPO_ROOT / "answers/data/tiny_sft.jsonl").exists()
    # 全链验收脚本在单分支模式下已移除（由 verify.py 按课过滤替代）
    assert not (REPO_ROOT / "tools/check_all_branches.sh").exists()
    # 按课过滤可运行：verify.py 03 只执行第 01~03 课的检查（跳过 04+ 模块）
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "verify.py"), "03", "--fast"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
    )
    out = r.stdout
    assert "第 01~03 课" in out, f"过滤器作用域声明缺失:\n{out[-300:]}"
    assert "跳过 11 个后续检查模块" in out, f"未跳过 04+ 模块:\n{out[-300:]}"
    assert "04." not in out and "14." not in out, "过滤器仍执行了后续课程检查"
    # 01.x（按课留白）在毕业态属预期失败（工作区已含全部文件），不要求 0 failed


@register("14.3 检查模块完整性（全部 14 课检查模块非空且总量足够）")
def check_scale(args):
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "verify").glob("[0-9]*.py"))
    assert files, "无检查模块"
    total = 0
    for m in files:
        import importlib

        mod = importlib.import_module(f"verify.{m.stem}")
        assert mod.CHECKS, f"{m.stem} 为空模块"
        total += len(mod.CHECKS)
    # 课程锚点：14 门课累计至少 70 项（01..13 课 67 项 + 本模块 3 项）
    assert len(files) >= 14, f"检查模块数不足: {len(files)}"
    assert total >= 70, f"检查项数量异常: {total} < 70"