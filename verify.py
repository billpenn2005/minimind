#!/usr/bin/env python
"""MiniMind3 验收入口。

单分支教学模式下，验收按课进行：`verify.py NN` 只运行第 01~NN 课的检查模块
（每课一个 verify/NN_xxx.py，逐课累积）；不带参数则运行全部（毕业验收）。

用法：
    python verify.py                 # 全部检查（毕业验收 / 参考答案自检）
    python verify.py --fast          # 全部，但跳过耗时训练类检查
    python verify.py 05              # 只验收第 01~05 课
    python verify.py 05 --fast       # 第 01~05 课，跳过耗时检查

任一检查失败则退出码非 0。
"""
import argparse
import importlib
import pathlib
import re
import sys
import time


def _lesson_no(stem: str) -> int | None:
    m = re.match(r"^(\d+)_", stem)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="跳过耗时检查（训练/生成类）")
    ap.add_argument(
        "lesson", nargs="?", type=int, default=None, metavar="NN",
        help="只验收第 01~NN 课（缺省 = 全部）",
    )
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    checks: list = []
    skipped = 0
    for mod_path in sorted((root / "verify").glob("[0-9]*.py")):
        no = _lesson_no(mod_path.stem)
        if no is None:
            continue
        if args.lesson is not None and no > args.lesson:
            skipped += 1
            continue
        try:
            mod = importlib.import_module(f"verify.{mod_path.stem}")
        except Exception as exc:  # 未完成当课手写时，友好提示而不是崩溃
            checks.append((
                f"verify.{mod_path.stem} 导入失败（第 {no:02d} 课可能尚未完成）",
                (lambda e: (lambda a: (_ for _ in ()).throw(e)))(exc),
            ))
            continue
        checks.extend(mod.CHECKS)

    if skipped:
        print(f"（按第 {args.lesson:02d} 课过滤，已跳过 {skipped} 个后续检查模块）")

    passed = failed = 0
    for name, fn in checks:
        t0 = time.time()
        try:
            fn(args)
            print(f"  [PASS] {name}  ({time.time() - t0:.2f}s)")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            failed += 1
    scope = f"（第 01~{args.lesson:02d} 课）" if args.lesson is not None else "（全部 14 课）"
    print(f"\n== minimind3 verify [{scope}]: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())