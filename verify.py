#!/usr/bin/env python
"""MiniMind3 验收入口。

加载 verify/ 目录下所有检查模块（逐课累积，每课一个文件），
按文件名顺序执行，任一失败则退出码非 0。

用法：
    python verify.py            # 全部检查
    python verify.py --fast     # 跳过耗时训练类检查
"""
import argparse
import importlib
import pathlib
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="跳过耗时检查（训练/生成类）")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    checks = []
    for mod_path in sorted((root / "verify").glob("[0-9]*.py")):
        mod = importlib.import_module(f"verify.{mod_path.stem}")
        checks.extend(mod.CHECKS)

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
    print(f"\n== minimind3 verify: {passed} passed, {failed} failed ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())