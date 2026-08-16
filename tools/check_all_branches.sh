#!/usr/bin/env bash
# minimind3 全链验收：依次 checkout 每个教程分支并运行 verify.py（默认 --fast），
# 汇总绿/红表。用法：tools/check_all_branches.sh [--full]
set -euo pipefail

MODE="${1:---fast}"
[[ "$MODE" != "--fast" && "$MODE" != "--full" ]] && { echo "用法: $0 [--fast|--full]"; exit 2; }

PY="${PY:-.venv/Scripts/python.exe}"
[ -x "$PY" ] || [ -f "$PY" ] || PY=python

BRANCHES=$(git for-each-ref --format='%(refname:short)' refs/heads/tutorial/ | sort -V)
ORIG=$(git branch --show-current)
[ -n "$ORIG" ] || ORIG=master

echo "== minimind3 全链验收（$MODE，共 $(echo "$BRANCHES" | wc -l) 个分支）=="
overall=0
while read -r b; do
    echo ""
    echo "===== $b ====="
    git checkout -q "$b"
    if ! "$PY" verify.py "$MODE"; then
        echo "!!!! FAIL: $b"
        overall=1
    else
        echo "---- ok: $b"
    fi
done <<< "$BRANCHES"

git checkout -q "$ORIG"
echo ""
if [ "$overall" -eq 0 ]; then
    echo "== 全链验收：ALL GREEN =="
else
    echo "== 全链验收：存在失败分支 =="
fi
exit $overall