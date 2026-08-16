"""合成微型数据集生成器（无网络、无大文件、可复现）。

生成两类数据（与 minimind 仓库 dataset/ 目录的 jsonl 格式一致）：
- 预训练样本：{"text": "..."}，文本来自封闭小词表模板，便于微型模型快速学会；
- SFT 样本：{"conversations": [{"role": ..., "content": ...}, ...]}。

用法：
    python tools/make_synthetic_data.py --out data --num 200 --seed 0
"""
import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------- 预训练模板
_NOUNS = ["猫", "狗", "鸟", "鱼", "马", "牛"]
_VERBS = ["跑", "跳", "飞", "游", "叫", "吃"]
_PLACES = ["草地", "天空", "水里", "山上", "田野", "树下"]
_ADJS = ["快", "慢", "高", "低", "轻", "重"]

_PRETRAIN_TEMPLATES = [
    "{noun}{verb}在{place}。",
    "一只{noun}在{place}{verb}。",
    "{adj}的{noun}正在{verb}。",
    "今天的{noun}{verb}得很{adj}。",
    "请问：{noun}会{verb}吗？答案是{noun}会{verb}。",
]


def make_pretrain_corpus(num: int, rng: random.Random) -> list[str]:
    """用封闭小词表生成 num 条可重复、可记忆的文本（保证微型模型能快速降低 loss）。"""
    texts = []
    for _ in range(num):
        tpl = rng.choice(_PRETRAIN_TEMPLATES)
        texts.append(
            tpl.format(
                noun=rng.choice(_NOUNS),
                verb=rng.choice(_VERBS),
                place=rng.choice(_PLACES),
                adj=rng.choice(_ADJS),
            )
        )
    return texts


# ---------------------------------------------------------------- SFT 模板
_FACTS = {
    "猫": "猫是一种会抓老鼠的动物。",
    "狗": "狗是人类忠诚的朋友。",
    "鸟": "鸟有翅膀，可以在天空飞翔。",
    "鱼": "鱼生活在水中，用鳃呼吸。",
    "马": "马擅长奔跑，可以载人。",
    "牛": "牛吃草，能产奶。",
}
_QA_TEMPLATES = [
    ("什么是{noun}？", "{fact}"),
    ("介绍一下{noun}。", "{fact}"),
    ("请用一个句子描述{noun}。", "{fact}"),
]


def make_sft_conversations(num: int, rng: random.Random) -> list[dict]:
    """生成 num 条 QA 对话（assistant 回答来自封闭事实表，便于 SFT 快速见效）。"""
    convos = []
    for _ in range(num):
        noun = rng.choice(list(_FACTS))
        q, a = rng.choice(_QA_TEMPLATES)
        convos.append(
            {
                "conversations": [
                    {"role": "user", "content": q.format(noun=noun)},
                    {"role": "assistant", "content": a.format(fact=_FACTS[noun])},
                ]
            }
        )
    return convos


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="data", help="输出目录")
    ap.add_argument("--num", type=int, default=200, help="每个数据集样本数")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（保证可复现）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pretrain = [{"text": t} for t in make_pretrain_corpus(args.num, rng)]
    sft = make_sft_conversations(args.num, rng)

    write_jsonl(out / "tiny_pretrain.jsonl", pretrain)
    write_jsonl(out / "tiny_sft.jsonl", sft)

    for p in (out / "tiny_pretrain.jsonl", out / "tiny_sft.jsonl"):
        print(f"{p}: {p.stat().st_size} bytes")
    print(f"done: {len(pretrain)} pretrain / {len(sft)} sft samples -> {out}")


if __name__ == "__main__":
    main()