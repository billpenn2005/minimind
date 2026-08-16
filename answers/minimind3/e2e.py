"""minimind3 端到端流水线：空权重 → 预训练 → SFT → HF 转换 → 对话测试。

全程 CPU、合成数据、微型配置；输出每个阶段的关键信息。
验收（verify/14_e2e.py）将解析本脚本 stdout 的 `[E2E]` 行做质量断言。

用法：
    python -m minimind3.e2e                      # 全部（约 1~2 分钟）
    python -m minimind3.e2e --skip-train         # 跳过训练（复用已有权重，仅转换+对话）
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    if r.returncode != 0:
        sys.exit(f"[e2e] 失败: {' '.join(cmd)}\n{r.stderr[-600:] if r.stderr else ''}")


def stage(name: str) -> None:
    print(f"\n========== [E2E] {name} ==========", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=str, default="out/e2e")
    ap.add_argument("--hidden", type=int, default=128, help="隐藏维度（CPU 演示用微型尺寸）")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pt-epochs", type=int, default=1)
    ap.add_argument("--sft-epochs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--skip-train", action="store_true", help="跳过训练，仅转换+对话")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # 0. 合成数据
    stage("合成数据")
    data_dir = REPO_ROOT / "data"
    if not (data_dir / "tiny_pretrain.jsonl").exists():
        run([py, "-m", "tools.make_synthetic_data", "--out", str(data_dir), "--num", "300"])

    if not args.skip_train:
        # 1. 预训练
        stage(f"预训练 hidden={args.hidden} layers={args.layers} epochs={args.pt_epochs}")
        run([
            py, "-m", "minimind3.train_pretrain",
            "--data_path", str(data_dir / "tiny_pretrain.jsonl"),
            "--save_dir", str(workdir), "--save_weight", "e2e_pt",
            "--epochs", str(args.pt_epochs), "--batch_size", "8",
            "--max_seq_len", str(args.seq), "--hidden_size", str(args.hidden),
            "--num_hidden_layers", str(args.layers), "--accumulation_steps", "2",
            "--log_interval", "25", "--save_interval", "0", "--seed", "0",
        ])

        # 2. SFT（配置对齐验收探针：acc=1、4 epoch 可达内容级收敛）
        stage(f"SFT epochs={args.sft_epochs}")
        run([
            py, "-m", "minimind3.train_full_sft",
            "--data_path", str(data_dir / "tiny_sft.jsonl"),
            "--save_dir", str(workdir), "--save_weight", "e2e_sft",
            "--from_weight", "e2e_pt", "--epochs", str(args.sft_epochs),
            "--batch_size", "8", "--max_seq_len", str(args.seq),
            "--hidden_size", str(args.hidden), "--num_hidden_layers", str(args.layers),
            "--accumulation_steps", "1", "--log_interval", "25",
            "--save_interval", "0", "--seed", "0", "--learning_rate", "5e-4",
        ])
    else:
        stage("跳过训练（--skip-train）")

    # 3. HF 转换
    stage("HF 转换（自包含 remote-code）")
    pth = workdir / f"e2e_sft_{args.hidden}.pth"
    assert pth.exists(), f"缺 SFT 权重: {pth}"
    hf_dir = workdir / f"hf_{args.hidden}"
    run([
        py, "-c",
        "import sys; sys.path.insert(0, '.');"
        f"from minimind3.convert import convert_torch2transformers;"
        f"convert_torch2transformers(r'{pth}', r'{hf_dir}')",
    ])

    # 4. 对话测试（加载转换产物，不 import minimind3 模型类里的手工路径——用 Auto 类）
    stage("对话测试（AutoModelForCausalLM + 本机分词器）")
    _chat_demo(hf_dir, args.hidden)


def _chat_demo(hf_dir: Path, hidden: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    sys.path.insert(0, str(REPO_ROOT))  # 供 remote-code 解析（KeyError: 看 lesson13 讨论）
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))

    from minimind3.tokenizer_utils import build_chat_prompt

    questions = ["什么是猫？", "介绍一下狗。"]
    for q in questions:
        prompt = build_chat_prompt([{"role": "user", "content": q}])
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        out = model.generate(
            torch.tensor([ids]),
            max_new_tokens=24,
            do_sample=False,
            top_k=0,
            top_p=1.0,
        )
        answer = tokenizer.decode(out[0][len(ids):], skip_special_tokens=False)
        answer = answer.replace("<|im_end|>", "").replace("\n", " ").strip()
        print(f"[E2E] Q: {q}")
        print(f"[E2E] A: {answer}")


if __name__ == "__main__":
    main()