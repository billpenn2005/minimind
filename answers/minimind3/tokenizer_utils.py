"""minimind3 tokenizer 工具（基于 minimind 仓库自带 BPE 词表，手写封装）。

说明：词表/配置文件来自 minimind 自训练的 6400 词 BPE（随仓库分发，零下载），
本教程**不重新训练词表**（训练分词器留作进阶练习），只做封装与对话模板工具。

对话模板（与 ChatML 一致的简化版，便于手写掩码扫描）：
    <|im_start|>user\n{content}<|im_end|>\n
    <|im_start|>assistant\n{content}<|im_end|>\n
"""
from pathlib import Path

TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"

# ChatML 特殊 token id（由词表约定：0=endoftext, 1=im_start, 2=im_end）
IM_START_ID = 1
IM_END_ID = 2

# 占位：真正加载在 load_tokenizer()
_tokenizer = None


def load_tokenizer():
    """惰性加载预训练 BPE 分词器（缓存单例）。"""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return _tokenizer


def build_chat_prompt(conversations: list[dict]) -> str:
    """把 [{'role','content'}, ...] 渲染为 ChatML 文本（不含 special token 的额外处理）。"""
    parts = []
    for msg in conversations:
        role = msg["role"]
        assert role in ("system", "user", "assistant"), f"未知角色: {role}"
        parts.append(f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n")
    return "".join(parts)


def encode_chat(conversations: list[dict], tokenizer=None, max_length: int | None = None) -> list[int]:
    """对话 -> input_ids（截断到 max_length）。"""
    tok = tokenizer or load_tokenizer()
    text = build_chat_prompt(conversations)
    return tok(text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids


def generate_labels(input_ids: list[int], tokenizer=None) -> list[int]:
    """按 assistant 片段构造损失掩码：仅 assistant 内容参与训练，其余 -100。

    扫描策略（自包含、不依赖"对整个模板正则"）：
    找 <|im_start|>assistant\n 起点，标记到下一个 <|im_end|> 之间为有效标签。
    """
    tok = tokenizer or load_tokenizer()
    start_marker = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    end_marker = tok("<|im_end|>", add_special_tokens=False).input_ids  # [2]

    labels = [-100] * len(input_ids)
    i = 0
    n = len(input_ids)
    while i < n:
        if input_ids[i:i + len(start_marker)] == start_marker:
            start = i + len(start_marker)
            end = start
            while end < n and input_ids[end:end + len(end_marker)] != end_marker:
                end += 1
            stop = min(end + len(end_marker), n)
            for j in range(start, stop):
                labels[j] = input_ids[j]
            i = stop if end < n else n
        else:
            i += 1
    return labels