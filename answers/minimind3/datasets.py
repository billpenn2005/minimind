"""minimind3 数据集（镜像 minimind dataset/lm_dataset.py 的核心语义，手写）。

- PretrainDataset：文本 → bos+内容+eos，padding 到固定长，pad 位置标签 -100；
- SFTDataset：对话 → ChatML 文本 → 仅 assistant 片段作为标签（其余 -100）。
"""
import json

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .tokenizer_utils import IM_END_ID, IM_START_ID, encode_chat, generate_labels, load_tokenizer


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer=None, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer or load_tokenizer()
        self.max_length = max_length
        # 用 HF datasets 读取 jsonl（内存友好，兼容大数据集）
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = str(self.samples[index]["text"])
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        # 包一层 bos/eos，构成"完整文档"样本
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # padding 不计损失
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer=None, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer or load_tokenizer()
        self.max_length = max_length
        self.samples = self._load_jsonl(jsonl_path)

    @staticmethod
    def _load_jsonl(jsonl_path: str) -> list[dict]:
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        conversations = self.samples[index]["conversations"]
        input_ids = encode_chat(conversations, self.tokenizer, max_length=self.max_length)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = generate_labels(input_ids, self.tokenizer)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )