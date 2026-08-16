# 第 10 课 · 分词器与数据集

> [上一课 09-causal-lm](../09-causal-lm/README.md) ← [目录](../../README.md) → [下一课 11-pretrain](../11-pretrain/README.md)
>
> 难度：★★☆☆☆ ｜ 预计用时 2 小时 ｜ 前置：09 课

## 0. 本课目标

- [ ] 理解 BPE 子词分词与 ChatML 模板；-100 掩码标签；
- [ ] 拷贝词表素材，手写 `tokenizer_utils.py` + `datasets.py`；
- [ ] 生成 `data/` 合成数据；
- [ ] 验收 56 项累计（本课新增 10.1~10.5）。

## 1. 理论速览

- **分词器**：文本 ⇄ token id（"猫会跑。" → [1968, 294, …]）；BPE 从字符开始迭代合并最频繁相邻对直到词表满 6400。**教程不重训词表**：复用 minimind 自训练词表（零下载），词表文件从 `answers/minimind3/tokenizer/` 拷贝。
- **特殊 token**：`<|endoftext|>=0`、`<|im_start|>=1`、`<|im_end|>=2`（pad 惯例取 0）。
- **ChatML**：
  ```
  <|im_start|>user\n什么是猫？<|im_end|>\n
  <|im_start|>assistant\n猫是一种会抓老鼠的动物。<|im_end|>\n
  ```
- **-100 掩码**：SFT 只监督 assistant 回答；user/模板/pad 标签 = -100（隐身）。`generate_labels` 的扫描起点 = `tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids`（依赖 BPE 上下文无关切分：独立编码的子串 == 长句里的切片）。

## 2. 任务要求（精确规格）

### 2.1 拷贝素材（本课非手写部分）

```bash
cp -r answers/minimind3/tokenizer minimind3/tokenizer   # 词表 + 配置（约 0.5MB）
```
（也可 `git show` 任一含该目录的提交；最终效果一致：`minimind3/tokenizer/` 下要有 `tokenizer.json` 与 `tokenizer_config.json`。）

### 2.2 新建 `minimind3/tokenizer_utils.py`

**模块级**：
- `TOKENIZER_DIR: Path` = `Path(__file__).resolve().parent / "tokenizer"`；
- `IM_START_ID: int = 1`；`IM_END_ID: int = 2`；
- 模块私有 `_tokenizer = None`（惰性单例）。

| 函数 | 签名 | 返回（类型） | 行为 |
|---|---|---|---|
| `load_tokenizer` | `() -> AutoTokenizer` | tokenizer | 惰性 `AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))`，缓存单例 |
| `build_chat_prompt` | `(conversations: list[dict]) -> str` | `str` | 逐条 `<|im_start|>{role}\n{content}<|im_end|>\n` 拼接；`role` 必须 ∈ {system,user,assistant}（否则 assert） |
| `encode_chat` | `(conversations: list[dict], tokenizer=None, max_length: int \| None = None) -> list[int]` | `list[int]` | `tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids`（**必须 add_special_tokens=False**） |
| `generate_labels` | `(input_ids: list[int], tokenizer=None) -> list[int]` | `list[int]` | 全 -100 起；找 `start_marker` 起点，标记到（含）下一个 `<|im_end|>` 为真实标签 |

### 2.3 新建 `minimind3/datasets.py`

**模块级 import**：`json`、`torch`、`from datasets import load_dataset`、`from torch.utils.data import Dataset`、`from .tokenizer_utils import IM_END_ID, IM_START_ID, encode_chat, generate_labels, load_tokenizer`。

**`class PretrainDataset(Dataset)`**：

| 成员 | 类型 | 值 |
|---|---|---|
| `tokenizer` | AutoTokenizer | 构造参数 2 或 `load_tokenizer()` |
| `max_length` | `int` | 构造参数 3（默认 512） |
| `samples` | Dataset | `load_dataset("json", data_files=data_path, split="train")` |

- `__init__(self, data_path: str, tokenizer=None, max_length: int = 512)`；
- `__len__() -> int`；
- `__getitem__(self, index) -> tuple[Tensor, Tensor]`：
  - 取 `samples[index]["text"]`；`tokenizer(text, add_special_tokens=False, max_length=max_length-2, truncation=True)`；
  - 包 `[bos] + tokens + [eos]`，右侧补 `pad_token_id` 到 max_length；
  - 返回 `(input_ids long (max_length,), labels long (max_length,))`；`labels = input_ids.clone()`，**pad 位 = -100**。

**`class SFTDataset(Dataset)`**：

| 成员 | 类型 | 值 |
|---|---|---|
| `tokenizer` | AutoTokenizer | 构造参数 2 或 `load_tokenizer()` |
| `max_length` | `int` | 构造参数 3（默认 512） |
| `samples` | `list[dict]` | `_load_jsonl(jsonl_path)` |

- `__init__(self, jsonl_path: str, tokenizer=None, max_length: int = 512)`；
- `@staticmethod _load_jsonl(jsonl_path: str) -> list[dict]`：utf-8 逐行 `json.loads`（跳过空行）；
- `__len__() -> int`；
- `__getitem__`：`encode_chat(conversations, tokenizer, max_length)` → 右侧补 pad → `generate_labels`；返回 `(input_ids long, labels long)`（labels 里 assistant 区是真实 id、其余 -100）。

### 2.4 生成数据（第 10 课起 `data/` 可存在）

```bash
.venv/Scripts/python.exe -m tools.make_synthetic_data --out data --num 300 --seed 0
```
产物 `data/tiny_pretrain.jsonl`（`{"text":…}` ×300）与 `data/tiny_sft.jsonl`（`{"conversations":[…]}` ×300）；内容与 `answers/data/` 逐字节相同（同种子）。

## 3. 手写步骤

拷贝 §2.1 → 写 §2.2 → 写 §2.3 → 生成 §2.4 → `verify.py 10`（**训练类检查用 `.venv/Scripts/python.exe`，datasets 依赖在其 venv 里**）。

## 4. 验收解读（verify/10_data.py）

| 检查（新） | 验什么 |
|---|---|
| 10.1 | 分词器可加载：vocab 6400、bos=1/eos=2/pad=0；encode→decode 往返一致 |
| 10.2 | ChatML 拼接含 `<|im_start|>`/`<|im_end|>`；IM 恰为单 id 1/2 |
| 10.3 | PretrainDataset：形状、bos/eos 包裹、pad 区标签 -100 |
| 10.4 | SFTDataset：**user 区全 -100、assistant 区 == 原文 id**（整段子序列扫描断言——单 token 定位会撞到文本里的相同字符） |
| 10.5 | 每条样本非 -100 标签 > 0 |

## 5. 参考答案

`answers/minimind3/tokenizer_utils.py`、`answers/minimind3/datasets.py`（先写后对）。

## 6. 常见坑

- **Windows pyarrow DLL 冲突**：任何脚本顶部 **先 import datasets 再 import torch**；
- **`add_special_tokens` 忘关**：自动加 bos/eos 弄乱掩码扫描；
- **截断吃掉回答**：`max_seq_len` 必须 ≥ 模板+问题+回答（`truncation` 从右切，回答在尾部最先被切）；
- **用 `ids.index` 找标记**：会命中文本中相同的首个字符——要整段连续子序列匹配；
- **GBK 控制台打印中文乱码**：正常；数据正确性以 encode/decode 往返为准。

## 7. 对照标准实现

| 你手写 | minimind 标准 |
|---|---|
| `minimind3/tokenizer_utils.py` | `model/` 词表 + 模板逻辑 |
| `minimind3/datasets.py` | `dataset/lm_dataset.py` |

教程有意简化（去掉 tools/reasoning_content 支持）；`generate_labels` 用独立 marker 扫描，比 minimind 的 `f'{bos}assistant\n'` 字符串扫法更稳（不受特殊 token 粘连影响），数学等价。

## 8. 小结

✅ 文本 ⇄ id 全链路打通（词表复用 + ChatML + 掩码 + 可复现数据）。
**下一课**：终于开始训练——预训练循环（cosine 退火、裁剪、累积、断点续训）。