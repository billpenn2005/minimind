# 第 10 课 · 分词器与数据集（tokenizer + ChatML + 掩码标签）

> 难度：★★☆☆☆ ｜ 预计用时：2~3 小时 ｜ 前置：09 课（知道 -100 是什么）
> 学完本课你应能回答：**文本怎么变成 id？ChatML 模板长什么样？为什么 user 的 token 要被 -100 屏蔽？**

---

## 0. 本课目标

- [ ] 理解 BPE 分词原理与 minimind 6400 词表的来龙去脉；
- [ ] 把仓库里的 tokenizer 文件接进 `minimind3/tokenizer/`，写 `tokenizer_utils.py` 包装层；
- [ ] 理解 ChatML 模板（`<|im_start|>user/assistant`）与特殊 token id；
- [ ] 手写 `datasets.py`：PretrainDataset + SFTDataset（含回答掩码标签）；
- [ ] 验收 5 项新检查（10.1~10.5，累计 55 项全过）。

---

## 1. 理论：分词器

### 1.1 为什么需要分词器？

模型吃数字，人写字。分词器 = 双向的"编码/解码器"：

```
"猫会跑。" ──encode──▶ [1968, 294, 1950, ...]  ──decode──▶ "猫会跑。"
```

### 1.2 BPE：子词（subword）分词

把词拆成"比字大、比词小"的**子词单元**：高频词整体一个 token（"猫"），罕见词拆成片段
（"猫粮" → 猫 + 粮）。BPE 从字符开始迭代合并**最频繁出现的相邻对**，直到词表满 6400。

minimind 的词表：**6400 tokens**（含特殊 token `<|endoftext|>=0`、`<|im_start|>=1`、`<|im_end|>=2`、
对象引用/框选等工具类 token）。规模小的原因：6400 的词表让 embedding + lm_head 参数占比极小，
对小模型更划算（499M 参数的等效占比 vs 64000 词表会吃掉一大块）。

> 本教程**不重训**分词器：直接复用仓库已验证的词表文件（`model/tokenizer.json` +
> `tokenizer_config.json`，已拷贝进 `minimind3/tokenizer/` 成为跟踪文件，见第 10 课提交）——
> 零成本、生态兼容（`AutoTokenizer.from_pretrained("./minimind3/tokenizer")` 直接可用）。

### 1.3 特殊 token 与 ChatML 模板

对话不是"一句话"，而是**角色轮流说话**。ChatML 模板把对话线性化成一段文本：

```
<|im_start|>user
什么是猫？<|im_end|>
<|im_start|>assistant
猫是一种会抓老鼠的动物。<|im_end|>
<|im_start|>assistant
```

- `<|im_start|>`（id=1）开启一个角色轮次，`<|im_end|>`（id=2）关闭；
- 训练时模型看到这样的序列，学"用户问什么 + 助手答什么"的完整因果模式；
- 推理时我们手动拼一段以 `assistant\n` 结尾的模板，让模型接着续写回答（14 课的 e2e 就这么干）。

### 1.4 -100 掩码标签：只监督"回答"

SFT 的目标不是"让模型背下用户的问题"，而是"学会回答"。所以标签设计：

| token 来源 | 标签 | 参与损失？ |
|---|---|---|
| system / user 的提问、模板标签 | **-100** | ❌（隐身） |
| assistant 的回答内容 | 真实 token id | ✅ |

实现 `generate_labels` 的经典手法：先用 `tokenizer("<|im_start|>assistant\n")`（无 add_special_tokens）
得到**起始标记 token 序列**，在整段 id 里**扫描**它出现的位置，从那里一路标记到下一个
`<|im_end|>`（含），其余全部 -100。

> 这个记号扫描依赖一个 BPE 特性：**同样的子串在上下文里独立编码时，切分结果与它在长句里一致**
> （编码是上下文无关的分片）。本课验收 10.3/10.4 会精确验证"user 区全 -100、assistant 区等于原文"。

---

## 2. 阅读参考答案

### 2.1 `tokenizer_utils.py`

```python
TOKENIZER_DIR = Path(__file__).parent / "tokenizer"

def load_tokenizer():          # 懒加载单例（只加载一次）
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

IM_START_ID, IM_END_ID = 1, 2  # 与词表一致

def build_chat_prompt(conversations):   # [{role, content}, ...] → ChatML 文本
    ...

def encode_chat(text, max_length):      # 编码 + 截断，特殊 token 由模板负责
    ...

def generate_labels(input_ids):         # assistant 掩码扫描
    start_marker = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    ... # 扫到 start_marker 就从这里标记至下一个 <|im_end|>，其余 -100
```

### 2.2 `datasets.py`

```python
class PretrainDataset(Dataset):
    # load_dataset("json", data_files=...) → 每条 text 切成 max_length-2
    # 包 <bos> + text + <eos>，右边 pad 到 max_length
    # labels = input_ids 复制，pad 位标 -100
    # 返回 (input_ids, labels)

class SFTDataset(Dataset):
    # 手读 jsonl（utf-8）→ build_chat_prompt → encode → generate_labels
    # 返回 (input_ids, labels)
```

---

## 3. 手写任务清单

1. 确认 `minimind3/tokenizer/` 下有 `tokenizer.json` + `tokenizer_config.json`，`AutoTokenizer` 能加载（词表 6400、bos=1、eos=2、pad=0）；
2. `tokenizer_utils.py`：懒加载单例 + 三个 helper；
3. `datasets.py`：两个 Dataset 类（注意 Windows 上**先 import datasets 再 import torch**，规避 pyarrow DLL 冲突）；
4. 生成/确认合成数据：`tools/make_synthetic_data.py --out data --num 300`；
5. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 4. 验收解读（verify/10_data.py）

| 检查 | 验什么 |
|---|---|
| 10.1 | 分词器可加载：vocab 6400、bos/eos/pad id 正确、`encode→decode` 往返一致 |
| 10.2 | `build_chat_prompt` 产出含 `<|im_start|>`/`<|im_end|>` 的模板；IM 恰为单 id 1/2 |
| 10.3 | PretrainDataset：形状、bos/eos 包裹、pad 区标签为 -100 |
| 10.4 | SFTDataset：**user 区全 -100、assistant 区 == 原文 id**（整段子序列扫描断言） |
| 10.5 | 样本里非 -100 标签数 > 0 | 

> 10.4 是本课灵魂：用整段连续子序列扫描（而非 `ids.index`）找标记——`index` 找首个匹配会撞上
> 文本里恰好相同的字符（例如"猫"出现在提问里），必须全窗口匹配。

## 5. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/tokenizer_utils.py` | `model/` 的 tokenizer 文件（打包层） |
| `minimind3/datasets.py` | `dataset/lm_dataset.py` |

教程有意**简化**：标准实现的 SFTDataset 支持 tools / reasoning_content / 多模态占位等；教程只保留
核心 ChatML + 掩码（进阶见 10 课"进阶"）。`generate_labels` 的"起始标记通过独立 tokenizer 调用得到"
这一手与 minimind 的 `f'{bos}assistant\n'` 字符串扫法**数学等价但更稳**（不受特殊 token 粘连影响）。

## 6. 常见坑

- **Windows pyarrow DLL 冲突**：datasets 必须在 torch 之前 import（或用单进程加载）；
- **`add_special_tokens=False`** 忘写：编码时自动加 bos/eos，掩码扫描全乱；
- **截断把 assistant 回答切掉**：SFT 的 `max_seq_len` 必须 ≥ 模板+问题+回答；`truncation` 从**右**切，
  回答在尾部最容易被切——实测教训（12 课还会强调）；
- **右 pad 与左 pad**：训练用右 pad；生成时若用左 pad 要小心位置错位；
- **GBK 控制台打印中文**：终端乱码 ≠ 数据错（编码/解码往返正确即真值）。

## 7. 小结 & 下一课

- ✅ BPE 子词分词 + 6400 词表复用（零下载、生态兼容）；
- ✅ ChatML 模板与特殊 token（`<|im_start|>`/`<|im_end|>`）；
- ✅ -100 掩码：user 隐身、只监督 assistant；
- ✅ 合成数据 schema 与 minimind 真实数据完全一致——交换 `--data_path` 即可升级大语料。

**下一课（11-pretrain）**：终于开始训练！手写**预训练循环**：余弦退火学习率、梯度裁剪、
梯度累积、checkpoint/断点续训（`_resume.pth`）、可复现种子。验收会真的在 CPU 上跑一个小模型，
断言 loss 逐轮下降。