# 参考答案文件夹（answers/）

> **规则只有一条：先自己写，写不出来再看；看完要能默写。**
> 不要把这些文件复制进工作区 `minimind3/`——`verify/01_skeleton.py` 的"起点留白"检查
> 会拒绝任何偷跑文件，而且复制答案 = 跳过学习本身。

## 结构

| 路径 | 内容 | 对应课程 |
|---|---|---|
| `answers/minimind3/` | **14 门课的完整实现**（每个文件对应一课，与教程第 14 课成果等价） | 01~14 |
| `answers/data/` | 规范合成数据（`tiny_pretrain.jsonl` / `tiny_sft.jsonl`，各 300 条） | 10~12（第 10 课可用 `tools/make_synthetic_data.py` 复现出完全相同的内容） |

## 使用姿势

1. 完成第 N 课手写、`verify.py NN` 通过后 → 再打开 `answers/minimind3/<对应文件>` 逐行对照；
2. 对照时关注：**命名是否一致**（state_dict 键名关系到第 13 课转换）、形状、返回结构、边界处理；
3. 想看得更"原汁原味"：`bash tools/fetch_reference.sh` 会把 minimind 标准实现导出到 `reference/`，
   三份文件（你的 / answers / minimind 标准）可以并排对比。

## 对照表：课程 → 参考答案文件

| 课 | 手写文件（工作区 minimind3/） | 参考答案（answers/minimind3/） |
|---|---|---|
| 01 | `__init__.py`（包入口） | `__init__.py` |
| 02 | `config.py` | `config.py` |
| 03 | `rms_norm.py` | `rms_norm.py` |
| 04 | `rope.py` | `rope.py` |
| 05 | `attention.py` | `attention.py` |
| 06 | `feed_forward.py` | `feed_forward.py` |
| 07 | `block.py` | `block.py` |
| 08 | `model_body.py` | `model_body.py` |
| 09 | `causal_lm.py` | `causal_lm.py` |
| 10 | `tokenizer_utils.py` + `datasets.py`（词表目录 `tokenizer/` 需拷贝：见第 10 课） | 同名文件 |
| 11 | `train_utils.py` + `train_pretrain.py` | 同名文件 |
| 12 | `train_full_sft.py` | 同名文件 |
| 13 | `convert.py` | `convert.py` |
| 14 | `e2e.py` | `e2e.py` |

> `answers/minimind3/tokenizer/` 里的词表文件是第 10 课的**拷贝素材**（不是本课手写产物）——
> 教程不重训分词器，直接复用 minimind 的 6400 词 BPE 词表。