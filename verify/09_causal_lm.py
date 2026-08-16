"""第 09 课验收：MiniMindForCausalLM（损失、绑定、生成、HF 集成）。"""
import math
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minimind3.causal_lm import MiniMindForCausalLM  # noqa: E402
from minimind3.config import MiniMindConfig  # noqa: E402
from verify._common import TINY_CONFIG, set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _make():
    cfg = MiniMindConfig(**TINY_CONFIG)
    model = MiniMindForCausalLM(cfg)
    return cfg, model


@register("09.1 logits 形状与权重绑定")
def check_logits_tie(args):
    cfg, model = _make()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert model.lm_head.weight is model.model.embed_tokens.weight, "lm_head 未与 embed 绑定"
    # 绑定后：同一位置输出经 lm_head 的投影 == embed 查表
    h = model.model(x)[0]
    assert torch.allclose(out.logits, h @ model.lm_head.weight.T, atol=1e-5)
    assert out.loss is None


@register("09.2 损失：shift 与 ignore_index")
def check_loss(args):
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

    cfg, model = _make()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    o1 = model(x, labels=x)
    assert isinstance(o1, MoeCausalLMOutputWithPast)
    # 手动参考：CE(logits[t] -> x[t+1])
    logits = o1.logits
    ref = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].reshape(-1, cfg.vocab_size),
        x[..., 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(o1.loss, ref, atol=1e-6)
    # -100 标签被忽略：只保留第一个预测位置有效时，loss == 仅首位置贡献
    y = x.clone()
    y[:, 2:] = -100  # 保留样本的 token[1] 作为唯一标签（对 logits[0] 求 CE）
    o2 = model(x, labels=y)
    first_tok_loss = torch.nn.functional.cross_entropy(
        logits[:, 0:1, :].reshape(-1, cfg.vocab_size),
        x[:, 1:2].reshape(-1),
    )
    assert torch.allclose(o2.loss, first_tok_loss, atol=1e-5)


@register("09.3 几步优化后 loss 下降（可学习性）")
def check_learnable(args):
    set_seed(0)
    cfg, model = _make()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (4, 16))  # 固定小数据，便于记忆式学习
    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss = model(x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0] * 0.95, f"loss 未下降: {losses[0]:.4f} -> {losses[-1]:.4f}"


@register("09.4 贪心生成确定性与长度")
def check_greedy(args):
    set_seed(1)
    cfg, model = _make()
    model.eval()
    x = torch.tensor([[10, 20, 30]])
    out = model.generate(x, max_new_tokens=6, do_sample=False, temperature=1.0, top_k=0, top_p=1.0)
    assert out.shape == (1, 9)
    assert torch.equal(out[:, :3], x)
    out2 = model.generate(x, max_new_tokens=6, do_sample=False, temperature=1.0, top_k=0, top_p=1.0)
    assert torch.equal(out, out2), "贪心生成必须确定"


@register("09.5 eos 提前终止")
def check_eos(args):
    cfg, model = _make()
    model.eval()
    eos = 2
    x = torch.tensor([[eos]])
    out = model.generate(x, max_new_tokens=20, do_sample=False, eos_token_id=eos)
    assert out.shape[1] == 2, f"应在 1 个新 token 后停止，实际生成长度 {out.shape[1]}"
    assert out[0, -1] == eos


@register("09.6 采样超参不报错（temperature/top_k/top_p/repetition_penalty）")
def check_sampling_args(args):
    set_seed(2)
    cfg, model = _make()
    model.eval()
    x = torch.tensor([[5, 6, 7]])
    out = model.generate(
        x, max_new_tokens=5, do_sample=True, temperature=0.7, top_k=20, top_p=0.9,
        repetition_penalty=1.2,
    )
    assert out.shape[1] == 8


@register("09.7 KV-Cache 开关下生成一致")
def check_kv_generate(args):
    set_seed(3)
    cfg, model = _make()
    model.eval()
    x = torch.tensor([[3 if i % 5 else 9 for i in range(8)]])
    a = model.generate(x, max_new_tokens=6, do_sample=False, use_cache=True, eos_token_id=None)
    b = model.generate(x, max_new_tokens=6, do_sample=False, use_cache=False, eos_token_id=None)
    assert torch.equal(a, b), "KV-Cache 开关不应改变贪心生成结果"


@register("09.8 HF save_pretrained / from_pretrained round-trip")
def check_hf_roundtrip(args):
    set_seed(4)
    cfg, model = _make()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 6))
    expected = model(x).logits
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        loaded = MiniMindForCausalLM.from_pretrained(td)
        loaded.eval()
        got = loaded(x).logits
    assert torch.allclose(expected, got, atol=1e-5), "save/load 后前向不一致"