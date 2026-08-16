"""第 13 课验收：torch(pth) ⇄ HF transformers 互转。"""
import json
import subprocess
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


def _make_random_pth(tmp: Path) -> tuple[Path, MiniMindForCausalLM]:
    """生成一个随机权重的 .pth（含 config sidecar）与原模型（验证转换正确性无需真实训练）。"""
    set_seed(5)
    cfg = MiniMindConfig(**{**TINY_CONFIG, "vocab_size": 512})
    model = MiniMindForCausalLM(cfg)
    pth = tmp / "model_96.pth"
    torch.save({k: v.float().cpu() for k, v in model.state_dict().items()}, pth)
    import json

    (tmp / "model_96.config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    return pth, model


@register("13.1 从权重形状推断配置")
def check_infer(args):
    from minimind3.convert import infer_config_from_state_dict

    cfg = MiniMindConfig(**TINY_CONFIG)
    sd = MiniMindForCausalLM(cfg).state_dict()
    inferred = infer_config_from_state_dict(sd)
    assert inferred.hidden_size == 96
    assert inferred.num_hidden_layers == 2
    assert inferred.vocab_size == 512
    assert inferred.num_attention_heads == 4
    assert inferred.num_key_value_heads == 2
    assert inferred.intermediate_size == 128


@register("13.2 pth -> HF 目录产物齐全 + auto_map 指向本地模块")
def check_convert_files(args):
    with tempfile.TemporaryDirectory() as td:
        pth, _ = _make_random_pth(Path(td))
        out = Path(td) / "hf"
        from minimind3.convert import convert_torch2transformers

        convert_torch2transformers(str(pth), str(out), standalone=True)
        for f in ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "modeling_minimind3.py"):
            assert (out / f).exists(), f"缺 {f}"
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert cfg["model_type"] == "minimind3"
        assert cfg["hidden_size"] == 96 and cfg["num_hidden_layers"] == 2
        assert cfg["pad_token_id"] == 0
        am = cfg["auto_map"]
        assert am["AutoModelForCausalLM"].startswith("modeling_minimind3."), f"auto_map 未指向本地模块: {am}"


@register("13.3 AutoModelForCausalLM 加载 HF 目录并复现 logits")
def check_autoload(args):
    from transformers import AutoModelForCausalLM

    with tempfile.TemporaryDirectory() as td:
        pth, model = _make_random_pth(Path(td))
        out = Path(td) / "hf"
        from minimind3.convert import convert_torch2transformers

        convert_torch2transformers(str(pth), str(out))
        model.eval()
        x = torch.randint(0, 512, (2, 6))
        expected = model(x).logits
        loaded = AutoModelForCausalLM.from_pretrained(str(out), trust_remote_code=True).eval()
        got = loaded(x).logits
        assert torch.allclose(expected, got, atol=1e-5), f"logits 不一致 {(expected - got).abs().max():.2e}"


@register("13.4 自包含：脱离仓库目录也能 remote-code 加载并生成")
def check_standalone(args):
    code = (
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "import sys\n"
        "m = AutoModelForCausalLM.from_pretrained(sys.argv[1], trust_remote_code=True)\n"
        "m.eval()\n"
        "ids = m.generate(torch.tensor([[10, 20, 30]]), max_new_tokens=5, do_sample=False, top_k=0, top_p=1.0)\n"
        "assert ids.shape[1] == 8, ids.shape\n"
        "print('OK', type(m).__name__)\n"
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pth, _ = _make_random_pth(root)
        out = root / "hf"
        from minimind3.convert import convert_torch2transformers

        convert_torch2transformers(str(pth), str(out))
        # 在独立临时目录中运行，仓库路径不在 sys.path -> 只能靠建模文件本身
        r = subprocess.run(
            [sys.executable, "-c", code, str(out.resolve())],
            capture_output=True, text=True, cwd=str(root / "sandbox" if (root / "sandbox").mkdir(exist_ok=True) else root),
        )
        assert r.returncode == 0, f"standalone 加载失败:\n{r.stdout}\n{r.stderr}"
        assert "OK MiniMindForCausalLM" in r.stdout


@register("13.5 反向转换 transformers -> torch 并 round-trip")
def check_reverse(args):
    with tempfile.TemporaryDirectory() as td:
        pth, model = _make_random_pth(Path(td))
        out = Path(td) / "hf"
        back = Path(td) / "back.pth"
        from minimind3.convert import convert_torch2transformers, convert_transformers2torch, _load_from_hf

        convert_torch2transformers(str(pth), str(out))
        convert_transformers2torch(str(out), str(back))
        model.eval()
        x = torch.randint(0, 512, (2, 5))
        expected = model(x).logits
        reloaded = MiniMindForCausalLM(_load_from_hf(out).config)
        reloaded.load_state_dict(torch.load(back, map_location="cpu"), strict=True)
        reloaded.eval()
        got = reloaded(x).logits
        assert torch.allclose(expected, got, atol=1e-5), f"round-trip 不一致 {(expected - got).abs().max():.2e}"