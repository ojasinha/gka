"""The absorbed cached decode path of every variant must reproduce the full forward pass."""

import pytest
import torch

from gka.model import GPT, Config, VARIANTS
from gka.runs import DEFAULT_RUNS, parse_run_name, run_name

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n_group", [1, 2, 4])
def test_cached_decode_matches_forward(variant, n_group):
    torch.manual_seed(0)
    cfg = Config(attn=variant, d_model=128, n_layer=2, n_head=8, n_group=n_group, head_dim=32, rope_dim=16, seq_len=64)
    m = GPT(cfg).to(DEV).float().eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 40), device=DEV)
    with torch.no_grad():
        ref, _ = m(idx)
        m.setup_cache(2, 64, dtype=torch.float32)
        out = [m.step(idx[:, :24], 0)]  # chunked prefill
        for t in range(24, 40):  # token-by-token decode
            out.append(m.step(idx[:, t : t + 1], t))
    got = torch.cat(out, 1)
    want = torch.cat([ref[:, 23:24], ref[:, 24:40]], 1)
    assert (got - want).abs().max().item() < 1e-3


@pytest.mark.parametrize("variant", VARIANTS)
def test_cache_size(variant):
    cfg = Config(attn=variant)
    G, dh, dr = cfg.n_group, cfg.head_dim, cfg.rope_dim
    expected = {"gqa": 2 * G * dh, "gva_drope": G * dh + dr, "gka_drope": G * dh + dr}.get(variant, G * dh)
    assert GPT(cfg).cache_scalars_per_token() == expected * cfg.n_layer


@pytest.mark.parametrize("name", DEFAULT_RUNS + ["gka_g4_s7", "gka_drope_s3", "gka_c2048", "gqa_g1_c2048_s2"])
def test_run_name_roundtrip(name):
    assert run_name(**parse_run_name(name)) == name


def test_rope_tables_grow_and_ntk_one_is_identity():
    torch.manual_seed(0)
    cfg = Config(attn="gka", d_model=128, n_layer=2, n_head=8, n_group=2, head_dim=32, seq_len=16)
    m = GPT(cfg).to(DEV).float().eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 300), device=DEV)  # far beyond the initial 8 x 16 tables
    with torch.no_grad():
        ref, _ = m(idx)
        m.set_rope_ntk(4.0)
        scaled, _ = m(idx)
        m.set_rope_ntk(1.0)
        back, _ = m(idx)
    assert (ref - back).abs().max().item() < 1e-5
    assert (ref - scaled).abs().max().item() > 1e-3