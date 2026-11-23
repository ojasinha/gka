"""Model correctness: absorbed cached decode == full forward, cache sizes, init scales, run names, ZClip."""

import math

import pytest
import torch

from gka.model import GPT, Config, VARIANTS
from gka.runs import DEFAULT_RUNS, MODELS, OPTIONAL_MODELS, parse_run_name, run_name
from gka.zclip import ZClip

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n_group", [1, 2, 4])
def test_cached_decode_matches_forward(variant, n_group):
    torch.manual_seed(0)
    cfg = Config(attn=variant, d_model=128, n_layer=2, n_head=8, n_group=n_group, head_dim=32, rope_dim=16, kv_latent=48, seq_len=64)
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
    expected = {"gqa": 2 * G * dh, "gva_drope": G * dh + dr, "gka_drope": G * dh + dr, "mla": cfg.kv_latent + dr}.get(variant, G * dh)
    assert GPT(cfg).cache_scalars_per_token() == expected * cfg.n_layer


@pytest.mark.parametrize("name", DEFAULT_RUNS + OPTIONAL_MODELS + ["gka_g4_s7", "gka_drope_s3", "mla_s2", "gka_c2048", "gqa_g1_c2048_s2"])
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

def test_planned_cache_sizes():
    """PLAN.md: MLA's cache matches GVA+DRoPE's (144 scalars/token/layer), MQA/GKA/GVA are half of GQA."""
    per_layer = {m: GPT(Config(attn=parse_run_name(m)["attn"], n_group=parse_run_name(m)["n_group"])).blocks[0].attn.cache_scalars_per_token()
                 for m in MODELS}
    assert per_layer["gqa"] == 256
    assert per_layer["mla"] == per_layer["gva_drope"] == 144
    assert per_layer["gqa_g1"] == per_layer["gka"] == per_layer["gva"] == per_layer["sharedkv"] == 128
    assert per_layer["gka_xg"] == per_layer["gka_vrope"] == per_layer["gka_xg_vrope"] == 128


@pytest.mark.parametrize("variant", ["gka", "gka_xg", "gka_vrope", "gka_xg_vrope"])
def test_gka_family_matches_naive_definition(variant):
    """The factorised forward (rotate keys, aggregate, rotate the output back, then N_h) equals the
    definition written out per query/key pair: value_{t,j} = (k_j R_{j-t}, or unrotated k_j) N_h,
    using the head's own group (gka, gka_vrope) or every group (the cross-group variants)."""
    torch.manual_seed(0)
    cfg = Config(attn=variant, d_model=64, n_layer=1, n_head=4, n_group=2, head_dim=16, seq_len=16)
    a = GPT(cfg).blocks[0].attn.float()
    B, T, H, G, dh = 2, 9, cfg.n_head, cfg.n_group, cfg.head_dim
    x = torch.randn(B, T, cfg.d_model)
    with torch.no_grad():
        got = a(x)
        q = a.q_proj(x).view(B, T, H, dh)
        k = a.k_proj(x).view(B, T, G, dh)
        pos = torch.arange(T)
        qr = a.rope(q.transpose(1, 2), pos).transpose(1, 2)  # (B,T,H,dh)
        kr = a.rope(k.transpose(1, 2), pos).transpose(1, 2)  # (B,T,G,dh)
        o = torch.zeros(B, T, H, dh)
        for h in range(H):
            g = h // (H // G)
            for t in range(T):
                s = torch.stack([(qr[:, t, h] * kr[:, j, g]).sum(-1) for j in range(t + 1)], -1) / math.sqrt(dh)
                w = s.softmax(-1)  # (B, t+1)
                for j in range(t + 1):
                    src = k[:, j] if a.xg else k[:, j, g : g + 1]  # (B, n, dh): the groups the value reads
                    if a.vrope:  # relative rotation R_{j-t} = R_{t-j}^T on each dh-wide chunk
                        src = a.rope(src.unsqueeze(-2), torch.tensor([t - j]), inverse=True).squeeze(-2)
                    o[:, t, h] += w[:, j, None] * (src.reshape(B, -1) @ a.N[h])
        want = a.o_proj(o.reshape(B, T, H * dh))
    assert (got - want).abs().max().item() < 1e-4


def _rms(t):
    return t.float().pow(2).mean().sqrt().item()


@pytest.mark.parametrize("variant", ["gva", "gka", "gka_xg", "mla"])
def test_reconstructed_streams_start_at_gqa_scale(variant):
    """Scale-matched init: reconstructed K (GVA, MLA) and V (GKA, MLA) start at the RMS of GQA's projections."""
    torch.manual_seed(0)
    x = torch.nn.functional.rms_norm(torch.randn(4, 256, 512), (512,))
    ref = GPT(Config(attn="gqa")).blocks[0].attn.k_proj(x)
    a = GPT(Config(attn=variant)).blocks[0].attn
    with torch.no_grad():
        if variant == "gva":
            streams = [torch.einsum("btd,hde->bhte", a.v_proj(x)[..., :64], a.M)]
        elif variant == "gka":
            streams = [torch.einsum("btd,hde->bhte", a.k_proj(x)[..., :64], a.N)]
        elif variant == "gka_xg":  # every group's key, fan-in G*dh
            streams = [torch.einsum("btc,hce->bhte", a.k_proj(x), a.N)]
        else:
            c = a.kv_norm(a.kv_down(x))
            streams = [a.k_up(c), a.v_up(c)]
    for s in streams:
        assert abs(_rms(s) / _rms(ref) - 1) < 0.1


def _grads(norm):
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), norm / 2.0)  # ||g|| = norm
    return [p]


def test_zclip_warmup_then_clips_spikes():
    z = ZClip(warmup_steps=5, max_norm=None)
    for n in [1.0, 1.1, 0.9, 1.0, 1.0]:
        out = z.step(_grads(n))
        assert not out["zclipped"] and out["clipped_norm"] == pytest.approx(n)
    assert z.mean == pytest.approx(1.0) and z.var == pytest.approx(0.004)
    ps = _grads(1.02)  # ordinary step: untouched
    assert not z.step(ps)["zclipped"]
    mean_before, std_before = z.mean, math.sqrt(z.var)
    ps = _grads(10.0)  # spike: clipped to mean + z_thresh^2 / z * std, below mean + z_thresh * std
    out = z.step(ps)
    zscore = (10.0 - mean_before) / (std_before + z.eps)
    expected = mean_before + z.z_thresh**2 / zscore * std_before
    assert out["zclipped"] and out["clipped_norm"] == pytest.approx(expected)
    assert ps[0].grad.norm().item() == pytest.approx(expected, rel=1e-4)
    assert expected < mean_before + z.z_thresh * std_before
    assert z.mean < 1.1  # statistics were updated with the clipped norm, not the spike
    assert z.n_clipped == 1


def test_zclip_max_norm_ceiling_and_state_roundtrip():
    z = ZClip(warmup_steps=2, max_norm=1.0)
    ps = _grads(5.0)
    assert z.step(ps)["clipped_norm"] == pytest.approx(1.0)  # ceiling applies during warmup
    assert ps[0].grad.norm().item() == pytest.approx(1.0, rel=1e-4)
    z.step(_grads(0.5))
    z2 = ZClip(warmup_steps=2, max_norm=1.0)
    z2.load_state_dict(z.state_dict())
    assert (z2.mean, z2.var, z2.n_clipped) == (z.mean, z.var, z.n_clipped)
