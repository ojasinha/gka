"""Inference benchmark: KV-cache footprint, prefill latency, decode throughput, peak memory.

Every variant decodes through its own cached path (model.step), i.e. the absorbed path for
gva_drope / gka / gka_drope. Speed does not depend on weight values, so trained checkpoints
are used when present and random init otherwise. Plain PyTorch eager, no custom kernels.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from gka.paths import RESULTS_DIR
from gka.runs import DEFAULT_RUNS, load_model

DEV = "cuda"


def load(name):
    return load_model(name, DEV, allow_random_init=True).to(torch.bfloat16)


@torch.no_grad()
def bench(m, B, ctx, n_dec=64, reps=3):
    prompt = ctx - n_dec
    idx = torch.randint(0, 50257, (B, prompt), device=DEV)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    m.setup_cache(B, ctx)
    cache_bytes = sum(t.numel() * t.element_size() for b in m.blocks for t in b.attn.cache.values())
    # Chunked prefill keeps the fp32 score block (B, H, chunk, ctx) at <= ~256 MB per layer.
    chunk = max(16, min(512, 2 ** int(torch.log2(torch.tensor(2**28 / (B * 8 * ctx * 4))).item())))
    pre, dec = [], []
    for r in range(reps + 1):  # first rep is warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(0, prompt, chunk):
            logits = m.step(idx[:, s : s + chunk], s)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        tok = logits.argmax(-1)
        for t in range(n_dec):
            tok = m.step(tok, prompt + t).argmax(-1)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if r:
            pre.append(t1 - t0)
            dec.append(t2 - t1)
    peak = torch.cuda.max_memory_allocated() - base  # decode phase, includes the cache itself
    m.free_cache()
    pre, dec = min(pre), min(dec)
    return {
        "B": B, "ctx": ctx, "prefill_chunk": chunk,
        "cache_MB": cache_bytes / 2**20,
        "prefill_ms": pre * 1e3,
        "decode_ms_per_step": dec / n_dec * 1e3,
        "decode_tok_s": B * n_dec / dec,
        "peak_extra_MB": peak / 2**20,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run names, e.g. gqa gka gka_g4")
    p.add_argument("--settings", default="1x512,1x2048,16x2048,32x4096", help="comma-separated BATCHxCONTEXT")
    p.add_argument("--out", default=str(RESULTS_DIR / "inference.json"))
    args = p.parse_args()
    settings = [tuple(map(int, s.split("x"))) for s in args.settings.split(",")]
    out = {}
    for v in args.runs:
        m = load(v)
        a = m.blocks[0].attn
        r = {
            "cache_scalars_per_token_per_layer": a.cache_scalars_per_token(),
            "cache_bytes_per_token_bf16": 2 * m.cache_scalars_per_token(),
            "params": m.num_params(False),
            "runs": [],
        }
        for B, ctx in settings:
            try:
                res = bench(m, B, ctx)
            except torch.OutOfMemoryError:
                m.free_cache()
                torch.cuda.empty_cache()
                res = {"B": B, "ctx": ctx, "oom": True}
            r["runs"].append(res)
            print(v, json.dumps(res), flush=True)
        out[v] = r
        del m
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
