"""Timing run: training throughput and peak memory per model, and the projected cost of a full run.

Times the real training step (bf16 autocast forward/backward, ZClip, fused AdamW) on random tokens at
the training settings, then projects hours and Colab compute units for --tokens per run.
"""

import argparse
import time

import torch

from gka.model import GPT, Config
from gka.runs import MODELS, parse_run_name
from gka.zclip import ZClip

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=MODELS, help="models, e.g. gqa gqa_g1 mla gka")
p.add_argument("--micro_bs", type=int, default=16)
p.add_argument("--seq_len", type=int, default=Config.seq_len)
p.add_argument("--steps", type=int, default=20, help="timed micro-steps (after warm-up)")
p.add_argument("--warmup", type=int, default=5, help="untimed micro-steps (includes compilation)")
p.add_argument("--compile", action="store_true")
p.add_argument("--tokens", type=int, default=1_000_000_000, help="tokens per run, for the projection")
p.add_argument("--cu_per_hour", type=float, default=11.77, help="Colab compute units per hour of this GPU")
args = p.parse_args()

torch.set_float32_matmul_precision("high")
print(f"{torch.cuda.get_device_name()}, micro_bs={args.micro_bs}, seq_len={args.seq_len}, compile={args.compile}")
total_h = 0.0
for name in args.runs:
    spec = parse_run_name(name)
    torch.manual_seed(0)
    cfg = Config(attn=spec["attn"], n_group=spec["n_group"], seq_len=args.seq_len)
    raw = GPT(cfg).cuda()
    model = torch.compile(raw) if args.compile else raw
    opt = torch.optim.AdamW(raw.parameters(), lr=1e-4, fused=True)
    zclip = ZClip()
    x = torch.randint(0, cfg.vocab_size, (args.micro_bs, cfg.seq_len), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, x)
        loss.backward()
        loss.item()  # the training loop syncs on the loss every micro-step
        zclip.step(raw.parameters())
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    tps = args.steps * x.numel() / (time.time() - t0)
    hours = args.tokens / tps / 3600
    total_h += hours
    print(f"{name:10s} params={raw.num_params(False) / 1e6:.2f}M (non-emb {raw.num_params() / 1e6:.2f}M)  "
          f"{tps / 1e3:7.1f}k tok/s  peak={torch.cuda.max_memory_allocated() / 2**30:5.2f}GB  "
          f"-> {hours:.2f} h, {hours * args.cu_per_hour:.1f} CU per {args.tokens / 1e9:g}B-token run", flush=True)
    del model, raw, opt
    torch.cuda.empty_cache()
print(f"one run of each listed model: {total_h:.1f} h, {total_h * args.cu_per_hour:.0f} CU (training only)")
