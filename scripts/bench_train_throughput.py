"""Training throughput and peak memory of every variant at the default config (random data)."""

import argparse
import time

import torch

from gka.model import GPT, Config, VARIANTS

p = argparse.ArgumentParser()
p.add_argument("--micro_bs", type=int, default=8)
p.add_argument("--steps", type=int, default=10)
args = p.parse_args()

for v in VARIANTS:
    torch.manual_seed(0)
    cfg = Config(attn=v)
    m = GPT(cfg).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, cfg.vocab_size, (args.micro_bs, cfg.seq_len), device="cuda")
    for i in range(args.steps + 3):  # 3 warm-up steps
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = m(x, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    tps = args.steps * x.numel() / (time.time() - t0)
    print(f"{v:10s} params={m.num_params(False) / 1e6:.2f}M (non-emb {m.num_params() / 1e6:.2f}M)  "
          f"{tps / 1e3:.1f}k tok/s  peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
    del m, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
