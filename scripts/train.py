"""Train one attention variant on the prepared FineWeb-Edu tokens.

All variants share the same data order (seeded), optimizer, schedule and token budget.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from gka.model import GPT, Config, VARIANTS
from gka.paths import DATA_DIR, RUNS_DIR
from gka.runs import DEFAULT_N_GROUP, DEFAULT_SEED, DEFAULT_SEQ_LEN, run_name

p = argparse.ArgumentParser()
p.add_argument("--attn", required=True, choices=VARIANTS)
p.add_argument("--tokens", type=int, default=40_000_000)
p.add_argument("--micro_bs", type=int, default=8)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--min_lr_frac", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--eval_every", type=int, default=200)
p.add_argument("--eval_batches", type=int, default=40)
p.add_argument("--n_group", type=int, default=DEFAULT_N_GROUP)
p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN, help="training context; keep micro_bs*accum*seq_len at 16384")
p.add_argument("--seed", type=int, default=DEFAULT_SEED)
p.add_argument("--data_seed", type=int, default=0)
p.add_argument("--data", default=str(DATA_DIR))
p.add_argument("--out", default=str(RUNS_DIR))
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
cfg = Config(attn=args.attn, n_group=args.n_group, seq_len=args.seq_len)
T = cfg.seq_len

data_dir = Path(args.data)
train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
tok_per_step = args.micro_bs * args.accum * T
steps = args.tokens // tok_per_step

# Fixed data order shared by every variant: a permutation of non-overlapping windows.
n_windows = (len(train) - 1) // T
order = np.random.default_rng(args.data_seed).permutation(n_windows)
assert steps * args.micro_bs * args.accum <= n_windows, "not enough training tokens"


def get_train(i):
    idx = order[i * args.micro_bs : (i + 1) * args.micro_bs]
    x = np.stack([train[j * T : j * T + T + 1] for j in idx]).astype(np.int64)
    x = torch.from_numpy(x).pin_memory().to(dev, non_blocking=True)
    return x[:, :-1], x[:, 1:]


def get_val(i, bs=8):
    x = np.stack([val[(i * bs + b) * T : (i * bs + b) * T + T + 1] for b in range(bs)]).astype(np.int64)
    x = torch.from_numpy(x).to(dev)
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(model):
    model.eval()
    tot = 0.0
    for i in range(args.eval_batches):
        x, y = get_val(i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / args.eval_batches


model = GPT(cfg).to(dev)
decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "wte" not in n]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "wte" in n]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
)


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    prog = (s - args.warmup) / max(1, steps - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


name = run_name(args.attn, args.n_group, args.seed, args.seq_len)
run_dir = Path(args.out) / name
run_dir.mkdir(parents=True, exist_ok=True)
log = open(run_dir / "log.jsonl", "w")
print(f"{name}: {model.num_params(False) / 1e6:.2f}M params ({model.num_params() / 1e6:.2f}M non-emb), "
      f"{steps} steps x {tok_per_step} tok, cache/token={model.cache_scalars_per_token()} scalars")

t0 = time.time()
micro = 0
for step in range(steps + 1):
    if step % args.eval_every == 0 or step == steps:
        vl = evaluate(model)
        rec = {"step": step, "tokens": step * tok_per_step, "val_loss": vl, "time": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {step:5d} | val {vl:.4f} | {time.time() - t0:.0f}s", flush=True)
    if step == steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    tl = 0.0
    for _ in range(args.accum):
        x, y = get_train(micro)
        micro += 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / args.accum).backward()
        tl += loss.item() / args.accum
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 20 == 0:
        el = time.time() - t0
        tps = (step + 1) * tok_per_step / el
        log.write(json.dumps({"step": step, "train_loss": tl, "grad_norm": gn, "lr": lr_at(step), "tok_s": tps}) + "\n")
        print(f"step {step:5d} | loss {tl:.4f} | gn {gn:.2f} | {tps / 1e3:.1f}k tok/s", flush=True)

torch.save({"cfg": cfg.to_dict(), "model": model.state_dict()}, run_dir / "model.pt")
json.dump({"attn": args.attn, "n_group": args.n_group, "seq_len": args.seq_len, "seed": args.seed, "params": model.num_params(False),
           "non_emb_params": model.num_params(), "final_val_loss": vl, "train_time_s": time.time() - t0,
           "tokens": steps * tok_per_step},
          open(run_dir / "summary.json", "w"), indent=2)
print("saved", run_dir)
