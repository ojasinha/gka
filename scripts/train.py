"""Train one attention variant on the prepared FineWeb-Edu tokens.

All variants share the same data order, optimizer, schedule, gradient clipping (ZClip) and token budget;
--seed changes only the initialisation. Defaults are the PLAN.md setup: 1B tokens (Chinchilla for 48M
params), 1024 context, 131072-token batches.

The run is resumable: a checkpoint (model + optimizer + ZClip state) is written to runs/<name>/ckpt.pt
every --ckpt_every steps, and re-launching the same command continues from it with the exact same data
order. A finished run (summary.json present) is skipped.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gka.model import GPT, Config, VARIANTS
from gka.paths import DATA_DIR, RUNS_DIR
from gka.runs import DEFAULT_N_GROUP, DEFAULT_SEED, DEFAULT_SEQ_LEN, parse_run_name, run_name
from gka.zclip import ZClip

p = argparse.ArgumentParser()
what = p.add_mutually_exclusive_group(required=True)
what.add_argument("--attn", choices=VARIANTS)
what.add_argument("--run", help="a run name (e.g. gqa_g1_s1); sets --attn/--n_group/--seed/--seq_len from it")
p.add_argument("--tokens", type=int, default=1_000_000_000)
p.add_argument("--micro_bs", type=int, default=64)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--min_lr_frac", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--eval_batches", type=int, default=40, help="periodic eval: batches of 8 sequences")
p.add_argument("--final_eval_batches", type=int, default=400, help="final eval: batches of 8 sequences")
p.add_argument("--ckpt_every", type=int, default=500, help="steps between resume checkpoints")
p.add_argument("--n_group", type=int, default=DEFAULT_N_GROUP)
p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
p.add_argument("--seed", type=int, default=DEFAULT_SEED)
p.add_argument("--data_seed", type=int, default=0)
p.add_argument("--compile", action="store_true", help="torch.compile the model (needs Triton; use on Linux/Colab)")
p.add_argument("--data", default=str(DATA_DIR))
p.add_argument("--out", default=str(RUNS_DIR))
args = p.parse_args()
if args.run:
    spec = parse_run_name(args.run)
    args.attn, args.n_group, args.seed, args.seq_len = spec["attn"], spec["n_group"], spec["seed"], spec["seq_len"]

dev = "cuda"
torch.set_float32_matmul_precision("high")
cfg = Config(attn=args.attn, n_group=args.n_group, seq_len=args.seq_len)
T = cfg.seq_len

name = run_name(args.attn, args.n_group, args.seed, args.seq_len)
run_dir = Path(args.out) / name
if (run_dir / "summary.json").exists():
    print(f"{name}: already finished ({run_dir / 'summary.json'}), skipping")
    sys.exit(0)
run_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = run_dir / "ckpt.pt"

# ----------------------------------------------------------------------------- data
data_dir = Path(args.data)
train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
tok_per_step = args.micro_bs * args.accum * T
steps = args.tokens // tok_per_step

# Fixed data order shared by every variant and seed: a permutation of non-overlapping windows.
n_windows = (len(train) - 1) // T
order = np.random.default_rng(args.data_seed).permutation(n_windows)
assert steps * args.micro_bs * args.accum <= n_windows, (
    f"not enough training tokens: need {steps * tok_per_step:,}, train.bin has {len(train):,}")
assert max(args.eval_batches, args.final_eval_batches) * 8 * T + 1 <= len(val), "not enough validation tokens"


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
def evaluate(n_batches):
    tot = 0.0
    for i in range(n_batches):
        x, y = get_val(i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item()
    return tot / n_batches


# ----------------------------------------------------------------------------- model / optimizer
torch.manual_seed(args.seed)
raw = GPT(cfg).to(dev)
decay = [p for n, p in raw.named_parameters() if p.dim() >= 2 and "wte" not in n]
no_decay = [p for n, p in raw.named_parameters() if p.dim() < 2 or "wte" in n]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
)
zclip = ZClip()


def lr_at(s):
    if s < args.warmup:
        return args.lr * (s + 1) / args.warmup
    prog = (s - args.warmup) / max(1, steps - args.warmup)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


start, elapsed = 0, 0.0
if ckpt_path.exists():
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    assert ck["cfg"] == cfg.to_dict() and ck["steps"] == steps and ck["tok_per_step"] == tok_per_step, \
        "checkpoint was written with different settings; delete it to start over"
    raw.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    zclip.load_state_dict(ck["zclip"])
    start, elapsed = ck["next_step"], ck["elapsed"]
    # drop log records from after the checkpoint (they will be recomputed)
    recs = [l for l in open(run_dir / "log.jsonl") if json.loads(l)["step"] < start] if (run_dir / "log.jsonl").exists() else []
    open(run_dir / "log.jsonl", "w").writelines(recs)
    print(f"{name}: resuming at step {start}/{steps}")
    del ck
elif (run_dir / "log.jsonl").exists():
    (run_dir / "log.jsonl").unlink()  # stale log of a run that never checkpointed

model = torch.compile(raw) if args.compile else raw
log = open(run_dir / "log.jsonl", "a")


def save_ckpt(next_step):
    log.flush()  # every record before next_step is on disk before the checkpoint that resumes after them
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save({"cfg": cfg.to_dict(), "model": raw.state_dict(), "opt": opt.state_dict(), "zclip": zclip.state_dict(),
                "next_step": next_step, "elapsed": time.time() - t0, "steps": steps, "tok_per_step": tok_per_step}, tmp)
    os.replace(tmp, ckpt_path)


print(f"{name}: {raw.num_params(False) / 1e6:.2f}M params ({raw.num_params() / 1e6:.2f}M non-emb), "
      f"{steps} steps x {tok_per_step} tok, cache/token={raw.cache_scalars_per_token()} scalars, "
      f"compile={args.compile}, {torch.cuda.get_device_name()}", flush=True)

# ----------------------------------------------------------------------------- training loop
t0 = time.time() - elapsed
win_t, win_step = time.time(), start
for step in range(start, steps):
    if step % args.eval_every == 0:
        vl = evaluate(args.eval_batches)
        log.write(json.dumps({"step": step, "tokens": step * tok_per_step, "val_loss": vl, "time": time.time() - t0}) + "\n")
        log.flush()
        print(f"step {step:5d} | val {vl:.4f} | {time.time() - t0:.0f}s", flush=True)
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    tl = 0.0
    for a in range(args.accum):
        x, y = get_train(step * args.accum + a)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / args.accum).backward()
        tl += loss.item() / args.accum
    if not math.isfinite(tl):
        raise RuntimeError(f"non-finite training loss at step {step}; resume restarts from the last checkpoint")
    clip = zclip.step(raw.parameters())
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 20 == 0:
        now = time.time()
        tps = (step + 1 - win_step) * tok_per_step / max(now - win_t, 1e-9)
        win_t, win_step = now, step + 1
        log.write(json.dumps({"step": step, "train_loss": tl, "grad_norm": clip["grad_norm"], "clipped_norm": clip["clipped_norm"],
                              "zclipped": clip["zclipped"], "lr": lr_at(step), "tok_s": tps}) + "\n")
        log.flush()
        print(f"step {step:5d} | loss {tl:.4f} | gn {clip['grad_norm']:.2f} | {tps / 1e3:.1f}k tok/s", flush=True)
    if (step + 1) % args.ckpt_every == 0 and step + 1 < steps:
        save_ckpt(step + 1)

# ----------------------------------------------------------------------------- final eval + save
vl = evaluate(args.final_eval_batches)
log.write(json.dumps({"step": steps, "tokens": steps * tok_per_step, "val_loss": vl, "time": time.time() - t0,
                      "final": True, "eval_tokens": args.final_eval_batches * 8 * T}) + "\n")
log.close()
print(f"final val {vl:.4f} over {args.final_eval_batches * 8 * T:,} tokens", flush=True)
torch.save({"cfg": cfg.to_dict(), "model": raw.state_dict()}, run_dir / "model.pt")
json.dump({"attn": args.attn, "n_group": args.n_group, "seq_len": args.seq_len, "seed": args.seed,
           "params": raw.num_params(False), "non_emb_params": raw.num_params(),
           "cache_scalars_per_token": raw.cache_scalars_per_token(),
           "final_val_loss": vl, "final_eval_tokens": args.final_eval_batches * 8 * T,
           "train_time_s": time.time() - t0, "tokens": steps * tok_per_step, "steps": steps,
           "zclip_clips": zclip.n_clipped, "gpu": torch.cuda.get_device_name(), "compile": args.compile,
           "args": vars(args)},
          open(run_dir / "summary.json", "w"), indent=2)
if ckpt_path.exists():
    ckpt_path.unlink()
print("saved", run_dir)
