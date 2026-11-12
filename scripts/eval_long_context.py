"""Long-context evaluation: held-out FineWeb-Edu loss at contexts beyond the training length.

For every run and context length, reports the mean loss and the loss per position bucket (where in
the window the model starts to break down). With --ntk, each context longer than the training length
is also evaluated with NTK-aware RoPE scaling (factor = context / training length), the standard
inference-time context-extension trick, as a baseline. Writes runs/<name>/long_context.json.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from gka.paths import DATA_DIR
from gka.runs import DEFAULT_RUNS, load_model, run_dir

DEV = "cuda"


@torch.no_grad()
def eval_context(model, val, ctx, n_tokens, bucket):
    """Mean loss and per-bucket loss over n_tokens of held-out data split into ctx-long windows."""
    n_seq = max(1, n_tokens // ctx)
    bs = max(1, 16384 // ctx)
    tot = torch.zeros(ctx, dtype=torch.float64, device=DEV)  # summed loss per position
    for s in range(0, n_seq, bs):
        b = min(bs, n_seq - s)
        x = np.stack([val[(s + i) * ctx : (s + i) * ctx + ctx + 1] for i in range(b)]).astype(np.int64)
        x = torch.from_numpy(x).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.hidden(x[:, :-1])
        # LM head in chunks: full fp32 logits for 16k tokens would be 3 GB
        for i in range(0, ctx, 2048):
            logits = model.lm_head(h[:, i : i + 2048]).float()
            loss = F.cross_entropy(logits.transpose(1, 2), x[:, 1 + i : 1 + i + 2048], reduction="none")
            tot[i : i + 2048] += loss.sum(0).double()
    per_pos = (tot / n_seq).cpu().numpy()
    return {
        "loss": float(per_pos.mean()),
        "bucket": bucket,
        "bucket_loss": [float(per_pos[i : i + bucket].mean()) for i in range(0, ctx, bucket)],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run names, e.g. gqa gka gka_c2048_s1")
    p.add_argument("--contexts", default="512,1024,2048,4096,8192,16384")
    p.add_argument("--ntk", action="store_true", help="also evaluate NTK RoPE scaling beyond the training length")
    p.add_argument("--tokens", type=int, default=262_144, help="held-out tokens per context length")
    p.add_argument("--bucket", type=int, default=256, help="position bucket size for the loss-vs-position curve")
    args = p.parse_args()
    val = np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r")
    contexts = [int(c) for c in args.contexts.split(",")]
    assert max(contexts) * max(1, args.tokens // max(contexts)) < len(val), "not enough held-out tokens"

    for run in args.runs:
        model = load_model(run, DEV)
        train_len = model.cfg.seq_len
        out = {"train_len": train_len, "plain": {}, "ntk": {}}
        for ctx in contexts:
            model.set_rope_ntk(1.0)
            out["plain"][ctx] = eval_context(model, val, ctx, args.tokens, args.bucket)
            line = f"{run:16s} ctx {ctx:6d}  plain {out['plain'][ctx]['loss']:.4f}"
            if args.ntk and ctx > train_len:
                model.set_rope_ntk(ctx / train_len)
                out["ntk"][ctx] = eval_context(model, val, ctx, args.tokens, args.bucket)
                line += f"  ntk {out['ntk'][ctx]['loss']:.4f}"
            print(line, flush=True)
        model.set_rope_ntk(1.0)
        json.dump(out, open(run_dir(run) / "long_context.json", "w"), indent=2)
        del model
        torch.cuda.empty_cache()
