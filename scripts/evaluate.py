"""Zero-shot evaluation of trained checkpoints (lm-eval-harness conventions).

Tasks: HellaSwag, ARC-Easy, ARC-Challenge, OpenBookQA, PIQA, WinoGrande (partial scoring),
LAMBADA (last-word accuracy + perplexity), and FineWeb-Edu held-out perplexity at context
512 (training length) plus 1024/2048 (length extrapolation).
"""

import argparse
import json
import math
import re

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from gka.paths import DATA_DIR
from gka.runs import DEFAULT_RUNS, load_model, run_dir

enc = tiktoken.get_encoding("gpt2")
DEV = "cuda"


# ----------------------------------------------------------------------------- scoring

@torch.no_grad()
def score_pairs(model, pairs, bs=32, max_len=512):
    """pairs: list of (context, continuation). Returns (sum logprob, all-greedy) per pair."""
    toks = []
    for ctx, cont in pairs:
        c = enc.encode_ordinary(ctx) if ctx else [enc.eot_token]
        k = enc.encode_ordinary(cont)
        seq = (c + k)[-(max_len + 1):]
        toks.append((seq, len(k)))
    order = sorted(range(len(toks)), key=lambda i: -len(toks[i][0]))
    out = [None] * len(toks)
    for s in range(0, len(order), bs):
        ids = order[s : s + bs]
        L = max(len(toks[i][0]) for i in ids) - 1
        x = torch.zeros(len(ids), L, dtype=torch.long)
        for r, i in enumerate(ids):
            seq = toks[i][0]
            x[r, : len(seq) - 1] = torch.tensor(seq[:-1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x.to(DEV))
        lp = F.log_softmax(logits.float(), -1)
        for r, i in enumerate(ids):
            seq, n = toks[i]
            tgt = torch.tensor(seq[-n:], device=DEV)
            l = lp[r, len(seq) - 1 - n : len(seq) - 1]
            out[i] = (l.gather(-1, tgt[:, None]).sum().item(), bool((l.argmax(-1) == tgt).all().item()))
    return out


def multiple_choice(model, items):
    """items: list of (context, [continuations], gold). Returns acc, acc_norm."""
    pairs, spans = [], []
    for ctx, conts, _ in items:
        spans.append((len(pairs), len(conts)))
        pairs += [(ctx, c) for c in conts]
    res = score_pairs(model, pairs)
    acc = accn = 0
    for (ctx, conts, gold), (s, n) in zip(items, spans):
        ll = np.array([res[s + j][0] for j in range(n)])
        ln = np.array([len(c) for c in conts])
        acc += int(ll.argmax() == gold)
        accn += int((ll / ln).argmax() == gold)
    return acc / len(items), accn / len(items)


# ----------------------------------------------------------------------------- tasks

def hs_pre(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def build_tasks():
    tasks = {}
    d = load_dataset("Rowan/hellaswag", split="validation")
    tasks["hellaswag"] = [
        (hs_pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()),
         [" " + hs_pre(e) for e in r["endings"]], int(r["label"])) for r in d]
    for name, sub in [("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")]:
        d = load_dataset("allenai/ai2_arc", sub, split="test")
        tasks[name] = [
            (f"Question: {r['question']}\nAnswer:", [" " + c for c in r["choices"]["text"]],
             r["choices"]["label"].index(r["answerKey"])) for r in d]
    d = load_dataset("allenai/openbookqa", "main", split="test")
    tasks["openbookqa"] = [
        (r["question_stem"], [" " + c for c in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"]))
        for r in d]
    d = load_dataset("baber/piqa", split="validation")
    tasks["piqa"] = [(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])) for r in d]
    d = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    wg = []
    for r in d:
        i = r["sentence"].index("_")
        cont = " " + r["sentence"][i + 1 :].strip()
        ctxs = [r["sentence"][:i] + r["option1"], r["sentence"][:i] + r["option2"]]
        wg.append((ctxs, cont, int(r["answer"]) - 1))
    tasks["winogrande"] = wg
    d = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    tasks["lambada"] = [(r["text"].rsplit(" ", 1)[0], " " + r["text"].rsplit(" ", 1)[1]) for r in d]
    return tasks


def eval_winogrande(model, items):
    pairs = [(c, cont) for ctxs, cont, _ in items for c in ctxs]
    res = score_pairs(model, pairs)
    acc = sum(int(np.argmax([res[2 * i][0], res[2 * i + 1][0]]) == g) for i, (_, _, g) in enumerate(items))
    return acc / len(items)


def eval_lambada(model, items):
    res = score_pairs(model, items)
    nll = -sum(r[0] for r in res)
    return sum(r[1] for r in res) / len(res), math.exp(nll / len(res))


@torch.no_grad()
def val_ppl(model, ctx, n_tokens=1_048_576):
    val = np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r")
    bs = max(1, 8 * 512 // ctx)
    n_seq = n_tokens // ctx
    tot, cnt = 0.0, 0
    # also track the loss on the final 256 positions (where extrapolation bites)
    tail, tcnt = 0.0, 0
    for s in range(0, n_seq, bs):
        b = min(bs, n_seq - s)
        x = torch.from_numpy(np.stack([val[(s + i) * ctx : (s + i) * ctx + ctx + 1] for i in range(b)]).astype(np.int64)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, :-1])
        l = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none")
        tot += l.sum().item()
        cnt += l.numel()
        tail += l[:, -256:].sum().item()
        tcnt += l[:, -256:].numel()
    return tot / cnt, tail / tcnt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run names, e.g. gqa gka gka_g4")
    args = p.parse_args()
    tasks = build_tasks()
    for run in args.runs:
        model = load_model(run, DEV)
        r = {}
        for t in ["hellaswag", "arc_easy", "arc_challenge", "openbookqa", "piqa"]:
            acc, accn = multiple_choice(model, tasks[t])
            r[t] = {"acc": acc, "acc_norm": accn}
            print(run, t, f"{acc:.4f} / {accn:.4f}", flush=True)
        r["winogrande"] = {"acc": eval_winogrande(model, tasks["winogrande"])}
        acc, ppl = eval_lambada(model, tasks["lambada"])
        r["lambada"] = {"acc": acc, "ppl": ppl}
        for ctx in [512, 1024, 2048]:
            loss, tail = val_ppl(model, ctx)
            r[f"val_ctx{ctx}"] = {"loss": loss, "ppl": math.exp(loss), "tail256_loss": tail}
        print(run, json.dumps(r), flush=True)
        json.dump(r, open(run_dir(run) / "benchmarks.json", "w"), indent=2)
        del model
        torch.cuda.empty_cache()
