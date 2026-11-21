"""Collect training logs, benchmark results and inference numbers into results/ (tables + plots)."""

import json
import math
import sys

import matplotlib.pyplot as plt

from gka.paths import FIGURES_DIR, RESULTS_DIR, RUNS_DIR
from gka.style import COLOR, LABEL, ORDER

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TASKS = ["hellaswag", "winogrande", "openbookqa", "arc_easy", "arc_challenge"]  # GVA paper, Table 2
EXTRA_TASKS = ["piqa"]
TASK_LABEL = {"hellaswag": "HellaSwag", "winogrande": "WinoG", "openbookqa": "OBQA", "arc_easy": "ARC-E",
              "arc_challenge": "ARC-C", "piqa": "PIQA"}
CHANCE = {"hellaswag": 25, "winogrande": 50, "openbookqa": 25, "arc_easy": 25, "arc_challenge": 25, "piqa": 50}


def runs_for(v):
    """All finished seeds of a model: runs/<v> and runs/<v>_s<seed>."""
    return [d for d in [RUNS_DIR / v] + sorted(RUNS_DIR.glob(f"{v}_s[0-9]*")) if (d / "summary.json").exists()]


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(xs, d=2):
    s = f"{mean(xs):.{d}f}"
    return s + (f" ± {std(xs):.{d}f}" if len(xs) > 1 else "")


def acc(b, t):
    return 100 * b[t]["acc_norm" if "acc_norm" in b[t] else "acc"]


variants = [v for v in ORDER if runs_for(v)]
if not variants:
    raise SystemExit(f"no finished runs in {RUNS_DIR}")
INF_PATH = RESULTS_DIR / "inference.json"
inf = json.load(open(INF_PATH)) if INF_PATH.exists() else {}
summ = {v: [json.load(open(r / "summary.json")) for r in runs_for(v)] for v in variants}
bench = {v: [json.load(open(r / "benchmarks.json")) for r in runs_for(v) if (r / "benchmarks.json").exists()] for v in variants}

# ---------------------------------------------------------------- training curves (first seed of each model)
fig, ax = plt.subplots(figsize=(8, 5))
for v in variants:
    recs = [json.loads(l) for l in open(runs_for(v)[0] / "log.jsonl")]
    ev = [r for r in recs if "val_loss" in r and r["step"] > 0 and not r.get("final")]
    ax.plot([r["tokens"] / 1e9 for r in ev], [r["val_loss"] for r in ev], lw=2, color=COLOR[v], label=LABEL[v])
ax.set_xlabel("training tokens (B)")
ax.set_ylabel("validation loss (FineWeb-Edu)")
lo = min(s["final_val_loss"] for v in variants for s in summ[v])
ax.set_ylim(lo - 0.05, lo + 1.5)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.set_title("Validation loss during training")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "val_loss.png", dpi=150)

# zoomed final 40%
ax.set_xlim(left=ax.get_xlim()[1] * 0.6)
ax.set_ylim(lo - 0.03, lo + 0.3)
ax.set_title("Validation loss (final 40% of training)")
fig.savefig(FIGURES_DIR / "val_loss_zoom.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- quality table
lines = []
P = lines.append
gqa_vl = mean([s["final_val_loss"] for s in summ["gqa"]]) if "gqa" in summ else None
gqa_cache = summ["gqa"][0]["cache_scalars_per_token"] if "gqa" in summ else None

P("## Quality at the training length\n")
P("Mean ± std across seeds (sample std; no ± means a single run). Val loss is measured on "
  f"{summ[variants[0]][0]['final_eval_tokens']:,} held-out FineWeb-Edu tokens at the training context.\n")
hdr = ["Method", "Seeds", "Cache vs GQA", "Params", "Non-emb params", "Val loss", "Δ vs GQA", "Val PPL"] + \
      [TASK_LABEL[t] for t in PAPER_TASKS] + ["**Avg (5, paper)**"] + [TASK_LABEL[t] for t in EXTRA_TASKS] + ["LAMBADA acc", "LAMBADA ppl"]
P("| " + " | ".join(hdr) + " |")
P("|" + "---|" * len(hdr))
for v in variants:
    ss, bm = summ[v], bench[v]
    vl = [s["final_val_loss"] for s in ss]
    cache = ss[0]["cache_scalars_per_token"]
    row = [LABEL[v], str(len(ss)), f"{100 * cache / gqa_cache:.1f}%" if gqa_cache else "-",
           f"{ss[0]['params'] / 1e6:.2f}M", f"{ss[0]['non_emb_params'] / 1e6:.2f}M", fmt(vl, 4),
           f"{mean(vl) - gqa_vl:+.4f}" if gqa_vl is not None and v != "gqa" else "-", fmt([math.exp(x) for x in vl], 1)]
    if bm:
        row += [fmt([acc(b, t) for b in bm], 1) for t in PAPER_TASKS]
        row.append("**" + fmt([mean([acc(b, t) for t in PAPER_TASKS]) for b in bm], 2) + "**")
        row += [fmt([acc(b, t) for b in bm], 1) for t in EXTRA_TASKS]
        row.append(fmt([100 * b["lambada"]["acc"] for b in bm], 1))
        row.append(fmt([b["lambada"]["ppl"] for b in bm], 0))
        if len(bm) != len(ss):
            row[1] += f" ({len(bm)} benchmarked)"
    else:
        row += ["-"] * (len(hdr) - len(row))
    P("| " + " | ".join(row) + " |")
P("\nAccuracies in %. HellaSwag/ARC/OBQA/PIQA use length-normalised accuracy (acc_norm); WinoGrande uses partial scoring. "
  "Avg (5, paper) is the unweighted mean of the GVA paper's five tasks, averaged per seed. "
  "Chance: " + ", ".join(f"{TASK_LABEL[k]} {c}%" for k, c in CHANCE.items()) + ".\n")

# ---------------------------------------------------------------- inference
inf_models = [v for v in variants if v in inf]
if inf_models:
    meta = inf.get("_meta", {})
    settings = [(r["B"], r["ctx"]) for r in inf[inf_models[0]]["runs"]]
    P(f"## Inference ({meta.get('gpu', 'GPU')}, bf16, PyTorch eager, 64 decoded tokens)\n")
    P("| Method | KV bytes/token (all layers) | " + " | ".join(f"cache MB B={b} T={t}" for b, t in settings) + " |")
    P("|---|---|" + "---|" * len(settings))
    for v in inf_models:
        r = inf[v]
        P(f"| {LABEL[v]} | {r['cache_bytes_per_token_bf16']} | " +
          " | ".join("OOM" if x.get("oom") else f"{x['cache_MB']:.1f}" for x in r["runs"]) + " |")
    P("")
    P("| Method | " + " | ".join(f"decode tok/s B={b} T={t}" for b, t in settings) + " | " +
      " | ".join(f"prefill ms B={b} T={t}" for b, t in settings) + " |")
    P("|---|" + "---|" * (2 * len(settings)))
    for v in inf_models:
        r = inf[v]["runs"]
        P(f"| {LABEL[v]} | " + " | ".join("OOM" if x.get("oom") else f"{x['decode_tok_s']:.0f}" for x in r) + " | " +
          " | ".join("OOM" if x.get("oom") else f"{x['prefill_ms']:.0f}" for x in r) + " |")
    P("")
    P("| Method | " + " | ".join(f"decode peak MB B={b} T={t}" for b, t in settings) + " |")
    P("|---|" + "---|" * len(settings))
    for v in inf_models:
        r = inf[v]["runs"]
        P(f"| {LABEL[v]} | " + " | ".join("OOM" if x.get("oom") else f"{x['peak_extra_MB']:.0f}" for x in r) + " |")
    P("")

# ---------------------------------------------------------------- quality vs cache scatter (direct-labelled)
fig, ax = plt.subplots(figsize=(7, 4.5))
for v in variants:
    x = 2 * summ[v][0]["cache_scalars_per_token"]  # bf16 bytes per token, all layers
    ys = [s["final_val_loss"] for s in summ[v]]
    ax.errorbar(x, mean(ys), yerr=std(ys) if len(ys) > 1 else None, fmt="o", ms=8, color=COLOR[v],
                mec="white", mew=2, capsize=3, zorder=3)
    ax.annotate(LABEL[v], (x, mean(ys)), textcoords="offset points", xytext=(8, 4), fontsize=9, color="#0b0b0b")
ax.set_xlabel("KV cache bytes per token (bf16, all layers)")
ax.set_ylabel("final validation loss (mean ± std over seeds)")
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Quality vs KV-cache size (lower-left is better)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "quality_vs_cache.png", dpi=150)
plt.close(fig)

open(RESULTS_DIR / "results.md", "w", encoding="utf-8").write("\n".join(lines))
sys.stdout.reconfigure(encoding="utf-8")  # tables contain ± and Δ; the Windows console defaults to cp1252
print("\n".join(lines))
