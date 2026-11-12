"""Collect training logs, benchmark results and inference numbers into results/ (tables + plots)."""

import json
import math

import matplotlib.pyplot as plt

from gka.paths import FIGURES_DIR, RESULTS_DIR, RUNS_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ORDER = ["gqa", "gka", "gka_drope", "gva_drope", "gva", "sharedkv", "gqa_g1", "gka_g4"]
LABEL = {
    "gqa": "GQA (baseline)",
    "gka": "GKA (ours, RoPE)",
    "gka_drope": "GKA + DRoPE16 (ours)",
    "gva_drope": "GVA + DRoPE16",
    "gva": "GVA (RoPE, no absorb)",
    "sharedkv": "Shared KV (K=V)",
    "gqa_g1": "MQA (GQA, G=1)",
    "gka_g4": "GKA, G=4 (ours)",
}
# Reference categorical palette (dataviz skill), slots 1-6 in fixed order, light mode.
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))
MC_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "openbookqa", "piqa", "winogrande"]
CHANCE = {"hellaswag": 25, "arc_easy": 25, "arc_challenge": 25, "openbookqa": 25, "piqa": 50, "winogrande": 50}


def runs_for(v):
    """All seeds of a variant: runs/<v> and runs/<v>_s<seed>."""
    return [d for d in [RUNS_DIR / v] + sorted(RUNS_DIR.glob(f"{v}_s[0-9]*")) if (d / "summary.json").exists()]


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


variants = [v for v in ORDER if runs_for(v)]
INF_PATH = RESULTS_DIR / "inference.json"
inf = json.load(open(INF_PATH)) if INF_PATH.exists() else {}

# ---------------------------------------------------------------- training curves
fig, ax = plt.subplots(figsize=(8, 5))
for v in variants:
    recs = [json.loads(l) for l in open(runs_for(v)[0] / "log.jsonl")]
    ev = [r for r in recs if "val_loss" in r and r["step"] > 0]
    ax.plot([r["tokens"] / 1e6 for r in ev], [r["val_loss"] for r in ev], lw=2, color=COLOR[v], label=LABEL[v])
ax.set_xlabel("training tokens (M)")
ax.set_ylabel("validation loss (FineWeb-Edu)")
ax.set_ylim(top=5.0)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.set_title("Validation loss during training")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "val_loss.png", dpi=150)

# zoomed final third
ax.set_xlim(left=ax.get_xlim()[1] * 0.6)
lo = min(json.load(open(runs_for(v)[0] / "summary.json"))["final_val_loss"] for v in variants)
ax.set_ylim(lo - 0.03, lo + 0.35)
ax.set_title("Validation loss (final 40% of training)")
fig.savefig(FIGURES_DIR / "val_loss_zoom.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- tables
lines = []
P = lines.append
P("## Quality\n")
P("Seeds per variant: " + ", ".join(f"{v}={len(runs_for(v))}" for v in variants) + "  (mean ± std across seeds where >1)\n")
hdr = ["Method", "Cache/token/layer", "vs GQA", "Params", "Val loss", "Val PPL"] + \
      ["HellaSwag", "ARC-E", "ARC-C", "OBQA", "PIQA", "WinoG", "Avg (6)", "LAMBADA acc", "LAMBADA ppl"]
P("| " + " | ".join(hdr) + " |")
P("|" + "---|" * len(hdr))
quality = {}
for v in variants:
    rs = runs_for(v)
    summ = [json.load(open(r / "summary.json")) for r in rs]
    bm = [json.load(open(r / "benchmarks.json")) for r in rs if (r / "benchmarks.json").exists()]
    cache = inf.get(v, {}).get("cache_scalars_per_token_per_layer")
    gqa_cache = inf.get("gqa", {}).get("cache_scalars_per_token_per_layer")
    vl = [s["final_val_loss"] for s in summ]

    def fmt(xs, scale=1.0, d=2):
        xs = [x * scale for x in xs]
        s = f"{mean(xs):.{d}f}"
        return s + (f" ± {std(xs):.{d}f}" if len(xs) > 1 else "")

    row = [LABEL[v], str(cache or "-"), f"{100 * (1 - cache / gqa_cache):.1f}% smaller" if cache and gqa_cache and v != "gqa" else "-",
           f"{summ[0]['params'] / 1e6:.2f}M", fmt(vl, d=4), fmt([math.exp(x) for x in vl])]
    q = {"val_loss": mean(vl)}
    if bm:
        accs = []
        for t in MC_TASKS:
            key = "acc_norm" if t in ("hellaswag", "arc_easy", "arc_challenge", "openbookqa", "piqa") else "acc"
            xs = [b[t][key] for b in bm]
            row.append(fmt(xs, 100, 1))
            accs.append([x * 100 for x in xs])
            q[t] = mean(xs) * 100
        per_seed_avg = [mean([a[i] for a in accs]) for i in range(len(bm))]
        row.append(fmt(per_seed_avg, 1, 2))
        row.append(fmt([b["lambada"]["acc"] for b in bm], 100, 1))
        row.append(fmt([b["lambada"]["ppl"] for b in bm], 1, 0))
        q["avg"] = mean(per_seed_avg)
        q["ctx"] = {c: mean([b[f"val_ctx{c}"]["loss"] for b in bm]) for c in (512, 1024, 2048)}
        q["tail"] = {c: mean([b[f"val_ctx{c}"]["tail256_loss"] for b in bm]) for c in (512, 1024, 2048)}
    else:
        row += ["-"] * 9
    quality[v] = q
    P("| " + " | ".join(row) + " |")
P("\nAccuracies in %. HellaSwag/ARC/OBQA/PIQA use length-normalised acc (acc_norm); WinoGrande uses partial scoring. "
  "Chance: " + ", ".join(f"{k} {c}%" for k, c in CHANCE.items()) + ".\n")

if any("ctx" in q for q in quality.values()):
    P("## Length extrapolation (trained at 512 tokens)\n")
    P("| Method | loss @512 | loss @1024 | loss @2048 | last-256-token loss @2048 |")
    P("|---|---|---|---|---|")
    for v in variants:
        q = quality[v]
        if "ctx" in q:
            P(f"| {LABEL[v]} | {q['ctx'][512]:.4f} | {q['ctx'][1024]:.4f} | {q['ctx'][2048]:.4f} | {q['tail'][2048]:.4f} |")
    P("")

if inf:
    P("## Inference (RTX 3050 6GB laptop, bf16, PyTorch eager, 64 decoded tokens)\n")
    settings = [(r["B"], r["ctx"]) for r in inf[variants[0]]["runs"]]
    P("| Method | KV bytes/token (all layers) | " + " | ".join(f"cache MB B={b} T={t}" for b, t in settings) + " |")
    P("|---|---|" + "---|" * len(settings))
    for v in variants:
        r = inf[v]
        P(f"| {LABEL[v]} | {r['cache_bytes_per_token_bf16']} | " +
          " | ".join("OOM" if x.get("oom") else f"{x['cache_MB']:.1f}" for x in r["runs"]) + " |")
    P("")
    P("| Method | " + " | ".join(f"decode tok/s B={b} T={t}" for b, t in settings) + " | " +
      " | ".join(f"prefill ms B={b} T={t}" for b, t in settings) + " |")
    P("|---|" + "---|" * (2 * len(settings)))
    for v in variants:
        r = inf[v]["runs"]
        P(f"| {LABEL[v]} | " + " | ".join("OOM" if x.get("oom") else f"{x['decode_tok_s']:.0f}" for x in r) + " | " +
          " | ".join("OOM" if x.get("oom") else f"{x['prefill_ms']:.0f}" for x in r) + " |")
    P("")
    P("| Method | " + " | ".join(f"decode peak MB B={b} T={t}" for b, t in settings) + " |")
    P("|---|" + "---|" * len(settings))
    for v in variants:
        r = inf[v]["runs"]
        P(f"| {LABEL[v]} | " + " | ".join("OOM" if x.get("oom") else f"{x['peak_extra_MB']:.0f}" for x in r) + " |")
    P("")

    # quality vs cache scatter (<=3 series rule doesn't apply: each point is direct-labelled)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        if v in inf and "val_loss" in quality[v]:
            x = inf[v]["cache_bytes_per_token_bf16"]
            y = quality[v]["val_loss"]
            ax.scatter(x, y, s=70, color=COLOR[v], edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(LABEL[v], (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9, color="#0b0b0b")
    ax.set_xlabel("KV cache bytes per token (bf16, all layers)")
    ax.set_ylabel("final validation loss")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Quality vs KV-cache size (lower-left is better)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_vs_cache.png", dpi=150)
    plt.close(fig)

open(RESULTS_DIR / "results.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
