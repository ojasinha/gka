"""Aggregate runs/*/long_context.json into results/long_context.md and loss-vs-position plots."""

import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from gka.paths import FIGURES_DIR, RESULTS_DIR, RUNS_DIR
from gka.runs import parse_run_name
from gka.style import COLOR, LABEL, ORDER

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# model key (as in gka.style, e.g. "gqa_g1") -> train_len -> list of per-seed results
groups = defaultdict(lambda: defaultdict(list))
for path in sorted(RUNS_DIR.glob("*/long_context.json")):
    spec = parse_run_name(path.parent.name)
    key = spec["attn"] + ("" if spec["n_group"] == 2 else f"_g{spec['n_group']}")
    res = json.load(open(path))
    groups[key][res["train_len"]].append(res)

train_lens = sorted({t for g in groups.values() for t in g})
models = [m for m in ORDER if m in groups]


def stat(results, mode, ctx):
    xs = [r[mode][ctx]["loss"] for r in results if ctx in r[mode]]
    if not xs:
        return None
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return mean, std


def fmt(s):
    if s is None:
        return "-"
    return f"{s[0]:.3f}" + (f" ± {s[1]:.3f}" if s[1] is not None else "")


lines = ["# Long-context evaluation\n",
         "Held-out FineWeb-Edu loss (lower is better) when running models beyond their training context. "
         "**plain** = RoPE as trained; **NTK** = inference-time NTK-aware RoPE scaling with factor context/training length. "
         "Mean ± std across seeds.\n"]
for tl in train_lens:
    rows = [(m, groups[m][tl]) for m in models if tl in groups[m]]
    ctxs = sorted({int(c) for _, rs in rows for r in rs for c in r["plain"]})
    longer = [c for c in ctxs if c > tl]
    lines.append(f"## Trained at {tl} tokens\n")
    hdr = ["Model", "Seeds"] + [f"plain @{c}" for c in ctxs] + [f"NTK @{c}" for c in longer]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for m, rs in rows:
        cells = [LABEL[m], str(len(rs))]
        cells += [fmt(stat(rs, "plain", str(c))) for c in ctxs]
        cells += [fmt(stat(rs, "ntk", str(c))) for c in longer]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

# loss vs position at the longest context: one row per training length, plain vs NTK
fig, axes = plt.subplots(len(train_lens), 2, figsize=(12, 4.2 * len(train_lens)), squeeze=False, sharey="row")
for row, tl in enumerate(train_lens):
    for col, mode in enumerate(["plain", "ntk"]):
        ax = axes[row][col]
        for m in models:
            rs = groups[m].get(tl, [])
            if not rs:
                continue
            ctx = str(max(int(c) for c in rs[0][mode])) if rs[0][mode] else None
            if ctx is None:
                continue
            curves = np.array([r[mode][ctx]["bucket_loss"] for r in rs if ctx in r[mode]])
            b = rs[0][mode][ctx]["bucket"]
            x = (np.arange(curves.shape[1]) + 0.5) * b
            ax.plot(x, curves.mean(0), lw=2, color=COLOR[m], label=f"{LABEL[m]} ({len(curves)} seed{'s' * (len(curves) > 1)})")
        ax.axvline(tl, color="#52514e", lw=1, ls="--")
        ax.text(tl, ax.get_ylim()[1] if ax.lines else 0, " training length", fontsize=8, color="#52514e", va="top")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("token position in the window")
        ax.set_title(f"Trained at {tl}: {'plain RoPE' if mode == 'plain' else 'NTK-scaled RoPE'}")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[row][0].set_ylabel("loss at that position")
axes[0][0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "long_context.png", dpi=150)
lines.append("![Loss by token position](figures/long_context.png)\n")

open(RESULTS_DIR / "long_context.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
