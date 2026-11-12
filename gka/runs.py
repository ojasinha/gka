"""Run naming and checkpoint loading shared by the training, evaluation and benchmark scripts.

A run is named after its attention variant, with suffixes for non-default settings:
  gka         -> attn=gka, n_group=2, seed=1337
  gka_g4      -> attn=gka, n_group=4
  gka_g4_s7   -> attn=gka, n_group=4, seed=7
  gka_c2048   -> attn=gka, trained with seq_len=2048
"""

import re
from pathlib import Path

import torch

from .model import GPT, Config, VARIANTS
from .paths import RUNS_DIR

DEFAULT_N_GROUP = Config.n_group
DEFAULT_SEED = 1337
DEFAULT_SEQ_LEN = Config.seq_len

# The eight runs reported in the README (all six variants + two equal-cache controls).
DEFAULT_RUNS = VARIANTS + ["gqa_g1", "gka_g4"]

_NAME = re.compile(rf"^({'|'.join(sorted(VARIANTS, key=len, reverse=True))})(?:_g(\d+))?(?:_c(\d+))?(?:_s(\d+))?$")


def run_name(attn: str, n_group: int = DEFAULT_N_GROUP, seed: int = DEFAULT_SEED, seq_len: int = DEFAULT_SEQ_LEN) -> str:
    return (attn + ("" if n_group == DEFAULT_N_GROUP else f"_g{n_group}")
            + ("" if seq_len == DEFAULT_SEQ_LEN else f"_c{seq_len}")
            + ("" if seed == DEFAULT_SEED else f"_s{seed}"))


def parse_run_name(name: str) -> dict:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a run name: {name!r}")
    return {
        "attn": m.group(1),
        "n_group": int(m.group(2) or DEFAULT_N_GROUP),
        "seq_len": int(m.group(3) or DEFAULT_SEQ_LEN),
        "seed": int(m.group(4) or DEFAULT_SEED),
    }


def run_dir(name: str) -> Path:
    return RUNS_DIR / name


def load_model(name: str, device="cuda", allow_random_init=False) -> GPT:
    """Load runs/<name>/model.pt. With allow_random_init, fall back to a fresh model of that config."""
    path = run_dir(name) / "model.pt"
    if path.exists():
        ck = torch.load(path, map_location=device)
        model = GPT(Config(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
    elif allow_random_init:
        spec = parse_run_name(name)
        model = GPT(Config(attn=spec["attn"], n_group=spec["n_group"], seq_len=spec["seq_len"])).to(device)
    else:
        raise FileNotFoundError(f"{path} not found; train it with scripts/train.py first")
    return model.eval()
