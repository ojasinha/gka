"""Stream FineWeb-Edu (sample-10BT), tokenize with GPT-2 BPE, write uint16 train/val shards."""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from gka.paths import DATA_DIR

p = argparse.ArgumentParser()
p.add_argument("--train_tokens", type=int, default=45_000_000)
p.add_argument("--val_tokens", type=int, default=2_000_000)
p.add_argument("--out", default=str(DATA_DIR))
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)


def fill(it, n, path):
    buf = np.empty(n, dtype=np.uint16)
    i = 0
    with tqdm(total=n, desc=path, unit="tok", unit_scale=True) as bar:
        for batch in it:
            toks = [eot] + enc.encode_ordinary(batch["text"])
            k = min(len(toks), n - i)
            buf[i : i + k] = toks[:k]
            i += k
            bar.update(k)
            if i >= n:
                break
    buf.tofile(path)


it = iter(ds)
# Validation comes first in the stream so it never overlaps training documents.
fill(it, args.val_tokens, os.path.join(args.out, "val.bin"))
fill(it, args.train_tokens, os.path.join(args.out, "train.bin"))
print("done")
