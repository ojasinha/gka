"""Stream FineWeb-Edu (sample-10BT), tokenize with GPT-2 BPE, write uint16 train/val shards.

Defaults: 4M validation tokens (taken first from the stream, so they never overlap training documents)
and 1.02B training tokens (1B for training plus slack). Files are written to *.tmp and renamed when
complete, so an interrupted run never leaves a truncated train.bin behind.
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from gka.paths import DATA_DIR

p = argparse.ArgumentParser()
p.add_argument("--train_tokens", type=int, default=1_020_000_000)
p.add_argument("--val_tokens", type=int, default=4_000_000)
p.add_argument("--out", default=str(DATA_DIR))
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
assert enc.n_vocab <= 2**16, "uint16 shards need a vocabulary below 65536"
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)


def documents():
    """Token lists, one per document (EOT-prefixed), in stream order. Tokenizes batches on all cores."""
    for batch in ds.iter(batch_size=1024):
        for toks in enc.encode_ordinary_batch(batch["text"], num_threads=os.cpu_count() or 8):
            yield [eot] + toks


def fill(docs, n, path):
    tmp = path + ".tmp"
    buf = np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(n,))
    i = 0
    with tqdm(total=n, desc=os.path.basename(path), unit="tok", unit_scale=True) as bar:
        for toks in docs:
            k = min(len(toks), n - i)
            buf[i : i + k] = toks[:k]
            i += k
            bar.update(k)
            if i >= n:
                break
    assert i == n, f"stream ended after {i:,} of {n:,} tokens"
    buf.flush()
    del buf
    os.replace(tmp, path)


docs = documents()
fill(docs, args.val_tokens, os.path.join(args.out, "val.bin"))
fill(docs, args.train_tokens, os.path.join(args.out, "train.bin"))
print("done")
