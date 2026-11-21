#!/usr/bin/env bash
# The PLAN.md experiment on a Colab A100, in resumable stages.
#
# In a Colab notebook (Runtime -> Change runtime type -> A100):
#   from google.colab import drive; drive.mount('/content/drive')
#   !git clone -b from-scratch https://github.com/ojasinha/gka.git /content/gka
#   !bash /content/gka/scripts/colab.sh setup     # uv + dependencies, tokenized data (built once, cached on Drive)
#   !bash /content/gka/scripts/colab.sh smoke     # every model: compiled train + checkpoint + eval + save (~15 min)
#   !bash /content/gka/scripts/colab.sh timing    # tok/s per model -> hours and CU per 1B-token run
#   !bash /content/gka/scripts/colab.sh train     # the 19 runs; re-run the same command after a disconnect
#   !bash /content/gka/scripts/colab.sh eval      # zero-shot benchmarks + long-context eval (plain and NTK)
#   !bash /content/gka/scripts/colab.sh bench     # inference benchmark
#   !bash /content/gka/scripts/colab.sh analyze   # results/*.md and figures
#
# Every new Colab session needs `setup` again (it reuses the data cached on Drive, ~2 min).
# `train` with OPTIONAL=1 also trains the optional runs (gka_g4) after the main ones.
#
# Runs, results and logs live on Google Drive ($GKA_DRIVE, default MyDrive/gka), so nothing is lost when
# the runtime disconnects. The tokenized data is kept on local disk for fast random access.
set -euo pipefail
cd "$(dirname "$0")/.."

DRIVE="${GKA_DRIVE:-/content/drive/MyDrive/gka}"
export GKA_DATA_DIR="${GKA_DATA_DIR:-/content/data}"
export GKA_RUNS_DIR="$DRIVE/runs" GKA_RESULTS_DIR="$DRIVE/results" GKA_LOGS_DIR="$DRIVE/logs"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$GKA_RUNS_DIR" "$GKA_RESULTS_DIR" "$GKA_LOGS_DIR"
log() { echo "$(date -Iseconds) $*" | tee -a "$GKA_LOGS_DIR/pipeline.log"; }
py() { uv run --frozen python "$@"; }
# python list -> bash words, e.g. runs DEFAULT_RUNS
runs() { py -c "import gka.runs as r; print(' '.join(r.$1))"; }

case "${1:-}" in
setup)
  [[ -d /content/drive/MyDrive ]] || { echo "mount Google Drive first: from google.colab import drive; drive.mount('/content/drive')"; exit 1; }
  command -v uv > /dev/null || pip install -q uv
  uv sync --frozen
  py -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
  mkdir -p "$GKA_DATA_DIR" "$DRIVE/data"
  if [[ ! -f "$GKA_DATA_DIR/train.bin" ]]; then
    if [[ -f "$DRIVE/data/train.bin" && -f "$DRIVE/data/val.bin" ]]; then
      log "copy data from Drive"
      cp "$DRIVE/data/val.bin" "$DRIVE/data/train.bin" "$GKA_DATA_DIR/"
    else
      log "prepare data (first session only)"
      py scripts/prepare_data.py 2>&1 | tee "$GKA_LOGS_DIR/prepare_data.log"
      cp "$GKA_DATA_DIR/val.bin" "$GKA_DATA_DIR/train.bin" "$DRIVE/data/"
    fi
  fi
  ls -la "$GKA_DATA_DIR"
  log "setup done"
  ;;

smoke)
  # Short runs at the real batch settings with --compile, into a throwaway folder on local disk.
  out=/content/smoke
  rm -rf "$out"
  for name in $(runs MODELS); do
    log "smoke $name"
    py scripts/train.py --run "$name" --compile --tokens $((20 * 131072)) --warmup 5 --eval_every 10 \
      --eval_batches 2 --final_eval_batches 4 --ckpt_every 10 --out "$out" 2>&1 | tee "$GKA_LOGS_DIR/smoke_$name.log" | grep -E "params|final val|Error"
    [[ -f "$out/$name/summary.json" && -f "$out/$name/model.pt" ]] || { log "smoke $name FAILED"; exit 1; }
  done
  log "smoke passed for: $(runs MODELS)"
  ;;

timing)
  py scripts/bench_train_throughput.py --compile 2>&1 | tee "$GKA_LOGS_DIR/timing.log"
  ;;

train)
  todo=$(runs DEFAULT_RUNS)
  [[ "${OPTIONAL:-0}" == 1 ]] && todo="$todo $(runs OPTIONAL_MODELS)"
  for name in $todo; do
    if [[ -f "$GKA_RUNS_DIR/$name/summary.json" ]]; then log "skip $name (done)"; continue; fi
    log "train $name"
    py scripts/train.py --run "$name" --compile >> "$GKA_LOGS_DIR/train_$name.log" 2>&1 \
      || { log "train $name FAILED, see $GKA_LOGS_DIR/train_$name.log"; tail -20 "$GKA_LOGS_DIR/train_$name.log"; exit 1; }
    log "train $name done: $(py -c "import json; print(round(json.load(open('$GKA_RUNS_DIR/$name/summary.json'))['final_val_loss'], 4))")"
  done
  log "all training done"
  ;;

eval)
  done_runs=$(for d in "$GKA_RUNS_DIR"/*/; do [[ -f "$d/model.pt" ]] && basename "$d"; done | tr '\n' ' ')
  [[ -n "$done_runs" ]] || { echo "no finished runs"; exit 1; }
  log "zero-shot eval: $done_runs"
  py scripts/evaluate.py --skip_done --runs $done_runs > >(tee -a "$GKA_LOGS_DIR/eval.log" | grep --line-buffered "5-task avg" | cut -c1-80) 2>&1
  log "long-context eval: $done_runs"
  py scripts/eval_long_context.py --ntk --skip_done --runs $done_runs 2>&1 | tee -a "$GKA_LOGS_DIR/long_context.log"
  log "eval done"
  ;;

bench)
  py scripts/bench_inference.py 2>&1 | tee "$GKA_LOGS_DIR/bench_inference.log"
  ;;

analyze)
  py scripts/analyze.py
  py scripts/analyze_long_context.py > /dev/null
  log "results in $GKA_RESULTS_DIR"
  ;;

*)
  sed -n '2,20p' "$0"
  exit 1
  ;;
esac
