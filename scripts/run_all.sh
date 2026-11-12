#!/usr/bin/env bash
# Full pipeline: prepare data -> train every run -> zero-shot eval -> inference bench -> analysis.
# Each stage skips work that is already done, so the script is safe to re-launch after an interruption.
# Usage (from anywhere):  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
log() { echo "$(date -Iseconds) $*" | tee -a logs/pipeline.log; }

# run name -> train.py arguments (see gka/runs.py for the naming scheme)
declare -A RUNS=(
  [gqa]="--attn gqa"
  [gka]="--attn gka"
  [gka_drope]="--attn gka_drope"
  [gva_drope]="--attn gva_drope"
  [gva]="--attn gva"
  [sharedkv]="--attn sharedkv"
  [gqa_g1]="--attn gqa --n_group 1"
  [gka_g4]="--attn gka --n_group 4"
)
ORDER=(gqa gka gka_drope gva_drope gva sharedkv gqa_g1 gka_g4)

if [[ ! -f data/train.bin ]]; then
  log "prepare data"
  uv run python scripts/prepare_data.py > logs/prepare_data.log 2>&1
fi

for name in "${ORDER[@]}"; do
  if [[ -f "runs/$name/summary.json" ]]; then log "skip train $name"; continue; fi
  log "train $name"
  # shellcheck disable=SC2086
  uv run python scripts/train.py ${RUNS[$name]} > "logs/train_$name.log" 2>&1
  log "train $name done"
done

todo=()
for name in "${ORDER[@]}"; do
  [[ -f "runs/$name/model.pt" && ! -f "runs/$name/benchmarks.json" ]] && todo+=("$name")
done
if (( ${#todo[@]} )); then
  log "eval ${todo[*]}"
  uv run python scripts/evaluate.py --runs "${todo[@]}" > logs/eval.log 2>&1
fi

log "bench inference"
uv run python scripts/bench_inference.py --runs "${ORDER[@]}" > logs/bench_inference.log 2>&1

uv run python scripts/analyze.py > logs/analyze.log 2>&1
log "done -> results/results.md"
