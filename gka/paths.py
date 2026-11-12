"""Repository layout. Every script resolves paths through here, so they work from any directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"  # tokenized FineWeb-Edu shards (train.bin, val.bin)
RUNS_DIR = ROOT / "runs"  # one folder per trained model: log.jsonl, summary.json, benchmarks.json, model.pt
RESULTS_DIR = ROOT / "results"  # aggregated tables (results.md, inference.json)
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"  # raw stdout logs from run_all
