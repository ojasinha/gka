"""Repository layout. Every script resolves paths through here, so they work from any directory.

Each location can be overridden with an environment variable (used on Colab to keep runs, results
and logs on Google Drive and the tokenized data on fast local disk).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GKA_DATA_DIR", ROOT / "data"))  # tokenized FineWeb-Edu (train.bin, val.bin)
RUNS_DIR = Path(os.environ.get("GKA_RUNS_DIR", ROOT / "runs"))  # one folder per trained model
RESULTS_DIR = Path(os.environ.get("GKA_RESULTS_DIR", ROOT / "results"))  # aggregated tables + figures
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = Path(os.environ.get("GKA_LOGS_DIR", ROOT / "logs"))  # raw stdout logs from the pipeline
