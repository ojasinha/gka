# Full pipeline: prepare data -> train every run -> zero-shot eval -> inference bench -> analysis.
# Each stage skips work that is already done, so the script is safe to re-launch after an interruption.
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
New-Item -ItemType Directory -Force logs | Out-Null
function Log($m) { $line = "$(Get-Date -Format s) $m"; Write-Host $line; Add-Content logs/pipeline.log $line }

# run name -> train.py arguments (see gka/runs.py for the naming scheme)
$runs = [ordered]@{
    'gqa'       = @('--attn', 'gqa')
    'gka'       = @('--attn', 'gka')
    'gka_drope' = @('--attn', 'gka_drope')
    'gva_drope' = @('--attn', 'gva_drope')
    'gva'       = @('--attn', 'gva')
    'sharedkv'  = @('--attn', 'sharedkv')
    'gqa_g1'    = @('--attn', 'gqa', '--n_group', '1')
    'gka_g4'    = @('--attn', 'gka', '--n_group', '4')
}

if (-not (Test-Path data/train.bin)) {
    Log 'prepare data'
    uv run python scripts/prepare_data.py *> logs/prepare_data.log
}

foreach ($name in $runs.Keys) {
    if (Test-Path "runs/$name/summary.json") { Log "skip train $name"; continue }
    Log "train $name"
    $trainArgs = $runs[$name]
    uv run python scripts/train.py @trainArgs *> "logs/train_$name.log"
    Log "train $name exit=$LASTEXITCODE"
}

$todo = @($runs.Keys | Where-Object { (Test-Path "runs/$_/model.pt") -and -not (Test-Path "runs/$_/benchmarks.json") })
if ($todo.Count) {
    Log "eval $($todo -join ' ')"
    uv run python scripts/evaluate.py --runs @todo *> logs/eval.log
    Log "eval exit=$LASTEXITCODE"
}

Log 'bench inference'
$all = @($runs.Keys)
uv run python scripts/bench_inference.py --runs @all *> logs/bench_inference.log
Log "bench exit=$LASTEXITCODE"

uv run python scripts/analyze.py *> logs/analyze.log
Log "analyze exit=$LASTEXITCODE -> results/results.md"
