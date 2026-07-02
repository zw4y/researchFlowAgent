param(
    [string]$Dataset = "demo/evaluation/retrieval_cases.jsonl",
    [string]$Output = "data/evaluations/latest"
)

$ErrorActionPreference = "Stop"
python -m app.evaluation.cli --dataset $Dataset --output $Output
