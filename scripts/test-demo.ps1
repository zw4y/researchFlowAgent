$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$conda = "C:\Users\lenovo\anaconda3\Scripts\conda.exe"

Set-Location -LiteralPath $projectRoot
& $conda run --no-capture-output -n researchflow python -m ruff check backend
& $conda run --no-capture-output -n researchflow python -m mypy backend/app
& $conda run --no-capture-output -n researchflow python -m pytest --basetemp .pytest-conda

