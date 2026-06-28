$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$conda = "C:\Users\lenovo\anaconda3\Scripts\conda.exe"

if (-not (Test-Path -LiteralPath $conda)) {
    throw "Conda executable not found: $conda"
}

Set-Location -LiteralPath $projectRoot
Write-Host "ResearchFlow API: http://127.0.0.1:8000"
Write-Host "Press Ctrl+C to stop."
& $conda run --no-capture-output -n researchflow python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

