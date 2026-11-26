Write-Host "Starting SyllabAI Backend (FastAPI on :8000) & Frontend (Vite on :5173)..." -ForegroundColor Cyan

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; python -m uvicorn backend.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root\frontend'; pnpm dev"

Write-Host "Open http://localhost:5173 once both servers finish booting." -ForegroundColor Green
