@echo off
title SyllabAI Launcher
echo Starting SyllabAI Backend (FastAPI :8000) and Frontend (Vite :5173)...

cd /d "%~dp0"

start "SyllabAI Backend (FastAPI)" cmd /k "python -m uvicorn backend.main:app --reload --port 8000"
start "SyllabAI Frontend (Vite)" cmd /k "cd frontend && pnpm dev"

timeout /t 5 /nobreak >nul
start http://localhost:5173
echo Both servers launched! Open http://localhost:5173 in your browser.
