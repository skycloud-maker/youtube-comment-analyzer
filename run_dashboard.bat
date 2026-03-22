@echo off
cd /d C:\codex
title VoC Dashboard Server

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8505 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

:restart
echo [VoC Dashboard] Starting Streamlit server on http://127.0.0.1:8505
C:\codex\python312\python.exe -m streamlit run C:\codex\streamlit_app.py --server.headless=true --server.port=8505 --server.address=127.0.0.1
echo.
echo [VoC Dashboard] Server stopped. Restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto restart
