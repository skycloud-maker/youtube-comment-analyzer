@echo off
setlocal

cd /d C:\codex
title VoC Dashboard Preview

set PORT=8505
set URL=http://127.0.0.1:%PORT%

echo [1/3] 기존 %PORT% 포트 점유 프로세스 정리...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo [2/3] Streamlit 서버 시작...
start "" /B C:\codex\python312\python.exe -m streamlit run C:\codex\streamlit_app.py --server.headless=true --server.port=%PORT% --server.address=127.0.0.1 > C:\codex\streamlit_preview_out.log 2> C:\codex\streamlit_preview_err.log

echo [3/3] 브라우저 오픈...
timeout /t 5 /nobreak >nul
start "" "%URL%"

echo.
echo Preview URL: %URL%
echo 종료하려면: Ctrl+C 후 필요 시 taskkill /F /IM python.exe
echo 로그: C:\codex\streamlit_preview_out.log / C:\codex\streamlit_preview_err.log

endlocal
