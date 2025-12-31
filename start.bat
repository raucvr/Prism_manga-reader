@echo off
title Prism AI

echo Starting Prism AI...
echo.

:: Start backend
start "Prism Backend" cmd /k "cd /d "%~dp0backend" && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

:: Wait a moment
timeout /t 2 /nobreak >nul

:: Start frontend
start "Prism Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

:: Wait for services
timeout /t 4 /nobreak >nul

echo.
echo Prism AI Started!
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8000
echo.

:: Open browser
start http://localhost:3000
