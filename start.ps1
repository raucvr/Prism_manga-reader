# Prism AI - PowerShell Startup Script
# Usage: Right-click and "Run with PowerShell" or run in terminal: powershell -ExecutionPolicy Bypass -File start.ps1

$Host.UI.RawUI.WindowTitle = "Prism AI - Starting..."

Write-Host ""
Write-Host "  =========================================" -ForegroundColor Cyan
Write-Host "       Prism AI - Starting...             " -ForegroundColor Cyan
Write-Host "  =========================================" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python not found! Please install Python first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Node.js
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Node.js not found! Please install Node.js first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/2] Starting Backend (FastAPI)..." -ForegroundColor Yellow

# Start backend in new window
$backendPath = Join-Path $ProjectRoot "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendPath'; `$Host.UI.RawUI.WindowTitle = 'Prism Backend'; python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

Write-Host "      Waiting for backend to start..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "      Backend started successfully!" -ForegroundColor Green
} catch {
    Write-Host "      Backend starting... please wait" -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}

Write-Host "[2/2] Starting Frontend (Next.js)..." -ForegroundColor Yellow

# Start frontend in new window
$frontendPath = Join-Path $ProjectRoot "frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; `$Host.UI.RawUI.WindowTitle = 'Prism Frontend'; npm run dev"

Write-Host "      Waiting for frontend to start..." -ForegroundColor Gray
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "  =========================================" -ForegroundColor Green
Write-Host "       Prism AI Started!                  " -ForegroundColor Green
Write-Host "  =========================================" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:3000         " -ForegroundColor White
Write-Host "  Backend:  http://localhost:8000         " -ForegroundColor White
Write-Host "  =========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Enter to open browser..." -ForegroundColor Gray
Read-Host

Start-Process "http://localhost:3000"
