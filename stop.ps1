# Prism AI - PowerShell Stop Script
# Stops all Prism AI services

Write-Host ""
Write-Host "  Stopping Prism AI services..." -ForegroundColor Yellow
Write-Host ""

# Kill processes on port 8000 (backend)
$backendProcesses = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($backendProcesses) {
    foreach ($procId in $backendProcesses) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Write-Host "  [OK] Stopped backend process (PID: $procId)" -ForegroundColor Green
        } catch {
            Write-Host "  [WARN] Could not stop process $procId" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  [INFO] No backend process found on port 8000" -ForegroundColor Gray
}

# Kill processes on port 3000 (frontend)
$frontendProcesses = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($frontendProcesses) {
    foreach ($procId in $frontendProcesses) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Write-Host "  [OK] Stopped frontend process (PID: $procId)" -ForegroundColor Green
        } catch {
            Write-Host "  [WARN] Could not stop process $procId" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  [INFO] No frontend process found on port 3000" -ForegroundColor Gray
}

Write-Host ""
Write-Host "  All services stopped." -ForegroundColor Green
Write-Host ""
