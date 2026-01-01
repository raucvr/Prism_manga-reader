@echo off
echo Stopping Prism AI...

:: Use PowerShell to reliably kill processes by port
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
powershell -Command "Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

:: Also kill any remaining cmd windows with Prism in title
taskkill /f /fi "WINDOWTITLE eq Prism Backend*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Prism Frontend*" >nul 2>&1

echo Done.
pause
