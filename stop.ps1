# Prism - stop the processes start.ps1 started (PIDs in .pids\). Anything else is left alone.
$Root = $PSScriptRoot
foreach ($name in "frontend", "backend") {
    $file = Join-Path $Root ".pids\$name.pid"
    if (-not (Test-Path -LiteralPath $file)) { continue }
    $lines = @(Get-Content -LiteralPath $file)
    $procId = 0
    $started = [long]0
    if ($lines.Count -lt 2 -or -not [int]::TryParse($lines[0], [ref]$procId) -or -not [long]::TryParse($lines[1], [ref]$started)) {
        Write-Host "[Prism] $file was not written by start.ps1; leaving it alone"
        continue
    }
    Remove-Item -LiteralPath $file -Force
    $process = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    $ours = $false
    try { $ours = ($process.StartTime.ToFileTimeUtc() -eq $started) } catch { $ours = $false }
    if ($ours) {
        Write-Host "[Prism] Stopping $name (pid $procId)"
        # the whole tree: cmd.exe, then python (the venv launcher and the interpreter) or npm, node and the Next.js server
        & taskkill.exe /PID $procId /T /F | Out-Null
    } else {
        Write-Host "[Prism] pid $procId is no longer Prism's $name; leaving it alone"
    }
}
Write-Host "[Prism] Stopped"
