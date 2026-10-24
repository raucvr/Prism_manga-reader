# Prism - start backend (FastAPI) and frontend (Next.js) on Windows. The twin of start.sh.
# Usage: double-click start.bat, or: powershell -ExecutionPolicy Bypass -File start.ps1 [-NoBrowser]
#        Other ports: $env:BACKEND_PORT = "8001"; $env:FRONTEND_PORT = "3001"; .\start.ps1
# Logs in .\logs, PIDs in .\.pids, stop with stop.bat. Both servers listen on 127.0.0.1 only:
# anything that can reach them can start paid jobs.
param([switch]$NoBrowser)

$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Frontend = Join-Path $Root "frontend"
$HasUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)
$BackendPort = 8000
$FrontendPort = 3000
if ($env:BACKEND_PORT) { $BackendPort = [int]$env:BACKEND_PORT }
if ($env:FRONTEND_PORT) { $FrontendPort = [int]$env:FRONTEND_PORT }

function Fail([string]$Message) {
    Write-Host "[Prism] $Message" -ForegroundColor Red
    exit 1
}

function Test-Python([string]$Exe, [string[]]$Arguments) {
    # True when the interpreter runs and is 3.10+ (the Microsoft Store "python" alias fails this test)
    try {
        $out = & $Exe @Arguments -c "import sys; print(sys.version_info >= (3, 10))" 2>$null
        return ($LASTEXITCODE -eq 0) -and ("$out".Trim() -eq "True")
    } catch {
        return $false
    }
}

function Read-Stamp([string]$Path) {
    if (Test-Path -LiteralPath $Path) { return "$(Get-Content -LiteralPath $Path -Raw)".Trim() }
    return ""
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
}

function Test-Health {
    try {
        $request = [System.Net.WebRequest]::Create("http://127.0.0.1:$BackendPort/health")
        $request.Proxy = $null  # a configured proxy must not see loopback requests
        $request.Timeout = 2000
        $request.GetResponse().Close()
        return $true
    } catch {
        return $false
    }
}

function Start-Logged([string]$Name, [string]$Command, [string]$WorkingDirectory) {
    # cmd.exe sends stdout and stderr to one log; the hidden window keeps running after this console closes
    $log = Join-Path $Root "logs\$Name.log"
    $arguments = "/d /s /c "" $Command > ""$log"" 2>&1 """
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList $arguments -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru
    $started = 0
    try { $started = $process.StartTime.ToFileTimeUtc() } catch { $started = 0 }
    # pid and start time: stop.ps1 never stops a process that merely reuses the pid
    Set-Content -LiteralPath (Join-Path $Root ".pids\$Name.pid") -Value @($process.Id, $started) -Encoding ASCII
    return $process
}

function Stop-Startup([string]$Name, [string]$Message) {
    # show the log tail, stop whatever this script started, exit
    Write-Host "[Prism] $Message Last lines of logs\$Name.log:" -ForegroundColor Red
    Get-Content -LiteralPath (Join-Path $Root "logs\$Name.log") -Tail 40 -Encoding UTF8 -ErrorAction SilentlyContinue | Out-Host
    & (Join-Path $Root "stop.ps1") | Out-Null
    exit 1
}

# First, before anything is installed or rewritten underneath a running copy
foreach ($port in @($BackendPort, $FrontendPort)) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $owner = (Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue).ProcessName
        Fail "Port $port is already in use by $owner (pid $($listener.OwningProcess)). Prism already running? stop.bat stops it. Or pick other ports with BACKEND_PORT and FRONTEND_PORT."
    }
}

# ---- Python virtualenv (Python 3.10+), reinstalled whenever requirements.txt changes
if ((Test-Path -LiteralPath $Venv) -and -not (Test-Path -LiteralPath $Py)) {
    Fail ".venv exists but has no Scripts\python.exe (unfinished, or made on another OS). Delete .venv and run start.bat again."
}
if ((Test-Path -LiteralPath $Py) -and -not (Test-Python $Py @())) {
    Fail ".venv does not run Python 3.10+. Delete .venv and run start.bat again."
}
if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "[Prism] Creating the virtualenv in .venv"
    if ($HasUv) {
        & uv venv $Venv --python 3.12
        if ($LASTEXITCODE -ne 0) { & uv venv $Venv --python ">=3.10" }
        if ($LASTEXITCODE -ne 0) { Fail "uv could not create .venv. Behind a firewall? See the README for UV_PYTHON_INSTALL_MIRROR." }
    } else {
        $base = $null
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($version in "3.12", "3.13", "3.11", "3.10") {
                if (Test-Python "py" @("-$version")) { $base = @("py", "-$version"); break }
            }
        }
        if (-not $base -and (Test-Python "python" @())) { $base = @("python") }
        if (-not $base) {
            Fail "Python 3.10+ not found. Install uv (https://docs.astral.sh/uv/) or Python 3.10+ from python.org, then run start.bat again."
        }
        $exe = $base[0]
        $rest = @($base | Select-Object -Skip 1)
        & $exe @rest -m venv $Venv
        if ($LASTEXITCODE -ne 0) { Fail "Could not create .venv with $($base -join ' ')." }
    }
}
$Requirements = Join-Path $Root "requirements.txt"
$ReqStamp = Join-Path $Venv ".prism-requirements.sha256"
if ((Read-Stamp $ReqStamp) -ne (Get-Sha256 $Requirements)) {
    Write-Host "[Prism] Installing Python dependencies (requirements.txt)"
    if ($HasUv) {
        & uv pip install --python $Py -r $Requirements
    } else {
        & $Py -m pip install -r $Requirements
    }
    if ($LASTEXITCODE -ne 0) { Fail "Installing the Python dependencies failed (see above)." }
    Set-Content -LiteralPath $ReqStamp -Value (Get-Sha256 $Requirements) -Encoding ASCII
}

$Config = Join-Path $Root "config\api_config.yaml"
if (-not (Test-Path -LiteralPath $Config)) {
    Write-Host "[Prism] config\api_config.yaml not found - copying the example. Set ARK_API_KEY (BytePlus ModelArk) before generating." -ForegroundColor Yellow
    Copy-Item -LiteralPath (Join-Path $Root "config\api_config.yaml.example") -Destination $Config
}

# ---- Node.js 18.17+ (Next.js 14) and the frontend dependencies, reinstalled whenever package-lock.json changes
$Npm = "npm.cmd"  # not npm.ps1, which the execution policy may block
if (-not (Get-Command $Npm -ErrorAction SilentlyContinue)) { $Npm = "npm" }
if (-not (Get-Command node -ErrorAction SilentlyContinue) -or -not (Get-Command $Npm -ErrorAction SilentlyContinue)) {
    Fail "Node.js 18.17+ with npm is required: https://nodejs.org"
}
$nodeVersion = "$(& node -v)".Trim()
$parsed = $null
if ([version]::TryParse($nodeVersion.TrimStart("v"), [ref]$parsed) -and $parsed -lt [version]"18.17") {
    Fail "Node.js 18.17+ is required (found $nodeVersion)."
}
$Lock = Join-Path $Frontend "package-lock.json"
$LockStamp = Join-Path $Frontend "node_modules\.prism-lock.sha256"
if (-not (Test-Path -LiteralPath (Join-Path $Frontend "node_modules")) -or (Read-Stamp $LockStamp) -ne (Get-Sha256 $Lock)) {
    Write-Host "[Prism] Installing frontend dependencies"
    Push-Location -LiteralPath $Frontend
    & $Npm install --no-audit --no-fund
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Fail "npm install failed (see above)." }
    Set-Content -LiteralPath $LockStamp -Value (Get-Sha256 $Lock) -Encoding ASCII
}

# ASCII: Next.js would read a UTF-8 byte order mark as part of the variable name
Set-Content -LiteralPath (Join-Path $Frontend ".env.local") -Value "NEXT_PUBLIC_API_URL=http://127.0.0.1:$BackendPort" -Encoding ASCII

# ---- Start both servers (inherited by both: the ports, UTF-8 output, no telemetry)
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs"), (Join-Path $Root ".pids") | Out-Null
$env:BACKEND_PORT = "$BackendPort"    # the backend accepts changes only from pages on these two ports
$env:FRONTEND_PORT = "$FrontendPort"
$env:PYTHONUTF8 = "1"                 # log lines contain Chinese: redirected output must be UTF-8, not the ANSI code page
$env:PYTHONUNBUFFERED = "1"
$env:NEXT_TELEMETRY_DISABLED = "1"

Write-Host "[Prism] Starting backend on 127.0.0.1:$BackendPort"
$backend = Start-Logged "backend" """$Py"" -m uvicorn main:app --host 127.0.0.1 --port $BackendPort" (Join-Path $Root "backend")
$healthy = $false
for ($i = 0; $i -lt 90; $i++) {
    if (Test-Health) { $healthy = $true; break }
    if ($backend.HasExited) { break }
    Start-Sleep -Seconds 1
}
if (-not $healthy) {
    if ($backend.HasExited) { Stop-Startup "backend" "The backend stopped during startup." }
    Stop-Startup "backend" "The backend did not answer /health within 90 s."
}

Write-Host "[Prism] Starting frontend on 127.0.0.1:$FrontendPort"
$frontendProcess = Start-Logged "frontend" "$Npm run dev -- -p $FrontendPort" $Frontend
for ($i = 0; $i -lt 60; $i++) {
    if (Get-NetTCPConnection -State Listen -LocalPort $FrontendPort -ErrorAction SilentlyContinue) { break }
    if ($frontendProcess.HasExited) { Stop-Startup "frontend" "The frontend stopped during startup." }
    if ($i -eq 59) { Write-Host "[Prism] The frontend is still starting; see logs\frontend.log" -ForegroundColor Yellow }
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "  Prism is running" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:$FrontendPort"
Write-Host "  Backend:  http://127.0.0.1:$BackendPort  (API docs at /docs)"
Write-Host "  Logs:     logs\backend.log, logs\frontend.log   Stop with stop.bat"
Write-Host ""
if (-not $NoBrowser) { Start-Process "http://localhost:$FrontendPort" }
