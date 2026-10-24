#!/usr/bin/env bash
# Prism - start backend (FastAPI) and frontend (Next.js) on macOS / Linux. On Windows use start.bat (or WSL2).
# Usage: ./start.sh                                    (logs in ./logs, PIDs in ./.pids, stop with ./stop.sh)
#        BACKEND_PORT=8001 FRONTEND_PORT=3001 ./start.sh
# Both servers listen on 127.0.0.1 only: anything that can reach them can start paid jobs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs .pids

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
export BACKEND_PORT FRONTEND_PORT   # the backend accepts changes only from pages on these two ports
PY="$ROOT/.venv/bin/python"

die() { echo "[Prism] $*" >&2; exit 1; }
py_ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; }
sha256() { "$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
healthy() {  # no proxy: a configured http_proxy must not see loopback requests
  "$PY" -c 'import sys, urllib.request as u; u.build_opener(u.ProxyHandler({})).open(sys.argv[1], timeout=2)' \
    "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1
}
fail_startup() {  # name, message: show the log tail, stop whatever started, exit
  echo "[Prism] $2 Last lines of logs/$1.log:" >&2
  tail -n 40 "logs/$1.log" >&2 || true
  "$ROOT/stop.sh" >/dev/null 2>&1 || true
  exit 1
}

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) die "On Windows run start.bat (or start.sh inside WSL2)." ;;
esac

# First, before anything is installed or rewritten underneath a running copy
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if port_busy "$port"; then
    die "Port $port is already in use. Prism already running? ./stop.sh stops it. Or pick others: BACKEND_PORT=8001 FRONTEND_PORT=3001 ./start.sh"
  fi
done
# Both ports are free, so PID files left from an earlier run cannot name a running Prism (stop.sh must not act on them)
rm -f "$ROOT/.pids/backend.pid" "$ROOT/.pids/frontend.pid"

# ---- Python virtualenv (Python 3.10+), reinstalled whenever requirements.txt changes
if [ -d .venv ] && [ ! -x "$PY" ]; then
  die ".venv exists but has no bin/python (unfinished, or made on another OS). Delete .venv and run ./start.sh again."
fi
if [ -x "$PY" ] && ! py_ok "$PY"; then
  die ".venv uses $("$PY" -V 2>&1); Prism needs Python 3.10+. Delete .venv and run ./start.sh again."
fi
if [ ! -x "$PY" ]; then
  echo "[Prism] Creating the virtualenv in .venv"
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.12 || uv venv .venv --python ">=3.10" \
      || die "uv could not create .venv. Behind a firewall? See the README for UV_PYTHON_INSTALL_MIRROR."
  else
    BASE=""
    for cand in python3.12 python3.13 python3.11 python3.10 python3; do
      if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then BASE="$cand"; break; fi
    done
    [ -n "$BASE" ] || die "Python 3.10+ not found ($(python3 -V 2>&1 || echo 'no python3')). Install uv (https://docs.astral.sh/uv/) or Python 3.10+, then run ./start.sh again."
    "$BASE" -m venv .venv || die "Could not create .venv with $BASE (Debian / Ubuntu: sudo apt install python3-venv)."
  fi
fi
REQ_STAMP=".venv/.prism-requirements.sha256"
if [ "$(cat "$REQ_STAMP" 2>/dev/null || true)" != "$(sha256 requirements.txt)" ]; then
  echo "[Prism] Installing Python dependencies (requirements.txt)"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r requirements.txt || die "Installing the Python dependencies failed (see above)."
  else
    "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null
    "$PY" -m pip install -r requirements.txt || die "Installing the Python dependencies failed (see above)."
  fi
  sha256 requirements.txt > "$REQ_STAMP"
fi

if [ ! -f config/api_config.yaml ]; then
  echo "[Prism] config/api_config.yaml not found - copying the example. Set ARK_API_KEY (BytePlus ModelArk) before generating."
  cp config/api_config.yaml.example config/api_config.yaml
fi

# ---- Node.js 18.17+ (Next.js 14) and the frontend dependencies, reinstalled whenever package-lock.json changes
command -v npm >/dev/null 2>&1 || die "Node.js 18.17+ with npm is required: https://nodejs.org"
node -e 'const [a, b] = process.versions.node.split(".").map(Number); process.exit(a > 18 || (a === 18 && b >= 17) ? 0 : 1)' \
  || die "Node.js 18.17+ is required (found $(node -v))."
LOCK_STAMP="frontend/node_modules/.prism-lock.sha256"
if [ ! -d frontend/node_modules ] || [ "$(cat "$LOCK_STAMP" 2>/dev/null || true)" != "$(sha256 frontend/package-lock.json)" ]; then
  echo "[Prism] Installing frontend dependencies"
  (cd frontend && npm install --no-audit --no-fund) || die "npm install failed (see above)."
  sha256 frontend/package-lock.json > "$LOCK_STAMP"
fi

echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:${BACKEND_PORT}" > frontend/.env.local

# ---- Start both servers, each in its own process group (set -m) so that stop.sh can stop all of its processes
set -m
echo "[Prism] Starting backend on 127.0.0.1:${BACKEND_PORT}"
cd "$ROOT/backend"
PYTHONUNBUFFERED=1 PYTHONUTF8=1 nohup "$PY" -m uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT" \
  </dev/null >"$ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!
{ echo "$BACKEND_PID"; ps -o lstart= -p "$BACKEND_PID"; } > "$ROOT/.pids/backend.pid" 2>/dev/null  # pid, start time
cd "$ROOT"

for i in $(seq 1 90); do
  healthy && break
  kill -0 "$BACKEND_PID" 2>/dev/null || fail_startup backend "The backend stopped during startup."
  [ "$i" -lt 90 ] || fail_startup backend "The backend did not answer /health within 90 s."
  sleep 1
done

echo "[Prism] Starting frontend on 127.0.0.1:${FRONTEND_PORT}"
cd "$ROOT/frontend"
NEXT_TELEMETRY_DISABLED=1 nohup npm run dev -- -p "$FRONTEND_PORT" </dev/null >"$ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
{ echo "$FRONTEND_PID"; ps -o lstart= -p "$FRONTEND_PID"; } > "$ROOT/.pids/frontend.pid" 2>/dev/null
cd "$ROOT"
set +m

for i in $(seq 1 60); do
  port_busy "$FRONTEND_PORT" && break
  kill -0 "$FRONTEND_PID" 2>/dev/null || fail_startup frontend "The frontend stopped during startup."
  [ "$i" -lt 60 ] || echo "[Prism] The frontend is still starting; see logs/frontend.log"
  sleep 1
done

echo
echo "  Prism is running"
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo "  Backend:  http://127.0.0.1:${BACKEND_PORT}  (API docs at /docs)"
echo "  Logs:     logs/backend.log, logs/frontend.log   Stop with ./stop.sh"
