#!/usr/bin/env bash
# Prism - stop the processes start.sh started (PIDs in ./.pids). Anything else is left alone.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stop_one() {  # name, pattern the process's command line must match
  local name="$1" pattern="$2" file="$ROOT/.pids/$1.pid" pid started target
  [ -f "$file" ] || return 0
  pid="$(head -n 1 "$file" | tr -cd '0-9')"
  started="$(sed -n 2p "$file")"   # its start time (start.sh writes it; older files have none)
  rm -f "$file"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then return 0; fi
  if { [ -n "$started" ] && [ "$(ps -o lstart= -p "$pid" 2>/dev/null)" != "$started" ]; } ||
     ! ps -o command= -p "$pid" 2>/dev/null | grep -Eq "$pattern"; then
    echo "[Prism] pid $pid is no longer Prism's $name; leaving it alone"
    return 0
  fi
  echo "[Prism] Stopping $name (pid $pid)"
  if [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pid" ]; then
    target="-$pid"  # its own process group (start.sh): uvicorn, or npm with next dev and its server process
  else
    target="$pid"   # started by an older start.sh: stop its direct children too
    pkill -TERM -P "$pid" 2>/dev/null
  fi
  kill -TERM -- "$target" 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 -- "$target" 2>/dev/null || return 0
    sleep 1
  done
  echo "[Prism] $name did not stop within 10 s; killing it"
  kill -KILL -- "$target" 2>/dev/null
  return 0
}

stop_one frontend 'npm run dev|next dev|next-server'
stop_one backend 'uvicorn main:app'
echo "[Prism] Stopped"
