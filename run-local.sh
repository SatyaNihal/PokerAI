#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  python_bin=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13)' 2>/dev/null; then
      python_bin="$candidate"
      break
    fi
  done
  if [[ -z "$python_bin" ]]; then
    echo "Install Python 3.11 or 3.12, then run this script again." >&2
    exit 1
  fi
  "$python_bin" -m venv .venv
fi

if ! .venv/bin/python -c 'import fastapi, numpy, torch, uvicorn, yaml, tensorboard, psutil' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install --no-deps -e .
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Install Node.js and npm, then run this script again." >&2
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  npm --prefix frontend ci --no-audit --no-fund
fi
npm --prefix frontend run build

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Open http://127.0.0.1:${PORT:-8765} — press Ctrl+C to stop."
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8765}" --workers 1
