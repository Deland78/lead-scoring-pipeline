#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-1}"

exec uvicorn main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level info