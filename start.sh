#!/usr/bin/env bash
set -e

# Render define $PORT. Se não existir, usa 10000 (útil localmente).
PORT="${PORT:-10000}"

exec uvicorn app:app --host 0.0.0.0 --port "$PORT"

