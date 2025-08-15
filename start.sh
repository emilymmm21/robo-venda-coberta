#!/usr/bin/env bash
set -e

# Render define $PORT; se não, usa 10000
PORT="${PORT:-10000}"

# Se seu app principal é app.py e a instância FastAPI chama-se "app":
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"

