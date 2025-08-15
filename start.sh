#!/usr/bin/env bash
set -e
PORT=${PORT:-10000}
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
