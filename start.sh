#!/usr/bin/env bash
set -euo pipefail

# Uvicorn com o módulo main:app (renomeado)
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-10000}"


