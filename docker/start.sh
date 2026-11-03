#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${OUTPUT_DIR:-/workspace/outputs}" "${HF_HOME:-/workspace/hf-cache}"
if [ -n "${HF_TOKEN:-}" ]; then huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true; fi
cd /app
exec uvicorn thumbgen.app.server:get_app --factory --host 0.0.0.0 --port "${PORT:-8000}"
