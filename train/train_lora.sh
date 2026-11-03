#!/usr/bin/env bash
# Train the SDXL LoRA with kohya sd-scripts. Override the config with CONFIG=...
set -euo pipefail
WORK=${WORK:-/workspace}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-$REPO_DIR/configs/train_lora.toml}

cd "$WORK/sd-scripts"
accelerate launch --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file "$CONFIG" "$@"
echo "done -> $WORK/output/"
ls -la "$WORK/output/"*.safetensors
