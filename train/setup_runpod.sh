#!/usr/bin/env bash
# One-time pod setup: kohya sd-scripts + SDXL base + VAE fix. Run from the repo root on the pod.
set -euo pipefail
WORK=${WORK:-/workspace}
mkdir -p "$WORK"/{models,dataset,output,logs}

if [ ! -d "$WORK/sd-scripts" ]; then
  git clone --depth 1 -b sd3 https://github.com/kohya-ss/sd-scripts.git "$WORK/sd-scripts"
fi
cd "$WORK/sd-scripts"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q bitsandbytes tensorboard "huggingface_hub>=0.24"

cd "$WORK/models"
[ -f sd_xl_base_1.0.safetensors ] || hf download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir . 2>/dev/null || \
  huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir .
[ -f sdxl_vae_fp16_fix.safetensors ] || { huggingface-cli download madebyollin/sdxl-vae-fp16-fix diffusion_pytorch_model.safetensors --local-dir ./vae_tmp && mv vae_tmp/diffusion_pytorch_model.safetensors sdxl_vae_fp16_fix.safetensors && rm -rf vae_tmp; }

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cp "$REPO_DIR/train/sample_prompts.txt" "$WORK/sample_prompts.txt"

# accelerate default config (single GPU, bf16)
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
num_machines: 1
num_processes: 1
use_cpu: false
YAML

echo "setup done. Copy work/dataset/10_rbxthumb -> $WORK/dataset/ then run train/train_lora.sh"
