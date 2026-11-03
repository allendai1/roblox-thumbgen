# Generation server for a RunPod pod. Expose port 8000 (HTTP) in the pod template.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
COPY pyproject.toml README.md ./
COPY thumbgen ./thumbgen
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[app]"

ENV BASE_MODEL=stabilityai/stable-diffusion-xl-base-1.0 \
    VAE_MODEL=madebyollin/sdxl-vae-fp16-fix \
    LORA_PATH=/workspace/output/rbxthumb-v1.safetensors \
    LORA_SCALE=0.8 \
    OUTPUT_DIR=/workspace/outputs \
    HF_HOME=/workspace/hf-cache

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 8000
CMD ["/start.sh"]
