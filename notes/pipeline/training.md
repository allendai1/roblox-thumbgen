---
title: Training on RunPod
date: 2026-11-03
tags: [pipeline, training, kohya, runpod]
---

# Training on RunPod

Target: one SDXL LoRA, `work/output/rbxthumb-v1.safetensors`.

## Pod

- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or newer), 24 GB+ GPU (4090 / A5000 / A100). Mount a network volume at `/workspace` so models and datasets survive pod restarts.
- `bash train/setup_runpod.sh` clones kohya sd-scripts (sd3 branch), installs deps, downloads the SDXL base and VAE-fp16-fix.
- Copy `work/dataset/` to `/workspace/dataset/` (rsync or `runpodctl send`).

## Run

`bash train/train_lora.sh` → `accelerate launch sdxl_train_network.py --config_file configs/train_lora.toml`.
Hyper-parameters are in [Model choice](../research/model-choice.md). Checkpoints every 500 steps; sample prompts render at each save for quick visual comparison.

## Evaluation checklist

- [ ] Both triggers produce their style without bleed.
- [ ] Faces read at thumbnail size (downscale samples to 150 px wide and look).
- [ ] No caption-tag leakage (e.g. game-title text artefacts).
- [ ] Preset prompts from `configs/presets.yaml` respond to `text_side` (clean space where asked).

Related: [Dataset pipeline](dataset.md), [Model choice](../research/model-choice.md), [App](app.md)
