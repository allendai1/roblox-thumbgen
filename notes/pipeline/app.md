---
title: Generation app
date: 2026-11-03
tags: [pipeline, app, fastapi, diffusers]
---

# Generation app

`thumbgen serve` — FastAPI + diffusers SDXL pipeline with the LoRA loaded, static single-page UI.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/presets` | Preset list, palettes, expressions, style options |
| POST | `/api/prompt` | Expand a preset + slots into positive/negative prompt (no GPU) |
| POST | `/api/generate` | Generate N candidates; returns image URLs + seeds + the exact prompt used |
| POST | `/api/variations` | img2img from an uploaded image or a previous result |
| GET | `/outputs/<file>` | Serve generated PNGs |

## Generation defaults

1344×768 (16:9 SDXL bucket), 28 steps, DPM++ 2M Karras, CFG 6, LoRA scale 0.8, batch of 4 seeds.
Optional `upscale_1080p` does a 0.3-strength img2img pass at 1920×1080.

> [!NOTE] Prompt weights
> diffusers does not parse A1111 `(text:1.2)` weights; the pipeline uses **Compel** (SDXL dual-encoder, penultimate hidden states, pooled from encoder 2) and converts `(text:1.2)` → `(text)1.2`. Without Compel the weights are stripped and prompts truncate at 77 tokens.

## UI

Left column: preset cards, style toggle (3D / anime), palette, slot fields filled from preset defaults,
free-text extras, seed, count. Right: gallery grid with seed + prompt copy buttons and a download link.

## Running on the pod

`docker/Dockerfile` builds an image that starts the server on port 8000; expose 8000 as an HTTP port in the
RunPod template and open the proxied URL. `LORA_PATH`, `BASE_MODEL`, `HF_TOKEN` are env vars.

Related: [CTR strategies](../research/ctr-strategies.md), [Training](training.md)
