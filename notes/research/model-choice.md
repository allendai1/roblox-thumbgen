---
title: Base model and LoRA choice
date: 2026-11-03
tags: [research, models, sdxl, lora]
---

# Base model and LoRA choice

## Candidates considered

| Model | Pros | Cons | Verdict |
|---|---|---|---|
| SDXL 1.0 base | Neutral style, huge LoRA ecosystem, 1024² native, trains on 24 GB | Weaker prompt adherence than FLUX | **Chosen for v1** |
| Juggernaut XL / RealVisXL | Better 3D-render polish out of the box | Photoreal bias may fight the anime half | First alternative |
| Illustrious / Pony | Excellent anime, booru-tag prompting | Strongly anime-biased; poor for blocky 3D avatars | Only if we split into two LoRAs |
| FLUX.1-dev | Best quality, text rendering, composition | ~2× VRAM and time to train; slower inference on rented pods | Revisit for v2 |

## LoRA hyper-parameters (starting point)

- `network_dim 32`, `network_alpha 16`, UNet + both text encoders (TE lr lower).
- `unet_lr 1e-4`, `text_encoder_lr 2e-5`, cosine schedule, 5 % warmup, AdamW8bit.
- Batch 4 with gradient checkpointing, bf16, cached latents; `resolution 1024`, buckets 512–2048.
- 1200 imgs × 10 repeats / batch 4 ≈ 3000 steps per epoch; run 1 epoch with the repeat trick, save every 500 steps and compare.
- `min_snr_gamma 5`, `noise_offset 0.0357` (SDXL default-ish), caption dropout 0.05, shuffle tags after the first 3 fixed tokens (`keep_tokens 3` protects `rbxthumb, 3d render, <genre>`).

## Captioning stack

- **WD14 EVA02-large v3** (ONNX) for dense tags — fast, consistent, works on 3D renders despite anime training.
- **Florence-2-large** `<MORE_DETAILED_CAPTION>` for a natural-language sentence that SDXL's CLIP encoders understand.
- Order in the final caption: trigger, style, genre, user tags, Florence sentence, WD14 tags (truncated to fit 225 tokens).

Related: [Architecture decisions](../decisions/2026-11-03-architecture.md), [Training](../pipeline/training.md)
