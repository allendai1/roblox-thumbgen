---
title: Architecture decisions
date: 2026-11-03
tags: [decision, architecture]
---

# Architecture decisions

Decisions taken in the first planning session, with the user's answers that drove them.

## Inputs from the user

| Question | Answer |
|---|---|
| Compute | Cloud GPU rental (RunPod) |
| Dataset | 1000+ thumbnails, some tagged; CSV/JSON metadata with `filename`, `tags`, game name/genre |
| Source resolution | 768×432 or smaller (Roblox CDN size) |
| Output goal | Scene/background only; text, arrows, badges added later in an editor |
| Base model | SDXL family (user preference) |
| Visual style | Mixed 3D-render + anime, switchable by prompt |
| Tooling | Full app: pick preset/style, get N candidates |
| Inference | RunPod pod running app + model, UI tunnelled through exposed port |
| Code home | GitHub `allendai1/roblox-thumbgen` (private) |

## Decisions

1. **Base: SDXL 1.0** (`stabilityai/stable-diffusion-xl-base-1.0`), configurable. Reason: a neutral base handles both 3D and anime; Illustrious/Pony are anime-biased and would fight the 3D half of the data. Juggernaut XL is the first alternative to try if 3D renders look plasticky.
2. **One LoRA, two trigger phrases** (`rbxthumb, 3d render` / `rbxthumb, anime style`) instead of two LoRAs. With 1000+ images there is enough data for both modes, and one adapter keeps the app simple. Split into two if style bleed is visible after v1.
3. **Style labels via CLIP zero-shot** at ingest (ViT-L/14, laion2b). The user's `style` column overrides when present.
4. **Upscale 2× before training** (Real-ESRGAN x4plus → downsample to 2×, Lanczos fallback). 768×432 is well under SDXL's native megapixel; training at the 1344×768 bucket after upscaling beats training at 768 base.
5. **Captions = trigger + style + genre + user tags + WD14 tags + Florence-2 description**, capped at 225 tokens (kohya `max_token_length`). Game titles excluded by default: they do not generalise.
6. **Trainer: kohya sd-scripts** (`sdxl_train_network.py`). Most battle-tested SDXL LoRA path, TOML-configurable, runs on RunPod with no GUI.
7. **CTR strategies live in prompt presets, not in the LoRA.** The LoRA learns the *look* (lighting, saturation, render style); composition hooks (face close-up, VS, loot reveal, …) are explicit prompt templates so they can be mixed and edited. See [CTR strategies](../research/ctr-strategies.md).
8. **No text generation.** SDXL text is unreliable; presets reserve negative space on a chosen side and the user adds title text afterwards.

Related: [CTR strategies](../research/ctr-strategies.md), [Model choice](../research/model-choice.md), [Dataset pipeline](../pipeline/dataset.md), [Training](../pipeline/training.md), [App](../pipeline/app.md)
