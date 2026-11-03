# roblox-thumbgen

Personal Roblox game-thumbnail generator: an SDXL LoRA trained on high-performing
thumbnails plus a small web app that encodes well-known high-CTR composition
strategies as prompt presets.

```
dataset (CSV/JSON + images)
   │  thumbgen ingest        upscale → bucket → caption → kohya folder
   ▼
train/train_lora.sh         kohya sd-scripts, SDXL LoRA, RunPod
   │
   ▼
thumbgen serve              FastAPI + diffusers + LoRA, browser UI with CTR presets
```

## Layout

| Path | Purpose |
|---|---|
| `configs/dataset.yaml` | Column mapping for your metadata file, paths, upscale/caption options |
| `configs/presets.yaml` | CTR composition presets (prompt fragments + layout hints) |
| `configs/train_lora.toml` | kohya `sdxl_train_network.py` config |
| `thumbgen/data/` | `ingest.py`, `caption.py`, `upscale.py`, `style.py` |
| `thumbgen/app/` | FastAPI server, diffusers pipeline wrapper, prompt builder, static UI |
| `train/` | `setup_runpod.sh`, `train_lora.sh` |
| `docker/` | Image that runs the app on a RunPod pod |

## Quick start

### 1. Prepare the dataset (any machine with a GPU, or CPU for Lanczos-only)

```bash
pip install -e ".[data]"
# edit configs/dataset.yaml: metadata path, image dir, column names
thumbgen ingest --config configs/dataset.yaml
```

Output: `work/dataset/10_rbxthumb/*.png|*.txt` ready for kohya, plus
`work/dataset/manifest.jsonl` with the per-image style label and caption.

### 2. Train the LoRA (RunPod, 24 GB+)

```bash
bash train/setup_runpod.sh          # installs kohya sd-scripts, downloads SDXL base
bash train/train_lora.sh            # ~1.5-2 h on an A100/4090 for ~1200 images
```

Outputs `work/output/rbxthumb-v1.safetensors`.

### 3. Serve

```bash
pip install -e ".[app]"
LORA_PATH=work/output/rbxthumb-v1.safetensors thumbgen serve --port 8000
```

or `docker build -f docker/Dockerfile -t roblox-thumbgen .` and run it as a RunPod
pod template with port 8000 exposed.

## Style switching

One LoRA, two trigger phrases baked into captions at ingest time:

- `rbxthumb, 3d render` — blocky/Rthro avatars, glossy Blender-style renders
- `rbxthumb, anime style` — 2D illustrated anime-game thumbnails

The style label is assigned automatically with CLIP zero-shot (overridable per row
via a `style` column).

## CTR presets

See `configs/presets.yaml`. Each preset is a named composition strategy
(face close-up with exaggerated expression, VS face-off, loot reveal, looming threat,
chase, before/after, update hero shot, etc.) that expands to prompt fragments,
a negative prompt, and a "text side" hint so the generated scene leaves negative space
for your title text. Text, arrows and badges are added afterward in your editor —
diffusion models are unreliable at rendering legible text.
