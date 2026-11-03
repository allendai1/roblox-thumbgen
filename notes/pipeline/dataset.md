---
title: Dataset pipeline
date: 2026-11-03
tags: [pipeline, dataset, ingest]
---

# Dataset pipeline

`thumbgen ingest --config configs/dataset.yaml` turns `metadata.csv` + `images/` into a kohya folder.

## Steps

1. **Load metadata** (`.csv` / `.json` / `.jsonl`); column names come from `configs/dataset.yaml → columns`.
2. **Filter**: aspect ratio 1.5–2.0 (drops square game icons), width ≥ 384, perceptual-hash dedupe.
3. **Upscale** 2× (Real-ESRGAN x4plus via spandrel, Lanczos fallback) then **fit-and-crop** to 1344×768.
4. **Style label**: `style` column if present, else CLIP zero-shot (`3d` / `anime`) with confidence recorded.
5. **Caption**: `rbxthumb, <style phrase>, <genre>, <user tags>, <Florence sentence>, <WD14 tags>`.
6. **Write** `work/dataset/10_rbxthumb/<id>.png` + `<id>.txt` and `work/dataset/manifest.jsonl` (one row per image: source file, style, confidence, caption, dropped reason if any).

## Expected metadata shape

```
filename,tags,game_name,genre,style
abc123.png,"zombie, night, flashlight",Dead Rails,horror,
...
```

Blank `tags` and `style` are fine. Absolute paths in `filename` are honoured.

## Known gaps

- No per-image CTR/visits data, so no weighting toward top performers yet. If stats arrive, add a `weight` column and emit per-image repeats (kohya supports a `.toml` dataset config with multiple subsets).

Related: [Architecture decisions](../decisions/2026-11-03-architecture.md), [Training](training.md)
