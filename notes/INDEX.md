---
title: roblox-thumbgen notes
date: 2026-11-03
tags: [index]
---

# roblox-thumbgen notes

Second brain for the thumbnail generator. Open this folder as an Obsidian vault.
Every decision, research finding and session log lives here; code lives one level up.

## Folders

- [Decisions](decisions/INDEX.md): architecture and trade-off decisions, dated. Read before changing the model, trainer or caption format.
- [Research](research/INDEX.md): CTR composition strategies, base model comparison, LoRA hyper-parameters.
- [Pipeline](pipeline/INDEX.md): how each stage works — dataset ingest, training on RunPod, the generation app.
- [Log](log/INDEX.md): one note per working session with what changed and what is next.

## Start here

- [Architecture decisions](decisions/2026-11-03-architecture.md): why SDXL, one LoRA with two triggers, presets for CTR hooks.
- [High-CTR Roblox thumbnail strategies](research/ctr-strategies.md): the strategies behind `configs/presets.yaml`.
- [Session 1 — planning and scaffold](log/2026-11-03-session-1.md): latest session log.

## Conventions

- Frontmatter `title`, `date`, `tags` on every note; dated basenames for decisions and logs.
- Relative markdown links only (render on GitHub and in Obsidian). End each note with a `Related:` line.
- New session → new `log/YYYY-MM-DD-session-N.md`, update the index above, record decisions under `decisions/`.
