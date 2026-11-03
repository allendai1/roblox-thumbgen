---
title: High-CTR Roblox thumbnail strategies
date: 2026-11-03
tags: [research, ctr, composition]
---

# High-CTR Roblox thumbnail strategies

Patterns that recur on the Roblox front page and in creator advice (DevForum thumbnail
threads, YouTube thumbnail-design channels adapted to Roblox). These are the source for
`configs/presets.yaml`. Each maps to a preset id.

## Core rules (apply to every preset)

- **One focal point** that reads at 150 px wide — the discovery page shows thumbnails tiny on mobile.
- **Face first.** A big expressive face with eyes toward the camera beats any environment shot. Shock, excitement, fear and smug grins all outperform neutral.
- **Saturation and contrast up**, 2–3 dominant colours, usually complementary (orange/teal, purple/yellow, red/cyan). Muddy or grey thumbnails lose.
- **Rim light / glow outline** separates the subject from the background so the silhouette survives downscaling.
- **Subject 40–60 % of the frame**, placed on one third; the other third is kept clean for title text.
- **Depth:** foreground element, mid subject, background environment; low angle or fisheye for drama.
- **Energy:** sparkles, explosions, speed lines, godrays. Static scenes under-perform.

## Composition hooks → preset

| Hook | Why it works | Preset id |
|---|---|---|
| Face close-up, exaggerated expression | Emotion is the fastest-read signal | `face_closeup` |
| VS face-off, two characters, clear centre | Conflict + curiosity, room for a VS badge | `vs_faceoff` |
| Giant glowing item / pet, small reacting player | Reward anticipation (simulators, pet games) | `loot_reveal` |
| Small player, huge monster behind | Scale contrast + fear (horror, survival) | `looming_threat` |
| Sprint toward camera, hazard behind | Motion = energy (obby, racing, battle) | `chase_action` |
| Split before/after | Progression promise (simulators, tycoons) | `before_after` |
| Hero shot of the new thing + proud character | Update hook; corner left for NEW badge | `update_hero` |
| 3–4 distinct characters posing | Social proof (roleplay, tycoon) | `group_squad` |
| First-person hands | Immersion (shooters, horror) | `pov_hands` |
| Seasonal dressing of a hero composition | Timeliness | `seasonal_event` |

## Post-generation (done in the editor, not the model)

Title text in a heavy rounded font with thick stroke, arrows/circles pointing at the hook,
"NEW"/"UPDATE"/"x2" badges, and a thin outer glow on cut-out characters. The presets' `text_side`
field says where the clean space was requested.

## Open questions

- Does the LoRA pick up the "glossy Blender render" look well enough at rank 32, or does it need rank 64?
- Is CTR correlated with warm vs cool dominant palette for a given genre? Needs stats the dataset does not have yet.

Related: [Architecture decisions](../decisions/2026-11-03-architecture.md), [App](../pipeline/app.md)
