"""Expand CTR presets into SDXL prompts."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import yaml

STYLE_PHRASES = {"3d": "3d render, glossy blender style render, blocky roblox avatar",
                 "anime": "anime style, 2d illustration, cel shaded"}
TRIGGER = "rbxthumb"
OPPOSITE = {"left": "right", "right": "left", "center": "center"}


def _squash(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(, )+", ", ", s)
    return s.strip(", ")


@dataclass
class PromptBuilder:
    cfg: Dict[str, Any]
    trigger: str = TRIGGER
    style_phrases: Dict[str, str] = field(default_factory=lambda: dict(STYLE_PHRASES))

    @classmethod
    def from_yaml(cls, path: str) -> "PromptBuilder":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    @property
    def presets(self) -> Dict[str, Any]:
        return self.cfg["presets"]

    def catalog(self) -> Dict[str, Any]:
        """What the UI needs to render the form."""
        return {
            "presets": [{"id": k, "label": v["label"], "description": v["description"],
                         "slots": v.get("defaults", {}), "text_side": v.get("text_side", "right")}
                        for k, v in self.presets.items()],
            "palettes": self.cfg.get("palettes", {}),
            "expressions": self.cfg.get("expressions", []),
            "styles": list(self.style_phrases.keys()),
        }

    def build(self, preset_id: str, style: str = "3d", palette: Optional[str] = "orange_teal",
              slots: Optional[Dict[str, str]] = None, text_side: Optional[str] = None,
              extra: str = "", extra_negative: str = "", seed: Optional[int] = None,
              ) -> Tuple[str, str, Dict[str, Any]]:
        if preset_id not in self.presets:
            raise KeyError(f"unknown preset {preset_id!r}; have {sorted(self.presets)}")
        if style not in self.style_phrases:
            raise KeyError(f"unknown style {style!r}")
        p = self.presets[preset_id]
        rng = random.Random(seed)

        palettes = self.cfg.get("palettes", {})
        if palette in (None, "", "random"):
            palette = rng.choice(sorted(palettes)) if palettes else None
        palette_text = palettes.get(palette, palette or "")

        side = text_side or p.get("text_side", "right")
        vars_: Dict[str, str] = dict(p.get("defaults", {}))
        vars_.update({k: v for k, v in (slots or {}).items() if v})
        if vars_.get("expression") in (None, "", "random") and "expression" in p.get("defaults", {}):
            vars_["expression"] = rng.choice(self.cfg.get("expressions") or ["excited"])
        vars_.update(palette=palette_text, text_side=side, subject_side=OPPOSITE.get(side, "left"))

        def fill(t: str) -> str:
            return re.sub(r"\{(\w+)\}", lambda m: vars_.get(m.group(1), m.group(0)), t)

        subject = fill(p["subject"])
        w = float(p.get("weight", 1.0))
        if abs(w - 1.0) > 1e-3:
            subject = f"({subject}:{w:.2f})"

        positive = _squash(", ".join([
            self.trigger, self.style_phrases[style], subject, fill(p["layout"]), fill(p["lighting"]),
            fill(self.cfg.get("global_positive", "")), extra,
        ]))
        negative = _squash(", ".join([
            self.cfg.get("global_negative", ""), p.get("negative", ""), extra_negative,
        ]))
        meta = {"preset": preset_id, "style": style, "palette": palette, "text_side": side, "slots": vars_}
        return positive, negative, meta
