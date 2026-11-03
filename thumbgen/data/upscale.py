"""Upscaling for small CDN thumbnails (768x432 -> SDXL bucket size).

Real-ESRGAN via spandrel when available; Lanczos fallback otherwise.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PIL import Image


class Upscaler:
    def __init__(self, method: str = "realesrgan", model_url: str | None = None,
                 scale: int = 2, device: str = "cpu", cache_dir: str = "work/models"):
        self.method = method
        self.scale = scale
        self.device = device
        self._model = None
        if method == "realesrgan":
            try:
                self._model = self._load_esrgan(model_url, cache_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[upscale] Real-ESRGAN unavailable ({e}); falling back to Lanczos")
                self.method = "lanczos"

    def _load_esrgan(self, model_url: str | None, cache_dir: str):
        import torch
        from spandrel import ModelLoader

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fname = Path(cache_dir) / Path(model_url).name
        if not fname.exists():
            import urllib.request
            print(f"[upscale] downloading {model_url}")
            urllib.request.urlretrieve(model_url, fname)
        model = ModelLoader().load_from_file(str(fname)).eval().to(self.device)
        if self.device != "cpu":
            model = model.half()
        self._torch = torch
        return model

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.method == "none":
            return img
        if self.method == "lanczos" or self._model is None:
            w, h = img.size
            return img.resize((w * self.scale, h * self.scale), Image.LANCZOS)
        return self._esrgan(img)

    def _esrgan(self, img: Image.Image) -> Image.Image:
        import numpy as np
        torch = self._torch
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.device != "cpu":
            t = t.half()
        with torch.no_grad():
            out = self._model(t)
        out = out.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
        res = Image.fromarray((out * 255).round().astype("uint8"))
        # ESRGAN x4plus is 4x; downscale to requested scale
        native = res.width // img.width
        if native != self.scale:
            res = res.resize((img.width * self.scale, img.height * self.scale), Image.LANCZOS)
        return res


def fit_to_bucket(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize to cover the (width, height) bucket, then center-crop. Never upscales
    beyond what's needed - caller should upscale first if the source is small."""
    sw, sh = img.size
    scale = max(width / sw, height / sh)
    nw, nh = max(width, round(sw * scale)), max(height, round(sh * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - width) // 2
    top = (nh - height) // 2
    return img.crop((left, top, left + width, top + height))
