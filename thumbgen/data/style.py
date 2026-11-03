"""CLIP zero-shot style classifier: '3d' vs 'anime'."""
from __future__ import annotations

from typing import Dict, List

from PIL import Image


class StyleClassifier:
    def __init__(self, model: str = "ViT-L-14", pretrained: str = "laion2b_s32b_b82k",
                 prompts: Dict[str, List[str]] | None = None, device: str = "cpu"):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model)
        self.labels = list(prompts.keys())
        feats = []
        with torch.no_grad():
            for label in self.labels:
                toks = tokenizer(prompts[label]).to(device)
                f = self.model.encode_text(toks)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.mean(0))
        self.text_feats = torch.stack(feats)
        self.text_feats = self.text_feats / self.text_feats.norm(dim=-1, keepdim=True)

    def __call__(self, img: Image.Image) -> tuple[str, float]:
        torch = self.torch
        with torch.no_grad():
            x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ self.text_feats.T).softmax(dim=-1)[0]
        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])
