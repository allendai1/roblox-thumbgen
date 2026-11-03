"""Autocaptioning: WD14 tagger (ONNX) + Florence-2 natural-language caption, then merge."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

from PIL import Image

_WD14_SKIP = {"general", "sensitive", "questionable", "explicit"}


class WD14Tagger:
    """SmilingWolf WD14-style ONNX tagger."""

    def __init__(self, repo: str, threshold: float = 0.35, max_tags: int = 18, device: str = "cpu"):
        import csv
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.threshold, self.max_tags = threshold, max_tags
        model_path = hf_hub_download(repo, "model.onnx")
        tags_path = hf_hub_download(repo, "selected_tags.csv")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[1] if isinstance(inp.shape[1], int) else 448
        with open(tags_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"] for r in rows]
        self.categories = [int(r["category"]) for r in rows]  # 0 general, 4 character, 9 rating

    def __call__(self, img: Image.Image) -> List[str]:
        import numpy as np
        im = img.convert("RGB")
        # pad to square on white, resize, BGR float32 (WD14 convention)
        w, h = im.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), (255, 255, 255))
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        canvas = canvas.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1][None]
        probs = self.sess.run(None, {self.input_name: arr})[0][0]
        scored = [(self.names[i], float(p)) for i, p in enumerate(probs)
                  if self.categories[i] == 0 and p >= self.threshold and self.names[i] not in _WD14_SKIP]
        scored.sort(key=lambda x: -x[1])
        return [n.replace("_", " ") for n, _ in scored[: self.max_tags]]


class FlorenceCaptioner:
    def __init__(self, model_id: str = "microsoft/Florence-2-large", task: str = "<MORE_DETAILED_CAPTION>",
                 device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch, self.device, self.task = torch, device, task
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                          trust_remote_code=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.dtype = dtype

    def __call__(self, img: Image.Image) -> str:
        inputs = self.processor(text=self.task, images=img.convert("RGB"), return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                  for k, v in inputs.items()}
        with self.torch.no_grad():
            ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                      max_new_tokens=160, num_beams=3, do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=self.task,
                                                        image_size=(img.width, img.height))
        return clean_sentence(parsed.get(self.task, ""))


def clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    # Florence loves "The image shows/is ..." - drop the preamble
    s = re.sub(r"^the (image|picture|thumbnail) (shows|is|depicts|features)\s*", "", s, flags=re.I)
    return s.rstrip(".")


def split_tags(s: Optional[str]) -> List[str]:
    if not s or (isinstance(s, float)):
        return []
    return [t.strip().lower() for t in re.split(r"[,;|]", str(s)) if t.strip()]


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = it.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out


def approx_tokens(s: str) -> int:
    """Cheap CLIP-ish token estimate: ~1.3 tokens per word + punctuation."""
    words = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", s)
    return int(len(words) * 1.1) + 1


def build_caption(trigger: str, style_phrase: str, genre: Optional[str], user_tags: List[str],
                  sentence: str, auto_tags: List[str], max_tokens: int = 225) -> str:
    """Assemble: trigger, style, genre, user tags, sentence, auto tags - truncating auto tags to fit.

    The first three comma-separated chunks are stable so kohya `keep_tokens=3` protects them
    when shuffling.
    """
    head = [trigger, style_phrase]
    head.append(genre.strip().lower() if genre and str(genre).strip() and str(genre) != "nan" else "game")
    parts = head + dedupe_keep_order(user_tags)
    if sentence:
        parts.append(sentence)
    base = ", ".join(parts)
    budget = max_tokens - approx_tokens(base) - 1
    extra = []
    existing = {p.lower() for p in parts}
    for t in auto_tags:
        if t.lower() in existing:
            continue
        cost = approx_tokens(t) + 1
        if cost > budget:
            break
        extra.append(t)
        budget -= cost
    return ", ".join(parts + extra)
