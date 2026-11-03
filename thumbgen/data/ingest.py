"""Build a kohya-ready LoRA dataset from CSV/JSON metadata + images."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image
from tqdm import tqdm

from .caption import build_caption, split_tags
from .upscale import Upscaler, fit_to_bucket


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        import csv
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if p.suffix.lower() == ".jsonl":
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # {"filename": {...}} or {"images": [...]}
            if "images" in data and isinstance(data["images"], list):
                return data["images"]
            return [{"filename": k, **(v if isinstance(v, dict) else {"tags": v})} for k, v in data.items()]
        return data
    raise ValueError(f"unsupported metadata format: {p.suffix}")


def dhash(img: Image.Image, size: int = 8) -> str:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.get_flattened_data()) if hasattr(g, "get_flattened_data") else list(g.getdata())
    bits = []
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        bits += [1 if row[c] > row[c + 1] else 0 for c in range(size)]
    return "%016x" % int("".join(map(str, bits)), 2)


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _get(row: Dict[str, Any], col: Optional[str]) -> Optional[str]:
    if not col:
        return None
    v = row.get(col)
    if v is None:
        return None
    v = str(v).strip()
    return v if v and v.lower() != "nan" else None


def filter_row(img: Image.Image, cfg: Dict[str, Any]) -> Optional[str]:
    """Return a drop reason or None if the image passes."""
    f, u = cfg["filters"], cfg["upscale"]
    w, h = img.size
    ar = w / h
    if ar < f["min_aspect"] or ar > f["max_aspect"]:
        return f"aspect {ar:.2f}"
    if w < u["min_source_width"]:
        return f"too small {w}x{h}"
    return None


def run_ingest(config_path: str, limit: Optional[int] = None, caption: bool = True,
               classify_style: bool = True, device: Optional[str] = None) -> int:
    cfg = load_config(config_path)
    paths, cols, cap, up = cfg["paths"], cfg["columns"], cfg["captions"], cfg["upscale"]

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[ingest] device={device}")

    rows = load_metadata(paths["metadata"])
    if limit:
        rows = rows[:limit]
    image_dir = Path(paths["image_dir"])
    out_root = Path(paths["output_dir"])
    out_img = out_root / f"{paths['repeats']}_{paths['instance_name']}"
    out_img.mkdir(parents=True, exist_ok=True)

    upscaler = Upscaler(up["method"], up.get("model_url"), up.get("scale", 2), device=device)
    style_clf = tagger = florence = None
    if classify_style:
        from .style import StyleClassifier
        sc = cfg["style_classifier"]
        style_clf = StyleClassifier(sc["model"], sc["pretrained"], sc["prompts"], device=device)
    if caption:
        from .caption import FlorenceCaptioner, WD14Tagger
        if cap.get("wd14_model"):
            tagger = WD14Tagger(cap["wd14_model"], cap["wd14_threshold"], cap["wd14_max_tags"], device)
        if cap.get("florence_model"):
            florence = FlorenceCaptioner(cap["florence_model"], cap["florence_task"], device)

    manifest = open(out_root / "manifest.jsonl", "w", encoding="utf-8")
    hashes: List[str] = []
    stats = {"kept": 0, "dropped": 0, "styles": {}}

    for row in tqdm(rows, desc="ingest"):
        fn = _get(row, cols["filename"])
        if not fn:
            continue
        src = Path(fn) if os.path.isabs(fn) else image_dir / fn
        rec: Dict[str, Any] = {"source": str(src)}
        try:
            img = Image.open(src)
            img.load()
        except Exception as e:  # noqa: BLE001
            rec["dropped"] = f"unreadable: {e}"
            manifest.write(json.dumps(rec) + "\n")
            stats["dropped"] += 1
            continue

        reason = filter_row(img, cfg)
        if reason is None and cfg["filters"].get("dedupe"):
            h = dhash(img)
            if any(hamming(h, o) <= 4 for o in hashes):
                reason = "duplicate"
            else:
                hashes.append(h)
        if reason:
            rec["dropped"] = reason
            manifest.write(json.dumps(rec) + "\n")
            stats["dropped"] += 1
            continue

        style = (_get(row, cols.get("style")) or "").lower() or None
        conf = 1.0
        if style not in cap["style_phrases"]:
            if style_clf is not None:
                style, conf = style_clf(img)
            else:
                style = "3d"
        rec.update(style=style, style_confidence=round(conf, 3))

        img = img.convert("RGB")
        img = upscaler(img)
        img = fit_to_bucket(img, up["target_width"], up["target_height"])

        user_tags = split_tags(_get(row, cols.get("tags")))
        genre = _get(row, cols.get("genre")) if cap.get("include_genre", True) else None
        game = _get(row, cols.get("game"))
        if cap.get("include_game_name") and game:
            user_tags = [game.lower()] + user_tags
        sentence = florence(img) if florence else ""
        auto_tags = tagger(img) if tagger else []
        text = build_caption(cap["trigger"], cap["style_phrases"][style], genre, user_tags,
                             sentence, auto_tags, cap.get("max_caption_tokens", 225))

        stem = hashlib.sha1(str(src).encode()).hexdigest()[:12]
        img.save(out_img / f"{stem}.png", optimize=True)
        (out_img / f"{stem}.txt").write_text(text, encoding="utf-8")
        rec.update(output=f"{stem}.png", caption=text, game=game, genre=genre)
        manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats["kept"] += 1
        stats["styles"][style] = stats["styles"].get(style, 0) + 1

    manifest.close()
    print(f"[ingest] kept={stats['kept']} dropped={stats['dropped']} styles={stats['styles']}")
    print(f"[ingest] dataset at {out_img}; manifest at {out_root / 'manifest.jsonl'}")
    return 0
