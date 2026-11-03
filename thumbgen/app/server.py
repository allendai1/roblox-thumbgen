"""FastAPI app: presets + prompt expansion + generation."""
from __future__ import annotations

import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .pipeline import Generator
from .prompts import PromptBuilder

STATIC = Path(__file__).parent / "static"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "work/outputs"))


class PromptRequest(BaseModel):
    preset: str
    style: str = "3d"
    palette: Optional[str] = "orange_teal"
    slots: Dict[str, str] = Field(default_factory=dict)
    text_side: Optional[str] = None
    extra: str = ""
    extra_negative: str = ""
    seed: Optional[int] = None


class GenerateRequest(PromptRequest):
    count: int = Field(4, ge=1, le=12)
    width: int = 1344
    height: int = 768
    steps: int = Field(28, ge=8, le=60)
    cfg: float = Field(6.0, ge=1.0, le=15.0)
    lora_scale: Optional[float] = Field(None, ge=0.0, le=1.5)
    prompt_override: Optional[str] = None
    negative_override: Optional[str] = None
    upscale_1080p: bool = False


def create_app(presets_path: str = "configs/presets.yaml", generator: Optional[Generator] = None) -> FastAPI:
    pb = PromptBuilder.from_yaml(presets_path)
    gen = generator or Generator()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="roblox-thumbgen")
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
    app.state.builder, app.state.generator = pb, gen

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/presets")
    def presets() -> Dict[str, Any]:
        return pb.catalog()

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "lora": gen.lora_path or None, "base": gen.base, "loaded": gen._pipe is not None}

    @app.post("/api/prompt")
    def prompt(req: PromptRequest) -> Dict[str, Any]:
        try:
            pos, neg, meta = pb.build(req.preset, req.style, req.palette, req.slots, req.text_side,
                                      req.extra, req.extra_negative, req.seed)
        except KeyError as e:
            raise HTTPException(400, str(e))
        return {"positive": pos, "negative": neg, "meta": meta}

    def _save(images: List[Image.Image], seeds: List[int], pos: str, neg: str, meta: Dict[str, Any],
              params: Dict[str, Any]) -> List[Dict[str, Any]]:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        results = []
        for img, s in zip(images, seeds):
            name = f"{stamp}_{meta.get('preset', 'img')}_{s}.png"
            img.save(OUTPUT_DIR / name)
            results.append({"url": f"/outputs/{name}", "seed": s, "width": img.width, "height": img.height})
        (OUTPUT_DIR / f"{stamp}_{meta.get('preset', 'img')}.json").write_text(json.dumps(
            {"positive": pos, "negative": neg, "meta": meta, "params": params, "results": results}, indent=2))
        return results

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> Dict[str, Any]:
        try:
            pos, neg, meta = pb.build(req.preset, req.style, req.palette, req.slots, req.text_side,
                                      req.extra, req.extra_negative, req.seed)
        except KeyError as e:
            raise HTTPException(400, str(e))
        pos = req.prompt_override or pos
        neg = req.negative_override or neg
        base_seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        seeds = [base_seed + i for i in range(req.count)]
        params = req.model_dump()
        try:
            images = gen.generate(pos, neg, seeds, req.width, req.height, req.steps, req.cfg, req.lora_scale)
            if req.upscale_1080p:
                images = [gen.upscale_1080p(im, pos, neg, s) for im, s in zip(images, seeds)]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"generation failed: {e}")
        return {"positive": pos, "negative": neg, "meta": meta, "results": _save(images, seeds, pos, neg, meta, params)}

    @app.post("/api/variations")
    async def variations(image: UploadFile = File(...), prompt: str = Form(...), negative: str = Form(""),
                         count: int = Form(4), strength: float = Form(0.55), seed: Optional[int] = Form(None),
                         steps: int = Form(28), cfg: float = Form(6.0)) -> Dict[str, Any]:
        data = await image.read()
        init = Image.open(io.BytesIO(data)).convert("RGB")
        base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        seeds = [base_seed + i for i in range(max(1, min(count, 12)))]
        try:
            images = gen.variations(init, prompt, negative, seeds, strength, steps, cfg, 1344, 768)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"variation failed: {e}")
        meta = {"preset": "variation"}
        return {"positive": prompt, "negative": negative, "meta": meta,
                "results": _save(images, seeds, prompt, negative, meta, {"strength": strength})}

    @app.get("/api/history")
    def history(limit: int = 20) -> List[Dict[str, Any]]:
        files = sorted(OUTPUT_DIR.glob("*.json"), reverse=True)[:limit]
        return [json.loads(f.read_text()) for f in files]

    return app


app = None  # created lazily by `thumbgen serve` or `uvicorn thumbgen.app.server:get_app --factory`


def get_app() -> FastAPI:
    return create_app(os.environ.get("PRESETS_PATH", "configs/presets.yaml"))
