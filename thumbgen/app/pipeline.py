"""SDXL + LoRA generation wrapper (lazy-loaded so the API can start without a GPU)."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import List, Optional

from PIL import Image

_WEIGHT_RE = re.compile(r"\(([^()]*):([0-9.]+)\)")


def strip_weights(prompt: str) -> str:
    """Drop A1111-style (text:1.2) weights for backends without Compel."""
    return _WEIGHT_RE.sub(r"\1", prompt)

DEFAULT_BASE = os.environ.get("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
DEFAULT_VAE = os.environ.get("VAE_MODEL", "madebyollin/sdxl-vae-fp16-fix")
DEFAULT_LORA = os.environ.get("LORA_PATH", "")
DEFAULT_LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.8"))


class Generator:
    def __init__(self, base: str = DEFAULT_BASE, vae: str = DEFAULT_VAE, lora: str = DEFAULT_LORA,
                 lora_scale: float = DEFAULT_LORA_SCALE, device: Optional[str] = None):
        self.base, self.vae_id, self.lora_path, self.lora_scale = base, vae, lora, lora_scale
        self.device = device
        self._pipe = None
        self._img2img = None
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------
    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        kw = {"torch_dtype": dtype, "use_safetensors": True}
        if self.vae_id:
            kw["vae"] = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)
        if self.base.endswith(".safetensors"):
            pipe = StableDiffusionXLPipeline.from_single_file(self.base, **kw)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(self.base, **kw)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        pipe.to(device)
        if device == "cuda":
            pipe.enable_vae_slicing()
        if self.lora_path and Path(self.lora_path).exists():
            pipe.load_lora_weights(str(Path(self.lora_path).parent), weight_name=Path(self.lora_path).name,
                                   adapter_name="rbx")
            pipe.set_adapters(["rbx"], adapter_weights=[self.lora_scale])
            print(f"[pipeline] LoRA loaded: {self.lora_path} @ {self.lora_scale}")
        else:
            print("[pipeline] no LoRA loaded (set LORA_PATH)")
        self._pipe = pipe
        self._device = device
        self._compel = None
        try:
            from compel import Compel, ReturnedEmbeddingsType
            self._compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2], text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True], truncate_long_prompts=False)
            print("[pipeline] compel enabled: (text:1.2) weights and long prompts supported")
        except ImportError:
            print("[pipeline] compel not installed; prompt weights stripped, prompts truncated at 77 tokens")

    def _cond(self, prompt: str, negative: str) -> dict:
        """Prompt kwargs for the pipeline, via Compel when available."""
        if self._compel is None:
            return {"prompt": strip_weights(prompt), "negative_prompt": strip_weights(negative)}
        # Compel uses (text)1.2 syntax; convert from (text:1.2)
        conv = lambda s: _WEIGHT_RE.sub(r"(\1)\2", s)  # noqa: E731
        c, pc = self._compel(conv(prompt))
        n, pn = self._compel(conv(negative))
        [c, n] = self._compel.pad_conditioning_tensors_to_same_length([c, n])
        return {"prompt_embeds": c, "pooled_prompt_embeds": pc,
                "negative_prompt_embeds": n, "negative_pooled_prompt_embeds": pn}

    def set_lora_scale(self, scale: float) -> None:
        self.lora_scale = scale
        if self._pipe is not None and self.lora_path:
            self._pipe.set_adapters(["rbx"], adapter_weights=[scale])

    # -- generation ----------------------------------------------------------
    def generate(self, prompt: str, negative: str, seeds: List[int], width: int = 1344, height: int = 768,
                 steps: int = 28, cfg: float = 6.0, lora_scale: Optional[float] = None) -> List[Image.Image]:
        import torch
        with self._lock:
            self._load()
            if lora_scale is not None and lora_scale != self.lora_scale:
                self.set_lora_scale(lora_scale)
            out: List[Image.Image] = []
            for s in seeds:
                g = torch.Generator(device=self._device).manual_seed(int(s))
                img = self._pipe(**self._cond(prompt, negative), width=width, height=height,
                                 num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0]
                out.append(img)
            return out

    def variations(self, init: Image.Image, prompt: str, negative: str, seeds: List[int], strength: float = 0.55,
                   steps: int = 28, cfg: float = 6.0, width: Optional[int] = None,
                   height: Optional[int] = None) -> List[Image.Image]:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline
        with self._lock:
            self._load()
            if self._img2img is None:
                self._img2img = StableDiffusionXLImg2ImgPipeline(**self._pipe.components)
            if width and height:
                init = init.resize((width, height), Image.LANCZOS)
            out: List[Image.Image] = []
            for s in seeds:
                g = torch.Generator(device=self._device).manual_seed(int(s))
                img = self._img2img(**self._cond(prompt, negative), image=init, strength=strength,
                                    num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0]
                out.append(img)
            return out

    def upscale_1080p(self, img: Image.Image, prompt: str, negative: str, seed: int) -> Image.Image:
        """Hi-res pass: Lanczos to 1920x1080 then a light img2img to re-sharpen."""
        return self.variations(img, prompt, negative, [seed], strength=0.3, steps=20, width=1920, height=1080)[0]
