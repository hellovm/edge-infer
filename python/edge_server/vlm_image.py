"""VLM image inference via OpenVINO GenAI's VLMPipeline.

OpenVINO IR for vision-language models (e.g. Qwen3-VL) exposes separate
vision-embedding graphs; driving them by hand means re-implementing the
Qwen2VL image processor (patchify, merge, rotary embeds). ``VLMPipeline``
already does all of that, so we use it and only supply the preprocessed
``ov.Tensor`` images it expects.

Preprocessing (per ``preprocessor_config.json``): RGB, resize so both sides are
multiples of ``patch_size * merge_size`` (= 32), scale to [0,1], normalise with
mean=std=0.5, layout ``channels_first``. ``temporal_patch_size=2`` means the
image is duplicated along a leading frame axis.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    import openvino as ov
    import openvino_genai as ov_genai
    _AVAILABLE = True
except Exception:  # noqa: BLE001
    _AVAILABLE = False


_PATCH = 16
_MERGE = 2
_FACTOR = _PATCH * _MERGE          # 32
_TEMPORAL = 2


def vlm_available() -> bool:
    return _AVAILABLE


def _decode_data_url(url: str) -> bytes:
    """Accept both raw base64 and ``data:image/png;base64,...`` URLs."""
    if "," in url and url.strip().startswith("data:"):
        url = url.split(",", 1)[1]
    return base64.b64decode(url)


def preprocess_image(data_url: str, max_side: int = 1024) -> "ov.Tensor":
    """Decode one image into the raw uint8 RGB tensor VLMPipeline expects.

    VLMPipeline runs the model's own image processor (resize / normalise /
    patchify) internally, so we only decode to RGB uint8 here.
    """
    from PIL import Image

    raw = _decode_data_url(data_url)
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    w, h = img.size
    # Cap the long side so huge screenshots don't blow up NPU/GPU memory.
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.uint8)        # HWC, RGB, 0..255
    return ov.Tensor(np.ascontiguousarray(arr))


@dataclass
class VlmResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    device: str

    @property
    def est_tok_s(self) -> float:
        return self.completion_tokens / max(self.seconds, 1e-3)


class VlmImagePipeline:
    """Thin wrapper: lazily build a VLMPipeline, run text+image generation."""

    def __init__(self, model_dir: str, device: str = "GPU.0") -> None:
        self.model_dir = model_dir
        self.device = device
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return self._pipe
        if not _AVAILABLE:
            raise RuntimeError("openvino_genai not installed")
        try:
            self._pipe = ov_genai.VLMPipeline(self.model_dir, self.device)
        except Exception:
            self._pipe = ov_genai.VLMPipeline(self.model_dir, "CPU")
            self.device = "CPU"
        return self._pipe

    def chat(
        self,
        content: str,
        images: Sequence[str],
        system: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> VlmResult:
        import time

        pipe = self._ensure()
        tensors = [preprocess_image(u) for u in images] if images else []

        prompt = ""
        if system:
            prompt += f"<|im_start|>system\n{system}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.temperature = temperature
        else:
            cfg.do_sample = False

        t0 = time.time()
        if tensors:
            out = pipe.generate(prompt, images=tensors, generation_config=cfg)
        else:
            out = pipe.generate(prompt, generation_config=cfg)
        dt = max(time.time() - t0, 1e-3)

        text = out.texts[0] if hasattr(out, "texts") else str(out)
        comp = max(len(text) // 2, 1)
        return VlmResult(text=text, prompt_tokens=max(len(content) // 2, 1),
                         completion_tokens=comp, seconds=dt, device=self.device)
