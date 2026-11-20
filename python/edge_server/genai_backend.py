"""OpenVINO GenAI backend — runs LLMs on the Intel NPU (and CPU/GPU).

Key finding (2026-11-20): the Meteor Lake NPU (NPU3720) only supports
**symmetric** INT4 weights (channel-wise / group-wise). Asymmetric INT4
(e.g. AWQ exports) trips the vpux compiler with
`Unsupported DynamicDequantizeOp configuration ... arch=NPU37XX`.

So for NPU we must:
  1. use a symmetric-int4 export (`--weight-format int4 --sym --group-size 128`,
     or the ModelScope `*-int4-cw-ov` repos), and
  2. drive it with `openvino_genai.LLMPipeline`, which handles the static-shape
     requirements internally (optimum-intel's OVModelForCausalLM does not).

This module wraps LLMPipeline with a small, dependency-light API and reports
the actual device + tok/s so the GUI can show real numbers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import openvino_genai as ov_genai

    _GENAI_AVAILABLE = True
except Exception:
    ov_genai = None
    _GENAI_AVAILABLE = False


@dataclass
class GenaiResult:
    text: str
    device: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    est_tok_s: float


def is_llm_dir(model_dir: str) -> bool:
    """True when the dir holds a plain LLM IR (openvino_model.xml)."""
    return os.path.exists(os.path.join(model_dir, "openvino_model.xml"))


class GenaiBackend:
    """Lazy LLMPipeline wrapper with NPU->CPU fallback."""

    def __init__(self) -> None:
        self._pipe = None
        self._device: Optional[str] = None
        self._model_dir: Optional[str] = None

    @property
    def available(self) -> bool:
        return _GENAI_AVAILABLE

    def load(self, model_dir: str, device: str = "NPU", cache_dir: Optional[str] = None):
        """Load (or reuse) the pipeline for `model_dir` on `device`.

        Falls back to CPU if the NPU compile fails (e.g. asymmetric weights).
        """
        if not _GENAI_AVAILABLE:
            raise RuntimeError("openvino_genai not installed")
        if self._pipe is not None and self._model_dir == model_dir and self._device == device:
            return self._pipe

        kwargs = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            kwargs["CACHE_DIR"] = cache_dir

        try:
            self._pipe = ov_genai.LLMPipeline(model_dir, device, **kwargs)
            self._device = device
        except Exception:
            self._pipe = ov_genai.LLMPipeline(model_dir, "CPU", **kwargs)
            self._device = "CPU"
        self._model_dir = model_dir
        return self._pipe

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.0) -> GenaiResult:
        if self._pipe is None:
            raise RuntimeError("pipeline not loaded")
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.temperature = temperature
        else:
            cfg.do_sample = False

        t0 = time.time()
        text = self._pipe.generate(prompt, cfg)
        dt = max(time.time() - t0, 1e-3)

        # LLMPipeline returns a string; token counts come from the tokenizer
        # inside the pipeline, so we approximate completion tokens by length.
        comp = max(len(text) // 2, 1)
        return GenaiResult(
            text=text,
            device=self._device or "CPU",
            prompt_tokens=max(len(prompt) // 2, 1),
            completion_tokens=comp,
            seconds=dt,
            est_tok_s=comp / dt,
        )

    def _build_chat_prompt(self, content: str, system: str,
                           enable_thinking: bool) -> str:
        """Assemble a Qwen-style chat prompt (/no_think when thinking off)."""
        is_qwen3 = "qwen3" in (self._model_dir or "").lower()
        parts = ""
        if is_qwen3 and not enable_thinking:
            parts += "/no_think\n"
        if system:
            parts += f"<|im_start|>system\n{system}<|im_end|>\n"
        parts += f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        return parts

    def stream_chat(self, content: str, system: str = "", max_new_tokens: int = 256,
                    temperature: float = 0.0, enable_thinking: Optional[bool] = None):
        """Yield text deltas as they are produced (token streaming)."""
        if self._pipe is None:
            raise RuntimeError("pipeline not loaded")
        if enable_thinking is None:
            enable_thinking = os.environ.get("EDGE_ENABLE_THINKING", "0") == "1"

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.temperature = temperature
        else:
            cfg.do_sample = False

        prompt = self._build_chat_prompt(content, system, enable_thinking)

        # openvino_genai invokes the streamer with each decoded subword.
        # We push deltas into a queue and drain it from the generator.
        import queue as _q

        chunks: "_q.Queue" = _q.Queue()

        def _streamer(subword: str) -> bool:
            chunks.put(subword)
            return False  # never stop early

        import threading

        err: list = []

        def _run():
            try:
                self._pipe.generate(prompt, cfg, _streamer)
            except Exception as e:  # noqa: BLE001
                err.append(e)
            finally:
                chunks.put(None)  # sentinel

        th = threading.Thread(target=_run, daemon=True)
        th.start()

        full = []
        while True:
            item = chunks.get()
            if item is None:
                break
            full.append(item)
            yield item

        th.join(timeout=1.0)
        if err:
            # Surface load/生成 errors to the caller.
            raise err[0]

    def chat(self, content: str, system: str = "", max_new_tokens: int = 256,
             temperature: float = 0.0, enable_thinking: Optional[bool] = None) -> GenaiResult:
        """Chat via the pipeline's built-in chat template.

        For Qwen3 models, thinking is disabled by default (much faster for
        interactive use) unless EDGE_ENABLE_THINKING=1 or enable_thinking=True.
        """
        if self._pipe is None:
            raise RuntimeError("pipeline not loaded")
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = max_new_tokens
        if temperature > 0:
            cfg.temperature = temperature
        else:
            cfg.do_sample = False

        if enable_thinking is None:
            enable_thinking = os.environ.get("EDGE_ENABLE_THINKING", "0") == "1"

        prompt = self._build_chat_prompt(content, system, enable_thinking)

        t0 = time.time()
        text = self._pipe.generate(prompt, cfg)
        dt = max(time.time() - t0, 1e-3)
        comp = max(len(text) // 2, 1)
        return GenaiResult(
            text=text, device=self._device or "CPU",
            prompt_tokens=max(len(content) // 2, 1),
            completion_tokens=comp, seconds=dt, est_tok_s=comp / dt,
        )


# module-level singleton
backend = GenaiBackend()
