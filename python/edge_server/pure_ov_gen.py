"""Pure-OpenVINO text generation for Qwen3-VL-4B INT4.

Deliberately independent of torch / transformers (which drag in heavy deps and
version hell). Only three things:
  - openvino  : compile the shipped IR graphs (text-embeddings + language model)
  - tokenizers : read the model's tokenizer.json directly
  - numpy      : tensor assembly

The language-model IR in this repo is *stateless* (full-context, no KV-cache
inputs), so each decode step re-runs the whole sequence. That is O(n^2) in
context length - slow on CPU, fine for an end-to-end validation slice. A
KV-cache'd graph (compiled from optimum-exported IR) is the performance follow-up.

Device policy: NPU preferred (the small-model tier target); auto-fall back to
CPU when the NPU plugin is absent or compile fails.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

# Special token ids used by the Qwen3 chat template (from tokenizer.json).
_IM_START = 151644  # <|im_start|>
_IM_END = 151645    # <|im_end|>
_EOT = 151643       # <|endoftext|>


@dataclass
class GenResult:
    text: str
    device: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    est_tok_s: float
    new_ids: List[int] = field(default_factory=list)


class PureOVGenerator:
    """Self-contained OpenVINO generator for a Qwen3-VL-4B INT4 model dir."""

    def __init__(self, model_dir: str, device: str = "NPU") -> None:
        self.model_dir = model_dir
        self.core = ov.Core()
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.device = device
        self._load()

    def _load(self) -> None:
        """Compile the two IR graphs, NPU->CPU fallback."""
        avail = self.core.get_available_devices()
        dev = self.device if self.device in avail else "CPU"
        try:
            self.embeds = self.core.compile_model(
                self.core.read_model(os.path.join(self.model_dir, "openvino_text_embeddings_model.xml")),
                dev,
            )
            self.lm = self.core.compile_model(
                self.core.read_model(os.path.join(self.model_dir, "openvino_language_model.xml")),
                dev,
            )
            self.embed_req = self.embeds.create_infer_request()
            self.lm_req = self.lm.create_infer_request()
            self.device = dev
        except Exception:
            # NPU failed (driver / model too big for NPU) -> retry on CPU.
            self.device = "CPU"
            self.embeds = self.core.compile_model(
                self.core.read_model(os.path.join(self.model_dir, "openvino_text_embeddings_model.xml")),
                "CPU",
            )
            self.lm = self.core.compile_model(
                self.core.read_model(os.path.join(self.model_dir, "openvino_language_model.xml")),
                "CPU",
            )
            self.embed_req = self.embeds.create_infer_request()
            self.lm_req = self.lm.create_infer_request()

    # ------------------------------------------------------------------
    def build_prompt(self, content: str, system: str = "") -> List[int]:
        """Assemble the Qwen3 chat template into token ids."""
        parts = ""
        if system:
            parts += f"<|im_start|>system\n{system}<|im_end|>\n"
        parts += f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        return self.tok.encode(parts).ids

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        """Run the text-embeddings IR -> [1, L, 2560]."""
        out = self.embed_req.infer({"input": ids.astype(np.int64)})
        return out["inputs_embeds"].astype(np.float32)

    def _lm_forward(self, embeds: np.ndarray, past_len: int, total_len: int) -> np.ndarray:
        """One LM forward for the *new* tokens only (stateful KV cache).

        `embeds` covers the newly appended tokens (prefill: whole prompt;
        decode: 1 token). `past_len` is how many tokens are already cached in
        the model state, `total_len` is past_len + new tokens.
        """
        b, L, H = embeds.shape
        attention = np.ones((b, total_len), dtype=np.int64)
        # position_ids: [3, B, L] for the NEW tokens, offset by past_len.
        pos = np.zeros((3, b, L), dtype=np.int64)
        for c in range(3):
            pos[c] = np.arange(past_len, past_len + L, dtype=np.int64)
        vis_mask = np.zeros((b, L), dtype=np.int8)
        deepstack = np.zeros((b, L, H), dtype=np.float32)
        beam = np.zeros((b,), dtype=np.int32)
        res = self.lm_req.infer({
            "attention_mask": attention,
            "position_ids": pos,
            "inputs_embeds": embeds,
            "visual_pos_masks": vis_mask,
            "deepstack_visual_embeds": deepstack,
            "beam_idx": beam,
        })
        return res["logits"]

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
    ) -> GenResult:
        t0 = time.time()
        prompt = list(prompt_ids)

        # Reset the KV-cache state so this request starts clean.
        self.lm_req.reset_state()

        # --- prefill: whole prompt at once, positions 0..P-1 ---
        cur = np.array([prompt], dtype=np.int64)
        embeds = self._embed(cur)                       # (1, P, H)
        logits = self._lm_forward(embeds, 0, len(prompt))  # (1, P, V)

        new_ids: List[int] = []
        past = len(prompt)
        for _ in range(max_new_tokens):
            last = logits[0, -1, :].astype(np.float64)

            # Repetition penalty over tokens already generated.
            if repetition_penalty and repetition_penalty != 1.0 and new_ids:
                seen = np.unique(np.array(new_ids, dtype=np.int64))
                last[seen] = np.where(
                    last[seen] > 0,
                    last[seen] / repetition_penalty,
                    last[seen] * repetition_penalty,
                )

            if do_sample and temperature > 0:
                z = last / max(temperature, 1e-5)
                z = z - z.max()
                p = np.exp(z)
                p = p / p.sum()
                nxt = int(np.random.choice(len(p), p=p))
            else:
                nxt = int(np.argmax(last))

            new_ids.append(nxt)
            if nxt in (_IM_END, _EOT):
                break

            # --- decode: feed one new token, cache advances by 1 ---
            one = np.array([[nxt]], dtype=np.int64)
            e1 = self._embed(one)                        # (1, 1, H)
            logits = self._lm_forward(e1, past, past + 1)
            past += 1

        dt = max(time.time() - t0, 1e-3)
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        return GenResult(
            text=text,
            device=self.device,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(new_ids),
            seconds=dt,
            est_tok_s=(len(new_ids) / dt) if new_ids else 0.0,
            new_ids=new_ids,
        )

    def chat(self, content: str, system: str = "", max_new_tokens: int = 128,
             do_sample: bool = False, temperature: float = 1.0,
             repetition_penalty: float = 1.2) -> GenResult:
        ids = self.build_prompt(content, system)
        return self.generate(ids, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, temperature=temperature,
                             repetition_penalty=repetition_penalty)
