"""Benchmark OpenVINO GenAI LLMPipeline on one device.

Reports **prompt processing (prefill)** and **token generation (decode)**
separately -- the two numbers the README table needs.

openvino_genai 2026.x ``generate()`` returns a plain ``str`` and exposes no
``perf_metrics``, so timing is measured directly:

  * prefill  = n_prompt_tokens / ttft        (ttft = time to first streamed token)
  * decode   = n_streamed_subwords / (t_last - t_first)

Usage:
    python scripts/bench_ov.py <model_dir> <DEVICE> [--prompt 512] [--gen 128]

DEVICE: CPU | GPU.0 (Intel Arc) | GPU.1 (NVIDIA via OpenVINO) | NPU
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _prompt_tokens(pipe, text: str) -> int:
    """Best-effort token count of `text` using the pipeline's tokenizer."""
    try:
        tok = pipe.get_tokenizer()
        ti = tok.encode(text)
        ids = getattr(ti, "input_ids", None)
        if ids is not None and hasattr(ids, "shape"):
            return int(ids.shape[-1])
        if ids is not None:
            try:
                return len(ids)
            except TypeError:
                pass
    except Exception:
        pass
    return max(len(text) // 2, 1)   # CJK ~2 chars/token fallback


def _build_prompt(pipe, target_tokens: int) -> str:
    unit = "请逐条说明下面事项的细节、理由与可能的例外情况。"
    text = ""
    for i in range(1, 500):
        text += f"{i}. {unit}\n"
        if _prompt_tokens(pipe, text) >= target_tokens:
            break
    return text


def run(model_dir: str, device: str, n_prompt: int, n_gen: int,
        runs: int, cache: str) -> dict:
    import openvino_genai as ov_genai

    kw = {}
    if cache:
        os.makedirs(cache, exist_ok=True)
        kw["CACHE_DIR"] = cache

    t0 = time.time()
    pipe = ov_genai.LLMPipeline(model_dir, device, **kw)
    load_s = time.time() - t0

    prompt = _build_prompt(pipe, n_prompt)
    n_in = _prompt_tokens(pipe, prompt)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = n_gen

    best = None
    for _ in range(runs):
        marks: list[float] = []

        def cb(_sub: str) -> bool:
            marks.append(time.perf_counter())
            return False

        t = time.perf_counter()
        pipe.generate(prompt, cfg, cb)
        t_end = time.perf_counter()

        if not marks:
            continue
        ttft = marks[0] - t
        n_out = len(marks)
        dec = (n_out - 1) / (marks[-1] - marks[0]) if n_out > 1 else 0.0
        rec = {
            "wall_s": t_end - t,
            "ttft_s": ttft,
            "n_gen": n_out,
            "prefill_tok_s": n_in / ttft if ttft > 0 else None,
            "decode_tok_s": dec or None,
        }
        if best is None or rec["wall_s"] < best["wall_s"]:
            best = rec

    if best is None:
        return {"device": device, "error": "no tokens streamed"}

    return {
        "device": device,
        "model": os.path.basename(model_dir.rstrip("/\\")),
        "load_s": round(load_s, 2),
        "n_prompt": n_in,
        "n_gen": best["n_gen"],
        "prefill_tok_s": round(best["prefill_tok_s"], 1) if best["prefill_tok_s"] else None,
        "decode_tok_s": round(best["decode_tok_s"], 1) if best["decode_tok_s"] else None,
        "ttft_s": round(best["ttft_s"], 3),
        "wall_s": round(best["wall_s"], 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("device")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--cache", default="")
    a = ap.parse_args()

    try:
        out = run(a.model_dir, a.device, a.prompt, a.gen, a.runs, a.cache)
    except Exception as e:  # noqa: BLE001
        out = {"device": a.device, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    print(json.dumps(out, ensure_ascii=False))
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
