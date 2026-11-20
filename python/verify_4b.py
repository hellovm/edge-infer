"""Verify the 4B small-model tier end-to-end.

Steps:
  1. List the OpenVINO devices actually visible on this box (NPU? CPU? GPU?).
  2. Confirm the OpenAI-compatible server routes exist.
  3. Load the Qwen3-VL-4B INT4 IR with the pure-OpenVINO generator and run one
     real chat completion (NPU preferred, CPU fallback).

Run:
    python verify_4b.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_REPO_ROOT, "models", "qwen3-vl-4b-int4")


def main() -> int:
    # 1) device discovery
    print("=== 1. OpenVINO devices ===")
    try:
        import openvino as ov

        core = ov.Core()
        devs = core.get_available_devices()
        print(f"OpenVINO {ov.__version__} available devices: {devs}")
        for d in devs:
            try:
                props = {k: str(core.get_property(d, k)) for k in ("AVAILABLE_DEVICES",)}
                print(f"  - {d}: {json.dumps(props, ensure_ascii=False)}")
            except Exception as e:  # noqa: BLE001
                print(f"  - {d}: (prop read failed: {e})")
    except Exception as e:  # noqa: BLE001
        print(f"OpenVINO import/init failed: {e}")
        return 1

    # 2) OpenAI contract
    print("\n=== 2. OpenAI-compatible routes ===")
    try:
        from edge_server.openai_server import app, _GEN_AVAILABLE

        routes = sorted({r.path for r in app.routes if r.path.startswith("/v1")})
        print(f"routes: {routes}")
        print(f"generator available: {_GEN_AVAILABLE}")
    except Exception as e:  # noqa: BLE001
        print(f"server import failed: {e}")

    # 3) real inference
    print("\n=== 3. real inference ===")
    if not os.path.isdir(_MODEL_DIR):
        print(f"model dir not found: {_MODEL_DIR}")
        print("run: python -m edge_server.download_modelscope")
        return 0
    from edge_server.pure_ov_gen import PureOVGenerator

    gen = PureOVGenerator(_MODEL_DIR, device="NPU")
    print(f"loaded on device: {gen.device}")
    q = "你好，请用一句话介绍三层卸载推理架构。"
    r = gen.chat(q, max_new_tokens=32)
    print(f"Q: {q}")
    print(f"A: {r.text}")
    print(f"   [{r.device} {r.completion_tokens}tok {r.seconds:.1f}s {r.est_tok_s:.2f} tok/s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
