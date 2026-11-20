"""Start the edge-infer OpenAI-compatible server (real 4B inference).

Usage (from the python/ dir):
    python run_server.py                 # NPU preferred, CPU fallback, port 8080
    EDGE_DEVICE=CPU python run_server.py # force CPU
    EDGE_API_KEY=secret python run_server.py

Env:
    EDGE_DEVICE     "NPU" (default) | "CPU" | "GPU.0" ...
    EDGE_API_KEY    bearer token for /v1/* (empty = auth disabled)
    EDGE_MODEL_DIR  override model dir (default: <repo>/models/qwen3-vl-4b-int4)
    EDGE_PORT       listen port (default 8080)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    import uvicorn

    host = os.environ.get("EDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("EDGE_PORT", "8080"))
    print(f"[edge-infer] serving OpenAI-compatible API on {host}:{port}")
    print(f"[edge-infer] device={os.environ.get('EDGE_DEVICE','NPU')} "
          f"auth={'on' if os.environ.get('EDGE_API_KEY') else 'off'}")
    uvicorn.run("edge_server.openai_server:app", host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
