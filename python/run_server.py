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
    EDGE_LOG_LEVEL  DEBUG | INFO (default) | WARNING
    EDGE_DIAG_BOOT  1 = write an environment report at startup
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    import logging

    from edge_server.logging_setup import setup_logging, LOG_FILE
    from edge_server import diagnostics

    log_path = setup_logging()
    log = logging.getLogger("edge.boot")

    import uvicorn

    host = os.environ.get("EDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("EDGE_PORT", "8080"))
    device = os.environ.get("EDGE_DEVICE", "NPU")
    log.info("serving OpenAI-compatible API on %s:%s (device=%s, auth=%s)",
             host, port, device, "on" if os.environ.get("EDGE_API_KEY") else "off")
    log.info("log file: %s", LOG_FILE)

    # Startup environment snapshot. Cheap (fast=True skips the slow probes) and
    # invaluable when a user on a different machine (e.g. AMD) reports trouble.
    try:
        rep = diagnostics.collect_report(tag="boot", fast=True)
        a = rep.get("assessment", {})
        log.info(
            "hardware: nvidia_cuda=%s intel_gpu=%s intel_npu=%s "
            "amd_npu_detected=%s amd_gpu_detected=%s",
            a.get("nvidia_cuda", {}).get("supported"),
            a.get("intel_gpu", {}).get("supported"),
            a.get("intel_npu", {}).get("supported"),
            a.get("amd", {}).get("npu_hardware_detected"),
            a.get("amd", {}).get("gpu_hardware_detected"),
        )
        if os.environ.get("EDGE_DIAG_BOOT") == "1":
            from edge_server.diagnostics import build_bundle
            log.info("boot diagnostics bundle: %s", build_bundle(tag="boot"))
    except Exception as e:  # noqa: BLE001
        log.warning("startup diagnostics failed: %s", e)
        diagnostics.record_problem("startup_diagnostics", str(e), exc=e)

    try:
        uvicorn.run("edge_server.openai_server:app", host=host, port=port,
                    log_config=None)
    except Exception as e:  # noqa: BLE001
        log.exception("server crashed")
        diagnostics.record_problem("server_crash", str(e), exc=e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
