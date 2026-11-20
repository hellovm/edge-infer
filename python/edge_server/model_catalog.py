"""Recommended model catalog — one entry per model, with a variant per device.

Design
------
A *model* here is a logical name (e.g. "Qwen3.6-35B-A3B"). Each model exposes
one or more **variants**, and every variant targets a specific device:

  - ``nvidia``     → GGUF, run by **llama.cpp CUDA**
  - ``intel-gpu``  → OpenVINO IR, run by **OpenVINO**
  - ``npu``        → OpenVINO IR (symmetric INT4 only)

The GUI groups by model, then shows/vends the variant that matches the user's
selected device — so choosing the NVIDIA card downloads the GGUF, choosing the
Intel iGPU downloads the INT4 IR. Nothing is hand-picked by the user.

Why the split exists
--------------------
Measured on this machine (RTX 4070 Laptop + Intel Arc iGPU + NPU):

  llama.cpp CUDA (RTX)   126.7 tok/s   ← NVIDIA's native path
  OpenVINO  Arc iGPU      27.1 tok/s   ← Intel's native path
  OpenVINO  CPU           28.8 tok/s
  OpenVINO  RTX (GPU.1)    6.1 tok/s   ← ~20× slower; not its target hardware

So NVIDIA must use llama.cpp/GGUF, Intel must use OpenVINO/IR. One model,
two files, auto-chosen by device.

Quantization note
-----------------
FP4/NVFP4 needs Blackwell (RTX 50, sm120) FP4 tensor cores. Ada (RTX 40, sm89)
has no FP4 datapath — NVFP4 falls back to software emulation and can be slower.
For RTX 40-class, GGUF Q4_K_M/Q5_K_M (llama.cpp) or FP8 (vLLM/TensorRT) are the
right choices.

All repos verified reachable on ModelScope (2026-11-20) via the platform API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Each model: metadata + variants keyed by device ("nvidia" | "intel-gpu" | "npu").
MODELS: List[dict] = [
    # ------------------------------------------------------------------ Qwen3.6-35B-A3B
    {
        "name": "Qwen3.6-35B-A3B",
        "params": "35B",
        "active": "3B",
        "mm": True,
        "note": "多模态 MoE · 激活仅 3B · 三层卸载首选（本机跑不动 → 卸载的原点场景）",
        "variants": {
            "nvidia": {
                "repo": "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
                "quant": "GGUF Q4_K_M",
                "backend": "llama.cpp (CUDA)",
                "size_gb": 21.2,
                "ram_hint": "≥ 16 GB 内存；8 GB 显存可常驻热专家 + 热 KV",
                "min_ssd_gb": 22,
                "files": ["*Q4_K_M.gguf", "*mmproj*.gguf"],
            },
            "intel-gpu": {
                "repo": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
                "quant": "INT4",
                "backend": "OpenVINO (GPU)",
                "size_gb": 19.4,
                "ram_hint": "≥ 24 GB 内存；Arc 核显共享内存，较吃整机 RAM",
                "min_ssd_gb": 20,
            },
        },
    },
    # ------------------------------------------------------------------ Qwen3-30B-A3B
    {
        "name": "Qwen3-30B-A3B",
        "params": "30B",
        "active": "3B",
        "mm": False,
        "note": "MoE · 激活仅 3B · NVIDIA 与 Intel 均可跑（各自原生后端）",
        "variants": {
            "nvidia": {
                "repo": "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF",
                "quant": "GGUF Q4_K_M",
                "backend": "llama.cpp (CUDA)",
                "size_gb": 18.6,
                "ram_hint": "≥ 16 GB 内存；8 GB 显存 + 三层卸载",
                "min_ssd_gb": 19,
                "files": ["*Q4_K_M.gguf"],
            },
            "intel-gpu": {
                "repo": "OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov",
                "quant": "INT4",
                "backend": "OpenVINO (GPU)",
                "size_gb": 16.3,
                "ram_hint": "≥ 16 GB 内存；Arc 核显共享系统内存，整机建议 32 GB",
                "min_ssd_gb": 17,
            },
        },
    },
    # ------------------------------------------------------------------ Qwen3-Next-80B-A3B
    {
        "name": "Qwen3-Next-80B-A3B",
        "params": "80B",
        "active": "3B",
        "mm": False,
        "note": "超大规模 MoE · 本机显存装不下 → 三层卸载的核心场景（仅 GGUF）",
        "variants": {
            "nvidia": {
                "repo": "lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF",
                "quant": "GGUF Q4_K_M",
                "backend": "llama.cpp (CUDA)",
                "size_gb": 48.5,
                "ram_hint": "≥ 16 GB 内存；约 50 GB SSD 承载全量；激活仅 3B，速度可控",
                "min_ssd_gb": 50,
                "files": ["*Q4_K_M.gguf"],
            },
        },
    },
    # ------------------------------------------------------------------ Qwen3-8B
    {
        "name": "Qwen3-8B",
        "params": "8B",
        "active": "8B",
        "mm": False,
        "note": "通用小模型 · NPU 低功耗常驻 · Intel GPU / CPU 亦可",
        "variants": {
            "nvidia": {
                "repo": "unsloth/Qwen3-8B-GGUF",
                "quant": "GGUF Q4_K_M",
                "backend": "llama.cpp (CUDA)",
                "size_gb": 5.0,
                "ram_hint": "≥ 8 GB 内存；6 GB 显存可全量常驻",
                "min_ssd_gb": 6,
                "files": ["*Q4_K_M.gguf"],
            },
            "intel-gpu": {
                "repo": "OpenVINO/Qwen3-8B-int4-cw-ov",
                "quant": "INT4_SYM",
                "backend": "OpenVINO (GPU)",
                "size_gb": 4.75,
                "ram_hint": "≥ 12 GB 内存",
                "min_ssd_gb": 5,
            },
            "npu": {
                "repo": "OpenVINO/Qwen3-8B-int4-cw-ov",
                "quant": "INT4_SYM",
                "backend": "OpenVINO (NPU)",
                "size_gb": 4.75,
                "ram_hint": "≥ 12 GB 内存；NPU 常驻约 5 GB",
                "min_ssd_gb": 5,
                "npu_compatible": True,
            },
        },
    },
    # ------------------------------------------------------------------ MiniCPM-V-4.5
    {
        "name": "MiniCPM-V-4.5",
        "params": "9B",
        "active": "9B",
        "mm": True,
        "note": "多模态 · 单图/多图/视频 · NPU 实测可跑（首次加载约 50s）",
        "variants": {
            "nvidia": {
                "repo": "OpenBMB/MiniCPM-V-4_5-GGUF",
                "quant": "GGUF Q4_K_M",
                "backend": "llama.cpp (CUDA)",
                "size_gb": 6.0,
                "ram_hint": "≥ 12 GB 内存；8 GB 显存可全量常驻",
                "min_ssd_gb": 7,
                "files": ["*Q4_K_M.gguf", "*mmproj*.gguf"],
            },
            "intel-gpu": {
                "repo": "OpenVINO/MiniCPM-V-4_5-int4-cw-ov",
                "quant": "INT4_SYM",
                "backend": "OpenVINO (GPU)",
                "size_gb": 5.26,
                "ram_hint": "≥ 16 GB 内存",
                "min_ssd_gb": 6,
            },
            "npu": {
                "repo": "OpenVINO/MiniCPM-V-4_5-int4-cw-ov",
                "quant": "INT4_SYM",
                "backend": "OpenVINO (NPU)",
                "size_gb": 5.26,
                "ram_hint": "≥ 16 GB 内存；NPU 实测可跑（首次加载约 50s）",
                "min_ssd_gb": 6,
                "npu_compatible": True,
            },
        },
    },
]

# GUI / API device keys → display label
DEVICE_LABELS = {
    "nvidia": "NVIDIA GPU",
    "intel-gpu": "Intel GPU",
    "npu": "Intel NPU",
}


def all_variants() -> List[dict]:
    """Flatten to one row per (model, device) variant."""
    out = []
    for m in MODELS:
        for dev, v in m["variants"].items():
            out.append({
                **v, "device": dev, "device_label": DEVICE_LABELS.get(dev, dev),
                "name": m["name"], "params": m["params"], "active": m["active"],
                "mm": m["mm"], "note": m["note"],
            })
    return out


def find(repo: str) -> Optional[dict]:
    for it in all_variants():
        if it["repo"] == repo:
            return it
    return None


def resolve_device(device: Optional[str]) -> Optional[str]:
    """Map a GUI device alias to a catalog device key."""
    if not device:
        return None
    d = str(device).upper().replace(" ", "").replace("_", "")
    if d in ("NVIDIA-GPU", "NVIDIA", "CUDA", "GPU.1"):
        return "nvidia"
    if d in ("INTEL-GPU", "IGPU", "INTELGPU", "GPU.0"):
        return "intel-gpu"
    if d == "NPU":
        return "npu"
    return None
