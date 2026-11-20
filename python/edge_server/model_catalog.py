"""Recommended model catalog per accelerator.

All repos verified reachable on ModelScope (2026-11-20). Nothing here is guessed:
each `repo` was resolved via the ModelScope API and its size taken from the file
listing.

Design intent
-------------
This project exists to run models **larger than the host can hold in fast
memory**, by splitting weights across tiers (SSD / RAM / VRAM for discrete GPUs;
SSD / RAM for NPU & iGPU which draw from system memory). So the catalog favours
**Mixture-of-Experts (MoE)** checkpoints: their per-token working set is only the
*active* parameters, which is exactly what makes 3-tier offload pay off.

Backends — matched to what each vendor is actually fast at
---------------------------------------------------------
Measured on this machine (RTX 4070 Laptop + Intel Arc iGPU + Intel NPU):

  backend / device              tok/s   verdict
  llama.cpp CUDA (GPU.1=RTX)    126.7   ✅ NVIDIA's native path
  OpenVINO GPU.0 (Arc iGPU)      27.1   ✅ Intel's native path
  OpenVINO CPU                   28.8   ✅
  OpenVINO GPU.1 (RTX via OV)      6.1  ❌ 20× slower — OpenVINO's GPU plugin
                                            targets Intel GPUs; NVIDIA is not
                                            its optimisation target.

So: **NVIDIA → llama.cpp (GGUF, CUDA)**; **Intel GPU / NPU → OpenVINO (IR)**.
Passing an NVIDIA card through OpenVINO technically works but throws away ~20×;
we route it to llama.cpp instead.

Quantization note
-----------------
FP4/NVFP4 requires Blackwell (RTX 50 / B200, sm120). Ada (RTX 40, sm89) has FP8
tensor cores but **no FP4 datapath** — NVFP4 degrades to software emulation and
can be slower. For RTX 40-class cards the right formats are **FP8** or **GGUF
Q4_K_M/Q5_K_M**. FP8 needs vLLM/TensorRT-LLM; within llama.cpp the practical
choice is Q4_K_M/Q5_K_M.

Field notes
-----------
- ``layers``     : tiers the checkpoint can be split across.
- ``active``     : active params per token (drives the decode-speed floor).
- ``min_ssd_gb`` : disk needed to hold the full checkpoint for offload.
- ``files``      : download allow-list (GGUF repos bundle many quantizations).
"""

from __future__ import annotations

from typing import Dict, List

# device key: nvidia | intel-gpu | npu
CATALOG: Dict[str, List[dict]] = {
    "nvidia": [
        {
            "repo": "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
            "name": "Qwen3.6-35B-A3B",
            "params": "35B",
            "active": "3B",
            "size_gb": 21.2,
            "quant": "GGUF Q4_K_M",
            "backend": "llama.cpp (CUDA)",
            "mm": True,
            "layers": "SSD + 内存 + 显存（三层卸载）",
            "ram_hint": "≥ 16 GB 内存；8 GB 显存可常驻热专家 + 热 KV",
            "min_ssd_gb": 22,
            "note": "多模态 MoE · 激活仅 3B · NVIDIA 推荐走 CUDA（比 OpenVINO 快约 20×）",
            "files": ["*Q4_K_M.gguf", "*mmproj*.gguf"],
        },
        {
            "repo": "lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF",
            "name": "Qwen3-Next-80B-A3B",
            "params": "80B",
            "active": "3B",
            "size_gb": 48.5,
            "quant": "GGUF Q4_K_M",
            "backend": "llama.cpp (CUDA)",
            "mm": False,
            "layers": "SSD + 内存 + 显存（三层卸载）",
            "ram_hint": "≥ 16 GB 内存；约 50 GB SSD 承载全量；激活仅 3B，速度可控",
            "min_ssd_gb": 50,
            "note": "超大规模 MoE · 本机显存装不下 → 三层卸载的原点场景",
            "files": ["*Q4_K_M.gguf"],
        },
    ],
    "intel-gpu": [
        {
            "repo": "OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov",
            "name": "Qwen3-30B-A3B",
            "params": "30B",
            "active": "3B",
            "size_gb": 16.3,
            "quant": "INT4",
            "backend": "OpenVINO (GPU)",
            "mm": False,
            "layers": "SSD + 内存 + 共享显存（三层卸载）",
            "ram_hint": "≥ 16 GB 内存；Arc 核显共享系统内存，整机建议 32 GB",
            "min_ssd_gb": 17,
            "note": "MoE · 激活 3B · Intel GPU 走 OpenVINO（比 Vulkan 快约 1.8×）",
        },
        {
            "repo": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
            "name": "Qwen3.6-35B-A3B",
            "params": "35B",
            "active": "3B",
            "size_gb": 19.4,
            "quant": "INT4",
            "backend": "OpenVINO (GPU)",
            "mm": True,
            "layers": "SSD + 内存 + 共享显存（三层卸载）",
            "ram_hint": "≥ 24 GB 内存；多模态 MoE，i 卡共享内存更吃整机 RAM",
            "min_ssd_gb": 20,
            "note": "多模态 MoE · 与 N 卡同款，i 卡走 OpenVINO 共享内存",
        },
    ],
    "npu": [
        {
            "repo": "OpenVINO/Qwen3-8B-int4-cw-ov",
            "name": "Qwen3-8B",
            "params": "8B",
            "active": "8B",
            "size_gb": 4.75,
            "quant": "INT4_SYM",
            "backend": "OpenVINO (NPU)",
            "mm": False,
            "layers": "SSD + 内存（NPU 用系统内存，无独立显存层）",
            "ram_hint": "≥ 12 GB 内存；NPU 常驻约 5 GB",
            "min_ssd_gb": 5,
            "note": "对称量化 · 低功耗常驻 · NPU 实测可用",
        },
        {
            "repo": "OpenVINO/MiniCPM-V-4_5-int4-cw-ov",
            "name": "MiniCPM-V-4.5",
            "params": "9B",
            "active": "9B",
            "size_gb": 5.26,
            "quant": "INT4_SYM",
            "backend": "OpenVINO (NPU)",
            "mm": True,
            "layers": "SSD + 内存（NPU 用系统内存，无独立显存层）",
            "ram_hint": "≥ 16 GB 内存；NPU 实测可跑（首次加载约 50s）",
            "min_ssd_gb": 6,
            "note": "多模态 · 已实测通过 · NPU 实用的最大档（受对称 INT4 限制）",
        },
    ],
}


def all_repos() -> List[dict]:
    """Flatten the catalog into a list with the device key attached."""
    out = []
    for dev, items in CATALOG.items():
        for it in items:
            out.append({**it, "device": dev})
    return out


def find(repo: str) -> dict | None:
    for it in all_repos():
        if it["repo"] == repo:
            return it
    return None
