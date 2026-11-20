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

Backends
--------
- ``openvino``  — OpenVINO IR (``*-ov``). Works today; NPU requires symmetric
  INT4 (``*-int4-cw-ov`` / ``nf4``), asymmetric INT4 (AWQ) fails to compile on
  Meteor Lake (NPU3720).
- ``llama.cpp`` — GGUF. Required for MoE beyond ~35B (OpenVINO stops at 35B-A3B).
  llama.cpp's ``--n-cpu-moe`` is the natural 3-tier offload path.

Field notes
-----------
- ``layers``        : tiers the checkpoint can be split across.
- ``active``        : active params per token (drives decode-speed floor).
- ``min_ssd_gb``    : disk needed to hold the full checkpoint for offload.
"""

from __future__ import annotations

from typing import Dict, List

# device key: nvidia | intel-gpu | npu
CATALOG: Dict[str, List[dict]] = {
    "nvidia": [
        {
            "repo": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
            "name": "Qwen3.6-35B-A3B（小·N卡）",
            "scale": "small",
            "params": "35B",
            "active": "3B",
            "size_gb": 19.4,
            "quant": "INT4",
            "backend": "openvino",
            "mm": True,
            "layers": "SSD + 内存 + 显存（三层卸载）",
            "ram_hint": "≥ 16 GB 可用内存；8 GB 显存可常驻热专家 + 热 KV",
            "min_ssd_gb": 20,
            "note": "多模态 MoE · 激活仅 3B · 三层卸载首选",
        },
        {
            "repo": "lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF",
            "name": "Qwen3-Next-80B-A3B（大·N卡）",
            "scale": "large",
            "params": "80B",
            "active": "3B",
            "size_gb": 48.5,
            "quant": "GGUF Q4_K_M",
            "backend": "llama.cpp",
            "mm": False,
            "layers": "SSD + 内存 + 显存（三层卸载）",
            "ram_hint": "≥ 16 GB 可用内存；约 50 GB SSD 全量卸载；激活仅 3B，速度可控",
            "min_ssd_gb": 50,
            "note": "超大规模 MoE · 本机跑不动 → 三层卸载的原点场景",
            "files": ["*Q4_K_M.gguf", "README.md"],
        },
    ],
    "intel-gpu": [
        {
            "repo": "OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov",
            "name": "Qwen3-30B-A3B（小·i卡）",
            "scale": "small",
            "params": "30B",
            "active": "3B",
            "size_gb": 16.3,
            "quant": "INT4",
            "backend": "openvino",
            "mm": False,
            "layers": "SSD + 内存 + 共享显存（三层卸载）",
            "ram_hint": "≥ 16 GB 可用内存；i 卡共享系统内存，整机建议 32 GB",
            "min_ssd_gb": 17,
            "note": "MoE · 激活 3B · Arc 核显可卸载",
        },
        {
            "repo": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
            "name": "Qwen3.6-35B-A3B（大·i卡）",
            "scale": "large",
            "params": "35B",
            "active": "3B",
            "size_gb": 19.4,
            "quant": "INT4",
            "backend": "openvino",
            "mm": True,
            "layers": "SSD + 内存 + 共享显存（三层卸载）",
            "ram_hint": "≥ 24 GB 可用内存；多模态 MoE，i 卡共享内存更吃整机 RAM",
            "min_ssd_gb": 20,
            "note": "多模态 MoE · 与 N 卡同款，i 卡走共享内存",
        },
    ],
    "npu": [
        {
            "repo": "OpenVINO/Qwen3-8B-int4-cw-ov",
            "name": "Qwen3-8B（小·NPU）",
            "scale": "small",
            "params": "8B",
            "active": "8B",
            "size_gb": 4.75,
            "quant": "INT4_SYM",
            "backend": "openvino",
            "mm": False,
            "layers": "SSD + 内存（NPU 用系统内存，无独立显存层）",
            "ram_hint": "≥ 12 GB 可用内存；NPU 常驻约 5 GB",
            "min_ssd_gb": 5,
            "note": "对称量化 · NPU 实测可用 · 加载快",
        },
        {
            "repo": "OpenVINO/Qwen3-30B-A3B-nf4-ov",
            "name": "Qwen3-30B-A3B（大·NPU）",
            "scale": "large",
            "params": "30B",
            "active": "3B",
            "size_gb": 15.7,
            "quant": "NF4（对称）",
            "backend": "openvino",
            "mm": False,
            "layers": "SSD + 内存（NPU 用系统内存，无独立显存层）",
            "ram_hint": "≥ 24 GB 可用内存；MoE 激活仅 3B，SSD+内存两层分片",
            "min_ssd_gb": 16,
            "note": "MoE · 对称 NF4 · NPU 上限实测中（30B 级）",
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
