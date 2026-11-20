"""Curated model catalog — recommended multimodal (VLM) models per accelerator.

All repos verified reachable on ModelScope (2026-11-20). Each entry lists the
target device, parameter scale (small/large), approximate download size, and
the RAM/VRAM guidance shown next to the model in the GUI.

Key facts behind the recommendations:
  - NPU (Meteor Lake / NPU3720): only **symmetric INT4** works
    (`*-int4-cw-ov` exports). Asymmetric INT4 (AWQ) fails to compile.
  - NVIDIA GPU / Intel GPU: any INT4 export works (`*-int4-ov`).
  - Multimodal = vision-language (image understanding), which is what these
    Qwen3-VL / MiniCPM-V / Qwen2.5-VL repos provide.
"""

from __future__ import annotations

from typing import Dict, List

# device key: nvidia | intel-gpu | npu
CATALOG: Dict[str, List[dict]] = {
    "nvidia": [
        {
            "repo": "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov",
            "name": "Qwen3-VL-4B（小·N卡）",
            "scale": "small",
            "params": "4B",
            "size_gb": 3.1,
            "quant": "INT4",
            "mm": True,
            "ram_hint": "≥ 8 GB 可用内存；6–8 GB 显存可全量常驻",
            "note": "多模态 · 32 语言 OCR · 推荐 8 GB+ 显存",
        },
        {
            "repo": "OpenVINO/Qwen3-VL-8B-Instruct-int4-ov",
            "name": "Qwen3-VL-8B（大·N卡）",
            "scale": "large",
            "params": "8B",
            "size_gb": 5.5,
            "quant": "INT4",
            "mm": True,
            "ram_hint": "≥ 16 GB 可用内存；8 GB 显存走三层卸载",
            "note": "多模态 · 更强视觉推理 · 推荐 16 GB+ 内存",
        },
    ],
    "intel-gpu": [
        {
            "repo": "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov",
            "name": "Qwen2.5-VL-7B（小·i卡）",
            "scale": "small",
            "params": "7B",
            "size_gb": 5.2,
            "quant": "INT4",
            "mm": True,
            "ram_hint": "≥ 12 GB 可用内存；i 卡共享内存，建议 16 GB 整机",
            "note": "多模态 · 文档/图表理解 · Arc 核显可跑",
        },
        {
            "repo": "OpenVINO/MiniCPM-V-4_5-int4-cw-ov",
            "name": "MiniCPM-V-4.5（大·i卡）",
            "scale": "large",
            "params": "9B",
            "size_gb": 5.3,
            "quant": "INT4_SYM",
            "mm": True,
            "ram_hint": "≥ 16 GB 可用内存；对称量化，i 卡/NPU 通用",
            "note": "多模态 · 单图/多图/视频 · 同级 SOTA",
        },
    ],
    "npu": [
        {
            "repo": "OpenVINO/Gemma-3-4B-it-int4-cw-ov",
            "name": "Gemma-3-4B（小·NPU）",
            "scale": "small",
            "params": "4B",
            "size_gb": 3.4,
            "quant": "INT4_SYM",
            "mm": True,
            "ram_hint": "≥ 8 GB 可用内存；NPU 常驻约 3–4 GB",
            "note": "多模态 · 对称量化 · 加载快",
        },
        {
            "repo": "OpenVINO/MiniCPM-V-4_5-int4-cw-ov",
            "name": "MiniCPM-V-4.5（大·NPU）",
            "scale": "large",
            "params": "9B",
            "size_gb": 4.9,
            "quant": "INT4_SYM",
            "mm": True,
            "ram_hint": "≥ 16 GB 可用内存；NPU 实测可跑（加载约 50s）",
            "note": "多模态 · 已实测通过 · 端侧多模态 SOTA",
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
