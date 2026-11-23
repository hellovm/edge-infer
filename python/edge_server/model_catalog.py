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
    # ------------------------------------------------------------------ Qwen3.8-Flash-Next
    {
        "name": "Qwen3.8-Flash-Next",
        "params": "180B",
        "active": "6B",
        "mm": True,
        "note": "旗舰 MoE（180B 存储 / 6B 激活）· 1-bit 极小量化 · 三层卸载首选 · 可配 MTP 提速",
        "variants": {
            "nvidia": {
                "repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
                "subdir": "UD-IQ1_S",
                "quant": "GGUF UD-IQ1_S (1-bit)",
                "backend": "llama.cpp (CUDA)",
                # Weights shard ≈ 37.6 GB (>32 GB RAM → part streams from SSD);
                # the 28.8 GB engram/n-gram shard is mmap'd to SSD, not resident.
                # Smaller than UD-Q2_0 (41.7 GB weights) so more experts fit in
                # RAM. Measured on RTX 4070 8 GB + 32 GB: 1-bit cuts SSD traffic
                # per token ~10% vs Q2_0 — the decisive factor is RAM residency.
                "size_gb": 66.4,
                "weights_gb": 37.6,
                "ram_hint": "≥ 32 GB 内存；8 GB 显存 + 三层卸载（ncmoe≈30）",
                "min_ssd_gb": 70,
                "files": ["*UD-IQ1_S*.gguf"],
                "draft": {
                    "subdir": "MTP",
                    "file": "mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf",
                    "quant": "MTP draft shared-Q4_K_M",
                    "size_gb": 1.8,
                    "note": "投机解码 draft head（无损，实测 +17% decode）；ctx≥64K 自动关闭以省显存",
                },
            },
            # ------------------------------------------------------------------
            # "agention" variant: the 1-bit Gyro-S quant that actually FITS in
            # 32 GB RAM (weights 29.7 GB, no separate engram shard → nothing is
            # read from SSD per token). It needs AgentionAI's llama.cpp fork
            # (APR rotor kernels); stock llama.cpp cannot load it. The GUI/API
            # auto-selects this engine in super-model mode.
            # ------------------------------------------------------------------
            "nvidia-agention": {
                "repo": "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF",
                "quant": "GGUF Gyro-S TQ1_0 (1-bit, APR)",
                "backend": "AgentionAI llama.cpp (CUDA)",
                "engine": "agention",
                "size_gb": 29.7,
                "weights_gb": 29.7,
                "ram_hint": "≥ 32 GB 内存：权重全驻内存，无需每 token 读盘（速度档）",
                "min_ssd_gb": 32,
                # -noPLE = no PLE/engram table → nothing extra on SSD.
                # The draft ships at the repo root (Agention layout).
                "files": ["*Gyro-S-TQ1_0-noPLE.gguf", "*mmproj-F16.gguf",
                          "mtp-Qwen3.8-Flash-Next-draft.gguf"],
                "draft": {
                    "file": "mtp-Qwen3.8-Flash-Next-draft.gguf",
                    "quant": "MTP draft (Qwen3.8-Flash-Next-draft)",
                    "size_gb": 2.7,
                    "note": "AgentionAI 官方 MTP draft；ctx≥64K 自动关闭",
                },
            },
        },
    },
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
    "nvidia-agention": "NVIDIA GPU · Agention 引擎",
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


# Engine ids understood by the llama.cpp launcher.
ENGINE_UPSTREAM = "upstream"    # stock/unsloth llama.cpp (default)
ENGINE_AGENTION = "agention"    # AgentionAI fork — required for TQ1_0/APR GGUFs


def variant_for_engine(model_name: str, engine: str) -> Optional[dict]:
    """Return the catalog variant matching a given engine (NVIDIA path only).

    Super-model mode uses ``ENGINE_AGENTION`` to auto-pick the 1-bit Gyro-S
    quant, which fits fully in 32 GB RAM. Falls back to the plain ``nvidia``
    variant when no engine-specific entry exists.
    """
    for m in MODELS:
        if m["name"] != model_name:
            continue
        for dev, v in m["variants"].items():
            if v.get("engine") == engine:
                return {**v, "device": dev}
        return {**m["variants"].get("nvidia", {}), "device": "nvidia"} or None
    return None

