"""Hardware profile: memory-tier classification, offload strategy and
"super model" mode gating.

This is the decision layer between raw probes (diagnostics.py / edge-detect)
and the GUI. Pure functions where possible so it stays unit-testable.

Why memory capacity changes the strategy
----------------------------------------
All three-tier offloading is a race between "how much hot data fits in the
fast tiers" and "how much must be read from SSD per token". The RAM capacity
is the single biggest determinant:

  16 GB  -> after OS (≈4 GB) + DVMT/reserve, ~10-11 GB usable for weights+KV.
            A 66 GB Q2_0 model must keep most experts on SSD. Small ncmoe.
  32 GB  -> ~24-26 GB usable. A much larger hot expert set fits in RAM, so
            SSD reads per token drop sharply and long context becomes viable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


# Thresholds in GiB. Note: a "32 GB" machine reports ~31.5 GiB in the OS
# (binary GiB vs decimal GB, plus firmware reserve), so the effective
# threshold is slightly below 32 to honour the *marketed* capacity.
RAM_LOW_GB = 16.0
RAM_HIGH_GB = 30.0
# Super-model mode requirements (effective threshold, see note above).
SUPER_MODEL_MIN_RAM_GB = 30.0
SUPER_MODEL_MIN_VRAM_GB = 6.0
SUPER_MODEL_MIN_SSD_GB_S = 1.5       # 1.5 GB/s sequential read
SUPER_MODEL_TARGET_CTX = 262144      # 256K (model's native context)
DEFAULT_CTX = 65536                  # 64K for all non-super modes


class RamTier:
    LOW = "low"          # 16 GB class
    STANDARD = "standard"  # 24-31 GB
    HIGH = "high"        # >= 32 GB


@dataclass
class OffloadStrategy:
    """How the three tiers are used for a given device + RAM class."""
    device: str
    ram_tier: str
    tiers: Dict[str, str]
    ncmoe: int
    kv_quant: str
    max_recommended_ctx: int
    note: str


@dataclass
class SuperModelMode:
    available: bool
    reason: str
    model: str
    context: int
    requirements: Dict[str, Any]
    exploration: Optional[Dict[str, Any]] = None


def classify_ram(total_gb: float) -> str:
    if total_gb >= RAM_HIGH_GB:
        return RamTier.HIGH
    # 16 GB machines (and anything below ~24) are the constrained class:
    # real usable RAM after OS is ~12-13 GB, far less than the 32 GB path.
    if total_gb >= 24.0:
        return RamTier.STANDARD
    return RamTier.LOW


def _usable_ram_gb(total_gb: float) -> float:
    """Rough usable RAM after OS + driver overhead (matches edge-detect)."""
    overhead = 3.5 if total_gb >= RAM_HIGH_GB else 3.0
    return max(0.0, total_gb - overhead)


def offload_strategy(device: str, total_ram_gb: float) -> OffloadStrategy:
    """Pick the offload knobs for a device class + RAM capacity.

    ``device`` is one of NVIDIA-GPU / INTEL-GPU / NPU / CPU (case-insensitive),
    matching the GUI's device keys.
    """
    dev = (device or "").upper().replace(" ", "").replace("_", "")
    tier = classify_ram(total_ram_gb)
    usable = _usable_ram_gb(total_ram_gb)

    if dev in ("NVIDIA-GPU", "NVIDIA", "CUDA", "GPU.1"):
        if tier == RamTier.HIGH:
            return OffloadStrategy(
                device="NVIDIA-GPU", ram_tier=tier,
                tiers={
                    "vram": "注意力 + norm + 输出 + 热专家（ncmoe 之外）",
                    "ram": "主干 + 次热专家（大缓存池，命中率高）",
                    "ssd": "冷专家 + 28.8GB n-gram 表（lazy 按需读）",
                },
                ncmoe=35, kv_quant="q8_0", max_recommended_ctx=262144,
                note=f"32G+ 内存：{usable:.0f}GB 可用于热专家缓存，支持 256K 上下文与超级模型模式",
            )
        if tier == RamTier.STANDARD:
            return OffloadStrategy(
                device="NVIDIA-GPU", ram_tier=tier,
                tiers={
                    "vram": "注意力 + 热专家（显存优先）",
                    "ram": "主干 + 少量次热专家（缓存池紧张）",
                    "ssd": "大部分专家 + n-gram 表（读盘频繁）",
                },
                ncmoe=20, kv_quant="q8_0", max_recommended_ctx=65536,
                note=f"16G 内存：{usable:.0f}GB 可用，需减小 ncmoe 防换页；上下文建议 ≤128K",
            )
        # 16 GB class: the Qwen3.8-Flash-Next "exploration" tier. It *can* run
        # the 66 GB Q2_0 model on SSD, but only a small hot-expert set fits in
        # RAM, so ncmoe must be conservative and context capped.
        return OffloadStrategy(
            device="NVIDIA-GPU", ram_tier=tier,
            tiers={
                "vram": "注意力 + 少量热专家",
                "ram": "主干 + 极小热专家缓存（~6-8GB 余量）",
                "ssd": "绝大多数专家 + 28.8GB n-gram 表（读盘频繁）",
            },
            ncmoe=12, kv_quant="q4_0", max_recommended_ctx=65536,
            note=(f"{total_ram_gb:.0f}G 内存可跑 Qwen3.8-Flash-Next 但需降级："
                  f"{usable:.0f}GB 可用，ncmoe=12、上下文≤64K；"
                  "256K 与满速不可期，详见超级模型模式的探索评估"),
        )

    if dev in ("INTEL-GPU", "IGPU", "INTELGPU", "GPU.0"):
        # iGPU shares system RAM - no separate VRAM tier in practice.
        return OffloadStrategy(
            device="Intel-GPU", ram_tier=tier,
            tiers={
                "vram": "共享内存（DVMT 划分，实际走 RAM）",
                "ram": "主干 + 热专家（iGPU 直接访问）",
                "ssd": "冷专家 + n-gram 表",
            },
            ncmoe=0, kv_quant="q8_0" if tier == RamTier.HIGH else "q4_0",
            max_recommended_ctx=65536,
            note=("iGPU 无独立显存：权重与 KV 都走系统内存，SSD 承担冷专家。"
                  + (" 32G 内存可跑 30B/35B MoE。" if tier == RamTier.HIGH
                     else " 16G 内存建议 ≤30B-A3B 并限制上下文。")),
        )

    if dev == "NPU":
        # NPU has no VRAM; its budget is carved from system RAM.
        return OffloadStrategy(
            device="Intel-NPU", ram_tier=tier,
            tiers={
                "vram": "无独立显存（NPU 预算从 RAM 划分）",
                "ram": "NPU 常驻小模型（≤9B int4 对称量化）",
                "ssd": "超大模型不适用 NPU；需切 NVIDIA 路径",
            },
            ncmoe=0, kv_quant="int4",
            max_recommended_ctx=32768,
            note=("NPU 走 SSD+RAM 两层分片：仅小模型（NPU 常驻 ~5GB），"
                  + ("32G 内存可给更大 KV 缓存。" if tier == RamTier.HIGH
                     else "16G 内存需限制并发与上下文。")),
        )

    # CPU fallback
    return OffloadStrategy(
        device="CPU", ram_tier=tier,
        tiers={"vram": "无", "ram": "热专家 + KV", "ssd": "冷专家 + 全量权重"},
        ncmoe=0, kv_quant="q4_0", max_recommended_ctx=32768,
        note="CPU 走 SSD+RAM 两层；速度受限，仅作回退",
    )


def super_model_mode(
    total_ram_gb: float,
    has_nvidia: bool,
    nvidia_vram_gb: float,
    ssd_read_gb_s: float,
    model_available: bool = True,
) -> SuperModelMode:
    """Decide whether "super model mode" (Qwen3.8-Flash-Next @ 256K) is on.

    Requirements: NVIDIA dGPU + SSD + RAM >= 32 GB (+ enough VRAM/SSD speed).
    When RAM is only 16 GB we still report an *exploration* assessment instead
    of simply disabling, so the GUI can explain why and what to expect.
    """
    req = {
        "ram_gb": {"required": SUPER_MODEL_MIN_RAM_GB, "actual": round(total_ram_gb, 1),
                   "ok": total_ram_gb >= SUPER_MODEL_MIN_RAM_GB},
        "nvidia": {"required": True, "actual": bool(has_nvidia), "ok": bool(has_nvidia)},
        "vram_gb": {"required": SUPER_MODEL_MIN_VRAM_GB, "actual": round(nvidia_vram_gb, 1),
                    "ok": has_nvidia and nvidia_vram_gb >= SUPER_MODEL_MIN_VRAM_GB},
        "ssd_gb_per_s": {"required": SUPER_MODEL_MIN_SSD_GB_S,
                         "actual": round(ssd_read_gb_s, 1),
                         "ok": ssd_read_gb_s >= SUPER_MODEL_MIN_SSD_GB_S},
        "model": {"required": True, "actual": bool(model_available), "ok": bool(model_available)},
    }
    failed = [k for k, v in req.items() if not v["ok"]]
    base = SuperModelMode(
        available=not failed,
        reason=("满足全部条件：N 卡独显 + SSD + ≥32G 内存" if not failed
                else "未满足：" + "、".join(failed)),
        model="Qwen3.8-Flash-Next (UD-Q2_0 + MTP)",
        context=SUPER_MODEL_TARGET_CTX if not failed else 0,
        requirements=req,
    )

    # Exploration: NVIDIA + SSD + 16 GB. Can it still run the model?
    if failed and has_nvidia and ssd_read_gb_s >= SUPER_MODEL_MIN_SSD_GB_S \
            and RAM_LOW_GB <= total_ram_gb < RAM_HIGH_GB:
        usable = _usable_ram_gb(total_ram_gb)
        # Q2_0: ~70 GB weights, 28.8 GB n-gram stays on SSD. With ~13 GB RAM
        # you can hold only a small hot-expert set; still runs, but ctx and
        # speed drop. Estimate from measured 1.1-1.4 tok/s at 32 GB.
        base.exploration = {
            "feasible": True,
            "headline": f"{total_ram_gb:.0f}G + N卡 + SSD 也能跑 Qwen3.8-Flash-Next，但需降级配置",
            "usable_ram_gb": round(usable, 1),
            "recommended": {
                "ncmoe": 12,
                "ctx": 65536,
                "kv_quant": "q4_0",
                "mtp": True,
            },
            "expected": {
                "decode_tok_s": [0.4, 0.8],
                "prefill": "显著变慢（SSD 读 n-gram + 冷专家）",
                "context": "建议 ≤64K；256K 不现实（KV + 热集超内存）",
            },
            "reason": (f"{usable:.0f}GB 可用内存只能缓存少量热专家，"
                       "每 token 大量专家需从 SSD 读；能跑但不适合长上下文/高吞吐"),
        }
    return base


def build_profile(total_ram_gb: float, has_nvidia: bool, nvidia_vram_gb: float,
                  ssd_read_gb_s: float, device: str = "NVIDIA-GPU") -> Dict[str, Any]:
    """Combined profile payload for /edge/devices."""
    tier = classify_ram(total_ram_gb)
    return {
        "ram": {
            "total_gb": round(total_ram_gb, 1),
            "usable_gb": round(_usable_ram_gb(total_ram_gb), 1),
            "tier": tier,
            "tier_label": {"low": "16G 级", "standard": "24-31G 级", "high": "32G+ 级"}[tier],
        },
        "offload": asdict(offload_strategy(device, total_ram_gb)),
        "super_model": asdict(super_model_mode(total_ram_gb, has_nvidia,
                                                nvidia_vram_gb, ssd_read_gb_s)),
    }
