"""Unit tests for hardware_profile (pure logic, no real probes needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_server.hardware_profile import (  # noqa: E402
    classify_ram, offload_strategy, super_model_mode, build_profile,
)


def test_classify_ram():
    assert classify_ram(16) == "low"
    assert classify_ram(24) == "standard"
    assert classify_ram(32) == "high"
    assert classify_ram(64) == "high"


def test_offload_16_vs_32_nvidia():
    s16 = offload_strategy("NVIDIA-GPU", 16)
    s32 = offload_strategy("NVIDIA-GPU", 32)
    assert s16.ncmoe < s32.ncmoe                     # 16G uses smaller ncmoe
    assert s16.max_recommended_ctx < s32.max_recommended_ctx
    assert "16G" in s16.note and "32G" in s32.note
    assert "n-gram" in s32.tiers["ssd"]              # n-gram stays on SSD both


def test_offload_igpu_and_npu_no_vram():
    ig = offload_strategy("INTEL-GPU", 32)
    npu = offload_strategy("NPU", 16)
    assert "共享" in ig.tiers["vram"]
    assert "无独立显存" in npu.tiers["vram"]
    assert ig.ncmoe == 0 and npu.ncmoe == 0


def test_super_mode_32g_available():
    m = super_model_mode(32, True, 8, 6.0, True)
    assert m.available
    assert m.context >= 262_144
    assert "Qwen3.8" in m.model


def test_super_mode_16g_disabled_with_exploration():
    m = super_model_mode(16, True, 8, 6.0, True)
    assert not m.available
    assert "ram" in m.reason or "内存" in m.reason
    assert m.exploration is not None
    assert m.exploration["feasible"]
    assert m.exploration["recommended"]["ncmoe"] <= 16
    assert m.exploration["recommended"]["ctx"] <= 65536


def test_super_mode_no_nvidia():
    m = super_model_mode(64, False, 0, 6.0, True)
    assert not m.available
    assert "nvidia" in m.reason
    assert m.exploration is None                     # no exploration without N-card


def test_super_mode_slow_ssd():
    m = super_model_mode(64, True, 8, 0.4, True)
    assert not m.available
    assert "ssd" in m.reason


def test_build_profile_payload():
    p = build_profile(16, True, 8, 6.0, device="NVIDIA-GPU")
    assert p["ram"]["tier"] == "low"
    assert p["offload"]["ram_tier"] == "low"
    assert p["super_model"]["available"] is False
    assert p["super_model"]["exploration"]["feasible"] is True
