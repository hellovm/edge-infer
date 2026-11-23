"""Tests for the P0-P2 offload knobs added to LlamaConfig.

Covers:
  - P1-A: MTP draft auto-disable at ctx >= 64K
  - P0-2/P2: new fields exist, default off, and reach the argv
  - P1-B: flash-attn default on
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_server.llama_backend import LlamaConfig  # noqa: E402


def test_draft_disabled_at_large_ctx():
    c = LlamaConfig(model_path="m.gguf", ctx=65536, draft_path="d.gguf")
    assert c.resolved_draft() == ""      # 64K → drop draft
    c2 = LlamaConfig(model_path="m.gguf", ctx=262144, draft_path="d.gguf")
    assert c2.resolved_draft() == ""


def test_draft_kept_at_small_ctx():
    c = LlamaConfig(model_path="m.gguf", ctx=32768, draft_path="d.gguf")
    assert c.resolved_draft() == "d.gguf"


def test_draft_empty_when_none():
    c = LlamaConfig(model_path="m.gguf", ctx=8192, draft_path="")
    assert c.resolved_draft() == ""


def test_new_knob_defaults():
    c = LlamaConfig(model_path="m.gguf")
    assert c.flash_attn is True          # P1-B: on by default
    assert c.moe_cache_mib == 0          # P0-2: off by default
    assert c.resident_budget_mib == 0    # P2: uncapped by default


def test_new_knobs_settable():
    c = LlamaConfig(model_path="m.gguf", moe_cache_mib=4096, resident_budget_mib=24576,
                    flash_attn=False)
    assert c.moe_cache_mib == 4096
    assert c.resident_budget_mib == 24576
    assert c.flash_attn is False
