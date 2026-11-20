"""OpenVINO runtime for the 4B small-model tier.

Loads an OpenVINO IR / ONNX model (Qwen3-VL 4B INT4 class) and runs chat
completion. Device is auto-detected: NPU preferred, fall back to CPU when the
NPU plugin is absent or the model does not fit the NPU memory budget.

This is the "small model" path from the requirements doc: NPU only serves a
dense 3-4B multimodal model; anything bigger is routed to the N-card three-tier
path (Rust edge-engine).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import openvino as ov

    _OV_AVAILABLE = True
except Exception:  # openvino wheel not installed yet
    ov = None
    _OV_AVAILABLE = False

log = logging.getLogger("edge.openvino")


class DeviceNotAvailable(Exception):
    pass


@dataclass
class OVModel:
    """A loaded OpenVINO model with a device tag."""

    name: str
    request: object  # ov.CompiledModel
    device: str
    model_id: str = "qwen3-vl-4b-int4"

    def available(self) -> bool:
        return self.request is not None


class OVRuntime:
    """Loads and runs OpenVINO models on NPU or CPU."""

    def __init__(self) -> None:
        self.core = ov.Core() if _OV_AVAILABLE else None
        self._models: dict[str, OVModel] = {}

    @property
    def backend_available(self) -> bool:
        """True when the OpenVINO wheel is present."""
        return _OV_AVAILABLE

    # -- device discovery ----------------------------------------------------
    def available_devices(self) -> List[str]:
        """All device types the OpenVINO runtime can target.

        On an NPU-capable laptop this typically includes "NPU"; on a plain
        Windows box it will be ["CPU"] (or + "GPU" if an NVIDIA/Intel driver
        is present that OpenVINO's GPU plugin recognizes).
        """
        if not self.backend_available:
            return []
        return self.core.get_available_devices()

    def device_info(self, device: str) -> dict:
        info = self.core.get_property(device, "AVAILABLE_DEVICES")
        return {
            "device": device,
            "supported": list(info or []) if info else [],
        }

    # -- model loading -------------------------------------------------------
    def load(
        self,
        ir_path: str,
        device: str = "NPU",
        name: Optional[str] = None,
        memory_budget_gb: float = 4.0,
    ) -> OVModel:
        """Load an IR model onto `device`.

        Falls back to CPU when the NPU is unavailable or the model's memory
        footprint exceeds `memory_budget_gb` (NPU has a hard cap).
        """
        name = name or ir_path.rsplit("/", 1)[-1].replace(".xml", "")
        ox = self.core.read_model(ir_path)

        # Pick the real device.
        dev = device if device in self.available_devices() else "CPU"
        if dev == "NPU":
            # Heuristic: refuse if the model's weight blob is over the NPU cap.
            import os

            blob = os.path.getsize(ir_path.replace(".xml", ".bin"))
            if blob > int(memory_budget_gb * (1024 ** 3)):
                log.warning(
                    "model blob %.2fGB exceeds NPU budget %.1fGB; falling back to CPU",
                    blob / 1e9, memory_budget_gb,
                )
                dev = "CPU"

        compiled = self.core.compile_model(ox, dev)
        self._models[name] = OVModel(
            name=name,
            request=compiled,
            device=dev,
        )
        log.info("loaded model '%s' on %s", name, dev)
        return self._models[name]

    def devices_loaded_on(self) -> List[str]:
        return [m.device for m in self._models.values()]
