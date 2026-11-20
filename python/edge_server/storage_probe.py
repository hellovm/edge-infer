"""Windows storage probe — detect the real SSD model and estimate read bandwidth.

PCIe 5.0 NVMe (~12-14 GB/s), PCIe 4.0 NVMe (~7 GB/s), SATA SSD (~0.55 GB/s).
We query the physical disk model via PowerShell/CIM and map known families to a
bandwidth class. If detection fails we fall back to a conservative NVMe value.

This feeds the Rust three-tier planner so the "SSD-bound" tok/s estimate uses
the machine's actual drive speed (a PCIe 5.0 drive can 2x a PCIe 4.0 one).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

# known model substrings -> (read GB/s, pcie5?)
_FAMILY_HINTS = [
    # PCIe 5.0 consumer flagships
    ("990 pro", 7.0, False), ("990 evo", 7.0, False),
    ("9100 pro", 14.0, True), ("t705", 13.6, True), ("t700", 12.0, True),
    ("crucial t710", 14.5, True), ("t710", 14.5, True),
    ("sn8100", 14.0, True), ("sandisk", 14.0, True),
    ("mp700", 12.0, True), ("mp600 pro", 7.0, False),
    ("gm7000", 7.4, False), ("nv7000", 7.0, False),
    # PCIe 4.0 mainstream
    ("sn850", 7.0, False), ("sn770", 5.2, False), ("sn580", 4.1, False),
    ("980 pro", 7.0, False), ("970 evo", 3.5, False),
    ("crucial p3", 3.5, False), ("crucial p5", 6.6, False),
    ("kc3000", 7.0, False), ("fury", 7.3, False),
    ("kingston", 5.0, False), ("lexar", 7.0, False),
    ("xfusion", 7.0, False), ("致态", 7.0, False), ("tiplus", 7.0, False),
]

_DEFAULT_NVME_GB_S = 6.0  # conservative PCIe 4.0-class fallback


@dataclass
class SsdInfo:
    name: str
    read_gb_per_s: float
    pcie5: bool

    @property
    def read_bytes_per_s(self) -> int:
        return int(self.read_gb_per_s * 1_000_000_000)


def benchmark_read(path: str, threads: int = 4, cap_bytes: int = 2_000_000_000) -> float:
    """Measure multi-threaded sequential read bandwidth (GB/s) over a real file.

    Reads up to `cap_bytes` split across `threads` workers. This is a practical
    number: it reflects how fast the planner can stream expert weights in.
    Returns 0.0 on failure.
    """
    import threading
    import time

    try:
        size = min(os.path.getsize(path), cap_bytes)
    except OSError:
        return 0.0
    if size <= 0:
        return 0.0

    seg = size // threads
    results = [0] * threads

    def worker(idx: int, off: int, n: int) -> None:
        got = 0
        try:
            with open(path, "rb") as f:
                f.seek(off)
                while got < n:
                    b = f.read(min(16 * 1024 * 1024, n - got))
                    if not b:
                        break
                    got += len(b)
        except OSError:
            pass
        results[idx] = got

    ths = [threading.Thread(target=worker, args=(i, i * seg, seg)) for i in range(threads)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = max(time.time() - t0, 1e-3)
    total = sum(results)
    return (total / dt) / 1e9 if total else 0.0


def _run_ps(script: str, timeout: int = 8) -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout or ""
    except Exception:  # noqa: BLE001
        return ""


def detect_ssd(bench_file: Optional[str] = None) -> SsdInfo:
    """Return the best-guess system SSD model + bandwidth class.

    If `bench_file` points to a large local file, we *measure* read bandwidth
    (multi-threaded) and prefer that over the model-name heuristic -- real
    numbers beat marketing names, especially for white-label drives.
    """
    # CIM gives PhysicalDisk.Model; MediaType 4 == SSD.
    script = (
        "Get-PhysicalDisk | Select-Object FriendlyName,MediaType,Size | "
        "ConvertTo-Json -Compress"
    )
    raw = _run_ps(script).strip()
    name = ""
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            ssds = [d for d in data if str(d.get("MediaType", "")).lower() in ("4", "ssd")]
            pick = (ssds or data)[0]
            name = str(pick.get("FriendlyName", "")).strip()
        except Exception:  # noqa: BLE001
            name = ""

    low = name.lower()
    gb, pcie5 = _DEFAULT_NVME_GB_S, False
    for hint, speed, is5 in _FAMILY_HINTS:
        if hint in low:
            gb, pcie5 = speed, is5
            break

    # Prefer a real measured number when we have a big file to stream.
    if bench_file and os.path.isfile(bench_file):
        measured = benchmark_read(bench_file)
        if measured > 0.3:
            # OS cache inflates reads; discount slightly for cold-cache realism.
            gb = round(measured * 0.85, 2)
            pcie5 = gb >= 10.0

    return SsdInfo(name=name or "NVMe SSD (未探测到型号)", read_gb_per_s=gb, pcie5=pcie5)


if __name__ == "__main__":
    info = detect_ssd()
    print(f"SSD: {info.name}")
    print(f"read: {info.read_gb_per_s} GB/s  PCIe5={info.pcie5}")
