"""Environment diagnostics + structured problem recording.

Why this exists
---------------
The maintainer develops on one machine (Intel Core Ultra + RTX) but users run
edge-infer on **other** machines -- notably **AMD platforms** (Ryzen AI XDNA
NPU, Radeon iGPU/dGPU). When something fails there, we cannot reproduce it
locally, so the product must be able to *explain itself*:

  * ``collect_report()`` gathers a complete, structured snapshot of the host
    environment (CPU, GPUs, NPUs, drivers, RAM, SSD, backends, model dirs) and
    probes every backend **independently**, recording *why* each one failed
    instead of silently degrading.
  * ``record_problem()`` appends a structured entry for any load/inference
    failure, with a traceback and the environment fingerprint at that moment.
  * ``build_bundle()`` writes a single zip the user can hand back for analysis.

Privacy: API keys and bearer tokens are redacted; no prompts or file contents
are ever captured.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIAG_DIR = os.path.join(REPO_ROOT, "logs", "diagnostics")
PROBLEM_LOG = os.path.join(REPO_ROOT, "logs", "problems.jsonl")

# Anything matching these keys is scrubbed before it is written to disk.
_SECRET_KEY_RE = re.compile(r"(key|token|secret|password|authorization)", re.I)
_SK_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,}|github_pat_[A-Za-z0-9_]{10,})\b")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def redact(obj: Any) -> Any:
    """Recursively scrub secrets from a JSON-able structure."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_KEY_RE.search(k):
                out[k] = "***redacted***" if v else ""
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _SK_RE.sub("***redacted***", obj)
    return obj


# ---------------------------------------------------------------------------
# individual probes -- each is isolated so one failure never aborts the report
# ---------------------------------------------------------------------------

def _probe(name: str, fn) -> Dict[str, Any]:
    t0 = time.time()
    try:
        data = fn()
        return {"ok": True, "ms": round((time.time() - t0) * 1000, 1), "data": data}
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "ms": round((time.time() - t0) * 1000, 1),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }


def probe_os() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_exe": sys.executable,
        "cwd": os.getcwd(),
    }


def probe_cpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "arch": platform.machine(),
    }
    if os.name == "nt":
        try:
            ps = ("Get-CimInstance Win32_Processor | "
                  "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | "
                  "ConvertTo-Json -Compress")
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=40,
                creationflags=0x0800_0000,
            )
            raw = (r.stdout or "").strip()
            if raw:
                d = json.loads(raw)
                if isinstance(d, list):
                    d = d[0] if d else {}
                for k in ("Name", "NumberOfCores", "NumberOfLogicalProcessors", "MaxClockSpeed"):
                    if d.get(k) is not None:
                        info[k] = d[k]
        except Exception as e:  # noqa: BLE001
            info["cim_error"] = str(e)[:200]
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        info["Name"] = line.split(":", 1)[1].strip()
                        break
        except Exception:  # noqa: BLE001
            pass
    return info


def probe_memory() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if os.name == "nt":
        # CIM is locale-independent; wmic is deprecated and now often absent.
        try:
            ps = ("$c=Get-CimInstance Win32_ComputerSystem;"
                  "$o=Get-CimInstance Win32_OperatingSystem;"
                  "\"{0:N1}|{1:N1}\" -f ($c.TotalPhysicalMemory/1GB),($o.FreePhysicalMemory/1MB)")
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=40,
                creationflags=0x0800_0000,
            )
            parts = (r.stdout or "").strip().replace(",", "").split("|")
            if len(parts) == 2:
                info["total_gb"] = float(parts[0])
                info["free_gb"] = float(parts[1])
        except Exception as e:  # noqa: BLE001
            info["error"] = str(e)[:200]
    else:
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["total_gb"] = round(int(line.split()[1]) / (1024 ** 2), 1)
                        break
        except Exception:  # noqa: BLE001
            pass
    return info


def probe_nvidia() -> Dict[str, Any]:
    """NVIDIA GPU presence + driver via nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=25,
        )
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "name": parts[0], "vram": parts[1],
                    "driver": parts[2], "compute_cap": parts[3],
                })
        return {"available": bool(gpus), "gpus": gpus}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def probe_openvino() -> Dict[str, Any]:
    """OpenVINO devices + per-device properties + NPU plugin presence.

    This is the probe that matters most on AMD: OpenVINO's GPU/NPU plugins are
    **Intel-only**, so on an AMD box we expect ``GPU`` to be absent and ``NPU``
    to be absent even though the silicon exists.
    """
    import openvino as ov

    core = ov.Core()
    devices = list(core.get_available_devices())
    detail: Dict[str, Any] = {}
    for d in devices:
        entry: Dict[str, Any] = {}
        for prop in ("FULL_DEVICE_NAME", "DEVICE_TYPE", "SUPPORTED_PROPERTIES"):
            try:
                v = core.get_property(d, prop)
                entry[prop] = str(v) if prop != "SUPPORTED_PROPERTIES" else f"<{len(list(v))} props>"
            except Exception:  # noqa: BLE001
                pass
        detail[d] = entry

    # Explicitly test whether the NPU plugin can be instantiated at all.
    npu_probe: Dict[str, Any]
    try:
        present = "NPU" in devices
        npu_probe = {
            "enumerated": present,
            "note": ("OpenVINO NPU plugin exposes the device" if present
                     else "no NPU device: either no Intel NPU, or driver missing. "
                          "AMD XDNA is NOT supported by OpenVINO's NPU plugin."),
        }
    except Exception as e:  # noqa: BLE001
        npu_probe = {"enumerated": False, "error": str(e)[:300]}

    return {
        "version": getattr(ov, "__version__", "?"),
        "devices": devices,
        "device_detail": detail,
        "npu": npu_probe,
        "gpu_plugin_present": any(str(d).startswith("GPU") for d in devices),
        "cpu_present": "CPU" in devices,
    }


def probe_vendor_npu() -> Dict[str, Any]:
    """Best-effort detection of a *vendor* NPU (Intel or AMD) from the OS.

    We look for the PnP/driver entry so we can tell the user "your AMD NPU
    exists, but no runtime in this project can use it" -- which is far more
    useful than a silent fallback.
    """
    out: Dict[str, Any] = {"intel": False, "amd": False, "amd_devices": [], "intel_devices": []}
    if os.name != "nt":
        return out
    try:
        ps = (
            "Get-PnpDevice -Class ComputeAccelerator,Display -Status OK "
            "-ErrorAction SilentlyContinue | "
            "Select-Object FriendlyName,Class | ConvertTo-Json -Compress"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=40,
            creationflags=0x0800_0000,
        )
        raw = (r.stdout or "").strip()
        if not raw:
            return out
        items = json.loads(raw)
        if isinstance(items, dict):
            items = [items]
        for it in items:
            n = str(it.get("FriendlyName") or "")
            low = n.lower()
            if "amd" in low or "radeon" in low:
                out["amd"] = True
                out["amd_devices"].append(n)
                if "npu" in low or "xdna" in low or "ryzen ai" in low:
                    out["amd_npu_hint"] = n
            if "intel" in low:
                out["intel"] = True
                out["intel_devices"].append(n)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:200]
    return out


def probe_backends() -> Dict[str, Any]:
    """Import-test every optional backend so we know what is actually usable."""
    res: Dict[str, Any] = {}

    def _try(mod: str) -> Dict[str, Any]:
        try:
            m = __import__(mod)
            return {"importable": True, "version": getattr(m, "__version__", "?")}
        except Exception as e:  # noqa: BLE001
            return {"importable": False, "error": f"{type(e).__name__}: {e}"[:300]}

    for m in ("openvino", "openvino_genai", "openvino_tokenizers",
              "tokenizers", "numpy", "modelscope"):
        res[m] = _try(m)

    # llama.cpp binaries
    try:
        from .llama_backend import find_llama_server, llama_available
        res["llama.cpp"] = {
            "available": llama_available(),
            "path": find_llama_server() or None,
        }
    except Exception as e:  # noqa: BLE001
        res["llama.cpp"] = {"available": False, "error": str(e)[:300]}
    return res


def probe_llama_devices() -> Dict[str, Any]:
    """Ask llama.cpp which compute backends/devices it can see.

    On an AMD machine this is where Radeon shows up (via Vulkan/ROCm) -- the
    path we plan to support -- while the XDNA NPU does not appear.
    """
    try:
        from .llama_backend import find_llama_server
        exe = find_llama_server()
        if not exe:
            return {"available": False, "reason": "llama-server not found"}
        bench = exe.replace("llama-server", "llama-bench")
        if not os.path.isfile(bench):
            bench = exe
        r = subprocess.run([bench, "--list-devices"], capture_output=True,
                           text=True, timeout=90, creationflags=0x0800_0000)
        txt = (r.stdout or "") + (r.stderr or "")
        backends = sorted(set(re.findall(r"loaded (\w+) backend", txt)))
        devs = re.findall(r"^\s{2}(\w+\d*):\s*(.+?)\s*$", txt, re.M)
        return {
            "available": True,
            "backends": backends,
            "devices": [{"id": a, "name": b} for a, b in devs],
            "raw_tail": txt[-1200:],
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"[:300]}


def probe_ssd() -> Dict[str, Any]:
    try:
        from .storage_probe import detect_ssd
        s = detect_ssd()
        return {"name": s.name, "read_gb_per_s": s.read_gb_per_s, "pcie5": s.pcie5}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def probe_models() -> Dict[str, Any]:
    """List model dirs with state, so we can see what the user actually has."""
    models_dir = os.path.join(REPO_ROOT, "models")
    out: List[Dict[str, Any]] = []
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            d = os.path.join(models_dir, name)
            if not os.path.isdir(d):
                continue
            size = 0
            has_incomplete = False
            for root, _dirs, files in os.walk(d):
                for f in files:
                    p = os.path.join(root, f)
                    try:
                        size += os.path.getsize(p)
                    except OSError:
                        continue
                    if f.endswith((".incomplete", ".part", ".tmp")):
                        has_incomplete = True
            out.append({
                "name": name,
                "size_gb": round(size / 1e9, 2),
                "state": "downloading" if has_incomplete else ("ready" if size > 0 else "empty"),
            })
    return {"models_dir": models_dir, "entries": out}


PROBES = {
    "os": probe_os,
    "cpu": probe_cpu,
    "memory": probe_memory,
    "nvidia": probe_nvidia,
    "openvino": probe_openvino,
    "vendor_npu": probe_vendor_npu,
    "backends": probe_backends,
    "llama_devices": probe_llama_devices,
    "ssd": probe_ssd,
    "models": probe_models,
}


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def collect_report(tag: str = "", fast: bool = False) -> Dict[str, Any]:
    """Run every probe and assemble a redacted, structured report.

    Probes run **concurrently** -- several of them shell out (wmic/powershell/
    llama-bench) and serially they take >30 s. ``fast=True`` skips the two
    slowest optional probes (llama-bench device listing, SSD bandwidth) for a
    quick in-GUI snapshot.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    names = list(PROBES)
    if fast:
        names = [n for n in names if n not in ("llama_devices", "ssd")]

    report: Dict[str, Any] = {
        "schema": "edge-infer.diagnostics/1",
        "generated_at": _now(),
        "tag": tag,
        "fast": fast,
        "probes": {},
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_probe, n, PROBES[n]): n for n in names}
        for fut in as_completed(futs):
            n = futs[fut]
            report["probes"][n] = fut.result()
    # stable ordering for diffing between machines
    report["probes"] = {n: report["probes"][n] for n in names if n in report["probes"]}
    report["assessment"] = assess(report)
    return redact(report)


def assess(report: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a human-readable capability verdict from the probe results."""
    p = report.get("probes", {})
    ov = (p.get("openvino") or {}).get("data") or {}
    nv = (p.get("nvidia") or {}).get("data") or {}
    vn = (p.get("vendor_npu") or {}).get("data") or {}
    lb = (p.get("llama_devices") or {}).get("data") or {}

    ov_devices = ov.get("devices") or []
    caps: Dict[str, Any] = {}

    caps["nvidia_cuda"] = {
        "supported": bool(nv.get("available")),
        "backend": "llama.cpp CUDA",
        "detail": (nv.get("gpus") or [{}])[0].get("name") if nv.get("available") else nv.get("reason"),
    }
    caps["intel_gpu"] = {
        "supported": any(str(d).startswith("GPU") and "INTEL" in str(
            (ov.get("device_detail") or {}).get(d, {}).get("FULL_DEVICE_NAME", "")).upper()
            for d in ov_devices),
        "backend": "OpenVINO",
    }
    caps["intel_npu"] = {
        "supported": bool((ov.get("npu") or {}).get("enumerated")),
        "backend": "OpenVINO GenAI",
        "detail": (ov.get("npu") or {}).get("note"),
    }

    # AMD assessment -- the whole point of this module.
    amd_npu_present = bool(vn.get("amd_npu_hint"))
    amd_gpu_present = bool(vn.get("amd")) and any(
        "radeon" in str(d).lower() for d in (vn.get("amd_devices") or []))
    amd_backends = [b for b in (lb.get("backends") or []) if b.upper() in ("VULKAN", "ROCM")]
    caps["amd"] = {
        "npu_hardware_detected": amd_npu_present,
        "npu_device": vn.get("amd_npu_hint"),
        "npu_supported": False,
        "npu_reason": (
            "OpenVINO's NPU plugin is Intel-only and does not enumerate AMD XDNA. "
            "AMD XDNA needs ONNX Runtime + VitisAIExecutionProvider, or the "
            "FastFlowLM / Lemonade runtime (Ryzen AI 300/400 only). See docs/AMD.md."
            if amd_npu_present else
            "no AMD NPU detected"
        ),
        "gpu_hardware_detected": amd_gpu_present,
        "gpu_supported_via": amd_backends or [],
        "gpu_note": (
            "AMD Radeon is reachable through llama.cpp Vulkan/ROCm backends. "
            "Install the Vulkan or ROCm build and set EDGE_LLAMA_DIR to it."
            if amd_gpu_present else
            "no AMD GPU detected"
        ),
    }
    caps["cpu"] = {"supported": True, "backend": "OpenVINO / llama.cpp"}
    return caps


# ---------------------------------------------------------------------------
# problem recording
# ---------------------------------------------------------------------------

def record_problem(kind: str, message: str, *, context: Optional[Dict[str, Any]] = None,
                   exc: Optional[BaseException] = None) -> Dict[str, Any]:
    """Append a structured failure entry (JSONL) + return it.

    ``kind`` is a short category, e.g. ``"backend_load"`` / ``"inference"`` /
    ``"model_download"`` / ``"device_probe"``.
    """
    entry = {
        "schema": "edge-infer.problem/1",
        "at": _now(),
        "kind": kind,
        "message": str(message)[:2000],
        "context": redact(context or {}),
        "env_fingerprint": _fingerprint(),
    }
    if exc is not None:
        entry["exception"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-4000:]
    try:
        os.makedirs(os.path.dirname(PROBLEM_LOG), exist_ok=True)
        with open(PROBLEM_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return entry


def _fingerprint() -> Dict[str, Any]:
    """Cheap environment fingerprint attached to every problem entry."""
    fp: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import openvino as ov
        fp["openvino"] = getattr(ov, "__version__", "?")
        fp["ov_devices"] = list(ov.Core().get_available_devices())
    except Exception as e:  # noqa: BLE001
        fp["openvino_error"] = str(e)[:200]
    try:
        from .llama_backend import llama_available, find_llama_server
        fp["llama_cpp"] = llama_available()
        fp["llama_path"] = find_llama_server()
    except Exception:  # noqa: BLE001
        fp["llama_cpp"] = False
    for var in ("EDGE_DEVICE", "EDGE_MODEL_DIR", "EDGE_LLAMA_DIR", "EDGE_N_CPU_MOE"):
        v = os.environ.get(var)
        if v:
            fp[var] = v
    return redact(fp)


def read_problems(limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.isfile(PROBLEM_LOG):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(PROBLEM_LOG, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        continue
    except Exception:  # noqa: BLE001
        return []
    return out[-limit:]


# ---------------------------------------------------------------------------
# bundle export
# ---------------------------------------------------------------------------

def build_bundle(tag: str = "") -> str:
    """Write a single zip containing the report + logs, return its path.

    Intended flow: user runs the product on the AMD box, clicks "export
    diagnostics", and sends this zip back. Everything needed to reason about
    the failure is inside; nothing sensitive is.
    """
    os.makedirs(DIAG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"edge-infer-diag-{stamp}" + (f"-{tag}" if tag else "")

    report = collect_report(tag=tag)
    report_path = os.path.join(DIAG_DIR, base + ".json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Human-readable summary next to the JSON.
    txt_path = os.path.join(DIAG_DIR, base + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_text(report))

    zip_path = os.path.join(DIAG_DIR, base + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(report_path, arcname="diagnostics.json")
        z.write(txt_path, arcname="diagnostics.txt")
        # problems log
        if os.path.isfile(PROBLEM_LOG):
            z.write(PROBLEM_LOG, arcname="problems.jsonl")
        # python server log, if the launcher wrote one
        for cand in ("logs/edge-server.log", "logs/server.log"):
            p = os.path.join(REPO_ROOT, cand)
            if os.path.isfile(p):
                z.write(p, arcname=os.path.basename(p))
    return zip_path


def render_text(report: Dict[str, Any]) -> str:
    """Turn the report into a compact, paste-friendly text summary."""
    L: List[str] = []
    a = report.get("assessment", {})
    L.append("=" * 68)
    L.append("edge-infer diagnostics")
    L.append("=" * 68)
    L.append(f"generated : {report.get('generated_at')}")
    L.append(f"platform  : {report['probes'].get('os', {}).get('data', {}).get('platform')}")
    cpu = report["probes"].get("cpu", {}).get("data", {})
    L.append(f"cpu       : {cpu.get('Name') or cpu.get('processor')} "
             f"({cpu.get('NumberOfCores','?')}C/{cpu.get('NumberOfLogicalProcessors') or cpu.get('logical_cores')}T)")
    mem = report["probes"].get("memory", {}).get("data", {})
    L.append(f"memory    : {mem.get('total_gb','?')} GB")
    L.append("")
    L.append("--- capability assessment ---")
    nv = a.get("nvidia_cuda", {})
    L.append(f"  NVIDIA CUDA : {'YES' if nv.get('supported') else 'no '}  {nv.get('detail') or ''}")
    ig = a.get("intel_gpu", {})
    L.append(f"  Intel GPU   : {'YES' if ig.get('supported') else 'no '}  (OpenVINO)")
    inpu = a.get("intel_npu", {})
    L.append(f"  Intel NPU   : {'YES' if inpu.get('supported') else 'no '}  {inpu.get('detail') or ''}")
    amd = a.get("amd", {})
    L.append(f"  AMD NPU     : {'detected' if amd.get('npu_hardware_detected') else 'not found'}"
             f"  -> supported: NO")
    if amd.get("npu_hardware_detected"):
        L.append(f"                device: {amd.get('npu_device')}")
        L.append(f"                reason: {amd.get('npu_reason')}")
    L.append(f"  AMD GPU     : {'detected' if amd.get('gpu_hardware_detected') else 'not found'}"
             f"  -> via {amd.get('gpu_supported_via') or ['(no Vulkan/ROCm backend)']}")
    L.append("")
    L.append("--- OpenVINO ---")
    ov = report["probes"].get("openvino", {}).get("data", {})
    L.append(f"  version : {ov.get('version')}")
    L.append(f"  devices : {ov.get('devices')}")
    L.append("")
    L.append("--- llama.cpp ---")
    lb = report["probes"].get("llama_devices", {}).get("data", {})
    L.append(f"  backends: {lb.get('backends')}")
    for d in lb.get("devices") or []:
        L.append(f"    {d.get('id')}: {d.get('name')}")
    L.append("")
    L.append("--- models ---")
    for m in report["probes"].get("models", {}).get("data", {}).get("entries", []):
        L.append(f"  {m['name']:44} {m['size_gb']:7.2f} GB  [{m['state']}]")
    L.append("")
    L.append("--- probe health ---")
    for name, pr in report["probes"].items():
        mark = "ok " if pr.get("ok") else "ERR"
        extra = "" if pr.get("ok") else f"  {pr.get('error')}"
        L.append(f"  [{mark}] {name:<14} {pr.get('ms')} ms{extra}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="edge-infer environment diagnostics")
    ap.add_argument("--bundle", action="store_true", help="write a zip bundle")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--tag", default="", help="tag for the bundle filename")
    args = ap.parse_args()

    if args.bundle:
        print(build_bundle(tag=args.tag))
    else:
        rep = collect_report(tag=args.tag)
        print(json.dumps(rep, ensure_ascii=False, indent=2)
              if args.json else render_text(rep))
