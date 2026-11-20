"""OpenAI-compatible server for the edge-infer small-model tier.

Exposes POST /v1/chat/completions (and /v1/models) on 0.0.0.0 with API-key
auth, backed by an OpenVINO model on NPU/CPU. This is the "local inference
source" default; external OpenAI-compatible APIs are a separate profile the
GUI can switch to (per requirements decision #4).

Run:
    uvicorn edge_server.openai_server:app --host 0.0.0.0 --port 8080

Auth:
    Send `Authorization: Bearer <EDGE_API_KEY>`; env EDGE_API_KEY sets it.
    If unset, auth is disabled (local dev only).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# Real inference path: pure OpenVINO generator (no torch/transformers).
try:
    from .pure_ov_gen import PureOVGenerator
    _GEN_AVAILABLE = True
except Exception:  # openvino / tokenizers not installed yet
    PureOVGenerator = None  # type: ignore
    _GEN_AVAILABLE = False

# NPU path: OpenVINO GenAI LLMPipeline (handles static-shape NPU compilation).
try:
    from .genai_backend import GenaiBackend, is_llm_dir
    _GENAI_AVAILABLE = True
except Exception:
    GenaiBackend = None  # type: ignore
    is_llm_dir = None  # type: ignore
    _GENAI_AVAILABLE = False

# GGUF path: llama.cpp llama-server (needed for MoE beyond OpenVINO's 35B ceiling).
try:
    from .llama_backend import LlamaBackend, LlamaConfig, llama_available
    _LLAMA_AVAILABLE = True
except Exception:
    LlamaBackend = None  # type: ignore
    LlamaConfig = None  # type: ignore
    llama_available = None  # type: ignore
    _LLAMA_AVAILABLE = False

# VLM image path: OpenVINO GenAI VLMPipeline (handles the Qwen2VL image processor).
try:
    from .vlm_image import VlmImagePipeline, vlm_available
    _VLM_AVAILABLE = True
except Exception:
    VlmImagePipeline = None  # type: ignore
    vlm_available = None  # type: ignore
    _VLM_AVAILABLE = False

# Local agent tools (filesystem sandbox + tool loop).
try:
    from .agent_tools import (
        Workspace, parse_tool_call, run_tool, tools_prompt, TOOL_SCHEMAS,
    )
    _AGENT_AVAILABLE = True
except Exception:
    Workspace = None  # type: ignore
    _AGENT_AVAILABLE = False

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
EDGE_API_KEY = os.environ.get("EDGE_API_KEY", "")
DEFAULT_DEVICE = os.environ.get("EDGE_DEVICE", "NPU")
# Default model dir: an NPU-compatible symmetric-int4 LLM if present, else the
# VLM. Override with EDGE_MODEL_DIR.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Detect the discrete NVIDIA GPU once, so device aliases can route to the right
# backend. OpenVINO itself tells us which GPU.* id is discrete + NVIDIA-named.
_NVIDIA_DEVICE_IDS: set = set()
_INTEL_GPU_DEVICE_IDS: set = set()
_NVIDIA_AVAILABLE = False
_NPU_AVAILABLE = False


def _probe_gpu_ids() -> None:
    """Classify OpenVINO's GPU.* ids into NVIDIA (discrete) vs Intel (integrated).

    Also records whether an ``NPU`` is present. OpenVINO's NPU plugin targets
    **Intel** NPUs (VPU3720/4000) only -- an AMD XDNA / Ryzen AI NPU is *not*
    enumerated here, so ``_NPU_AVAILABLE`` stays False on AMD and callers must
    fall back to CPU instead of pretending the NPU exists.
    """
    global _NVIDIA_AVAILABLE, _NPU_AVAILABLE
    try:
        import openvino as ov

        core = ov.Core()
        for d in core.get_available_devices():
            if d.upper() == "NPU":
                _NPU_AVAILABLE = True
                continue
            if not d.startswith("GPU"):
                continue
            name = ""
            try:
                name = str(core.get_property(d, "FULL_DEVICE_NAME"))
            except Exception:  # noqa: BLE001
                pass
            if "NVIDIA" in name.upper():
                _NVIDIA_DEVICE_IDS.add(d.upper())
                _NVIDIA_AVAILABLE = True
            elif "INTEL" in name.upper() or "ARC" in name.upper():
                _INTEL_GPU_DEVICE_IDS.add(d.upper())
        # CUDA is available if nvidia-smi answers, even without OpenVINO.
        if not _NVIDIA_AVAILABLE:
            import shutil as _sh
            _NVIDIA_AVAILABLE = bool(_sh.which("nvidia-smi"))
    except Exception:  # noqa: BLE001
        pass


_probe_gpu_ids()


def _norm_dev(dev: Optional[str]) -> str:
    """Normalise a device alias: upper-case, strip spaces / underscores / hyphens.

    ``"NVIDIA GPU"``, ``"nvidia_gpu"`` and ``"NVIDIA-GPU"`` all collapse to
    ``"NVIDIAGPU"`` so aliases written naturally in docs/GUI still route right.
    (``GPU.0`` / ``GPU.1`` keep their dot.)
    """
    return (str(dev or "").upper()
            .replace(" ", "").replace("_", "").replace("-", ""))


def _is_nvidia(dev: Optional[str]) -> bool:
    """True when the requested device explicitly targets the NVIDIA GPU.

    Note: ``AUTO`` is intentionally NOT treated as NVIDIA here — auto means
    "pick the best device for the chosen model", which is handled per backend.
    """
    d = _norm_dev(dev)
    if not d:
        return False
    if d in ("NVIDIAGPU", "NVIDIA", "CUDA"):
        return True
    return d in _NVIDIA_DEVICE_IDS


def _is_intel_gpu(dev: Optional[str]) -> bool:
    d = _norm_dev(dev)
    if not d:
        return False
    if d in ("INTELGPU", "IGPU"):
        return True
    if d == "AUTO":
        return False
    return d in _INTEL_GPU_DEVICE_IDS


def _ov_device(dev: Optional[str], model_kind: str) -> str:
    """Map a GUI device alias to an OpenVINO device string.

    NVIDIA has no fast OpenVINO path (measured ~20× slower than CUDA), so an
    IR model requested on NVIDIA falls back to CPU rather than pretending.
    Likewise an ``NPU`` request on a machine without an **Intel** NPU (e.g. an
    AMD XDNA / Ryzen AI box, which OpenVINO does not enumerate) falls back to
    CPU instead of erroring out.
    """
    d = _norm_dev(dev)
    if d == "NPU":
        return "NPU" if _NPU_AVAILABLE else "CPU"
    if d in ("INTELGPU", "IGPU") or d in _INTEL_GPU_DEVICE_IDS:
        return "GPU.0"
    if d in ("NVIDIAGPU", "NVIDIA", "CUDA") or d in _NVIDIA_DEVICE_IDS:
        return "CPU"        # OpenVINO on NVIDIA is not the fast path
    if d in ("CPU",):
        return "CPU"
    # AUTO / empty: best OpenVINO device for the model kind.
    if model_kind == "llm" and _NPU_AVAILABLE:
        return "NPU"
    if any(x.startswith("GPU") for x in _ov_devices()):
        return "GPU.0"
    return "CPU"


def _ov_devices() -> list:
    try:
        import openvino as ov
        return ov.Core().get_available_devices()
    except Exception:  # noqa: BLE001
        return []


def _npu_compatible(model_dir: str, kind: str) -> bool:
    """Whether this model can actually compile on the Intel NPU.

    NPU (Meteor Lake / NPU3720) supports **symmetric INT4** (channel-wise
    ``int4-cw`` / group-wise ``int4-gq``) but **not** asymmetric AWQ or NF4 —
    those fail with ``Unsupported data type 'nf4'`` / DynamicDequantizeOp.
    Only a plain LLM IR in a symmetric-int4 export qualifies.
    """
    if kind != "llm":
        return False
    name = os.path.basename(model_dir).lower()
    if "int4-cw" in name or "int4-gq" in name or "int8-cw" in name:
        return True
    # Fall back to the export config when the name isn't conclusive.
    cfg = os.path.join(model_dir, "openvino_config.json")
    if os.path.exists(cfg):
        try:
            import json as _j
            q = (_j.load(open(cfg, encoding="utf-8")).get("quantization_config") or {})
            method = str((q.get("default_config") or {}).get("quant_method", "")).lower()
            if method in ("awq", "nf4", "gptq"):
                return False
        except Exception:  # noqa: BLE001
            pass
    return False


def _pick_default_model_dir() -> str:
    env = os.environ.get("EDGE_MODEL_DIR")
    if env:
        return env
    models = os.path.join(_REPO_ROOT, "models")
    # Prefer a plain-LLM IR (NPU-friendly) over the VLM.
    preferred = [
        "deepseek-r1-1.5b-int4-cw",
        "qwen3-0.6b-int4",
        "qwen3-vl-4b-int4",
    ]
    for name in preferred:
        if is_llm_dir and is_llm_dir(os.path.join(models, name)):
            return os.path.join(models, name)
    for name in preferred:
        if os.path.isdir(os.path.join(models, name)):
            return os.path.join(models, name)
    return os.path.join(models, "qwen3-vl-4b-int4")


DEFAULT_MODEL_DIR = _pick_default_model_dir()

app = FastAPI(title="edge-infer OpenAI-compatible API", version="0.1.0")

# Allow the Tauri native shell (and any local front-end) to call the API.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_gen = None  # lazily-loaded PureOVGenerator (VLM path)
_gen_device: Optional[str] = None
_genai = GenaiBackend() if _GENAI_AVAILABLE else None  # NPU/LLM path
_llama = LlamaBackend() if _LLAMA_AVAILABLE else None  # GGUF path (llama.cpp)
_vlm_img = None  # lazily-built VlmImagePipeline (VLM + images)
_MODELS_DIR = os.path.join(_REPO_ROOT, "models")

# Serve the GUI at "/" (works in a plain browser; the Tauri shell reuses the
# same backend commands).
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    from fastapi.staticfiles import StaticFiles

    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    from fastapi.responses import RedirectResponse

    @app.get("/")
    def _root():
        return RedirectResponse(url="/ui/")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3-vl-4b-int4"
    model_id: Optional[str] = None        # switch model per request (models/ <id>)
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    device: Optional[str] = None          # override EDGE_DEVICE per request
    enable_thinking: bool = False         # Qwen3 thinking mode
    attachments: List[dict] = []          # [{name, content}] inline text files
    images: List[str] = []                # [{data_url}] inline images (VLM)
    web_search: bool = False              # search the web and inject results


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------
def _check_auth(authorization: Optional[str]) -> None:
    if not EDGE_API_KEY:
        return  # auth disabled
    if not authorization or authorization.removeprefix("Bearer ").strip() != EDGE_API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": "qwen3-vl-4b-int4", "object": "model", "owned_by": "edge-infer"}],
    }


@app.get("/health")
def health():
    gguf = _find_gguf(DEFAULT_MODEL_DIR) if os.path.isdir(DEFAULT_MODEL_DIR) else None
    return {
        "ok": True,
        "backend_available": _GEN_AVAILABLE,
        "generator_ready": _gen is not None,
        "device": _gen_device,
        "model_dir": DEFAULT_MODEL_DIR,
        "model_dir_exists": os.path.isdir(DEFAULT_MODEL_DIR),
        "backends": {
            "openvino": _GEN_AVAILABLE,
            "openvino_genai": _GENAI_AVAILABLE,
            "llama_cpp": _LLAMA_AVAILABLE,
            "llama_cpp_running": bool(_llama and _llama.running()),
        },
        "gguf_model": gguf,
    }


# ---------------------------------------------------------------------------
# diagnostics -- environment snapshot + problem log (see diagnostics.py)
# ---------------------------------------------------------------------------
@app.get("/edge/diagnostics")
def edge_diagnostics(fast: int = 1, problems: int = 20):
    """Structured environment report + recent recorded problems.

    ``fast=0`` additionally runs the slow probes (llama-bench device listing,
    SSD bandwidth benchmark).
    """
    from . import diagnostics as diag

    report = diag.collect_report(tag="api", fast=bool(fast))
    return {"ok": True, "report": report,
            "problems": diag.read_problems(limit=max(0, min(problems, 200)))}


@app.post("/edge/diagnostics/bundle")
def edge_diagnostics_bundle(tag: str = ""):
    """Write a zip (report + logs) and return its path for the GUI to offer."""
    from . import diagnostics as diag

    try:
        path = diag.build_bundle(tag=tag)
        return {"ok": True, "path": path, "name": os.path.basename(path)}
    except Exception as e:  # noqa: BLE001
        diag.record_problem("diagnostics_bundle", str(e), exc=e)
        raise HTTPException(500, f"诊断包生成失败: {e}")


@app.get("/edge/problems")
def edge_problems(limit: int = 50):
    """Recent recorded problems (newest last)."""
    from . import diagnostics as diag

    return {"ok": True, "problems": diag.read_problems(limit=max(1, min(limit, 500)))}


@app.post("/edge/problems/clear")
def edge_problems_clear():
    """Truncate the problem log (after the user has exported it)."""
    from . import diagnostics as diag

    try:
        if os.path.isfile(diag.PROBLEM_LOG):
            open(diag.PROBLEM_LOG, "w", encoding="utf-8").close()
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))



@app.get("/edge/devices")
def edge_devices():
    """Real hardware discovery (OpenVINO devices + the three-tier report)."""
    import openvino as ov

    core = ov.Core()
    devs = core.get_available_devices()
    is_llm = bool(is_llm_dir(DEFAULT_MODEL_DIR)) if is_llm_dir else False
    ssd = _probe_ssd()
    # Requested device is only honoured when the hardware is actually present
    # (OpenVINO's NPU plugin is Intel-only, so AMD XDNA falls back to CPU).
    want = DEFAULT_DEVICE
    if want == "NPU" and not _NPU_AVAILABLE:
        want = "CPU"
    return {
        "openvino_devices": devs,
        "npu_available": _NPU_AVAILABLE,
        "npu_vendor": "Intel" if _NPU_AVAILABLE else None,
        "small_model_tier": {
            "device": want if want in devs else "CPU",
            "model": os.path.basename(DEFAULT_MODEL_DIR),
            "available": os.path.isdir(DEFAULT_MODEL_DIR),
            "npu_compatible": is_llm,
            "note": ("对称 int4 + LLMPipeline，可跑 NPU" if is_llm
                     else "该模型为非对称量化/VLM，NPU 不支持，将回退 CPU/GPU"),
        },
        "large_moe_tier": {
            "device": "NVIDIA-GPU",
            "note": "35B-A3B three-tier offload (SSD/RAM/VRAM) via edge-engine",
        },
        "storage": {
            "name": ssd.name,
            "read_gb_per_s": ssd.read_gb_per_s,
            "pcie5": ssd.pcie5,
            "note": f"{ssd.name} · {ssd.read_gb_per_s} GB/s 读取带宽"
                    + ("（PCIe 5.0 级）" if ssd.pcie5 else ""),
        },
    }


# cache the SSD probe (benchmark is a few hundred ms; no need to repeat)
_ssd_cache = None


def _probe_ssd():
    global _ssd_cache
    if _ssd_cache is None:
        try:
            from .storage_probe import detect_ssd

            bench = None
            for name in os.listdir(_MODELS_DIR) if os.path.isdir(_MODELS_DIR) else []:
                d = os.path.join(_MODELS_DIR, name)
                for fn in ("openvino_model.bin", "openvino_language_model.bin"):
                    p = os.path.join(d, fn)
                    if os.path.isfile(p) and os.path.getsize(p) > 500_000_000:
                        bench = p
                        break
                if bench:
                    break
            _ssd_cache = detect_ssd(bench_file=bench)
        except Exception as e:  # noqa: BLE001
            from types import SimpleNamespace
            print(f"[edge-infer] ssd probe failed: {e}")
            _ssd_cache = SimpleNamespace(name="NVMe SSD", read_gb_per_s=6.0, pcie5=False,
                                         read_bytes_per_s=6_000_000_000)
    return _ssd_cache


@app.get("/edge/run_plan")
def edge_run_plan(context: int = 8192, large_threshold_gb: int = 8):
    """Run the Rust edge-engine planner and return its JSON plan.

    Requires the `edge-engine` CLI on PATH (built from ../crates/edge-engine).
    Falls back to 503 with a clear message when it is not available.
    """
    import json as _json
    import shutil
    import subprocess
    import tempfile

    exe = shutil.which("edge-engine")
    if not exe:
        # Try the workspace target dir.
        cand = os.path.join(_REPO_ROOT, "target", "release", "edge-engine.exe")
        cand_d = os.path.join(_REPO_ROOT, "target", "debug", "edge-engine.exe")
        exe = cand if os.path.exists(cand) else (cand_d if os.path.exists(cand_d) else None)
    if not exe:
        raise HTTPException(503, "edge-engine CLI not found; build with `cargo build -p edge-engine`")

    ssd = _probe_ssd()
    probe = {
        # Real detected VRAM would come from NVML; 8 GiB is the 4070 Laptop total.
        "nvidia": {"name": "RTX 4070 Laptop", "vram_bytes": 8589934592,
                   "bandwidth_bytes_per_s": 288_000_000_000},
        "npu": {"name": "Intel AI Boost", "tops": 47, "memory_budget_bytes": 4294967296},
        "total_ram_bytes": 17179869184,
        "os_overhead": {"idle_baseline_bytes": 5368709120,
                        "igpu_dvmt_bytes": 536870912, "hidden_bytes": 1073741824},
        "ssd": {"name": ssd.name,
                "read_bytes_per_s": int(ssd.read_gb_per_s * 1_000_000_000),
                "pcie5": ssd.pcie5},
    }
    model = {"resident_bytes": 1500000000, "kv_bytes_per_token": 80000,
             "num_experts": 256, "expert_size_bytes": 11700000,
             "predicted": [0, 1, 2, 3, 4, 5, 6, 7, 8]}
    with tempfile.TemporaryDirectory() as d:
        pj = os.path.join(d, "probe.json")
        mj = os.path.join(d, "model.json")
        open(pj, "w", encoding="utf-8").write(_json.dumps(probe))
        open(mj, "w", encoding="utf-8").write(_json.dumps(model))
        out = subprocess.run([exe, pj, mj, str(context), str(large_threshold_gb)],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise HTTPException(500, f"edge-engine failed: {out.stderr[:300]}")
        return _json.loads(out.stdout)


def _dir_state(d: str) -> str:
    """Classify a models/ subdir: 'ready' | 'downloading' | 'empty'.

    A dir counts as **ready** only when it holds usable weights: a non-empty
    ``*.gguf``, or the OpenVINO IR pair (``openvino_model.xml`` +
    ``openvino_model.bin`` / ``openvino_language_model.xml``). While a download
    is in flight ModelScope leaves ``*.incomplete`` files, which we treat as
    'downloading' so the UI doesn't claim the model is installed.
    """
    try:
        entries = os.listdir(d)
    except OSError:
        return "empty"
    if not entries:
        return "empty"

    has_incomplete = any(e.endswith((".incomplete", ".part", ".tmp")) for e in entries)
    if has_incomplete:
        return "downloading"

    # GGUF: any real .gguf (weights, ignoring the tiny mmproj projector)
    ggufs = [e for e in entries if e.endswith(".gguf")]
    weights = [e for e in ggufs if "mmproj" not in e.lower()]
    if weights:
        # .gguf files are downloaded via ModelScope's own temp names, so a
        # present .gguf means it finished.
        return "ready"

    # OpenVINO IR: xml + matching bin
    for xml in ("openvino_model.xml", "openvino_language_model.xml"):
        if os.path.exists(os.path.join(d, xml)):
            base = xml[:-4]
            if os.path.exists(os.path.join(d, base + ".bin")):
                return "ready"
    return "downloading" if ggufs or os.path.exists(os.path.join(d, "openvino_config.json")) else "empty"


@app.get("/edge/catalog")
def edge_catalog(device: Optional[str] = None):
    """Recommended models, one entry per model with a variant per device.

    The GUI passes the currently-selected device so each card can highlight the
    matching variant (GGUF for NVIDIA, INT4 IR for Intel GPU / NPU).
    """
    try:
        from .model_catalog import MODELS, DEVICE_LABELS, resolve_device

        # Only dirs with complete weights count as installed.
        installed = set()
        if os.path.isdir(_MODELS_DIR):
            for name in os.listdir(_MODELS_DIR):
                if _dir_state(os.path.join(_MODELS_DIR, name)) == "ready":
                    installed.add(name)

        def _norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "").replace(".", "")

        def installed_for(repo: str) -> bool:
            short = _norm(repo.split("/")[-1])
            core = short[:-2] if short.endswith("ov") else short
            for n in installed:
                nn = _norm(n)
                if nn == short or nn.startswith(core[:14]) or core.startswith(nn[:14]):
                    return True
            return False

        sel = resolve_device(device)

        out = []
        for m in MODELS:
            variants = []
            for dev, v in m["variants"].items():
                variants.append({
                    **v,
                    "device": dev,
                    "device_label": DEVICE_LABELS.get(dev, dev),
                    "installed": installed_for(v["repo"]),
                })
            # Recommended variant: the one matching the selected device, else the
            # first (NVIDIA → Intel GPU → NPU order as authored).
            rec = next((v for v in variants if v["device"] == sel), None) or variants[0]
            out.append({
                "name": m["name"],
                "params": m["params"],
                "active": m["active"],
                "mm": m["mm"],
                "note": m["note"],
                "variants": variants,
                "recommended": rec,
                "selected_device": sel,
            })
        return {"models": out}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"catalog error: {e}")


@app.get("/edge/search")
def edge_search(q: str, n: int = 5):
    """Run a web search via browser automation (returns titles/urls/snippets)."""
    try:
        from .web_search import search_web, available

        if not available():
            raise HTTPException(503, "playwright 未安装，联网搜索不可用")
        return {"ok": True, "query": q, "results": search_web(q, max_results=n)}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"search error: {e}")


@app.get("/edge/models")
def edge_models():
    """List locally installed models with backend / device hints.

    The GUI uses `kind` + `devices` to build the per-device model picker.
    """
    out = []
    if os.path.isdir(_MODELS_DIR):
        for name in sorted(os.listdir(_MODELS_DIR)):
            d = os.path.join(_MODELS_DIR, name)
            if not os.path.isdir(d):
                continue
            kind = _model_kind(d)          # gguf | llm | vlm | unknown
            state = _dir_state(d)          # ready | downloading | empty
            label = {
                "gguf": "GGUF (llama.cpp)",
                "llm": "OpenVINO IR (LLM)",
                "vlm": "OpenVINO IR (VLM)",
            }.get(kind, "下载中" if state == "downloading" else "其他")
            # Which *devices this machine actually has* can run this model?
            # (Only list hardware that's present, so the GUI badge is honest.)
            npu_ok = _npu_compatible(d, kind)
            devices = []
            if kind in ("llm", "vlm"):
                if npu_ok and _NPU_AVAILABLE:
                    devices.append("NPU")
                if any(x.startswith("GPU") for x in _INTEL_GPU_DEVICE_IDS) or any(
                    x.startswith("GPU") for x in _ov_devices()
                ):
                    devices.append("Intel GPU")
                devices.append("CPU")
            if kind == "gguf":
                if _NVIDIA_AVAILABLE:
                    devices.append("NVIDIA GPU")
                devices.append("CPU")
            size = 0
            for root, _dirs, files in os.walk(d):
                for f in files:
                    try:
                        size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            out.append({
                "id": name,
                "note": label,
                "kind": kind,
                "state": state,            # ready | downloading | empty
                "usable": state == "ready",
                "size": f"{size / 1e9:.2f} GB",
                "size_gb": round(size / 1e9, 2),
                "path": d,
                "devices": devices,
                "device_label": devices[0] if devices else "—",
                "npu_compatible": npu_ok,
                "backend": "llama.cpp" if kind == "gguf" else "openvino",
            })
    return {"models": out}


# Track in-flight downloads so the UI can show progress.
_DOWNLOADS: dict[str, dict] = {}


@app.get("/edge/downloads")
def edge_downloads():
    """Status of in-flight / finished downloads."""
    return {"downloads": _DOWNLOADS}


@app.post("/edge/download")
def edge_download(repo: str, files: str = ""):
    """Kick off a ModelScope download in the background. Returns immediately.

    ``files`` is a comma-separated allow-list (optional). It exists because some
    GGUF repos bundle *many* quantizations (e.g. the 80B repo is >3 TB total) —
    we only want the one quant the catalog recommends.
    """
    import threading

    short = repo.split("/")[-1].lower()
    local = os.path.join(_MODELS_DIR, short)
    allow = [f.strip() for f in files.split(",") if f.strip()] or None

    if _DOWNLOADS.get(repo, {}).get("state") == "running":
        return {"ok": True, "message": f"{repo} 正在下载中。"}

    _DOWNLOADS[repo] = {"state": "running", "local": local, "files": allow or "all"}

    def _run():
        try:
            from modelscope import snapshot_download

            kw = {"local_dir": local}
            if allow:
                kw["allow_patterns"] = allow
            snapshot_download(repo, **kw)
            _DOWNLOADS[repo]["state"] = "done"
        except Exception as e:  # noqa: BLE001
            _DOWNLOADS[repo]["state"] = "error"
            _DOWNLOADS[repo]["error"] = str(e)[:300]
            print(f"[edge-infer] download failed for {repo}: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "message": f"已开始后台下载 {repo}，完成后刷新模型列表。"}


def _ensure_generator(model_dir: Optional[str] = None):
    """Lazily load the pure-OpenVINO VLM generator (device-aware)."""
    global _gen, _gen_device
    model_dir = model_dir or DEFAULT_MODEL_DIR
    if _gen is not None and getattr(_gen, "model_dir", None) == model_dir:
        return _gen
    if not _GEN_AVAILABLE:
        raise HTTPException(503, "OpenVINO generator unavailable (missing openvino/tokenizers)")
    if not os.path.isdir(model_dir):
        raise HTTPException(503, f"model dir not found: {model_dir}")
    _gen = PureOVGenerator(model_dir, device=DEFAULT_DEVICE)
    _gen_device = _gen.device
    return _gen


def _model_kind(d: str) -> str:
    """Classify a model dir: 'gguf' | 'llm' (OpenVINO LLM IR) | 'vlm' | 'unknown'."""
    if not os.path.isdir(d):
        return "none"
    if _find_gguf(d):
        return "gguf"
    if os.path.exists(os.path.join(d, "openvino_model.xml")):
        return "llm"
    if os.path.exists(os.path.join(d, "openvino_language_model.xml")):
        return "vlm"
    return "unknown"


def _resolve_model_dir(model_id: Optional[str]) -> str:
    """Resolve a request's model id to a local directory.

    ``model_id`` is a folder name under models/ (as returned by /edge/models).
    Falls back to the server default when absent or unknown.
    """
    if not model_id:
        return DEFAULT_MODEL_DIR
    # Accept either a bare folder name or an absolute path.
    if os.path.isabs(model_id) and os.path.isdir(model_id):
        return model_id
    cand = os.path.join(_MODELS_DIR, model_id)
    return cand if os.path.isdir(cand) else DEFAULT_MODEL_DIR


def _find_gguf(model_dir: str) -> Optional[str]:
    """Return the largest *.gguf under model_dir (a GGUF checkpoint), else None."""
    best, best_sz = None, -1
    if not os.path.isdir(model_dir):
        return None
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f.lower():
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > best_sz:
                    best, best_sz = p, sz
    return best


def _llama_stream(frame, gguf_path: str, req, system: str, user: str):
    """Proxy a streaming completion to a local llama-server (OpenAI API).

    Longer-term the Rust kernel will drive llama.cpp's C API directly for
    fine-grained tier control; via HTTP we still get ``--n-cpu-moe`` offload.
    """
    import json as _j
    import urllib.request

    # Think markers are assembled from char codes on purpose: written as
    # literals they keep getting mangled by tooling that eats "<...>",
    # which silently breaks the GUI think-block parser.
    _LT, _GT = chr(60), chr(62)
    T_OPEN = _LT + "think" + _GT
    T_CLOSE = _LT + "/think" + _GT

    want_think = bool(getattr(req, "enable_thinking", False))
    if not _llama.running() or (_llama._cfg and (
            _llama._cfg.model_path != gguf_path or _llama._cfg.reasoning != want_think)):
        # NVIDIA → offload everything to CUDA (discrete VRAM). MoE layers can be
        # pushed to CPU/RAM via EDGE_N_CPU_MOE when the model exceeds VRAM.
        n_cpu_moe = int(os.environ.get("EDGE_N_CPU_MOE", "0") or 0)
        _llama.load(LlamaConfig(model_path=gguf_path, port=8081,
                                n_cpu_moe=n_cpu_moe, reasoning=want_think))

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    # Qwen3 defaults to thinking mode via llama.cpp (content lands in
    # reasoning_content). Prepend /no_think unless the caller wants thinking.
    content = user
    if not getattr(req, "enable_thinking", False) and "/no_think" not in content:
        content = "/no_think\n" + content
    msgs.append({"role": "user", "content": content})
    payload = _j.dumps({
        "messages": msgs,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
        # Ask llama-server for real token counts + timings in the final chunk,
        # instead of estimating from character counts (which badly misreports
        # slow models -- e.g. an 80B MoE looked like 0.36 tok/s).
        "stream_options": {"include_usage": True},
    }).encode()

    r = urllib.request.Request(
        _llama.base + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    nchars = 0
    in_think = False  # currently inside a think block
    server_usage = None
    timings = None
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = _j.loads(data)
                if obj.get("usage"):
                    server_usage = obj["usage"]
                if obj.get("timings"):
                    timings = obj["timings"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                d = choices[0].get("delta", {})
                # llama-server reports reasoning separately in
                # `reasoning_content` (Qwen3 / MiniCPM-V-4.5). Some models
                # also echo the literal markers inside the text -- strip them.
                content = (d.get("content") or "")
                reasoning = (d.get("reasoning_content") or "")
                for _tag in (T_OPEN, T_CLOSE):
                    content = content.replace(_tag, "")
                    reasoning = reasoning.replace(_tag, "")

                delta = ""
                if reasoning and want_think:
                    if not in_think:
                        delta += T_OPEN
                        in_think = True
                    delta += reasoning
                # Thinking off -> drop the reasoning; it would otherwise leak
                # into the visible answer as plain text.
                if content:
                    if in_think:
                        delta += T_CLOSE
                        in_think = False
                    delta += content
            except Exception:
                continue
            if delta:
                nchars += len(delta)
                yield frame(delta)
    if in_think:
        yield frame(T_CLOSE)
    dt = max(time.time() - t0, 1e-3)

    # Prefer llama-server's own measurements; fall back to the char estimate.
    if server_usage:
        p_tok = int(server_usage.get("prompt_tokens") or 0)
        c_tok = int(server_usage.get("completion_tokens") or 0)
    else:
        p_tok = max(len(user) // 2, 1)
        c_tok = max(nchars // 2, 1)
    if timings and timings.get("predicted_per_second"):
        est = float(timings["predicted_per_second"])
    else:
        est = c_tok / dt
    usage = {
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": p_tok + c_tok,
        "est_tok_s": round(est, 2),
        "device": "llama.cpp",
    }
    if timings:
        if timings.get("prompt_per_second"):
            usage["prefill_tok_s"] = round(float(timings["prompt_per_second"]), 2)
        if timings.get("prompt_ms") is not None:
            usage["ttft_ms"] = round(float(timings["prompt_ms"]), 1)
    yield frame("", finish="stop", usage=usage)


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)

    system = next((m.content for m in req.messages if m.role == "system"), "")
    user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not user:
        user = "\n".join(m.content for m in req.messages)

    # Fold attachments (inline text files) into the prompt.
    if req.attachments:
        extra = []
        for a in req.attachments:
            name = a.get("name", "file")
            content = a.get("content", "")
            extra.append(f"\n\n[附件 {name}]\n{content[:6000]}")
        user = user + "".join(extra)

    # Optional web search: fetch fresh results and inject them into the prompt.
    if req.web_search:
        try:
            from . import web_search as _ws_mod

            hits = _ws_mod.search_web(user, max_results=5)
            block = _ws_mod.format_results(hits)
            if block:
                user = (f"{block}\n\n请基于以上联网搜索结果回答下面的问题；"
                        f"若结果不相关，请说明并凭已有知识作答。\n\n【用户问题】\n{user}")
        except Exception as e:  # noqa: BLE001
            print(f"[edge-infer] web search failed: {e}")

    # Resolve the per-request model (falls back to the server default).
    model_dir = _resolve_model_dir(req.model_id)
    model_ok = os.path.isdir(model_dir)
    dev = req.device or DEFAULT_DEVICE
    model_kind = _model_kind(model_dir) if model_ok else "none"
    # OpenVINO device string (NVIDIA maps to CPU — CUDA is handled by llama.cpp).
    ov_dev = _ov_device(dev, model_kind)

    def _degrade_text() -> str:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        reason = []
        if not _GEN_AVAILABLE:
            reason.append("openvino/tokenizers 未安装")
        if not model_ok:
            reason.append(f"模型目录不存在: {model_dir}")
        elif model_kind == "gguf" and not _LLAMA_AVAILABLE:
            reason.append("模型是 GGUF 但未找到 llama-server（设置 EDGE_LLAMA_DIR）")
        elif model_kind == "llm" and not _GENAI_AVAILABLE:
            reason.append("模型是 OpenVINO IR 但 openvino_genai 不可用")
        if dev == "NPU" and not _NPU_AVAILABLE:
            reason.append("请求了 NPU 但本机没有 Intel NPU（OpenVINO 的 NPU 插件为 Intel 专属）")
        if not reason:
            reason.append(f"无可用推理路径（模型类型 '{model_kind}'）")
        return ("[edge-infer 降级响应] " + "；".join(reason)
                + "\n运行 `python -m edge_server.diagnostics` 查看完整环境报告。"
                + "\n| echo: " + prompt)

    # ---- streaming (SSE) ----
    if req.stream:
        from fastapi.responses import StreamingResponse

        def sse(obj: dict) -> str:
            import json as _j
            return f"data: {_j.dumps(obj, ensure_ascii=False)}\n\n"

        def event_stream():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            def frame(delta: str, finish=None, usage=None) -> str:
                obj = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": ({"content": delta} if delta else {}),
                                 "finish_reason": finish}],
                }
                if usage:
                    obj["usage"] = usage
                return sse(obj)

            gguf = _find_gguf(model_dir) if model_ok else None
            use_llama = _LLAMA_AVAILABLE and gguf is not None
            use_genai = (not use_llama) and _GENAI_AVAILABLE and model_ok and model_kind == "llm"
            use_vlm_img = (not use_llama and not use_genai) and bool(req.images) and model_kind == "vlm" and _VLM_AVAILABLE
            # NVIDIA + non-GGUF model: OpenVINO would run ~20× slower there, so
            # warn loudly instead of silently degrading.
            if _is_nvidia(dev) and model_kind in ("llm", "vlm") and not use_llama:
                yield sse({"notice": "所选模型为 OpenVINO IR；NVIDIA 走 CUDA 需 GGUF 模型。"
                                     "当前将用 OpenVINO 的 CPU 路径（较慢）。"
                                     "建议在模型页选择 GGUF 模型（如 Qwen3.6-35B-A3B）。"})
            try:
                if use_vlm_img:
                    # VLM with images → VLMPipeline; emit the whole answer at once.
                    global _vlm_img
                    if _vlm_img is None or _vlm_img.model_dir != model_dir:
                        _vlm_img = VlmImagePipeline(model_dir, device=ov_dev)
                    r = _vlm_img.chat(user, req.images, system=system,
                                      max_new_tokens=req.max_tokens, temperature=req.temperature)
                    yield frame(r.text)
                    yield frame("", finish="stop", usage={
                        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                        "total_tokens": r.prompt_tokens + r.completion_tokens,
                        "est_tok_s": round(r.est_tok_s, 2), "device": r.device,
                    })
                elif use_llama:
                    yield from _llama_stream(frame, gguf, req, system, user)
                elif use_genai:
                    _genai.load(
                        model_dir, device=ov_dev,
                        cache_dir=os.path.join(model_dir, ".npu_cache", dev),
                    )
                    t0 = time.time()
                    nchars = 0
                    got = False
                    for delta in _genai.stream_chat(
                        user, system=system, max_new_tokens=req.max_tokens,
                        temperature=req.temperature, enable_thinking=req.enable_thinking,
                    ):
                        got = True
                        nchars += len(delta)
                        yield frame(delta)
                    if not got:
                        yield frame("")
                    dt = max(time.time() - t0, 1e-3)
                    comp = max(nchars // 2, 1)   # CJK ~2 chars/token
                    yield frame("", finish="stop", usage={
                        "prompt_tokens": max(len(user) // 2, 1),
                        "completion_tokens": comp,
                        "total_tokens": max(len(user) // 2, 1) + comp,
                        "est_tok_s": round(comp / dt, 2),
                        "device": _genai._device or dev,
                    })
                else:
                    text = _degrade_text()
                    if _GEN_AVAILABLE and model_ok and model_kind == "vlm":
                        try:
                            gen = _ensure_generator(model_dir)
                            text = gen.chat(user, system=system, max_new_tokens=req.max_tokens,
                                            do_sample=req.temperature > 0.01,
                                            temperature=max(req.temperature, 0.01)).text
                        except Exception as e:  # noqa: BLE001
                            from . import diagnostics as _diag
                            _diag.record_problem(
                                "backend_pure_ov_fallback", str(e),
                                context={"model_dir": model_dir, "device": dev}, exc=e)
                    else:
                        from . import diagnostics as _diag
                        _diag.record_problem(
                            "degraded_response", text.split("\n")[0][:300],
                            context={"model_dir": model_dir, "model_kind": model_kind,
                                     "device": dev, "ov_device": ov_dev,
                                     "llama_available": _LLAMA_AVAILABLE,
                                     "genai_available": _GENAI_AVAILABLE,
                                     "npu_available": _NPU_AVAILABLE})
                    yield frame(text)
                    yield frame("", finish="stop")
            except Exception as e:  # noqa: BLE001
                from . import diagnostics as _diag
                _diag.record_problem(
                    "inference_stream",
                    str(e),
                    context={"model_dir": model_dir, "kind": model_kind,
                             "device": dev, "ov_device": ov_dev,
                             "use_llama": use_llama, "use_genai": use_genai,
                             "use_vlm_img": use_vlm_img},
                    exc=e,
                )
                yield sse({"error": str(e)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---- non-streaming ----
    # VLM with images: use VLMPipeline (handles the image processor).
    if req.images and model_kind == "vlm" and _VLM_AVAILABLE:
        try:
            global _vlm_img
            if _vlm_img is None or _vlm_img.model_dir != model_dir:
                _vlm_img = VlmImagePipeline(model_dir, device=ov_dev)
            r = _vlm_img.chat(user, req.images, system=system,
                              max_new_tokens=req.max_tokens, temperature=req.temperature)
            return _resp(req.model, r.text, prompt_tokens=r.prompt_tokens,
                         completion_tokens=r.completion_tokens, tok_s=r.est_tok_s,
                         device=r.device)
        except Exception as e:  # noqa: BLE001
            from . import diagnostics as _diag
            _diag.record_problem("vlm_image", str(e),
                                 context={"model_dir": model_dir, "device": ov_dev}, exc=e)
            print(f"[edge-infer] VLM image path failed: {e}")

    gguf = _find_gguf(model_dir) if model_ok else None
    if _LLAMA_AVAILABLE and gguf is not None:
        try:
            t0 = time.time()
            chunks = []
            for piece in _llama_stream(lambda d, **k: d, gguf, req, system, user):
                chunks.append(piece)
            text = "".join(chunks)
            dt = max(time.time() - t0, 1e-3)
            comp = max(len(text) // 2, 1)
            return _resp(req.model, text, prompt_tokens=max(len(user) // 2, 1),
                         completion_tokens=comp, tok_s=comp / dt, device="llama.cpp")
        except Exception as e:  # noqa: BLE001
            from . import diagnostics as _diag
            _diag.record_problem("backend_llama_fallback", str(e),
                                 context={"model_dir": model_dir, "gguf": gguf, "device": dev}, exc=e)
            print(f"[edge-infer] llama.cpp path failed, falling back: {e}")

    if _GENAI_AVAILABLE and model_ok and model_kind == "llm":
        try:
            _genai.load(
                model_dir,
                device=ov_dev,
                cache_dir=os.path.join(model_dir, ".npu_cache", dev),
            )
            r = _genai.chat(
                user, system=system,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                enable_thinking=req.enable_thinking,
            )
            return _resp(req.model, r.text, prompt_tokens=r.prompt_tokens,
                         completion_tokens=r.completion_tokens, tok_s=r.est_tok_s,
                         device=r.device)
        except Exception as e:  # noqa: BLE001
            from . import diagnostics as _diag
            _diag.record_problem("backend_genai_fallback", str(e),
                                 context={"model_dir": model_dir, "device": dev,
                                          "ov_device": ov_dev}, exc=e)
            print(f"[edge-infer] genai path failed, falling back: {e}")

    # VLM path (pure OpenVINO) / degradation.
    if not _GEN_AVAILABLE or not model_ok or model_kind != "vlm":
        # Not a VLM-IR directory (e.g. GGUF-only) → don't try PureOVGenerator.
        # A degradation is exactly the kind of thing users report as "it's
        # broken but I can't say why", so record it (with the reason) here.
        from . import diagnostics as _diag
        reasons = []
        if not _GEN_AVAILABLE:
            reasons.append("openvino/tokenizers 未安装")
        if not model_ok:
            reasons.append(f"模型目录不存在: {model_dir}")
        if model_kind != "vlm":
            reasons.append(f"模型类型为 '{model_kind}'（非 VLM IR），"
                           f"且无可用 GGUF / GenAI 路径")
        _diag.record_problem(
            "degraded_response",
            "；".join(reasons) or "unknown",
            context={"model_dir": model_dir, "model_kind": model_kind,
                     "device": dev, "ov_device": ov_dev,
                     "llama_available": _LLAMA_AVAILABLE,
                     "genai_available": _GENAI_AVAILABLE},
        )
        return _resp(req.model, _degrade_text(), prompt_tokens=len(req.messages),
                     completion_tokens=-1, tok_s=0.0, degraded=True)

    gen = _ensure_generator(model_dir)
    result = gen.chat(
        user,
        system=system,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=max(req.temperature, 0.01),
    )
    return _resp(
        req.model, result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tok_s=result.est_tok_s,
        device=result.device,
    )


def _resp(model: str, text: str, *, prompt_tokens: int, completion_tokens: int,
          tok_s: float, device: Optional[str] = None, degraded: bool = False) -> dict:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + max(completion_tokens, 0),
        "est_tok_s": round(tok_s, 2),
    }
    if device:
        usage["device"] = device
    if degraded:
        usage["degraded"] = True
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text},
             "finish_reason": "stop"}
        ],
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# local agent (filesystem tools + tool loop)
# ---------------------------------------------------------------------------
_ws = Workspace() if _AGENT_AVAILABLE else None


class AgentRequest(BaseModel):
    message: str
    max_steps: int = Field(default=5, ge=1, le=12)
    max_tokens: int = Field(default=512, ge=1, le=4096)


@app.get("/agent/workspace")
def agent_workspace():
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    return {
        "root": str(_ws.root),
        "current": str(_ws.current.relative_to(_ws.root)) or ".",
        "tools": [t["name"] for t in TOOL_SCHEMAS],
    }


@app.get("/agent/list")
def agent_list(path: str = "."):
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    return {"path": path, "content": _ws.list_dir(path)}


# ---- project management ---------------------------------------------------
@app.get("/project/list")
def project_list():
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    projs = []
    pd = _ws._projects_dir()
    for c in sorted(pd.iterdir()):
        if c.is_dir():
            nfiles = sum(1 for _ in c.rglob("*") if _.is_file())
            meta = c / ".edge-project.json"
            projs.append({
                "name": c.name, "files": nfiles,
                "has_venv": (c / ".venv").exists(),
                "created": (json.loads(meta.read_text(encoding="utf-8")).get("created")
                            if meta.exists() else ""),
            })
    return {"projects": projs, "current": str(_ws.current.relative_to(_ws.root))}


@app.post("/project/create")
def project_create(name: str, template: str = "empty"):
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    return {"ok": True, "message": _ws.create_project(name, template),
            "current": str(_ws.current.relative_to(_ws.root))}


@app.post("/project/open")
def project_open(name: str):
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    return {"ok": True, "message": _ws.open_project(name),
            "current": str(_ws.current.relative_to(_ws.root))}


@app.get("/project/tree")
def project_tree(path: str = "."):
    """Recursive project tree (dirs + files), for the GUI file explorer.

    `path` is resolved relative to the *workspace root* (e.g. "projects/foo").
    """
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")

    def walk(p, rel="", depth=0):
        if depth > 4:
            return []
        out = []
        try:
            for c in sorted(p.iterdir()):
                rp = f"{rel}/{c.name}".lstrip("/")
                if c.name in (".venv", "__pycache__", "node_modules", ".git"):
                    out.append({"name": c.name, "type": "dir", "path": rp, "skipped": True})
                    continue
                if c.is_dir():
                    out.append({"name": c.name, "type": "dir", "path": rp,
                                "children": walk(c, rp, depth + 1)})
                else:
                    out.append({"name": c.name, "type": "file", "path": rp,
                                "size": c.stat().st_size})
        except OSError:
            pass
        return out

    # resolve from the workspace root (not the current dir)
    base = (_ws.root / path).resolve()
    try:
        base.relative_to(_ws.root)
    except ValueError:
        raise HTTPException(400, "path escapes workspace")
    if not base.is_dir():
        return {"root": path, "tree": []}
    return {"root": path, "tree": walk(base)}


@app.get("/project/file")
def project_file(path: str):
    """Read a file's content for the editor (path is workspace-root relative)."""
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    p = (_ws.root / path).resolve()
    try:
        p.relative_to(_ws.root)
    except ValueError:
        raise HTTPException(400, "path escapes workspace")
    if not p.is_file():
        return {"path": path, "content": ""}
    return {"path": path, "content": _ws.read_file(path)}


class WriteFileReq(BaseModel):
    path: str
    content: str


@app.post("/project/file")
def project_file_write(req: WriteFileReq):
    """Save file content from the editor (path is workspace-root relative)."""
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    p = (_ws.root / req.path).resolve()
    try:
        p.relative_to(_ws.root)
    except ValueError:
        raise HTTPException(400, "path escapes workspace")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(req.content, encoding="utf-8")
    return {"ok": True, "message": f"已保存 {req.path} ({len(req.content)} 字符)"}


@app.post("/agent/sandbox_run")
def agent_sandbox_run(cmd: str):
    """Run a command inside the current project's sandbox (used by the GUI)."""
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    return {"ok": True, "output": _ws.sandbox_run(cmd)}


@app.post("/project/delete")
def project_delete(name: str):
    """Delete a project folder (and everything in it). Confined to workspace."""
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    import shutil as _sh

    base = (_ws._projects_dir() / name).resolve()
    try:
        base.relative_to(_ws.root)
    except ValueError:
        raise HTTPException(400, "path escapes workspace")
    if not base.is_dir():
        raise HTTPException(404, f"project not found: {name}")
    _sh.rmtree(base, ignore_errors=True)
    # if we were inside the deleted project, reset to workspace root
    try:
        _ws.current.relative_to(base)
        _ws.current = _ws.root
    except ValueError:
        pass
    return {"ok": True, "message": f"已删除项目 {name}"}


class AttachmentReq(BaseModel):
    name: str
    content: str = ""
    data_b64: Optional[str] = None       # binary payload (images, folders, etc.)
    rel_path: Optional[str] = None       # for folder uploads (keeps sub-paths)


@app.post("/agent/upload")
def agent_upload(req: AttachmentReq):
    """Save an uploaded attachment into the workspace's uploads/ dir.

    Supports text (``content``) and binary (``data_b64``) payloads. Folder
    uploads pass ``rel_path`` so the directory structure is preserved.
    """
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    import base64 as _b64

    updir = _ws.root / "uploads"
    updir.mkdir(parents=True, exist_ok=True)

    # Preserve folder structure when rel_path is given; else flatten to basename.
    rel = (req.rel_path or req.name).replace("\\", "/").lstrip("/")
    rel = "/".join(seg for seg in rel.split("/") if seg not in ("", ".", ".."))
    target = (updir / rel).resolve()
    try:
        target.relative_to(updir.resolve())
    except ValueError:
        raise HTTPException(400, "path escapes uploads dir")

    target.parent.mkdir(parents=True, exist_ok=True)
    if req.data_b64:
        try:
            target.write_bytes(_b64.b64decode(req.data_b64))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"invalid base64 payload: {e}")
    else:
        target.write_text(req.content, encoding="utf-8")

    return {"ok": True, "path": f"uploads/{rel}", "size": target.stat().st_size,
            "message": f"已保存附件 {rel}"}


@app.post("/edge/upload")
def edge_upload(req: AttachmentReq):
    """Upload an attachment to the shared uploads dir (works without agent mode)."""
    import base64 as _b64

    updir = os.path.join(_MODELS_DIR, "..", "uploads")
    updir = os.path.abspath(updir)
    os.makedirs(updir, exist_ok=True)
    rel = (req.rel_path or req.name).replace("\\", "/").lstrip("/")
    rel = "/".join(seg for seg in rel.split("/") if seg not in ("", ".", ".."))
    target = os.path.abspath(os.path.join(updir, rel))
    if not target.startswith(updir):
        raise HTTPException(400, "path escapes uploads dir")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if req.data_b64:
        try:
            with open(target, "wb") as f:
                f.write(_b64.b64decode(req.data_b64))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"invalid base64 payload: {e}")
    else:
        with open(target, "w", encoding="utf-8") as f:
            f.write(req.content)
    return {"ok": True, "path": os.path.relpath(target, updir).replace("\\", "/"),
            "size": os.path.getsize(target)}


@app.post("/edge/device/set")
def edge_device_set(device: str):
    """Switch the inference device at runtime (unloads the current pipeline)."""
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device
    if _genai is not None:
        try:
            _genai.unload()
        except Exception:  # noqa: BLE001
            pass
    return {"ok": True, "device": DEFAULT_DEVICE,
            "message": f"已切换设备为 {device}（下次推理生效）"}


@app.get("/edge/device/current")
def edge_device_current():
    return {"device": DEFAULT_DEVICE,
            "loaded_on": (_genai._device if _genai else None)}


@app.post("/project/sandbox_setup")
def project_sandbox_setup():
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    return {"ok": True, "message": _ws.sandbox_setup()}


@app.post("/agent/chat")
def agent_chat(req: AgentRequest):
    """Run a tool-using agent loop: model decides tools -> execute -> repeat."""
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    if not (_GENAI_AVAILABLE and is_llm_dir(DEFAULT_MODEL_DIR)):
        # fall back to a plain answer when no tool-capable LLM is present
        return {"steps": [], "answer": "[degradation] 本地 LLM 不可用，无法执行 Agent 循环。",
                "workspace": str(_ws.root)}

    _genai.load(DEFAULT_MODEL_DIR, device=DEFAULT_DEVICE,
                cache_dir=os.path.join(DEFAULT_MODEL_DIR, ".npu_cache", DEFAULT_DEVICE))

    system = ("你是一个本地项目开发助手，能创建项目、写代码、建沙箱、装依赖、跑程序。\n"
              + tools_prompt())
    steps = []
    # Keep a growing scratchpad so the model sees the full tool history.
    scratch = f"用户任务：{req.message}\n"
    for _ in range(req.max_steps):
        resp = _genai.chat(scratch, system=system,
                           max_new_tokens=req.max_tokens, temperature=0.0)
        text = resp.text
        call = parse_tool_call(text)
        if not call:
            return {"steps": steps, "answer": _strip_think(text),
                    "workspace": str(_ws.current)}
        name = call.get("name")
        args = call.get("arguments", {}) or {}
        result = run_tool(_ws, name, args)
        steps.append({"tool": name, "arguments": args, "result": result})
        scratch += (f"\n[调用 {name}({json.dumps(args, ensure_ascii=False)})]\n"
                    f"<tool_result>\n{result}\n</tool_result>\n")
        if name == "get_workspace":
            scratch += "继续下一步。\n"
    return {"steps": steps, "answer": "（达到最大步骤数，任务可能未完成）",
            "workspace": str(_ws.current)}


def _strip_think(text: str) -> str:
    """Remove Qwen3  thinking... response blocks from the final answer."""
    import re as _re
    pat = r"<think.*?</think>"
    text = _re.sub(pat, "", text, flags=_re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# parent-process watchdog (auto-exit when the GUI dies)
# ---------------------------------------------------------------------------
def _start_parent_watchdog() -> None:
    """Exit this process when EDGE_PARENT_PID is gone.

    When the Tauri GUI is force-killed, the child Python process would
    otherwise linger. Poll the parent and shut down cleanly once it disappears.
    """
    ppid = os.environ.get("EDGE_PARENT_PID")
    if not ppid:
        return
    try:
        ppid_i = int(ppid)
    except ValueError:
        return

    import threading

    def _watch():
        import time as _t
        while True:
            _t.sleep(3)
            if not _pid_alive(ppid_i):
                os._exit(0)

    threading.Thread(target=_watch, daemon=True).start()


def _pid_alive(pid: int) -> bool:
    import subprocess as _sp
    try:
        if os.name == "nt":
            out = _sp.run(["tasklist", "/FI", f"PID eq {pid}"],
                          capture_output=True, text=True, timeout=5, errors="replace")
            return str(pid) in (out.stdout or "")
        os.kill(pid, 0)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    _start_parent_watchdog()
    uvicorn.run(app, host="0.0.0.0", port=8080)

