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
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    device: Optional[str] = None          # override EDGE_DEVICE per request
    enable_thinking: bool = False         # Qwen3 thinking mode
    attachments: List[dict] = []          # [{name, content}] inline text files


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


@app.get("/edge/devices")
def edge_devices():
    """Real hardware discovery (OpenVINO devices + the three-tier report)."""
    import openvino as ov

    core = ov.Core()
    devs = core.get_available_devices()
    is_llm = bool(is_llm_dir(DEFAULT_MODEL_DIR)) if is_llm_dir else False
    ssd = _probe_ssd()
    return {
        "openvino_devices": devs,
        "small_model_tier": {
            "device": DEFAULT_DEVICE if DEFAULT_DEVICE in devs else "CPU",
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


@app.get("/edge/catalog")
def edge_catalog():
    """Curated multimodal model recommendations per accelerator."""
    try:
        from .model_catalog import CATALOG

        installed = set()
        if os.path.isdir(_MODELS_DIR):
            installed = set(os.listdir(_MODELS_DIR))

        def _norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "").replace(".", "")

        def installed_for(repo: str) -> bool:
            # repo short name (e.g. MiniCPM-V-4_5-int4-cw-ov) vs local dir
            # (e.g. minicpm-v-4_5-int4-cw). Match on a normalized core token.
            short = _norm(repo.split("/")[-1])
            core = short.replace("ov", "") if short.endswith("ov") else short
            for n in installed:
                nn = _norm(n)
                if nn == short or nn.startswith(core[:14]) or core.startswith(nn[:14]):
                    return True
            return False

        out = {}
        for dev, items in CATALOG.items():
            out[dev] = [{**it, "installed": installed_for(it["repo"])} for it in items]
        return {"catalog": out}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"catalog error: {e}")


@app.get("/edge/models")
def edge_models():
    out = []
    if os.path.isdir(_MODELS_DIR):
        for name in sorted(os.listdir(_MODELS_DIR)):
            d = os.path.join(_MODELS_DIR, name)
            if not os.path.isdir(d):
                continue
            # Detect the layout / backend.
            #   OpenVINO LLM -> openvino_model.xml            (NPU-able with sym int4)
            #   OpenVINO VLM -> openvino_language_model.xml   (CPU/GPU)
            #   GGUF         -> *.gguf                        (llama.cpp, 3-tier offload)
            is_llm = os.path.exists(os.path.join(d, "openvino_model.xml"))
            is_vlm = os.path.exists(os.path.join(d, "openvino_language_model.xml"))
            has_gguf = any(f.endswith(".gguf") for f in os.listdir(d))
            if has_gguf:
                kind = "GGUF (llama.cpp)"
            elif is_llm:
                kind = "OpenVINO IR (LLM)"
            elif is_vlm:
                kind = "OpenVINO IR (VLM)"
            else:
                kind = "其他"
            size = 0
            for root, _dirs, files in os.walk(d):
                for f in files:
                    try:
                        size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            out.append({
                "id": name,
                "note": kind,
                "size": f"{size / 1e9:.2f} GB",
                "path": d,
                "npu_compatible": is_llm,
                "backend": "llama.cpp" if has_gguf else "openvino",
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


def _ensure_generator():
    """Lazily load the pure-OpenVINO generator (NPU preferred, CPU fallback)."""
    global _gen, _gen_device
    if _gen is not None:
        return _gen
    if not _GEN_AVAILABLE:
        raise HTTPException(503, "OpenVINO generator unavailable (missing openvino/tokenizers)")
    if not os.path.isdir(DEFAULT_MODEL_DIR):
        raise HTTPException(503, f"model dir not found: {DEFAULT_MODEL_DIR}")
    _gen = PureOVGenerator(DEFAULT_MODEL_DIR, device=DEFAULT_DEVICE)
    _gen_device = _gen.device
    return _gen


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

    if not _llama.running() or (_llama._cfg and _llama._cfg.model_path != gguf_path):
        _llama.load(LlamaConfig(model_path=gguf_path, port=8081))

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
    }).encode()

    r = urllib.request.Request(
        _llama.base + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    nchars = 0
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
                d = obj["choices"][0].get("delta", {})
                delta = d.get("content") or d.get("reasoning_content") or ""
            except Exception:
                continue
            if delta:
                nchars += len(delta)
                yield frame(delta)
    dt = max(time.time() - t0, 1e-3)
    comp = max(nchars // 2, 1)
    yield frame("", finish="stop", usage={
        "prompt_tokens": max(len(user) // 2, 1),
        "completion_tokens": comp,
        "total_tokens": max(len(user) // 2, 1) + comp,
        "est_tok_s": round(comp / dt, 2),
        "device": "llama.cpp",
    })


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

    model_ok = os.path.isdir(DEFAULT_MODEL_DIR)
    dev = req.device or DEFAULT_DEVICE

    def _degrade_text() -> str:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
        reason = []
        if not _GEN_AVAILABLE:
            reason.append("openvino/tokenizers not installed")
        if not model_ok:
            reason.append(f"no model at {DEFAULT_MODEL_DIR}")
        return "[edge-infer degradation] " + "; ".join(reason) + " | echo: " + prompt

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

            gguf = _find_gguf(DEFAULT_MODEL_DIR) if model_ok else None
            use_llama = _LLAMA_AVAILABLE and gguf is not None
            use_genai = (not use_llama) and _GENAI_AVAILABLE and model_ok and is_llm_dir(DEFAULT_MODEL_DIR)
            try:
                if use_llama:
                    yield from _llama_stream(frame, gguf, req, system, user)
                elif use_genai:
                    _genai.load(
                        DEFAULT_MODEL_DIR, device=dev,
                        cache_dir=os.path.join(DEFAULT_MODEL_DIR, ".npu_cache", dev),
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
                    if _GEN_AVAILABLE and model_ok:
                        try:
                            gen = _ensure_generator()
                            text = gen.chat(user, system=system, max_new_tokens=req.max_tokens,
                                            do_sample=req.temperature > 0.01,
                                            temperature=max(req.temperature, 0.01)).text
                        except Exception:  # noqa: BLE001
                            pass
                    yield frame(text)
                    yield frame("", finish="stop")
            except Exception as e:  # noqa: BLE001
                yield sse({"error": str(e)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---- non-streaming ----
    gguf = _find_gguf(DEFAULT_MODEL_DIR) if model_ok else None
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
            print(f"[edge-infer] llama.cpp path failed, falling back: {e}")

    if _GENAI_AVAILABLE and model_ok and is_llm_dir(DEFAULT_MODEL_DIR):
        try:
            _genai.load(
                DEFAULT_MODEL_DIR,
                device=dev,
                cache_dir=os.path.join(DEFAULT_MODEL_DIR, ".npu_cache", dev),
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
            print(f"[edge-infer] genai path failed, falling back: {e}")

    # VLM path (pure OpenVINO) / degradation.
    if not _GEN_AVAILABLE or not model_ok or (gguf is not None and not is_llm_dir(DEFAULT_MODEL_DIR)):
        # Not a VLM-IR directory (e.g. GGUF-only) → don't try PureOVGenerator.
        return _resp(req.model, _degrade_text(), prompt_tokens=len(req.messages),
                     completion_tokens=-1, tok_s=0.0, degraded=True)

    gen = _ensure_generator()
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
    content: str


@app.post("/agent/upload")
def agent_upload(req: AttachmentReq):
    """Save an uploaded attachment into the workspace's uploads/ dir."""
    if not _AGENT_AVAILABLE:
        raise HTTPException(503, "agent tools unavailable")
    updir = _ws.root / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    safe = os.path.basename(req.name) or "file.txt"
    (updir / safe).write_text(req.content, encoding="utf-8")
    return {"ok": True, "path": f"uploads/{safe}", "message": f"已保存附件 {safe}"}


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

