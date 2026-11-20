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
    return {
        "ok": True,
        "backend_available": _GEN_AVAILABLE,
        "generator_ready": _gen is not None,
        "device": _gen_device,
        "model_dir": DEFAULT_MODEL_DIR,
        "model_dir_exists": os.path.isdir(DEFAULT_MODEL_DIR),
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


@app.get("/edge/models")
def edge_models():
    """List locally installed models under models/."""
    out = []
    if os.path.isdir(_MODELS_DIR):
        for name in sorted(os.listdir(_MODELS_DIR)):
            d = os.path.join(_MODELS_DIR, name)
            if not os.path.isdir(d):
                continue
            # Detect an OpenVINO IR layout.
            #   plain LLM -> openvino_model.xml    (NPU-able with sym int4)
            #   VLM       -> openvino_language_model.xml (CPU/GPU only)
            is_llm = os.path.exists(os.path.join(d, "openvino_model.xml"))
            is_vlm = os.path.exists(os.path.join(d, "openvino_language_model.xml"))
            size = 0
            for root, _dirs, files in os.walk(d):
                for f in files:
                    try:
                        size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            out.append({
                "id": name,
                "note": "OpenVINO IR (LLM)" if is_llm else (
                    "OpenVINO IR (VLM)" if is_vlm else "其他"),
                "size": f"{size / 1e9:.2f} GB",
                "path": d,
                "npu_compatible": is_llm,  # plain LLM IR + symmetric int4 -> NPU
            })
    return {"models": out}


@app.post("/edge/download")
def edge_download(repo: str):
    """Kick off a ModelScope download in the background. Returns immediately."""
    import threading

    def _run():
        try:
            from modelscope import snapshot_download

            short = repo.split("/")[-1].lower()
            local = os.path.join(_MODELS_DIR, short)
            snapshot_download(repo, local_dir=local)
        except Exception as e:  # noqa: BLE001
            print(f"[edge-infer] download failed for {repo}: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
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


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)

    system = next((m.content for m in req.messages if m.role == "system"), "")
    user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not user:
        user = "\n".join(m.content for m in req.messages)

    model_ok = os.path.isdir(DEFAULT_MODEL_DIR)

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

            def frame(delta: str, finish=None) -> str:
                return sse({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": ({"content": delta} if delta else {}),
                                 "finish_reason": finish}],
                })

            use_genai = _GENAI_AVAILABLE and model_ok and is_llm_dir(DEFAULT_MODEL_DIR)
            try:
                if use_genai:
                    _genai.load(
                        DEFAULT_MODEL_DIR, device=DEFAULT_DEVICE,
                        cache_dir=os.path.join(DEFAULT_MODEL_DIR, ".npu_cache", DEFAULT_DEVICE),
                    )
                    got = False
                    for delta in _genai.stream_chat(
                        user, system=system, max_new_tokens=req.max_tokens,
                        temperature=req.temperature,
                    ):
                        got = True
                        yield frame(delta)
                    if not got:
                        yield frame("")
                    yield frame("", finish="stop")
                else:
                    # Non-streaming backends: emit the whole answer as one chunk.
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
    # Prefer the NPU/GenAI LLMPipeline path when the model is a plain LLM IR.
    if _GENAI_AVAILABLE and model_ok and is_llm_dir(DEFAULT_MODEL_DIR):
        try:
            _genai.load(
                DEFAULT_MODEL_DIR,
                device=DEFAULT_DEVICE,
                cache_dir=os.path.join(DEFAULT_MODEL_DIR, ".npu_cache", DEFAULT_DEVICE),
            )
            r = _genai.chat(
                user, system=system,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            return _resp(req.model, r.text, prompt_tokens=r.prompt_tokens,
                         completion_tokens=r.completion_tokens, tok_s=r.est_tok_s,
                         device=r.device)
        except Exception as e:  # noqa: BLE001
            print(f"[edge-infer] genai path failed, falling back: {e}")

    # VLM path (pure OpenVINO) / degradation.
    if not _GEN_AVAILABLE or not model_ok:
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

    system = ("你是一个本地文件助手，可以创建任务文件夹、读写文件。\n"
              + tools_prompt())
    ctx = req.message
    steps = []
    for _ in range(req.max_steps):
        resp = _genai.chat(ctx, system=system, max_new_tokens=req.max_tokens, temperature=0.0)
        text = resp.text
        call = parse_tool_call(text)
        if not call:
            return {"steps": steps, "answer": text, "workspace": str(_ws.root)}
        name = call.get("name")
        args = call.get("arguments", {}) or {}
        result = run_tool(_ws, name, args)
        steps.append({"tool": name, "arguments": args, "result": result})
        # feed the result back and let the model continue
        ctx = (f"{ctx}\n\n[你调用了 {name}，结果如下]\n<tool_result>\n{result}\n</tool_result>\n"
               f"请继续：如已完成请直接给出最终答复（不要再包含 <tool_call>）。")
    return {"steps": steps, "answer": "（达到最大步骤数）", "workspace": str(_ws.root)}


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

