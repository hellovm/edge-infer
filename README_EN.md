# edge-infer

**On-device 3-tier offloading inference platform · local AI workbench**

[![Version](https://img.shields.io/badge/version-0.2.1-2ea44f.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.4-0091D5.svg)](https://docs.openvino.ai/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-CUDA%20%7C%20CPU-6E6E6E.svg)](https://github.com/ggml-org/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](#quick-start)
[![Tauri](https://img.shields.io/badge/Tauri-2.x-24C8DB.svg)](https://tauri.app/)

**English** · [简体中文](README.md)

Run local LLMs on Windows: a **self-built Rust 3-tier offloading kernel**
(SSD / RAM / VRAM) + **two inference backends** (NVIDIA → llama.cpp CUDA;
Intel GPU / NPU → OpenVINO) + a **local agent workbench** that can actually
touch the filesystem, scaffold projects and run commands.

> Goal: let a Windows laptop that "can't run big models" run them anyway,
> with longer context. Kernel and scheduler are all implemented from scratch.

---

## ✨ Features

- **3-tier offloading kernel (Rust)** — distributes MoE experts across
  SSD / RAM / VRAM by heat; dynamic VRAM probing (NVML); automatic OOM
  fallback (shorter context / demoted experts).
- **Prerouter prefetch** — predicts the experts the next token needs and pulls
  them from the slow tier to the fast tier *while* the current forward pass
  runs, hiding IO behind compute.
- **Two inference routes (pick the fastest backend per device)** — same config,
  auto-dispatched:

  | Hardware | Backend | Measured tok/s (Qwen3-8B Q4_K_M) |
  |---|---|---|
  | NVIDIA GPU (RTX 4070 Laptop) | **llama.cpp CUDA** | **43.3** |
  | Intel GPU (Arc iGPU) | OpenVINO (GPU.0) | 12.0 |
  | Intel NPU (AI Boost) | OpenVINO (NPU) | 6.4 |
  | CPU | llama.cpp | 4.5 |
  | CPU | OpenVINO | 3.5 |
  | NVIDIA via OpenVINO (GPU.1) | *(not recommended)* | 2.3 ← ~19× slower |

  Conclusion: **OpenVINO's GPU plugin is tuned for Intel; handing it an NVIDIA
  card wastes roughly 19× the throughput.** So NVIDIA always goes through
  llama.cpp CUDA, while Intel GPU / NPU go through OpenVINO.
  (Same-run prefill/decode numbers are in [Performance](#-performance).)

- **One model, format auto-matched to the device** — the catalog is organised
  as *model → variants*: a single model name carries `nvidia` / `intel-gpu` /
  `npu` variants, and the matching format is selected from the active hardware
  (NVIDIA → **GGUF**, Intel GPU / NPU → **OpenVINO IR**). A mismatched
  combination is called out explicitly instead of silently degrading.

- **Pure OpenVINO inference path** — no torch / transformers, only `openvino` +
  `tokenizers` + `numpy`, dodging version hell (the VLM path).
- **OpenVINO GenAI backend** — `LLMPipeline` drives NPU / Intel GPU / CPU, with
  **SSE streaming**.
- **llama.cpp backend** — runs GGUF models (NVIDIA CUDA path) and covers MoE
  far beyond what OpenVINO IR can hold (80B-A3B and up); `--n-cpu-moe` keeps
  MoE layers on CPU/RAM for 3-tier offloading.
- **Quantization notes** — FP4/NVFP4 needs Blackwell (RTX 50, sm120) FP4 tensor
  cores; **RTX 40 (Ada, sm89) has no FP4 datapath**, so NVFP4 degrades to
  software emulation (often slower). For Ada, the right choices are **FP8**
  (via vLLM/TensorRT-LLM) or **GGUF Q4_K_M / Q5_K_M**.
- **Local agent** — the model can call tools: create task folders, read/write
  files, search, run commands — all sandboxed inside the workspace.
- **Web search** — a built-in Playwright driver reuses your local Edge to fetch
  search results and inject them into context (toggleable).
- **OpenAI-compatible API** — exposes `/v1/chat/completions` (including
  `stream=true`), callable from any client; optional API-key auth.
- **Workbench GUI** — five views (Chat / Agent workspace / 3-tier plan / Models
  / Settings); a clickable device chip in the header; a composer with model
  picker, file / image / folder upload and a web-search toggle; persisted
  conversations; switch between the local engine and an external OpenAI API.
- **One-click launch** — the Tauri native window (a 6.7 MB exe, not a browser
  shell) starts the Python inference service on launch and reclaims it on exit,
  leaving no stray processes.

---

## ❗ NPU notes (important)

The NPU path is verified on **Intel Core Ultra Series 1 (Meteor Lake, NPU3720)**.
That NPU compiler **only supports symmetric INT4** (channel-wise / group-wise)
and **not asymmetric INT4 (AWQ)**:

```
Unsupported DynamicDequantizeOp configuration:
  inElemType=!quant.uniform<u4:f16, ...>, arch=NPU37XX
```

So running on NPU requires both:

1. **A symmetrically quantized model** — export with
   `optimum-cli export openvino --weight-format int4 --sym --group-size 128`,
   or just use a `*-int4-cw-ov` repo on ModelScope;
2. **Driving it with `openvino_genai.LLMPipeline`** — it handles the NPU's
   static-shape requirement internally; optimum-intel's
   `OVModelForCausalLM` forces dynamic shapes, which the NPU rejects.

Models verified to run on NPU:

| Model | Size | Device | Notes |
|---|---|---|---|
| `OpenVINO/Qwen3-8B-int4-cw-ov` | 4.7 GB | NPU | symmetric INT4 · fast load |
| `OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov` | 1.1 GB | NPU | ~19 tok/s |
| `OpenVINO/MiniCPM-V-4_5-int4-cw-ov` | 5.3 GB | NPU | multimodal · 9B class |

> Qwen3 defaults to thinking mode (slow); the platform appends `/no_think` to
> suppress it. Set `EDGE_ENABLE_THINKING=1` for deep reasoning.
>
> `Qwen3-VL-4B-*-int4-ov` (AWQ, asymmetric) **cannot run on NPU** — CPU/GPU only.

---

## 📊 Performance

Measured on this machine: **Intel Core Ultra 7 155H** (16 threads) +
**Intel Arc iGPU** + **Intel AI Boost NPU (NPU3720)** +
**NVIDIA RTX 4070 Laptop 8 GB (sm_89)** + **32 GB DDR5-5600**, Windows 11.
Toolchain: llama.cpp `b12559` (CUDA 12), OpenVINO `2026.4.1`.

Model: **Qwen3-8B Q4_K_M (GGUF)** / **Qwen3-8B int4-cw (OpenVINO IR)** —
the same model, in the format each device prefers.

### Decode throughput (token generation)

| Backend · Device | Prefill tok/s | **Decode tok/s** | Relative |
|---|---|---|---|
| llama.cpp CUDA · RTX 4070 8 GB | 2381.8 | **43.3** | 1.00× |
| OpenVINO · Intel Arc iGPU | 1297.7 | 12.0 | 3.6× slower |
| OpenVINO · Intel NPU | 35.0 | 6.4 | 6.8× slower |
| llama.cpp · CPU (16 threads) | 198.7 | 4.5 | 9.6× slower |
| OpenVINO · CPU | 283.3 | 3.5 | 12.4× slower |
| OpenVINO · NVIDIA (GPU.1) | 26.2 | **2.3** | **18.8× slower** |

Measured with `llama-bench` (`-p 512 -n 128`, 3 reps) and `scripts/bench_ov.py`
(531-token prompt, 128 generated tokens, 2 reps). Prefill = prompt processing,
decode = steady-state generation.

### Large MoE 3-tier offloading (the headline)

**An 80B model runs on 8 GB VRAM + 32 GB RAM** — this is the project's core
scenario. `Qwen3-Next-80B-A3B` (48.5 GB, 80B total / 3B active) uses
`--n-cpu-moe` to keep MoE layers on CPU/RAM while attention and dense layers
stay in VRAM:

| Backend · Device | Offload config | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|---|
| llama.cpp CUDA · RTX 4070 8 GB | `ncmoe=48` | **34.2** | **9.5** | best; ~48/64 MoE layers in RAM |
| llama.cpp Vulkan · Intel Arc | `ncmoe=48` | 6.4 | 3.9 | 15.5 GB shared-memory ceiling |
| ~~llama.cpp CUDA · `ngl=999` full offload~~ | — | — | — | ❌ immediate OOM (needs 46 GB VRAM) |

`--n-cpu-moe` sweep (NVIDIA, `-p 128 -n 64`):

| ncmoe | Prefill tok/s | Decode tok/s |
|---|---|---|
| 30 | 4.3 | 3.0 |
| 40 | 3.6 | 3.6 |
| 44 | 9.8 | 3.9 |
| **48** | **10.8** | **7.3** ← best |
| 52 | 12.6 | 7.0 |
| 56 | 15.2 | 6.2 |
| 60 | 13.3 | 4.7 |

> More offload means less VRAM pressure but the CPU-side bandwidth becomes the
> bottleneck — around 48 layers is the sweet spot ("fit every layer that fits in
> VRAM, spill the rest to RAM"). **Going 8B → 80B is 10× the parameters, yet
> decode only drops from 43.3 to 9.5 tok/s (~4.6×)** — that is exactly the value
> of MoE plus 3-tier offloading.

### Reading the numbers

- **NVIDIA → llama.cpp CUDA, always.** The RTX 4070 beats every other option on
  both axes. Routing it through OpenVINO (GPU.1) costs **18.8×** on decode and
  **91×** on prefill — OpenVINO's GPU plugin is built for Intel silicon.
- **Intel Arc iGPU is the strongest non-NVIDIA option** — prefill is within 1.8×
  of the 4070, and it shares system RAM, so it can host models larger than its
  dedicated budget (80B runs too, via Vulkan + 3-tier offloading).
- **NPU is for always-on, low-power inference**, not throughput: 6.4 tok/s at a
  fraction of the power. It has no dedicated VRAM, so it uses a SSD + RAM
  two-tier split.
- **Prefill vs decode diverge sharply on CPU.** CPUs vectorize prompt
  processing well (198–283 tok/s) but are memory-bandwidth-bound per generated
  token, hence the 4.5 tok/s decode.
- **iGPU/Vulkan trade-off**: for 8B on the Arc iGPU, Vulkan (140/9.2) is
  **slower** than OpenVINO (1298/12.0) — OpenVINO is Intel's native optimized
  path. But no OpenVINO IR exists for 80B, so Vulkan + `--n-cpu-moe` is the only
  way to run it there.

---

## 📦 Recommended models (by hardware · MoE-first)

The project's value is **making models that otherwise won't fit actually run**,
so the recommendations are **MoE (mixture-of-experts)** models — their per-token
working set is just the *active* parameters (e.g. 35B-A3B activates 3B), which
is exactly what 3-tier offloading exploits.

The workbench's *Menu → Models* page shows the same list; **pick a device and it
auto-matches the format, then downloads on click**.

| Model | Params | Backend · format | Size | Tiers |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | 35B-A3B · multimodal | llama.cpp CUDA · GGUF / OpenVINO · INT4 | 21.2 / 19.4 GB | SSD + RAM + VRAM |
| **Qwen3-30B-A3B** | 30B-A3B | llama.cpp CUDA · GGUF / OpenVINO · INT4 | 18.6 / 16.3 GB | SSD + RAM + VRAM |
| **Qwen3-Next-80B-A3B** | 80B-A3B | llama.cpp CUDA · GGUF | 48.5 GB | SSD + RAM + VRAM |
| **Qwen3-8B** | 8B | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 5.0 / 4.7 GB | SSD + RAM (+ VRAM) |
| **MiniCPM-V-4.5** | 9B · multimodal | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 6.0 / 5.3 GB | SSD + RAM (+ VRAM) |

- **NVIDIA** — always **llama.cpp CUDA** (43.3 tok/s for 8B; OpenVINO only 2.3).
  From 35B-A3B to 80B-A3B: even 8 GB of VRAM works, thanks to 3-tier offloading.
- **Intel GPU** — **OpenVINO** (~1.8× faster than Vulkan on Arc); 30B / 35B MoE
  can all be offloaded.
- **NPU** — uses system RAM (no dedicated VRAM tier), so it runs a
  **SSD + RAM two-tier split**; 9B multimodal (MiniCPM-V-4.5, symmetric INT4)
  is verified, 30B-class MoE validation is ongoing.

> **One model, format auto-switched per device**: the same model name resolves to
> GGUF on NVIDIA and OpenVINO IR on Intel GPU / NPU — handled by the
> *model → variant* structure plus `resolve_device()` in `model_catalog.py`.
> No manual format picking.
>
> Note: GGUF models under the llama.cpp backend use `--n-cpu-moe N` to keep N
> MoE layers on CPU/RAM while the rest stay on the GPU — a natural 3-tier
> offload path. OpenVINO IR models get their tiering planned by the Rust kernel.

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────┐
│  GUI (Tauri native window / browser workbench)              │
│  Chat · Agent workspace · 3-tier plan · Models · Settings    │
└───────────────────────┬────────────────────────────────────┘
                        │  OpenAI-compatible HTTP
                        ▼
┌────────────────────────────────────────────────────────────┐
│  FastAPI service  python/edge_server/openai_server.py       │
│  /v1/chat/completions   inference (SSE streaming)           │
│  /agent/chat            local agent tool loop               │
│  /edge/devices          real hardware probe (incl. SSD BW)  │
│  /edge/run_plan         calls the Rust kernel for a plan    │
│  /edge/catalog          device-matched model variants       │
│  /edge/download(s)      model download & status             │
└───┬──────────────┬───────────────┬─────────────────────────┘
    │              │               │
    ▼              ▼               ▼
┌────────────┐ ┌──────────────┐ ┌────────────────────────────┐
│ llama_     │ │ genai_       │ │ pure_ov_gen.py             │
│ backend.py │ │ backend.py   │ │ hand-written OpenVINO VLM   │
│ GGUF →     │ │ LLMPipeline  │ │ inference                   │
│ llama.cpp  │ │ NPU/GPU/CPU  │ │ (no torch/transformers)     │
│ (CUDA)     │ │              │ │                             │
└────────────┘ └──────────────┘ └────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  Rust kernel  crates/                                       │
│  edge-detect    hardware probe (GPU/NPU/CPU/SSD)             │
│  edge-tiers     3-tier (SSD/RAM/VRAM) heat allocator         │
│  edge-prefetch  Prerouter prefetch engine                    │
│  edge-engine    assembly layer + CLI                         │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick start

### Prerequisites

- Windows 11
- Rust 1.75+ (GNU or MSVC toolchain)
- Python 3.10+
- Node.js 18+ (only to build the Tauri native window)

### 1. One-click launch (recommended)

```bat
run_windows.bat
```

Starts the Python inference service, then opens the Tauri native window; falls
back to the browser if the native window isn't built.

### 2. Manual start

```bash
# install Python deps
cd python
pip install -r requirements.txt

# fetch an NPU-compatible model (symmetric quantization)
python -m edge_server.download_modelscope OpenVINO/Qwen3-8B-int4-cw-ov

# start the service (default device=NPU, listening on 0.0.0.0:8080)
EDGE_DEVICE=NPU EDGE_MODEL_DIR=../models/qwen3-8b-int4-cw python run_server.py
```

Open <http://127.0.0.1:8080/ui/> for the workbench.

### 3. Build the Rust kernel

```bash
cargo build --release          # -> target/release/edge-engine.exe
cargo test                     # unit tests
```

### 4. Build the Tauri native window

```bash
# on this machine: MinGW-w64 (winlibs), unpacked to C:\mingw64
export PATH="/c/mingw64/mingw64/bin:$HOME/.cargo/bin:$PATH"
cd tauri-app/src-tauri
cargo build --release          # -> target/release/edge-gui.exe
```

---

## 🤖 Local agent (project-style development)

The *Projects* and *Agent workspace* tabs let you give development tasks in
natural language; the model calls tools and **actually creates projects on disk,
writes code, sets up sandboxes, installs dependencies and runs programs**.

Available tools:

| Tool | Description |
|---|---|
| `create_project` | Create a project folder (`src/data/output/notes`), empty/python template |
| `open_project` / `list_projects` | Open / list projects |
| `sandbox_setup` | Create a Python venv sandbox inside the project (`.venv/`) |
| `sandbox_run` | Run commands in the sandbox (`pip install ...` / `python xxx.py`) |
| `read_file` / `write_file` | Read / write files |
| `edit_file` / `append_file` | Find-replace / append |
| `list_dir` / `make_dir` / `search` | Directories and search |
| `run_command` / `cd` / `get_workspace` | Dev commands / change dir / workspace info |

**Typical workflow** (verified):

```bash
curl -X POST http://127.0.0.1:8080/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Create a Python project named Fibonacci that prints the first 10 Fibonacci numbers and run it","max_steps":6}'
```

The agent will `create_project` → `write_file` → `sandbox_setup` → `sandbox_run`,
and a real project folder appears on disk with the program's output.

The GUI *Projects* view offers a project list, file tree, file editor
(open/save) and a sandbox terminal.

> All operations are confined to the `workspace/` sandbox; out-of-bounds paths
> are rejected. The agent needs decent instruction following — **8B recommended**
> (1.5B is too weak).

---

## 🌐 OpenAI-compatible API

```bash
# non-streaming
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}],"temperature":0}'

# streaming (SSE)
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Or with the official OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen", messages=[{"role": "user", "content": "hello"}])
print(resp.choices[0].message.content)
```

---

## ⚙️ Environment variables

| Variable | Default | Description |
|---|---|---|
| `EDGE_DEVICE` | `NPU` | Inference device: `NPU` / `CPU` / `GPU.0` |
| `EDGE_MODEL_DIR` | auto | Model directory |
| `EDGE_API_KEY` | empty | API auth (empty = disabled) |
| `EDGE_PORT` | `8080` | Listen port |
| `EDGE_WORKSPACE` | `./workspace` | Agent sandbox root |
| `EDGE_ENABLE_THINKING` | `0` | Whether Qwen3 runs in thinking mode |
| `EDGE_LLAMA_DIR` | `D:\AI\tools\llama` | Directory holding `llama-server.exe` |
| `EDGE_N_CPU_MOE` | `0` | MoE layers kept on CPU/RAM (3-tier offload knob) |

---

## 📁 Layout

```
edge-infer/
├── crates/                       self-built Rust kernel (4 crates)
│   ├── edge-detect/              hardware probe (GPU/NPU/CPU/SSD)
│   ├── edge-tiers/               3-tier (SSD/RAM/VRAM) allocator
│   ├── edge-prefetch/            Prerouter prefetch engine
│   └── edge-engine/              assembly layer + CLI
├── python/
│   ├── edge_server/
│   │   ├── openai_server.py      FastAPI OpenAI-compatible service
│   │   ├── llama_backend.py      llama.cpp backend (GGUF / CUDA 3-tier offload)
│   │   ├── genai_backend.py      OpenVINO GenAI backend (LLMPipeline)
│   │   ├── pure_ov_gen.py        pure OpenVINO VLM inference
│   │   ├── model_catalog.py      device-matched catalog (model -> variant)
│   │   ├── web_search.py         web search (Playwright + local Edge)
│   │   ├── agent_tools.py        local agent tools + sandbox
│   │   ├── storage_probe.py      SSD model + bandwidth probe
│   │   ├── download_modelscope.py ModelScope downloader
│   │   └── static/index.html     workbench GUI
│   ├── requirements.txt
│   ├── run_server.py
│   └── verify_4b.py
├── scripts/
│   ├── bench_llama.py            llama.cpp benchmark (llama-bench wrapper)
│   └── bench_ov.py               OpenVINO benchmark (prefill / decode split)
├── tauri-app/                    Tauri native shell
├── run_windows.bat               one-click launch
├── CHANGELOG.md
└── LICENSE
```

---

## ✅ Verification

```bash
cd python && python verify_4b.py    # device probe + OpenAI contract + real inference
cargo test                          # Rust kernel unit tests

# performance
python scripts/bench_llama.py models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf --ngl 999
python scripts/bench_ov.py models/qwen3-8b-int4-cw-ov NPU

# 80B 3-tier offloading
D:/AI/tools/llama/llama-bench.exe \
  -m models/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 999 -ncmoe 48 -p 512 -n 128 -r 2
```

---

## 📄 License

[Apache-2.0](LICENSE) © 2026 David
