# edge-infer

**Run very large models locally on an ordinary PC · energy-efficient on-device inference**

[![Version](https://img.shields.io/badge/version-0.3.1-2ea44f.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Free%20personal%20%7C%20Commercial-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.4-0091D5.svg)](https://docs.openvino.ai/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-CUDA%20%7C%20CPU-6E6E6E.svg)](https://github.com/ggml-org/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](#quick-start)
[![Tauri](https://img.shields.io/badge/Tauri-2.x-24C8DB.svg)](https://tauri.app/)

**English** · [简体中文](README.md)

**Goal: let an ordinary PC run very large models locally, and run them energy-efficiently.**

An ordinary Windows PC with **8 GB VRAM + 32 GB RAM** can run a **180B-class MoE**
model smoothly — via a self-built **Rust 3-tier offloading kernel** (SSD / RAM / VRAM)
plus **MTP speculative decoding**, arranged as one local pipeline.

- 🧱 **3-tier offloading** — weights are placed across SSD / RAM / VRAM by heat; big models run even without enough VRAM
- ⚡ **MTP speculative decoding** — the target verifies several proposed tokens per pass; output is unchanged, ~17% faster
- 🔌 **OpenAI-compatible API** — a standard `/v1/chat/completions` endpoint, pluggable into any agent workbench
- 🖥 **Multi-hardware backends** — NVIDIA via llama.cpp CUDA; Intel GPU / NPU via OpenVINO

**Next goal**: keep pushing local inference performance (faster decode, longer context, lower footprint).

---

## ✨ Features

- **3-tier offloading kernel (Rust)** — distributes MoE experts across
  SSD / RAM / VRAM by heat; dynamic VRAM probing (NVML); automatic OOM
  fallback (shorter context / demoted experts).
- **MTP speculative decoding** — a model's built-in MTP draft head proposes
  several candidate tokens, verified by the target in one pass (**output is
  bit-for-bit identical to running without it**); measured **+17%** generation
  speed here. Auto-enabled for models that ship a draft (e.g. Qwen3.8-Flash-Next),
  toggleable in Settings.
- **Prerouter prefetch** — predicts the experts the next token needs and pulls
  them from the slow tier to the fast tier *while* the current forward pass
  runs, hiding IO behind compute.
- **Two inference routes (pick the fastest backend per device)** — same config,
  auto-dispatched:

  | Hardware | Backend | Measured decode tok/s (Qwen3-8B Q4_K_M) |
  |---|---|---|
  | NVIDIA GPU (RTX 4070 Laptop) | **llama.cpp CUDA** | **43.3** |
  | Intel GPU (Arc iGPU) | OpenVINO (GPU.0) | 12.0 |
  | Intel NPU (AI Boost · Core Ultra 7 155H) | OpenVINO (NPU) | 6.4 |

  Conclusion: **OpenVINO's GPU plugin is tuned for Intel; handing it an NVIDIA
  card wastes roughly 19× the throughput** (measured decode 2.3 vs 43.3 tok/s).
  So NVIDIA always goes through llama.cpp CUDA, while Intel GPU / NPU go through
  OpenVINO. Large MoE (Qwen3.8-Flash-Next, 180B stored) still runs on 8 GB VRAM
  via `--n-cpu-moe` 3-tier offloading.
  (Full data, the `ncmoe` sweep and the MTP comparison are in
  [Performance](#-performance).)

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
  far beyond what OpenVINO IR can hold (e.g. Qwen3.8-Flash-Next, 180B stored);
  `--n-cpu-moe` keeps MoE layers on CPU/RAM for 3-tier offloading.
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
| llama.cpp CUDA · NVIDIA RTX 4070 Laptop 8 GB | 2381.8 | **43.3** | 1.00× |
| OpenVINO · Intel Arc iGPU | 1297.7 | 12.0 | 3.6× slower |
| OpenVINO · Intel NPU (AI Boost · Core Ultra 7 155H) | 35.0 | 6.4 | 6.8× slower |

Measured with `llama-bench` (`-p 512 -n 128`, 3 reps).
Prefill = prompt processing, decode = steady-state generation.

> Two further references (also Qwen3-8B, kept out of the main table): **CPU**
> via llama.cpp is 198.7 / 4.5 and via OpenVINO 283.3 / 3.5 tok/s;
> **NVIDIA through OpenVINO (GPU.1)** is only 26.2 / 2.3 tok/s, ~19× slower than
> CUDA — hence not recommended.

### Large MoE 3-tier offloading (the headline)

**Qwen3.8-Flash-Next (180B stored / 6B active) runs on 8 GB VRAM +
32 GB RAM** — this is the project's core scenario. `--n-cpu-moe` keeps MoE
experts on CPU/RAM while attention and dense layers stay in VRAM; the 28.8 GB
n-gram table stays on SSD via `--lazy-mode`:

| Target · quant | Backend · device | Offload | Decode tok/s |
|---|---|---|---|
| **UD-Q2_0 (65.7 GB)** | llama.cpp CUDA · RTX 4070 8 GB | `ncmoe=35` | **1.1** |
| UD-IQ1_S (66.4 GB) | same | `ncmoe=30` | 1.1 |
| UD-IQ4_XS (87 GB) | same | `ncmoe=45` | 1.2 |

> All three quantization levels decode about the same — the bottleneck is the
> **SSD paging of expert weights**, not compute or quantization precision. This
> is exactly why 3-tier offloading, not a fancier quant, is what makes big
> models runnable.

**MTP speculative decoding** (same real task, `-c 4096 -n 192`, 2 runs, averaged):

| Target | no MTP | + MTP (shared draft, n-max 2) | Gain |
|---|---|---|---|
| **UD-Q2_0** | 1.2 t/s | **1.4 t/s** | **+17%** |
| UD-IQ1_S | 1.1 t/s | 1.3 t/s | +18% |
| UD-IQ4_XS | 1.2 t/s | 1.4 t/s | +17% |

> MTP lets the draft head propose several tokens that the target verifies in a
> single pass (mathematically lossless — output matches token-for-token). The
> gain sits at **~17%**, again capped by SSD bandwidth: it cuts the number of
> decode steps but cannot avoid loading expert weights each step.
> Prefer a `shared-` draft (borrows the target's embedding — smaller and faster).

### Reading the numbers

- **NVIDIA → llama.cpp CUDA, always.** The RTX 4070 beats every other option on
  both axes. Routing it through OpenVINO (GPU.1) costs **18.8×** on decode and
  **91×** on prefill — OpenVINO's GPU plugin is built for Intel silicon.
- **Intel Arc iGPU is the strongest non-NVIDIA option** — prefill is within 1.8×
  of the 4070, and it shares system RAM, so it can host models larger than its
  dedicated budget (30B/35B MoE via Vulkan + 3-tier offloading).
- **Intel NPU (AI Boost on Core Ultra 7 155H) is for always-on, low-power
  inference**, not throughput: 6.4 tok/s at a fraction of the power. It has no
  dedicated VRAM, so it uses a SSD + RAM two-tier split.
- **iGPU/Vulkan trade-off**: for 8B on the Arc iGPU, Vulkan (140/9.2) is
  **slower** than OpenVINO (1298/12.0) — OpenVINO is Intel's native optimized
  path; for 30B/35B MoE with no IR build, Vulkan + `--n-cpu-moe` works.
- **Qwen3.8-Flash-Next is NVIDIA-only**: 66+ GB weights and no OpenVINO IR —
  the iGPU / NPU paths are not viable.

### How approaches compare

Different ways to run large models on limited VRAM involve different trade-offs:

| Approach | VRAM needed | Speed | Accuracy | Notes |
|---|---|---|---|---|
| Quantized, fully resident | High (must fit weights) | Fast | Some loss | Best when VRAM is enough |
| Layer-by-layer streaming | Low (one layer) | Slow (seconds/token) | Can stay full precision | Reads disk every token; offline use |
| **This project: 3-tier offload** | **Low (from 8 GB)** | **Moderate** | GGUF quantization | Experts resident in RAM, large tables on SSD — balances threshold and speed |

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
| **Qwen3.8-Flash-Next** ⭐ | 180B stored / 6B active | llama.cpp CUDA · GGUF (UD-Q2_0) + MTP | 65.7 + 1.8 GB | SSD + RAM + VRAM |
| **Qwen3.6-35B-A3B** | 35B-A3B · multimodal | llama.cpp CUDA · GGUF / OpenVINO · INT4 | 21.2 / 19.4 GB | SSD + RAM + VRAM |
| **Qwen3-8B** | 8B | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 5.0 / 4.7 GB | SSD + RAM (+ VRAM) |
| **MiniCPM-V-4.5** | 9B · multimodal | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 6.0 / 5.3 GB | SSD + RAM (+ VRAM) |

- ⭐ **Default flagship combo: Qwen3.8-Flash-Next (UD-Q2_0) + shared-Q4_K_M MTP
  draft** — a 180B-stored flagship MoE that runs on 8 GB VRAM / 32 GB RAM with
  MTP speculative decoding enabled automatically.
- **NVIDIA** — always **llama.cpp CUDA** (43.3 tok/s for 8B; OpenVINO only 2.3).
  From 35B-A3B to 180B-stored Flash-Next: even 8 GB of VRAM works, thanks to
  3-tier offloading.
- **Intel GPU** — **OpenVINO** (~1.8× faster than Vulkan on Arc); 30B / 35B MoE
  can all be offloaded.
- **NPU** — uses system RAM (no dedicated VRAM tier), so it runs a
  **SSD + RAM two-tier split**; 9B multimodal (MiniCPM-V-4.5, symmetric INT4) is verified.

> **One model, format auto-switched per device**: the same model name resolves to
> GGUF on NVIDIA and OpenVINO IR on Intel GPU / NPU — handled by the
> *model → variant* structure plus `resolve_device()` in `model_catalog.py`.
> No manual format picking.
>
> Note: GGUF models under the llama.cpp backend use `--n-cpu-moe N` to keep N
> MoE layers on CPU/RAM while the rest stay on the GPU — a natural 3-tier
> offload path. OpenVINO IR models get their tiering planned by the Rust kernel.

---

## 🧭 3-tier offload planner

Before downloading a 60 GB+ model, know whether it will fit. `tier_plan.py`
reads **only the GGUF header and tensor table** (never the weights), so it
answers instantly even for models far larger than RAM:

```bash
cd python
python -m edge_server.tier_plan <model.gguf> --vram 8 --ram 32
```

It reports per-group sizes (n-gram table / MoE experts / attention / other) and
a per-group tier recommendation (VRAM / RAM / SSD), flagging when SSD must join.

HTTP: `GET /edge/tier_plan?model=<dir-or-path>&vram=8&ram=32`

---

## 🚀 Very large MoE: Qwen3.8-Flash-Next (180B stored / 6B active)

**A 66 GB model on 8 GB VRAM + 32 GB RAM, keeping 256K context.**

Its architecture (`qwen4exp`) has three structural traits that fit 3-tier
offloading unusually well:

| Trait | Benefit |
|---|---|
| **48 layers = 36 GDN + 12 QSA**; only QSA layers keep a KV cache | **256K context needs just 2.67 GB** KV (f16) — 1/4 of a dense model its size |
| **The n-gram table is a lookup, not a matmul** (16 rows per token) | Its 28.8 GB can live on **SSD**; `--lazy-mode auto` reads rows on demand |
| **512 experts, only 10 active** (2%) | Expert weights tier by heat |

Tier allocation:

| Tier | Capacity | Holds |
|---|---|---|
| VRAM | 8 GB | attention + norm + output + some experts |
| RAM | 32 GB | backbone + hot experts |
| **SSD** | — | **the 28.8 GB n-gram table** (lazy, not resident) + cold experts |

Recommended combo **UD-Q2_0 + shared-Q4_K_M MTP draft** (set up automatically in-app):

```bash
export LLAMA_ATTN_ROT_DISABLE=1     # required by qwen4exp

llama-server -m .../Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf \
  -md .../MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 999 -ncmoe 35 --tensor-split 1 -c 32768 -fa on
```

> `-ncmoe 35` is the stable sweet spot. `--tensor-split 1` is required when an
> MTP draft is loaded (otherwise the second model's free-VRAM probe sees 0 and
> loading fails). Measured **+17%** generation with MTP on 8 GB VRAM / 32 GB RAM.

> Key mechanism: this build's `--lazy-mode` defaults to `auto`, which reads
> **any tensor larger than 4 GiB** on demand — the 28.8 GB n-gram table
> qualifies, so it stays on disk out of the box.

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

## 🪟 Windows installer (no Python required)

Don't want to wrangle an environment? Grab **`edge-infer-<version>-setup.exe`**
from GitHub Releases and double-click — **the target machine needs no Python
installed**.

What the installer ships:

| Item | Notes |
|---|---|
| `edge-gui.exe` | Tauri native window (auto-starts the backend) |
| Rust CLI kernel | `edge-detect` / `edge-engine` / `edge-tiers`, … |
| **Portable Python** | A self-contained embeddable Python runtime + all deps (`fastapi`/`uvicorn`/`openvino` …) under `runtime\python` — **does not touch your system Python** |
| `models\` dir | Created automatically; weights are *not* bundled — download them in the app's *Models* page |
| Shortcuts | Desktop + Start menu |

### Will a new machine lose the Python environment?

**Not with the installer.** The key is how the app resolves its interpreter:

- **Old (run from source)** — the app only searched the system `PATH` for
  `python.exe`. On a different machine without Python on `PATH` it failed with
  "no Python interpreter found".
- **New (installer)** — `find_python()` now **prefers the portable runtime
  inside the app folder** (`runtime\python\python.exe`) and automatically sets
  `PYTHONPATH` to its bundled `Lib\site-packages`. The deps travel with the app,
  so machine or directory changes don't matter. Search order:
  **bundled runtime → project `.venv` → user-level Python → system `PATH`**.

> Why the old version broke: a GUI launched by double-clicking inherits the
> *system* PATH, not the PATH you exported in a terminal. So even if `python`
> works in your shell, the double-clicked exe may not find it — which is exactly
> why the bundled runtime now sits first in the search chain.

### Building the installer yourself

```powershell
# requires: Rust, Node (Tauri), internet, Inno Setup 6
powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1
# output: dist\edge-infer-<version>-setup.exe
```

The script builds Rust/Tauri → downloads embeddable Python → `pip install`s the
deps into that runtime → stages the backend + exes → runs Inno Setup.

---

## 🔄 Update checking

*Settings → About* has a **Check for updates** button: it queries the latest
GitHub Release and either

- shows `new version x.y.z available (current a.b.c) · download` linking to the
  installer, or
- says `up to date`, or
- degrades quietly to "check failed (offline?)" when there's no network.

Under Tauri it calls the native `check_update` command (PowerShell → GitHub API);
in a browser it calls the GitHub API via `fetch`. Both use the same semver
segment comparison (tolerating a `v` prefix).

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

Exposes a **standard OpenAI interface**, so it can plug into **any agent
workbench / client that speaks the OpenAI standard** (local or remote), turning
this machine's 3-tier offloading compute into an OpenAI endpoint.

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

## 🩺 Diagnostics & problem recording

Running on a machine **other than the dev box** (especially an **AMD platform**)
means a failure often cannot be reproduced locally, so the product can
**explain itself**:

- **One-shot diagnostics** — 10 independent probes covering CPU / GPU / NPU /
  drivers / RAM / backends / model dirs; every probe records *why* it failed
  instead of silently degrading. Outputs structured JSON + a readable summary.
- **Problem log** — backend-load failures, inference errors and download
  failures are appended to `logs/problems.jsonl` with a traceback and the
  **environment fingerprint** at that moment.
- **Capability assessment** — states plainly which paths are usable on *this*
  machine, and why the others are not.

```bash
cd python
python -m edge_server.diagnostics            # summary
python -m edge_server.diagnostics --bundle   # zip to send back
```

In the GUI: *Settings → Diagnostics* — refresh, browse problems, export a bundle.

> Sample output on this dev box (Intel + NVIDIA):
> ```
> NVIDIA CUDA : YES  NVIDIA GeForce RTX 4070 Laptop GPU
> Intel GPU   : YES  (OpenVINO)
> Intel NPU   : YES  OpenVINO NPU plugin exposes the device
> AMD NPU     : not found  -> supported: NO
> AMD GPU     : not found  -> via ['(no Vulkan/ROCm backend)']
> ```

Probes run **concurrently** and a failure in one never aborts the others.

**Privacy**: API keys / tokens / `sk-*` / `github_pat_*` are redacted; no prompts
or file contents are ever captured.

---

## 🔵 AMD platform compatibility

v0.3.0 supports **NVIDIA GPU** and **Intel GPU / NPU**. On AMD the product
**falls back to CPU safely** (no crash, but no acceleration).

| Hardware | Status |
|---|---|
| AMD Radeon dGPU | Can be accelerated via llama.cpp Vulkan |
| AMD Radeon iGPU | Supported, modest gain |
| AMD XDNA NPU (Ryzen AI) | Not supported yet |

OpenVINO's GPU / NPU plugins are Intel-only and cannot drive AMD XDNA NPUs;
support is on the roadmap.

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
| `EDGE_LLAMA_DIR` | `D:\AI\tools\llama` | Directory holding `llama-server.exe` (point it at a Vulkan/ROCm build on AMD) |
| `EDGE_N_CPU_MOE` | `0` | MoE layers kept on CPU/RAM (3-tier offload knob) |
| `EDGE_CTX` | `32768` | llama.cpp context window (max 1048576) |
| `EDGE_KV_QUANT` | auto | KV cache quantisation; `q8_0` auto-applied at ≥64K ctx |
| `EDGE_MTP_DRAFT` | auto | MTP draft model path; blank = auto-find under sibling `MTP/` |
| `EDGE_LOG_LEVEL` | `INFO` | Log level: `DEBUG` / `INFO` / `WARNING` |
| `EDGE_DIAG_BOOT` | `0` | `1` = write a diagnostics bundle at startup |

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
│   │   ├── diagnostics.py        env diagnostics + problem log + bundle export
│   │   ├── tier_plan.py          GGUF 3-tier planner (header-only)
│   │   ├── logging_setup.py      unified logging (rotating file + excepthooks)
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
│   └── run_server.py
├── logs/                         runtime logs (diagnostics / problems.jsonl)
├── tauri-app/                    Tauri native shell
├── run_windows.bat               one-click launch
├── CHANGELOG.md
└── LICENSE
```

---

## ✅ Verification

```bash
cargo test    # Rust kernel unit tests
```

---

## 📄 License

**Free for personal use; commercial use requires a license** — see [LICENSE](LICENSE).

Personal learning, research, and non-profit use is free, including modification.
For commercial use within an organization, or for inclusion in a paid product or
service, please contact the author via [GitHub Issues](https://github.com/hellovm/edge-infer/issues)
to obtain a commercial license.

© 2026 David
