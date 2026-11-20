# edge-infer

**端侧三层卸载多硬件推理平台 · 本地 AI 工作台**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.4-0091D5.svg)](https://docs.openvino.ai/)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](#快速开始)
[![Tauri](https://img.shields.io/badge/Tauri-2.x-24C8DB.svg)](https://tauri.app/)

在 Windows 上运行本地大模型：**Rust 自研三层卸载内核**（SSD / 内存 / 显存）+
**OpenVINO 推理服务**（NVIDIA GPU / Intel NPU / CPU）+ **本地
Agent 工作台**（可真正操作文件系统、创建项目文件夹、执行命令）。

> 目标：让原本跑不动大模型的 Windows 笔记本也能跑起来，并支持更长的上下文。
> 从内核到调度全部独立自主实现。

---

## ✨ 特性

- **三层卸载内核（Rust）** — 把 MoE 专家按热度分布在 SSD / RAM / VRAM 三层，
  显存动态探测（NVML），OOM 自动回退（缩短上下文 / 降级专家）。
- **Prerouter 预取** — 一步前瞻预测下一 token 要用的专家，在当前前向计算时
  并行把它从慢层拉到快层，用计算掩盖 IO。
- **多硬件加速** — NVIDIA GPU（CUDA，大 MoE 三层）、Intel NPU（AI Boost，
  小模型）、CPU 回退；NPU→CPU 自动降级。
- **纯 OpenVINO 推理链路** — 不依赖 torch / transformers，只用 `openvino` +
  `tokenizers` + `numpy`，绕开版本地狱（VLM 路径）。
- **OpenVINO GenAI 后端** — `LLMPipeline` 驱动 NPU/GPU/CPU，支持 **SSE 流式输出**。
- **llama.cpp 后端** — 运行 GGUF 模型，覆盖 OpenVINO IR 无法承载的更大 MoE
  （80B-A3B 及以上）；用 `--n-cpu-moe` 把 MoE 层放到 CPU/内存实现三层卸载。
- **本地 Agent** — 模型可自主调用工具：创建任务文件夹、
  读写文件、搜索、执行命令；全部限制在沙箱工作区内。
- **OpenAI 兼容 API** — 对外暴露 `/v1/chat/completions`（含 `stream=true`），
  可被任意客户端调用；支持 API Key 鉴权。
- **工作台 GUI** — 对话 / Agent 工作区 / 三层方案 / 模型 / 设置
  五个视图；会话列表持久化；可切换本地引擎 or 外接 OpenAI API。
- **Tauri 原生窗口** — 6.7MB 原生 exe，非浏览器外壳。

---

## ❗ NPU 使用说明（重要）

本项目的 NPU 路径在 **Intel Core Ultra Series 1（Meteor Lake，NPU3720）**
上实测通过。该 NPU 的编译器**只支持对称 INT4**（channel-wise / group-wise），
**不支持非对称 INT4（AWQ）**：

```
Unsupported DynamicDequantizeOp configuration:
  inElemType=!quant.uniform<u4:f16, ...>, arch=NPU37XX
```

因此跑 NPU 必须同时满足两个条件：

1. **对称量化模型** — 用 `optimum-cli export openvino --weight-format int4 --sym
   --group-size 128`，或直接用 ModelScope 上的 `*-int4-cw-ov` 仓库；
2. **用 `openvino_genai.LLMPipeline` 驱动** — 它内部处理 NPU 的静态 shape 要求；
   optimum-intel 的 `OVModelForCausalLM` 会强制动态 shape，NPU 不支持。

实测可跑 NPU 的模型：

| 模型 | 大小 | 设备 | 说明 |
|---|---|---|---|
| `OpenVINO/Qwen3-8B-int4-cw-ov` | 4.8 GB | NPU | 对称量化 · 加载快 |
| `OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov` | 1.1 GB | NPU | ~19 tok/s |
| `OpenVINO/MiniCPM-V-4_5-int4-cw-ov` | 5.3 GB | NPU | 多模态 · 9B 级 |

> Qwen3 系默认进 thinking 模式（慢）；平台默认加 `/no_think` 抑制，
> 需深度推理时设 `EDGE_ENABLE_THINKING=1`。
>
> `Qwen3-VL-4B-*-int4-ov`（AWQ 非对称）**无法在 NPU 上运行**，仅 CPU/GPU。

---

## 📦 推荐模型（按硬件 · 偏重超大 MoE）

本项目的价值在于**让原本跑不动的大模型跑起来**，所以推荐的都是 **MoE 混合专家**模型 ——
它们的单 token 工作集只是*激活参数*（如 35B-A3B 只激活 3B），这正是三层卸载能获益的关键。

工作台「菜单 → 模型」页展示同一份清单，**点击卡片即自动下载**。

| 硬件 | 小模型 | 大模型 | 分层 |
|---|---|---|---|
| **NVIDIA GPU** | Qwen3.6-35B-A3B（19.4 GB） | Qwen3-Next-80B-A3B（48.5 GB） | SSD + 内存 + 显存 |
| **Intel GPU** | Qwen3-30B-A3B（16.3 GB） | Qwen3.6-35B-A3B（19.4 GB） | SSD + 内存 + 共享显存 |
| **Intel NPU** | Qwen3-8B（4.8 GB） | Qwen3-30B-A3B-nf4（15.7 GB） | SSD + 内存（无独立显存层） |

- **N 卡**：从 35B-A3B 起（多模态），大到 80B-A3B —— 8 GB 显存也能跑，靠三层卸载。
  80B 走 **llama.cpp**（GGUF Q4_K_M），因为 OpenVINO IR 目前最大只到 35B-A3B。
- **i 卡**：共享系统内存，30B / 35B MoE 均可卸载。
- **NPU**：本身就用系统内存（无独立显存层），因此走 **SSD + 内存两层分片**；
  已实测 9B 多模态（MiniCPM-V-4.5），30B 级 MoE 验证进行中。

> 说明：`llama.cpp` 后端的 GGUF 模型用 `--n-cpu-moe N` 把 MoE 层放到 CPU/内存、
> 其余留在显存，是天然的三层卸载路径；OpenVINO IR 模型则由本项目 Rust 内核规划分层。

---

## 🏗 架构

```
┌────────────────────────────────────────────────────────────┐
│  GUI（Tauri 原生窗口 / 浏览器工作台）                        │
│  对话 · Agent 工作区 · 三层方案 · 模型 · 设置                │
└───────────────────────┬────────────────────────────────────┘
                        │  OpenAI 兼容 HTTP
                        ▼
┌────────────────────────────────────────────────────────────┐
│  FastAPI 服务  python/edge_server/openai_server.py          │
│  /v1/chat/completions   推理（SSE 流式）                     │
│  /agent/chat            本地 Agent 工具循环                  │
│  /edge/devices          真实硬件探测（含 SSD 带宽）           │
│  /edge/run_plan         调 Rust 内核计算三层方案              │
│  /edge/models,/download 模型管理                            │
└───────┬──────────────────────────────┬─────────────────────┘
        │                              │
        ▼                              ▼
┌──────────────────┐        ┌──────────────────────────────┐
│ genai_backend.py │        │ pure_ov_gen.py               │
│ LLMPipeline      │        │ 纯 OpenVINO 手写 VLM 推理      │
│ NPU/GPU/CPU      │        │ （无 torch/transformers）      │
└──────────────────┘        └──────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Rust 内核  crates/                                         │
│  edge-detect    硬件探测（GPU/NPU/CPU/SSD）                  │
│  edge-tiers     三层（SSD/RAM/VRAM）热度分配器               │
│  edge-prefetch  Prerouter 预取引擎                          │
│  edge-engine    组装层 + CLI                                │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 前置

- Windows 11
- Rust 1.75+（GNU 或 MSVC 工具链）
- Python 3.10+
- Node.js 18+（仅构建 Tauri 原生窗口需要）

### 1. 一键启动（推荐）

```bat
run_windows.bat
```

先起 Python 推理服务，再打开 Tauri 原生窗口；若未编译原生窗口则自动回退浏览器。

### 2. 手动启动

```bash
# 安装 Python 依赖
cd python
pip install -r requirements.txt

# 下载一个 NPU 兼容模型（含对称量化）
python -m edge_server.download_modelscope OpenVINO/Qwen3-8B-int4-cw-ov

# 启动服务（默认 device=NPU，监听 0.0.0.0:8080）
EDGE_DEVICE=NPU EDGE_MODEL_DIR=../models/qwen3-8b-int4-cw python run_server.py
```

打开 <http://127.0.0.1:8080/ui/> 即为工作台。

### 3. 构建 Rust 内核

```bash
cargo build --release          # 产出 target/release/edge-engine.exe
cargo test                     # 单元测试
```

### 4. 构建 Tauri 原生窗口

```bash
# 本机用 MinGW-w64（winlibs）：解压到 C:\mingw64
export PATH="/c/mingw64/mingw64/bin:$HOME/.cargo/bin:$PATH"
cd tauri-app/src-tauri
cargo build --release          # 产出 target/release/edge-gui.exe
```

---

## 🤖 本地 Agent（项目化开发）

工作台「项目」与「Agent 工作区」标签页：用自然语言下达开发任务，模型会自主调用工具
**真正在磁盘上创建项目、写代码、建沙箱、装依赖、跑程序**。

可用工具：

| 工具 | 说明 |
|---|---|
| `create_project` | 创建项目文件夹（含 `src/data/output/notes`），支持 empty/python 模板 |
| `open_project` / `list_projects` | 打开 / 列出项目 |
| `sandbox_setup` | 在项目内创建 Python venv 沙箱（`.venv/`） |
| `sandbox_run` | 在沙箱内执行命令（`pip install ...` / `python xxx.py`） |
| `read_file` / `write_file` | 读取 / 写入文件 |
| `edit_file` / `append_file` | 查找替换 / 追加 |
| `list_dir` / `make_dir` / `search` | 目录与搜索 |
| `run_command` / `cd` / `get_workspace` | 开发命令 / 切目录 / 工作区信息 |

**典型工作流**（已实测通过）：

```bash
curl -X POST http://127.0.0.1:8080/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"创建一个 Python 项目「斐波那契」，写一个打印前10个斐波那契数的程序并运行","max_steps":6}'
```

Agent 会自动：`create_project` → `write_file` → `sandbox_setup` → `sandbox_run`
→ 在磁盘上真出现项目文件夹并输出运行结果。

GUI 的「项目」视图提供：项目列表、文件树、文件编辑器（打开/保存）、沙箱终端。

> 所有操作被限制在 `workspace/` 沙箱内，越权路径会被拒绝。
> Agent 需要指令遵循较强的模型，**建议用 8B**（1.5B 太弱）。

---

## 🌐 OpenAI 兼容 API

```bash
# 非流式
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你好"}],"temperature":0}'

# 流式（SSE）
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

也可以用官方 OpenAI SDK：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen", messages=[{"role": "user", "content": "你好"}])
print(resp.choices[0].message.content)
```

---

## ⚙️ 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `EDGE_DEVICE` | `NPU` | 推理设备：`NPU` / `CPU` / `GPU.0` |
| `EDGE_MODEL_DIR` | 自动选择 | 模型目录 |
| `EDGE_API_KEY` | 空 | 对外 API 鉴权（空 = 不鉴权） |
| `EDGE_PORT` | `8080` | 监听端口 |
| `EDGE_WORKSPACE` | `./workspace` | Agent 沙箱根目录 |
| `EDGE_ENABLE_THINKING` | `0` | Qwen3 是否开启 thinking |

---

## 📁 目录结构

```
edge-infer/
├── crates/                       Rust 自研内核（4 crate）
│   ├── edge-detect/              硬件探测（GPU/NPU/CPU/SSD）
│   ├── edge-tiers/               三层（SSD/RAM/VRAM）分配器
│   ├── edge-prefetch/            Prerouter 预取引擎
│   └── edge-engine/              组装层 + CLI
├── python/
│   ├── edge_server/
│   │   ├── openai_server.py      FastAPI OpenAI 兼容服务
│   │   ├── genai_backend.py      NPU/LLM 后端（LLMPipeline）
│   │   ├── pure_ov_gen.py        纯 OpenVINO VLM 推理
│   │   ├── agent_tools.py        本地 Agent 工具 + 沙箱
│   │   ├── storage_probe.py      SSD 型号 + 带宽探测
│   │   ├── download_modelscope.py ModelScope 下载
│   │   └── static/index.html     工作台 GUI
│   ├── requirements.txt
│   ├── run_server.py
│   └── verify_4b.py
├── tauri-app/                    Tauri 原生壳
├── run_windows.bat               一键启动
├── CHANGELOG.md
└── LICENSE
```

---

## ✅ 验证

```bash
cd python && python verify_4b.py    # 设备探测 + OpenAI 契约 + 真实推理
cargo test                          # Rust 内核单测
```

---

## 📄 License

[Apache-2.0](LICENSE) © 2026 David
