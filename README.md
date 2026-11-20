# edge-infer

**端侧三层卸载多硬件推理平台 · 本地 AI 工作台**

[![Version](https://img.shields.io/badge/version-0.2.1-2ea44f.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.4-0091D5.svg)](https://docs.openvino.ai/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-CUDA%20%7C%20CPU-6E6E6E.svg)](https://github.com/ggml-org/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](#-快速开始)
[![Tauri](https://img.shields.io/badge/Tauri-2.x-24C8DB.svg)](https://tauri.app/)

[English](README_EN.md) · **简体中文**

在 Windows 上运行本地大模型：**Rust 自研三层卸载内核**（SSD / 内存 / 显存）+
**双推理后端**（NVIDIA → llama.cpp CUDA；Intel GPU / NPU → OpenVINO）+
**本地 Agent 工作台**（可真正操作文件系统、创建项目文件夹、执行命令）。

> 目标：让原本跑不动大模型的 Windows 笔记本也能跑起来，并支持更长的上下文。
> 从内核到调度全部独立自主实现。

---

## ✨ 特性

- **三层卸载内核（Rust）** — 把 MoE 专家按热度分布在 SSD / RAM / VRAM 三层，
  显存动态探测（NVML），OOM 自动回退（缩短上下文 / 降级专家）。
- **Prerouter 预取** — 一步前瞻预测下一 token 要用的专家，在当前前向计算时
  并行把它从慢层拉到快层，用计算掩盖 IO。
- **双推理路线（按硬件匹配最快的后端）** — 同一份配置下自动分流：

  | 硬件 | 后端 | 实测 decode tok/s（Qwen3-8B Q4_K_M） |
  |---|---|---|
  | NVIDIA GPU (RTX 4070 Laptop) | **llama.cpp CUDA** | **43.3** |
  | Intel GPU (Arc iGPU) | OpenVINO (GPU.0) | 12.0 |
  | Intel NPU (AI Boost) | OpenVINO (NPU) | 6.4 |
  | CPU | llama.cpp | 4.5 |
  | CPU | OpenVINO | 3.5 |
  | NVIDIA via OpenVINO (GPU.1) | （不推荐） | 2.3 ← 慢约 19× |

  关键结论：**OpenVINO 的 GPU 插件面向 Intel 优化，把 N 卡交给它等于浪费约 19 倍性能**。
  因此 N 卡一律走 llama.cpp CUDA，Intel GPU / NPU 走 OpenVINO。
  **超大 MoE（80B-A3B）在 8 GB 显存上靠 `--n-cpu-moe` 三层卸载也能跑：N 卡 decode 9.5 tok/s。**
  （完整的 prefill / decode、`ncmoe` 扫描与测试条件见 [性能实测](#-性能实测)。）

- **同一模型按设备自动匹配格式** — 模型目录按「模型 → 变体」组织：同一个模型名下
  挂 `nvidia` / `intel-gpu` / `npu` 多套变体，选中硬件后自动取对应格式
  （N 卡 → **GGUF**，Intel GPU / NPU → **OpenVINO IR**），用户无需关心底层格式；
  选错组合会明确提示，不会静默走慢路径。

- **纯 OpenVINO 推理链路** — 不依赖 torch / transformers，只用 `openvino` +
  `tokenizers` + `numpy`，绕开版本地狱（VLM 路径）。
- **OpenVINO GenAI 后端** — `LLMPipeline` 驱动 NPU / Intel GPU / CPU，支持 **SSE 流式输出**。
- **llama.cpp 后端** — 运行 GGUF 模型（NVIDIA CUDA 路径），覆盖 OpenVINO IR
  无法承载的更大 MoE（80B-A3B 及以上）；用 `--n-cpu-moe` 把 MoE 层放到
  CPU/内存实现三层卸载。
- **量化说明** — FP4/NVFP4 需 Blackwell（RTX 50 系，sm120）的 FP4 张量核；
  **RTX 40 系（Ada，sm89）没有 FP4 通路**，NVFP4 会退化为软件模拟（可能更慢）。
  Ada 卡的正确选择是 **FP8**（需 vLLM/TensorRT-LLM）或 **GGUF Q4_K_M / Q5_K_M**。
- **本地 Agent** — 模型可自主调用工具：创建任务文件夹、
  读写文件、搜索、执行命令；全部限制在沙箱工作区内。
- **联网搜索** — 内置 Playwright 驱动本机 Edge 抓取搜索结果并注入上下文，
  可与本地模型组合使用（开关式）。
- **OpenAI 兼容 API** — 对外暴露 `/v1/chat/completions`（含 `stream=true`），
  可被任意客户端调用；支持 API Key 鉴权。
- **工作台 GUI** — 对话 / Agent 工作区 / 三层方案 / 模型 / 设置
  五个视图；顶部设备 chip 可点击切换；输入区支持模型选择、文件 / 图片 /
  文件夹上传与联网搜索开关；会话列表持久化，可切换本地引擎 or 外接 OpenAI API。
- **一键启动** — Tauri 原生窗口（6.7MB exe，非浏览器外壳）启动时自动拉起
  Python 推理服务，关闭时自动回收，不留残留进程。

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

## 📊 性能实测

测试机：**Intel Core Ultra 7 155H**（16 线程）+ **Intel Arc iGPU** +
**Intel AI Boost NPU（NPU3720）** + **NVIDIA RTX 4070 Laptop 8 GB（sm_89）** +
**32 GB DDR5-5600**，Windows 11。
工具链：llama.cpp `b12559`（CUDA 12）、OpenVINO `2026.4.1`。

测试模型：**Qwen3-8B Q4_K_M（GGUF）** / **Qwen3-8B int4-cw（OpenVINO IR）**——
同一个模型，各自用对应设备偏好的格式。

| 后端 · 设备 | Prefill tok/s | **Decode tok/s** | 相对倍率 |
|---|---|---|---|
| llama.cpp CUDA · RTX 4070 8 GB | 2381.8 | **43.3** | 1.00× |
| OpenVINO · Intel Arc iGPU | 1297.7 | 12.0 | 慢 3.6× |
| OpenVINO · Intel NPU | 35.0 | 6.4 | 慢 6.8× |
| llama.cpp · CPU（16 线程） | 198.7 | 4.5 | 慢 9.6× |
| OpenVINO · CPU | 283.3 | 3.5 | 慢 12.4× |
| OpenVINO · NVIDIA（GPU.1） | 26.2 | **2.3** | **慢 18.8×** |

测法：`llama-bench`（`-p 512 -n 128`，3 次取优）与 `scripts/bench_ov.py`
（531 token 提示词、生成 128 token、2 次取优）。Prefill = 提示词处理，
decode = 稳态生成。

### 超大 MoE 三层卸载（重点）

**80B 模型在 8 GB 显存 + 32 GB 内存上跑起来** —— 这是本项目的核心场景。
模型 `Qwen3-Next-80B-A3B`（48.5 GB，总参 80B / 激活 3B），用
`--n-cpu-moe` 把 MoE 层放 CPU/内存、注意力与 dense 层留显存：

| 后端 · 设备 | 卸载配置 | Prefill tok/s | Decode tok/s | 备注 |
|---|---|---|---|---|
| llama.cpp CUDA · RTX 4070 8 GB | `ncmoe=48` | **34.2** | **9.5** | 最优；约 48/64 层 MoE 落在内存 |
| llama.cpp Vulkan · Intel Arc | `ncmoe=48` | 6.4 | 3.9 | 共享内存上限 15.5 GB |
| ~~llama.cpp CUDA · `ngl=999` 全量卸载~~ | — | — | — | ❌ 直接 OOM（需 46 GB 显存） |

`--n-cpu-moe` 扫描结果（N 卡，`-p 128 -n 64`）：

| ncmoe | Prefill tok/s | Decode tok/s |
|---|---|---|
| 30 | 4.3 | 3.0 |
| 40 | 3.6 | 3.6 |
| 44 | 9.8 | 3.9 |
| **48** | **10.8** | **7.3** ← 最优 |
| 52 | 12.6 | 7.0 |
| 56 | 15.2 | 6.2 |
| 60 | 13.3 | 4.7 |

> 卸载越多，显存压力越小但 CPU 侧带宽成为瓶颈 —— 48 层附近是
> 「显存装得下的层都装下、剩下的丢内存」的平衡点。**8B → 80B 参数量涨 10×，
> decode 只从 43.3 掉到 9.5 tok/s（约 4.6×）**，这正是 MoE + 三层卸载的价值。

### 怎么读这组数

- **N 卡一律走 llama.cpp CUDA。** RTX 4070 在两个维度都碾压其它选项；
  交给 OpenVINO（GPU.1）decode 慢 **18.8×**、prefill 慢 **91×** ——
  OpenVINO 的 GPU 插件是为 Intel 芯片打造的。
- **Intel Arc 核显是非 N 卡里的最强选项** —— prefill 只比 4070 慢 1.8×，
  而且它共享系统内存，能承载超出独显预算的模型（80B 也能跑，靠 Vulkan + 三层卸载）。
- **NPU 面向常驻低功耗**，不是吞吐：6.4 tok/s，但功耗只有零头。
  它没有独立显存，因此走 SSD + 内存两层分片。
- **CPU 上 prefill 与 decode 差距悬殊。** CPU 对提示词处理做了良好向量化
  （198–283 tok/s），但每生成一个 token 都受内存带宽限制，所以 decode 只有 4.5 tok/s。
- **核显/Vulkan 路径的取舍**：I 卡跑 8B 时 Vulkan（140/9.2）**慢于** OpenVINO
  （1298/12.0）——OpenVINO 是 Intel 的原生优化路径；但 80B 没有 OpenVINO IR 版，
  此时 Vulkan + `--n-cpu-moe` 是唯一可行解。

---

## 📦 推荐模型（按硬件 · 偏重超大 MoE）

本项目的价值在于**让原本跑不动的大模型跑起来**，所以推荐的都是 **MoE 混合专家**模型 ——
它们的单 token 工作集只是*激活参数*（如 35B-A3B 只激活 3B），这正是三层卸载能获益的关键。

工作台「菜单 → 模型」页展示同一份清单，**选中设备后自动匹配对应格式并点击下载**。

| 模型 | 参数 | 后端 · 格式 | 体积 | 分层 |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | 35B-A3B · 多模态 | llama.cpp CUDA · GGUF / OpenVINO · INT4 | 21.2 / 19.4 GB | SSD + 内存 + 显存 |
| **Qwen3-30B-A3B** | 30B-A3B | llama.cpp CUDA · GGUF / OpenVINO · INT4 | 18.6 / 16.3 GB | SSD + 内存 + 显存 |
| **Qwen3-Next-80B-A3B** | 80B-A3B | llama.cpp CUDA · GGUF | 48.5 GB | SSD + 内存 + 显存 |
| **Qwen3-8B** | 8B | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 5.0 / 4.75 GB | SSD + 内存（+ 显存） |
| **MiniCPM-V-4.5** | 9B · 多模态 | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 6.0 / 5.26 GB | SSD + 内存（+ 显存） |

- **N 卡**：一律走 **llama.cpp CUDA**（8B 实测 43.3 tok/s；交给 OpenVINO 只有 2.3）。
  从 35B-A3B 到 80B-A3B —— 8 GB 显存也能跑，靠三层卸载。
- **i 卡**：走 **OpenVINO**（Arc 上比 Vulkan 快约 1.8×），30B / 35B MoE 均可卸载。
- **NPU**：本身就用系统内存（无独立显存层），因此走 **SSD + 内存两层分片**；
  已实测 9B 多模态（MiniCPM-V-4.5，对称 INT4），30B 级 MoE 验证进行中。

> **同一模型按设备自动切换格式**：上表里同一模型名在 N 卡下取 GGUF、在 i 卡 / NPU 下取
> OpenVINO IR —— 由 `model_catalog.py` 的「模型 → 变体」结构 + `resolve_device()`
> 自动完成，用户无需手动选格式。
>
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
│  /edge/catalog          按设备匹配模型变体（GGUF / IR）       │
│  /edge/download(s)      模型下载与状态                       │
└───┬──────────────┬───────────────┬─────────────────────────┘
    │              │               │
    ▼              ▼               ▼
┌────────────┐ ┌──────────────┐ ┌────────────────────────────┐
│ llama_     │ │ genai_       │ │ pure_ov_gen.py             │
│ backend.py │ │ backend.py   │ │ 纯 OpenVINO 手写 VLM 推理    │
│ GGUF →     │ │ LLMPipeline  │ │ （无 torch/transformers）    │
│ llama.cpp  │ │ NPU/GPU/CPU  │ │                            │
│ (CUDA)     │ │              │ │                            │
└────────────┘ └──────────────┘ └────────────────────────────┘
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
| `EDGE_LLAMA_DIR` | `D:\AI\tools\llama` | 存放 `llama-server.exe` 的目录 |
| `EDGE_N_CPU_MOE` | `0` | 放在 CPU/内存的 MoE 层数（三层卸载开关） |

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
│   │   ├── llama_backend.py      llama.cpp 后端（GGUF / CUDA 三层卸载）
│   │   ├── genai_backend.py      OpenVINO GenAI 后端（LLMPipeline）
│   │   ├── pure_ov_gen.py        纯 OpenVINO VLM 推理
│   │   ├── model_catalog.py      按设备匹配的模型目录（模型 → 变体）
│   │   ├── web_search.py         联网搜索（Playwright + 本机 Edge）
│   │   ├── agent_tools.py        本地 Agent 工具 + 沙箱
│   │   ├── storage_probe.py      SSD 型号 + 带宽探测
│   │   ├── download_modelscope.py ModelScope 下载
│   │   └── static/index.html     工作台 GUI
│   ├── requirements.txt
│   ├── run_server.py
│   └── verify_4b.py
├── scripts/
│   ├── bench_llama.py            llama.cpp 基准（封装 llama-bench）
│   └── bench_ov.py               OpenVINO 基准（拆分 prefill / decode）
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

# 性能复现
python scripts/bench_llama.py models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf --ngl 999
python scripts/bench_ov.py models/qwen3-8b-int4-cw-ov NPU

# 80B 三层卸载
EDGE_LLAMA_DIR=D:/AI/tools/llama D:/AI/tools/llama/llama-bench.exe \
  -m models/qwen3-next-80b-a3b-instruct-gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 999 -ncmoe 48 -p 512 -n 128 -r 2
```

---

## 📄 License

[Apache-2.0](LICENSE) © 2026 David
