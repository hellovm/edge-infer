# edge-infer

**在普通人的电脑上本地运行超大规模模型 · 高能效端侧推理平台**

[![Version](https://img.shields.io/badge/version-0.3.0-2ea44f.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-个人免费%20·%20商用授权-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.4-0091D5.svg)](https://docs.openvino.ai/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-CUDA%20%7C%20CPU-6E6E6E.svg)](https://github.com/ggml-org/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D6.svg)](#-快速开始)
[![Tauri](https://img.shields.io/badge/Tauri-2.x-24C8DB.svg)](https://tauri.app/)

[English](README_EN.md) · **简体中文**

**目标：让普通人的电脑也能本地运行超大规模模型，并以高能效的方式运行。**

一台 8 GB 显存 + 32 GB 内存的普通 Windows 电脑，就能流畅运行 180B 级 MoE 大模型。
靠的是自研的 **Rust 三层卸载内核**（SSD / 内存 / 显存）+ **MTP 投机解码**，
在本地把显存、内存、SSD 组织成一条流水线。

- 🧱 **三层卸载** — 权重按热度分布在 SSD / 内存 / 显存，显存不够也能跑大模型
- ⚡ **MTP 投机解码** — 主模型一次校验多个预判 token，输出完全不变，速度再提约 17%
- 🔌 **OpenAI 兼容 API** — 对外提供标准 `/v1/chat/completions`，可对接任意 Agent 工作台
- 🖥 **多硬件后端** — NVIDIA 走 llama.cpp CUDA；Intel GPU / NPU 走 OpenVINO

**下一步目标**：持续提升本地运行性能（更快的 decode、更长的上下文、更低的资源占用）。

---

## ✨ 特性

- **三层卸载内核（Rust）** — 把 MoE 专家按热度分布在 SSD / RAM / VRAM 三层，
  显存动态探测（NVML），OOM 自动回退（缩短上下文 / 降级专家）。
- **MTP 投机解码** — 用模型自带的 MTP draft head 预判多个候选 token，
  主模型一次并行校验（**输出与不开时完全一致**），本机实测生成速度 **+17%**。
  带 draft 的模型（如 Qwen3.8-Flash-Next）自动启用，可在设置中开关。
- **Prerouter 预取** — 一步前瞻预测下一 token 要用的专家，在当前前向计算时
  并行把它从慢层拉到快层，用计算掩盖 IO。
- **双推理路线（按硬件匹配最快的后端）** — 同一份配置下自动分流：

  | 硬件 | 后端 | 实测 decode tok/s（Qwen3-8B Q4_K_M） |
  |---|---|---|
  | NVIDIA GPU (RTX 4070 Laptop) | **llama.cpp CUDA** | **43.3** |
  | Intel GPU (Arc iGPU) | OpenVINO (GPU.0) | 12.0 |
  | Intel NPU (AI Boost · Core Ultra 7 155H) | OpenVINO (NPU) | 6.4 |

  关键结论：**OpenVINO 的 GPU 插件面向 Intel 优化，把 N 卡交给它等于浪费约 19 倍性能**
  （实测 decode 2.3 vs 43.3 tok/s）。因此 N 卡一律走 llama.cpp CUDA，
  Intel GPU / NPU 走 OpenVINO。
  **超大 MoE（Qwen3.8-Flash-Next，180B 存储）在 8 GB 显存上靠 `--n-cpu-moe` 三层卸载也能跑。**
  （完整数据、`ncmoe` 扫描与 MTP 对比见 [性能实测](#-性能实测)。）

- **同一模型按设备自动匹配格式** — 模型目录按「模型 → 变体」组织：同一个模型名下
  挂 `nvidia` / `intel-gpu` / `npu` 多套变体，选中硬件后自动取对应格式
  （N 卡 → **GGUF**，Intel GPU / NPU → **OpenVINO IR**），用户无需关心底层格式；
  选错组合会明确提示，不会静默走慢路径。

- **纯 OpenVINO 推理链路** — 不依赖 torch / transformers，只用 `openvino` +
  `tokenizers` + `numpy`，绕开版本地狱（VLM 路径）。
- **OpenVINO GenAI 后端** — `LLMPipeline` 驱动 NPU / Intel GPU / CPU，支持 **SSE 流式输出**。
- **llama.cpp 后端** — 运行 GGUF 模型（NVIDIA CUDA 路径），覆盖 OpenVINO IR
  无法承载的更大 MoE（如 Qwen3.8-Flash-Next，180B 存储）；用 `--n-cpu-moe` 把 MoE 层放到
  CPU/内存实现三层卸载。
- **量化说明** — FP4/NVFP4 需 Blackwell（RTX 50 系，sm120）的 FP4 张量核；
  **RTX 40 系（Ada，sm89）没有 FP4 通路**，NVFP4 会退化为软件模拟（可能更慢）。
  Ada 卡的正确选择是 **FP8**（需 vLLM/TensorRT-LLM）或 **GGUF 量化**。
- **本地 Agent** — 模型可自主调用工具：创建任务文件夹、
  读写文件、搜索、执行命令；全部限制在沙箱工作区内。
- **联网搜索** — 内置 Playwright 驱动本机 Edge 抓取搜索结果并注入上下文，
  可与本地模型组合使用（开关式）。
- **OpenAI 兼容 API** — 对外暴露 `/v1/chat/completions`（含 `stream=true`），
  标准协议，可被任意客户端或 Agent 工作台调用；支持 API Key 鉴权。
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
| llama.cpp CUDA · NVIDIA RTX 4070 Laptop 8 GB | 2381.8 | **43.3** | 1.00× |
| OpenVINO · Intel Arc iGPU | 1297.7 | 12.0 | 慢 3.6× |
| OpenVINO · Intel NPU（AI Boost · Core Ultra 7 155H） | 35.0 | 6.4 | 慢 6.8× |

测法：`llama-bench`，`-p 512 -n 128`，3 次取优。
Prefill = 提示词处理，decode = 稳态生成。

> 另外两路对照（同为 Qwen3-8B，未计入主表）：**CPU** 上 llama.cpp 198.7 / 4.5、
> OpenVINO 283.3 / 3.5 tok/s；**NVIDIA 经 OpenVINO（GPU.1）** 仅 26.2 / 2.3 tok/s，
> 比 CUDA 慢约 19×，故不推荐。

### 超大 MoE 三层卸载（重点）

**Qwen3.8-Flash-Next（180B 存储 / 6B 激活）在 8 GB 显存 + 32 GB 内存上跑起来** ——
这是本项目的核心场景。`--n-cpu-moe` 把 MoE 专家放 CPU/内存、注意力与 dense 层留显存，
28.8 GB 的 n-gram 表靠 `--lazy-mode` 留在 SSD：

| 主模型 · 量化 | 后端 · 设备 | 卸载配置 | Decode tok/s |
|---|---|---|---|
| **UD-Q2_0（65.7 GB）** | llama.cpp CUDA · RTX 4070 8 GB | `ncmoe=35` | **1.1** |
| UD-IQ1_S（66.4 GB） | 同上 | `ncmoe=30` | 1.1 |
| UD-IQ4_XS（87 GB） | 同上 | `ncmoe=45` | 1.2 |

> 三种量化等级 decode 几乎相同 —— 因为瓶颈是 **SSD 分页加载专家权重的 IO**，
> 不是计算或量化精度。这也决定了「跑得动大模型」的性价比主要来自三层卸载本身。

**MTP 投机解码对比**（同一真实任务，`-c 4096 -n 192`，各测 2 次取稳）：

| 主模型 | 无 MTP | + MTP（shared draft，n-max 2） | 提升 |
|---|---|---|---|
| **UD-Q2_0** | 1.2 t/s | **1.4 t/s** | **+17%** |
| UD-IQ1_S | 1.1 t/s | 1.3 t/s | +18% |
| UD-IQ4_XS | 1.2 t/s | 1.4 t/s | +17% |

> MTP 用 draft head 预判多个候选 token，主模型**一次并行校验**（数学上无损，
> 输出与不开时逐字一致）。提速稳定在 **~17%**，且同样受限于 SSD 带宽 ——
> 它减少 decode 步数，但绕不过每步的专家权重加载。
> draft 优先选 `shared-` 变体（借用主模型 embedding，更省内存、更快）。

### 怎么读这组数

- **N 卡一律走 llama.cpp CUDA。** RTX 4070 在两个维度都碾压其它选项；
  交给 OpenVINO（GPU.1）decode 慢 **18.8×**、prefill 慢 **91×** ——
  OpenVINO 的 GPU 插件是为 Intel 芯片打造的。
- **Intel Arc 核显是非 N 卡里的最强选项** —— prefill 只比 4070 慢 1.8×，
  而且它共享系统内存，能承载超出独显预算的模型（30B/35B MoE 靠 Vulkan + 三层卸载）。
- **Intel NPU（Core Ultra 7 155H 上的 AI Boost）面向常驻低功耗**，不是吞吐：
  6.4 tok/s，但功耗只有零头。它没有独立显存，因此走 SSD + 内存两层分片。
- **核显/Vulkan 路径的取舍**：I 卡跑 8B 时 Vulkan（140/9.2）**慢于** OpenVINO
  （1298/12.0）——OpenVINO 是 Intel 的原生优化路径；30B/35B MoE 无 IR 版时，
  Vulkan + `--n-cpu-moe` 是可行解。
- **Qwen3.8-Flash-Next 仅 N 卡可行**：66.4 GB 权重、无 OpenVINO IR 版，
  I 卡 / NPU 路径均不可行。

### 大模型运行方式对比

同样是"小显存跑大模型"，不同方式的取舍不同：

| 方式 | 显存门槛 | 速度 | 精度 | 特点 |
|---|---|---|---|---|
| 量化后全量驻留显存 | 高（需装下权重） | 快 | 有量化损失 | 显存够时体验最好 |
| 逐层流式加载 | 低（单层即可） | 慢（秒级/token） | 可保持原精度 | 每个 token 都读盘，适合离线 |
| **本项目三层卸载** | **低（8 GB 起）** | **中等** | GGUF 量化 | 专家常驻内存、大表留 SSD，兼顾门槛与速度 |

---

## 📦 推荐模型（按硬件 · 偏重超大 MoE）

本项目的价值在于**让原本跑不动的大模型跑起来**，所以推荐的都是 **MoE 混合专家**模型 ——
它们的单 token 工作集只是*激活参数*（如 35B-A3B 只激活 3B），这正是三层卸载能获益的关键。

工作台「菜单 → 模型」页展示同一份清单，**选中设备后自动匹配对应格式并点击下载**。

| 模型 | 参数 | 后端 · 格式 | 体积 | 分层 |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next** ⭐ | 180B 存储 / 6B 激活 | llama.cpp CUDA · GGUF（UD-Q2_0）+ MTP | 65.7 + 1.8 GB | SSD + 内存 + 显存 |
| **Qwen3.6-35B-A3B** | 35B-A3B · 多模态 | llama.cpp CUDA · GGUF / OpenVINO · INT4 | 21.2 / 19.4 GB | SSD + 内存 + 显存 |
| **Qwen3-30B-A3B** | 30B-A3B | llama.cpp CUDA · GGUF / OpenVINO · INT4 | 18.6 / 16.3 GB | SSD + 内存 + 显存 |
| **Qwen3-8B** | 8B | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 5.0 / 4.75 GB | SSD + 内存（+ 显存） |
| **MiniCPM-V-4.5** | 9B · 多模态 | llama.cpp CUDA · GGUF / OpenVINO · INT4_SYM | 6.0 / 5.26 GB | SSD + 内存（+ 显存） |

- ⭐ **默认旗舰组合：Qwen3.8-Flash-Next（UD-Q2_0）+ shared-Q4_K_M MTP draft** ——
  180B 存储的旗舰 MoE，在 8 GB 显存 / 32 GB 内存上即可运行，并自动启用 MTP 投机解码。
- **N 卡**：一律走 **llama.cpp CUDA**（8B 实测 43.3 tok/s；交给 OpenVINO 只有 2.3）。
  从 35B-A3B 到 180B 存储的 Flash-Next —— 8 GB 显存也能跑，靠三层卸载。
- **i 卡**：走 **OpenVINO**（Arc 上比 Vulkan 快约 1.8×），30B / 35B MoE 均可卸载。
- **NPU**：本身就用系统内存（无独立显存层），因此走 **SSD + 内存两层分片**；
  已实测 9B 多模态（MiniCPM-V-4.5，对称 INT4）。

> **同一模型按设备自动切换格式**：上表里同一模型名在 N 卡下取 GGUF、在 i 卡 / NPU 下取
> OpenVINO IR —— 由 `model_catalog.py` 的「模型 → 变体」结构 + `resolve_device()`
> 自动完成，用户无需手动选格式。
>
> 说明：`llama.cpp` 后端的 GGUF 模型用 `--n-cpu-moe N` 把 MoE 层放到 CPU/内存、
> 其余留在显存，是天然的三层卸载路径；OpenVINO IR 模型则由本项目 Rust 内核规划分层。

---

## 🧭 三层卸载规划工具

在下载 60 GB+ 的模型之前，先算清楚它能不能放下 —— `tier_plan.py` **只读 GGUF 头
与张量表**（不加载权重），对任意大小的模型都能秒出结论：

```bash
cd python
python -m edge_server.tier_plan <model.gguf> --vram 8 --ram 32
```

输出会给出**分组体积**（n-gram 表 / MoE 专家 / 注意力 / 其它）与
**逐组分层建议**（VRAM / RAM / SSD），并提示是否需要 SSD 参与。

HTTP：`GET /edge/tier_plan?model=<模型目录或路径>&vram=8&ram=32`

---

## 🚀 超大 MoE：Qwen3.8-Flash-Next（180B 存储 / 6B 激活）

**在 8 GB 显存 + 32 GB 内存上跑 66 GB 模型，并保持 256K 上下文。**

该模型（`qwen4exp` 架构）有三个结构特性，天然契合三层卸载：

| 特性 | 收益 |
|---|---|
| **48 层 = 36 GDN + 12 QSA**，只有 QSA 层有 KV cache | **256K 上下文仅需 2.67 GB** KV（f16），是同规模 dense 的 1/4 |
| **n-gram 表是查表不是矩阵乘**（每 token 只读 16 行） | 28.8 GB 的表可放 **SSD**，`--lazy-mode auto` 自动按需读 |
| **512 专家只激活 10 个**（2%） | 专家权重可按热度分层 |

三层分配：

| 层 | 容量 | 放什么 |
|---|---|---|
| VRAM | 8 GB | 注意力 + norm + 输出 + 部分专家 |
| RAM | 32 GB | 主干权重 + 热专家 |
| **SSD** | — | **n-gram 表 28.8 GB**（lazy 按需读，不占常驻内存）+ 冷专家 |

推荐组合 **UD-Q2_0 + shared-Q4_K_M MTP draft**（应用内自动配置）：

```bash
export LLAMA_ATTN_ROT_DISABLE=1     # qwen4exp 需要

llama-server -m .../Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf \
  -md .../MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 999 -ncmoe 35 --tensor-split 1 -c 32768 -fa on
```

> `-ncmoe 35` 是稳定最优点；`--tensor-split 1` 在加载 MTP draft 时必须显式指定
> （否则第二个模型加载时会因空闲显存探测为 0 而失败）。
> 实测：8 GB 显存 / 32 GB 内存下，开启 MTP 生成速度 **+17%**。

> 关键机制：llama.cpp 的 `--lazy-mode` 默认 `auto`，
> **对 >4 GiB 的张量自动按需读取** —— n-gram 表 28.8 GB 正好命中，
> 开箱即用就会留在磁盘上。

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

对外提供**标准 OpenAI 接口**，因此理论上可对接**任意支持该标准的 Agent 工作台 / 客户端**
（本地或远程），把本机的三层卸载算力当成一个 OpenAI 端点来用。

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

## 🩺 环境诊断与问题记录

在**非开发机**上运行时（尤其是 **AMD 平台**），问题往往无法本地复现，
因此产品内置了完整的自查能力：

- **一键诊断** — 采集 CPU / GPU / NPU / 驱动 / 内存 / 后端 / 模型目录等
  10 类探针，逐项探测并**记录失败原因**，输出结构化 JSON + 可读摘要。
- **问题记录** — 后端加载失败、推理异常、模型下载失败都会自动落盘
  （`logs/problems.jsonl`），含 traceback 与当时的**环境指纹**。
- **能力评估** — 明确告诉你在本机哪些路径可用、哪些不可用，以及原因。

```bash
cd python
python -m edge_server.diagnostics              # 看摘要
python -m edge_server.diagnostics --bundle     # 生成诊断包（回传即可排查）
```

GUI 里：「设置 → 环境诊断」可刷新诊断、查看问题记录、一键导出诊断包。

> 本机示例输出（Intel + NVIDIA）：
> ```
> NVIDIA CUDA : YES  NVIDIA GeForce RTX 4070 Laptop GPU
> Intel GPU   : YES  (OpenVINO)
> Intel NPU   : YES  OpenVINO NPU plugin exposes the device
> AMD NPU     : not found  -> supported: NO
> AMD GPU     : not found  -> via ['(no Vulkan/ROCm backend)']
> ```

诊断包含 10 类探针（`os` / `cpu` / `memory` / `nvidia` / `openvino` /
`vendor_npu` / `backends` / `llama_devices` / `ssd` / `models`），
并发执行，每项失败都**不会**影响其它探针。

**隐私**：API Key / token / `sk-*` / `github_pat_*` 全部脱敏；不采集
prompt 或文件内容。

---

## 🔵 AMD 平台兼容性

当前版本（v0.3.0）支持 **NVIDIA GPU** 与 **Intel GPU / NPU**；
在 AMD 平台会**安全回退 CPU**（不崩溃，但无加速）。

| 硬件 | 状态 |
|---|---|
| AMD Radeon 独显 | 可通过 llama.cpp Vulkan 加速 |
| AMD Radeon 核显 | 可用，收益有限 |
| AMD XDNA NPU（Ryzen AI） | 暂不支持 |

OpenVINO 的 GPU / NPU 插件为 Intel 专属，无法驱动 AMD XDNA NPU，
相关支持仍在规划中。

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
| `EDGE_LLAMA_DIR` | `D:\AI\tools\llama` | 存放 `llama-server.exe` 的目录（AMD 可指向 Vulkan/ROCm 构建） |
| `EDGE_N_CPU_MOE` | `0` | 放在 CPU/内存的 MoE 层数（三层卸载开关） |
| `EDGE_CTX` | `32768` | llama.cpp 上下文窗口（最大 1048576） |
| `EDGE_KV_QUANT` | 自动 | KV cache 量化；≥64K 上下文时自动用 `q8_0` |
| `EDGE_MTP_DRAFT` | 自动 | MTP draft 模型路径；留空则自动在同级 `MTP/` 目录查找 |
| `EDGE_LOG_LEVEL` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` |
| `EDGE_DIAG_BOOT` | `0` | `1` = 启动时自动生成诊断包 |

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
│   │   ├── diagnostics.py        环境诊断 + 问题记录 + 诊断包导出
│   │   ├── tier_plan.py          三层卸载规划（只读 GGUF 头，不加载权重）
│   │   ├── logging_setup.py      统一日志（滚动文件 + 异常捕获）
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
│   └── run_server.py
├── logs/                         运行日志（诊断包 / problems.jsonl）
├── tauri-app/                    Tauri 原生壳
├── run_windows.bat               一键启动
├── CHANGELOG.md
└── LICENSE
```

---

## ✅ 验证

```bash
cargo test    # Rust 内核单测
```

---

## 📄 License

**个人免费使用，商业使用需授权** —— 详见 [LICENSE](LICENSE)。

个人学习、研究、非营利用途可免费使用与修改；在公司/组织中用于商业目的、
或集成进收费产品/服务前，请通过 [GitHub Issues](https://github.com/hellovm/edge-infer/issues)
联系作者获取商业授权。

© 2026 David
