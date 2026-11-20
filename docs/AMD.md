# AMD 平台支持规划

> 状态：**规划中（未实现）** · 更新于 2026-11-20
>
> 本文档说明 edge-infer 在 **AMD 平台**（Ryzen AI XDNA NPU、Radeon 核显 / 独显）上的
> 支持可行性与实施路线。当前版本（v0.2.1）**仅支持 NVIDIA GPU + Intel GPU/NPU**，
> AMD 会安全回退到 CPU（不会崩溃，但也没有加速）。

---

## 0. 一句话结论

| 目标 | 可行性 | 结论 |
|---|---|---|
| **AMD Radeon 独显**（RX 6000/7000/9000） | ✅ **可行** | 走 **llama.cpp ROCm / Vulkan**，与本项目现有 llama.cpp 路径天然兼容 —— **P0 优先级** |
| **AMD Radeon 核显**（RDNA 2/3/3.5 iGPU） | 🟡 **可行但收益有限** | 走 llama.cpp **Vulkan**；ROCm 在核显上常静默回退 CPU，需实测 |
| **AMD XDNA NPU**（Ryzen AI 300/400，XDNA2） | 🔴 **架构不兼容** | 需换成 ONNX Runtime + VitisAI EP 或 FastFlowLM，**无法复用 OpenVINO 路径** |
| **AMD XDNA NPU**（Ryzen 7040/8040/200，XDNA1） | ⛔ **不可用** | 无任何主流运行时支持 LLM 推理 |

---

## 1. 为什么 AMD NPU 不能用现有的 OpenVINO 路径

这是最关键的一点，避免走弯路：

### 1.1 OpenVINO 的 NPU 插件是 Intel 专属

本项目 NPU 路径是 `openvino_genai.LLMPipeline(model, "NPU")`。
OpenVINO 的 NPU 插件**只识别 Intel 的 VPU 系列**（Meteor Lake NPU3720、
Lunar Lake NPU4000、Arrow Lake 等）：

```python
import openvino as ov
ov.Core().get_available_devices()
# Intel 平台: ['CPU', 'GPU.0', 'GPU.1', 'NPU']
# AMD 平台  : ['CPU']            ← NPU 不会出现
```

在 AMD 机器上：
- OpenVINO **只有 CPU 后端**可用（x86 通用指令集）；
- `GPU` 设备**不会出现**（OpenVINO GPU 插件只支持 Intel GPU）；
- `NPU` 设备**不会出现**，即使硅片就在那里。

> 已在本项目 `diagnostics.py` 的 `probe_openvino()` 中显式记录这一点：
> 报告会输出「AMD XDNA is NOT supported by OpenVINO's NPU plugin」。

### 1.2 AMD NPU 是另一套完全独立的软件栈

| 维度 | Intel NPU（本项目已支持） | AMD XDNA NPU |
|---|---|---|
| 运行时 | OpenVINO Runtime | ONNX Runtime + **VitisAIExecutionProvider** |
| 模型格式 | OpenVINO IR（`.xml`/`.bin`） | ONNX（量化后 `.qdq.onnx`） |
| 高层封装 | `openvino_genai.LLMPipeline` | AMD **Ryzen AI Software** / **FastFlowLM** / **Lemonade** |
| 社区 llama.cpp | 无 NPU 后端 | 有实验性 fork（`mlir-aie` 编译 `.xclbin`） |
| 操作系统 | Windows 11 / Linux | 主要 Linux（`amdxdna` 驱动），Windows 支持有限 |

两套栈在**模型格式、运行时 API、算子支持**上完全不兼容。
不存在「改个 device 字符串就能跑」的路径。

### 1.3 硬件代际门槛

- **Ryzen AI 300 / 400 系列（XDNA2，Strix/Krackan Point、Strix Halo）**：
  有可用 LLM 运行时（FastFlowLM、Lemonade v10.6+）。
  实测参考：Ryzen AI MAX 385 上 XDNA2 负责 decode，可达 **43.7 tok/s、0.947 J/token**。
- **Ryzen 7040 / 8040 / 200 系列（XDNA1，Phoenix / Hawk Point）**：
  NPU 存在，但**没有任何主流运行时支持 LLM 推理**。只能跑 CV 类模型（INT8 CNN）。

---

## 2. 推荐路线：先做 AMD GPU（ROCm / Vulkan）

AMD GPU 是**低成本高回报**的方向，因为本项目已经有成熟的 llama.cpp 路径。

### 2.1 为什么可行

llama.cpp 支持多种 AMD 后端，且与本项目 `llama_backend.py` 的
`--n-cpu-moe` 三层卸载机制**完全兼容**：

| 后端 | 适用硬件 | 说明 |
|---|---|---|
| **ROCm / HIP** | Radeon 独显（gfx1030/1100/1200…） | 性能最好，需装 ROCm 运行时 |
| **Vulkan** | 独显 + 核显 | 兼容性最好，无需额外运行时，性能约为 ROCm 的 70–90% |
| CPU | 一切 | 兜底 |

### 2.2 实施步骤（P0）

1. **发布 AMD 版 llama.cpp 后端**
   - 从 llama.cpp Releases 取 `llama-*-bin-win-vulkan-x64.zip`（通用）
     或 `llama-*-bin-win-rocm-x64.zip`（AMD 官方构建）；
   - 解压到 `tools/llama-amd/`，设置 `EDGE_LLAMA_DIR` 指向它。

2. **扩展设备探测**（`openai_server.py`）
   - `_probe_gpu_ids()` 已经能识别 `FULL_DEVICE_NAME` 里的 `NVIDIA` / `INTEL`；
   - 新增 `AMD` / `RADEON` 分支 → 归类为 `amd-gpu`；
   - 设备别名增加 `"AMD-GPU"` / `"Radeon"` → 映射到 llama.cpp 的 `--device Vulkan0`。

3. **模型目录增加 AMD 变体**（`model_catalog.py`）
   - AMD 与 NVIDIA 共用 **GGUF**，无需额外转换 ——
     只是在 `variants` 里加一个 `amd-gpu` 键，指向同一个 repo；
   - 这样「同一模型按设备自动匹配格式」继续成立。

4. **GUI 增加 AMD GPU 选项**
   - 设备 chip 下拉加「AMD GPU」；
   - 模型页徽章显示对应变体。

### 2.3 已知风险

- **核显上 ROCm 会静默回退 CPU**。必须用 `llama-bench --list-devices` 二次确认
  实际命中的设备（本项目的 `probe_llama_devices()` 已做这件事），否则会出现
  「/health 正常但实际在 CPU 上跑」的假象。
- Vulkan 在部分老驱动上不稳定，需要提示用户更新显卡驱动。

---

## 3. 次选路线：AMD NPU（仅 Ryzen AI 300/400）

如果确实要用上 NPU，**不能复用现有路径**，需要独立后端。三种方案：

### 方案 A：ONNX Runtime + VitisAI EP（最贴近本项目架构）

```
模型: GGUF/IR → ONNX → 量化(INT8/INT4, A16W8) → .qdq.onnx
运行时: onnxruntime + VitisAIExecutionProvider
设备: "VitisAI"
```

- **优点**：Python 侧可复用现有 OpenAI 兼容服务框架；控制粒度细。
- **缺点**：需要模型转换流水线（`optimum` → ONNX → AMD 量化）；
  LLM 支持受限于 AMD 的算子覆盖表（BF16 / A16W8 / A8W8 / XINT8 各行约束不同）。
- **工作量**：新增 `amd_npu_backend.py` + 转换脚本，约等于当年做 OpenVINO 路径的 60%。

### 方案 B：调用 FastFlowLM / Lemonade（最快见效）

- **FastFlowLM**：专为 XDNA2 优化的 NPU 推理运行时，命令行调用；
- **Lemonade**（AMD 官方，Apache-2.0）：OpenAI 兼容 HTTP 服务，
  NPU 负责 prefill、iGPU 负责 decode 的混合模式。
- **优点**：能快速让 NPU 跑起来；
- **缺点**：等于**再套一层外部服务**，与本项目「自研内核」的定位有冲突 ——
  建议做成「外接推理源」的一种（`state.useExternal`），而不是内置后端。

### 方案 C：llama.cpp NPU fork（实验性，不建议）

社区有把 `GGML_OP_MUL_MAT` 派发到 XDNA2 的 fork（`mlir-aie` 编译 `.xclbin`），
但：仅 Linux、需自编译、算子覆盖有限、无法保证维护。
**建议仅作为技术观察，不纳入产品路线。**

### 3.4 NPU 路线建议

**短期（v0.3.x）不做**。理由：

1. 覆盖硬件窄（仅 Ryzen AI 300/400）；
2. 实现成本高（独立后端 + 模型转换流水线）；
3. 收益有限（NPU 是低功耗场景，吞吐不如 iGPU；参考 Lemonade 的 NPU decode 约 43 tok/s，
   与 Radeon iGPU 持平，但功耗显著更低）。

**优先级：AMD GPU（P0） >> AMD NPU（P2，视用户需求）**

---

## 4. 已落地的「异地排查」基础设施 ✅

由于开发机是 Intel + NVIDIA 平台，**AMD 机器上的问题无法本地复现**，
因此本项目已内置产品级诊断能力，这是 AMD 支持的前置条件：

### 4.1 环境诊断（`python/edge_server/diagnostics.py`）

采集 10 类探针，**每个探针独立隔离**，一项失败不影响整体：

| 探针 | 内容 |
|---|---|
| `os` / `cpu` / `memory` | 平台、CPU 型号核心数、内存容量 |
| `nvidia` | nvidia-smi、驱动版本、算力 |
| `openvino` | 版本、设备列表、逐设备属性、NPU 插件是否枚举 |
| **`vendor_npu`** | **从 PnP 设备树识别 Intel / AMD NPU 硬件**（关键：能区分「没有 NPU」与「有 NPU 但没驱动/不支持」） |
| `backends` | 逐模块 import 测试（openvino / openvino_genai / tokenizers / modelscope…） |
| **`llama_devices`** | **llama.cpp 能看到的后端与设备**（AMD Radeon 会在这里出现） |
| `ssd` | 硬盘型号与读取带宽 |
| `models` | 模型目录清单与状态（ready / downloading） |

**能力评估**（`assess()`）会输出结构化结论，例如在 AMD 机器上：

```
AMD NPU     : detected  -> supported: NO
              device: AMD Ryzen AI NPU
              reason: OpenVINO's NPU plugin is Intel-only and does not
                      enumerate AMD XDNA. See docs/AMD.md.
AMD GPU     : detected  -> via ['VULKAN']
```

### 4.2 问题记录（`logs/problems.jsonl`）

任何**后端加载失败 / 推理异常 / 模型下载失败**都会自动落盘一条 JSONL，
包含：时间、类别、消息、上下文（模型目录 / 设备 / 走的哪条路径）、
traceback、**当时的环境指纹**。

已接入的失败点：
- 流式推理异常（`inference_stream`）
- llama.cpp 路径回退（`backend_llama_fallback`）
- OpenVINO GenAI 路径回退（`backend_genai_fallback`）
- VLM 图片路径失败（`vlm_image`）
- 启动诊断失败 / 服务崩溃

### 4.3 一键导出诊断包

三种方式：

```bash
# 1) 命令行
cd python && python -m edge_server.diagnostics --bundle --tag amd

# 2) HTTP
curl -X POST "http://127.0.0.1:8080/edge/diagnostics/bundle?tag=amd"

# 3) GUI：「菜单 → 设置 → 环境诊断 → 导出诊断包」
```

产出 `logs/diagnostics/edge-infer-diag-<时间>-<tag>.zip`，内含：

```
diagnostics.json   完整结构化报告
diagnostics.txt    可读摘要（贴 issue 用）
problems.jsonl     问题记录
edge-server.log    服务日志
```

### 4.4 隐私

- API Key / Token / `sk-*` / `github_pat_*` 一律 **`***redacted***`**；
- **不采集**任何 prompt 内容或文件内容；
- 日志为滚动文件（5 MB × 3），不会无限增长。

### 4.5 日志位置

| 文件 | 内容 |
|---|---|
| `logs/edge-server.log` | 服务日志（滚动，含启动环境快照） |
| `logs/problems.jsonl` | 结构化问题记录 |
| `logs/diagnostics/*.zip` | 导出的诊断包 |

启动时可设 `EDGE_DIAG_BOOT=1` 自动生成一份启动诊断包。

---

## 5. AMD 平台使用 SOP（给持有 AMD 机器的用户）

1. **先跑诊断**，确认现状：
   ```bash
   cd python && python -m edge_server.diagnostics
   ```
   看 `AMD GPU` 一行是 `detected` 还是 `not found`。

2. **若只有 NPU、没有 Radeon 独显**：当前版本无法加速，
   会回退 CPU。可选择：
   - 等 NPU 后端（见 §3，优先级 P2）；
   - 或把本项目的 OpenAI 兼容接口指向 AMD 的 Lemonade / FastFlowLM 服务
     （走「设置 → 推理源 → 使用外接 API」）。

3. **若有 Radeon 独显/核显**：可先手动验证 llama.cpp Vulkan 后端：
   ```bash
   # 下载 llama-bXXXX-bin-win-vulkan-x64.zip，解压后
   EDGE_LLAMA_DIR=/path/to/vulkan ./llama-bench.exe --list-devices
   ```
   若能看到 `Vulkan0: AMD Radeon ...`，说明 §2 的路线在你的机器上可行。

4. **遇到问题**：导出诊断包并回传。
   ```bash
   cd python && python -m edge_server.diagnostics --bundle --tag amd-<你的型号>
   ```

---

## 6. 任务清单

| 优先级 | 任务 | 状态 |
|---|---|---|
| — | 环境诊断模块 `diagnostics.py` | ✅ 已完成 |
| — | 结构化日志 + 问题记录 | ✅ 已完成 |
| — | 诊断 HTTP 端点 + GUI 一键导出 | ✅ 已完成 |
| — | 本文档 + 排障指南 | ✅ 已完成 |
| **P0** | llama.cpp ROCm/Vulkan 后端接入（AMD GPU） | ⬜ 待办 |
| **P0** | 设备探测识别 AMD GPU（`_probe_gpu_ids` 扩展） | ⬜ 待办 |
| **P0** | 模型目录增加 `amd-gpu` 变体 | ⬜ 待办 |
| **P0** | GUI 设备下拉增加「AMD GPU」 | ⬜ 待办 |
| P1 | AMD 实机验证 + 性能实测（补进 README 性能表） | ⬜ 待办 |
| P2 | AMD NPU 后端调研（ONNX Runtime + VitisAI EP） | ⬜ 待办 |
| P2 | 评估把 Lemonade / FastFlowLM 作为外接推理源 | ⬜ 待办 |

---

## 7. 参考资料

- OpenVINO 设备支持：[docs.openvino.ai](https://docs.openvino.ai/)（NPU/GPU 插件限 Intel）
- llama.cpp 硬件后端：`ggml-org/llama.cpp` Releases 提供
  `vulkan-x64` / `rocm-x64` / `sycl-x64` 等 Windows 构建
- AMD Ryzen AI Software：ONNX Runtime + VitisAI Execution Provider
- AMD Lemonade（Apache-2.0）：NPU(prefill) + iGPU(decode) 混合推理服务
- FastFlowLM：XDNA2 专用 NPU 运行时
