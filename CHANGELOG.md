# Changelog

本文件记录项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本 SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Planned
- Intel Arc iGPU 后端（Vulkan / SYCL）
- 100B+ MoE 分层参数支持
- Tauri 安装包（NSIS / WiX）

## [0.2.0] - 2026-11-20

项目化 Agent 工作流：把"单一沙箱"升级为多项目开发环境。

### Added
- **项目管理**：`create_project` / `open_project` / `list_projects` 工具，
  每个项目独立文件夹（含 `src/ data/ output/ notes/` 结构 + `.edge-project.json` 元数据），
  支持 `empty` / `python` 模板。
- **项目沙箱**：`sandbox_setup` 在项目内创建 Python venv（`.venv/`），
  `sandbox_run` 用沙箱解释器执行命令（`pip install`、`python xxx.py` 等）。
- **文件编辑工具**：`edit_file`（查找替换）、`append_file`（追加）。
- **GUI「项目」视图**：项目列表 + 文件树浏览 + 文件编辑器（打开/保存）+
  沙箱终端（在项目内跑命令）。
- 项目 HTTP 端点：`/project/list`、`/project/create`、`/project/open`、
  `/project/tree`、`/project/file`（GET/POST）、`/agent/sandbox_run`、
  `/project/sandbox_setup`。
- Agent 循环改进：保留完整工具调用/结果历史（scratchpad），多步任务更可靠；
  最终答复自动剥离 Qwen3 的 `think` 块。

### Changed
- `run_command` 白名单扩展为开发类命令（python/pip/git/npm/node/cargo/go/dotnet…），
  工作目录始终限定在项目内。

## [0.1.0] - 2026-11-20

首个公开版本：端侧三层卸载多硬件推理平台 + 本地 AI 工作台。

### Added

**Rust 三层卸载内核**（cargo workspace，4 crate）
- `edge-detect`：硬件探测（NVIDIA GPU / Intel NPU / CPU / SSD 带宽）
- `edge-tiers`：三层（SSD / RAM / VRAM）热度分配器，含 OOM 自动回退
- `edge-prefetch`：Prerouter 一步前瞻预取引擎
- `edge-engine`：组装层 + CLI，输出可序列化 RunPlan

**推理链路**
- 纯 OpenVINO 推理（`pure_ov_gen.py`）：仅依赖 openvino + tokenizers + numpy，
  无 torch/transformers；实现 stateful KV-cache 的 prefill + 增量解码
- OpenVINO GenAI 后端（`genai_backend.py`）：`LLMPipeline` 驱动 NPU/GPU/CPU，
  NPU→CPU 自动回退
- **流式输出**：`/v1/chat/completions` 支持 `stream=true`（OpenAI 兼容 SSE）

**本地 Agent（WorkBuddy 式）**
- `agent_tools.py`：文件系统工具（创建任务文件夹、读写文件、搜索、执行命令等），
  带 `workspace/` 沙箱隔离
- 端点：`/agent/chat`（工具循环）、`/agent/workspace`、`/agent/list`
- GUI「Agent 工作区」视图，可自然语言下达任务并可视化执行步骤

**服务与界面**
- OpenAI 兼容服务（`openai_server.py`，FastAPI）：`/v1/chat/completions`、
  `/v1/models`、`/health`，支持 API Key 鉴权
- 类 WorkBuddy 工作台 GUI：会话列表、对话 / Agent 工作区 / 三层方案 / 模型 / 设置
  五视图、设备选择器、推理源切换（本地 / 外接 OpenAI API）
- Tauri 2 原生窗口（设备选择器 + 对话 + 性能面板）
- SSD 真实探测（`storage_probe.py`）：CIM 读磁盘型号 + 多线程实测带宽
- ModelScope 模型下载（含 NPU 兼容模型映射）
- `run_windows.bat` 一键启动

### Fixed
- NPU 编译失败根因：Meteor Lake（NPU3720）不支持非对称 INT4（AWQ），
  需对称量化 + `LLMPipeline`；已提供可用模型清单
- 三层方案文案：明确区分「受硬盘带宽限制」与「不受硬盘限制（热专家全在显存）」
- 模型列表 NPU 兼容性判断错误（LLM IR 用 `openvino_model.xml`）
- Tauri windows-gnu 工具链 `dlltool` 缺失 → 改用完整 MinGW-w64
- Tauri v2 命令宏命名冲突 → 命令移入 `commands.rs` 子模块

[Unreleased]: https://github.com/OWNER/edge-infer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/OWNER/edge-infer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OWNER/edge-infer/releases/tag/v0.1.0
