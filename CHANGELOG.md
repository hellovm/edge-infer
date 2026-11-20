# Changelog

本文件记录项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本 SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

v0.2.0 之后的增量（版本号待正式发布时再定）。

### Added
- **一键启动**：Tauri 原生窗口启动时自动定位项目根目录与 Python 解释器，
  后台拉起 `uvicorn` 推理服务（TCP 健康探测，就绪后通知前端）；普通用户
  双击 exe 即用，无需手动运行 `run_server.py`。
- **进程生命周期管理**：关闭窗口时结束子进程；即使被强制结束，Python 端的
  **父进程看门狗**（`EDGE_PARENT_PID`）也会在数秒内自我退出，不留残留进程。
- 新增 Tauri 命令 `start_service` / `service_status`（供前端手动重试）。
- GUI 启动状态提示：「正在启动推理服务…」→ 就绪 / 失败可点击重试。
- **模型推荐目录**（`model_catalog.py`）：NVIDIA / Intel GPU / Intel NPU 各
  2 个模型（一大一小），**以超大 MoE 为主**（利用三层卸载的优势），
  标注参数量、激活参数、量化方式、体积、内存建议与分层方式；新增
  `/edge/catalog` 端点（含已安装标记），点击卡片直接下载。
  支持 GGUF 仓库按文件白名单下载（避免误拉多个量化版本，80B 仓库全量 >3 TB）。
  另新增 `/edge/downloads` 查看下载状态。
- **llama.cpp 后端**（`llama_backend.py`）：支持 GGUF 模型，覆盖 OpenVINO IR
  无法承载的更大 MoE（80B-A3B 及以上）。以子进程方式管理 `llama-server`，
  通过 `--n-cpu-moe` 把 MoE 层放到 CPU/内存实现三层卸载；`/v1/chat/completions`
  自动按模型格式（GGUF vs OpenVINO IR）路由到对应后端，含 SSE 流式。
- **运行时设备切换**：`/edge/device/set`、`/edge/device/current`，
  NVIDIA GPU / Intel NPU / CPU / AUTO 真正生效。
- **思考模式开关**：关闭时过滤 `think` 块，开启时以折叠框展示思考过程。
- 对话区显示 **token 数 / tok-s / 推理设备**。
- 历史对话**归档 / 取消归档 / 删除**（可连带删除关联项目文件夹）。
- 输入框**附件上传**（文本内容随请求注入，`/agent/upload`）。
- 「关于」页开发者信息。
- 实测：NPU 可跑 9B 多模态（MiniCPM-V-4.5 `int4-cw`，对称量化 + `VLMPipeline`），
  首次加载含编译约 50s。

### Changed
- 布局重构：右侧专注对话，项目 / Agent / 三层方案 / 模型 / 设置收入左下角菜单弹层。
- `run_windows.bat` 简化为「优先启动原生窗口」；无原生窗口时回退「服务 + 浏览器」。
- 文档移除第三方产品名，统一为「本地 Agent」表述。

### Fixed
- **Tauri 前端误判服务离线**（核心 bug）：页面源 `http://tauri.localhost` 的
  asset server 对未知路径（如 `/health`）会 fallback 返回 `index.html`
  （HTTP 200 + HTML）。此前只检查 `r.ok`，误把它当作有效 API，导致后续请求
  拿到 HTML 解析失败、界面显示「服务未运行」。现改为**校验响应为 JSON 且
  `ok === true`**，并显式跳开 Tauri/file 资产源；`loadHealth` 失败时会自动
  重新探测 API 地址。
- **删除对话后界面残留**：空态原本用 `$("chatEmpty").outerHTML` 获取，但该元素
  首次渲染即被 `innerHTML` 覆盖销毁导致取不到；改为函数生成空态，
  删除当前对话后自动切到最近对话，无对话则新建空对话并清空输入框。
- 流式结束后「Agent 运行中」状态残留；SSE 末帧补齐 usage（token / tok-s / 设备）。
- 模型列表 NPU 兼容性判断错误（LLM IR 用 `openvino_model.xml`）。

## [0.2.0] - 2026-11-20

项目化 Agent 工作流：把「单一沙箱」升级为多项目开发环境。

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

**本地 Agent**
- `agent_tools.py`：文件系统工具（创建任务文件夹、读写文件、搜索、执行命令等），
  带 `workspace/` 沙箱隔离
- 端点：`/agent/chat`（工具循环）、`/agent/workspace`、`/agent/list`
- GUI「Agent 工作区」视图，可自然语言下达任务并可视化执行步骤

**服务与界面**
- OpenAI 兼容服务（`openai_server.py`，FastAPI）：`/v1/chat/completions`、
  `/v1/models`、`/health`，支持 API Key 鉴权
- 本地 AI 工作台 GUI：会话列表、对话 / Agent 工作区 / 三层方案 / 模型 / 设置
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

[Unreleased]: https://github.com/hellovm/-edge-infer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hellovm/-edge-infer/releases/tag/v0.2.0
[0.1.0]: https://github.com/hellovm/-edge-infer/releases/tag/v0.1.0
