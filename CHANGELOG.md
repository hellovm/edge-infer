# Changelog

本文件记录项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本 SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

（暂无）

## [0.2.1] - 2026-11-20

本次版本聚焦 **「同一模型按设备自动切换格式」** + **双推理路线** +
**UI 全面重构为可交付工作台**，并修复了一批下载状态与思考输出的问题。

### Added
- **一键启动**：Tauri 原生窗口启动时自动定位项目根目录与 Python 解释器，
  后台拉起 `uvicorn` 推理服务（TCP 健康探测，就绪后通知前端）；普通用户
  双击 exe 即用，无需手动运行 `run_server.py`。
- **进程生命周期管理**：关闭窗口时结束子进程；即使被强制结束，Python 端的
  **父进程看门狗**（`EDGE_PARENT_PID`）也会在数秒内自我退出，不留残留进程。
- 新增 Tauri 命令 `start_service` / `service_status`（供前端手动重试）。
- GUI 启动状态提示：「正在启动推理服务…」→ 就绪 / 失败可点击重试。
- **模型推荐目录**（`model_catalog.py`）：NVIDIA / Intel GPU / Intel NPU 各
  2 个模型，**以超大 MoE 为主**（利用三层卸载的优势），标注参数量、激活参数、
  量化方式、体积、内存建议与分层方式；新增 `/edge/catalog` 端点（含已安装标记），
  点击卡片直接下载。支持 GGUF 仓库按文件白名单下载（避免误拉多个量化版本，
  80B 仓库全量 >3 TB）。另新增 `/edge/downloads` 查看下载状态。
- **同一模型按设备自动匹配格式**（`model_catalog.py` 重构为「按模型组织 + variants」）：
  同一个模型名下挂 `nvidia` / `intel-gpu` / `npu` 多套变体，`resolve_device()`
  按当前硬件自动选出对应格式 —— N 卡取 **GGUF**，Intel GPU / NPU 取 **OpenVINO IR**，
  用户无需关心底层格式。
- **llama.cpp 后端**（`llama_backend.py`）：支持 GGUF 模型，覆盖 OpenVINO IR
  无法承载的更大 MoE（80B-A3B 及以上）。以子进程方式管理 `llama-server`，
  通过 `--n-cpu-moe` 把 MoE 层放到 CPU/内存实现三层卸载；`/v1/chat/completions`
  自动按模型格式（GGUF vs OpenVINO IR）路由到对应后端，含 SSE 流式。
- **双推理路线**：按硬件匹配最快的后端。实测（Qwen3-0.6B Q4）——
  llama.cpp CUDA **126.7 tok/s**、OpenVINO Intel iGPU 27.1、CPU 28.8，
  而 **NVIDIA 经 OpenVINO 只有 6.1 tok/s（慢约 20×）**。
  因此 NVIDIA → llama.cpp CUDA，Intel GPU / NPU → OpenVINO；
  选 N 卡却用 IR 模型时会明确提示改用 GGUF，不再静默走慢路径。
- **顶部设备 chip 可点击切换**：NVIDIA GPU / Intel NPU / CPU / AUTO 实时生效。
- **模型选择器**：输入区下拉，按当前设备过滤可用模型；「模型」页展示**支持的
  模型清单 + 下载状态**（✓ 已下载 / ⏳ 下载中 / 下载），点击卡片自助下载。
- **联网搜索**（`web_search.py`）：Playwright 驱动本机 Edge 抓取搜索结果，
  在对话上下文中注入，可与本地模型组合使用（开关式）。
- **输入区重构**：模型选择 + 文件 / 图片 / 文件夹上传 + 联网搜索开关，
  全部收敛到 composer 卡片。
- 对话区显示 **token 数 / tok·s⁻¹ / 推理设备**。
- 历史对话**归档 / 取消归档 / 删除**（可连带删除关联项目文件夹）。
- 「关于」页开发者信息。

### Changed
- 推荐模型目录：N 卡组改为 **GGUF（llama.cpp CUDA）**；卡片去掉「大/小」标注，
  只保留模型名称、介绍与推荐硬件规格。
- 模型选择器徽章改显示 **设备**（NVIDIA GPU / Intel NPU…），不再显示后端名。
- 布局重构：右侧专注对话，项目 / Agent / 三层方案 / 模型 / 设置收入左下角菜单弹层。
- 生成提示词改用**模型自身的 chat 模板**（`tokenizer.apply_chat_template`），
  不再写死 Qwen 模板。
- `run_windows.bat` 简化为「优先启动原生窗口」；无原生窗口时回退「服务 + 浏览器」。
- 文档移除第三方产品名，统一为「本地 Agent」表述。

### Fixed
- **思考内容泄漏到正文（llama.cpp 路径）**：开启「思考模式」时，整段推理过程
  被当作正文显示，并夹杂 `thinking…/think` 字样，「思考过程」折叠框始终为空。
  根因是 `_llama_stream()` 里开标签被写坏成字面量（丢了 `<` 和 `>`），
  前端 `splitThink()` 找 `<think` 永远匹配不到。修复：用 `chr(60)/chr(62)`
  拼接标签（避免工具链再吞尖括号）、改为状态机只输出一对 ` thinking…</think>`、
  先剥离模型自吐的字面标签、思考关闭时彻底丢弃 `reasoning_content`。
- **思考内容泄漏**：llama.cpp 路径下 Qwen3 的推理过程被当作正文输出，回答里出现
  `im_start` / `im_end` 等特殊标记 → 给 `llama-server` 加
  `--reasoning off --reasoning-format deepseek`，思考块不再混入正文。
- **思考模板误用**：`genai_backend.py` 原写死 Qwen `<|im_start|>` 模板，导致
  DeepSeek 等模型把模板标记当纯文本讨论 → 改用模型自带 chat 模板。
- **模型下载状态误判**：80B 模型只有 `.incomplete` 临时文件却被当成「已安装」
  → 新增 `_dir_state()` 三态（`ready` / `downloading` / `empty`），`*.incomplete`
  识别为下载中；`/edge/models` 增加 `state` / `usable` 字段，前端区分展示。
- **模型选择器永久为空**：`loadModelsForPicker` 只在 init 时调用一次，之后
  后台新下载的模型不出现 → 改为每次打开菜单都重新拉取。
- **Tauri 前端误判服务离线**（核心 bug）：页面源 `http://tauri.localhost` 的
  asset server 对未知路径（如 `/health`）会 fallback 返回 `index.html`
  （HTTP 200 + HTML）。此前只检查 `r.ok`，误把它当作有效 API，导致后续请求
  拿到 HTML 解析失败、界面显示「服务未运行」。现改为**校验响应为 JSON 且
  `ok === true`**，并显式跳开 Tauri/file 资产源；`loadHealth` 失败时会自动
  重新探测 API 地址。
- **删除对话后界面残留**：空态原本用 `$("chatEmpty").outerHTML` 获取，但该元素
  首次渲染即被 `innerHTML` 覆盖销毁导致取不到；改为函数生成空态，
  删除当前对话后自动切到最近对话，无对话则新建空对话并清空输入框。
- **联网搜索返回 0 条**：自定义 `User-Agent` 导致 Bing 返回不同页面结构
  → 改用浏览器默认 UA。
- 流式结束后「Agent 运行中」状态残留；SSE 末帧补齐 usage（token / tok·s⁻¹ / 设备）。
- 模型列表 NPU 兼容性判断错误（LLM IR 用 `openvino_model.xml`）。
- 路由误判：原按**激活参数**判定大小模型，导致 35B-A3B（激活 3B）被错判为小模型
  → 改为按**总参数量**判定。

### Notes
- **量化说明**：FP4/NVFP4 需 Blackwell（RTX 50 系，sm120）的 FP4 张量核；
  RTX 40 系（Ada，sm89）**无 FP4 通路**，NVFP4 退化为软件模拟可能更慢。
  Ada 卡宜用 FP8（vLLM/TensorRT-LLM）或 GGUF Q4_K_M/Q5_K_M。
- 实测：NPU 可跑 9B 多模态（MiniCPM-V-4.5 `int4-cw`，对称量化 + `VLMPipeline`），
  首次加载含编译约 50s；NPU 实测约 **18.98 tok/s**。

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

[Unreleased]: https://github.com/hellovm/edge-infer/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/hellovm/edge-infer/releases/tag/v0.2.1
[0.2.0]: https://github.com/hellovm/edge-infer/releases/tag/v0.2.0
[0.1.0]: https://github.com/hellovm/edge-infer/releases/tag/v0.1.0
