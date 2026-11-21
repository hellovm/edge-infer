# Changelog

本文件记录项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本 SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added
- **Qwen3.8-Flash-Next 实测跑通**（`qwen4exp`，180B 存储 / 6B 激活）：
  在 **8 GB 显存 + 32 GB 内存 + SSD** 上用三层卸载成功加载并推理，
  66.38 GB 权重（UD-IQ1_S）实测通过。
  - `--n-cpu-moe 30` 为实测最优点：prefill 15.5 / decode 1.93 tok/s（p512）
  - **256K 上下文可用**（`n_ctx: 262144` 确认）；2K 提示长文召回测试通过
  - 详见 `docs/QWEN38_FLASH_NEXT.md`
- **架构自适应后端**（`llama_backend.py`）：
  - `_read_arch()` 只读 GGUF 头解析 `general.architecture`，60 GB 文件也是毫秒级
  - `_ARCH_FLAGS` 表按架构自动注入必需参数 —— `qwen4exp` 自动加
    `LLAMA_ATTN_ROT_DISABLE=1`（该架构不支持上游 RoPE rotation）
  - 非思考模式自动加 `--no-reasoning-preserve`：实测发现 qwen4exp 的
    思考通道会吃光 token 预算（24 token 预算产出空回答）
  - 新增 `kv_quant` / `ctx` / `extra_args` 配置项
- **KV cache 量化**：`EDGE_KV_QUANT`（默认按窗口自动决定），
  `EDGE_CTX` 控制上下文（默认 32K，最大 1M）。
  大上下文时自动用 `q8_0`，256K KV cache 从 2.67 GB 降到 1.34 GB。
- 新增 `python/edge_server/tier_plan.py`（三层卸载规划，只读 GGUF 头）
  与 GUI「设置 → 三层卸载规划」面板。
- 新增 `docs/QWEN38_FLASH_NEXT.md`：完整方案（规格 / 预算 / 调优 / 实测 / 限制）。

### Fixed
- **分片 GGUF 选错分片（关键）**：`_find_gguf()` 原来返回**最大的单个** .gguf，
  对分片模型会返回 shard-2（37.57 GB），而 llama.cpp 必须拿到 **shard-00001**
  （元数据与张量表在那里）才能加载，否则报 `failed to load GGUF split`。
  实测 Qwen3.8-Flash-Next 时暴露：端到端只返回降级响应。
  现改为识别 `-0000N-of-0000M` 命名并优先返回**分片 1**；
  单文件模型保持原「取最大」行为，并跳过 mmproj 与未下完的文件。
- **GGUF 类型表缺新增类型**：`Q1_0`(41) / `Q2_0`(42) 未收录导致权重虚高
  2.9 倍（190.9 GB vs 实际 66.4 GB）；`IQ4_NL` 块大小应为 32 而非 256。
- **tier_plan 支持分片与未下完的文件**：自动聚合 `-0000N-of-0000M`，
  `.incomplete` 也能读（header 在文件开头）。
- **分层决策**：n-gram 表是纯查表，超预算时整表下 SSD 优先于任何专家，
  不为它分配 VRAM。

## [Unreleased]（上一批）

### Added
- **三层卸载规划工具** `edge_server/tier_plan.py`：**只读 GGUF 头与张量表**
  （不加载权重），对任意大小模型秒出结论 —— 分组体积（n-gram 表 / MoE 专家 /
  注意力 / 其它）+ 逐组分层建议（VRAM / RAM / SSD）+ 是否需 SSD 参与。
  自带的 GGUF 解析器零依赖（仅 `struct`），能安全跳过超大字符串数组。
  CLI：`python -m edge_server.tier_plan <model.gguf> --vram 8 --ram 32`；
  HTTP：`GET /edge/tier_plan?model=<dir>&vram=8&ram=32`。
- **Qwen3.8-Flash-Next（`qwen4exp`）支持方案** `docs/QWEN38_FLASH_NEXT.md`：
  从 GGUF 元数据核实规格（48 层 = 36 GDN + 12 QSA / 512 专家 top-10 /
  256K 原生上下文）、三层内存预算、启动配置、已知限制。
  - 关键结论：**只有 12 个 QSA 层有 KV cache** → 256K 上下文仅需 2.67 GB（f16），
    是同规模 dense 的 1/4；
  - n-gram 表（28.8 GB）是**查表非矩阵乘**，本机 llama.cpp 的 `--lazy-mode`
    默认 `auto`（>4 GiB 张量按需读）正好命中 → 表留在 SSD，不占常驻内存。
- README 中英双版新增「三层卸载规划工具」「超大 MoE：Qwen3.8-Flash-Next」章节；
  目录结构同步。

### Added（上一批）
- **环境诊断模块** `edge_server/diagnostics.py`：10 类探针（OS / CPU / 内存 /
  NVIDIA / OpenVINO / 厂商 NPU / 后端 import / llama.cpp 设备 / SSD / 模型目录）
  并发执行、逐项隔离，输出结构化 JSON + 可读摘要，并给出**能力评估**
  （明确哪些路径可用、哪些不可用及原因）。含 API Key / token 脱敏。
- **问题记录** `logs/problems.jsonl`：后端加载失败、推理异常、降级响应、
  模型下载失败自动落盘，含 traceback 与**环境指纹**。
- **诊断包导出**：`python -m edge_server.diagnostics --bundle`、
  `POST /edge/diagnostics/bundle`、GUI「设置 → 环境诊断 → 导出诊断包」，
  产出含 `diagnostics.json` / `.txt` / `problems.jsonl` / `edge-server.log` 的 zip。
- **统一日志** `edge_server/logging_setup.py`：滚动文件日志（5 MB × 3）
  + 主线程/子线程未捕获异常钩子；启动时输出硬件能力快照。
- 新增 HTTP 端点：`GET /edge/diagnostics`、`POST /edge/diagnostics/bundle`、
  `GET /edge/problems`、`POST /edge/problems/clear`。
- 新增环境变量 `EDGE_LOG_LEVEL`、`EDGE_DIAG_BOOT`。
- 新增文档：`docs/AMD.md`（AMD 平台支持规划）、`docs/TROUBLESHOOTING.md`（排障指南）。

### Changed
- `run_server.py` 接入统一日志与启动诊断；改用 `log_config=None` 避免
  uvicorn 覆盖日志配置。
- 降级响应体不再只回显 prompt，而是**明确说明原因**
  （模型目录不存在 / GGUF 缺 llama-server / IR 缺 openvino_genai /
  请求 NPU 但本机无 Intel NPU），并指向诊断命令。
- 两版 README 新增「环境诊断与问题记录」「AMD 平台支持（规划中）」章节；
  环境变量表与目录结构同步更新。
- `.gitignore` 忽略 `logs/`。

### Fixed
- **降级路径静默无记录**：此前走到 `_degrade_text()` 时用户只看到一个回显，
  既没有日志也没有可排查的线索。现在降级会记录问题条目并在响应中说明原因。

## [0.2.1] - 2026-11-20

本次版本聚焦 **「同一模型按设备自动切换格式」** + **双推理路线** +
**UI 全面重构为可交付工作台**，并修复了一批下载状态与思考输出的问题。

### Changed
- README 性能表由 **Qwen3-0.6B** 换为 **Qwen3-8B** 实测，并新增超大 MoE
  三层卸载章节（80B-A3B 在 8 GB 显存上 `--n-cpu-moe` 扫描结果）。
- README 特性表精简：移除 CPU 两行与「NVIDIA via OpenVINO (GPU.1)」行
  （转为表下脚注），NPU 行标注具体平台 **AI Boost · Core Ultra 7 155H**；
  中英双版同步。
- 设备别名归一化（`_norm_dev`）：`NVIDIA GPU` / `nvidia_gpu` / `NVIDIA-GPU`
  等等价写法统一路由，避免文档里的自然写法落空。
- README 新增中英双版：`README.md`（简体中文，默认）+ `README_EN.md`（English）。
- 新增 `scripts/bench_llama.py` / `scripts/bench_ov.py` 性能基准脚本。

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
- **速度统计严重失真**：llama.cpp 路径原来用「字符数 ÷ 2」估算 token 与
  tok/s，慢模型会被大幅低估（80B 实测被报成 0.36 tok/s，真实为 5.0）。
  改为向 llama-server 请求 `stream_options.include_usage`，直接采用其
  真实 `usage` 与 `timings`（`predicted_per_second` / `prompt_per_second`），
  并在 SSE 末帧补充 `prefill_tok_s` / `ttft_ms`。
- **NPU 可用性校验缺失**：此前显式请求 `NPU` 时不检查硬件是否存在。
  新增 `_NPU_AVAILABLE`（由 OpenVINO `available_devices` 探测），
  无 NPU 的机器（含 **AMD XDNA / Ryzen AI**，OpenVINO 不枚举该设备）
  自动回退 CPU；模型列表的设备徽章与 `/edge/devices` 也据此修正。
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
