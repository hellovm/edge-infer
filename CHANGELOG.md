# 更新日志

记录每个版本面向用户的主要变化。

## [Unreleased]

### 新增
- **Windows 安装包（免装 Python）**：`installer/` 提供 Inno Setup 脚本与一键构建脚本，
  产出 `edge-infer-<版本>-setup.exe`。安装包内置**便携 Python 运行时**与全部依赖，
  目标机器无需预装 Python；自动创建桌面 / 开始菜单快捷方式与 `models\` 目录。
- **Anthropic 兼容接口** `/v1/messages`：可让 Claude Code / Anthropic SDK 直连本地引擎
  （支持流式 SSE 事件与 `x-api-key` / Bearer 双鉴权），与 `/v1/chat/completions` 复用同一后端。
- **版本更新检测**：设置 → 关于新增「检查更新」，查询 GitHub Releases 最新版本，
  提示新版并直达安装包下载（Tauri 走原生命令，浏览器走 GitHub API）。

### 改进
- **解释器查找更稳健**：`find_python()` 优先使用应用自带的便携运行时
  （`runtime\python`，并自动设置 `PYTHONPATH`），其次项目 `.venv`、用户级 Python、
  最后系统 `PATH`。**解决「换一台电脑后找不到 Python / 依赖」的问题**——依赖随应用走。
- 首次运行自动创建 `models\` 目录并写入说明，避免用户不清楚模型放哪。
- **专家画像持久化**（Strata 同款）：学习型缓存的专家热度画像可存盘 / 载入，
  跨会话免冷启动（`learned_cache` 新增 `save_profile` / `load_profile` + 单测）。

### 移除
- **移除 Qwen3-Next-80B 与 Qwen3-30B-A3B**：模型清单（`model_catalog.py`）、UI、
  README 中相应条目已删除；本地已下载的 46 GB `qwen3-next-80b-a3b-instruct-gguf`
  目录一并清理。

### 修复
- **NPU 运行 MiniCPM-V 4.5 纯文本无推理路径**：多模态 VLM 之前只有带图片时
  才走 VLMPipeline；纯文本查询会落入"无可用推理路径（模型类型 vlm）"降级。
  现 VLM 无论是否带图都走 VLMPipeline（内部 NPU→CPU 自动回退），纯文本也可用。
- **超级模式 256K 上下文启动失败**：大上下文的 KV cache 放显存会 OOM
  （8 GB 卡即使 q8_0 量化仍需 ~1.8 GB），现自动把 KV 放系统内存。
- **超级模式按钮竞态**：界面刚打开时硬件信息尚未加载完，点击会误报"硬件不支持"；
  现显示"检测中"，点击时自动等待检测结果。
- 模型加载期间给出明确提示（首次约 1-2 分钟），推理错误直接显示在对话里。

### 新增
- **外部服务对接说明**：设置页新增"OpenAI 兼容 API · 外部服务使用说明"，
  可填写 Base URL / 模型名（model_id）/ 上下文窗口（ctx），一键生成
  curl 与 Python 调用代码（含复制按钮），方便外部服务接入本地推理引擎。

### 诊断改进
- llama-server 的启动输出写入 `logs/llama-server.log`，加载失败可查。

## [0.3.1] - 2026-11-23

**本版本主题：超级模型模式 + 按内存分级的卸载策略。**

### 新增
- **超级模型模式**：检测到独立 N 卡 + SSD + 32G 内存时自动可用，一键启用
  Qwen3.8-Flash-Next 与 **256K 上下文**；硬件不足时按钮置灰并说明原因。
- **模型清单整理**：模型只显示名称，不再出现 `-gguf` / `-ov` 等实现后缀；
  超级模型在清单中带 ⚡ 标记，选中即开启超级模式。
- **按内存分级的卸载策略**：区分 16G 与 32G 电脑的三层卸载配置
  （ncmoe、KV 量化、上下文上限），并说明 iGPU / NPU 下的 RAM + SSD 加载策略。
- **16G 探索评估**：16G + N 卡 + SSD 也能运行 Qwen3.8-Flash-Next，
  但需降级（ncmoe=12、≤64K 上下文），界面会给出预期速度。

### 改进
- 非超级模式的上下文上限统一调整为 **64K**。
- 模型清单隐藏临时/测试目录，只显示可正常使用的模型。

## [0.3.0] - 2026-11-22

**本版本主题：让普通人的电脑也能本地运行超大规模模型，并以高能效方式运行。**

一台 8 GB 显存 + 32 GB 内存的普通 Windows 电脑，即可流畅运行 180B 级 MoE 大模型。
新增 **MTP 投机解码**，在保持输出完全不变的前提下进一步提速。

### 新增
- **MTP 投机解码**：模型自带的 draft head 预判多个候选 token，主模型一次并行校验，
  **输出与不开时逐字一致**，实测生成速度 **+17%**。带 draft 的模型自动启用，
  可在「设置 → 性能加速」开关。
- **旗舰模型组合：Qwen3.8-Flash-Next（180B 存储 / 6B 激活）+ MTP draft** ——
  在 8 GB 显存 / 32 GB 内存上开箱即用，应用自动配置三层卸载与 MTP 参数。
- **多量化档位实测支持**：UD-Q2_0 / UD-IQ1_S / UD-IQ4_XS 均可运行；
  三种量化速度几乎相同，瓶颈在 SSD 带宽而非精度。
- 设置新增「性能加速」面板，可开关 MTP 投机解码。

### 改进
- 后端对 Qwen3.8-Flash-Next 自动应用最优卸载参数（`ncmoe` / `tensor-split`），
  无需手动配置。
- 「关于」页面更新产品定位与架构说明。

### 许可
- 采用双许可：**个人使用免费，商业使用需获得授权**。详情见 LICENSE。

## [0.2.1] - 2026-11-20

聚焦"按设备自动匹配格式"与"双推理路线"，界面重构为可交付工作台。

- 新增中英文双版 README。
- 同一模型按硬件自动选择最快后端与格式，用户无需关心底层细节。
- 支持双击一键启动，自动拉起本地服务。
- 模型目录按硬件推荐，点击即可下载；新增下载状态显示。
- 设备选择可实时切换，对话区显示 token 数与速度。
- 修复速度统计、思考内容显示、模型下载状态判断等一批问题。

## [0.2.0] - 2026-11-20

项目化 Agent 工作流：从单一沙箱升级为多项目开发环境。

- 新增项目管理：独立文件夹、文件树浏览与编辑器、内置沙箱终端。
- Agent 支持多步工具调用，执行过程可视化。

## [0.1.0] - 2026-11-20

首个公开版本。

- 端侧三层卸载（SSD / 内存 / 显存）推理内核。
- 按硬件自动选择推理路径，支持 NVIDIA / Intel GPU / Intel NPU。
- OpenAI 兼容 API 与 SSE 流式输出。
- 本地 AI 工作台界面与 Tauri 原生窗口。
- 本地 Agent 工具、联网搜索、ModelScope 模型下载。

[0.3.1]: https://github.com/hellovm/edge-infer/releases/tag/v0.3.1
[0.3.0]: https://github.com/hellovm/edge-infer/releases/tag/v0.3.0
[0.2.1]: https://github.com/hellovm/edge-infer/releases/tag/v0.2.1
[0.2.0]: https://github.com/hellovm/edge-infer/releases/tag/v0.2.0
[0.1.0]: https://github.com/hellovm/edge-infer/releases/tag/v0.1.0
