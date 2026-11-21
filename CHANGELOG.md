# 更新日志

记录每个版本面向用户的主要变化。

## [Unreleased]

### 新增
- 支持 **Qwen3.8-Flash-Next**（180B 存储 / 6B 激活）：在 8 GB 显存 + 32 GB 内存
  的普通电脑上即可运行，支持 256K 上下文，长文本召回正常。
- 推荐 ncmoe=30 为稳定配置：prefill 16.9 / decode 3.6 tok/s。
- **架构自适应**：按模型自动匹配运行参数，无需手动设置。
- **KV cache 量化**与上下文配置：大上下文自动降低显存占用（256K 从 2.67 GB 降到 1.34 GB）。
- **三层卸载规划面板**：在设置里即可查看模型的分层建议，无需加载权重。

### 改进
- 更大的 MoE 模型（Qwen3.8-Flash-Next）现可自动识别分片并正确加载。
- 诊断与问题记录能力增强，导出诊断包更方便排查。

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

[Unreleased]: https://github.com/hellovm/edge-infer/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/hellovm/edge-infer/releases/tag/v0.2.1
[0.2.0]: https://github.com/hellovm/edge-infer/releases/tag/v0.2.0
[0.1.0]: https://github.com/hellovm/edge-infer/releases/tag/v0.1.0
