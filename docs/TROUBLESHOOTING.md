# 疑难排查指南

> 适用于在**非开发机**上运行 edge-infer 时遇到的问题。
> 遇到任何异常，第一步永远是**导出诊断包**（见下），它能省掉大量来回问答。

---

## 0. 出问题时先做这一步

### GUI（推荐）

「菜单（左下角）→ 设置 → 环境诊断」：

- **刷新诊断** — 立即看到本机硬件与后端可用性
- **问题记录** — 展开已记录的后端失败 / 推理异常
- **导出诊断包** — 生成 zip，回传即可

### 命令行

```bash
cd python

# 看摘要
python -m edge_server.diagnostics

# 看原始 JSON
python -m edge_server.diagnostics --json

# 生成诊断包（回传这个文件）
python -m edge_server.diagnostics --bundle --tag <你的机器型号>
```

产物：`logs/diagnostics/edge-infer-diag-<时间>-<tag>.zip`

---

## 1. 诊断包里有什么

| 文件 | 用途 |
|---|---|
| `diagnostics.json` | 完整结构化报告（10 类探针 + 能力评估） |
| `diagnostics.txt` | 可读摘要，可直接贴到 issue |
| `problems.jsonl` | 结构化问题记录（含 traceback 与环境指纹） |
| `edge-server.log` | 服务日志（滚动，含启动快照） |

**隐私**：API Key / token / `sk-*` / `github_pat_*` 全部脱敏为 `***redacted***`；
不采集任何 prompt 或文件内容。

---

## 2. 日志文件位置

| 文件 | 内容 |
|---|---|
| `logs/edge-server.log` | 服务日志；单文件 5 MB，滚动保留 3 份 |
| `logs/problems.jsonl` | 每次失败一条 JSON |
| `logs/diagnostics/*` | 导出的报告与包 |

提高日志详细度：

```bash
EDGE_LOG_LEVEL=DEBUG python run_server.py
```

启动时自动生成诊断包：

```bash
EDGE_DIAG_BOOT=1 python run_server.py
```

---

## 3. 常见问题

### 3.1 「服务未运行」但进程还在

**历史原因**（v0.2.0 已修）：Tauri 的 asset server 对未知路径返回 `index.html`
（HTTP 200 + HTML），前端只检查 `r.ok` 会误判成有效 API。

**现在**：前端会校验响应是 JSON 且 `ok === true`。若仍出现：

1. 确认端口：`netstat -ano | findstr :8080`
2. 直接访问 `http://127.0.0.1:8080/health`
3. 看 `logs/edge-server.log` 最后 20 行

### 3.2 NPU 不可用 / 报 `Unsupported data type 'nf4'`

**原因**：Intel NPU（Meteor Lake NPU3720）只支持**对称 INT4**
（`int4-cw` / `int4-gq`），不支持 AWQ/NF4 等非对称量化。

**解决**：用 `*-int4-cw-ov` 仓库，或自行导出：

```bash
optimum-cli export openvino --model <hf-model> --weight-format int4 --sym --group-size 128 out/
```

诊断报告里 `Intel NPU` 一行会显示 `✅ 可用` 或 `❌ 不可用`。

### 3.3 在 AMD 机器上跑，没有任何加速

**这是预期行为**（当前版本）。OpenVINO 的 GPU / NPU 插件都是 Intel 专属，
AMD 上会安全回退 CPU。

诊断报告会明确写出：

```
AMD NPU     : detected  -> supported: NO
              reason: OpenVINO's NPU plugin is Intel-only ...
AMD GPU     : detected  -> via ['VULKAN']
```

详见 [`docs/AMD.md`](AMD.md)。

### 3.4 llama.cpp 路径很慢

诊断报告里看 `llama.cpp 后端` 一行：

- 期望 `CUDA`（NVIDIA）或 `VULKAN`/`ROCM`（AMD）；
- 若只看到 `CPU`，说明没加载到 GPU 后端 ——
  检查 `EDGE_LLAMA_DIR` 是否指向正确的后端构建。

```bash
# 手动确认 llama.cpp 能看到哪些设备
"$EDGE_LLAMA_DIR/llama-bench.exe" --list-devices
```

> 注意：核显上 ROCm 会**静默回退 CPU**，务必用上面的命令二次确认。

### 3.5 模型显示「下载中」但实际已停止

`*.incomplete` 临时文件会被识别为下载中。若下载已中断：

```bash
# 查看残留
find models -name "*.incomplete"

# 删除后重新下载（在 GUI 模型页点击下载）
```

诊断报告的 `models` 探针会列出每个模型目录的状态。

### 3.6 显存不足 / OOM

用三层卸载把 MoE 层放到内存：

```bash
# N 卡：把 48 层 MoE 放 CPU/内存
EDGE_N_CPU_MOE=48 python run_server.py
```

参考 README「性能实测」章节的 `ncmoe` 扫描结果
（8 GB 显存 + 32 GB 内存可跑 80B-A3B）。

---

## 4. 提交 issue 时请附上

1. **诊断包**（`logs/diagnostics/*.zip`）—— 最重要
2. 复现步骤（点了什么 / 发了什么请求）
3. 期望行为 vs 实际行为
4. 若与模型相关，说明模型来源与量化格式（如 `OpenVINO/Qwen3-8B-int4-cw-ov`）

有了诊断包，通常无需再追问环境细节。

---

## 5. 相关文档

- [`docs/AMD.md`](AMD.md) — AMD 平台支持规划
- [`README.md`](README.md) — 项目总览与性能实测
- [`CHANGELOG.md`](CHANGELOG.md) — 变更记录
