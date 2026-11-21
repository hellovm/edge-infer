# Qwen3.8-Flash-Next 支持方案

> 模型：`Qwen/Qwen3.8-Flash-Next`（2026-08-26 开源，Qwen4 架构预览）
> 目标：在 **8 GB 显存 + 32 GB 内存 + NVMe SSD** 上跑起来，并保持**大上下文**
> 状态：**✅ 已实测跑通** —— 权重 66.38 GB 已下载，三层卸载生效，256K 上下文可用

---

## 1. 模型规格（已从 GGUF 元数据核实）

```
general.architecture                        = qwen4exp
qwen4exp.block_count                        = 48
qwen4exp.context_length                     = 262144        (256K 原生, YaRN 可扩展)
qwen4exp.embedding_length                   = 2560
qwen4exp.attention.head_count               = 24
qwen4exp.attention.head_count_kv            = 2
qwen4exp.expert_count                       = 512
qwen4exp.expert_used_count                  = 10            (+1 shared)
qwen4exp.expert_feed_forward_length         = 640
qwen4exp.embedding_length_per_layer_input   = 160           (n-gram 表每行宽度)
```

| 项目 | 值 |
|---|---|
| 层结构 | 48 层 = **36 × Gated DeltaNet (GDN)** + **12 × Qwen Sparse Attention (QSA)** |
| 参数 | 125B 主干 + **51B n-gram 表** + 4B MTP = 180B 存储 |
| 激活 | **6B / token** |
| 上下文 | 262,144 原生；YaRN 扩展至 1,000,000 |
| 其他 | Gated Residual（4 分支）、Muon 优化器、多模态（视觉编码器） |

---

## 2. 为什么这个架构特别适合三层卸载

三个结构性优势，都是**白送**的：

### 2.1 GDN 层没有 KV cache

48 层里只有 **12 层**（每 4 层 1 个）是 QSA，需要 KV cache；
其余 36 层是 GDN 线性注意力，**状态固定大小、不随上下文增长**。

| 上下文 | KV cache (f16) | KV cache (q8_0) |
|---|---|---|
| 32K | 0.33 GB | 0.17 GB |
| 64K | 0.67 GB | 0.33 GB |
| 128K | 1.33 GB | 0.67 GB |
| **256K** | **2.67 GB** | **1.34 GB** |

对比：同规模 dense 模型需要 **4×** 的 KV cache。**256K 上下文只花 2.67 GB** ——
这就是「保持大上下文支持」能成立的根本原因。

### 2.2 n-gram 表是**查表**，不是矩阵乘

`per_layer_token_embd.weight` 有 3.2 亿行 × 160 值（51B 参数），但
**每个 token 只读 16 行**（几 KB），且模式随机 ——
对显存带宽几乎无要求，可以安心放 SSD。

llama.cpp 原生支持这一点（见 §3.2）。

### 2.3 专家稀疏度极高

512 个专家，每 token 只激活 **10 个 + 1 个共享**（约 2%）。
专家权重占绝大部分体积，但访问极其稀疏 —— 典型的三层卸载受益场景。

---

## 3. llama.cpp 侧的支撑

### 3.1 架构支持 ✅

本机 llama.cpp `b12559`（build 12559, commit 7e960f40f）：

```
llama.dll 内含符号: qwen4exp  qwen3next  qwen3moe  qwen3vl
```

上游 PR #27742（2026-08-27，模型发布次日）合并，稳定性自 b12233 起可用。

### 3.2 `--lazy-mode` —— SSD 层的实现机制

```
--lazy-mode MODE    on-demand reading of certain tensors,
                    for example per-layer embeddings  (default: auto)
  - on    : read rows from disk on demand (requires mmap)
  - auto  : on, but only for tensors larger than 4 GiB   ← 默认
  - off   : always keep them resident
```

**默认的 `auto` 已经够用**：n-gram 表 28.8 GB > 4 GiB，
所以开箱即用就会按需读取，不必常驻内存。

> 更新版本/fork 另有 `on-direct`（跳过 page cache 直读），
> 本机 b12559 无此取值，用 `on` 即可。

### 3.3 `--ple` —— 外挂独立的 n-gram 表

```
--ple, --shard FNAME,...    extra GGUF shards for the model
```

用于**单独替换** n-gram 表精度（如主干用 Q4_K_XL + 表用 Q8_0），
不重下整个模型。本例中 UD-IQ1_S 的 `-00003-of-00003` **就是**内置的
IQ4_NL 表，因此**无需** `--ple`。

> 已从 unsloth 文档核实：UD-IQ1_S / IQ1_M / Q2_0 / IQ2_XS 的表 shard
> 均为 `-00003-of-00003`，类型 IQ4_NL。

### 3.4 `--n-cpu-moe` —— 专家卸载

```
-ngl    N     期望放进 VRAM 的层数（可设 999，会自动取实际能放下的量）
-ncmoe  N     前 N 层的 MoE 专家权重留在 CPU/RAM
-nkvo   1     不把 KV cache 放显存
-ctk/-ctv      KV cache 量化类型（q8_0 / q4_0）
-c     N      上下文长度
```

---

## 4. 内存预算（本机）

### 4.1 权重分解（UD-IQ1_S，66.38 GB）

| 分片 | 内容 | 大小 |
|---|---|---|
| `-00001-of-00003` | 元数据 + 少量张量 | 0.01 GB |
| `-00002-of-00003` | 主干（125B 参数） | **37.57 GB** |
| `-00003-of-00003` | n-gram 表（51B 参数，IQ4_NL） | **28.80 GB** |

### 4.2 三层分配方案

| 层 | 容量 | 放什么 | 依据 |
|---|---|---|---|
| **VRAM** | 8 GB | 注意力 + norm + 输出层（约 1.7 GB）+ 部分专家 | 每 token 必用，收益最高 |
| **RAM** | 32 GB | 主干剩余部分（专家权重） | 访问稀疏 |
| **SSD** | 1.1 TB | **n-gram 表 28.8 GB**（lazy 按需读） | 查表，每 token 只读 16 行 |

**关键算术**：
```
主干 37.57 GB   vs   VRAM 8 + RAM 32 = 40 GB   → ✅ 主干能完全驻留
n-gram 28.8 GB  →  SSD, lazy 按需读           → ✅ 不占常驻内存
KV cache 256K   →  2.67 GB (f16) / 1.34 GB (q8_0) → ✅ 大上下文无忧
```

### 4.3 风险点

- **RAM 实际可用量**：32 GB 物理内存，扣掉 OS + 应用后常驻约 **20–25 GB**。
  若主干无法完全驻留，多余部分会落到 SSD —— 此时**每一层的前向都会触发磁盘读**，
  速度会显著下降（SSD 带宽 ≠ 内存带宽）。
- **缓解**：优先保证注意力层 + 高频专家在 VRAM/RAM；必要时降上下文
  或改用更激进的 KV 量化。

---

## 5. 启动配置

### 5.0 实测结果（本机：RTX 4070 8GB + 32GB + NVMe）

**权重已完整下载 66.38 GB，模型成功加载并推理。**

```
llama_model_loader: qwen4exp A3B IQ1_S - 1.5625 bpw
                   61.81 GiB / 176.94 B params      ← 架构与参数量均正确识别
KV self size    = n_ctx 262144 确认 ✅
```

| 配置 | Prefill tok/s | Decode tok/s | 备注 |
|---|---|---|---|
| `ncmoe=30`（r=2） | **16.88 ± 1.07** | **3.57 ± 0.03** | ✅ **稳定最优点** |
| `ncmoe=24`（r=2） | 16.87 ± 1.20 | 3.15 ± 0.04 | 次优，decode 略低 |
| `ncmoe=36`（r=2） | 13.44 ± 0.68 | 3.97 ± 0.02 | pp 下降，decode 略升 |
| `ncmoe=38`（r=3） | 13.84 ± 0.68 | 2.91 ± 0.06 | — |
| `ncmoe=42`（r=3） | 17.63 ± 1.88 | 3.89 ± 0.72 | ⚠️ 方差大，已越过内存边界 |
| `ncmoe=44`（r=2/3） | 43.98 ± 7.12 / 46.01 ± 10.30 | 3.46 ± 0.04 | ⚠️ pp 虚高、±10 抖动，不可信 |
| `ncmoe=48`（r=2） | 43.45 ± 12.78 | 3.20 ± 0.04 | ⚠️ ±12 抖动，不可信 |
| `ncmoe=40`（r=2） | 10.30 ± 0.49 | 4.85 ± 0.05 | ⚠️ 下面三次复测全部推翻此值 |
| `ncmoe=40`（r=3 复测 ×2） | 13.90 / 12.98 | 2.59 / 2.06 | ❌ decode 从 4.85 掉到 ~2，抖动剧烈 |
| `ncmoe=20` | OOM | — | ❌ `cudaMalloc failed`，显存不足 |

> 测法：`llama-bench -p 512 -n 128`，`LLAMA_ATTN_ROT_DISABLE=1`，
> CUDA(ngl=999)。r=2/3 为重复次数。
>
> **关于「省电模式」复测（2026-11-21）**：用户怀疑之前处于省电模式，
> 要求重测。核查发现电源方案一直是 **「平衡」**（`381b4222…`），CPU 未限频，
> 并非省电模式。但重测数值确比首轮偏高（首轮 ncmoe=30 decode 1.93 → 本轮 3.57，
> +85%），差异主要来自**首轮未预热 + 单次采样偶然性**，而非电源设置。
>
> **关键结论**：ncmoe≥40 时，CPU 侧需约 30 GB 专家权重常驻，
> 已超过本机空闲物理内存（~21 GB），触发 SSD/RAM 换页。
> 表现是 decode 在 2.0–4.9 之间剧烈抖动、pp 虚高且 ±10 波动，
> **这些数字不可复现、不可信**。稳定可用区间是 **ncmoe 24–36，推荐 30**。

**长上下文召回测试通过** ✅

```
提示 2014 tok（内文首行埋入「紫罗兰-7391」，中间填充 2000 行无关文本）
问题：本次项目的内部代号是什么？
回答：紫罗兰-7391          ← 准确召回
```

**256K 上下文可用** ✅

```bash
curl http://127.0.0.1:8091/props
# → n_ctx: 262144
```

> **速度解读**：稳定 decode 约 **3.5 tok/s**（首轮单次采样曾低至 1.9，属未预热偶然值）。
> 瓶颈是**权重总量 61.81 GiB > VRAM 8 + RAM 32**，相当一部分专家权重必须从 SSD 分页读取。
> 这是物理约束，不是配置问题。ncmoe≥40 会越过空闲内存边界触发抖动，
> 数字看似更高但不可复现；实测最优点是 ncmoe=30。若内存升到 64 GB，
> 可预期 decode 稳定提升到 5+ tok/s。

### 5.1 推荐（保守起步）

```bash
export LLAMA_ATTN_ROT_DISABLE=1     # qwen4exp 不支持 upstream #21038 的 RoPE rotation

llama-server \
  -m models/qwen3.8-flash-next-gguf/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  -ngl 999 \
  -ncmoe 30 \
  -c 32768 \
  -ctk q8_0 -ctv q8_0 \
  --no-reasoning-preserve            # 思考模式会吃光 token 预算
```

> **注意**：`--n-cpu-moe 30` 是实测最优点。更小的值（20）会挤压显存，
> 更大的值（44/48）会把过多专家推到慢层，**两头都变慢**。

### 5.2 大上下文（256K）

```bash
llama-server \
  -m .../Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  -ngl 999 -ncmoe 30 \
  -c 262144 \
  -ctk q8_0 -ctv q8_0 \
  --no-reasoning-preserve
```

> KV cache 实测：256K × q8_0 ≈ 1.34 GB，加载正常。
> 但**长提示的 prefill 极慢**（10 tok/s），
> 11 万 token 的提示约需 3 小时 —— 实用上下文建议 ≤ 32K。

### 5.3 通过本项目启动

```bash
cd python
EDGE_DEVICE="NVIDIA GPU" \
EDGE_MODEL_DIR="../models/qwen3.8-flash-next-gguf/UD-IQ1_S" \
EDGE_N_CPU_MOE=30 \
EDGE_LLAMA_DIR="D:/AI/tools/llama" \
python run_server.py
```

`llama_backend.py` 会自动加 `--n-cpu-moe`、`--ctx-size`，
并按 GGUF 类型走 llama.cpp 路径。

> ⚠️ 需手动补 `LLAMA_ATTN_ROT_DISABLE=1` 与 `--no-reasoning-preserve`
> （当前 `llama_backend.py` 尚未内置，见待办）。

---

## 6. 规划工具

新增 `python/edge_server/tier_plan.py` —— **不加载权重**（只读 GGUF 头 + 张量表），
对任意大小的模型都能算：

```bash
cd python
python -m edge_server.tier_plan <model.gguf> --vram 8 --ram 32
```

### 6.1 本模型实测输出

```
架构        : qwen4exp  (GGUF v3, 1224 tensors / 3 shards)
  qwen4exp.block_count                        = 48
  qwen4exp.context_length                     = 262144
  qwen4exp.expert_count                       = 512
  qwen4exp.expert_used_count                  = 10
  qwen4exp.embedding_length_per_layer_input   = 160

权重总计    : 66.37 GB          ← 与实际文件 66.38 GB 吻合

--- 分组 ---
  ngram_table       28.80 GB       1 tensors   查表用，可放 RAM/SSD
  experts           33.40 GB     144 tensors   MoE 专家，可按热度分层
  attention          1.94 GB     336 tensors   注意力，常驻快层
  other              2.23 GB     743 tensors   embedding / norm / 输出层

--- 分层建议 (VRAM 8 GB + RAM 32 GB) ---
  attention      -> VRAM       [VRAM 1.94 GB]
  other          -> VRAM       [VRAM 2.23 GB]
  experts        -> RAM        [VRAM 3.83 GB + RAM 29.57 GB]
  ngram_table    -> SSD        [SSD 28.80 GB]

合计: VRAM 8.00 / RAM 29.57 / SSD 28.80 GB
```

### 6.2 关键点

- **支持分片与未下完的文件**：自动聚合 `-0000N-of-0000M` 全部分片；
  header 在文件开头，所以 `.incomplete` 也能读（下载途中即可规划）。
- **分层决策逻辑**：n-gram 表是**纯查表**（非矩阵乘），
  超预算时**整表下 SSD 优先于任何专家** —— 它常驻快层几乎无收益，
  而专家是矩阵乘、受益于快内存。
- 权重算术已用 80B / Flash-Next 两个模型交叉校验。

HTTP 端点：`GET /edge/tier_plan?model=<dir>&vram=8&ram=32`
GUI：「设置 → 三层卸载规划」

### 6.3 踩坑记录

实现时踩到一个真坑：**GGUF 类型表需要跟上 llama.cpp 的新增类型**。
`Q1_0`(41) / `Q2_0`(42) 是较新的 1/2-bit simple-block 类型
（Q2_0 = 64 值/块 18 字节 = 2.25 bpw），表里缺失会按默认 4 字节计算，
导致总权重虚高到 **190.9 GB**（实际 66.4 GB，差 2.9 倍）。
另外 `IQ4_NL` 的块大小是 **32** 而非 256，也需修正。

---

## 7. 已知限制

| 限制 | 说明 | 状态 |
|---|---|---|
| **速度 ~3.5 tok/s** | 权重 61.81 GiB > VRAM 8 + RAM 32，部分专家需从 SSD 分页。物理约束，非配置问题 | 需 64 GB 内存改善 |
| **ncmoe≥40 抖动** | CPU 侧专家需 ~30 GB > 空闲内存 ~21 GB，触发换页，decode 在 2.0–4.9 间剧烈波动、不可复现 | 锁定 ncmoe=30 |
| **长提示 prefill 慢** | 10 tok/s，11 万 token 约需 3 小时。实用上下文建议 ≤ 32K | 可接受 |
| **上游无 SWA 优化** | qwen4exp 的 dense attention 层 decode 成本随上下文线性增长。第三方 fork（llama.cpp-turboq-mtp）用 Sliding-Window Attention 解决（100K 上下文仍有 15.7 t/s） | 上游缺；可评估 fork |
| **RoPE rotation** | 必须 `LLAMA_ATTN_ROT_DISABLE=1` | 已确认 |
| **思考模式吃 token** | 默认开启，短 `max_tokens` 会导致空回答。加 `--no-reasoning-preserve` 或 `/no_think` | 已确认 |
| **i 卡不可行** | 无 OpenVINO IR 版；Vulkan 路径需 66 GB+ 显存/内存 | 排除 |
| **无 Ollama 支持** | Ollama 0.33.2 对该 GGUF 报校验错 | 用 llama.cpp 直连 |
| **mmap + CPU 覆盖的提示** | 加载时警告 `consider using --load-mode none` —— 但那会一次性占用全部内存，**不建议** | 忽略即可 |
| 首次加载耗时 | 66 GB 从磁盘读入，实测约 45 秒（mmap 惰性） | 正常 |

---

## 8. 待办

- [x] 验证 `qwen4exp` 架构支持
- [x] 核实 GGUF 元数据（层数/专家/上下文）
- [x] 确认 `--lazy-mode` / `--ple` / `--n-cpu-moe` 可用
- [x] 内存预算分析
- [x] 新增 `tier_plan.py` 规划工具（含 GUI 面板）
- [x] **下载 66.38 GB 权重**
- [x] **实测启动**（ncmoe 20/30/40/44/48 全部试过，30 最优）
- [x] **长上下文验证**（256K 可加载；2K 提示召回通过）
- [x] 性能数据记录进本文档
- [ ] **把 `LLAMA_ATTN_ROT_DISABLE` 与 `--no-reasoning-preserve` 内置进 `llama_backend.py`**
      （目前需手动设置；应按模型架构自动检测 `qwen4exp` 并加参数）
- [x] 评估第三方 SWA fork 的必要性（见 §10；长上下文必要，短上下文收益有限）
- [x] 性能数据摘要补进 README 性能表
- [ ] MTP / EAGLE 投机解码实测（需专门分支 + MTP GGUF，见 §10.2）

---

## 10. 加速路线调研（2026-11-21）

### 10.1 第三方 SWA fork：`Indras-Mirror/llama.cpp-turboq-mtp`

非官方 llama.cpp 分支（MIT，~94 star），专为 qwen4exp 提供三件事：
- **SWA**：把多数 dense 全注意力层改为滑动窗口（默认 4096），保留 8 个 Bresenham
  分布的全局层维持长程召回。解决「dense 层每步扫全部历史 → decode 随上下文线性变慢」。
- **MTP draft head**（`blk.48`，4B/512-expert）：`--spec-type draft-mtp`。
- **TBQ4 4-bit KV**（`-ctk tbq4_0 -ctv tbq4_0`）+ PLE-on-SWA。

fork 在 **RTX 4090 24GB** 实测：prefill 200–235 tok/s，decode 22.8 t/s @12K →
**15.7 t/s @100K**，100K 长程召回通过。启动需加：
`--override-kv "qwen4exp.attention.sliding_window=int:4096,qwen4exp.attention.swa_global_layers=int:8"`。

**对本机的判断**：本机瓶颈是专家权重从 SSD 分页（物理约束），SWA 优化的是
>32K 长上下文的注意力衰减。**短上下文（≤32K）收益有限；做 100K 级长文档时这是唯一可行方案。**
代价：非上游分支、需重编译（`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`）、有维护风险。

### 10.2 MTP / EAGLE 投机解码

原理：草稿器（n-gram / MTP 头 / EAGLE-3）一次猜 K 个 token，目标模型一次并行验证。
**无损**（输出与不投机一致）。接受率 α≥70% 时提速 1.5–2×；α<60% 反而变慢。

公开实测（注意均非本机配置）：
| 来源 | 配置 | 结果 |
|---|---|---|
| dev.to A100 | Flash-Next + MTP | avg 32.6 → **56.1 tok/s**（1.72×），额外 ~6 GiB |
| Qwen3.6-27B A10G 24G | MTP | 25 → 45 tok/s（+78%） |
| fork RTX 4090 | Flash-Next MTP | 74.1 t/s（vs 44–50），接受率 0.52–0.67 |

**对 8GB 显存场景的关键分析**：投机解码在**IO/带宽受限**场景理论上收益更大
（验证 K 个 token 只读一遍专家权重，摊薄 SSD 分页成本）；但 MoE 不同 token 激活的
专家集合部分重叠，实际节省约 2–3× 而非 4×。约束是 MTP 头额外占 3–6 GiB，
会把 ncmoe 从 30 进一步压低、加剧显存压力 —— 存在权衡，必须实测接受率。
上游已合并 MTP（PR #22673），但 qwen4exp 的 MTP 需专门构建
（`danielhanchen/llama.cpp` 的 `qwen4exp/mtp` 分支 + 单独 MTP GGUF）或上述 fork。

### 10.3 NPU 协同：现阶段对 Flash-Next 不可行

本机诊断确认：Intel Core Ultra 7 155H + **Intel AI Boost NPU**（Meteor Lake 第一代，
OpenVINO 2026.4 可见，~11 TOPS）+ Arc iGPU + RTX 4070 8GB + 32GB RAM。

- llama.cpp 后端只有 **CPU / CUDA / RPC**，**无 OpenVINO NPU 后端** →
  Flash-Next 的任何层都无法交给 NPU。
- NPU SRAM **不可扩展**，只适合 3–7B INT4 小模型；66.38 GB 的 Flash-Next 完全装不下。
- 第三方 `openvino-llama.cpp` backend 存在但仅支持小模型 + Q4_K_M/Q5_K_M，
  **不支持 IQ1_S 等 importance-weighted 量化**。
- 理论上可行但工程未支持：**异构投机解码**（NPU 跑小 draft 模型，CUDA 跑大模型验证，
  即 SSD / Speculative Speculative Decoding）——llama.cpp 目前草稿与验证必须同后端，做不到。

**结论**：NPU 帮不上 Flash-Next 这个量级的模型。它的合理用途是独立跑小模型
（3–7B INT4，NPU 18–28 tok/s @ 11W）做低功耗任务，或与独显分工而非协同同一模型。
**最高优先级仍是升级到 64GB 内存**（消除 SSD 分页，decode 预期 5+ tok/s），
其次才是 MTP 与 SWA。

---

## 9. 参考

- 官方博客：https://qwen.ai/blog?id=qwen3.8-flash-next
- 上游 PR：ggml-org/llama.cpp #27742（qwen4exp 支持）
- `--ple` 支持 PR：#31185（b12233+）
- PLE 侧车文件说明：`Delgon/Qwen3.8-Flash-Next-PLE-GGUF`
- SWA fork：`Indras-Mirror/llama.cpp-turboq-mtp`
- 量化仓库：`unsloth/Qwen3.8-Flash-Next-GGUF`

---

## 11. AirLLM 调研（2026-11-21，与本项目对比）

> 结论先行：**AirLLM 与本项目目标不同——它解决「极小显存能否跑超大模型」，
> 本项目解决「8GB 显存 + 32GB 内存下尽量快」。同模型（Flash-Next）本项目快约一个数量级。
> 不建议集成进本项目主干，其价值限于「低于本方案硬件门槛」的兜底场景。**

### 11.1 项目概况
- `lyogavin/airllm`，Apache-2.0，~33.8k star（2026-09），pip 一行安装。
- 版本：v4.0.0（2026-09-05）。2026/08 加入 **Qwen3.8-Flash-Next** 支持
  （`Qwen4ExpForConditionalGeneration`，125B + 51B n-gram，实测 **5.95 GB 显存**），
  2026/09 加入流式 LoRA 训练（125B 在 RTX 3060 Ti 6GB 内可训）。

### 11.2 实现方式（四条机制）
| 机制 | 做法 | 代价 |
|---|---|---|
| 分层加载 | checkpoint 按层切 shard；模型在 meta 设备实例化（0 显存），forward hook 在每层计算前把权重从盘搬到 GPU、算完即释放 | 每 token 重读全模型，磁盘 IO 成瓶颈 |
| 块级量化 | bitsandbytes NF4(4bit)/absmax(8bit)，只量化权重不量化激活，离线压盘 | **与预取互斥**；GPU 反量化开销 |
| 预取 | 单 worker 线程，算层 N 时读层 N+1 到 pinned memory；官方称 ~10% 提速 | 开量化即被禁用 |
| 逐专家流式(v3.0) | MoE 层用 safetensors seek 只读 token 路由到的专家（K3 每层 896 专家只路由 16 个，单层 55GB→实际读 ~1GB） | 路由调度开销；batch>1 时 IO 放大 |

Flash-Next 特例：51B n-gram 表以 **file-map 留在主机内存**（64GB 机器够），decoder 层照常流式；
需 Git 版 transformers（内置 qwen4_exp）+ 约 **360 GB checkpoint 磁盘**。

### 11.3 真实性能（官方只报显存、从不报速度）
速度上限 = 存储读速 ÷ 每 token 权重读取量。Gen4 NVMe ~7GB/s：
- 70B fp16（140GB）→ 理论下限 ~20 s/token；4bit（35GB）→ ~5 s/token；SATA SSD 更慢到分钟级。
社区实测：GB10 + Qwen2.5-32B 无压缩 13.3 s/token，4bit 反慢至 106.7 s/token（issue #330，
因存储够快时反量化只有成本没有收益）；70B 4bit + 热页缓存 ~0.3 tok/s；
消费级 SSD + 70B 约 0.5–2 tok/s。**同 Flash-Next：本项目 3.6 tok/s，快一个数量级。**

### 11.4 与本项目对比 & 决策
- 项目定位高度一致（低门槛跑大模型），但**技术路线相反**：AirLLM 把压力全转嫁磁盘，
  本项目用 32GB 内存常驻换速度。对 16–32GB 内存的普通用户，本项目已在优势区间。
- **不集成**。理由：① 同硬件速度差一个数量级；② 依赖 PyTorch/transformers Git 版，
  与本项目「不依赖 torch、纯 llama.cpp/OpenVINO」的轻量链路冲突；③ 360GB 磁盘门槛过高。
- **可借鉴**：逐专家流式（seek 只读路由专家）思想，与 llama.cpp `--lazy-mode` +
  ncmoe 专家分层同源；若未来支持显存 <8GB 的兜底设备可参考其 per-expert 思路。
- **适用场景（供文档标注）**：显存低于 6GB、能接受秒级/token、离线批处理/论文复现。
