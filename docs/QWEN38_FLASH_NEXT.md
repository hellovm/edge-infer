# Qwen3.8-Flash-Next 支持方案

> 模型：`Qwen/Qwen3.8-Flash-Next`（2026-08-26 开源，Qwen4 架构预览）
> 目标：在 **8 GB 显存 + 32 GB 内存 + NVMe SSD** 上跑起来，并保持**大上下文**
> 状态：架构已验证 ✅ · 权重下载中 · 实测待补

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

### 5.1 推荐（保守起步）

```bash
export LLAMA_ATTN_ROT_DISABLE=1     # qwen4exp 不支持 upstream #21038 的 RoPE rotation

llama-server \
  -m models/qwen3.8-flash-next-gguf/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  -ngl 999 \
  -ncmoe 40 \
  -c 32768 \
  -ctk q8_0 -ctv q8_0 \
  -fa on
```

### 5.2 大上下文（256K）

```bash
llama-server \
  -m .../Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  -ngl 999 -ncmoe 44 \
  -c 262144 \
  -ctk q8_0 -ctv q8_0 \
  -fa on \
  --lazy-mode on
```

> KV cache: 256K × q8_0 ≈ 1.34 GB，加上 262144 上下文的其他开销，需留 3–4 GB。

### 5.3 通过本项目启动

```bash
cd python
EDGE_DEVICE="NVIDIA GPU" \
EDGE_MODEL_DIR="../models/qwen3.8-flash-next-gguf/UD-IQ1_S" \
EDGE_N_CPU_MOE=40 \
EDGE_LLAMA_DIR="D:/AI/tools/llama" \
python run_server.py
```

`llama_backend.py` 会自动加 `--n-cpu-moe`、`--ctx-size`，
并按 GGUF 类型走 llama.cpp 路径。

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
| **上游无 SWA 优化** | qwen4exp 的 dense attention 层 decode 成本随上下文线性增长。第三方 fork（llama.cpp-turboq-mtp）用 Sliding-Window Attention 解决（100K 上下文仍有 15.7 t/s） | 上游缺；可评估 fork |
| **RoPE rotation** | `LLAMA_ATTN_ROT_DISABLE=1` 可能必需 | 实测确认 |
| **i 卡不可行** | 无 OpenVINO IR 版；Vulkan 路径需 66 GB+ 显存/内存 | 排除 |
| **无 Ollama 支持** | Ollama 0.33.2 对该 GGUF 报校验错 | 用 llama.cpp 直连 |
| 首次加载慢 | 66 GB 从磁盘读入，首次启动需数分钟 | 正常 |

---

## 8. 待办

- [x] 验证 `qwen4exp` 架构支持
- [x] 核实 GGUF 元数据（层数/专家/上下文）
- [x] 确认 `--lazy-mode` / `--ple` / `--n-cpu-moe` 可用
- [x] 内存预算分析
- [x] 新增 `tier_plan.py` 规划工具
- [ ] **下载 66.4 GB 权重**（进行中）
- [ ] 实测启动 + 调优 `ncmoe`
- [ ] 长上下文实测（32K / 128K / 256K）
- [ ] 性能数据补进 README 性能表
- [ ] 评估第三方 SWA fork 的必要性

---

## 9. 参考

- 官方博客：https://qwen.ai/blog?id=qwen3.8-flash-next
- 上游 PR：ggml-org/llama.cpp #27742（qwen4exp 支持）
- `--ple` 支持 PR：#31185（b12233+）
- PLE 侧车文件说明：`Delgon/Qwen3.8-Flash-Next-PLE-GGUF`
- SWA fork：`Indras-Mirror/llama.cpp-turboq-mtp`
- 量化仓库：`unsloth/Qwen3.8-Flash-Next-GGUF`
