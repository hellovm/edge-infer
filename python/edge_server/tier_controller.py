"""自适应三层卸载规划器（超级模式）。

目标
----
家用电脑配置差异极大：显存 6–16 GB、内存 32–64 GB，还要和 Windows、浏览器、
IDE 抢内存。手工调 `--n-cpu-moe` 不可能覆盖这些组合，所以这里把"放哪里、放多少"
变成一道可解的约束优化题，**用实测标定的速度模型挑最优解**。

三层边界（本文件只负责 llama.cpp 这一路，即 N 卡 + CPU）：
    VRAM  非专家权重（注意力/SSM/共享专家/输出）→ 余量给专家层
    RAM   专家权重；KV 视余量放在 VRAM 或 RAM
    SSD   n-gram / PLE 表，按需读

> i 卡（OpenVINO）与 NPU 的实现在 `openvino_runtime.py` / `pure_ov_gen.py`，
> 本模块不引用、不修改它们。

速度模型（两条，都带置信度标注）
-------------------------------
A. 整层放置（当前二进制唯一可用）：按实测斜率外推
   `--n-cpu-moe 44`（4 层在卡）→ 1.760 t/s；`42`（6 层在卡）→ 2.219 t/s
   ⇒ tps ≈ 0.842 + 0.2295 × L（L = 显存里的专家层数）
   这是**局部线性外推**，L > 10 后不可信（已标注）。

B. 专家级缓存（需要 llama.cpp ≥ v0.7.0 的 `--moe-cache-mib`）：
   `tps ≈ base × (1 + 0.805 × hit)`，其中 0.805 由 PR #29887 实测标定
   （77% 命中 → 1.62×），hit 由 `strata-lab` 的复刻策略给出。
   这是跨硬件单点标定，只当**量级估计**。

两条都算，取大者，并在输出里写明用的是哪条、置信度如何。
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
from ctypes import wintypes
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import tier_plan as tp

GB = 1024 ** 3
MB = 1024 ** 2

# --- 实测标定常数（本机 RTX 4070 Laptop 8GB + Ultra 7 155H + 31.6GB）---------
TPS_AT_0_GPU_LAYERS = 0.842      # 外推截距
TPS_PER_GPU_LAYER = 0.2295       # 4→6 层的实测斜率
CACHE_MULTIPLIER_K = 0.805       # PR #29887: 77% 命中 → 1.62×
KV_Q8_FACTOR = 0.53              # q8_0 相对 f16 的近似体积比
KV_Q4_FACTOR = 0.29

# --- 实测：flash-attn + KV 量化在 Qwen4Exp 混合注意力上是净损失 ---------------
# ctx 8192、3 层专家在卡、长输出：
#   -fa off + KV f16（内存）  0.627 t/s
#   -fa on  + KV q8_0（显存） 0.511 t/s   → 量化路径慢 18%
# 原因：48 层里只有 12 层全注意力，36 层 GDN 的收益抵不过量化/fa 的开销。
# 所以"省显存"和"跑得快"在这里是两件事，不能默认开 fa。
FA_PENALTY = 0.511 / 0.627       # ≈0.815，只应在短上下文（≤32K）套用
FA_PENALTY_MAX_CTX = 32768
VRAM_KV_BONUS = 1.03             # KV 在显存省的算力收益（128K 实测 1.033 vs 1.040 近似）

# --- 安全余量 ---------------------------------------------------------------
VRAM_SAFETY_MB = 420             # 桌面合成/驱动/其他进程
VRAM_DESKTOP_MB = 1150           # 实测：桌面+驱动基线占用（8GB 卡上约 1.1 GB）
VRAM_FLOOR_MB = 320              # 低于此值一定会 stall
UBATCH_MIN = 64
UBATCH_DEFAULT = 512


# ---------------------------------------------------------------------------
# 硬件探测
# ---------------------------------------------------------------------------
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


@dataclass
class HostProfile:
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    ram_total_gb: float = 0.0
    ram_avail_gb: float = 0.0
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_name: str = ""
    gpu_name: str = ""
    source: str = "probe"

    @property
    def vram_free_mb(self) -> int:
        return max(0, self.vram_total_mb - self.vram_used_mb)

    def describe(self) -> str:
        return (f"{self.gpu_name or 'GPU'} VRAM {self.vram_total_mb} MB"
                f"（空闲 {self.vram_free_mb} MB） | RAM {self.ram_total_gb:.1f} GB"
                f"（可用 {self.ram_avail_gb:.1f} GB） | {self.cpu_name} "
                f"{self.cpu_cores}C/{self.cpu_threads}T")


def probe_host() -> HostProfile:
    h = HostProfile()
    st = _MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    h.ram_total_gb = st.ullTotalPhys / GB
    h.ram_avail_gb = st.ullAvailPhys / GB

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            name, total, used = [x.strip() for x in out.splitlines()[0].split(",")]
            h.gpu_name, h.vram_total_mb, h.vram_used_mb = name, int(total), int(used)
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Processor | Select-Object -First 1 "
             "Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            j = json.loads(out)
            h.cpu_name = (j.get("Name") or "").strip()
            h.cpu_cores = int(j.get("NumberOfCores") or 0)
            h.cpu_threads = int(j.get("NumberOfLogicalProcessors") or 0)
    except Exception:
        pass
    return h


# ---------------------------------------------------------------------------
# 模型几何（只读 GGUF 头，不加载权重）
# ---------------------------------------------------------------------------
@dataclass
class ModelProfile:
    path: str
    arch: str = ""
    n_layer: int = 0
    n_expert: int = 0
    top_k: int = 0
    vram_fixed_bytes: int = 0        # 注意力/SSM/共享专家/输出/norm
    expert_bytes: int = 0            # 全部专家
    ple_bytes: int = 0               # SSD 层的 n-gram 表
    kv_bytes_per_token_f16: int = 0
    ctx_native: int = 0
    has_trellis: bool = False

    @property
    def expert_per_layer(self) -> int:
        return self.expert_bytes // max(1, self.n_layer)

    def kv_bytes(self, ctx: int, dtype: str = "f16") -> int:
        f = {"f16": 1.0, "q8_0": KV_Q8_FACTOR, "q4_0": KV_Q4_FACTOR}.get(dtype, 1.0)
        return int(self.kv_bytes_per_token_f16 * ctx * f)

    def describe(self) -> str:
        return (f"{Path(self.path).name}  {self.arch}  {self.n_layer} 层 × "
                f"{self.n_expert} 专家 top-{self.top_k} | 固定 {self.vram_fixed_bytes/GB:.2f} GB"
                f" | 专家 {self.expert_bytes/GB:.2f} GB"
                f"（{self.expert_per_layer/MB:.0f} MB/层）"
                f" | KV {self.kv_bytes_per_token_f16/1024:.1f} KB/token(f16)")


def probe_model(path: str, ple_sidecar: Optional[str] = None) -> ModelProfile:
    model = tp.read_model(path)
    meta = model.get("meta", {}) or {}
    groups = tp.classify(model["tensors"])
    mp = ModelProfile(path=path, arch=str(meta.get("general.architecture", "")))

    def g(key: str, default: int = 0) -> int:
        v = meta.get(key, default)
        return int(v) if isinstance(v, (int, float)) else default

    mp.n_layer = g(f"{mp.arch}.block_count", 0)
    mp.n_expert = g(f"{mp.arch}.expert_count", 0)
    mp.top_k = g(f"{mp.arch}.expert_used_count", 0)
    mp.ctx_native = g(f"{mp.arch}.context_length", 0)
    mp.vram_fixed_bytes = int(groups.get("attention", {}).get("bytes", 0)
                             + groups.get("other", {}).get("bytes", 0))
    mp.expert_bytes = int(groups.get("experts", {}).get("bytes", 0))
    mp.ple_bytes = int(groups.get("ngram_table", {}).get("bytes", 0))

    # KV：只有全注意力层需要 cache，SSM/GDN 层是固定状态
    interval = g(f"{mp.arch}.full_attention_interval", 1) or 1
    n_attn = max(1, mp.n_layer // interval) if interval > 1 else mp.n_layer
    kvh = g(f"{mp.arch}.attention.head_count_kv", 0)
    klen = g(f"{mp.arch}.attention.key_length", 0)
    vlen = g(f"{mp.arch}.attention.value_length", 0)
    if kvh and klen and vlen:
        mp.kv_bytes_per_token_f16 = n_attn * kvh * (klen + vlen) * 2

    if ple_sidecar and os.path.isfile(ple_sidecar):
        try:
            p = tp.read_model(ple_sidecar)
            mp.ple_bytes += sum(int(t.get("bytes", 0)) for t in p["tensors"])
        except Exception:
            pass

    mp.has_trellis = any("TQ" in (t.get("type_name") or "")
                         for t in model["tensors"])
    return mp


# ---------------------------------------------------------------------------
# 规划
# ---------------------------------------------------------------------------
@dataclass
class LaunchPlan:
    ctx: int
    n_gpu_layers: int
    n_cpu_moe: int
    kv_location: str               # vram | ram
    kv_type: str
    flash_attn: bool
    moe_cache_mib: int
    threads: int
    ubatch: int
    batch: int
    expected_tps: float
    speed_model: str               # whole-layer | cache
    vram_used_mb: int
    ram_used_gb: float
    headroom_vram_mb: int
    headroom_ram_gb: float
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def args(self, exe: str, model_path: str) -> List[str]:
        a = [exe, "-m", model_path, "-c", str(self.ctx),
             "-ngl", str(self.n_gpu_layers), "-t", str(self.threads),
             "-ub", str(self.ubatch), "-b", str(self.batch),
             "-ctk", self.kv_type, "-ctv", self.kv_type,
             "--parallel", "1", "--no-warmup"]
        # flash-attn 只在需要 KV 量化时才开：短上下文下它是净损失（实测 -18%）
        if self.flash_attn:
            a += ["-fa", "on"]
        if self.n_cpu_moe:
            a += ["--n-cpu-moe", str(self.n_cpu_moe)]
        if self.kv_location == "ram":
            a += ["--no-kv-offload"]
        if self.moe_cache_mib:
            a += ["--moe-cache-mib", str(self.moe_cache_mib)]
        return a

    def summary(self) -> str:
        return (f"ctx {self.ctx:>7} | 显存专家层 {self.n_gpu_layers:>2}/"
                f"{self.n_cpu_moe + self.n_gpu_layers:>2} | KV {self.kv_location}"
                f"/{self.kv_type} | fa {'on' if self.flash_attn else 'off'} | "
                f"缓存 {self.moe_cache_mib} MiB | "
                f"t{self.threads} ub{self.ubatch} | "
                f"预计 {self.expected_tps:.2f} t/s | "
                f"VRAM {self.vram_used_mb} MB (余 {self.headroom_vram_mb}) | "
                f"RAM {self.ram_used_gb:.1f} GB (余 {self.headroom_ram_gb:.1f})")


def _host_budget(host: HostProfile, reserve_ram_gb: Optional[float] = None):
    """Windows + 后台应用的可用额度。

    显存看"当前空闲"（驱动和桌面已经占走的部分拿不回来）。

    内存**不能**看"当前可用"：模型 mmap 之后，它的页面就进了 Windows 的
    standby/page cache，反而让 `ullAvailPhys` 变小——用它做预算会得出
    "内存只够十几层"的错误结论（实测 48 层全驻留时解码只读盘 0.18 MB/token）。
    所以按**容量**规划：总内存减去给 OS 与应用的余量；当前可用只用于告警。
    """
    # 显存预算按"总量 − 桌面基线"算，而不是"当前空闲"：
    # 实测 8GB 卡上 6 层专家的峰值是 7041 MB（= 8188 − 1147），
    # 用当前空闲会低估（历史进程残留会让 used 虚高）。
    baseline = max(VRAM_DESKTOP_MB, host.vram_used_mb)
    vram_budget = host.vram_total_mb - baseline - 120
    vram_free = max(0, vram_budget)
    if reserve_ram_gb is None:
        reserve_ram_gb = max(5.0, min(8.0, host.ram_total_gb * 0.20))
    ram_capacity = max(0.0, host.ram_total_gb - reserve_ram_gb)
    ram_now = max(0.0, host.ram_avail_gb - 1.5)   # 只看是否已有强压力
    return max(0, vram_free), ram_capacity, ram_now, reserve_ram_gb


def _threads_for(host: HostProfile) -> tuple[int, List[str]]:
    """大小核机器上，线程数不是越多越好（实测 12 线程比 6 慢 17%）。"""
    notes = []
    if not host.cpu_cores:
        return 6, ["未探测到 CPU 拓扑，按 6 线程保守设定"]
    # 155H = 6P + 8E + 2LPE：只用 P 核最快
    if "155H" in host.cpu_name or "125H" in host.cpu_name:
        notes.append("检测到 Intel 大小核（Meteor Lake-H），只用 6 个 P 核")
        return 6, notes
    t = max(4, min(host.cpu_cores, 12))
    if t < host.cpu_threads:
        notes.append(f"按物理核数取 {t} 线程（HT/SMT 对这类内存受限负载无益）")
    return t, notes


# 真实路由轨迹的实测锚点（2026-11-24，Qwen3.8 Gyro-S，48 层 × 512 专家 top-10，
# 134 个 token；静态最优命中率 vs 等效缓存容量）。
#
# 之前用合成 Zipf(α=1.1) 拟合出"4 GB → 58%"，实测量到只有 20.3%：
# 真实路由远比假设均匀（137 个 token 内每层 251/256 个专家对都被用到）。
# 用合成值会把预计速度抬高 3 倍，导致规划器推荐了实际并不值得的缓存配置。
_HIT_ANCHORS = ((0.0, 0.0), (0.98, 0.056), (2.02, 0.108),
                (4.03, 0.203), (6.45, 0.309), (12.9, 0.570))
# 自适应策略实测只吃到静态最优的约 59%（4 GB：11.9% vs 20.3%），
# 因为它每轮换入过于激进、且没有真实热度先验。
_POLICY_EFFICIENCY = 0.60


def _hit_rate(cache_mb: int) -> float:
    """专家热缓存命中率的估计（真实轨迹锚点 + 分段线性插值）。"""
    if cache_mb < 512:
        return 0.0
    gb = cache_mb / 1024
    if gb >= _HIT_ANCHORS[-1][0]:
        return min(0.62, _HIT_ANCHORS[-1][1]) * _POLICY_EFFICIENCY
    for (g0, h0), (g1, h1) in zip(_HIT_ANCHORS, _HIT_ANCHORS[1:]):
        if g0 <= gb <= g1:
            t = (gb - g0) / (g1 - g0) if g1 > g0 else 0.0
            return (h0 + (h1 - h0) * t) * _POLICY_EFFICIENCY
    return 0.0


def _fit_plan(host: HostProfile, model: ModelProfile, ctx: int,
              cache_supported: bool) -> LaunchPlan:
    vram_free, ram_capacity, ram_now, reserve = _host_budget(host)
    threads, tnotes = _threads_for(host)

    # 1) 固定部分（非专家权重）优先上卡；放不下就整体降级
    fixed_mb = model.vram_fixed_bytes / MB
    buffer_mb = 520          # 计算缓冲（随 ubatch 变化，取典型值）
    vram_left = vram_free - fixed_mb - buffer_mb
    warnings: List[str] = []
    notes: List[str] = list(tnotes)

    if vram_left < 0:
        warnings.append(
            f"非专家权重 {fixed_mb:.0f} MB + 缓冲放不进显存（可用 {vram_free:.0f} MB）："
            "将退回部分层到 CPU，速度会显著下降")
        vram_left = 0

    # 2) KV：在 (f16 / q8_0) × (显存 / 内存) 四个候选里打分
    #    实测依据（ctx 8192，3 层专家在卡）：
    #      f16 KV + 关 fa = 0.627 t/s   ← 快
    #      q8_0 KV + 开 fa = 0.511 t/s  ← 慢 18%
    #    所以短上下文要主动关掉 flash-attn，别为了省显存牺牲速度；
    #    只有上下文长到 f16 放不下时，才走 q8_0 + fa。
    best = None
    for ktype in ("f16", "q8_0"):
        need_mb = model.kv_bytes(ctx, ktype) / MB
        for loc in ("vram", "ram"):
            if loc == "vram" and need_mb > vram_left - 512:
                continue                    # 至少给专家层留 512 MB
            if loc == "ram" and (ram_capacity - model.expert_bytes / GB
                                 - need_mb / 1024) < 0.5:
                continue                    # 内存要留 0.5 GB 余量
            score = 1.0
            if ktype == "f16" and ctx <= FA_PENALTY_MAX_CTX:
                score *= 1.0 / FA_PENALTY   # 关 fa 的收益，只对短上下文成立
            if loc == "vram":
                score *= VRAM_KV_BONUS
            score *= 1.0 - 0.05 * min(1.0, need_mb / max(1.0, vram_left))
            if best is None or score > best[0]:
                best = (score, loc, ktype, need_mb)

    if best is None:
        kv_location, kv_type, kv_need = "ram", "q8_0", model.kv_bytes(ctx, "q8_0") / MB
        warnings.append("KV 在显存和内存都放不下：建议缩小上下文或再加内存")
    else:
        _, kv_location, kv_type, kv_need = best
        if kv_location == "vram":
            vram_left -= kv_need
        notes.append(
            f"KV {kv_type} 放{kv_location}（{kv_need/1024:.2f} GiB），"
            f"flash-attn {'开' if kv_type == 'q8_0' else '关'}"
            + ("：短上下文下 f16 KV 比量化快 18%" if kv_type == "f16" else ""))

    flash_attn = kv_type == "q8_0"       # KV 量化依赖 flash-attn

    kv_ram_gb = (kv_need if kv_location == "ram" else 0) / 1024
    ram_left = ram_capacity - kv_ram_gb
    if ram_left < 0:
        warnings.append("KV 放内存后内存不足：建议减小上下文")
    if ram_now < model.expert_bytes / GB * 0.5 and ram_now < 8:
        notes.append(f"当前可用内存仅 {host.ram_avail_gb:.1f} GB（其余在页面缓存里），"
                     "若其他应用持续增长，可能触发换页")

    # 3) 专家层：整层放置 vs 专家级缓存，取预计速度高者
    per_layer_mb = model.expert_per_layer / MB
    max_gpu_layers = 0 if per_layer_mb <= 0 else int(max(0, vram_left) // per_layer_mb)
    max_gpu_layers = max(0, min(max_gpu_layers, model.n_layer))
    ram_layers_capacity = int(max(0.0, ram_left) * 1024 // per_layer_mb) if per_layer_mb else 0
    cpu_layers = model.n_layer - max_gpu_layers
    if cpu_layers > ram_layers_capacity:
        warnings.append(
            f"专家层需 {cpu_layers} 层驻留内存（{cpu_layers*per_layer_mb/1024:.1f} GB），"
            f"内存只够 {ram_layers_capacity} 层：超出部分每 token 从 SSD 读，速度会被 IO 拖住")

    tps_whole = TPS_AT_0_GPU_LAYERS + TPS_PER_GPU_LAYER * max_gpu_layers
    if max_gpu_layers > 10:
        notes.append("整层速度模型超出实测区间（>10 层），预测仅供排序")

    # 3b) 支持 GPU 专家缓存时：在"整层"与"缓存"之间**分配**显存
    #     实测/复刻依据：strata-lab 的命中率曲线 h ≈ 0.342·GB^0.379
    #     （4 GB→58%，2 GB→45%，1 GB→34%），乘法收益来自 PR #29887 的 1.62×。
    model_kind, tps, n_gpu_layers, cache_keep = "whole-layer", tps_whole, max_gpu_layers, 0
    if cache_supported:
        best = (tps_whole, max_gpu_layers, 0, 0.0)
        for L in range(0, max_gpu_layers + 1):
            remain = vram_left - L * per_layer_mb
            cache_mb = int(max(0, remain - 180))
            hit = _hit_rate(cache_mb)
            cand = ((TPS_AT_0_GPU_LAYERS + TPS_PER_GPU_LAYER * L)
                    * (1 + CACHE_MULTIPLIER_K * hit))
            if cand > best[0]:
                best = (cand, L, cache_mb, hit)
        tps, n_gpu_layers, cache_keep, best_hit = best
        if cache_keep:
            model_kind = "cache+layers" if n_gpu_layers else "cache"
            notes.append(f"显存分配：{n_gpu_layers} 层整层 + {cache_keep} MiB 热缓存"
                         f"（预计命中 {best_hit*100:.0f}%）")

    # 4) ubatch：长上下文时 prefill 缓冲会吃掉显存，按需缩小
    ub = UBATCH_DEFAULT
    if ctx >= 65536:
        ub = 128
    if ctx >= 131072:
        ub = 64
    batch = max(512, ub * 4)

    vram_used = int(fixed_mb + buffer_mb + (kv_need if kv_location == "vram" else 0)
                    + n_gpu_layers * per_layer_mb + (cache_keep if model_kind == "cache" else 0))
    ram_used = (model.expert_bytes - n_gpu_layers * model.expert_per_layer) / GB + kv_ram_gb

    return LaunchPlan(
        ctx=ctx, n_gpu_layers=n_gpu_layers,
        n_cpu_moe=model.n_layer - n_gpu_layers,
        kv_location=kv_location, kv_type=kv_type, flash_attn=flash_attn,
        moe_cache_mib=(cache_keep if model_kind.startswith("cache") else 0),
        threads=threads, ubatch=ub, batch=batch,
        expected_tps=tps, speed_model=model_kind,
        vram_used_mb=vram_used, ram_used_gb=ram_used,
        headroom_vram_mb=int(vram_free - vram_used),
        headroom_ram_gb=round(ram_capacity - ram_used, 2),
        warnings=warnings, notes=notes)


def plan_super(host: HostProfile, model: ModelProfile, ctx: int,
               cache_supported: bool = False) -> LaunchPlan:
    """超级模式：在安全边界内最大化预计速度。"""
    return _fit_plan(host, model, ctx, cache_supported)


def simulate_matrix(model: ModelProfile, ctx: int,
                    vrams=(6, 8, 12, 16), rams=(32, 48, 64),
                    cache_supported: bool = False) -> List[dict]:
    """给不同家用配置生成对照表（用于验证策略的单调性与边界）。"""
    rows = []
    for v in vrams:
        for r in rams:
            h = HostProfile(vram_total_mb=v * 1024, vram_used_mb=int(0.9 * 1024),
                            ram_total_gb=float(r), ram_avail_gb=r * 0.72,
                            cpu_cores=6, cpu_threads=12, cpu_name="sim")
            p = plan_super(h, model, ctx, cache_supported)
            rows.append({"vram_gb": v, "ram_gb": r, "tps": round(p.expected_tps, 2),
                         "gpu_layers": p.n_gpu_layers, "kv": p.kv_location,
                         "cache_mib": p.moe_cache_mib, "model": p.speed_model,
                         "vram_used": p.vram_used_mb,
                         "ram_used": round(p.ram_used_gb, 1),
                         "warn": len(p.warnings)})
    return rows


def _main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="自适应三层卸载规划（超级模式）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ple", default="")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--cache", action="store_true",
                    help="假设二进制支持 --moe-cache-mib")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--matrix", action="store_true", help="输出 6-16G × 32-64G 对照表")
    a = ap.parse_args(argv)

    host = probe_host()
    model = probe_model(a.model, a.ple or None)
    hdr = sys.stderr if a.json else sys.stdout
    print("主机:", host.describe(), file=hdr)
    print("模型:", model.describe(), file=hdr)
    print(file=hdr)

    if a.matrix:
        rows = simulate_matrix(model, a.ctx, cache_supported=a.cache)
        print(f"{'显存':>5} {'内存':>5} {'显存专家层':>10} {'KV':>5} {'缓存MiB':>8} "
              f"{'速度模型':>12} {'预计t/s':>9} {'RAM占用':>8}")
        print("-" * 78)
        for r in rows:
            print(f"{r['vram_gb']:>4}G {r['ram_gb']:>4}G {r['gpu_layers']:>10} "
                  f"{r['kv']:>5} {r['cache_mib']:>8} {r['model']:>12} "
                  f"{r['tps']:>8.2f} {r['ram_used']:>7.1f}G")
        return 0

    p = plan_super(host, model, a.ctx, a.cache)
    if a.json:
        print(json.dumps(asdict(p), ensure_ascii=False, indent=2))
        return 0
    print("→ 超级模式方案")
    print(" ", p.summary())
    for n in p.notes:
        print("  ·", n)
    for w in p.warnings:
        print("  ⚠", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
