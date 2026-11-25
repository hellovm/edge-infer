"""llama.cpp backend — runs GGUF models (incl. MoE beyond OpenVINO's 35B ceiling).

Why this exists
---------------
OpenVINO IR top out at ~35B-A3B on ModelScope; **larger MoE (80B / 235B / 397B)
only exist as GGUF**. llama.cpp is also the most mature place for 3-tier offload:
``--n-cpu-moe N`` keeps N MoE layers in CPU/RAM while the rest stay on GPU, and
mmap'd weights stream from SSD on demand — exactly the SSD/RAM/VRAM split this
project is about.

Usage
-----
Start ``llama-server`` as a subprocess with an OpenAI-compatible endpoint, then
proxy to it. We deliberately reuse llama-server's own HTTP API rather than
re-implementing inference.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator, Optional

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=16)
def _supports_flag(exe: str, flag: str) -> bool:
    """Whether this llama-server build knows a flag.

    Feature flags move fast: ``--moe-cache-mib`` (the GPU LRU cache that lets
    RAM-resident experts be computed *on the GPU*) landed in llama.cpp v0.7.0 /
    PR #29887, while the Agention fork we build is pinned at 2026-10-07 and does
    not have it. Passing an unknown flag makes llama-server exit immediately, so
    probe the binary once and cache the answer.
    """
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True,
                             timeout=30, errors="replace")
        return flag in (out.stdout or "")
    except Exception:
        return False

# Where we look for the prebuilt binaries (downloaded from llama.cpp releases).
# The Qwen3.8-Flash-Next build (with MTP speculative decoding) lives under
# ``llama-unsloth``; the plain build stays as a fallback.
_LLAMA_DIRS = [
    os.environ.get("EDGE_LLAMA_DIR", ""),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools", "llama-unsloth"),
    r"D:\AI\tools\llama-unsloth",
    r"D:\AI\tools\llama",
    r"C:\llama.cpp",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools", "llama"),
]

# Engine → extra binary search dirs. The AgentionAI fork (APR rotor kernels,
# required by the TQ1_0 Gyro GGUFs) lives in its own folder so both engines can
# coexist. A per-engine env var overrides the default locations.
_ENGINE_LLAMA_DIRS = {
    "agention": [
        os.environ.get("EDGE_LLAMA_AGENTION_DIR", ""),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools", "llama-agention"),
        r"D:\AI\tools\llama-agention",
    ],
}


def find_llama_server(engine: str = "upstream") -> Optional[str]:
    """Locate ``llama-server.exe`` for a given engine.

    ``engine="upstream"`` (default) → stock/unsloth build.
    ``engine="agention"``          → AgentionAI fork (TQ1_0/APR models).
    Falls back to the upstream search when the engine-specific dirs are empty,
    so a missing fork degrades gracefully instead of hard-failing.
    """
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    dirs = list(_ENGINE_LLAMA_DIRS.get(engine, [])) + _LLAMA_DIRS
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return shutil.which("llama-server") or shutil.which("llama-server.exe")


def engine_available(engine: str) -> bool:
    """True when the engine's own binary (not the fallback) is present."""
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    for d in _ENGINE_LLAMA_DIRS.get(engine, []):
        if d and os.path.isfile(os.path.join(d, name)):
            return True
    return False


def llama_available() -> bool:
    return find_llama_server() is not None


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _read_arch(model_path: str) -> str:
    """Best-effort read of ``general.architecture`` from a GGUF header.

    Only the first metadata block is parsed, so this is cheap even on a 60 GB
    file. Split models: pass the ``-00001-of-0000N`` shard.
    """
    import struct
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return ""
            struct.unpack("<I", f.read(4))           # version
            struct.unpack("<Q", f.read(8))           # n_tensors
            n_kv, = struct.unpack("<Q", f.read(8))
            for _ in range(n_kv):
                n, = struct.unpack("<Q", f.read(8))
                key = f.read(n).decode("utf-8", "replace")
                t, = struct.unpack("<I", f.read(4))
                if t == 8:                            # string
                    ln, = struct.unpack("<Q", f.read(8))
                    val = f.read(ln).decode("utf-8", "replace")
                elif t == 9:                          # array -> skip
                    et, = struct.unpack("<I", f.read(4))
                    cnt, = struct.unpack("<Q", f.read(8))
                    if et == 8:
                        for _ in range(cnt):
                            sl, = struct.unpack("<Q", f.read(8))
                            f.seek(sl, 1)
                    else:
                        f.seek({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                                7: 1, 10: 8, 11: 8, 12: 8}.get(et, 4) * cnt, 1)
                    continue
                else:
                    f.seek({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                            7: 1, 10: 8, 11: 8, 12: 8}.get(t, 4), 1)
                    continue
                if key == "general.architecture":
                    return val
    except Exception:  # noqa: BLE001
        pass
    return ""


# Architectures that need extra flags. Add here rather than in the GUI so the
# behaviour travels with the backend.
_ARCH_FLAGS = {
    # qwen4exp (Qwen3.8-Flash-Next) does not implement upstream's RoPE rotation
    # (#21038); without this the results are wrong or the load fails.
    # ``--tensor-split 1`` is required whenever a second model (the MTP draft)
    # loads after the target: otherwise the free-VRAM-based default split sees
    # 0 free and aborts with "invalid vector subscript".
    # ``ncmoe`` is the measured stable offload point on 8 GB VRAM + 32 GB RAM.
    "qwen4exp": {
        "env": {"LLAMA_ATTN_ROT_DISABLE": "1"},
        "args": ["--tensor-split", "1"],
        "ncmoe": 35,
    },
}


@dataclass
class LlamaConfig:
    model_path: str
    port: int = 8081
    n_gpu_layers: int = 999      # put as much as fits on GPU
    n_cpu_moe: int = 0           # MoE layers kept on CPU (3-tier offload knob)
    ctx: int = 8192
    threads: int = 0             # 0 -> auto
    reasoning: bool = False      # Qwen3 thinking; off by default for chat
    kv_quant: str = ""           # "" | q8_0 | q4_0 ... (applied to K and V)
    kv_offload: bool = True      # put KV cache on GPU; False -> host RAM (large ctx)
    draft_path: str = ""         # MTP draft head (enables speculative decoding)
    spec_n_max: int = 2          # draft tokens per step (2 measured best here)
    flash_attn: bool = True      # -fa; required for KV quantization + faster attn
    # P0-2: bounded GPU expert cache (llama.cpp's answer to Strata's expert
    # cache). 0 = let llama.cpp decide (default). ⚠️ Upstream reports it can
    # split the graph (2 -> ~50 splits/token) and cost ~10% even when everything
    # fits, so we only pass it when the caller opts in.
    moe_cache_mib: int = 0
    # 超级模式：按本机显存(6-16G)/内存(32-64G)自动解出三层分配
    # （专家层数、KV 放显存还是内存、是否开专家缓存、线程与 ubatch）。
    # 只在 llama.cpp 的 N 卡/CPU 路径生效，i 卡 OpenVINO 与 NPU 不受影响。
    auto_tier: bool = False
    # P2: cap how much of RAM the engine may pin for resident experts, so the OS
    # page cache can't evict them mid-generation (Strata's `resident-budget`;
    # measured +20% decode when set). 0 = no explicit cap.
    resident_budget_mib: int = 0
    engine: str = "upstream"     # llama.cpp engine: "upstream" | "agention"
    extra_args: tuple = ()       # escape hatch for arch-specific flags

    def resolved_draft(self) -> str:
        """MTP draft head, auto-disabled for very large contexts.

        P1-A: on an 8 GB card the draft layer costs ~820 MiB–1.8 GiB, and it is
        measured to *zero* decode benefit there while cutting prefill. When the
        context is large enough that KV cache is already tight, drop the draft.
        """
        if not self.draft_path:
            return ""
        if self.ctx >= 65536:
            return ""
        return self.draft_path


class LlamaBackend:
    """Manage a llama-server subprocess speaking the OpenAI API."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._cfg: Optional[LlamaConfig] = None
        self._arch: str = ""

    @property
    def base(self) -> str:
        port = self._cfg.port if self._cfg else 8081
        return f"http://127.0.0.1:{port}"

    @property
    def arch(self) -> str:
        return self._arch

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _plan_tier(self, exe: str, cfg: LlamaConfig):
        """超级模式：调用 tier_controller 解出本机最优三层分配。

        失败时返回 None，调用方回退到手工参数——规划器是增强，不是必需路径。
        """
        try:
            from .tier_controller import plan_super, probe_host, probe_model
            ple = None
            cand = os.path.join(os.path.dirname(cfg.model_path),
                                "Qwen3.8-Flash-Next-PLE-IQ4_NL-GyroS.gguf")
            if os.path.isfile(cand):
                ple = cand
            return plan_super(probe_host(), probe_model(cfg.model_path, ple),
                              cfg.ctx,
                              cache_supported=_supports_flag(exe, "--moe-cache-mib"))
        except Exception as exc:      # 规划失败绝不能让加载失败
            log.warning("超级模式规划失败，回退到手工参数：%s", exc)
            return None

    def load(self, cfg: LlamaConfig) -> None:
        """Start llama-server (idempotent: restarts only if config changed)."""
        if self.running() and self._cfg == cfg:
            return
        self.stop()

        exe = find_llama_server(cfg.engine)
        if not exe:
            raise RuntimeError(
                "未找到 llama-server；请从 llama.cpp Releases 下载对应后端的"
                " Windows 预编译包，解压到 tools/llama/ 或设置 EDGE_LLAMA_DIR"
            )
        if cfg.engine == "agention" and not engine_available("agention"):
            # Stock llama.cpp cannot load TQ1_0/APR GGUFs — running them on the
            # upstream binary would fail with an obscure tensor error, so fail
            # fast with an actionable message instead.
            raise RuntimeError(
                "Agention 引擎未安装：TQ1_0/Gyro 模型需要 AgentionAI 的 llama.cpp "
                "fork（APR 内核），stock llama.cpp 无法加载。请将 fork 的预编译包"
                "解压到 tools/llama-agention/，或设置 EDGE_LLAMA_AGENTION_DIR；"
                "也可退出超级模式改用 upstream 引擎的 UD-IQ1_S 量化。"
            )

        # Architecture-aware flags: qwen4exp needs LLAMA_ATTN_ROT_DISABLE=1 and
        # (because its thinking mode eats the token budget) --no-reasoning-preserve.
        self._arch = _read_arch(cfg.model_path)
        env = os.environ.copy()
        if cfg.engine == "agention":
            # Vulkan + coopmat2 crashes on the TQ/trellis expert matmul kernels:
            # the server dies silently during the first prefill (measured on an
            # RTX 4070 Laptop, driver 32.0.16.2124, our MinGW build). With
            # coopmat2 disabled the same run completes and decodes ~2.0 t/s.
            # See bench/BLUE_SCREEN_INCIDENT.md.
            env.setdefault("GGML_VK_DISABLE_COOPMAT2", "1")
        arch_extra: list = []
        spec = _ARCH_FLAGS.get(self._arch)
        n_cpu_moe = cfg.n_cpu_moe
        if spec:
            env.update(spec.get("env", {}))
            arch_extra.extend(spec.get("args", []))
            # Apply the arch's measured offload default unless the caller set one.
            if not n_cpu_moe and spec.get("ncmoe"):
                n_cpu_moe = int(spec["ncmoe"])
        if cfg.engine == "agention" and not cfg.n_cpu_moe and \
                "gyro" in os.path.basename(cfg.model_path).lower():
            # Gyro-S carries only ~3.9 GB of attention/norm/output, so more
            # expert layers fit in VRAM than the 69 GB upstream model allows.
            # On an 8 GB card ncmoe 44 is the measured boundary: 44 keeps 4
            # expert layers on the GPU (VRAM peaks at 6.99 GB), while ncmoe 40
            # overflows and kills the driver.
            n_cpu_moe = 44

        # 超级模式：按本机显存/内存自动解三层分配。只作用于 llama.cpp 的
        # N 卡 + CPU 路径；OpenVINO 的 i 卡（Arc）与 NPU 有各自实现，不受影响。
        n_gpu_layers = cfg.n_gpu_layers
        threads = cfg.threads
        if not threads:
            # -t 0 让引擎自己决定，在 Intel 大小核上会把 E 核/LPE 核也算进去。
            # 干净内存状态实测（8 GB N 卡、UD-Q2_0、ctx 8192）：
            #   6 个 P 核 2.77 t/s ｜ 自动 22 线程 2.52 ｜ 8 线程 2.33 ｜ 12 线程 2.52
            # 即只用 P 核约 +10%；内存紧张时差距会更大。
            try:
                from .tier_controller import probe_host, _threads_for
                threads, tnotes = _threads_for(probe_host())
                for n in tnotes:
                    log.info("线程数：%s", n)
            except Exception as exc:
                log.debug("线程探测失败，交回引擎自动决定：%s", exc)
                threads = 0
        kv_type = cfg.kv_quant
        kv_offload = cfg.kv_offload
        ubatch = 0
        cache_mib = cfg.moe_cache_mib
        flash_attn = cfg.flash_attn
        if cfg.auto_tier and cfg.engine in ("agention", "upstream"):
            tier = self._plan_tier(exe, cfg)
            if tier is not None:
                n_cpu_moe = tier.n_cpu_moe
                n_gpu_layers = tier.n_gpu_layers
                threads = tier.threads
                ubatch = tier.ubatch
                kv_type = tier.kv_type
                kv_offload = tier.kv_location == "vram"
                cache_mib = tier.moe_cache_mib
                flash_attn = tier.flash_attn
                log.info("超级模式：%s", tier.summary())
                for w in tier.warnings:
                    log.warning("超级模式：%s", w)

        # 容量预检：专家权重 vs 真正能装下的内存。
        # 装不下时引擎不会报错，只会退化成"每请求几十 GB 换页"，速度还会随
        # 内存状态剧烈抖动（同一配置实测 1.40 t/s 脏内存 vs 2.77 t/s 干净内存，
        # 近 2 倍）。提前告警远比
        # 事后调参有用——这是「小显存跑大模型」唯一真正决定性的因素。
        try:
            from .tier_controller import probe_host, probe_model
            _host = probe_host()
            _mprof = probe_model(cfg.model_path)
            _layers = getattr(_mprof, "n_layer", 0) or 1
            _cpu_frac = 1.0 if not n_cpu_moe else min(1.0, n_cpu_moe / _layers)
            _need = _mprof.expert_bytes * _cpu_frac / (1024 ** 3)
            _cap_total = _host.ram_total_gb * 0.80
            _cap_now = max(0.0, _host.ram_avail_gb * 0.95)
            _cap = min(_cap_total, _cap_now) if _cap_now else _cap_total
            if _need > _cap:
                log.warning(
                    "容量预检未通过：常驻专家约 %.1f GB，可用内存约 %.1f GB"
                    "（总 %.1f GB，当前空闲 %.1f GB）。专家装不下会以换页为主，"
                    "decode 可能只有 0.5~1.5 t/s。建议换更小的量化档位"
                    "（IQ1_S / TQ1_0 这类），或先关掉占内存的程序。",
                    _need, _cap, _host.ram_total_gb, _host.ram_avail_gb)
            else:
                log.info("容量预检通过：常驻专家 %.1f GB ≤ 可用 %.1f GB（余量 %.1f GB）",
                         _need, _cap, _cap - _need)
        except Exception as exc:
            log.debug("容量预检跳过：%s", exc)

        args = [
            exe,
            "-m", cfg.model_path,
            "--host", "127.0.0.1",
            "--port", str(cfg.port),
            "-c", str(cfg.ctx),
            "-ngl", str(n_gpu_layers),
        ]
        if n_cpu_moe:
            # llama.cpp 的 3 层卸载关键参数：把 n 个 MoE 层放 CPU/内存。
            args += ["--n-cpu-moe", str(n_cpu_moe)]
        if threads:
            args += ["-t", str(threads)]
        if ubatch:
            args += ["-ub", str(ubatch), "-b", str(max(512, ubatch * 4))]
        # P1-B: flash-attention is required for KV quantization and speeds up
        # attention; enable it for every path that supports it.
        if flash_attn:
            args += ["-fa", "on"]
        if kv_type:
            # KV 量化是大上下文的关键（256K × f16 会吃掉数 GB 显存）。
            args += ["-ctk", kv_type, "-ctv", kv_type]
        if not kv_offload:
            # Large context on a small VRAM card: keep KV in host RAM instead
            # of failing cudaMalloc. Measured: 256K needs ~1.8 GB even at q8_0,
            # which does not fit alongside a 66 GB MoE on an 8 GB card.
            args += ["--no-kv-offload"]
        # P0-2: bounded GPU expert cache (opt-in; see LlamaConfig note).
        if cache_mib > 0:
            if _supports_flag(exe, "--moe-cache-mib"):
                args += ["--moe-cache-mib", str(cache_mib)]
            else:
                log.warning(
                    "llama-server 不支持 --moe-cache-mib（需 llama.cpp >= v0.7.0，"
                    "PR #29887）：专家只能由 CPU 计算，decode 约为 GPU 缓存路径的 "
                    "1/2~1/3。已忽略该参数。")
        # P2: cap pinning so the OS can't evict resident experts mid-run.
        # ⚠️ 没有任何已发布的 llama.cpp（含 d4f7a21 与 unsloth b10715）认识 --set，
        # 传了会直接 "error: invalid argument: --set" 启动失败。先探测再用。
        if cfg.resident_budget_mib > 0:
            if _supports_flag(exe, "--set"):
                args += ["--set", f"resident-budget-mib={cfg.resident_budget_mib}"]
            else:
                log.warning(
                    "当前引擎不支持 --set（resident-budget），已忽略该设置。"
                    "该参数需要特定分支的 llama.cpp，升级引擎后再启用。")

        # MTP speculative decoding: a small draft head proposes tokens that the
        # target verifies in one pass (lossless). Measured ~+17% decode here;
        # the gain is capped by SSD paging, not by the head's precision.
        # resolved_draft() auto-disables it at ctx >= 64K (8 GB cards).
        draft = cfg.resolved_draft()
        if draft and os.path.isfile(draft):
            args += [
                "-md", draft,
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(cfg.spec_n_max),
            ]
            if cfg.engine == "agention":
                # Adaptive speculation (fork headline feature): draft length
                # follows measured acceptance instead of a fixed n — the gyro
                # recipe's own setting. Also keeps the draft warm on device.
                args += ["--spec-draft-adaptive", "--spec-draft-n-min", "2"]

        # Qwen3-style thinking: off by default (much faster, cleaner answers);
        # `--reasoning-format deepseek` keeps any thoughts out of `content`.
        args += ["--reasoning", "on" if cfg.reasoning else "off",
                 "--reasoning-format", "deepseek"]
        if not cfg.reasoning:
            # Keeps the reasoning channel from consuming the whole max_tokens
            # (observed on qwen4exp: a 24-token budget produced an empty answer).
            args += ["--no-reasoning-preserve"]

        # AgentionAI fork: the TQ1_0 Gyro GGUFs need the Jinja chat template.
        # The Gyro arch requires the PLE (per_layer_token_embd) table: either
        # embedded in the main file (full Gyro-S) or provided via --model-ple
        # sidecar (the "-noPLE" files). The table itself stays on disk
        # (--ngram-on-disk); each batch only reads the rows it needs.
        if cfg.engine == "agention":
            args += ["--jinja"]
            model_dir = os.path.dirname(cfg.model_path)
            ple_sidecar = os.path.join(model_dir, "Qwen3.8-Flash-Next-PLE-IQ4_NL-GyroS.gguf")
            if "nople" in cfg.model_path.lower():
                if os.path.isfile(ple_sidecar):
                    args += ["--model-ple", ple_sidecar, "--ngram-on-disk"]
                # no sidecar → cannot load at all; llama.cpp will fail with a
                # clear check_tensor_dims error, surfaced by the caller.
            else:
                args += ["--ngram-on-disk"]

        args += arch_extra
        args += list(cfg.extra_args)

        kwargs = {"env": env}
        if os.name == "nt":
            kwargs["creationflags"] = 0x0800_0000  # CREATE_NO_WINDOW
        # Capture llama-server output: a failed load (e.g. cudaMalloc OOM at
        # 256K context) must be diagnosable instead of silently dying.
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "logs", "llama-server.log")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_f = open(log_path, "w", encoding="utf-8", errors="replace")
            kwargs["stdout"] = log_f
            kwargs["stderr"] = subprocess.STDOUT
        except OSError:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        self._proc = subprocess.Popen(args, **kwargs)
        self._cfg = cfg

        # Wait until /health reports 200 (llama-server answers 503 while the
        # model is still loading, so a bare port check is not enough).
        import json as _json
        import urllib.request

        start = time.time()
        while time.time() - start < 600:
            if self._proc.poll() is not None:
                raise RuntimeError("llama-server 启动失败（进程已退出）")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=3) as r:
                    body = _json.loads(r.read().decode("utf-8", "replace"))
                    if r.status == 200 and body.get("status") in ("ok", None):
                        return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("llama-server 启动超时（模型加载过慢）")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._cfg = None
