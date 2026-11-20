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

import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator, Optional

# Where we look for the prebuilt binaries (downloaded from llama.cpp releases).
_LLAMA_DIRS = [
    os.environ.get("EDGE_LLAMA_DIR", ""),
    r"D:\AI\tools\llama",
    r"C:\llama.cpp",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools", "llama"),
]


def find_llama_server() -> Optional[str]:
    """Locate ``llama-server.exe`` (or ``llama-server`` on POSIX)."""
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    for d in _LLAMA_DIRS:
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return shutil.which("llama-server") or shutil.which("llama-server.exe")


def llama_available() -> bool:
    return find_llama_server() is not None


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


@dataclass
class LlamaConfig:
    model_path: str
    port: int = 8081
    n_gpu_layers: int = 999      # put as much as fits on GPU
    n_cpu_moe: int = 0           # MoE layers kept on CPU (3-tier offload knob)
    ctx: int = 8192
    threads: int = 0             # 0 -> auto


class LlamaBackend:
    """Manage a llama-server subprocess speaking the OpenAI API."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._cfg: Optional[LlamaConfig] = None

    @property
    def base(self) -> str:
        port = self._cfg.port if self._cfg else 8081
        return f"http://127.0.0.1:{port}"

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def load(self, cfg: LlamaConfig) -> None:
        """Start llama-server (idempotent: restarts only if config changed)."""
        if self.running() and self._cfg == cfg:
            return
        self.stop()

        exe = find_llama_server()
        if not exe:
            raise RuntimeError(
                "未找到 llama-server；请从 llama.cpp Releases 下载对应后端的"
                " Windows 预编译包，解压到 tools/llama/ 或设置 EDGE_LLAMA_DIR"
            )

        args = [
            exe,
            "-m", cfg.model_path,
            "--host", "127.0.0.1",
            "--port", str(cfg.port),
            "-c", str(cfg.ctx),
            "-ngl", str(cfg.n_gpu_layers),
        ]
        if cfg.n_cpu_moe:
            # llama.cpp 的 3 层卸载关键参数：把 n 个 MoE 层放 CPU/内存。
            args += ["--n-cpu-moe", str(cfg.n_cpu_moe)]
        if cfg.threads:
            args += ["-t", str(cfg.threads)]

        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x0800_0000  # CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs
        )
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
