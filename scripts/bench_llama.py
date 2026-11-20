"""Benchmark GGUF models with llama-bench (CUDA / CPU / 3-tier offload).

Reports prefill (pp) and decode (tg) separately, matching the README table.

Usage:
    python scripts/bench_llama.py <model.gguf> [--ngl 999] [--ncmoe 0]
                                  [--pp 512] [--tg 128] [--label "NVIDIA GPU"]

Environment:
    EDGE_LLAMA_DIR  directory holding llama-bench.exe (default D:\\AI\\tools\\llama)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

_DIRS = [
    os.environ.get("EDGE_LLAMA_DIR", ""),
    r"D:\AI\tools\llama",
    r"C:\llama.cpp",
]


def find_bench() -> str:
    name = "llama-bench.exe" if os.name == "nt" else "llama-bench"
    for d in _DIRS:
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    raise SystemExit("llama-bench not found; set EDGE_LLAMA_DIR")


def run(model: str, ngl: int, ncmoe: int, pp: int, tg: int, reps: int,
        extra: list[str] | None = None) -> dict:
    exe = find_bench()
    args = [exe, "-m", model, "-ngl", str(ngl), "-p", str(pp), "-n", str(tg),
            "-r", str(reps), "-o", "json", "--no-warmup"]
    if ncmoe:
        args += ["-ncmoe", str(ncmoe)]
    if extra:
        args += extra

    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=7200)
    out = p.stdout.strip()
    # llama-bench prints backend-load noise before the JSON array; slice it out.
    i = out.find("[")
    if i < 0:
        return {"error": (p.stderr or out)[-400:]}
    try:
        rows = json.loads(out[i:])
    except Exception:
        # some builds emit JSON lines instead
        rows = []
        for line in out[i:].splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if not rows:
            return {"error": out[-400:]}

    def pick(n_type: str):
        for r in rows:
            if str(r.get("n_prompt")) == str(pp) and str(r.get("n_gen")) == str(tg) \
                    and r.get("n_prompt", 0) > 0 and n_type == "pp":
                return r.get("avg_ts")
        return None

    pp_row = next((r for r in rows if r.get("n_prompt") and r.get("n_gen") == 0), None)
    tg_row = next((r for r in rows if r.get("n_gen") and (r.get("n_prompt") or 0) == 0), None)
    if pp_row is None:
        pp_row = next((r for r in rows if r.get("n_prompt")), None)
    if tg_row is None:
        tg_row = next((r for r in rows if r.get("n_gen")), None)

    return {
        "prefill_tok_s": round(pp_row["avg_ts"], 1) if pp_row and pp_row.get("avg_ts") else None,
        "decode_tok_s": round(tg_row["avg_ts"], 1) if tg_row and tg_row.get("avg_ts") else None,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ngl", type=int, default=999)
    ap.add_argument("--ncmoe", type=int, default=0)
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()

    extra = [x for x in a.extra.split() if x] or None
    try:
        r = run(a.model, a.ngl, a.ncmoe, a.pp, a.tg, a.reps, extra)
    except Exception as e:  # noqa: BLE001
        r = {"error": f"{type(e).__name__}: {e}"}

    lab = a.label or (f"ngl={a.ngl} ncmoe={a.ncmoe}")
    print(json.dumps({"label": lab, **r}, ensure_ascii=False))
    return 0 if "error" not in r else 1


if __name__ == "__main__":
    raise SystemExit(main())
