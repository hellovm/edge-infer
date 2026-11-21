"""Plan 3-tier offloading for a GGUF model without loading it.

Reads only the GGUF header (metadata + tensor table) so it works on models far
larger than RAM. Reports, per hardware tier, how the weights can be split, and
recommends ``--n-cpu-moe`` / ``--ple`` / ``--lazy-mode`` settings.

Why a separate tool
-------------------
``--n-cpu-moe N`` is a guess until you know each expert's size. For models like
Qwen3.8-Flash-Next (125B backbone + 51B n-gram table, but only 6B active per
token) the right split is unintuitive: most experts can live on SSD/RAM because
a token only touches ~11 of 512.

Usage:
    python -m edge_server.tier_plan <model.gguf> [--vram 8] [--ram 32]

The GGUF reader here is deliberately dependency-free (struct only).
"""

from __future__ import annotations

import argparse
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# minimal GGUF reader (header + tensor table only -- never reads tensor data)
# ---------------------------------------------------------------------------

_GGUF_TYPES = {
    0: ("U8", 1), 1: ("I8", 1), 2: ("U16", 2), 3: ("I16", 2),
    4: ("U32", 4), 5: ("I32", 4), 6: ("F32", 4), 7: ("BOOL", 1),
    8: ("STRING", 0), 9: ("ARRAY", 0), 10: ("U64", 8), 11: ("I64", 8), 12: ("F64", 8),
}

# ggml block sizes: (block_size, type_size) for the quantized types we may meet.
# NOTE: llama.cpp keeps adding types; ids >= 41 are recent additions (Q1_0/Q2_0
# are 1/2-bit "simple block" quants, not K-quants). Unknown ids fall back to
# 1 element x 4 bytes, which over-estimates -- so keep this table current.
_GGML_BLOCK = {
    0: (1, 4), 1: (1, 2),      # F32, F16
    2: (32, 18),               # Q4_0
    3: (32, 20),               # Q4_1
    6: (32, 22),               # Q5_0
    7: (32, 24),               # Q5_1
    8: (32, 34),               # Q8_0
    9: (32, 36),               # Q8_1
    10: (256, 84),             # Q2_K
    11: (256, 110),            # Q3_K
    12: (256, 144),            # Q4_K
    13: (256, 176),            # Q5_K
    14: (256, 210),            # Q6_K
    15: (256, 292),            # Q8_K
    16: (256, 66),             # IQ2_XXS
    17: (256, 74),             # IQ2_XS
    18: (256, 98),             # IQ3_XXS
    19: (256, 34),             # IQ1_S
    20: (32, 18),              # IQ4_NL  (block 32, not 256)
    21: (256, 110),            # IQ3_S
    22: (256, 82),             # IQ2_S
    23: (256, 136),            # IQ4_XS
    24: (1, 1),                # I8
    25: (1, 2),                # I16
    26: (1, 4),                # I32
    27: (1, 8),                # I64
    28: (1, 8),                # F64
    29: (256, 56),             # IQ1_M
    30: (1, 2),                # BF16
    34: (256, 54),             # TQ1_0
    35: (256, 66),             # TQ2_0
    36: (256, 54),             # TQ1_0 (alias seen in some builds)
    37: (256, 66),             # TQ2_0 (alias)
    41: (128, 18),             # Q1_0
    42: (64, 18),              # Q2_0
}

_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 41: "Q1_0", 42: "Q2_0",
}


class _Reader:
    def __init__(self, f):
        self.f = f

    def u32(self) -> int:
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.f.read(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.f.read(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.f.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.f.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.f.read(8))[0]

    def boolean(self) -> bool:
        return struct.unpack("<?", self.f.read(1))[0]

    def string(self) -> str:
        n = self.u64()
        if n > 1 << 24:           # sanity: refuse absurd lengths (corrupt file)
            raise ValueError(f"string length {n} too large")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t: int) -> Any:
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            if et == 9:            # array of arrays -- skip defensively
                for _ in range(n):
                    self.value(et)
                return f"<nested array x{n}>"
            # Strings dominate the metadata size; we only need counts, so skip
            # the payload for big string arrays instead of materialising them.
            if et == 8:
                for _ in range(n):
                    ln = self.u64()
                    self.f.seek(ln, 1)
                return f"<string array x{n}>"
            step = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}[et]
            self.f.seek(step * n, 1)
            return f"<array x{n}>"
        return {
            0: lambda: self.f.read(1)[0], 1: lambda: struct.unpack("<b", self.f.read(1))[0],
            2: lambda: struct.unpack("<H", self.f.read(2))[0],
            3: lambda: struct.unpack("<h", self.f.read(2))[0],
            4: self.u32, 5: self.i32, 6: self.f32, 7: self.boolean,
            10: self.u64, 11: self.i64, 12: self.f64,
        }[t]()


def read_gguf(path: str) -> Dict[str, Any]:
    """Parse header + metadata + tensor table. Returns the tensor list too.

    Works on a partially-downloaded file: the header and tensor table live at
    the start, so even a ``.incomplete`` shard can be inspected mid-download.
    """
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file (magic={magic!r})")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        meta: Dict[str, Any] = {}
        for _ in range(n_kv):
            k = r.string()
            t = r.u32()
            try:
                meta[k] = r.value(t)
            except Exception as e:  # noqa: BLE001
                meta[k] = f"<parse error: {e}>"
                break

        tensors: List[Dict[str, Any]] = []
        for _ in range(n_tensors):
            try:
                name = r.string()
                n_dims = r.u32()
                dims = [r.u64() for _ in range(n_dims)]
                ttype = r.u32()
                offset = r.u64()
            except Exception:  # noqa: BLE001
                break          # truncated download -- keep what we parsed
            n_elem = 1
            for d in dims:
                n_elem *= d
            bs, ts = _GGML_BLOCK.get(ttype, (1, 4))
            nbytes = (n_elem // bs) * ts if bs > 1 else n_elem * ts
            tensors.append({
                "name": name, "dims": dims, "type": ttype,
                "type_name": _TYPE_NAME.get(ttype, f"T{ttype}"),
                "bytes": nbytes,
            })
        return {"version": version, "meta": meta, "tensors": tensors,
                "n_tensors_declared": n_tensors}


_SPLIT_RE = __import__("re").compile(r"^(?P<pre>.+?)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf")


def find_shards(path: str) -> List[str]:
    """Return every shard of a split GGUF, including still-downloading ones.

    ``…-00002-of-00003.gguf.incomplete`` counts, so a plan can be produced
    while the download is still running.
    """
    d = os.path.dirname(path) or "."
    b = os.path.basename(path)
    m = _SPLIT_RE.match(b)
    if not m:
        return [path]
    total = int(m.group("n"))
    pre = m.group("pre")
    out: List[str] = []
    for i in range(1, total + 1):
        for cand in (f"{pre}-{i:05d}-of-{total:05d}.gguf",
                     f"{pre}-{i:05d}-of-{total:05d}.gguf.incomplete"):
            p = os.path.join(d, cand)
            if os.path.isfile(p):
                out.append(p)
                break
    return out or [path]


def read_model(path: str) -> Dict[str, Any]:
    """Read a (possibly split, possibly incomplete) model as one unit."""
    shards = find_shards(path)
    metas: Dict[str, Any] = {}
    tensors: List[Dict[str, Any]] = []
    version = 0
    declared = 0
    partial = []
    for s in shards:
        try:
            info = read_gguf(s)
        except Exception as e:  # noqa: BLE001
            partial.append(f"{os.path.basename(s)}: {e}")
            continue
        version = info["version"]
        declared += info["n_tensors_declared"]
        for k, v in info["meta"].items():
            metas.setdefault(k, v)      # shard 1 carries the real metadata
        tensors.extend(info["tensors"])
        if s.endswith(".incomplete"):
            partial.append(os.path.basename(s))
    return {
        "version": version, "meta": metas, "tensors": tensors,
        "n_shards": len(shards), "n_tensors_declared": declared,
        "partial": partial,
    }


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def _is_expert(name: str) -> bool:
    n = name.lower()
    return ("ffn_gate_exps" in n or "ffn_up_exps" in n or "ffn_down_exps" in n
            or "_exps." in n or ".exps" in n)


def _is_ngram(name: str) -> bool:
    n = name.lower()
    return ("per_layer_token_embd" in n or "ngram" in n or "engram" in n
            or "ple" in n.split(".")[-1])


def _is_attn(name: str) -> bool:
    n = name.lower()
    return ("attn_" in n or "attention" in n)


def classify(tensors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Split tensors into the groups that matter for tiering."""
    groups: Dict[str, Dict[str, Any]] = {
        "ngram_table": {"bytes": 0, "count": 0, "note": "查表用，可放 RAM/SSD"},
        "experts": {"bytes": 0, "count": 0, "note": "MoE 专家，可按热度分层"},
        "attention": {"bytes": 0, "count": 0, "note": "注意力，常驻快层"},
        "other": {"bytes": 0, "count": 0, "note": "embedding / norm / 输出层"},
    }
    for t in tensors:
        n = t["name"]
        if _is_ngram(n):
            k = "ngram_table"
        elif _is_expert(n):
            k = "experts"
        elif _is_attn(n):
            k = "attention"
        else:
            k = "other"
        groups[k]["bytes"] += t["bytes"]
        groups[k]["count"] += 1
    return groups


def decide_tiers(groups: Dict[str, Dict[str, Any]], vram: float,
                 ram: float) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Decide which group goes where.

    Priority reasoning:

    * ``attention`` / ``other`` are small and touched **every token** -- they
      must be resident (VRAM, else RAM).
    * ``experts`` are huge but only ~2% are read per token, so RAM is fine.
    * ``ngram_table`` is a **pure lookup** (16 rows/token, a few KB): it is the
      single best thing to park on SSD, and the only group whose cost on SSD is
      a page read rather than a matmul.

    So when the model does not fit in VRAM+RAM, the n-gram table is evicted to
    SSD *first*, before any expert.
    """
    tiers: Dict[str, str] = {}
    v_left, r_left = vram, ram
    detail: Dict[str, Dict[str, float]] = {}

    def place(k: str) -> None:
        """Fill VRAM first, then RAM, spill the remainder to SSD."""
        nonlocal v_left, r_left
        b = groups[k]["bytes"]
        into_v = min(b, v_left)
        rest = b - into_v
        into_r = min(rest, r_left)
        to_ssd = rest - into_r
        v_left -= into_v
        r_left -= into_r
        detail[k] = {"vram": into_v, "ram": into_r, "ssd": to_ssd}
        if to_ssd > 0:
            tiers[k] = "SSD" if (into_v + into_r) == 0 else "RAM+SSD"
        elif into_r > 0:
            tiers[k] = "RAM"
        else:
            tiers[k] = "VRAM"

    # 1) always-resident groups (small, touched every token)
    for k in ("attention", "other"):
        if groups[k]["bytes"]:
            place(k)

    # 2) the big two. The n-gram table is a pure lookup, so it is the cheapest
    #    thing to park on SSD -- evict it there before spilling any expert.
    ng_b = groups["ngram_table"]["bytes"]
    ex_b = groups["experts"]["bytes"]
    if (ng_b + ex_b) <= (v_left + r_left):
        # everything fits in VRAM+RAM, no SSD needed
        place("experts")
        place("ngram_table")
    else:
        # Over budget: the n-gram table goes **entirely** to SSD. It is a pure
        # lookup, so residency buys almost nothing, whereas the experts are
        # matmul'd and benefit from fast memory. Giving the leftover VRAM to
        # experts is strictly better than splitting it with the table.
        detail["ngram_table"] = {"vram": 0.0, "ram": 0.0, "ssd": ng_b}
        tiers["ngram_table"] = "SSD"
        place("experts")

    total = sum(g["bytes"] for g in groups.values())
    info = {
        "total_bytes": total,
        "fits_vram_ram": sum(d["ssd"] for d in detail.values()) == 0,
        "vram_used": sum(d["vram"] for d in detail.values()),
        "ram_used": sum(d["ram"] for d in detail.values()),
        "ssd_used": sum(d["ssd"] for d in detail.values()),
        "tier_detail": detail,
    }
    return tiers, info


def plan(path: str, vram_gb: float, ram_gb: float) -> Dict[str, Any]:
    """Analyse a GGUF (split / partial ok) and recommend a tier split."""
    info = read_model(path)
    meta, tensors = info["meta"], info["tensors"]
    groups = classify(tensors)

    vram = vram_gb * 1e9
    ram = ram_gb * 1e9
    tiers, summary = decide_tiers(groups, vram, ram)

    return {
        "path": path,
        "arch": meta.get("general.architecture", "?"),
        "name": meta.get("general.name", ""),
        "version": info["version"],
        "n_tensors": len(tensors),
        "n_tensors_declared": info.get("n_tensors_declared", len(tensors)),
        "n_shards": info.get("n_shards", 1),
        "partial": info.get("partial", []),
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "groups": groups,
        "tiers": tiers,
        **summary,
        "meta": {k: v for k, v in meta.items()
                 if any(s in k for s in ("arch", "block_count", "context_length",
                                         "expert", "embedding_length", "name"))},
    }


def render(p: Dict[str, Any]) -> str:
    g = p["groups"]
    L = []
    L.append("=" * 74)
    L.append(f"三层卸载规划  ·  {os.path.basename(p['path'])}")
    L.append("=" * 74)
    L.append(f"架构        : {p['arch']}  (GGUF v{p['version']}, "
             f"{p['n_tensors']} tensors"
             + (f" / {p['n_shards']} shards" if p.get("n_shards", 1) > 1 else "")
             + ")")
    if p.get("partial"):
        L.append(f"⚠ 下载未完成 : {len(p['partial'])} 个分片仍在下载中")
    for k, v in (p["meta"] or {}).items():
        if k.startswith("general."):
            continue
        L.append(f"  {k:34} = {v}")
    L.append("")
    L.append(f"权重总计    : {p['total_bytes']/1e9:.2f} GB")
    L.append("")
    L.append("--- 分组 ---")
    for k in ("ngram_table", "experts", "attention", "other"):
        v = g[k]
        if not v["count"]:
            continue
        L.append(f"  {k:14} {v['bytes']/1e9:8.2f} GB   {v['count']:5} tensors   {v['note']}")
    L.append("")
    L.append(f"--- 分层建议 (VRAM {p['vram_gb']:.0f} GB + RAM {p['ram_gb']:.0f} GB) ---")
    det = p.get("tier_detail") or {}
    for k in ("attention", "other", "experts", "ngram_table"):
        if k not in p["tiers"]:
            continue
        d = det.get(k)
        extra = ""
        if d:
            parts = []
            for t, short in (("vram", "VRAM"), ("ram", "RAM"), ("ssd", "SSD")):
                if d.get(t):
                    parts.append(f"{short} {d[t]/1e9:.2f} GB")
            extra = "   [" + " + ".join(parts) + "]"
        L.append(f"  {k:14} -> {p['tiers'][k]:8}{extra}")
    L.append("")
    L.append(f"合计: VRAM {p.get('vram_used',0)/1e9:.2f} / "
             f"RAM {p.get('ram_used',0)/1e9:.2f} / "
             f"SSD {p.get('ssd_used',0)/1e9:.2f} GB")
    L.append("")
    if p["fits_vram_ram"]:
        L.append("  ✓ 权重可完全放进 VRAM + RAM")
        L.append("    建议: -ngl 999  -ncmoe <放入RAM的专家层数>")
    else:
        L.append("  ⚠ 权重超出 VRAM+RAM —— 需要 SSD 参与")
        L.append("    建议: mmap 默认开启 + --n-cpu-moe 增大 + --lazy-mode on")
        L.append("          n-gram 表留在 SSD 上按需读取（查表非矩阵乘，每 token 只读 16 行）")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="GGUF 三层卸载规划")
    ap.add_argument("model")
    ap.add_argument("--vram", type=float, default=8.0)
    ap.add_argument("--ram", type=float, default=32.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    p = plan(a.model, a.vram, a.ram)
    if a.json:
        import json
        print(json.dumps(p, ensure_ascii=False, indent=2))
    else:
        print(render(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
