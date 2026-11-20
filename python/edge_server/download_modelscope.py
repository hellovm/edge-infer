"""Download the small-model (Qwen3-VL 4B INT4 class) from ModelScope.

Per requirements: model source is ModelScope only. This fetches the repo into
models/ so the OpenVINO runtime can load it. OpenVINO needs an IR (.xml/.bin)
or ONNX export; many ModelScope Qwen3-VL repos ship PyTorch weights, so a
second step converts them to ONNX/IR (wired in the next slice).

Usage:
    python -m edge_server.download_modelscope            # default repo
    python -m edge_server.download_modelscope Qwen3-VL-4B-INT4
"""

from __future__ import annotations

import os
import sys

# Real, verified-on-ModelScope OpenVINO INT4 repos for Qwen3-VL 4B (researched
# 2026-11-19). The official OpenVINO namespace has the smallest (~3.1GB) INT4
# export and is the preferred default.
DEFAULT_REPO = "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov"
DEFAULT_SHORT_ID = "qwen3-vl-4b-int4"
KNOWN_REPOS = {
    # preferred: official OpenVINO, ~3.1GB INT4
    "qwen3-vl-4b-int4": "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov",
    "Qwen3-VL-4B-Instruct-int4-ov": "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov",
    "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov": "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov",
    # alt: 魔搭 workshop copy (~3.3GB)
    "snake7gun/Qwen3-VL-4B-Instruct-int4-ov": "snake7gun/Qwen3-VL-4B-Instruct-int4-ov",
}


def _target_dir(short_id: str) -> str:
    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
    )
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, short_id)


def download(repo: str = DEFAULT_REPO, short_id: str = "qwen3-vl-4b-int4") -> str:
    """Download `repo` from ModelScope into models/<short_id>/. Returns the path."""
    from modelscope import snapshot_download

    local_dir = _target_dir(short_id)
    print(f"[modelscope] downloading {repo} -> {local_dir}")
    snapshot_download(
        repo,
        local_dir=local_dir,
    )
    print(f"[modelscope] done -> {local_dir}")
    return local_dir


def main(argv: list[str]) -> int:
    repo = argv[0] if argv else DEFAULT_REPO
    if repo in KNOWN_REPOS:
        repo = KNOWN_REPOS[repo]
    try:
        download(repo)
    except Exception as e:  # noqa: BLE001
        print(f"[modelscope] download failed: {e}", file=sys.stderr)
        print(
            "[modelscope] hint: if the repo has no INT4 / OpenVINO export, "
            "convert PyTorch weights -> ONNX next (separate step).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
