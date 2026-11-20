"""Unified logging setup for edge-infer.

Goals
-----
1. A single rotating log file (``logs/edge-server.log``) that survives across
   runs and is included in diagnostic bundles.
2. Unhandled exceptions in any thread / request are captured, not swallowed.
3. No dependency on a log viewer -- plain, greppable lines.

Call ``setup_logging()`` once, early, from ``run_server.py`` and from the
Tauri sidecar launcher.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "edge-server.log")

_configured = False
_lock = threading.Lock()

FMT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: Optional[str] = None, *, console: bool = True) -> str:
    """Configure root logging once. Returns the log file path."""
    global _configured
    with _lock:
        if _configured:
            return LOG_FILE
        _configured = True

    lvl = (level or os.environ.get("EDGE_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, lvl, logging.INFO))

    for h in list(root.handlers):
        root.removeHandler(h)

    os.makedirs(LOG_DIR, exist_ok=True)

    # Rotating file: 5 MB x 3 -> bounded, always safe to ship in a bundle.
    try:
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(FMT, DATEFMT))
        root.addHandler(fh)
    except Exception:  # noqa: BLE001
        pass  # read-only FS: keep going with console only

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(FMT, DATEFMT))
        root.addHandler(sh)

    # Third-party noise down.
    for noisy in ("uvicorn.access", "httpx", "urllib3", "modelscope"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Capture otherwise-silent failures.
    _install_excepthooks()

    logging.getLogger("edge").info(
        "logging initialised -> %s (level=%s)", LOG_FILE, lvl
    )
    return LOG_FILE


def _install_excepthooks() -> None:
    """Route uncaught exceptions (main + threads) into the log."""
    log = logging.getLogger("edge.unhandled")

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return
        log.error("uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook

    def _thread_hook(args):  # noqa: ANN001
        if issubclass(args.exc_type, SystemExit):
            return
        log.error("uncaught exception in thread '%s'",
                  args.thread.name if args.thread else "?",
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook


def log_dir() -> str:
    return LOG_DIR
