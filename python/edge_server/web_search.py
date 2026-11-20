"""Web search via browser automation (Playwright + the local Edge/Chrome).

Why browser automation instead of a plain HTTP request?
------------------------------------------------------
Search engines increasingly gate plain HTTP clients (JS challenges, consent
walls, bot fingerprints). Driving a *real* browser sidesteps that, and it also
lets us read JS-rendered result pages.

We use the system-installed Edge/Chrome (``channel="msedge"`` / ``"chrome"``) so
there's no ~150 MB browser download. Everything runs headless.

Public API
----------
``search_web(query, max_results=5) -> list[dict]`` — each item is
``{"title", "url", "snippet"}``. Returns ``[]`` on failure (caller degrades).

``fetch_page_text(url, max_chars=4000) -> str`` — open one page and return its
visible text, for "read this link" follow-ups.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

_lock = threading.Lock()
_browser = None
_playwright = None

try:
    from playwright.sync_api import sync_playwright
    _PW_AVAILABLE = True
except Exception:  # noqa: BLE001
    sync_playwright = None
    _PW_AVAILABLE = False


def available() -> bool:
    return _PW_AVAILABLE


def _get_browser():
    """Lazily launch a headless browser (reused across calls)."""
    global _browser, _playwright
    if _browser is not None:
        return _browser
    if not _PW_AVAILABLE:
        raise RuntimeError("playwright 未安装")
    _playwright = sync_playwright().start()
    last_err = None
    for ch in ("msedge", "chrome", None):
        try:
            kw = {"headless": True, "args": ["--no-sandbox", "--disable-gpu"]}
            if ch:
                kw["channel"] = ch
            _browser = _playwright.chromium.launch(**kw)
            return _browser
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"无法启动浏览器（请安装 Edge/Chrome 或 playwright install）：{last_err}")


def _ctx(browser):
    # NOTE: do NOT override user_agent — Bing serves a different (JS-only)
    # layout for custom UAs that lacks the `li.b_algo` result nodes. The
    # browser's default UA gets the plain, parseable results page.
    return browser.new_context(locale="zh-CN", viewport={"width": 1280, "height": 900})


def search_web(query: str, max_results: int = 5) -> List[dict]:
    """Search the web and return [{title, url, snippet}]."""
    with _lock:
        try:
            browser = _get_browser()
        except Exception as e:  # noqa: BLE001
            print(f"[edge-infer] search unavailable: {e}")
            return []
        try:
            page = _ctx(browser).new_page()
            return _search_bing(page, query, max_results)
        except Exception as e:  # noqa: BLE001
            print(f"[edge-infer] search failed: {e}")
            return []
        finally:
            try:
                page.close()
            except Exception:  # noqa: BLE001
                pass


def _search_bing(page, query: str, max_results: int) -> List[dict]:
    from urllib.parse import quote_plus

    page.goto(f"https://www.bing.com/search?q={quote_plus(query)}&setlang=zh-CN",
              wait_until="domcontentloaded", timeout=25000)

    # Wait for the results container. Bing may redirect once (consent/region),
    # which destroys the execution context — so retry the wait a couple of times.
    found = False
    for _ in range(3):
        try:
            page.wait_for_selector("li.b_algo", timeout=8000)
            found = True
            break
        except Exception:  # noqa: BLE001
            page.wait_for_timeout(1000)
    if not found:
        return []

    items = []
    for el in page.query_selector_all("li.b_algo"):
        if len(items) >= max_results:
            break
        try:
            a = el.query_selector("h2 a")
            if not a:
                continue
            title = (a.inner_text() or "").strip()
            url = a.get_attribute("href") or ""
            sn = el.query_selector(".b_caption p") or el.query_selector("p")
            snippet = (sn.inner_text() or "").strip() if sn else ""
            if title and url.startswith("http"):
                items.append({"title": title, "url": url, "snippet": snippet[:400]})
        except Exception:  # noqa: BLE001
            continue
    return items


def fetch_page_text(url: str, max_chars: int = 4000) -> str:
    """Open `url` in the browser and return its visible text."""
    with _lock:
        try:
            browser = _get_browser()
        except Exception:  # noqa: BLE001
            return ""
        try:
            page = _ctx(browser).new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(800)
            text = page.inner_text("body") or ""
            return " ".join(text.split())[:max_chars]
        except Exception as e:  # noqa: BLE001
            print(f"[edge-infer] fetch failed: {e}")
            return ""
        finally:
            try:
                page.close()
            except Exception:  # noqa: BLE001
                pass


def format_results(results: List[dict]) -> str:
    """Render search results as a compact prompt block."""
    if not results:
        return ""
    lines = ["【联网搜索结果】"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r.get('snippet','')}")
    return "\n".join(lines)


if __name__ == "__main__":
    for r in search_web("OpenVINO NPU 支持模型", 3):
        print("-", r["title"], r["url"])
