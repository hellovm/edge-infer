"""Local agent tools — give the model real hands on the filesystem.

This is the "WorkBuddy-like local agent" capability: the model can create task
folders, read/write files, list directories, search, and run whitelisted shell
commands. Everything is confined to a sandboxed workspace root so the agent
cannot touch the rest of the disk.

Design:
  - A tool is a Python callable + a JSON schema description (OpenAI tool format).
  - `run_tool(name, args)` executes it inside the workspace and returns text.
  - The chat loop asks the model for a tool call, runs it, feeds the result
    back, and repeats until the model answers (or the step budget is spent).

The model is a plain instruct LLM (no native tool-calling head), so we drive
tool use with a strict text protocol: the model emits a single JSON object
inside <tool_call>...</tool_call>. This is robust across Qwen/DeepSeek/Phi.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---- workspace sandbox -----------------------------------------------------
# All agent file ops are confined under this root.
_DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "workspace",
)


class Workspace:
    """A sandboxed directory the agent can freely read/write."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root or os.environ.get("EDGE_WORKSPACE", _DEFAULT_ROOT)).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = self.root  # the "cwd" for relative paths

    # -- path safety ---------------------------------------------------------
    def resolve(self, rel: str) -> Path:
        """Resolve `rel` against the current dir, refusing to escape root."""
        p = (self.current / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    # -- tools ---------------------------------------------------------------
    def list_dir(self, path: str = ".") -> str:
        p = self.resolve(path)
        if not p.exists():
            return f"(不存在) {path}"
        if p.is_file():
            return f"{path} 是一个文件 ({p.stat().st_size} 字节)"
        items = []
        for c in sorted(p.iterdir()):
            if c.name.startswith("."):
                continue
            if c.is_dir():
                items.append(f"[目录] {c.name}/")
            else:
                items.append(f"[文件] {c.name}  ({c.stat().st_size} 字节)")
        return "\n".join(items) or "(空目录)"

    def read_file(self, path: str, max_bytes: int = 200_000) -> str:
        p = self.resolve(path)
        if not p.is_file():
            return f"(不存在) {path}"
        data = p.read_bytes()[:max_bytes]
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return f"(二进制文件，{p.stat().st_size} 字节，无法以文本显示)"

    def write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {p.relative_to(self.root)} ({len(content)} 字符)"

    def make_dir(self, path: str) -> str:
        p = self.resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"已创建目录 {p.relative_to(self.root)}/"

    def create_task_folder(self, name: str, subdirs: Optional[List[str]] = None) -> str:
        """Create a task folder (like WorkBuddy's per-task workspace)."""
        base = self.resolve(name)
        subs = subdirs or ["input", "output", "notes"]
        for s in subs:
            (base / s).mkdir(parents=True, exist_ok=True)
        # seed a README so the folder is self-describing
        readme = base / "README.md"
        if not readme.exists():
            readme.write_text(f"# {name}\n\n由 edge-infer 本地 Agent 创建于 "
                              f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                              f"目录结构:\n" + "\n".join(f"- {s}/" for s in subs) + "\n",
                              encoding="utf-8")
        return f"已创建任务文件夹 {name}/（含 {'、'.join(subs)}）"

    def search(self, pattern: str, path: str = ".") -> str:
        p = self.resolve(path)
        hits = []
        rx = re.compile(pattern)
        for root, _dirs, files in os.walk(p):
            for f in files:
                fp = Path(root) / f
                try:
                    for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{fp.relative_to(self.root)}:{i}: {line[:120]}")
                            if len(hits) >= 50:
                                return "\n".join(hits) + "\n…（截断）"
                except Exception:  # noqa: BLE001
                    continue
        return "\n".join(hits) or "(无匹配)"

    def run_command(self, cmd: str, timeout: int = 20) -> str:
        """Run a whitelisted, read-mostly shell command inside the workspace."""
        allowed_prefixes = ("dir", "ls", "type ", "cat ", "echo ", "find", "grep",
                            "python ", "python3 ", "pip ", "git ", "where ", "which ")
        if not cmd.strip().lower().startswith(allowed_prefixes):
            return f"(命令被拒绝：仅允许只读/开发类命令) 收到: {cmd}"
        try:
            out = subprocess.run(cmd, shell=True, cwd=str(self.current),
                                 capture_output=True, text=True, timeout=timeout)
            text = (out.stdout or "") + (out.stderr or "")
            return text[:8000] or "(无输出)"
        except subprocess.TimeoutExpired:
            return f"(命令超时 >{timeout}s)"
        except Exception as e:  # noqa: BLE001
            return f"(执行失败) {e}"

    def cd(self, path: str) -> str:
        p = self.resolve(path)
        if not p.is_dir():
            return f"(不存在) {path}"
        self.current = p
        return f"当前目录 -> {p.relative_to(self.root) or '.'}"


# ---- tool registry (OpenAI-style schemas) ----------------------------------
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"name": "list_dir", "desc": "列出目录内容",
     "params": {"path": "目录路径(默认 .)"}},
    {"name": "read_file", "desc": "读取文本文件内容",
     "params": {"path": "文件路径"}},
    {"name": "write_file", "desc": "写入/覆盖文本文件",
     "params": {"path": "文件路径", "content": "文件内容"}},
    {"name": "make_dir", "desc": "创建目录",
     "params": {"path": "目录路径"}},
    {"name": "create_task_folder",
     "desc": "为一项任务创建标准工作文件夹(含 input/output/notes)",
     "params": {"name": "任务文件夹名", "subdirs": "子目录列表(可选)"}},
    {"name": "search", "desc": "在文件中按正则搜索文本",
     "params": {"pattern": "正则", "path": "搜索目录(默认 .)"}},
    {"name": "run_command", "desc": "在任务目录内运行只读/开发类命令",
     "params": {"cmd": "命令"}},
    {"name": "cd", "desc": "切换当前目录",
     "params": {"path": "目录"}},
    {"name": "get_workspace", "desc": "查看工作区根目录与当前目录", "params": {}},
]


def tools_prompt() -> str:
    """The system-prompt block that teaches the model the tool protocol."""
    lines = [
        "你可以使用以下本地工具操作文件系统（工作区沙箱内）：",
    ]
    for t in TOOL_SCHEMAS:
        args = ", ".join(f"{k}: {v}" for k, v in t["params"].items()) or "无"
        lines.append(f"- {t['name']}({args}) — {t['desc']}")
    lines += [
        "",
        "需要调用工具时，只输出一个 JSON 对象，放在 <tool_call></tool_call> 里，例如：",
        '<tool_call>{"name": "create_task_folder", "arguments": {"name": "周报"}}</tool_call>',
        "工具执行结果会以 <tool_result>...</tool_result> 返回给你；",
        "你可以连续调用多个工具，最后再用自然语言给出最终答复（不要再包含 <tool_call>）。",
    ]
    return "\n".join(lines)


def run_tool(ws: Workspace, name: str, args: Dict[str, Any]) -> str:
    """Dispatch a tool call inside the workspace."""
    try:
        if name == "list_dir":
            return ws.list_dir(args.get("path", "."))
        if name == "read_file":
            return ws.read_file(args["path"])
        if name == "write_file":
            return ws.write_file(args["path"], args.get("content", ""))
        if name == "make_dir":
            return ws.make_dir(args["path"])
        if name == "create_task_folder":
            return ws.create_task_folder(args["name"], args.get("subdirs"))
        if name == "search":
            return ws.search(args["pattern"], args.get("path", "."))
        if name == "run_command":
            return ws.run_command(args["cmd"])
        if name == "cd":
            return ws.cd(args["path"])
        if name == "get_workspace":
            return f"工作区根目录: {ws.root}\n当前目录: {ws.current.relative_to(ws.root) or '.'}"
        return f"(未知工具) {name}"
    except PermissionError as e:
        return f"(权限拒绝) {e}"
    except Exception as e:  # noqa: BLE001
        return f"(工具执行出错) {e}"


_TOOL_CALL_RX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first <tool_call>{...}</tool_call> JSON object, if any."""
    m = _TOOL_CALL_RX.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        if "name" in obj:
            obj.setdefault("arguments", {})
            return obj
    except json.JSONDecodeError:
        # try a lenient fix for trailing commas / single quotes
        raw = m.group(1).replace("'", '"')
        raw = re.sub(r",\s*}", "}", raw)
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return None
    return None
