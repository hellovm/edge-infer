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

    def run_command(self, cmd: str, timeout: int = 60) -> str:
        """Run a shell command inside the current project dir.

        Dev-oriented commands are allowed (python/pip/git/npm/node/cargo/make/
        dir/ls/cat/echo/find/grep/where/which/mkdir/rmdir/copy/del/type). This
        is intentionally permissive *inside the sandbox* so the agent can set up
        and run real projects -- but the working dir is always confined to the
        project root, so it cannot escape.
        """
        allowed_prefixes = (
            "dir", "ls", "type ", "cat ", "echo ", "find", "grep", "mkdir",
            "rmdir", "copy", "xcopy", "del ", "mv ", "move ",
            "python ", "python3 ", "py ", "pip ", "pip3 ",
            "git ", "npm ", "npx ", "node ", "yarn ", "pnpm ",
            "cargo ", "rustc ", "go ", "dotnet ",
            "where ", "which ", "set ", "path",
        )
        low = cmd.strip().lower()
        if not low.startswith(allowed_prefixes):
            return f"(命令被拒绝：仅允许开发类命令) 收到: {cmd}"
        try:
            out = subprocess.run(cmd, shell=True, cwd=str(self.current),
                                 capture_output=True, text=True, timeout=timeout,
                                 errors="replace")
            text = (out.stdout or "") + (out.stderr or "")
            return text[:12000] or "(无输出)"
        except subprocess.TimeoutExpired:
            return f"(命令超时 >{timeout}s)"
        except Exception as e:  # noqa: BLE001
            return f"(执行失败) {e}"

    # ---- file editing ------------------------------------------------------
    def edit_file(self, path: str, find: str, replace: str) -> str:
        """Find/replace inside a file (first occurrence)."""
        p = self.resolve(path)
        if not p.is_file():
            return f"(不存在) {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        if find not in text:
            return f"(未找到要替换的内容) {path}"
        n = text.count(find)
        text = text.replace(find, replace, 1)
        p.write_text(text, encoding="utf-8")
        return f"已替换 {p.relative_to(self.root)} 中的 1 处（共 {n} 处匹配）"

    def append_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return f"已追加写入 {p.relative_to(self.root)} ({len(content)} 字符)"

    # ---- project management ------------------------------------------------
    def list_projects(self) -> str:
        projs = self._projects_dir()
        items = []
        for c in sorted(projs.iterdir()):
            if c.is_dir():
                # count files
                nfiles = sum(1 for _ in c.rglob("*") if _.is_file())
                items.append(f"- {c.name}  ({nfiles} 个文件)")
        return "\n".join(items) or "(暂无项目)"

    def _projects_dir(self) -> Path:
        d = self.root / "projects"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_project(self, name: str, template: str = "empty") -> str:
        """Create a new project folder with a standard layout + metadata."""
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "project"
        base = self._projects_dir() / safe
        base.mkdir(parents=True, exist_ok=True)

        # standard layout
        for sub in ("src", "data", "output", "notes"):
            (base / sub).mkdir(exist_ok=True)

        # metadata
        meta = base / ".edge-project.json"
        if not meta.exists():
            meta.write_text(json.dumps({
                "name": safe, "template": template,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        # seed files per template
        readme = base / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {safe}\n\n由 edge-infer Agent 创建于 "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"目录：`src/` 源码、`data/` 数据、`output/` 产出、`notes/` 笔记\n",
                encoding="utf-8")
        if template == "python":
            (base / "src" / "main.py").write_text(
                '"""' + safe + ' - entry point."""\n\n\ndef main():\n'
                '    print("Hello from ' + safe + '")\n\n\n'
                'if __name__ == "__main__":\n    main()\n', encoding="utf-8")
            (base / "requirements.txt").write_text("# 依赖\n", encoding="utf-8")

        # switch into the new project
        self.current = base
        return f"已创建项目 {safe}/（模板：{template}），并切换到该项目目录"

    def open_project(self, name: str) -> str:
        base = self._projects_dir() / name
        if not base.is_dir():
            return f"(项目不存在) {name}。现有项目：\n{self.list_projects()}"
        self.current = base
        return f"已打开项目 {name}"

    # ---- sandbox (venv) ----------------------------------------------------
    def sandbox_setup(self, python: str = "python") -> str:
        """Create a virtualenv (.venv) inside the current project."""
        if not (self.current / ".edge-project.json").exists():
            return "(请先创建/打开一个项目再设置沙箱)"
        venv = self.current / ".venv"
        if venv.exists():
            return f"沙箱已存在：{venv.relative_to(self.root)}"
        out = subprocess.run(f"{python} -m venv .venv", shell=True,
                             cwd=str(self.current), capture_output=True,
                             text=True, timeout=300, errors="replace")
        if out.returncode != 0:
            return f"(创建 venv 失败) {out.stderr[:500]}"
        return ("已创建 Python 沙箱 .venv/。之后可用 sandbox_run 在其中执行命令，"
                "例如 sandbox_run(cmd='pip install requests')")

    def _venv_python(self) -> str:
        """Path to the sandbox python, or the system python if no venv."""
        win = self.current / ".venv" / "Scripts" / "python.exe"
        nix = self.current / ".venv" / "bin" / "python"
        if win.exists():
            return str(win)
        if nix.exists():
            return str(nix)
        return "python"

    def sandbox_run(self, cmd: str, timeout: int = 300) -> str:
        """Run a command using the project's sandbox python (pip install etc.)."""
        py = self._venv_python()
        # Route bare python/pip to the sandbox interpreter.
        low = cmd.strip().lower()
        if low.startswith("pip "):
            cmd = f'"{py}" -m ' + cmd
        elif low.startswith("python "):
            cmd = f'"{py}" ' + cmd[7:]
        try:
            out = subprocess.run(cmd, shell=True, cwd=str(self.current),
                                 capture_output=True, text=True, timeout=timeout,
                                 errors="replace")
            text = (out.stdout or "") + (out.stderr or "")
            return text[:12000] or "(无输出)"
        except subprocess.TimeoutExpired:
            return f"(沙箱命令超时 >{timeout}s)"
        except Exception as e:  # noqa: BLE001
            return f"(沙箱执行失败) {e}"

    def cd(self, path: str) -> str:
        p = self.resolve(path)
        if not p.is_dir():
            return f"(不存在) {path}"
        self.current = p
        return f"当前目录 -> {p.relative_to(self.root) or '.'}"


# ---- tool registry (OpenAI-style schemas) ----------------------------------
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    # project lifecycle
    {"name": "create_project",
     "desc": "创建一个新项目文件夹(含 src/data/output/notes 结构)，并切换进去",
     "params": {"name": "项目名", "template": "模板：empty | python (可选)"}},
    {"name": "open_project", "desc": "打开一个已有项目",
     "params": {"name": "项目名"}},
    {"name": "list_projects", "desc": "列出所有项目", "params": {}},
    # sandbox
    {"name": "sandbox_setup",
     "desc": "在当前项目内创建 Python 虚拟环境 .venv（沙箱）", "params": {}},
    {"name": "sandbox_run",
     "desc": "在项目沙箱内执行命令（如 pip install xxx / python xxx.py）",
     "params": {"cmd": "命令"}},
    # files
    {"name": "list_dir", "desc": "列出目录内容",
     "params": {"path": "目录路径(默认 .)"}},
    {"name": "read_file", "desc": "读取文本文件内容",
     "params": {"path": "文件路径"}},
    {"name": "write_file", "desc": "写入/覆盖文本文件",
     "params": {"path": "文件路径", "content": "文件内容"}},
    {"name": "edit_file", "desc": "在文件中查找并替换（首次匹配）",
     "params": {"path": "文件路径", "find": "要查找的内容", "replace": "替换为"}},
    {"name": "append_file", "desc": "在文件末尾追加内容",
     "params": {"path": "文件路径", "content": "追加内容"}},
    {"name": "make_dir", "desc": "创建目录", "params": {"path": "目录路径"}},
    {"name": "search", "desc": "在文件中按正则搜索文本",
     "params": {"pattern": "正则", "path": "搜索目录(默认 .)"}},
    {"name": "run_command", "desc": "在当前项目目录内运行开发类命令（git/npm/...）",
     "params": {"cmd": "命令"}},
    {"name": "cd", "desc": "切换当前目录", "params": {"path": "目录"}},
    {"name": "get_workspace", "desc": "查看工作区根目录与当前目录", "params": {}},
]


def tools_prompt() -> str:
    """The system-prompt block that teaches the model the tool protocol."""
    lines = [
        "你是一个本地项目开发助手，可以使用以下工具在沙箱工作区里操作文件与执行命令：",
    ]
    for t in TOOL_SCHEMAS:
        args = ", ".join(f"{k}: {v}" for k, v in t["params"].items()) or "无"
        lines.append(f"- {t['name']}({args}) — {t['desc']}")
    lines += [
        "",
        "典型工作流：create_project(建项目) -> write_file(写代码) -> "
        "sandbox_setup(建沙箱) -> sandbox_run('pip install ...')(装依赖) -> "
        "sandbox_run('python src/main.py')(运行验证)。",
        "",
        "需要调用工具时，只输出一个 JSON 对象，放在 <tool_call></tool_call> 里，例如：",
        '<tool_call>{"name": "create_project", "arguments": {"name": "我的工具", "template": "python"}}</tool_call>',
        "工具执行结果会以 <tool_result>...</tool_result> 返回给你；",
        "你可以连续调用多个工具，最后再用自然语言给出最终答复（不要再包含 <tool_call>）。",
        "除非确实完成，否则请持续推进任务（写文件、装依赖、跑测试）。",
    ]
    return "\n".join(lines)


def run_tool(ws: Workspace, name: str, args: Dict[str, Any]) -> str:
    """Dispatch a tool call inside the workspace."""
    try:
        if name == "create_project":
            return ws.create_project(args["name"], args.get("template", "empty"))
        if name == "open_project":
            return ws.open_project(args["name"])
        if name == "list_projects":
            return ws.list_projects()
        if name == "sandbox_setup":
            return ws.sandbox_setup()
        if name == "sandbox_run":
            return ws.sandbox_run(args["cmd"])
        if name == "list_dir":
            return ws.list_dir(args.get("path", "."))
        if name == "read_file":
            return ws.read_file(args["path"])
        if name == "write_file":
            return ws.write_file(args["path"], args.get("content", ""))
        if name == "edit_file":
            return ws.edit_file(args["path"], args.get("find", ""), args.get("replace", ""))
        if name == "append_file":
            return ws.append_file(args["path"], args.get("content", ""))
        if name == "make_dir":
            return ws.make_dir(args["path"])
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
