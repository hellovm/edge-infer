//! Sidecar server management: auto-launch the Python inference service when the
//! GUI starts, so end users never have to run `run_server.py` by hand.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Holds the spawned Python server process so we can kill it on exit.
pub struct ServerProc(pub Mutex<Option<Child>>);

/// Locate the repo root relative to the running exe.
/// Packaged layout:   <root>/edge-gui.exe            -> root
/// Dev layout:        <root>/tauri-app/src-tauri/target/release/edge-gui.exe
fn find_repo_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut dir = exe.parent()?.to_path_buf();
    // walk up looking for python/edge_server/openai_server.py
    for _ in 0..6 {
        if dir.join("python").join("edge_server").join("openai_server.py").exists() {
            return Some(dir);
        }
        if dir.join("edge_server").join("openai_server.py").exists() {
            // packaged: exe sits next to python/ contents
            return Some(dir);
        }
        dir = dir.parent()?.to_path_buf();
    }
    None
}

/// Find a usable Python interpreter (bundled venv -> python on PATH -> py).
fn find_python(repo_root: &PathBuf) -> Option<String> {
    let candidates = [
        repo_root.join("python").join(".venv").join("Scripts").join("python.exe"),
        repo_root.join(".venv").join("Scripts").join("python.exe"),
    ];
    for c in candidates {
        if c.exists() {
            return Some(c.to_string_lossy().into_owned());
        }
    }
    // fall back to PATH
    for name in ["python", "python3", "py"] {
        if Command::new(name).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(name.to_string());
        }
    }
    None
}

/// Check whether the service answers /health.
pub fn service_up() -> bool {
    use std::net::TcpStream;
    // quick TCP probe on 8080
    TcpStream::connect_timeout(
        &"127.0.0.1:8080".parse().unwrap(),
        Duration::from_millis(400),
    )
    .is_ok()
}

/// Start the Python server if it isn't already running.
/// Returns Ok(true) if we launched it, Ok(false) if it was already up.
pub fn ensure_server(proc: &ServerProc) -> Result<bool, String> {
    if service_up() {
        return Ok(false);
    }

    let repo_root = find_repo_root()
        .ok_or_else(|| "找不到项目根目录（python/edge_server 不存在）".to_string())?;
    let python = find_python(&repo_root)
        .ok_or_else(|| "未找到 Python 解释器，请安装 Python 3.10+ 并加入 PATH".to_string())?;

    // Working dir: <root>/python  (so `-m edge_server.openai_server` resolves)
    let workdir = if repo_root.join("python").join("edge_server").exists() {
        repo_root.join("python")
    } else {
        repo_root.clone()
    };

    let mut cmd = Command::new(&python);
    cmd.args(["-m", "uvicorn", "edge_server.openai_server:app",
              "--host", "0.0.0.0", "--port", "8080"])
        .current_dir(&workdir)
        .env("EDGE_DEVICE", std::env::var("EDGE_DEVICE").unwrap_or_else(|_| "AUTO".into()))
        .env("PYTHONIOENCODING", "utf-8")
        // Pass our PID so the Python side can self-terminate if we die
        // (covers the case where the GUI is force-killed).
        .env("EDGE_PARENT_PID", std::process::id().to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Point at the 8B model if present (better Agent behaviour).
    if std::env::var("EDGE_MODEL_DIR").is_err() {
        let m = repo_root.join("models").join("qwen3-8b-int4-cw");
        if m.exists() {
            cmd.env("EDGE_MODEL_DIR", m);
        }
    }

    // Hide the console window on Windows.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let child = cmd.spawn()
        .map_err(|e| format!("启动服务失败：{e}"))?;

    if let Ok(mut guard) = proc.0.lock() {
        *guard = Some(child);
    }

    // Wait until /health responds (up to ~60s; model load can be slow).
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(90) {
        if service_up() {
            return Ok(true);
        }
        std::thread::sleep(Duration::from_millis(700));
    }
    Err("服务启动超时（可能是首次加载模型较慢，请稍候重试）".into())
}

/// Kill the spawned server (called on app exit).
pub fn stop_server(proc: &ServerProc) {
    if let Ok(mut guard) = proc.0.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
