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

/// Find a usable Python interpreter.
///
/// A double-clicked GUI app inherits the *system* PATH, which usually does not
/// include developer-only shims (e.g. WorkBuddy's bundled python). So we probe
/// concrete locations first and only then fall back to PATH.
fn find_python(repo_root: &PathBuf) -> Option<String> {
    let mut candidates: Vec<PathBuf> = vec![
        repo_root.join("python").join(".venv").join("Scripts").join("python.exe"),
        repo_root.join(".venv").join("Scripts").join("python.exe"),
    ];
    // WorkBuddy's bundled interpreter (this machine) — common user-local paths.
    if let Some(home) = std::env::var_os("USERPROFILE") {
        let home = PathBuf::from(home);
        candidates.push(home.join(".workbuddy").join("binaries").join("python").join("versions").join("3.13.12").join("python.exe"));
        // Generic `%LOCALAPPDATA%\Programs\Python\Python3xx\python.exe`.
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            let local = PathBuf::from(local);
            if let Ok(entries) = std::fs::read_dir(local.join("Programs").join("Python")) {
                for e in entries.flatten() {
                    let p = e.path().join("python.exe");
                    if p.exists() { candidates.push(p); }
                }
            }
        }
    }
    for c in candidates {
        if c.exists()
            && Command::new(&c).arg("-c").arg("import uvicorn")
                .output().map(|o| o.status.success()).unwrap_or(false)
        {
            return Some(c.to_string_lossy().into_owned());
        }
    }
    // fall back to PATH (only if it really has uvicorn available)
    for name in ["python", "python3", "py"] {
        if Command::new(name).arg("-c").arg("import uvicorn")
            .output().map(|o| o.status.success()).unwrap_or(false)
        {
            return Some(name.to_string());
        }
    }
    None
}

/// Check whether the service really answers /health over HTTP.
///
/// A bare TCP probe is not enough: the socket can be bound while the ASGI app
/// is still importing (or even belongs to a stale process). Frontend probes
/// `/health` too, so use the same signal here.
pub fn service_up() -> bool {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let Ok(mut s) = TcpStream::connect_timeout(
        &"127.0.0.1:8080".parse().unwrap(), Duration::from_millis(400)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(800)));
    let req = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: close\r\n\r\n";
    if s.write_all(req).is_err() { return false; }
    let mut buf = [0u8; 512];
    match s.read(&mut buf) {
        Ok(n) if n > 0 => {
            let head = String::from_utf8_lossy(&buf[..n]);
            head.contains("200")
        }
        _ => false,
    }
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

    // Keep a sidecar log so a failed launch is diagnosable. Previously both
    // streams went to /dev/null, so a missing module or port conflict left the
    // GUI hanging with no clue.
    let log_path = repo_root.join("logs").join("server-sidecar.log");
    let _ = std::fs::create_dir_all(log_path.parent().unwrap());
    let log_file = std::fs::File::create(&log_path)
        .or_else(|_| std::fs::File::create(std::env::temp_dir().join("edge-sidecar.log")))
        .ok();

    let mut cmd = Command::new(&python);
    cmd.args(["-m", "uvicorn", "edge_server.openai_server:app",
              "--host", "127.0.0.1", "--port", "8080"])
        .current_dir(&workdir)
        .env("EDGE_DEVICE", std::env::var("EDGE_DEVICE").unwrap_or_else(|_| "AUTO".into()))
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        // Pass our PID so the Python side can self-terminate if we die
        // (covers the case where the GUI is force-killed).
        .env("EDGE_PARENT_PID", std::process::id().to_string());
    if let Some(f) = log_file {
        cmd.stdout(Stdio::from(f.try_clone().ok().unwrap()));
        cmd.stderr(Stdio::from(f));
    } else {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    }

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
