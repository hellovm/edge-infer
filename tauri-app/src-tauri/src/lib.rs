//! edge-infer Tauri GUI backend.
//!
//! The GUI is a thin shell over the real work:
//!   - hardware report  -> `edge-detect` CLI (probe JSON in, report JSON out)
//!   - run plan (3-tier)-> `edge-engine` CLI (probe+model in, plan JSON out)
//!   - inference        -> Python OpenAI-compatible service (127.0.0.1:8080)
//!
//! On startup the app AUTO-LAUNCHES the Python service (sidecar), so end users
//! never run `run_server.py` by hand. The child process is killed on exit.

mod commands;
mod sidecar;

use sidecar::ServerProc;
use std::sync::Mutex;
use tauri::Manager;

pub use commands::{list_devices, run_plan, service_health};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ServerProc(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            commands::list_devices,
            commands::run_plan,
            commands::service_health,
            commands::start_service,
            commands::service_status,
        ])
        .setup(|app| {
            // Auto-start the Python inference service in the background.
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let proc: tauri::State<ServerProc> = handle.state();
                match sidecar::ensure_server(&proc) {
                    Ok(launched) => {
                        if launched {
                            println!("[edge-infer] 推理服务已自动启动 (:8080)");
                        } else {
                            println!("[edge-infer] 推理服务已在运行");
                        }
                    }
                    Err(e) => eprintln!("[edge-infer] 自动启动服务失败: {e}"),
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            // Kill the sidecar when the main window closes.
            if let tauri::WindowEvent::Destroyed = event {
                let proc: tauri::State<ServerProc> = window.state();
                sidecar::stop_server(&proc);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
