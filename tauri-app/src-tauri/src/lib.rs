//! edge-infer Tauri GUI backend.
//!
//! The GUI is a thin shell over the real work:
//!   - hardware report  -> `edge-detect` CLI (probe JSON in, report JSON out)
//!   - run plan (3-tier)-> `edge-engine` CLI (probe+model in, plan JSON out)
//!   - 4B chat          -> Python OpenAI-compatible service (127.0.0.1:8080)
//!
//! Commands live in `commands.rs` to avoid macro-name collisions at the crate
//! root; this file only wires them into the Tauri builder.

mod commands;

pub use commands::{list_devices, run_plan, service_health};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            commands::list_devices,
            commands::run_plan,
            commands::service_health
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
