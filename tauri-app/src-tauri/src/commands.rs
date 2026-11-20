//! Tauri commands. Kept in a dedicated module so the `#[tauri::command]`
//! generated macros don't collide at the crate root.

use serde::{Deserialize, Serialize};
use std::process::Command;

const EDGE_DETECT: &str = "edge-detect.exe";
const EDGE_ENGINE: &str = "edge-engine.exe";
const OPENAI_BASE: &str = "http://127.0.0.1:8080";

#[derive(Serialize, Deserialize, Clone)]
pub struct ProbeJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvidia: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub npu: Option<serde_json::Value>,
    pub total_ram_bytes: u64,
    pub os_overhead: serde_json::Value,
}

/// Locate a sibling CLI exe (packaged layout), else fall back to bare name.
fn find_cli(name: &str) -> String {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(name);
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    name.to_string()
}

fn probe_json() -> ProbeJson {
    ProbeJson {
        nvidia: Some(serde_json::json!({
            "name": "RTX 4070 Laptop",
            "vram_bytes": 4294967296u64,
            "bandwidth_bytes_per_s": 309237632000u64
        })),
        npu: Some(serde_json::json!({
            "name": "Intel AI Boost",
            "tops": 47,
            "memory_budget_bytes": 4294967296u64
        })),
        total_ram_bytes: 17179869184,
        os_overhead: serde_json::json!({
            "idle_baseline_bytes": 5368709120u64,
            "igpu_dvmt_bytes": 536870912u64,
            "hidden_bytes": 1073741824u64
        }),
    }
}

#[tauri::command]
pub fn list_devices() -> Result<Vec<serde_json::Value>, String> {
    let probe = probe_json();
    let tmp = std::env::temp_dir().join("edge_probe.json");
    std::fs::write(&tmp, serde_json::to_vec(&probe).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let out = Command::new(find_cli(EDGE_DETECT))
        .arg(tmp.as_os_str())
        .output()
        .map_err(|e| format!("failed to run edge-detect: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let report: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
    report
        .get("devices")
        .cloned()
        .and_then(|d| d.as_array().cloned())
        .ok_or_else(|| "no devices in report".to_string())
}

#[tauri::command]
pub fn run_plan(context: u32, large_threshold_gb: u32) -> Result<serde_json::Value, String> {
    let probe = probe_json();
    let model = serde_json::json!({
        "resident_bytes": 1500000000u64,
        "kv_bytes_per_token": 80000u64,
        "num_experts": 256u32,
        "expert_size_bytes": 11700000u64,
        "predicted": [0,1,2,3,4,5,6,7,8]
    });

    let pt = std::env::temp_dir().join("edge_probe_cli.json");
    let mt = std::env::temp_dir().join("edge_model_cli.json");
    std::fs::write(&pt, serde_json::to_vec(&probe).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    std::fs::write(&mt, serde_json::to_vec(&model).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let out = Command::new(find_cli(EDGE_ENGINE))
        .arg(pt.as_os_str())
        .arg(mt.as_os_str())
        .arg(context.to_string())
        .arg(large_threshold_gb.to_string())
        .output()
        .map_err(|e| format!("failed to run edge-engine: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn service_health() -> Result<serde_json::Value, String> {
    // HTTP GET via PowerShell (no extra HTTP dependency).
    let script = format!(
        "(Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 '{OPENAI_BASE}/health').Content"
    );
    let out = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .map_err(|e| format!("failed to reach service: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())
}
