//! Tauri commands. Kept in a dedicated module so the `#[tauri::command]`
//! generated macros don't collide at the crate root.

use crate::sidecar::{ensure_server, service_up, ServerProc};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tauri::State;

const EDGE_DETECT: &str = "edge-detect.exe";
const EDGE_ENGINE: &str = "edge-engine.exe";
const OPENAI_BASE: &str = "http://127.0.0.1:8080";

#[tauri::command]
pub fn start_service(proc: State<'_, ServerProc>) -> Result<serde_json::Value, String> {
    let launched = ensure_server(&proc)?;
    Ok(serde_json::json!({
        "ok": true,
        "launched": launched,
        "up": service_up(),
    }))
}

#[tauri::command]
pub fn service_status() -> serde_json::Value {
    serde_json::json!({ "up": service_up() })
}

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

// ---------------------------------------------------------------------------
// update check (GitHub Releases)
// ---------------------------------------------------------------------------
const REPO: &str = "hellovm/edge-infer";

/// Parse a dotted version (tolerates a leading `v`) into comparable integers.
fn parse_version(v: &str) -> Vec<u64> {
    v.trim()
        .trim_start_matches('v')
        .split(['.', '-', '+'])
        .map(|p| p.parse::<u64>().unwrap_or(0))
        .collect()
}

/// True when `remote` is a strictly newer release than `local`.
fn is_newer(remote: &str, local: &str) -> bool {
    let (r, l) = (parse_version(remote), parse_version(local));
    for i in 0..r.len().max(l.len()) {
        let a = r.get(i).copied().unwrap_or(0);
        let b = l.get(i).copied().unwrap_or(0);
        if a != b {
            return a > b;
        }
    }
    false
}

/// Ask GitHub for the latest release and compare it with the running version.
///
/// Network failures are reported as `{ ok:false, ... }` rather than an Err so
/// the GUI can stay silent (an offline user must not see an error banner).
#[tauri::command]
pub fn check_update() -> serde_json::Value {
    let current = env!("CARGO_PKG_VERSION").to_string();
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    // Invoke-RestMethod parses JSON for us; force TLS 1.2 and send a UA
    // (GitHub rejects requests without one).
    let script = format!(
        "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; \
         $h=@{{'User-Agent'='edge-infer';'Accept'='application/vnd.github+json'}}; \
         (Invoke-RestMethod -UseBasicParsing -TimeoutSec 8 -Headers $h '{url}') | \
         ConvertTo-Json -Depth 5"
    );
    let out = match Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            return serde_json::json!({
                "ok": false, "current": current,
                "error": String::from_utf8_lossy(&o.stderr).trim().to_string(),
            });
        }
        Err(e) => {
            return serde_json::json!({"ok": false, "current": current, "error": e.to_string()});
        }
    };

    let rel: serde_json::Value = match serde_json::from_slice(&out.stdout) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({
                "ok": false, "current": current,
                "error": format!("解析发布信息失败: {e}"),
            });
        }
    };

    let tag = rel.get("tag_name").and_then(|v| v.as_str()).unwrap_or("");
    let html = rel.get("html_url").and_then(|v| v.as_str()).unwrap_or("");
    let name = rel.get("name").and_then(|v| v.as_str()).unwrap_or(tag);
    let notes = rel.get("body").and_then(|v| v.as_str()).unwrap_or("");

    // Preferred download: the .exe installer asset, else the release page.
    let mut download = html.to_string();
    if let Some(assets) = rel.get("assets").and_then(|a| a.as_array()) {
        for a in assets {
            let an = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if an.ends_with(".exe") || an.ends_with(".msi") {
                if let Some(u) = a.get("browser_download_url").and_then(|v| v.as_str()) {
                    download = u.to_string();
                    break;
                }
            }
        }
    }

    serde_json::json!({
        "ok": true,
        "current": current,
        "latest": tag.trim_start_matches('v'),
        "has_update": is_newer(tag, &current),
        "name": name,
        "notes": notes,
        "url": html,
        "download": download,
    })
}

#[cfg(test)]
mod tests {
    use super::{is_newer, parse_version};

    #[test]
    fn version_compare() {
        assert!(is_newer("v0.4.0", "0.3.1"));
        assert!(is_newer("0.3.2", "0.3.1"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(!is_newer("0.3.1", "0.3.1"));
        assert!(!is_newer("v0.3.0", "0.3.1"));
        assert_eq!(parse_version("v0.3.1"), vec![0, 3, 1]);
    }
}

