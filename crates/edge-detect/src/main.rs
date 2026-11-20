//! CLI: print a normalized hardware report from a JSON probe input.
//!
//! Usage: edge-detect probe.json
//!
//! In this slice the probe input is provided as JSON (the real NVML/OpenVINO
//! probing is the next slice; the reporting logic is already real).

use edge_detect::{build_report, ProbeInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: edge-detect <probe.json>")?;
    let probe: ProbeInput = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let report = build_report(&probe);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
