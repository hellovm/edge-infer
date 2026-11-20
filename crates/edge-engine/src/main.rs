//! CLI: build and pretty-print a run plan from a JSON probe + model spec.
//!
//! Usage:
//!   edge-engine probe.json model.json <context> <large_threshold_gb>

use edge_engine::build_run_plan;
use edge_detect::ProbeInput;
use edge_tiers::{ExpertPlacement, ModelParams};
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct ModelSpec {
    resident_bytes: u64,
    kv_bytes_per_token: u64,
    num_experts: u32,
    expert_size_bytes: u64,
    /// Heat-ordered predicted top-K for the *next* token.
    predicted: Vec<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (probe_path, model_path, context, threshold_gb) = match args.len() {
        4 => (
            PathBuf::from(&args[0]),
            PathBuf::from(&args[1]),
            args[2].parse::<usize>()?,
            args[3].parse::<u64>()?,
        ),
        _ => {
            eprintln!("usage: edge-engine <probe.json> <model.json> <context> <large_threshold_gb>");
            std::process::exit(2);
        }
    };

    let probe: ProbeInput = serde_json::from_str(&std::fs::read_to_string(&probe_path)?)?;
    let spec: ModelSpec = serde_json::from_str(&std::fs::read_to_string(&model_path)?)?;

    let model = ModelParams {
        resident_bytes: spec.resident_bytes,
        kv_bytes_per_token: spec.kv_bytes_per_token,
        num_experts: spec.num_experts,
        expert_size_bytes: spec.expert_size_bytes,
    };

    let experts: Vec<ExpertPlacement> = (0..spec.num_experts)
        .map(|i| ExpertPlacement {
            index: i,
            heat: (spec.num_experts as u64 - i as u64), // lower index = hotter
            size_bytes: spec.expert_size_bytes,
            tier: edge_tiers::Tier::Ssd,
        })
        .collect();

    let plan = build_run_plan(
        &probe,
        &model,
        &experts,
        &spec.predicted,
        context,
        threshold_gb * 1_000_000_000,
    )?;

    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}
