//! CLI: build and pretty-print a run plan from a JSON probe + model spec,
//! or benchmark the adaptive expert-cache policies.
//!
//! Usage:
//!   edge-engine probe.json model.json <context> <large_threshold_gb>
//!   edge-engine cache-bench [steps]

use edge_engine::build_run_plan;
use edge_detect::ProbeInput;
use edge_prefetch::{adaptive_hit_rate, lru_hit_rate, CacheConfig};
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

    if args.first().map(|s| s.as_str()) == Some("cache-bench") {
        let steps = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
        cache_bench(steps);
        return Ok(());
    }

    let (probe_path, model_path, context, threshold_gb) = match args.len() {
        4 => (
            PathBuf::from(&args[0]),
            PathBuf::from(&args[1]),
            args[2].parse::<usize>()?,
            args[3].parse::<u64>()?,
        ),
        _ => {
            eprintln!("usage: edge-engine <probe.json> <model.json> <context> <large_threshold_gb>");
            eprintln!("       edge-engine cache-bench [steps]");
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

/// Print a comparison of the adaptive cache against LRU on two MoE routing
/// workloads. Hit rate maps almost directly to decode tok/s here (a miss is
/// an SSD read on the critical path).
fn cache_bench(steps: usize) {
    use std::collections::HashMap;

    fn rng_next(s: &mut u64) -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }
    fn phase_shift(n: usize, ne: u32, hot: usize, plen: usize, seed: u64) -> Vec<u32> {
        let mut s = seed | 1;
        let mut out = Vec::new();
        for t in 0..n {
            let base = ((t / plen) as u32).wrapping_mul(131) % ne;
            for _ in 0..hot {
                let x = rng_next(&mut s);
                out.push(match x % 100 {
                    0..=59 => (base + (x >> 7) % 8) % ne,
                    60..=94 => (base + (x >> 3) % 64) % ne,
                    _ => x % ne,
                });
            }
        }
        out
    }
    fn scan(n: usize, ne: u32, hot: usize, slen: usize, seed: u64) -> Vec<u32> {
        let mut s = seed | 1;
        let mut out = Vec::new();
        for t in 0..n {
            if t % 400 == 0 && t > 0 {
                let base = rng_next(&mut s) % ne;
                for k in 0..slen as u32 { out.push((base.wrapping_add(k)) % ne); }
            }
            for _ in 0..hot {
                let x = rng_next(&mut s);
                out.push(match x % 100 { 0..=64 => x % 16, 65..=94 => x % 48, _ => x % ne });
            }
        }
        out
    }
    fn lfu(trace: &[u32], cap: usize) -> f64 {
        let mut m: HashMap<u32, f64> = HashMap::new();
        let mut h = 0;
        for &e in trace {
            if m.get(&e).copied().unwrap_or(0.0) > 0.0 { h += 1; }
            for v in m.values_mut() { *v *= 0.98; }
            *m.entry(e).or_insert(0.0) += 1.0;
            if m.len() > cap {
                if let Some((&k, _)) = m.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) { m.remove(&k); }
            }
        }
        h as f64 / trace.len().max(1) as f64
    }

    let ne = 512u32;
    let ps = phase_shift(steps, ne, 10, 400, 0xDEADBEEF);
    let sc = scan(steps, ne, 10, 300, 0x12345678);
    println!("edge-infer adaptive expert-cache benchmark ({steps} steps/trace)");
    for (name, tr) in [("phase-shift (hurts LFU)", &ps), ("scan/burst (hurts LRU)", &sc)] {
        println!("\n=== {name} ===");
        println!("{:>10}  {:>9}  {:>9}  {:>11}  {:>9}", "capacity", "LRU", "LFU", "adaptive", "vs best");
        for &cap in &[16usize, 32, 48, 64] {
            let lru = lru_hit_rate(tr, cap);
            let l = lfu(tr, cap);
            let (ad, _) = adaptive_hit_rate(tr, cap, CacheConfig::default());
            let best = lru.max(l);
            println!("{:>10}  {:>8.1}%  {:>8.1}%  {:>10.1}%  {:>+8.1}%",
                cap, lru * 100.0, l * 100.0, ad * 100.0, (ad / best - 1.0) * 100.0);
        }
    }
}
