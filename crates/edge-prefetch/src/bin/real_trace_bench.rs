//! Real-trace cache benchmark: validates the adaptive expert cache against
//! LRU/LFU on an actual MoE routing trace captured from llama.cpp's
//! expert-stats endpoint (not a synthetic trace).
//!
//! Run: cargo run --release -p edge-prefetch --bin real_trace_bench -- <trace.json>

use edge_prefetch::{adaptive_hit_rate, lru_hit_rate, CacheConfig};
use std::collections::{HashMap, HashSet};
use std::fs;

fn lfu_hit_rate(trace: &[u32], cap: usize, decay: f64) -> f64 {
    let mut freq: HashMap<u32, f64> = HashMap::new();
    let mut resident: HashSet<u32> = HashSet::new();
    let mut hits = 0usize;
    for &e in trace {
        if resident.contains(&e) {
            hits += 1;
        }
        for v in freq.values_mut() {
            *v *= decay;
        }
        *freq.entry(e).or_insert(0.0) += 1.0;
        resident.insert(e);
        if resident.len() > cap {
            if let Some((&victim, _)) = freq.iter().filter(|(k, _)| resident.contains(k))
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                resident.remove(&victim);
            }
        }
    }
    hits as f64 / trace.len().max(1) as f64
}

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "D:/AI/tools-src/real_trace_long.json".into());
    let raw = fs::read_to_string(&path).expect("trace file");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
    let meta = &doc["meta"];
    println!("真实 MoE 路由轨迹");
    println!("  模型: {}", meta["model"].as_str().unwrap_or("?"));
    println!("  生成 token: {}, 层: {}, 专家/层: {}, top-{}",
        meta["generated_tokens"], meta["n_layers"], meta["n_expert"], meta["n_expert_used"]);
    println!();

    let traces = doc["traces_by_layer"].as_object().expect("traces_by_layer");
    let layer_ids = ["0", "1", "2", "11", "23", "47"];
    println!("{:>8}  {:>9}  {:>9}  {:>11}  {:>10}",
        "capacity", "LRU", "LFU", "adaptive", "vs best");
    println!("{}", "-".repeat(54));

    let mut all_gains = Vec::new();
    for &cap in &[16usize, 32, 48, 64, 96] {
        let (mut lru_sum, mut lfu_sum, mut ad_sum) = (0.0, 0.0, 0.0);
        let mut n = 0;
        for il in layer_ids {
            let Some(arr) = traces.get(il).and_then(|v| v.as_array()) else { continue };
            if arr.len() < 100 { continue; }
            let tr: Vec<u32> = arr.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
            lru_sum += lru_hit_rate(&tr, cap);
            lfu_sum += lfu_hit_rate(&tr, cap, 0.98);
            ad_sum += adaptive_hit_rate(&tr, cap, CacheConfig::default()).0;
            n += 1;
        }
        if n == 0 { continue; }
        let (lru, lfu, ad) = (lru_sum / n as f64, lfu_sum / n as f64, ad_sum / n as f64);
        let best = lru.max(lfu);
        let gain = (ad / best - 1.0) * 100.0;
        all_gains.push(gain);
        println!("{:>8}  {:>8.1}%  {:>8.1}%  {:>10.1}%  {:>+9.1}%",
            cap, lru * 100.0, lfu * 100.0, ad * 100.0, gain);
    }
    if !all_gains.is_empty() {
        let mean = all_gains.iter().sum::<f64>() / all_gains.len() as f64;
        let min = all_gains.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = all_gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("\n平均提升: {:+.1}%  (范围 {:+.1}% ~ {:+.1}%)", mean, min, max);
    }
}
