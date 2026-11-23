//! Expert-cache policy benchmark.
//!
//! Compares LRU, LFU and the adaptive ghost-list policy on MoE routing traces
//! that reproduce the two classic failure modes deliberately:
//!
//!   - **Phase-shift workload** — the hot set moves over time (prose -> code ->
//!     tool calls). Pure LFU keeps stale favourites and loses.
//!   - **Scan/burst workload** — long one-pass scans interleaved with reuse of
//!     a working set. Pure LRU evicts the working set mid-scan and loses.
//!
//! The claim under test is robustness: the adaptive policy should stay close
//! to the best baseline on *both* workloads without being told which it is on.
//!
//! Run: `cargo run --release -p edge-prefetch --bin cache_bench`

use edge_prefetch::{adaptive_hit_rate, lru_hit_rate, CacheConfig};
use std::collections::HashMap;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn trace_phase_shift(n_steps: usize, n_experts: u32, hot: usize, phase_len: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed);
    let mut out = Vec::with_capacity(n_steps * hot);
    for t in 0..n_steps {
        let base = ((t / phase_len) as u32).wrapping_mul(131) % n_experts;
        for _ in 0..hot {
            let x = r.next();
            let e = match x % 100 {
                0..=59 => (base + (x >> 7) % 8) % n_experts,
                60..=94 => (base + (x >> 3) % 64) % n_experts,
                _ => x % n_experts,
            };
            out.push(e);
        }
    }
    out
}

fn trace_scan(n_steps: usize, n_experts: u32, hot: usize, scan_len: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng::new(seed);
    let mut out = Vec::with_capacity(n_steps * hot);
    for t in 0..n_steps {
        if t % 400 == 0 && t > 0 {
            let base = r.next() % n_experts;
            for k in 0..scan_len as u32 {
                out.push((base.wrapping_add(k)) % n_experts);
            }
        }
        for _ in 0..hot {
            let x = r.next();
            let e = match x % 100 {
                0..=64 => x % 16,
                65..=94 => x % 48,
                _ => x % n_experts,
            };
            out.push(e);
        }
    }
    out
}

fn lfu_hits(trace: &[u32], cap: usize, decay: f64) -> f64 {
    let mut m: HashMap<u32, f64> = HashMap::new();
    let mut h = 0;
    for &e in trace {
        if m.get(&e).copied().unwrap_or(0.0) > 0.0 { h += 1; }
        for v in m.values_mut() { *v *= decay; }
        *m.entry(e).or_insert(0.0) += 1.0;
        if m.len() > cap {
            if let Some((&k, _)) = m.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) { m.remove(&k); }
        }
    }
    h as f64 / trace.len().max(1) as f64
}

fn report(name: &str, tr: &[u32], caps: &[usize]) {
    println!("\n=== {name} ({:.0}k accesses) ===", tr.len() as f64 / 1000.0);
    println!("{:>10}  {:>9}  {:>9}  {:>11}  {:>11}  {:>9}",
        "capacity", "LRU", "LFU", "adaptive", "vs best", "rec_frac");
    println!("{}", "-".repeat(66));
    for &cap in caps {
        let lru = lru_hit_rate(tr, cap);
        let lfu = lfu_hits(tr, cap, 0.98);
        let (ad, rf) = adaptive_hit_rate(tr, cap, CacheConfig::default());
        let best = lru.max(lfu);
        let gain = if best > 0.0 { (ad / best - 1.0) * 100.0 } else { 0.0 };
        println!("{:>10}  {:>8.1}%  {:>8.1}%  {:>10.1}%  {gain:>+10.1}%  {:>9.3}",
            cap, lru * 100.0, lfu * 100.0, ad * 100.0, rf);
    }
}

fn main() {
    let n_experts = 512u32;
    let hot = 10usize;
    let caps = [16usize, 32, 48, 64];

    let ps = trace_phase_shift(20_000, n_experts, hot, 400, 0xDEADBEEF);
    let sc = trace_scan(20_000, n_experts, hot, 300, 0x12345678);

    println!("MoE expert-cache policy benchmark");
    println!("Two workloads that expose LRU and LFU failure modes respectively.");

    report("Phase-shift workload (hurts LFU)", &ps, &caps);
    report("Scan/burst workload (hurts LRU)", &sc, &caps);

    println!("\n=== Robustness summary ===");
    println!("worst-case hit rate relative to the best baseline, across both workloads:");
    for &cap in &caps {
        let mut lru_w = f64::INFINITY;
        let mut lfu_w = f64::INFINITY;
        let mut ad_w = f64::INFINITY;
        for tr in [&ps, &sc] {
            let lru = lru_hit_rate(tr, cap);
            let lfu = lfu_hits(tr, cap, 0.98);
            let (ad, _) = adaptive_hit_rate(tr, cap, CacheConfig::default());
            let best = lru.max(lfu);
            lru_w = lru_w.min(lru / best);
            lfu_w = lfu_w.min(lfu / best);
            ad_w = ad_w.min(ad / best);
        }
        println!("  cap={cap:>3}:  LRU {:>6.1}%   LFU {:>6.1}%   adaptive {:>6.1}%",
            lru_w * 100.0, lfu_w * 100.0, ad_w * 100.0);
    }
}
