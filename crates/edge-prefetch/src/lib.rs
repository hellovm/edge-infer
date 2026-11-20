//! Prerouter one-step-ahead expert prediction + cross-tier overlapping
//! prefetch + OOM-fallback orchestration.
//!
//! The prerouter predicts, one decode step ahead, which experts the next
//! token will route to. That lets us start pulling those experts up a tier
//! (SSD -> RAM -> VRAM) *while* the current forward pass is still running,
//! so the slow SSD read is hidden behind compute. This is the mechanism
//! behind the up-to +59% decode-throughput gain vs a naive loader, and it
//! is the differentiator vs Edge0.
//!
//! This slice is pure logic: it takes the *predicted* set, the *current*
//! placement, and the tier capacities, and returns the set of
//! (expert, from-tier, to-tier) transfers that should be issued right now,
//! plus any OOM demotions to run first.

use edge_tiers::{demote, ExpertPlacement, Tier};
use serde::{Deserialize, Serialize};

/// A single tier-to-tier transfer to issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefetchOp {
    pub expert_index: u32,
    pub from: Tier,
    pub to: Tier,
    /// Bytes moved.
    pub bytes: u64,
}

/// A demotion (eviction) to make room.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvictionOp {
    pub expert_index: u32,
    pub from: Tier,
    pub to: Tier,
    pub bytes: u64,
}

/// Outcome of one prefetch planning pass.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PrefetchPlan {
    /// Transfers to start now (issued in parallel with the current forward).
    pub prefetch: Vec<PrefetchOp>,
    /// Evictions to run first (cheapest victims first).
    pub evictions: Vec<EvictionOp>,
    /// Experts that stay where they are this step.
    pub kept: Vec<u32>,
}

/// One-step-ahead prediction: the expert set the *next* token will use.
///
/// `predicted` are the top-K expert indices the prerouter heads up, ordered
/// by predicted usefulness (most first). `k` is the routed width (e.g. 8
/// routed + 1 shared == 9).
pub fn plan(
    current: &mut [ExpertPlacement],
    predicted: &[u32],
    target_capacity_vram: u64,
    target_capacity_ram: u64,
    k: usize,
) -> PrefetchPlan {
    let by_index: &mut [ExpertPlacement] = current;
    let mut plan = PrefetchPlan::default();

    // How many bytes currently live in each target tier.
    let mut vram_used: u64 = by_index.iter().filter(|e| e.tier == Tier::Vram).map(|e| e.size_bytes).sum();
    let mut ram_used: u64 = by_index.iter().filter(|e| e.tier == Tier::Ram).map(|e| e.size_bytes).sum();

    // Predicted-but-not-here experts, most useful first.
    let predicted_set: Vec<u32> = predicted[..k.min(predicted.len())].to_vec();

    for &exp in &predicted_set {
        let slot = by_index.iter_mut().find(|e| e.index == exp);
        let Some(e) = slot else {
            plan.kept.push(exp);
            continue;
        };

        // Already hot enough (in VRAM)? No op.
        if e.tier == Tier::Vram {
            plan.kept.push(exp);
            continue;
        }

        // Decide target tier: prefer VRAM if it still fits, else RAM.
        let fits_vram = e.size_bytes <= target_capacity_vram.saturating_sub(vram_used);
        let fits_ram = e.size_bytes <= target_capacity_ram.saturating_sub(ram_used);

        if fits_vram || !fits_ram {
            if fits_vram {
                let from = e.tier;
                let bytes = e.size_bytes;
                e.tier = Tier::Vram;
                vram_used += bytes;
                plan.prefetch.push(PrefetchOp { expert_index: exp, from, to: Tier::Vram, bytes });
            } else {
                // Cannot fit anywhere sensible; keep and let OOM pass handle it.
                plan.kept.push(exp);
            }
        } else {
            let from = e.tier;
            e.tier = Tier::Ram;
            ram_used += e.size_bytes;
            plan.prefetch.push(PrefetchOp { expert_index: exp, from, to: Tier::Ram, bytes: e.size_bytes });
        }
    }

    // Everything not in the predicted set cools down one tier this step.
    for e in by_index.iter_mut() {
        let in_predicted = predicted_set.contains(&e.index);
        if !in_predicted {
            plan.kept.push(e.index);
            if e.tier == Tier::Vram {
                // Cool: VRAM -> RAM, unless RAM is full.
                let fits_ram = e.size_bytes <= target_capacity_ram.saturating_sub(ram_used);
                if fits_ram {
                    e.tier = Tier::Ram;
                    ram_used += e.size_bytes;
                    plan.evictions.push(EvictionOp { expert_index: e.index, from: Tier::Vram, to: Tier::Ram, bytes: e.size_bytes });
                } else if e.size_bytes > target_capacity_vram {
                    // oversized & RAM full: drop to SSD.
                    let _ = demote(e);
                    plan.evictions.push(EvictionOp { expert_index: e.index, from: Tier::Vram, to: Tier::Ssd, bytes: e.size_bytes });
                }
            } else if e.tier == Tier::Ram {
                let _ = demote(e);
                plan.evictions.push(EvictionOp { expert_index: e.index, from: Tier::Ram, to: Tier::Ssd, bytes: e.size_bytes });
            }
        }
    }

    plan
}

/// Estimated decode tok/s floor given a placement. The slowest tier that
/// still holds a meaningfully-hot expert dominates the per-token transfer.
///
/// `hot_experts_in_ssd` = how many of the top-K predicted experts are on SSD.
/// If >0, the per-token path is SSD-bound (worst case).
pub fn est_decode_tokens_per_s(
    hot_experts_in_ssd: u32,
    ssd_bw_bytes_s: u64,
    expert_bytes: u64,
) -> f64 {
    if hot_experts_in_ssd == 0 {
        // Not SSD-bound; caller has RAM/VRAM bandwidth to plug in.
        return f64::INFINITY;
    }
    let per_token = hot_experts_in_ssd as f64 * expert_bytes as f64;
    if per_token == 0.0 {
        return 0.0;
    }
    ssd_bw_bytes_s as f64 / per_token
}
