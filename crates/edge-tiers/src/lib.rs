//! Three-tier heat-based allocator for SSD / RAM / VRAM.
//!
//! This is the device-agnostic core of edge-infer. It decides, for a given
//! live VRAM budget and a user-requested context length, where each "expert"
//! and the KV cache live, and how to fall back when VRAM runs out. It is
//! pure logic: no driver calls, fully unit-testable.
//!
//! Tier model (requirements doc §7):
//!   L2 VRAM : resident layer + hottest experts + hot KV
//!   L1 RAM  : next-heat experts + cold KV
//!   L0 SSD  : full int4 checkpoint + coldest experts
//!
//! Experts are ranked by a "heat" score (historical routing frequency,
//! optionally augmented by the prerouter's one-step-ahead prediction). The
//! allocator places top-N in VRAM, N+1..M in RAM, the rest on SSD, where N
//! and M are derived from the live budgets.

use serde::{Deserialize, Serialize};

/// A placement tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    Vram,
    Ram,
    Ssd,
}

impl Tier {
    /// Slower == larger. Used for OOM demotion.
    pub fn slowness(self) -> u8 {
        match self {
            Tier::Vram => 0,
            Tier::Ram => 1,
            Tier::Ssd => 2,
        }
    }
}

/// One routed expert's placement + heat.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExpertPlacement {
    /// Expert index (0..num_experts).
    pub index: u32,
    /// Heat score, higher = used more / predicted sooner.
    pub heat: u64,
    /// Size of this expert's int4 weight slice in bytes.
    pub size_bytes: u64,
    pub tier: Tier,
}

/// The result of one allocation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub experts: Vec<ExpertPlacement>,
    pub vram_used_bytes: u64,
    pub ram_used_bytes: u64,
    /// Context length we actually committed to (may be reduced by OOM fallback).
    pub effective_context: usize,
    /// Whether OOM fallback engaged.
    pub oom_fallback: bool,
}

/// Budgets available for a single model run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Budget {
    /// Live free VRAM (after desktop/other-app reserve, from NVML).
    pub vram_free_bytes: u64,
    /// Usable system RAM for next-heat experts + cold KV.
    pub ram_free_bytes: u64,
    /// SSD is effectively unbounded; kept for completeness.
    pub ssd_free_bytes: u64,
}

/// Per-model constants the allocator needs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelParams {
    /// Resident layer int4 size (shared experts + attention + router +
    /// embedding). Always pinned to VRAM.
    pub resident_bytes: u64,
    /// KV cache bytes per token (at the chosen KV quant precision).
    pub kv_bytes_per_token: u64,
    /// Number of routed experts (e.g. 256 for a 35B-A3B class model).
    pub num_experts: u32,
    /// Expert size is uniform per expert (int4).
    pub expert_size_bytes: u64,
}

impl ModelParams {
    /// Total int4 checkpoint size.
    pub fn full_checkpoint_bytes(&self) -> u64 {
        self.resident_bytes + self.num_experts as u64 * self.expert_size_bytes
    }
}

/// Errors from allocation.
#[derive(Debug, thiserror::Error)]
pub enum AllocError {
    #[error("resident layer does not fit in VRAM ({res}B > {vram}B): need a bigger GPU or drop KV")]
    ResidentOverflow { res: u64, vram: u64 },
}

/// Allocate experts + KV across the three tiers.
///
/// `experts` is the full expert set with heat scores. `context` is the
/// user-requested context length. The allocator:
///   1. pins the resident layer to VRAM;
///   2. locks KV for the requested context (hot KV -> VRAM, overflow -> RAM);
///   3. fills VRAM with the hottest experts, then RAM with the next-heat,
///      the rest stays on SSD;
///   4. on OOM (VRAM short) it demotes experts from VRAM->RAM->SSD and, as a
///      last resort, shortens the effective context.
pub fn allocate(
    experts: &[ExpertPlacement],
    model: &ModelParams,
    budget: &Budget,
    context: usize,
) -> Result<Allocation, AllocError> {
    let vram = budget.vram_free_bytes;

    // 1. Resident layer must fit in VRAM or we cannot run at all.
    if model.resident_bytes > vram {
        return Err(AllocError::ResidentOverflow {
            res: model.resident_bytes,
            vram: vram,
        });
    }

    let mut vram_used = model.resident_bytes;

    // 2. KV cache: hot slice in VRAM, cold overflow in RAM.
    //    We pin as much of the requested context's KV into VRAM as we can,
    //    the remainder spills to RAM (and beyond that to SSD in a full run).
    let kv_total = context as u64 * model.kv_bytes_per_token;
    let kv_in_vram = kv_total.min(vram.saturating_sub(vram_used));
    vram_used += kv_in_vram;
    let mut ram_used = kv_total - kv_in_vram; // cold KV spills to RAM

    // 3. Place experts by heat: VRAM (remaining) -> RAM -> SSD.
    let mut ranked: Vec<usize> = (0..experts.len()).collect();
    ranked.sort_by_key(|&i| std::cmp::Reverse(experts[i].heat));

    let mut out: Vec<ExpertPlacement> = experts
        .iter()
        .map(|e| ExpertPlacement {
            index: e.index,
            heat: e.heat,
            size_bytes: e.size_bytes,
            tier: Tier::Ssd, // default coldest; upgraded below
        })
        .collect();

    let mut oom_fallback = false;
    let mut effective_context = context;

    let mut idx_in_ranked = 0;
    while idx_in_ranked < ranked.len() {
        let i = ranked[idx_in_ranked];
        let size = experts[i].size_bytes;

        // Try VRAM first (but only if it still fits after resident+KV).
        if size <= vram - vram_used {
            vram_used += size;
            out[i] = ExpertPlacement {
                index: experts[i].index,
                heat: experts[i].heat,
                size_bytes: size,
                tier: Tier::Vram,
            };
            idx_in_ranked += 1;
            continue;
        }

        // VRAM is full for this expert.
        if size <= budget.ram_free_bytes - ram_used {
            ram_used += size;
            out[i] = ExpertPlacement {
                index: experts[i].index,
                heat: experts[i].heat,
                size_bytes: size,
                tier: Tier::Ram,
            };
        } else {
            // Falls to SSD (default tier already Ssd).
            out[i] = ExpertPlacement {
                index: experts[i].index,
                heat: experts[i].heat,
                size_bytes: size,
                tier: Tier::Ssd,
            };
        }
        idx_in_ranked += 1;
    }

    // 4. OOM fallback: the cold KV (whatever did not fit in VRAM) must fit in
    //    RAM. If the requested context's cold KV exceeds the RAM budget, shorten
    //    the effective context proportionally and flag the fallback.
    if kv_total > 0 && kv_in_vram < kv_total {
        let cold_kv = kv_total - kv_in_vram;
        if cold_kv > budget.ram_free_bytes {
            let max_cold_tokens =
                (budget.ram_free_bytes as f64 / model.kv_bytes_per_token.max(1) as f64) as usize;
            // Keep the hot part already pinned in VRAM; cap the total context.
            let hot_tokens = (kv_in_vram as f64 / model.kv_bytes_per_token.max(1) as f64) as usize;
            effective_context = hot_tokens + max_cold_tokens;
            if effective_context == 0 {
                effective_context = 1;
            }
            oom_fallback = true;
        }
    }

    Ok(Allocation {
        experts: out,
        vram_used_bytes: vram_used,
        ram_used_bytes: ram_used,
        effective_context,
        oom_fallback,
    })
}

/// Demote one placed expert to the next slower tier. Returns the freed bytes
/// and the source tier, or None if already on SSD.
pub fn demote(expert: &mut ExpertPlacement) -> Option<(Tier, u64)> {
    match expert.tier {
        Tier::Vram => {
            expert.tier = Tier::Ram;
            Some((Tier::Vram, expert.size_bytes))
        }
        Tier::Ram => {
            expert.tier = Tier::Ssd;
            Some((Tier::Ram, expert.size_bytes))
        }
        Tier::Ssd => None,
    }
}
