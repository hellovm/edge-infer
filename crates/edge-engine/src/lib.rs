//! edge-engine: orchestration layer that wires detect -> tiers -> prefetch into
//! a concrete, serializable plan a GUI or OpenAI server can execute.
//!
//! It is deliberately hardware-agnostic: the *plan* (which expert in which
//! tier, what to prefetch, which OOM fallbacks) is pure data. Executing that
//! plan against a real CUDA / SYCL / OpenVINO backend is the next slice.

#[cfg(test)]
mod tests;


use edge_detect::{build_report, Device, DeviceClass, HardwareReport, ProbeInput};
use edge_prefetch::{
    est_decode_tokens_per_s, plan as prefetch_plan, plan_learned, CacheConfig, LearnedCache,
    PrefetchPlan,
};
use edge_tiers::{allocate, Allocation, Budget, ExpertPlacement, ModelParams, Tier};
use serde::{Deserialize, Serialize};

/// End-to-end planning result for one model + context + device set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunPlan {
    /// Which device class serves this model.
    pub target: DeviceClass,
    /// Three-tier allocation.
    pub allocation: Allocation,
    /// Per-step prefetch / eviction plan.
    pub prefetch: PrefetchPlan,
    /// Estimated decode floor (tok/s), accounting for SSD-bound hot experts.
    pub est_decode_tok_s: Option<f64>,
    /// OOM fallback engaged?
    pub oom_fallback: bool,
    /// Human-readable summary lines for the GUI.
    pub summary: Vec<String>,
    /// Learned-cache state, when the plan was built with one.
    #[serde(default)]
    pub cache: Option<CacheReport>,
}

/// Snapshot of the adaptive expert cache, for the GUI / diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheReport {
    /// Current recency/frequency split (0 = pure frequency, 1 = pure recency).
    pub recency_frac: f64,
    /// Observed hit-rate EMA (0..1).
    pub hit_rate: f64,
    /// Total routed-expert lookups seen.
    pub accesses: u64,
    /// Experts currently tracked.
    pub tracked_experts: usize,
}

/// Build a concrete `RunPlan` for a model.
///
/// # Arguments
/// * `probe`  - probed hardware (from `ProbeInput`)
/// * `model`  - per-model constants
/// * `experts`- the full expert set with heat + size
/// * `predicted` - the prerouter's one-step-ahead top-K for the next token
/// * `context` - user-requested context length
pub fn build_run_plan(
    probe: &ProbeInput,
    model: &ModelParams,
    experts: &[ExpertPlacement],
    predicted: &[u32],
    context: usize,
    large_threshold: u64,
) -> Result<RunPlan, String> {
    let report: HardwareReport = build_report(probe);

    if !report.mvp_supported() {
        return Err("no N-card or NPU available (MVP requires at least one)".into());
    }

    // Route by TOTAL model params (checkpoint size), not active params.
    // A 35B-A3B has ~3B *active* but a ~35B *total* checkpoint: the NPU can
    // only serve a few GB, so the whole-model footprint decides. int4 ==
    // 0.5 bytes/param, so params ~= bytes * 2.
    let total_expert_bytes = model.num_experts as u64 * model.expert_size_bytes;
    let total_params_estimate = model
        .resident_bytes
        .saturating_mul(2)
        .saturating_add(total_expert_bytes.saturating_mul(2));

    let target = report.route(total_params_estimate, large_threshold)
        .ok_or_else(|| "no eligible device for this model size".to_string())?;

    // Pick the target device's budget.
    let device: &Device = report.find(target).expect("route found a class");
    let vram_free = device.usable_bytes;
    let ram_free = report.usable_ram_bytes();

    let budget = Budget {
        vram_free_bytes: vram_free,
        ram_free_bytes: ram_free,
        ssd_free_bytes: u64::MAX / 1024, // effectively unbounded
    };

    let allocation =
        allocate(experts, model, &budget, context).map_err(|e| format!("{e}"))?;

    let mut allocation = allocation;
    let prefetch = prefetch_plan(
        &mut allocation.experts,
        predicted,
        allocation.vram_used_bytes,
        allocation.ram_used_bytes,
        model.num_experts as usize,
    );

    // Estimate the decode floor. Prefer SSD-bound (hot experts on SSD); when
    // nothing hot is on SSD, the path is bound by the fastest tier holding the
    // hot experts -- report a bandwidth-derived ceiling instead of "N/A".
    let hot_in_ssd = allocation
        .experts
        .iter()
        .filter(|e| predicted.contains(&e.index) && e.tier == Tier::Ssd)
        .count() as u32;
    let ssd_floor = est_decode_tokens_per_s(
        hot_in_ssd,
        report.ssd_read_bytes_per_s,
        model.expert_size_bytes,
    );

    // Bytes moved per token = hot experts' weights pulled from their tier.
    let hot_count = predicted.len().max(1) as u64;
    let per_token_bytes = hot_count * model.expert_size_bytes;
    // Effective bandwidth = the slowest tier that still holds a hot expert.
    let hot_in_ram = allocation
        .experts
        .iter()
        .filter(|e| predicted.contains(&e.index) && e.tier == Tier::Ram)
        .count();
    let effective_bw = if hot_in_ssd > 0 {
        report.ssd_read_bytes_per_s
    } else if hot_in_ram > 0 {
        40_000_000_000 // DDR5 dual-channel ~40 GB/s
    } else {
        device.est_bandwidth_bytes_per_s.max(1) // VRAM
    };
    let bw_ceiling = if per_token_bytes > 0 {
        Some(effective_bw as f64 / per_token_bytes as f64)
    } else {
        None
    };

    let mut summary = vec![
        format!(
            "路由 -> {}（总参约 {}B，阈值 {}B）",
            target,
            total_params_estimate / 1_000_000_000,
            large_threshold / 1_000_000_000
        ),
        format!(
            "显存可省 {} / 内存可省 {}；已用 显存 {}B、内存 {}B",
            fmt_bytes(vram_free),
            fmt_bytes(ram_free),
            fmt_bytes(allocation.vram_used_bytes),
            fmt_bytes(allocation.ram_used_bytes)
        ),
        format!(
            "上下文 请求 {} -> 生效 {}",
            context, allocation.effective_context
        ),
        format!(
            "存储层：{}（读带宽 {}）",
            report.ssd_name,
            fmt_bw(report.ssd_read_bytes_per_s)
        ),
    ];
    if allocation.oom_fallback {
        summary.push("已启用 OOM 回退：缩短上下文 / 降级部分专家".into());
    }
    if hot_in_ssd > 0 {
        summary.push(format!(
            "有 {hot_in_ssd} 个热专家在 SSD：decode 受硬盘带宽限制，约 {:.1} tok/s（换更快 SSD 或调大显存/内存可缓解）",
            ssd_floor
        ));
    } else {
        summary.push(format!(
            "热专家全部驻留显存/内存，无需读盘 —— 不受硬盘带宽限制，带宽上限约 {} tok/s",
            bw_ceiling.map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into())
        ));
    }
    let est = if hot_in_ssd > 0 { Some(ssd_floor) } else { None };

    let oom_fallback = allocation.oom_fallback;

    Ok(RunPlan {
        target,
        allocation,
        prefetch,
        est_decode_tok_s: est,
        oom_fallback,
        summary,
        cache: None,
    })
}

/// Like [`build_run_plan`], but drives the eviction side with the adaptive
/// expert cache instead of "everything not predicted cools".
///
/// `observed` is the recent routing trace (expert indices in order, the most
/// recent last); it warms the cache so the first planning step already has a
/// learned hot set. Pass an empty slice for a cold start.
#[allow(clippy::too_many_arguments)]
pub fn build_run_plan_learned(
    probe: &ProbeInput,
    model: &ModelParams,
    experts: &[ExpertPlacement],
    predicted: &[u32],
    observed: &[u32],
    context: usize,
    large_threshold: u64,
    cache_cfg: CacheConfig,
) -> Result<RunPlan, String> {
    let mut plan = build_run_plan(probe, model, experts, predicted, context, large_threshold)?;

    // Warm the cache on the recent trace, then re-plan evictions with it.
    let capacity = plan
        .allocation
        .experts
        .iter()
        .filter(|e| e.tier != Tier::Ssd)
        .count()
        .max(1);
    let mut cache = LearnedCache::with_capacity(cache_cfg, capacity);
    for &e in observed {
        cache.record(e, true);
        cache.tick();
    }

    let vram_used = plan.allocation.vram_used_bytes;
    let ram_used = plan.allocation.ram_used_bytes;
    plan.prefetch = plan_learned(
        &cache,
        &mut plan.allocation.experts,
        predicted,
        vram_used,
        ram_used,
        model.num_experts as usize,
    );
    plan.cache = Some(CacheReport {
        recency_frac: cache.recency_frac(),
        hit_rate: cache.hit_rate(),
        accesses: cache.accesses(),
        tracked_experts: capacity,
    });
    plan.summary.push(format!(
        "学习型专家缓存：recency/frequency 自适应比重 {:.2}，已跟踪 {} 个专家",
        cache.recency_frac(),
        capacity
    ));
    Ok(plan)
}

fn fmt_bytes(b: u64) -> String {
    const M: u64 = 1024 * 1024;
    const G: u64 = 1024 * 1024 * 1024;
    if b >= G {
        format!("{:.1}G", b as f64 / G as f64)
    } else if b >= M {
        format!("{:.0}M", b as f64 / M as f64)
    } else {
        format!("{b}B")
    }
}

/// Format a bandwidth in GB/s.
fn fmt_bw(bytes_per_s: u64) -> String {
    format!("{:.1} GB/s", bytes_per_s as f64 / 1_000_000_000.0)
}
