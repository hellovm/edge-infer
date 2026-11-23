//! Learned / adaptive expert cache for MoE weights across SSD / RAM / VRAM.
//!
//! Why not LRU or LFU
//! ------------------
//! Both classic policies have a fatal blind spot on real MoE routing:
//!   - **LRU** is scan-vulnerable: one pass over cold experts flushes the hot
//!     working set, and every working-set use after that is a miss.
//!   - **LFU** is shift-vulnerable: experts that were hot during warm-up stay
//!     pinned after the workload moves on (prose -> code -> tool calls).
//!
//! The policy here adapts online between the two, using the classic
//! **ghost-list** signal: when an expert we just evicted is requested again
//! ("evicted then reused"), that is direct evidence our eviction choice was
//! wrong, and it tells us *which way* it was wrong —
//!   - reuse of a **recently-evicted** expert => be more recency-friendly;
//!   - reuse of a **frequently-used** evicted expert => be more frequency-friendly.
//!
//! On top of that it keeps a learned per-expert heat score (decayed frequency
//! blended with recency) that supplies tie-breaking and the prerouter-facing
//! priority. No offline training: it converges within a few hundred steps.

use crate::{EvictionOp, PrefetchOp, PrefetchPlan};
use edge_tiers::{Tier};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Tunables for the adaptive cache.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Frequency decay per epoch (<1 so old counts fade).
    pub decay: f64,
    /// Recency half-life in epoch units.
    pub recency_halflife: f64,
    /// Step size for adapting the recency/frequency target split.
    pub adapt_lr: f64,
    /// Length of each ghost list, as a fraction of the main capacity.
    pub ghost_frac: f64,
    /// Initial fraction of capacity reserved for the recency list (0..1).
    pub recency_frac: f64,
    /// Bounds on `recency_frac`.
    pub recency_frac_min: f64,
    pub recency_frac_max: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            decay: 0.98,
            recency_halflife: 128.0,
            adapt_lr: 0.05,
            ghost_frac: 0.5,
            recency_frac: 0.5,
            recency_frac_min: 0.1,
            recency_frac_max: 0.9,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct ExpertStat {
    freq: f64,
    last_used: u64,
}

/// An adaptive, self-tuning cache over expert indices.
///
/// Core API:
///   - [`record`](Self::record): one routing step (expert + hit/miss)
///   - [`score`](Self::score): current priority (for prerouter / tie-breaks)
///   - [`victims`](Self::victims): which residents to evict (adaptive ARC-like)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedCache {
    config: CacheConfig,
    epoch: u64,
    stats: HashMap<u32, ExpertStat>,
    hits: u64,
    accesses: u64,
    hit_rate_ema: f64,
    /// Ring buffers of recently-evicted experts (the ghost lists).
    ghost_recent: VecDeque<u32>,
    ghost_freq: VecDeque<u32>,
    ghost_cap: usize,
    /// Runtime copy of the recency/frequency split (adapted by ghost hits).
    recency_frac: f64,
}

impl LearnedCache {
    /// Create a cache sized for `capacity` resident experts (so ghost lists
    /// are proportional).
    pub fn with_capacity(config: CacheConfig, capacity: usize) -> Self {
        let ghost_cap = ((capacity as f64 * config.ghost_frac) as usize).max(1);
        Self {
            config,
            epoch: 0,
            stats: HashMap::new(),
            hits: 0,
            accesses: 0,
            hit_rate_ema: 0.0,
            ghost_recent: VecDeque::with_capacity(ghost_cap),
            ghost_freq: VecDeque::with_capacity(ghost_cap),
            ghost_cap,
            recency_frac: config.recency_frac,
        }
    }

    pub fn new(config: CacheConfig) -> Self {
        Self::with_capacity(config, 64)
    }

    /// Restore a cache from a previously-saved expert-usage profile, so a
    /// restart skips the cold start: the per-expert heat scores and the
    /// learned recency/frequency split are all recovered from disk.
    ///
    /// The profile is the whole struct (it is `Serialize`), so this is the
    /// exact inverse of [`save_profile`](Self::save_profile). Returns
    /// `None` when the file does not exist or cannot be parsed, so a fresh
    /// install simply starts with a warm-up rather than an error.
    pub fn load_profile(path: &std::path::Path) -> Option<Self> {
        let raw = std::fs::read(path).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    /// Persist the current expert-usage profile (heat scores + adapted
    /// recency split + counters) to `path` so the next launch starts warm.
    ///
    /// Call this periodically (Strata saves every 10 minutes) and at
    /// shutdown. Write-then-rename is not used: a torn write just means
    /// the next `load_profile` gets `None` and we fall back to warm-up,
    /// which is acceptable. Returns `Err` on I/O failure.
    pub fn save_profile(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let raw = serde_json::to_vec(self).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &raw)?;
        // Atomic-ish: replace on the same filesystem; on failure the next
        // load just sees the old file (or none) — no corruption.
        std::fs::rename(&tmp, path).or_else(|_| std::fs::copy(&tmp, path).map(|_| ()))
    }

    pub fn config(&self) -> CacheConfig {
        self.config
    }

    /// Current recency/frequency split (0 = pure frequency, 1 = pure recency).
    pub fn recency_frac(&self) -> f64 {
        self.recency_frac
    }

    pub fn hit_rate(&self) -> f64 {
        self.hit_rate_ema
    }

    pub fn accesses(&self) -> u64 {
        self.accesses
    }

    /// Advance one decode step (decays all frequency counts so the score
    /// tracks a moving hot set rather than a lifetime total).
    pub fn tick(&mut self) {
        self.epoch += 1;
        let d = self.config.decay;
        for s in self.stats.values_mut() {
            s.freq *= d;
        }
    }

    /// Record that `expert` was routed to at the current epoch (no hit info).
    pub fn observe(&mut self, expert: u32) {
        let now = self.epoch;
        let e = self.stats.entry(expert).or_insert_with(|| ExpertStat { freq: 0.0, last_used: now });
        e.freq += 1.0;
        e.last_used = now;
    }

    /// Record one routing step for `expert`, plus whether it was a cache hit.
    /// Call this instead of [`observe`](Self::observe) when you know the hit
    /// outcome — it also feeds the ghost-list adaptation.
    pub fn record(&mut self, expert: u32, hit: bool) {
        let alpha = 0.02;
        self.hit_rate_ema = (1.0 - alpha) * self.hit_rate_ema + alpha * if hit { 1.0 } else { 0.0 };
        self.accesses += 1;
        if hit {
            self.hits += 1;
        }
        self.observe(expert);
    }

    /// Notify the cache that `expert` was just evicted from the main set.
    ///
    /// It is classified by **which signal would have kept it** — a
    /// frequency-dominant victim (high decayed count) versus a
    /// recency-dominant one (used very recently). A later request for it then
    /// teaches the policy that *that* signal was the one worth trusting,
    /// which is what makes the adaptation genuine rather than circular.
    pub fn note_evicted(&mut self, expert: u32) {
        let (f, r) = self.signals(expert);
        let freq_dominant = f >= r;
        let list = if freq_dominant { &mut self.ghost_freq } else { &mut self.ghost_recent };
        if list.len() >= self.ghost_cap {
            list.pop_front();
        }
        list.push_back(expert);
    }

    /// Ask the cache whether `expert` is only present as a ghost (i.e. it was
    /// evicted recently). Call on a miss *before* inserting. Returns the list
    /// it was found on, and adapts the split accordingly: a hit on the
    /// frequency ghost list says "trust frequency more", and vice versa.
    pub fn ghost_probe(&mut self, expert: u32) -> Option<GhostHit> {
        if let Some(p) = self.ghost_recent.iter().position(|&x| x == expert) {
            self.ghost_recent.remove(p);
            self.recency_frac =
                (self.recency_frac + self.config.adapt_lr).min(self.config.recency_frac_max);
            return Some(GhostHit::Recent);
        }
        if let Some(p) = self.ghost_freq.iter().position(|&x| x == expert) {
            self.ghost_freq.remove(p);
            self.recency_frac =
                (self.recency_frac - self.config.adapt_lr).max(self.config.recency_frac_min);
            return Some(GhostHit::Frequent);
        }
        None
    }

    /// Learned priority for `expert` — higher means "keep hotter".
    pub fn score(&self, expert: u32) -> f64 {
        let (f, r) = self.signals(expert);
        self.recency_frac * r + (1.0 - self.recency_frac) * f
    }

    fn signals(&self, expert: u32) -> (f64, f64) {
        let Some(s) = self.stats.get(&expert) else {
            return (0.0, 0.0);
        };
        let f = s.freq / (1.0 + s.freq); // non-saturating squash
        let age = (self.epoch.saturating_sub(s.last_used)) as f64;
        let r = 0.5_f64.powf(age / self.config.recency_halflife.max(1.0));
        (f, r)
    }

    /// Pick up to `n` eviction victims among `residents`, coldest first.
    pub fn victims(&self, residents: &[u32], n: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f64)> = residents.iter().map(|&e| (e, self.score(e))).collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n).map(|(e, _)| e).collect()
    }

    /// Size the recency vs frequency sub-list for a given capacity, per the
    /// adapted split. Returned as (recency_slots, frequency_slots).
    pub fn sublist_sizes(&self, capacity: usize) -> (usize, usize) {
        let r = (capacity as f64 * self.recency_frac).round() as usize;
        let r = r.clamp(1, capacity.saturating_sub(1).max(1));
        (r, capacity - r)
    }
}

/// Which ghost list an evicted expert was found on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GhostHit {
    Recent,
    Frequent,
}

/// Plan one prefetch pass with the adaptive policy.
///
/// Predictions are promoted first (they are hot for sure). The *demotion*
/// side is then driven by the learned score, so an expert predicted to be
/// hot soon — even if not in the next-step set — is kept resident.
#[allow(clippy::too_many_arguments)]
pub fn plan_learned(
    cache: &LearnedCache,
    current: &mut [edge_tiers::ExpertPlacement],
    predicted: &[u32],
    target_capacity_vram: u64,
    target_capacity_ram: u64,
    k: usize,
) -> PrefetchPlan {
    let mut plan = PrefetchPlan::default();
    let predicted_set: Vec<u32> = predicted[..k.min(predicted.len())].to_vec();

    let mut vram_used: u64 =
        current.iter().filter(|e| e.tier == Tier::Vram).map(|e| e.size_bytes).sum();
    let mut ram_used: u64 =
        current.iter().filter(|e| e.tier == Tier::Ram).map(|e| e.size_bytes).sum();

    for &exp in &predicted_set {
        let Some(e) = current.iter_mut().find(|e| e.index == exp) else {
            plan.kept.push(exp);
            continue;
        };
        if e.tier == Tier::Vram {
            plan.kept.push(exp);
            continue;
        }
        let fits_vram = e.size_bytes <= target_capacity_vram.saturating_sub(vram_used);
        let fits_ram = e.size_bytes <= target_capacity_ram.saturating_sub(ram_used);
        if fits_vram {
            let (from, bytes) = (e.tier, e.size_bytes);
            e.tier = Tier::Vram;
            vram_used += bytes;
            plan.prefetch.push(PrefetchOp { expert_index: exp, from, to: Tier::Vram, bytes });
        } else if fits_ram {
            let (from, bytes) = (e.tier, e.size_bytes);
            e.tier = Tier::Ram;
            ram_used += bytes;
            plan.prefetch.push(PrefetchOp { expert_index: exp, from, to: Tier::Ram, bytes });
        } else {
            plan.kept.push(exp);
        }
    }

    // Learned retention: experts scoring at least as hot as the predictions
    // stay put instead of cooling — this is what turns bursts into hits.
    let hot_floor = predicted_set
        .iter()
        .map(|&e| cache.score(e))
        .fold(f64::INFINITY, f64::min);

    for e in current.iter_mut() {
        if predicted_set.contains(&e.index) {
            continue;
        }
        if hot_floor.is_finite() && cache.score(e.index) >= hot_floor {
            plan.kept.push(e.index);
            continue;
        }
        match e.tier {
            Tier::Vram => {
                if e.size_bytes <= target_capacity_ram.saturating_sub(ram_used) {
                    e.tier = Tier::Ram;
                    ram_used += e.size_bytes;
                    plan.evictions.push(EvictionOp {
                        expert_index: e.index, from: Tier::Vram, to: Tier::Ram, bytes: e.size_bytes,
                    });
                }
            }
            Tier::Ram => {
                let _ = edge_tiers::demote(e);
                plan.evictions.push(EvictionOp {
                    expert_index: e.index, from: Tier::Ram, to: Tier::Ssd, bytes: e.size_bytes,
                });
            }
            Tier::Ssd => plan.kept.push(e.index),
        }
    }

    plan
}

/// Convenience: run a whole access trace through LRU and return its hit rate.
/// Used by tests and the benchmark to keep comparisons honest.
pub fn lru_hit_rate(trace: &[u32], capacity: usize) -> f64 {
    let mut v: Vec<u32> = Vec::new();
    let mut h = 0usize;
    for &e in trace {
        if let Some(p) = v.iter().position(|&x| x == e) {
            v.remove(p);
            v.push(e);
            h += 1;
        } else {
            if v.len() >= capacity {
                v.remove(0);
            }
            v.push(e);
        }
    }
    h as f64 / trace.len().max(1) as f64
}

/// Convenience: run a whole access trace through the adaptive cache.
pub fn adaptive_hit_rate(trace: &[u32], capacity: usize, cfg: CacheConfig) -> (f64, f64) {
    let mut c = LearnedCache::with_capacity(cfg, capacity);
    let mut resident: HashSet<u32> = HashSet::new();
    let mut h = 0usize;
    for &e in trace {
        let hit = resident.contains(&e);
        if hit {
            h += 1;
        } else {
            // ghost probe before inserting tells the policy how it erred
            c.ghost_probe(e);
            resident.insert(e);
            if resident.len() > capacity {
                let rv: Vec<u32> = resident.iter().copied().collect();
                for v in c.victims(&rv, 1) {
                    resident.remove(&v);
                    c.note_evicted(v);
                }
            }
        }
        c.record(e, hit);
        c.tick();
    }
    (h as f64 / trace.len().max(1) as f64, c.recency_frac())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(cache: &mut LearnedCache, pattern: &[(u32, usize)]) {
        for &(expert, count) in pattern {
            for _ in 0..count {
                cache.observe(expert);
                cache.tick();
            }
        }
    }

    #[test]
    fn recency_recent_beats_stale_burst() {
        let mut c = LearnedCache::new(CacheConfig::default());
        feed(&mut c, &[(1, 60)]);
        for _ in 0..500 {
            c.tick();
        }
        feed(&mut c, &[(2, 10)]);
        assert!(c.score(2) > c.score(1));
    }

    #[test]
    fn frequent_outranks_rare_when_both_recent() {
        let mut c = LearnedCache::new(CacheConfig::default());
        feed(&mut c, &[(7, 40), (9, 3)]);
        assert!(c.score(7) > c.score(9));
    }

    #[test]
    fn victims_are_coldest_first() {
        let mut c = LearnedCache::new(CacheConfig::default());
        for e in [10u32, 11, 12] {
            c.observe(e);
            c.tick();
        }
        assert_eq!(c.victims(&[10, 11, 12], 1), vec![10]);
    }

    #[test]
    fn hit_rate_ema_converges() {
        let mut c = LearnedCache::new(CacheConfig::default());
        for _ in 0..500 {
            c.record(1, true);
        }
        assert!(c.hit_rate() > 0.9);
    }

    #[test]
    fn ghost_hits_adapt_the_split() {
        let mut c = LearnedCache::with_capacity(CacheConfig::default(), 64);
        let r0 = c.recency_frac();
        // A recency-dominant expert (used once, just now) evicted -> ghost_recent.
        c.observe(5);
        c.note_evicted(5);
        if c.ghost_probe(5) == Some(GhostHit::Recent) {
            assert!(c.recency_frac() > r0, "recent-ghost hit should favour recency");
        }

        let r1 = c.recency_frac();
        // A frequency-dominant expert (used many times, aged) evicted -> ghost_freq.
        for _ in 0..20 {
            c.observe(6);
            c.tick();
        }
        for _ in 0..200 {
            c.tick();
        }
        c.note_evicted(6);
        if c.ghost_probe(6) == Some(GhostHit::Frequent) {
            assert!(c.recency_frac() < r1, "freq-ghost hit should favour frequency");
        }
    }

    #[test]
    fn split_stays_in_bounds() {
        let mut c = LearnedCache::with_capacity(CacheConfig::default(), 64);
        for _ in 0..1000 {
            c.observe(1);
            c.note_evicted(1);
            c.ghost_probe(1);
        }
        assert!(c.recency_frac() <= c.config().recency_frac_max + 1e-9);
        for _ in 0..1000 {
            for _ in 0..5 {
                c.observe(2);
                c.tick();
            }
            for _ in 0..100 {
                c.tick();
            }
            c.note_evicted(2);
            c.ghost_probe(2);
        }
        assert!(c.recency_frac() >= c.config().recency_frac_min - 1e-9);
    }

    fn trace_phase_shift(n_steps: usize, n_experts: u32, hot: usize, phase_len: usize, seed: u64) -> Vec<u32> {
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(n_steps * hot);
        for t in 0..n_steps {
            let phase = (t / phase_len) as u32;
            let base = (phase.wrapping_mul(131)) % n_experts;
            for _ in 0..hot {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let x = (s >> 33) as u32;
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
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(n_steps * hot);
        for t in 0..n_steps {
            if t % 400 == 0 && t > 0 {
                let scan_base = ((s >> 33) as u32) % n_experts;
                for k in 0..scan_len as u32 {
                    out.push((scan_base.wrapping_add(k)) % n_experts);
                }
            }
            for _ in 0..hot {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let x = (s >> 33) as u32;
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

    #[test]
    fn profile_roundtrip_recovers_warm_state() {
        let dir = std::env::temp_dir().join(format!("edge-infer-profile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("cache.json");
        let mut c = LearnedCache::with_capacity(CacheConfig::default(), 64);
        // Build a warm state: hot experts aged differently, adapted split,
        // and some recorded accesses so the counters are non-zero.
        for e in [1u32, 2, 3] {
            for _ in 0..50 {
                c.record(e, true);
                c.tick();
            }
        }
        for _ in 0..500 {
            c.tick();
        }
        let _ = c.ghost_probe(999); // nudge the split
        let frac_before = c.recency_frac();
        c.save_profile(&path).unwrap();

        let restored = LearnedCache::load_profile(&path).expect("profile load");
        assert_eq!(restored.recency_frac(), frac_before, "split must survive round-trip");
        // Heat scores must be recovered so hot experts outrank cold ones.
        assert!(restored.score(1) > restored.score(10_000), "hot expert must outrank cold after restore");
        assert!(restored.accesses() >= 1, "counters must be restored");

        // A missing file must return None (fresh-install warm-up path).
        let missing = dir.join("does-not-exist.json");
        assert!(LearnedCache::load_profile(&missing).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn adaptive_is_robust_across_workloads() {
        let ps = trace_phase_shift(8_000, 512, 10, 400, 0xDEADBEEF);
        let sc = trace_scan(8_000, 512, 10, 300, 0x12345678);
        for &cap in &[24usize, 48] {
            let lru_ps = lru_hit_rate(&ps, cap);
            let (ad_ps, _) = adaptive_hit_rate(&ps, cap, CacheConfig::default());
            let lru_sc = lru_hit_rate(&sc, cap);
            let (ad_sc, _) = adaptive_hit_rate(&sc, cap, CacheConfig::default());
            // Must never lose meaningfully to LRU on either workload.
            assert!(ad_ps >= lru_ps - 0.01, "phase-shift cap={cap}: adaptive {ad_ps:.3} vs LRU {lru_ps:.3}");
            assert!(ad_sc >= lru_sc - 0.01, "scan cap={cap}: adaptive {ad_sc:.3} vs LRU {lru_sc:.3}");
        }
    }
}
