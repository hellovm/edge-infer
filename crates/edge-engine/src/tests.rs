//! Unit tests for the pure core: three-tier allocation + prefetch planning.
//! These run with no hardware / no GPU.

#[cfg(test)]
mod tests {
    use crate::build_run_plan;
    use edge_detect::{NvidiaProbe, NpuProbe, OsOverhead, ProbeInput};
    use edge_prefetch::{est_decode_tokens_per_s, plan as prefetch_plan};
    use edge_tiers::{allocate, Budget, ExpertPlacement, ModelParams, Tier};

    fn probe_16gb() -> ProbeInput {
        ProbeInput {
            nvidia: Some(NvidiaProbe {
                name: "RTX 4070 Laptop".into(),
                vram_bytes: 8u64 << 30,
                bandwidth_bytes_per_s: 288u64 << 30,
            }),
            npu: Some(NpuProbe {
                name: "Intel AI Boost".into(),
                tops: 47,
                memory_budget_bytes: 4u64 << 30,
            }),
            total_ram_bytes: 16u64 << 30,
            os_overhead: OsOverhead {
                idle_baseline_bytes: 5u64 << 30,
                igpu_dvmt_bytes: (512u32 as u64) << 20,
                hidden_bytes: 1u64 << 30,
            },
            ..Default::default()
        }
    }

    /// 35B-A3B class: 256 experts, 3B active, ~1.5GB resident int4.
    fn model_35b() -> ModelParams {
        ModelParams {
            resident_bytes: 1_500_000_000,
            kv_bytes_per_token: 80_000, // 80KB/token at fp16
            num_experts: 256,
            expert_size_bytes: (3_000_000_000u64 / 256), // ~11.7MB each int4
        }
    }

    #[test]
    fn routes_large_moe_to_nvidia() {
        let p = probe_16gb();
        let m = model_35b();
        let experts: Vec<ExpertPlacement> = (0..256)
            .map(|i| ExpertPlacement { index: i as u32, heat: (256 - i) as u64, size_bytes: m.expert_size_bytes, tier: Tier::Ssd })
            .collect();
        let plan = build_run_plan(&p, &m, &experts, &[0, 1, 2, 3, 4, 5, 6, 7, 8], 8192, 8_000_000_000).unwrap();
        assert_eq!(plan.target, edge_detect::DeviceClass::NvidiaGpu);
    }

    #[test]
    fn routes_small_model_to_npu() {
        let p = probe_16gb();
        // A 4B dense-ish model: active below threshold -> NPU.
        let m = ModelParams {
            resident_bytes: 200_000_000,
            kv_bytes_per_token: 40_000,
            num_experts: 8,
            expert_size_bytes: 50_000_000,
        };
        let experts: Vec<ExpertPlacement> = (0..8).map(|i| ExpertPlacement { index: i as u32, heat: (8 - i) as u64, size_bytes: m.expert_size_bytes, tier: Tier::Ssd }).collect();
        let plan = build_run_plan(&p, &m, &experts, &[0,1,2], 4096, 8_000_000_000).unwrap();
        assert_eq!(plan.target, edge_detect::DeviceClass::IntelNpu);
    }

    #[test]
    fn allocation_fills_vram_with_hottest_experts() {
        let m = ModelParams {
            resident_bytes: 1_500_000_000,
            kv_bytes_per_token: 80_000,
            num_experts: 256,
            expert_size_bytes: 11_700_000,
        };
        let budget = Budget {
            vram_free_bytes: 3u64 << 30,
            ram_free_bytes: 1u64 << 30,
            ssd_free_bytes: u64::MAX,
        };
        let experts: Vec<ExpertPlacement> = (0..256)
            .map(|i| ExpertPlacement { index: i as u32, heat: (256 - i) as u64, size_bytes: 11_700_000, tier: Tier::Ssd })
            .collect();
        let alloc = allocate(&experts, &m, &budget, 8192).unwrap();
        // Hotter experts must not be in a slower tier than colder ones.
        let tiers: Vec<Tier> = alloc.experts.iter().map(|e| e.tier).collect();
        let vram_count = tiers.iter().filter(|t| **t == Tier::Vram).count();
        let ram_count = tiers.iter().filter(|t| **t == Tier::Ram).count();
        let ssd_count = tiers.iter().filter(|t| **t == Tier::Ssd).count();
        assert!(vram_count > 0, "some experts should be in VRAM");
        assert_eq!(vram_count + ram_count + ssd_count, 256);
        // index 0 (hottest) must be in VRAM
        assert_eq!(tiers[0], Tier::Vram, "hottest expert must be in VRAM");
        // coldest (index 255) should be in the slowest tier it can be in
        assert!(matches!(tiers[255], Tier::Ssd | Tier::Ram));
    }

    #[test]
    fn resident_overflow_is_an_error() {
        let m = ModelParams {
            resident_bytes: 10u64 << 30, // bigger than any VRAM here
            kv_bytes_per_token: 80_000,
            num_experts: 256,
            expert_size_bytes: 11_700_000,
        };
        let budget = Budget { vram_free_bytes: 6u64 << 30, ram_free_bytes: 6u64 << 30, ssd_free_bytes: u64::MAX };
        let experts: Vec<ExpertPlacement> = (0..4).map(|i| ExpertPlacement { index: i as u32, heat: (4 - i) as u64, size_bytes: 11_700_000, tier: Tier::Ssd }).collect();
        assert!(allocate(&experts, &m, &budget, 1024).is_err());
    }

    #[test]
    fn prefetch_pulls_predicted_experts_up_a_tier() {
        let mut cur: Vec<ExpertPlacement> = vec![
            ExpertPlacement { index: 0, heat: 10, size_bytes: 1000, tier: Tier::Ssd },
            ExpertPlacement { index: 1, heat: 9, size_bytes: 1000, tier: Tier::Ssd },
        ];
        let plan = prefetch_plan(&mut cur, &[0, 1], 100_000, 100_000, 2);
        // Both predicted experts should be moved into a faster tier.
        let moved: Vec<u32> = plan.prefetch.iter().map(|o| o.expert_index).collect();
        assert!(moved.contains(&0) && moved.contains(&1), "predicted experts must be prefetched, got {moved:?}");
        // and they actually moved to a faster tier
        assert!(matches!(cur[0].tier, Tier::Vram | Tier::Ram));
    }

    #[test]
    fn ssd_bound_decode_floor_math() {
        // 1 hot expert on SSD, 11.7MB, at 5GB/s SSD -> ~427 tok/s ceiling
        // (the transfer is tiny relative to SSD bandwidth).
        let v = est_decode_tokens_per_s(1, 5u64 << 30, 11_700_000);
        assert!(v > 400.0, "got {v}");
        // 0 hot on SSD -> not SSD-bound
        assert!(est_decode_tokens_per_s(0, 5u64 << 30, 11_700_000).is_infinite());
    }

    #[test]
    fn no_mvp_support_when_only_cpu() {
        let p = ProbeInput {
            total_ram_bytes: 16u64 << 30,
            os_overhead: OsOverhead { idle_baseline_bytes: 5u64 << 30, igpu_dvmt_bytes: 0, hidden_bytes: 1u64 << 30 },
            cpu: Some(edge_detect::CpuProbe { name: "cpu".into(), cores: 8, has_amx: true, has_avx512: true }),
            ..Default::default()
        };
        let m = model_35b();
        let experts: Vec<ExpertPlacement> = (0..256).map(|i| ExpertPlacement { index: i as u32, heat: (256 - i) as u64, size_bytes: m.expert_size_bytes, tier: Tier::Ssd }).collect();
        let res = build_run_plan(&p, &m, &experts, &[0], 8192, 8_000_000_000);
        assert!(res.is_err(), "CPU-only must be rejected in MVP");
    }
}
