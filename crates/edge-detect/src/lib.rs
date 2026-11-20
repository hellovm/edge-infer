//! Hardware probing for edge-infer.
//!
//! MVP targets: NVIDIA GPU (CUDA) + Intel NPU (OpenVINO), with iGPU / CPU as
//! reference reports. The probe is device-agnostic: it produces a normalized
//! `HardwareReport` that the upper layers (tier allocator, prefetch engine,
//! model router) consume. No GPU driver calls are made here in this slice;
//! detection is by presence + budget heuristics so the crate stays unit-testable
//! without hardware.

use serde::{Deserialize, Serialize};
use std::fmt;

/// High-level device class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceClass {
    /// NVIDIA discrete GPU, CUDA capable. Primary for large MoE models.
    NvidiaGpu,
    /// Intel discrete / integrated GPU. Second phase.
    IntelGpu,
    /// Intel AI Boost NPU. Small-model tier only.
    IntelNpu,
    /// CPU with AMX / AVX-512. Fallback (not in MVP).
    Cpu,
}

impl DeviceClass {
    /// Does this class support the large-MoE three-tier offload path?
    pub fn supports_large_moe(self) -> bool {
        matches!(self, DeviceClass::NvidiaGpu | DeviceClass::IntelGpu)
    }

    /// Does this class serve the small-model (NPU) tier?
    pub fn supports_small_model(self) -> bool {
        matches!(self, DeviceClass::IntelNpu | DeviceClass::Cpu)
    }
}

/// A single probed device with its budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub class: DeviceClass,
    pub name: String,
    /// Usable capacity in bytes (VRAM for GPU, NPU memory budget, RAM for CPU).
    pub usable_bytes: u64,
    /// Estimated read bandwidth in bytes/sec (drives the tok/s floor calc).
    pub est_bandwidth_bytes_per_s: u64,
    /// Preferred quantization the device handles well.
    pub preferred_quant: QuantKind,
}

/// Quantization precision a device handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantKind {
    Int4,
    Int8,
    Fp16,
}

impl fmt::Display for DeviceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DeviceClass::NvidiaGpu => "NVIDIA-GPU",
            DeviceClass::IntelGpu => "Intel-GPU",
            DeviceClass::IntelNpu => "Intel-NPU",
            DeviceClass::Cpu => "CPU",
        };
        write!(f, "{s}")
    }
}

/// Whole-system memory accounting for a 16GB Windows laptop (from the
/// requirements doc §6). Drives how much RAM is actually available to the
/// model + KV beyond OS overhead.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OsOverhead {
    /// Windows 11 idle baseline.
    pub idle_baseline_bytes: u64,
    /// iGPU DVMT hardware reservation.
    pub igpu_dvmt_bytes: u64,
    /// Hidden overhead (memory compression, SysMain, driver-locked pools).
    pub hidden_bytes: u64,
}

impl OsOverhead {
    pub fn total(self) -> u64 {
        self.idle_baseline_bytes + self.igpu_dvmt_bytes + self.hidden_bytes
    }
}

/// Normalized hardware report consumed by the allocator / router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareReport {
    pub devices: Vec<Device>,
    pub total_ram_bytes: u64,
    pub os_overhead: OsOverhead,
    pub has_nvidia: bool,
    pub has_npu: bool,
    /// Storage read bandwidth (drives the SSD-bound tok/s floor).
    pub ssd_read_bytes_per_s: u64,
    pub ssd_name: String,
}

impl HardwareReport {
    /// RAM actually usable for model weights + KV after OS overhead.
    pub fn usable_ram_bytes(&self) -> u64 {
        self.total_ram_bytes.saturating_sub(self.os_overhead.total())
    }

    /// A device of a given class, if present.
    pub fn find(&self, class: DeviceClass) -> Option<&Device> {
        self.devices.iter().find(|d| d.class == class)
    }

    /// MVP support check: is there at least one N-card or NPU?
    /// (CPU-only machines are not supported in MVP.)
    pub fn mvp_supported(&self) -> bool {
        self.has_nvidia || self.has_npu
    }

    /// Which device class should serve a model of the given active
    /// parameter count? Large MoE (>8B active) -> N-card; small -> NPU.
    pub fn route(&self, active_params: u64, large_threshold: u64) -> Option<DeviceClass> {
        if active_params > large_threshold {
            // Large MoE: N-card primary, iGPU secondary.
            if self.has_nvidia {
                Some(DeviceClass::NvidiaGpu)
            } else {
                self.find(DeviceClass::IntelGpu).map(|_| DeviceClass::IntelGpu)
            }
        } else {
            // Small model: NPU preferred, CPU fallback.
            if self.has_npu {
                Some(DeviceClass::IntelNpu)
            } else {
                self.find(DeviceClass::Cpu).map(|_| DeviceClass::Cpu)
            }
        }
    }
}

/// Build a report from a structured input. Kept pure so the same logic can be
/// unit-tested and later wired to real NVML / OpenVINO probing.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProbeInput {
    pub nvidia: Option<NvidiaProbe>,
    pub intel_gpu: Option<NvidiaProbe>,
    pub npu: Option<NpuProbe>,
    pub total_ram_bytes: u64,
    pub os_overhead: OsOverhead,
    pub cpu: Option<CpuProbe>,
    /// Storage tier. PCIe 5.0 NVMe ~ 12-14 GB/s; PCIe 4.0 ~ 7 GB/s;
    /// SATA SSD ~ 0.55 GB/s. Defaults to a 6 GB/s NVMe estimate.
    pub ssd: Option<SsdProbe>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SsdProbe {
    pub name: String,
    /// Sequential read bandwidth in bytes/sec.
    pub read_bytes_per_s: u64,
    /// Whether the drive is PCIe gen-5 class (>= 10 GB/s).
    #[serde(default)]
    pub pcie5: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct NvidiaProbe {
    pub name: String,
    /// Total VRAM; usable is derived after a desktop-reserve fraction.
    pub vram_bytes: u64,
    /// Bandwidth (e.g. GDDR6 on a 4070 Laptop ~ 288 GB/s).
    pub bandwidth_bytes_per_s: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct NpuProbe {
    pub name: String,
    /// TOPS is informational; NPU budget is a memory cap.
    pub tops: u32,
    /// NPU-accessible memory budget.
    pub memory_budget_bytes: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CpuProbe {
    pub name: String,
    pub cores: u32,
    pub has_amx: bool,
    pub has_avx512: bool,
}

/// Fraction of a discrete GPU's VRAM reserved for the desktop / other apps.
const DESKTOP_VRAM_RESERVE: f64 = 0.25;

pub fn build_report(input: &ProbeInput) -> HardwareReport {
    let mut devices: Vec<Device> = Vec::new();

    if let Some(nv) = &input.nvidia {
        let usable = (nv.vram_bytes as f64 * (1.0 - DESKTOP_VRAM_RESERVE)) as u64;
        devices.push(Device {
            id: "nv0".into(),
            class: DeviceClass::NvidiaGpu,
            name: nv.name.clone(),
            usable_bytes: usable,
            est_bandwidth_bytes_per_s: nv.bandwidth_bytes_per_s,
            preferred_quant: QuantKind::Int4,
        });
    }

    if let Some(g) = &input.intel_gpu {
        devices.push(Device {
            id: "i0".into(),
            class: DeviceClass::IntelGpu,
            name: g.name.clone(),
            usable_bytes: g.vram_bytes,
            est_bandwidth_bytes_per_s: g.bandwidth_bytes_per_s,
            preferred_quant: QuantKind::Int4,
        });
    }

    let mut has_npu = false;
    if let Some(npu) = &input.npu {
        has_npu = true;
        devices.push(Device {
            id: "npu0".into(),
            class: DeviceClass::IntelNpu,
            name: npu.name.clone(),
            usable_bytes: npu.memory_budget_bytes,
            est_bandwidth_bytes_per_s: 0, // NPU bandwidth is shared LPDDR5X, not a direct pull
            preferred_quant: QuantKind::Int4,
        });
    }

    if let Some(c) = &input.cpu {
        let bw = 40u64 * 1024 * 1024 * 1024; // ~DDR5 dual-channel estimate
        devices.push(Device {
            id: "cpu0".into(),
            class: DeviceClass::Cpu,
            name: format!("{} ({}c, AMX:{}, AVX512:{})", c.name, c.cores, c.has_amx, c.has_avx512),
            usable_bytes: 0, // bounded by system RAM, not a fixed budget
            est_bandwidth_bytes_per_s: bw,
            preferred_quant: QuantKind::Int4,
        });
    }

    HardwareReport {
        has_nvidia: input.nvidia.is_some(),
        has_npu,
        devices,
        total_ram_bytes: input.total_ram_bytes,
        os_overhead: input.os_overhead,
        ssd_read_bytes_per_s: input
            .ssd
            .as_ref()
            .map(|s| s.read_bytes_per_s)
            .filter(|&b| b > 0)
            .unwrap_or(6_000_000_000), // default: NVMe PCIe 4.0 estimate
        ssd_name: input
            .ssd
            .as_ref()
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "NVMe SSD (未探测)".into()),
    }
}
