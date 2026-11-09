// SPDX-License-Identifier: Apache-2.0
//! What this machine and this build can compute on, for `liam devices`.
//!
//! Every function here only reads: it counts cores, opens a Metal device and
//! drops it, or asks llama.cpp to list its backend devices.

use crate::llm::DevicePreference;

/// The sysctl naming the performance cores of an Apple Silicon chip.
#[cfg(target_os = "macos")]
const PERFORMANCE_CORES_SYSCTL: &std::ffi::CStr = c"hw.perflevel0.logicalcpu";

/// The sysctl naming the efficiency cores of an Apple Silicon chip.
#[cfg(target_os = "macos")]
const EFFICIENCY_CORES_SYSCTL: &std::ffi::CStr = c"hw.perflevel1.logicalcpu";

/// Whether this binary was built with CUDA, which only the `cuda` feature does.
pub const CUDA_BUILT: bool = cfg!(feature = "cuda");

/// CPU core counts. The split is known on Apple Silicon only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreCounts {
    pub performance: Option<usize>,
    pub efficiency: Option<usize>,
    pub logical: usize,
}

/// One device in llama.cpp's backend registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDevice {
    pub name: String,
    pub description: String,
    pub backend: String,
    pub memory_total_bytes: usize,
}

#[cfg(target_os = "macos")]
fn sysctl_count(name: &std::ffi::CStr) -> Option<usize> {
    let mut count: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    // SAFETY: `count` and `size` are valid for the sysctl to write, and `size`
    // is the byte length of `count`.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut count).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    usize::try_from(count)
        .ok()
        .filter(|n| status == 0 && *n > 0)
}

/// The performance cores of an Apple Silicon chip, `None` elsewhere.
pub fn performance_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        sysctl_count(PERFORMANCE_CORES_SYSCTL)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// The core counts of this machine.
pub fn core_counts() -> CoreCounts {
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    #[cfg(target_os = "macos")]
    let efficiency = sysctl_count(EFFICIENCY_CORES_SYSCTL);
    #[cfg(not(target_os = "macos"))]
    let efficiency = None;
    CoreCounts {
        performance: performance_cores(),
        efficiency,
        logical,
    }
}

/// Whether a Metal device opens through candle.
pub fn metal_usable() -> bool {
    #[cfg(target_os = "macos")]
    {
        candle_core::Device::new_metal(0).is_ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// The devices llama.cpp reports, or `None` when it cannot start.
pub fn llama_backend_devices() -> Option<Vec<BackendDevice>> {
    crate::llama::shared_backend().ok()?;
    Some(
        llama_cpp_2::list_llama_ggml_backend_devices()
            .into_iter()
            .map(|device| BackendDevice {
                name: device.name,
                description: device.description,
                backend: device.backend,
                memory_total_bytes: device.memory_total,
            })
            .collect(),
    )
}

/// The loaders an `embedder.device` or `llm.device` value reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Candle,
    LlamaEmbedder,
    LlamaLlm,
}

impl Engine {
    /// The config key that names this engine's device, for error messages.
    pub fn key(self) -> &'static str {
        match self {
            Self::Candle | Self::LlamaEmbedder => "embedder.device",
            Self::LlamaLlm => "llm.device",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::LlamaEmbedder | Self::LlamaLlm => "llama.cpp",
        }
    }

    /// Only candle has a CUDA path, behind the `cuda` feature.
    fn has_cuda_path(self) -> bool {
        self == Self::Candle
    }
}

/// What the machine and the build offer, behind a trait so tests can fake it.
pub trait HardwareFacts {
    fn is_macos(&self) -> bool;
    fn metal_usable(&self) -> bool;
    fn cuda_built(&self) -> bool;
}

/// The facts of the running machine.
pub struct SystemFacts;

impl HardwareFacts for SystemFacts {
    fn is_macos(&self) -> bool {
        cfg!(target_os = "macos")
    }
    fn metal_usable(&self) -> bool {
        metal_usable()
    }
    fn cuda_built(&self) -> bool {
        CUDA_BUILT
    }
}

/// A device a loader can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedDevice {
    Metal,
    Cuda,
    Cpu,
}

impl ResolvedDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

/// The device a loader uses and why, when the answer is not obvious.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceResolution {
    pub device: ResolvedDevice,
    pub note: Option<String>,
}

/// The one place a device setting becomes a device. The loaders call it to pick
/// their backend and `liam devices` calls it to report, so the two cannot differ.
/// An explicit device the engine cannot run is an error, never a quiet CPU run.
pub fn resolve_device(
    engine: Engine,
    preference: DevicePreference,
    facts: &dyn HardwareFacts,
) -> Result<DeviceResolution, String> {
    let key = engine.key();
    let ok = |device, note| Ok(DeviceResolution { device, note });
    match preference {
        DevicePreference::Cpu => ok(ResolvedDevice::Cpu, None),
        DevicePreference::Metal if !facts.is_macos() => {
            Err(format!("{key} = \"metal\" is only available on macOS"))
        }
        DevicePreference::Metal if engine == Engine::Candle && !facts.metal_usable() => Err(
            format!("{key} = \"metal\" found no Metal device for candle"),
        ),
        DevicePreference::Metal => ok(ResolvedDevice::Metal, None),
        DevicePreference::Cuda if !(engine.has_cuda_path() && facts.cuda_built()) => Err(format!(
            "{key} = \"cuda\" is not built into {} in this build",
            engine.name()
        )),
        DevicePreference::Cuda => ok(ResolvedDevice::Cuda, None),
        DevicePreference::Auto if facts.is_macos() && facts.metal_usable() => {
            ok(ResolvedDevice::Metal, None)
        }
        DevicePreference::Auto if facts.is_macos() => ok(
            ResolvedDevice::Cpu,
            Some("no Metal device opened, so auto falls back to the CPU".to_owned()),
        ),
        DevicePreference::Auto => ok(ResolvedDevice::Cpu, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_counts_report_at_least_one_logical_core() {
        // Act
        let cores = core_counts();

        // Assert
        assert!(cores.logical >= 1);
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(cores.performance.is_some());
        }
    }

    struct Fake {
        macos: bool,
        metal: bool,
        cuda: bool,
    }

    impl HardwareFacts for Fake {
        fn is_macos(&self) -> bool {
            self.macos
        }
        fn metal_usable(&self) -> bool {
            self.metal
        }
        fn cuda_built(&self) -> bool {
            self.cuda
        }
    }

    const MAC: Fake = Fake {
        macos: true,
        metal: true,
        cuda: false,
    };
    const LINUX: Fake = Fake {
        macos: false,
        metal: false,
        cuda: false,
    };

    fn device(
        engine: Engine,
        pref: DevicePreference,
        facts: &Fake,
    ) -> Result<ResolvedDevice, String> {
        resolve_device(engine, pref, facts).map(|r| r.device)
    }

    #[test]
    fn auto_is_metal_on_a_mac_with_metal_and_the_cpu_otherwise_for_every_engine() {
        for engine in [Engine::Candle, Engine::LlamaEmbedder, Engine::LlamaLlm] {
            // Act and assert
            assert_eq!(
                device(engine, DevicePreference::Auto, &MAC),
                Ok(ResolvedDevice::Metal)
            );
            assert_eq!(
                device(engine, DevicePreference::Auto, &LINUX),
                Ok(ResolvedDevice::Cpu)
            );
            let no_metal = Fake {
                metal: false,
                ..MAC
            };
            let fallback = resolve_device(engine, DevicePreference::Auto, &no_metal).unwrap();
            assert_eq!(fallback.device, ResolvedDevice::Cpu);
            assert!(fallback.note.unwrap().contains("Metal"));
        }
    }

    #[test]
    fn metal_is_refused_off_macos_for_every_engine() {
        for engine in [Engine::Candle, Engine::LlamaEmbedder, Engine::LlamaLlm] {
            // Act
            let error = device(engine, DevicePreference::Metal, &LINUX).unwrap_err();

            // Assert
            assert!(error.contains("only available on macOS"), "{error}");
            assert!(error.contains(engine.key()), "{error}");
        }
    }

    #[test]
    fn cuda_is_refused_for_llama_cpp_always_and_for_candle_without_the_feature() {
        // Arrange
        let cuda_host = Fake {
            cuda: true,
            ..LINUX
        };

        // Act and assert
        for engine in [Engine::LlamaEmbedder, Engine::LlamaLlm] {
            let error = device(engine, DevicePreference::Cuda, &cuda_host).unwrap_err();
            assert!(error.contains("not built into llama.cpp"), "{error}");
        }
        assert_eq!(
            device(Engine::Candle, DevicePreference::Cuda, &cuda_host),
            Ok(ResolvedDevice::Cuda)
        );
        assert!(device(Engine::Candle, DevicePreference::Cuda, &LINUX).is_err());
    }

    #[test]
    fn cpu_and_candle_metal_follow_the_probe() {
        // Arrange
        let no_metal = Fake {
            metal: false,
            ..MAC
        };

        // Act and assert
        assert_eq!(
            device(Engine::LlamaLlm, DevicePreference::Cpu, &LINUX),
            Ok(ResolvedDevice::Cpu)
        );
        assert_eq!(
            device(Engine::Candle, DevicePreference::Metal, &MAC),
            Ok(ResolvedDevice::Metal)
        );
        assert!(device(Engine::Candle, DevicePreference::Metal, &no_metal).is_err());
        assert_eq!(
            device(Engine::LlamaLlm, DevicePreference::Metal, &MAC),
            Ok(ResolvedDevice::Metal)
        );
    }
}
