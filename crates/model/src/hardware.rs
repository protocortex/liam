// SPDX-License-Identifier: Apache-2.0
//! What this machine and this build can compute on, for `liam devices`.
//!
//! Every function here only reads: it counts cores or asks llama.cpp whether it
//! found a GPU and to list its backend devices.

use crate::llm::DevicePreference;

/// The sysctl naming the performance cores of an Apple Silicon chip.
#[cfg(target_os = "macos")]
const PERFORMANCE_CORES_SYSCTL: &std::ffi::CStr = c"hw.perflevel0.logicalcpu";

/// The sysctl naming the efficiency cores of an Apple Silicon chip.
#[cfg(target_os = "macos")]
const EFFICIENCY_CORES_SYSCTL: &std::ffi::CStr = c"hw.perflevel1.logicalcpu";

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

/// Whether llama.cpp found a Metal device it can offload to.
pub fn metal_usable() -> bool {
    #[cfg(target_os = "macos")]
    {
        crate::llama::shared_backend().is_ok_and(|backend| backend.supports_gpu_offload())
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

/// The config key that names the device, for error messages.
const DEVICE_KEY: &str = "llm.device";

/// What the machine and the build offer, behind a trait so tests can fake it.
pub trait HardwareFacts {
    fn is_macos(&self) -> bool;
    fn metal_usable(&self) -> bool;
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
}

/// A device a loader can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedDevice {
    Metal,
    Cpu,
}

impl ResolvedDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
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

/// The one place a device setting becomes a device. The loader calls it to pick
/// its backend and `liam devices` calls it to report, so the two cannot differ.
/// An explicit device llama.cpp cannot run is an error, never a quiet CPU run.
pub fn resolve_device(
    preference: DevicePreference,
    facts: &dyn HardwareFacts,
) -> Result<DeviceResolution, String> {
    let ok = |device, note| Ok(DeviceResolution { device, note });
    match preference {
        DevicePreference::Cpu => ok(ResolvedDevice::Cpu, None),
        DevicePreference::Metal if !facts.is_macos() => Err(format!(
            "{DEVICE_KEY} = \"metal\" is only available on macOS"
        )),
        DevicePreference::Metal => ok(ResolvedDevice::Metal, None),
        DevicePreference::Cuda => Err(format!(
            "{DEVICE_KEY} = \"cuda\" is not built into llama.cpp in this build"
        )),
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
    fn auto_on_this_machine_picks_cpu_unless_metal_is_usable() {
        // Arrange
        let metal = metal_usable();

        // Act
        let resolved = resolve_device(DevicePreference::Auto, &SystemFacts).unwrap();

        // Assert
        let expected = if cfg!(target_os = "macos") && metal {
            ResolvedDevice::Metal
        } else {
            ResolvedDevice::Cpu
        };
        assert_eq!(resolved.device, expected);
        if !cfg!(target_os = "macos") {
            assert!(!metal);
        }
    }

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
    }

    impl HardwareFacts for Fake {
        fn is_macos(&self) -> bool {
            self.macos
        }
        fn metal_usable(&self) -> bool {
            self.metal
        }
    }

    const MAC: Fake = Fake {
        macos: true,
        metal: true,
    };
    const LINUX: Fake = Fake {
        macos: false,
        metal: false,
    };

    fn device(pref: DevicePreference, facts: &Fake) -> Result<ResolvedDevice, String> {
        resolve_device(pref, facts).map(|r| r.device)
    }

    #[test]
    fn auto_is_metal_on_a_mac_with_metal_and_the_cpu_otherwise() {
        // Arrange
        let no_metal = Fake {
            metal: false,
            ..MAC
        };

        // Act
        let fallback = resolve_device(DevicePreference::Auto, &no_metal).unwrap();

        // Assert
        assert_eq!(
            device(DevicePreference::Auto, &MAC),
            Ok(ResolvedDevice::Metal)
        );
        assert_eq!(
            device(DevicePreference::Auto, &LINUX),
            Ok(ResolvedDevice::Cpu)
        );
        assert_eq!(fallback.device, ResolvedDevice::Cpu);
        assert!(fallback.note.unwrap().contains("Metal"));
    }

    #[test]
    fn metal_is_refused_off_macos() {
        // Act
        let error = device(DevicePreference::Metal, &LINUX).unwrap_err();

        // Assert
        assert!(error.contains("only available on macOS"), "{error}");
        assert!(error.contains(DEVICE_KEY), "{error}");
    }

    #[test]
    fn cuda_is_always_refused() {
        for facts in [&MAC, &LINUX] {
            // Act
            let error = device(DevicePreference::Cuda, facts).unwrap_err();

            // Assert
            assert!(error.contains("not built into llama.cpp"), "{error}");
        }
    }

    #[test]
    fn cpu_and_metal_on_a_mac_are_taken_as_asked() {
        // Act and assert
        assert_eq!(
            device(DevicePreference::Cpu, &LINUX),
            Ok(ResolvedDevice::Cpu)
        );
        assert_eq!(
            device(DevicePreference::Metal, &MAC),
            Ok(ResolvedDevice::Metal)
        );
    }
}
