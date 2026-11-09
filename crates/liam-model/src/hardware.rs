// SPDX-License-Identifier: Apache-2.0
//! What this machine and this build can compute on, for `liam devices`.
//!
//! Every function here only reads: it counts cores, opens a Metal device and
//! drops it, or asks llama.cpp to list its backend devices.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_counts_report_at_least_one_logical_core() {
        // Act
        let cores = core_counts();

        // Assert
        assert!(cores.logical >= 1);
        assert_eq!(cores.performance.is_some(), cfg!(target_os = "macos"));
    }
}
