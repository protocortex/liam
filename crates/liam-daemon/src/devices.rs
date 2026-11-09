// SPDX-License-Identifier: Apache-2.0
//! `liam devices`: what this machine and this build can compute on, and what
//! `auto` resolves to for the embedder and the LLM under a config.
//!
//! The probing sits behind `HardwareProbe` so the report and the resolution
//! are tested with fakes, since a test host cannot stand in for every
//! platform.

use std::fmt::Write as _;

use liam_model::hardware::{
    resolve_device, BackendDevice, CoreCounts, Engine, HardwareFacts, SystemFacts,
};
use liam_model::llm::DevicePreference;
use serde::Serialize;

use crate::config::Config;
use crate::models::{LLAMA_PROVIDER, LOCAL_PROVIDER};

const BYTES_PER_MB: usize = 1024 * 1024;
const BACKEND_METAL: &str = "metal";
const BACKEND_CUDA: &str = "cuda";
const BACKEND_CPU: &str = "cpu";
const NOT_REPORTED: &str = "not reported";
const MACOS: &str = "macos";

/// What the machine and the binary can say about themselves.
pub trait HardwareProbe: HardwareFacts {
    fn os(&self) -> String;
    fn arch(&self) -> String;
    fn cores(&self) -> CoreCounts;
    /// `None` when llama.cpp does not report its devices.
    fn llama_devices(&self) -> Option<Vec<BackendDevice>>;
}

/// The probe of the running machine.
pub struct SystemProbe;

impl HardwareFacts for SystemProbe {
    fn is_macos(&self) -> bool {
        SystemFacts.is_macos()
    }
    fn metal_usable(&self) -> bool {
        SystemFacts.metal_usable()
    }
    fn cuda_built(&self) -> bool {
        SystemFacts.cuda_built()
    }
}

impl HardwareProbe for SystemProbe {
    fn os(&self) -> String {
        std::env::consts::OS.to_owned()
    }
    fn arch(&self) -> String {
        std::env::consts::ARCH.to_owned()
    }
    fn cores(&self) -> CoreCounts {
        liam_model::hardware::core_counts()
    }
    fn llama_devices(&self) -> Option<Vec<BackendDevice>> {
        liam_model::hardware::llama_backend_devices()
    }
}

/// What a `device` setting comes to on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Resolution {
    pub configured: String,
    pub resolved: String,
    pub engine: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceRow {
    pub name: String,
    pub description: String,
    pub backend: String,
    pub memory_mb: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DevicesReport {
    pub os: String,
    pub arch: String,
    pub performance_cores: Option<usize>,
    pub efficiency_cores: Option<usize>,
    pub logical_cores: usize,
    pub metal_usable: bool,
    pub built_backends: Vec<String>,
    pub llama_devices: Option<Vec<DeviceRow>>,
    pub embedder: Resolution,
    pub llm: Resolution,
}

/// Which backends this binary carries. CPU is always there.
fn built_backends(probe: &dyn HardwareProbe) -> Vec<String> {
    let mut built = Vec::new();
    if probe.os() == MACOS {
        built.push(BACKEND_METAL.to_owned());
    }
    if probe.cuda_built() {
        built.push(BACKEND_CUDA.to_owned());
    }
    built.push(BACKEND_CPU.to_owned());
    built
}

/// Resolve one `device` value with the same function the loaders call.
fn resolve(
    probe: &dyn HardwareProbe,
    configured: &str,
    engine: Engine,
    engine_name: &str,
) -> Resolution {
    let make = |resolved: &str, note: Option<String>| Resolution {
        configured: configured.to_owned(),
        resolved: resolved.to_owned(),
        engine: engine_name.to_owned(),
        note,
    };
    match DevicePreference::parse(configured) {
        None => make(
            "invalid",
            Some("not one of auto, metal, cuda, cpu".to_owned()),
        ),
        Some(preference) => match resolve_device(engine, preference, probe) {
            Ok(resolution) => make(resolution.device.as_str(), resolution.note),
            Err(refusal) => make("unavailable", Some(refusal)),
        },
    }
}

fn not_used(configured: &str, provider: &str) -> Resolution {
    Resolution {
        configured: configured.to_owned(),
        resolved: "not used".to_owned(),
        engine: format!("provider {provider}"),
        note: Some("a mock provider loads no model".to_owned()),
    }
}

/// The engine an embedder model id loads on. `None` for models that ignore the device.
fn embedder_engine(model: &str) -> Option<(Engine, &'static str)> {
    if liam_model::gguf_embedder_spec(model).is_some() {
        Some((Engine::LlamaEmbedder, "llama.cpp"))
    } else if liam_model::onnx_model_spec(model).is_some()
        || model.eq_ignore_ascii_case(liam_model::BGE_M3_MODEL_ID)
    {
        None
    } else {
        Some((Engine::Candle, "candle"))
    }
}

fn resolve_embedder(probe: &dyn HardwareProbe, config: &Config) -> Resolution {
    let configured = &config.embedder.device;
    if config.embedder.provider != LOCAL_PROVIDER {
        return not_used(configured, &config.embedder.provider);
    }
    match embedder_engine(&config.embedder.model) {
        Some((engine, name)) => resolve(probe, configured, engine, name),
        None => Resolution {
            configured: configured.clone(),
            resolved: BACKEND_CPU.to_owned(),
            engine: "onnx".to_owned(),
            note: Some("this model ignores embedder.device".to_owned()),
        },
    }
}

fn resolve_llm(probe: &dyn HardwareProbe, config: &Config) -> Resolution {
    let configured = &config.llm.device;
    if config.llm.provider != LLAMA_PROVIDER {
        return not_used(configured, &config.llm.provider);
    }
    resolve(probe, configured, Engine::LlamaLlm, "llama.cpp")
}

/// Gather everything `liam devices` prints.
pub fn report(probe: &dyn HardwareProbe, config: &Config) -> DevicesReport {
    let cores = probe.cores();
    DevicesReport {
        os: probe.os(),
        arch: probe.arch(),
        performance_cores: cores.performance,
        efficiency_cores: cores.efficiency,
        logical_cores: cores.logical,
        metal_usable: probe.metal_usable(),
        built_backends: built_backends(probe),
        llama_devices: probe.llama_devices().map(|devices| {
            devices
                .into_iter()
                .map(|device| DeviceRow {
                    name: device.name,
                    description: device.description,
                    backend: device.backend,
                    memory_mb: device.memory_total_bytes / BYTES_PER_MB,
                })
                .collect()
        }),
        embedder: resolve_embedder(probe, config),
        llm: resolve_llm(probe, config),
    }
}

fn resolution_line(key: &str, resolution: &Resolution) -> String {
    let mut line = format!(
        "{key} {} resolves to {} ({})",
        resolution.configured, resolution.resolved, resolution.engine
    );
    if let Some(note) = &resolution.note {
        let _ = write!(line, ": {note}");
    }
    line
}

/// The report as plain text, one fact per line.
pub fn format_text(report: &DevicesReport) -> String {
    let cores = match (report.performance_cores, report.efficiency_cores) {
        (Some(performance), Some(efficiency)) => format!(
            "{performance} performance, {efficiency} efficiency ({} logical)",
            report.logical_cores
        ),
        _ => format!("{} logical", report.logical_cores),
    };
    let metal = if report.metal_usable {
        "usable"
    } else {
        "not usable"
    };
    let llama = match &report.llama_devices {
        None => NOT_REPORTED.to_owned(),
        Some(devices) if devices.is_empty() => "none".to_owned(),
        Some(devices) => devices
            .iter()
            .map(|d| {
                format!(
                    "\n  {}: {} ({}, {} MB)",
                    d.name, d.description, d.backend, d.memory_mb
                )
            })
            .collect(),
    };
    [
        format!("platform: {} {}", report.os, report.arch),
        format!("cpu cores: {cores}"),
        format!("metal: {metal}"),
        format!("built backends: {}", report.built_backends.join(", ")),
        format!("llama.cpp devices: {llama}"),
        resolution_line("embedder.device", &report.embedder),
        resolution_line("llm.device", &report.llm),
    ]
    .join("\n")
}

/// The report as pretty JSON.
pub fn format_json(report: &DevicesReport) -> String {
    serde_json::to_string_pretty(report).expect("a plain struct serialises")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        os: &'static str,
        metal: bool,
        cuda: bool,
        split: bool,
        devices: Option<Vec<BackendDevice>>,
    }

    fn mac() -> Fake {
        Fake {
            os: MACOS,
            metal: true,
            cuda: false,
            split: true,
            devices: Some(vec![BackendDevice {
                name: "MTL0".into(),
                description: "Apple M1 Pro".into(),
                backend: "Metal".into(),
                memory_total_bytes: 10_922 * BYTES_PER_MB,
            }]),
        }
    }

    fn linux() -> Fake {
        Fake {
            os: "linux",
            metal: false,
            cuda: false,
            split: false,
            devices: None,
        }
    }

    impl HardwareFacts for Fake {
        fn is_macos(&self) -> bool {
            self.os == MACOS
        }
        fn metal_usable(&self) -> bool {
            self.metal
        }
        fn cuda_built(&self) -> bool {
            self.cuda
        }
    }

    impl HardwareProbe for Fake {
        fn os(&self) -> String {
            self.os.into()
        }
        fn arch(&self) -> String {
            "aarch64".into()
        }
        fn cores(&self) -> CoreCounts {
            CoreCounts {
                performance: self.split.then_some(8),
                efficiency: self.split.then_some(2),
                logical: 10,
            }
        }
        fn llama_devices(&self) -> Option<Vec<BackendDevice>> {
            self.devices.clone()
        }
    }

    fn config(provider: &str, model: &str, device: &str) -> Config {
        let mut config = Config::default();
        config.embedder.provider = provider.into();
        config.embedder.model = model.into();
        config.embedder.device = device.into();
        config.llm.provider = LLAMA_PROVIDER.into();
        config
    }

    #[test]
    fn auto_is_metal_on_a_mac_with_a_metal_device() {
        // Arrange
        let config = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "auto",
        );

        // Act
        let report = report(&mac(), &config);

        // Assert
        assert_eq!(report.embedder.resolved, "metal");
        assert_eq!(report.embedder.engine, "llama.cpp");
        assert_eq!(report.llm.resolved, "metal");
    }

    #[test]
    fn auto_falls_back_to_the_cpu_and_says_so_when_no_metal_device_opens() {
        // Arrange
        let probe = Fake {
            metal: false,
            ..mac()
        };
        let config = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "auto",
        );

        // Act
        let report = report(&probe, &config);

        // Assert
        assert_eq!(report.embedder.resolved, "cpu");
        assert!(report.embedder.note.unwrap().contains("Metal"));
    }

    #[test]
    fn auto_is_the_cpu_off_macos() {
        // Arrange
        let config = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "auto",
        );

        // Act
        let report = report(&linux(), &config);

        // Assert
        assert_eq!(report.embedder.resolved, "cpu");
        assert_eq!(report.embedder.note, None);
        assert_eq!(report.built_backends, ["cpu"]);
    }

    #[test]
    fn explicit_devices_are_checked_against_the_platform_and_the_build() {
        // Arrange
        let metal = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "metal",
        );
        let cuda = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "cuda",
        );
        let candle_cuda = config(LOCAL_PROVIDER, liam_model::QWEN3_EMBEDDING_MODEL_ID, "cuda");
        let cuda_build = Fake {
            cuda: true,
            ..linux()
        };

        // Act and assert
        assert_eq!(report(&linux(), &metal).embedder.resolved, "unavailable");
        assert_eq!(report(&cuda_build, &cuda).embedder.resolved, "unavailable");
        assert_eq!(report(&cuda_build, &candle_cuda).embedder.resolved, "cuda");
        assert_eq!(
            report(&cuda_build, &candle_cuda).built_backends,
            ["cuda", "cpu"]
        );
    }

    #[test]
    fn the_llm_report_matches_what_the_llama_cpp_loader_refuses() {
        // Arrange
        let mut cuda = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "auto",
        );
        cuda.llm.device = "cuda".into();
        let mut metal = cuda.clone();
        metal.llm.device = "metal".into();
        let cuda_build = Fake {
            cuda: true,
            ..linux()
        };

        // Act
        let cuda = report(&cuda_build, &cuda).llm;
        let metal = report(&linux(), &metal).llm;

        // Assert
        assert_eq!(cuda.resolved, "unavailable");
        assert!(cuda.note.unwrap().contains("not built into llama.cpp"));
        assert_eq!(metal.resolved, "unavailable");
    }

    #[test]
    fn candle_and_onnx_models_resolve_on_their_own_engines() {
        // Arrange
        let candle = config(LOCAL_PROVIDER, liam_model::QWEN3_EMBEDDING_MODEL_ID, "auto");
        let onnx = config(LOCAL_PROVIDER, "BAAI/bge-small-en-v1.5", "metal");

        // Act
        let candle = report(&mac(), &candle).embedder;
        let onnx = report(&mac(), &onnx).embedder;

        // Assert
        assert_eq!(
            (candle.engine.as_str(), candle.resolved.as_str()),
            ("candle", "metal")
        );
        assert_eq!(
            (onnx.engine.as_str(), onnx.resolved.as_str()),
            ("onnx", "cpu")
        );
    }

    #[test]
    fn a_mock_provider_and_a_typo_are_reported_plainly() {
        // Arrange
        let mock = Config::default();
        let typo = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "gpux",
        );

        // Act
        let mock = report(&mac(), &mock);
        let typo = report(&mac(), &typo);

        // Assert
        assert_eq!(mock.embedder.resolved, "not used");
        assert_eq!(mock.llm.resolved, "not used");
        assert_eq!(typo.embedder.resolved, "invalid");
    }

    #[test]
    fn text_names_the_core_split_the_devices_and_the_resolutions() {
        // Arrange
        let config = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "auto",
        );

        // Act
        let text = format_text(&report(&mac(), &config));

        // Assert
        assert!(text.contains("platform: macos aarch64"));
        assert!(text.contains("cpu cores: 8 performance, 2 efficiency (10 logical)"));
        assert!(text.contains("metal: usable"));
        assert!(text.contains("built backends: metal, cpu"));
        assert!(text.contains("MTL0: Apple M1 Pro (Metal, 10922 MB)"));
        assert!(text.contains("embedder.device auto resolves to metal (llama.cpp)"));
    }

    #[test]
    fn text_says_not_reported_and_uses_logical_cores_when_nothing_is_known() {
        // Arrange
        let config = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "auto",
        );

        // Act
        let text = format_text(&report(&linux(), &config));

        // Assert
        assert!(text.contains("cpu cores: 10 logical"));
        assert!(text.contains("llama.cpp devices: not reported"));
        assert!(text.contains("metal: not usable"));
    }

    #[test]
    fn json_carries_the_same_facts() {
        // Arrange
        let config = config(
            LOCAL_PROVIDER,
            liam_model::DEFAULT_EMBEDDER_MODEL_ID,
            "auto",
        );

        // Act
        let json: serde_json::Value =
            serde_json::from_str(&format_json(&report(&mac(), &config))).unwrap();

        // Assert
        assert_eq!(json["performance_cores"], 8);
        assert_eq!(json["metal_usable"], true);
        assert_eq!(json["embedder"]["resolved"], "metal");
        assert_eq!(json["llama_devices"][0]["memory_mb"], 10_922);
    }
}
