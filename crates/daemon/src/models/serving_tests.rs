// SPDX-License-Identifier: Apache-2.0
//! Choosing the serving LLM from config, without loading any weights.

use super::*;

fn config_for(provider: &str, device: &str) -> Config {
    let mut config = Config::default();
    config.llm.provider = provider.to_string();
    config.llm.device = device.to_string();
    config
}

#[test]
fn a_mock_provider_serves_directly_and_has_nothing_to_unload() {
    // Arrange
    let config = config_for(MOCK_PROVIDER, "auto");

    // Act
    let serving = build_serving_llm(&config).expect("the mock provider builds");

    // Assert
    assert!(serving.lazy.is_none());
    assert_eq!(serving.llm.backend(), MockLlm.backend());
}

#[test]
fn a_llama_provider_with_an_explicit_cpu_device_loads_nothing_until_first_use() {
    // Arrange
    let config = config_for(LLAMA_PROVIDER, "cpu");

    // Act
    let serving = build_serving_llm(&config).expect("an explicit cpu device is accepted");

    // Assert
    let expected = liam_model::llama::llama_runtime_backend().unwrap();
    assert!(serving.lazy.is_some(), "the handle that unloads when idle");
    assert_eq!(serving.llm.backend(), expected);
}

#[test]
fn a_llama_provider_with_an_unknown_device_names_the_bad_value() {
    // Arrange
    let config = config_for(LLAMA_PROVIDER, "tpu");

    // Act
    let error = build_serving_llm(&config).err().expect("a typo is refused");

    // Assert
    assert_eq!(
        error.to_string(),
        r#"llm.device = "tpu" is not one of auto, metal, cuda, cpu"#
    );
}
