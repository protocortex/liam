use tempfile::TempDir;

use super::*;

const LOOPBACK: &str = "127.0.0.1";
const LEAK_PROBE: &str = "LIAM_TEST_SIDECAR_LEAK";

#[test]
fn the_production_command_keeps_the_secrets_off_argv() {
    // Arrange
    let config = ExtractionConfig {
        extra_args: vec!["--threads".into(), "2".into()],
        ..ExtractionConfig::default()
    };
    let paths = Paths {
        server: "/bin/llama-server".into(),
        model: "/models/m.gguf".into(),
    };
    let launch = LaunchParams {
        addr: "127.0.0.1:9".parse().unwrap(),
        alias: "the-alias",
        key: "the-key",
    };

    // Act
    let command = server_command(&config, &paths, &launch);

    // Assert
    let args: Vec<String> = command
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert!(args
        .iter()
        .all(|a| !a.contains("the-key") && !a.contains("the-alias")));
    assert!(args.windows(2).any(|w| w == ["--host", LOOPBACK]));
    assert!(args.windows(2).any(|w| w == ["--threads", "2"]));
    let envs: Vec<_> = command.get_envs().collect();
    assert!(envs.contains(&(API_KEY_ENV.as_ref(), Some("the-key".as_ref()))));
    assert!(envs.contains(&(ALIAS_ENV.as_ref(), Some("the-alias".as_ref()))));
}

#[test]
fn bad_paths_are_typed_errors_naming_the_key() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let plain = dir.path().join("plain");
    std::fs::write(&plain, "x").unwrap();
    let plain = plain.to_str().unwrap();
    let missing = "/nonexistent/file";
    let cases = [
        (None, None, "llama_server_path"),
        (Some(missing), Some(missing), "llama_server_path"),
        (Some(plain), Some(plain), "llama_server_path"),
        (Some("/bin/sh"), Some(missing), "model_gguf"),
        (Some("/bin/sh"), None, "model_gguf"),
    ];

    for (server, model, key) in cases {
        // Act
        let config = ExtractionConfig {
            llama_server_path: server.map(Into::into),
            model_gguf: model.map(Into::into),
            ..ExtractionConfig::default()
        };
        let result = Paths::resolve(&config);

        // Assert
        assert!(
            matches!(result, Err(SidecarError::BadPath { key: k, .. }) if k == key),
            "{server:?} {model:?}"
        );
    }
}

#[test]
fn stderr_keeps_only_the_tail_and_never_the_secrets() {
    // Arrange
    let mut tail = StderrTail::default();
    let filler = "x".repeat(STDERR_CAP_BYTES);

    // Act
    tail.push(format!("key=SECRETKEY\n{filler}\nlast alias=ALIAS1 line\n").as_bytes());
    let text = tail.redacted(&["SECRETKEY", "ALIAS1"]);

    // Assert
    assert!(text.len() <= STDERR_CAP_BYTES);
    assert!(text.contains("last alias=[redacted] line"));
    assert!(!text.contains("SECRETKEY") && !text.contains("ALIAS1"));
}

#[test]
fn a_secret_cut_by_the_cap_leaves_no_fragment() {
    // Arrange
    let mut tail = StderrTail::default();
    let filler = "y".repeat(STDERR_CAP_BYTES - 5);

    // Act
    tail.push(format!("ABCDEFGH{filler}").as_bytes());
    let text = tail.redacted(&["ABCDEFGH"]);

    // Assert
    assert!(!text.contains("DEFGH"), "a fragment of the secret survived");
}

#[test]
fn the_child_inherits_only_the_allowlist_and_its_own_secrets() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let script = dir.path().join("print-env");
    std::fs::write(&script, "#!/bin/sh\nenv\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let paths = Paths {
        server: script,
        model: "/models/m.gguf".into(),
    };
    let launch = LaunchParams {
        addr: "127.0.0.1:9".parse().unwrap(),
        alias: "the-alias",
        key: "the-key",
    };
    std::env::set_var(LEAK_PROBE, "leaked");

    // Act
    let output = server_command(&ExtractionConfig::default(), &paths, &launch)
        .output()
        .unwrap();

    // Assert
    let printed = String::from_utf8_lossy(&output.stdout);
    assert!(printed.contains("LLAMA_API_KEY=the-key"), "{printed}");
    assert!(printed.contains("PATH="), "{printed}");
    assert!(!printed.contains(LEAK_PROBE), "{printed}");
}

#[test]
fn the_random_values_differ_and_are_hex() {
    // Arrange / Act
    let (a, b) = (
        random_hex(KEY_BYTES).unwrap(),
        random_hex(KEY_BYTES).unwrap(),
    );

    // Assert
    assert_ne!(a, b);
    assert_eq!(a.len(), KEY_BYTES * 2);
    assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
}
