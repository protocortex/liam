// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tempfile::TempDir;

use super::*;

const STUB_SERVER_SCRIPT: &str = "#!/bin/sh\nexit 0\n";
const STUB_ROOT_DIR: &str = "llama-test";

fn sha_of(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn asset<'a>(url: &'a str, sha256: &'a str, bytes: &[u8]) -> Asset<'a> {
    Asset {
        url,
        sha256,
        size_bytes: bytes.len() as u64,
    }
}

/// Serves fixed bytes and counts calls; `fail_after` cuts the body short.
struct FakeFetcher {
    body: Vec<u8>,
    calls: AtomicUsize,
    fail_after: Option<usize>,
}

impl FakeFetcher {
    fn serving(body: &[u8]) -> Self {
        Self {
            body: body.to_vec(),
            calls: AtomicUsize::new(0),
            fail_after: None,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Fetcher for FakeFetcher {
    fn fetch(&self, _url: &str, _max: u64, sink: &mut dyn Write) -> anyhow::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.fail_after {
            Some(cut) => {
                sink.write_all(&self.body[..cut])?;
                anyhow::bail!("connection lost")
            }
            None => Ok(sink.write_all(&self.body)?),
        }
    }
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// A gzip tarball with `llama-server` (mode 0755) and a library, under one
/// top-level directory like the real release.
fn server_archive(dir: &Path, with_server: bool) -> Vec<u8> {
    let root = dir.join("src").join(STUB_ROOT_DIR);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("libggml.dylib"), b"lib").unwrap();
    std::os::unix::fs::symlink("libggml.dylib", root.join("libggml.0.dylib")).unwrap();
    if with_server {
        fs::write(root.join(SERVER_BINARY), STUB_SERVER_SCRIPT).unwrap();
        fs::set_permissions(root.join(SERVER_BINARY), fs::Permissions::from_mode(0o755)).unwrap();
    }
    let archive = dir.join("server.tar.gz");
    let status = Command::new("tar")
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(dir.join("src"))
        .arg(STUB_ROOT_DIR)
        .status()
        .unwrap();
    assert!(status.success());
    fs::read(archive).unwrap()
}

#[test]
fn model_with_a_correct_checksum_is_installed() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("models").join("extraction.gguf");

    // Act
    let outcome = install_model(&fetcher, &asset("https://x/m", &sha, body), &dest).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
    assert_eq!(fs::read(&dest).unwrap(), body);
    assert_eq!(files_in(dest.parent().unwrap()), ["extraction.gguf"]);
}

#[test]
fn model_with_a_wrong_checksum_aborts_and_leaves_no_file() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let wrong = sha_of(b"something else");
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");

    // Act
    let error = install_model(&fetcher, &asset("https://x/m", &wrong, body), &dest).unwrap_err();

    // Assert
    assert!(error.to_string().contains("sha256"), "{error}");
    assert!(
        files_in(temp.path()).is_empty(),
        "{:?}",
        files_in(temp.path())
    );
}

#[test]
fn model_of_the_wrong_size_aborts_and_leaves_no_file() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");
    let short = Asset {
        url: "https://x/m",
        sha256: &sha,
        size_bytes: body.len() as u64 + 1,
    };

    // Act
    let error = install_model(&fetcher, &short, &dest).unwrap_err();

    // Assert
    assert!(error.to_string().contains("bytes"), "{error}");
    assert!(files_in(temp.path()).is_empty());
}

#[test]
fn model_download_cut_short_leaves_no_partial_file() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let mut fetcher = FakeFetcher::serving(body);
    fetcher.fail_after = Some(4);
    let dest = temp.path().join("extraction.gguf");

    // Act
    let result = install_model(&fetcher, &asset("https://x/m", &sha, body), &dest);

    // Assert
    assert!(result.is_err());
    assert!(files_in(temp.path()).is_empty());
}

#[test]
fn model_already_verified_is_not_downloaded_again() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");
    let pinned = asset("https://x/m", &sha, body);
    install_model(&fetcher, &pinned, &dest).unwrap();

    // Act
    let outcome = install_model(&fetcher, &pinned, &dest).unwrap();

    // Assert
    assert_eq!(outcome, Installed::AlreadyPresent);
    assert_eq!(fetcher.calls(), 1);
}

#[test]
fn model_that_fails_its_hash_is_replaced() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub";
    let sha = sha_of(body);
    let fetcher = FakeFetcher::serving(body);
    let dest = temp.path().join("extraction.gguf");
    fs::write(&dest, b"truncated").unwrap();

    // Act
    let outcome = install_model(&fetcher, &asset("https://x/m", &sha, body), &dest).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
    assert_eq!(fs::read(&dest).unwrap(), body);
}

#[test]
fn server_with_a_correct_checksum_is_installed_and_executable() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), true);
    let sha = sha_of(&archive);
    let fetcher = FakeFetcher::serving(&archive);
    let dir = temp.path().join("bin").join("llama-test");

    // Act
    let outcome = install_server(&fetcher, &asset("https://x/s", &sha, &archive), &dir).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
    assert!(is_executable_file(&dir.join(SERVER_BINARY)));
    assert!(dir.join("libggml.dylib").is_file());
    assert_eq!(files_in(dir.parent().unwrap()), ["llama-test"]);
}

#[test]
fn server_with_a_wrong_checksum_aborts_and_leaves_nothing() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), true);
    let wrong = sha_of(b"other");
    let fetcher = FakeFetcher::serving(&archive);
    let bin = temp.path().join("bin");
    let dir = bin.join("llama-test");

    // Act
    let result = install_server(&fetcher, &asset("https://x/s", &wrong, &archive), &dir);

    // Assert
    assert!(result.is_err());
    assert!(files_in(&bin).is_empty(), "{:?}", files_in(&bin));
}

#[test]
fn server_archive_without_the_binary_is_rejected_and_leaves_nothing() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), false);
    let sha = sha_of(&archive);
    let fetcher = FakeFetcher::serving(&archive);
    let bin = temp.path().join("bin");

    // Act
    let error = install_server(
        &fetcher,
        &asset("https://x/s", &sha, &archive),
        &bin.join("llama-test"),
    )
    .unwrap_err();

    // Assert
    assert!(error.to_string().contains(SERVER_BINARY), "{error}");
    assert!(files_in(&bin).is_empty());
}

/// Builds a gzip tarball whose single risky entry is described by `kind`.
fn unsafe_archive(dir: &Path, kind: &str) -> Vec<u8> {
    let work = dir.join("work");
    fs::create_dir_all(work.join("root")).unwrap();
    fs::write(work.join("x"), b"x").unwrap();
    let archive = dir.join("evil.tar.gz");
    let mut tar = Command::new("tar");
    tar.arg("-czPf").arg(&archive).arg("-C").arg(&work);
    match kind {
        "absolute" => tar.arg(work.join("x")),
        "parent" => tar.arg("root/../x"),
        "symlink_out" => {
            std::os::unix::fs::symlink("/etc/passwd", work.join("root/link")).unwrap();
            tar.arg("root/link")
        }
        "symlink_up" => {
            std::os::unix::fs::symlink("../x", work.join("root/link")).unwrap();
            tar.arg("root/link")
        }
        other => panic!("unknown kind {other}"),
    };
    assert!(tar.status().unwrap().success());
    fs::read(archive).unwrap()
}

#[test]
fn server_archive_with_an_unsafe_entry_is_rejected() {
    for (kind, message) in [
        ("absolute", "unsafe entry"),
        ("parent", "unsafe entry"),
        ("symlink_out", "unsafe link"),
        ("symlink_up", "unsafe link"),
    ] {
        // Arrange
        let temp = TempDir::new().unwrap();
        let archive = unsafe_archive(temp.path(), kind);
        let sha = sha_of(&archive);
        let fetcher = FakeFetcher::serving(&archive);
        let bin = temp.path().join("bin");

        // Act
        let error = install_server(
            &fetcher,
            &asset("https://x/s", &sha, &archive),
            &bin.join("llama-test"),
        )
        .unwrap_err();

        // Assert
        assert!(error.to_string().contains(message), "{kind}: {error}");
        assert!(files_in(&bin).is_empty(), "{kind}");
    }
}

#[test]
fn concurrent_model_installs_never_publish_unverified_bytes() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let body = b"gguf stub".to_vec();
    let sha = sha_of(&body);
    let wrong = sha_of(b"other");
    let dest = temp.path().join("extraction.gguf");

    // Act
    let results: Vec<bool> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..6)
            .map(|n| {
                let (body, dest) = (&body, &dest);
                let (sha, wrong) = (&sha, &wrong);
                scope.spawn(move || {
                    let fetcher = FakeFetcher::serving(body);
                    let expected = if n == 5 { wrong } else { sha };
                    install_model(&fetcher, &asset("https://x/m", expected, body), dest).is_ok()
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });

    // Assert
    assert_eq!(results.iter().filter(|ok| !**ok).count(), 1);
    assert_eq!(fs::read(&dest).unwrap(), body);
    assert_eq!(files_in(temp.path()), ["extraction.gguf"]);
}

#[test]
fn concurrent_server_installs_leave_one_complete_directory() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), true);
    let sha = sha_of(&archive);
    let dir = temp.path().join("bin").join("llama-test");

    // Act
    let results: Vec<_> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let (archive, sha, dir) = (&archive, &sha, &dir);
                scope.spawn(move || {
                    let fetcher = FakeFetcher::serving(archive);
                    install_server(&fetcher, &asset("https://x/s", sha, archive), dir)
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });

    // Assert
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    assert!(is_executable_file(&dir.join(SERVER_BINARY)));
    assert_eq!(files_in(dir.parent().unwrap()), ["llama-test"]);
}

#[test]
fn server_already_installed_is_not_downloaded_again() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let archive = server_archive(temp.path(), true);
    let sha = sha_of(&archive);
    let fetcher = FakeFetcher::serving(&archive);
    let dir = temp.path().join("bin").join("llama-test");
    let pinned = asset("https://x/s", &sha, &archive);
    install_server(&fetcher, &pinned, &dir).unwrap();

    // Act
    let outcome = install_server(&fetcher, &pinned, &dir).unwrap();

    // Assert
    assert_eq!(outcome, Installed::AlreadyPresent);
    assert_eq!(fetcher.calls(), 1);
}

fn extraction(enabled: bool, llm_enabled: bool) -> ExtractionConfig {
    ExtractionConfig {
        enabled,
        llm_enabled,
        ..ExtractionConfig::default()
    }
}

/// Serves the server archive for the `/s` URL and the model bytes otherwise.
struct RoutedFetcher {
    server: Vec<u8>,
    model: Vec<u8>,
    urls: std::sync::Mutex<Vec<String>>,
}

impl Fetcher for RoutedFetcher {
    fn fetch(&self, url: &str, _max: u64, sink: &mut dyn Write) -> anyhow::Result<()> {
        self.urls.lock().unwrap().push(url.to_owned());
        let body = if url.ends_with("/s") {
            &self.server
        } else {
            &self.model
        };
        Ok(sink.write_all(body)?)
    }
}

#[test]
fn extraction_files_are_not_fetched_when_extraction_is_off() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let fetcher = FakeFetcher::serving(b"unused");
    let mut out = Vec::new();

    // Act
    fetch_extraction(
        &extraction(false, true),
        temp.path().to_str().unwrap(),
        &fetcher,
        &PINNED,
        &mut out,
    )
    .unwrap();

    // Assert
    assert_eq!(fetcher.calls(), 0);
    assert!(out.is_empty());
    assert!(files_in(temp.path()).is_empty());
}

#[test]
fn extraction_with_rules_only_fetches_nothing() {
    // Arrange / Act / Assert
    assert!(!wants_files(&extraction(true, false)));
    assert!(wants_files(&extraction(true, true)));
    assert!(!wants_files(&extraction(false, true)));
}

#[test]
fn extraction_enabled_fetches_the_server_and_the_model() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    let scratch = temp.path().join("scratch");
    fs::create_dir_all(&scratch).unwrap();
    let server = server_archive(&scratch, true);
    let model = b"gguf stub".to_vec();
    let (server_sha, model_sha) = (sha_of(&server), sha_of(&model));
    let pinned = Pinned {
        server: asset("https://x/s", &server_sha, &server),
        model: asset("https://x/m", &model_sha, &model),
    };
    let fetcher = RoutedFetcher {
        server,
        model,
        urls: Default::default(),
    };
    let mut out = Vec::new();

    // Act
    fetch_extraction(
        &extraction(true, true),
        home.to_str().unwrap(),
        &fetcher,
        &pinned,
        &mut out,
    )
    .unwrap();

    // Assert
    assert_eq!(fetcher.urls.lock().unwrap().len(), 2);
    let server_path = default_server_dir(home.to_str().unwrap()).join(SERVER_BINARY);
    assert!(is_executable_file(&server_path));
    assert!(default_model_path(home.to_str().unwrap()).is_file());
    let report = String::from_utf8(out).unwrap();
    assert!(report.contains("llama_server_path"), "{report}");
}

#[test]
fn a_server_configured_elsewhere_is_kept() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let model = b"gguf stub".to_vec();
    let model_sha = sha_of(&model);
    let pinned = Pinned {
        server: asset("https://x/s", "unused", b""),
        model: asset("https://x/m", &model_sha, &model),
    };
    let fetcher = RoutedFetcher {
        server: Vec::new(),
        model,
        urls: Default::default(),
    };
    let mut config = extraction(true, true);
    config.llama_server_path = Some("/opt/own/llama-server".into());
    let mut out = Vec::new();

    // Act
    fetch_extraction(
        &config,
        temp.path().to_str().unwrap(),
        &fetcher,
        &pinned,
        &mut out,
    )
    .unwrap();

    // Assert
    assert_eq!(*fetcher.urls.lock().unwrap(), ["https://x/m"]);
}

#[test]
fn platform_check_accepts_only_apple_silicon_macos() {
    // Arrange
    let cases = [
        ("macos", "aarch64", true),
        ("macos", "x86_64", false),
        ("linux", "aarch64", false),
        ("linux", "x86_64", false),
    ];

    for (os, arch, ok) in cases {
        // Act
        let result = check_platform(os, arch);

        // Assert
        assert_eq!(result.is_ok(), ok, "{os}/{arch}");
        if let Err(error) = result {
            assert!(
                error.to_string().contains(&format!("{os}/{arch}")),
                "{error}"
            );
        }
    }
}

fn is_lower_hex_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[test]
fn pinned_constants_are_well_formed() {
    for (name, pinned) in [("server", SERVER_ASSET), ("model", MODEL_ASSET)] {
        assert!(
            is_lower_hex_sha256(pinned.sha256),
            "{name} sha256: {}",
            pinned.sha256
        );
        assert!(
            is_well_formed_https_url(pinned.url),
            "{name} url: {}",
            pinned.url
        );
        assert!(pinned.size_bytes > 0, "{name} size");
    }
    assert!(SERVER_ASSET.url.contains(SERVER_TAG));
}

#[test]
fn pinned_assets_name_the_expected_release_and_repository() {
    // Arrange
    let server_url = format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{SERVER_TAG}/\
         llama-{SERVER_TAG}-bin-macos-arm64.tar.gz"
    );
    let model_prefix = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/";
    let revision = MODEL_ASSET
        .url
        .strip_prefix(model_prefix)
        .and_then(|rest| rest.strip_suffix("/Qwen3.5-4B-Q4_K_M.gguf"))
        .expect("model url has an unexpected repository or file name");

    // Act / Assert
    assert_eq!(SERVER_TAG, "b11860");
    assert_eq!(SERVER_ASSET.url, server_url);
    assert_eq!(revision.len(), 40);
    assert!(revision.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(
        SERVER_ASSET.sha256,
        "204fe983df733653d1045d0dfb768f5604c18f8f2859ed804a561a54311a3a9d"
    );
    assert_eq!(
        MODEL_ASSET.sha256,
        "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
    );
    assert_eq!(SERVER_ASSET.size_bytes, 16_245_641);
    assert_eq!(MODEL_ASSET.size_bytes, 2_740_937_888);
}

#[test]
fn url_shape_check_rejects_typos() {
    for bad in [
        "http://example.com/a",
        "https://",
        "https:///path",
        "https://exa mple.com/a",
        "https://user@example.com/a",
        "example.com/a",
    ] {
        assert!(!is_well_formed_https_url(bad), "{bad}");
    }
    assert!(is_well_formed_https_url("https://example.com/a?b=c"));
}

/// A loopback HTTP server: `/ok`, `/redirect`, `/loop`, `/big` and a 404.
struct Loopback {
    port: u16,
    requests: Arc<AtomicUsize>,
}

const OK_BODY: &[u8] = b"payload";
const BIG_BODY_BYTES: usize = 1 << 20;

impl Loopback {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&requests);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                counter.fetch_add(1, Ordering::SeqCst);
                std::thread::spawn(move || serve(stream, port));
            }
        });
        Self { port, requests }
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
}

fn serve(mut stream: TcpStream, port: u16) {
    let mut request = Vec::new();
    let mut byte = [0u8; 1];
    while !request.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        request.push(byte[0]);
    }
    let path = String::from_utf8_lossy(&request)
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_owned();
    let (head, body): (String, Vec<u8>) = match path.as_str() {
        "/ok" => ("200 OK".into(), OK_BODY.to_vec()),
        "/redirect" => (
            format!("302 Found\r\nLocation: http://127.0.0.1:{port}/ok"),
            Vec::new(),
        ),
        "/loop" => (
            format!("302 Found\r\nLocation: http://127.0.0.1:{port}/loop"),
            Vec::new(),
        ),
        "/big" => ("200 OK".into(), vec![0; BIG_BODY_BYTES]),
        _ => ("404 Not Found".into(), Vec::new()),
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {head}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(&body);
}

fn fetch_bytes(fetcher: &CurlFetcher, url: &str, max: u64) -> anyhow::Result<Vec<u8>> {
    let mut sink = Vec::new();
    fetcher.fetch(url, max, &mut sink)?;
    Ok(sink)
}

#[test]
fn curl_fetcher_returns_the_body() {
    // Arrange
    let server = Loopback::start();

    // Act
    let body = fetch_bytes(&CurlFetcher::plain_http(), &server.url("/ok"), 1024).unwrap();

    // Assert
    assert_eq!(body, OK_BODY);
}

#[test]
fn curl_fetcher_follows_a_redirect() {
    // Arrange
    let server = Loopback::start();

    // Act
    let body = fetch_bytes(&CurlFetcher::plain_http(), &server.url("/redirect"), 1024).unwrap();

    // Assert
    assert_eq!(body, OK_BODY);
}

#[test]
fn curl_fetcher_gives_up_on_a_redirect_loop() {
    // Arrange
    let server = Loopback::start();

    // Act
    let error = fetch_bytes(&CurlFetcher::plain_http(), &server.url("/loop"), 1024).unwrap_err();

    // Assert
    assert!(error.to_string().contains("curl failed"), "{error}");
    assert!(server.requests.load(Ordering::SeqCst) <= 7);
}

#[test]
fn curl_fetcher_stops_at_the_size_cap() {
    // Arrange
    let server = Loopback::start();
    let mut sink = Vec::new();

    // Act
    let error = CurlFetcher::plain_http()
        .fetch(&server.url("/big"), 1000, &mut sink)
        .unwrap_err();

    // Assert
    assert!(error.to_string().contains("larger than"), "{error}");
    assert!(sink.len() <= 1000);
}

#[test]
fn curl_fetcher_reports_an_http_error() {
    // Arrange
    let server = Loopback::start();

    // Act
    let error = fetch_bytes(&CurlFetcher::plain_http(), &server.url("/missing"), 1024).unwrap_err();

    // Assert
    assert!(error.to_string().contains("curl failed"), "{error}");
}

#[test]
fn https_only_fetcher_refuses_plain_http_without_connecting() {
    // Arrange
    let server = Loopback::start();

    // Act
    let result = fetch_bytes(&CurlFetcher::https_only(), &server.url("/ok"), 1024);

    // Assert
    assert!(result.is_err());
    assert_eq!(server.requests.load(Ordering::SeqCst), 0);
}

#[test]
fn https_only_fetcher_restricts_redirects_and_ignores_curlrc() {
    // Arrange / Act
    let args = CurlFetcher::https_only().command_args("https://example.com/a");

    // Assert
    assert_eq!(args[0], "-q");
    for flag in ["--proto", "--proto-redir"] {
        let at = args.iter().position(|arg| arg == flag).unwrap();
        assert_eq!(args[at + 1], "=https", "{flag}");
    }
    assert_eq!(args.last().unwrap(), "https://example.com/a");
}

/// Runs `packaging/install.sh` from a scratch copy of an unpacked release, with
/// a stub `liam` that records its arguments.
struct InstallRun {
    status: std::process::ExitStatus,
    stderr: String,
    recorded_args: Option<String>,
}

fn run_install_script(platform: &str, extra: &[&str]) -> InstallRun {
    let temp = TempDir::new().unwrap();
    let src = temp.path().join("release");
    let home = temp.path().join("home");
    fs::create_dir_all(&src).unwrap();
    fs::create_dir_all(&home).unwrap();
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packaging/install.sh");
    fs::copy(script, src.join("install.sh")).unwrap();
    for name in ["liamd", "liam.toml", "dev.protocortex.liamd.plist"] {
        fs::write(src.join(name), "stub").unwrap();
    }
    let recorded = temp.path().join("args.txt");
    let stub = format!("#!/bin/sh\necho \"$@\" > '{}'\n", recorded.display());
    fs::write(src.join("liam"), stub).unwrap();
    fs::set_permissions(src.join("liam"), fs::Permissions::from_mode(0o755)).unwrap();

    let output = Command::new("sh")
        .arg(src.join("install.sh"))
        .arg("--prefix")
        .arg(temp.path().join("bin"))
        .arg("--no-launchd")
        .args(extra)
        .env("HOME", &home)
        .env("LIAM_PLATFORM_OVERRIDE", platform)
        .output()
        .unwrap();
    InstallRun {
        status: output.status,
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        recorded_args: fs::read_to_string(recorded).ok(),
    }
}

#[test]
fn install_script_fetches_models_through_the_cli() {
    // Arrange / Act
    let run = run_install_script("Darwin/arm64", &[]);

    // Assert
    assert!(run.status.success(), "{}", run.stderr);
    let args = run.recorded_args.expect("liam was not invoked");
    assert!(args.starts_with("fetch-models --config "), "{args}");
    assert!(args.trim_end().ends_with("/.liam/liam.toml"), "{args}");
}

#[test]
fn install_script_skips_the_fetch_when_asked() {
    // Arrange / Act
    let run = run_install_script("Darwin/arm64", &["--skip-models"]);

    // Assert
    assert!(run.status.success(), "{}", run.stderr);
    assert!(run.recorded_args.is_none());
}

#[test]
fn install_script_rejects_an_unsupported_platform() {
    // Arrange / Act
    let run = run_install_script("Linux/x86_64", &[]);

    // Assert
    assert_eq!(run.status.code(), Some(1));
    assert!(
        run.stderr
            .contains("macOS on Apple Silicon only, found Linux/x86_64"),
        "{}",
        run.stderr
    );
    assert!(run.recorded_args.is_none());
}

/// Downloads the real server archive and checks the pin. Run with
/// `cargo test -p liam-daemon -- --ignored real_server_asset`.
#[test]
#[ignore = "downloads the pinned llama.cpp release over the network"]
fn real_server_asset_matches_its_pin() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let dir = temp.path().join("llama");

    // Act
    let outcome = install_server(&CurlFetcher::https_only(), &SERVER_ASSET, &dir).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
    assert!(is_executable_file(&dir.join(SERVER_BINARY)));
}

/// Downloads the real model (about 2.7 GB) and checks the pin.
#[test]
#[ignore = "downloads the pinned model, several gigabytes"]
fn real_model_asset_matches_its_pin() {
    // Arrange
    let temp = TempDir::new().unwrap();
    let dest = temp.path().join("extraction.gguf");

    // Act
    let outcome = install_model(&CurlFetcher::https_only(), &MODEL_ASSET, &dest).unwrap();

    // Assert
    assert_eq!(outcome, Installed::Downloaded);
}
