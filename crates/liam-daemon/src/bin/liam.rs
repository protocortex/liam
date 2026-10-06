// SPDX-License-Identifier: Apache-2.0
//! `liam`: the command line tool a person runs.
//!
//! Split from `liamd` because the two serve different moments. `liamd` is
//! started by launchd or by an MCP client and speaks nothing but JSON-RPC on
//! its stdio, which is why it logs to stderr and never prints. `liam` is
//! typed at a prompt and writes for a human on stdout.
//!
//! Folding them into one binary would force a bad choice: either the daemon
//! grows human-readable output that corrupts the MCP stream an client is
//! parsing, or the CLI stays mute through a multi-gigabyte download that
//! looks indistinguishable from a hang.
//!
//! Both binaries come from the same crate so they cannot disagree about
//! config or model paths. See `lib.rs` for why that matters.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use clap::{Args, Parser, Subcommand};
use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{WalConfig, WalWriter};
use liam_store::{
    ContentEmbedder, DefaultGraph, Error as StoreError, EventLog, GraphConfig, MismatchSource,
    RebuildMode, RebuildReport, SharedLog,
};

use liam_daemon::config::{resolve_config_source, Config};
use liam_daemon::models::{self, StoreEmbedder};
use liam_daemon::storelock::StoreLock;

#[derive(Debug, Parser)]
#[command(
    name = "liam",
    version,
    about = "LIAM command line tool: prepares and inspects local memory state.",
    long_about = None,
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    /// Path to liam.toml. Overrides the LIAM_CONFIG environment variable.
    #[arg(long, value_name = "PATH", global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq, Subcommand)]
enum Command {
    /// Download and load every model liam.toml asks for, so the first daemon
    /// start is not also the first download.
    FetchModels,
    /// Rebuild a store's nodes and edges from its event log, then check the
    /// result against the log row by row. Refuses while a daemon holds the
    /// store.
    Rebuild(RebuildArgs),
}

#[derive(Debug, Clone, PartialEq, Eq, Args)]
struct RebuildArgs {
    /// The database to rebuild. It is created when it does not exist.
    #[arg(long, value_name = "PATH")]
    database: PathBuf,

    /// The event log directory. Defaults to `<database stem>.log` beside the
    /// database.
    #[arg(long, value_name = "DIR")]
    log_dir: Option<PathBuf>,

    /// Rebuild even when the database holds rows the log does not record, such
    /// as rows written before the log existed. Those rows are deleted.
    #[arg(long)]
    force: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config_path = resolve_config_source(
        cli.config.as_deref(),
        std::env::var("LIAM_CONFIG").ok().as_deref(),
    );
    let config = Config::load(&config_path)?;

    match cli.command {
        Command::FetchModels => fetch_models(&config, &config_path),
        Command::Rebuild(args) => {
            export_embedder_cache(&config)?;
            let runtime = tokio::runtime::Runtime::new()?;
            let code = runtime.block_on(rebuild(
                &args,
                &config,
                &mut std::io::stdout(),
                &mut std::io::stderr(),
            ));
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
    }
}

/// The segment sizes of the log a rebuild opens. A rebuild appends only the
/// voids of events it refuses, so these thresholds are rarely reached.
const REBUILD_WAL: WalConfig = WalConfig {
    segment_max_bytes: 64 << 20,
    rotate_interval_secs: 3600,
};

/// GC does not record what it deletes yet, so a rebuild cannot know it.
const GC_WARNING: &str = "warning: a store that ran GC before the log recorded deletions can get \
     the rows GC removed back from the log. Back up the database before relying on a rebuild.";

/// `<database stem>.log` beside the database, until the config names a log
/// directory.
fn default_log_dir(database: &Path) -> PathBuf {
    let stem = database.file_stem().unwrap_or(database.as_os_str());
    let mut name = stem.to_owned();
    name.push(".log");
    database.with_file_name(name)
}

/// fastembed reads its cache directory from the environment, which is only safe
/// to set while the process has a single thread, so this runs before the
/// runtime starts.
fn export_embedder_cache(config: &Config) -> anyhow::Result<()> {
    if config.embedder.provider != "local" {
        return Ok(());
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let cache_dir =
        models::resolve_path_with_home("embedder.cache_dir", &config.embedder.cache_dir, &home)?;
    std::env::set_var("FASTEMBED_CACHE_DIR", cache_dir);
    Ok(())
}

/// Runs `liam rebuild`: the report goes to `out`, a refusal or a failure to
/// `err`. Returns the process exit code, non zero on any failure.
async fn rebuild(
    args: &RebuildArgs,
    config: &Config,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    match try_rebuild(args, config, out, err).await {
        Ok(()) => 0,
        Err(error) => {
            let _ = writeln!(err, "liam rebuild: {error:#}");
            1
        }
    }
}

/// The steps in order: the log must exist and the store lock is ours before the
/// database is opened, and the store's own checks run before it deletes
/// anything. `--force` only widens the mode after the store refused the safe
/// one, so a refusal is never skipped unseen.
async fn try_rebuild(
    args: &RebuildArgs,
    config: &Config,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> anyhow::Result<()> {
    let log_dir = args
        .log_dir
        .clone()
        .unwrap_or_else(|| default_log_dir(&args.database));
    anyhow::ensure!(
        log_dir.is_dir(),
        "the log directory {} does not exist; pass --log-dir if the log lives elsewhere",
        log_dir.display()
    );
    if let Some(parent) = args.database.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    let _lock = StoreLock::acquire(&args.database)?;
    tracing::debug!("rebuild: store lock taken");

    let log = open_log(&log_dir)?;
    let embedder = store_embedder(config, err)?;
    let graph = open_graph(args, config, embedder.as_ref()).await?;
    let mode = if graph.projection_hash_multiset().await?.is_empty() {
        RebuildMode::RequireEmpty
    } else {
        RebuildMode::ResetChecked
    };

    writeln!(err, "{GC_WARNING}")?;
    let rebuilt = match graph.rebuild_from_log(log.clone(), mode).await {
        Err(refusal) if args.force && is_forceable(&refusal) => {
            writeln!(err, "{}", force_warning(&refusal))?;
            // The refused store was consumed, so the replacing run opens its own.
            open_graph(args, config, embedder.as_ref())
                .await?
                .rebuild_from_log(log, RebuildMode::Replace)
                .await
        }
        other => other,
    };
    let (_graph, report) = rebuilt.map_err(explain_refusal)?;
    tracing::debug!("rebuild: done");
    print_report(out, &args.database, &log_dir, &report)?;
    Ok(())
}

fn open_log(log_dir: &Path) -> anyhow::Result<SharedLog> {
    let writer = WalWriter::open_with_system_clock(log_dir, REBUILD_WAL)
        .with_context(|| format!("could not open the log in {}", log_dir.display()))?;
    let reader = SequentialScanReader::local(log_dir)
        .with_context(|| format!("could not read the log in {}", log_dir.display()))?;
    let log = EventLog::new(
        Box::new(writer),
        Arc::new(reader),
        HashBloom::new(BloomConfig::default()),
    );
    Ok(Arc::new(tokio::sync::Mutex::new(log)))
}

/// The embedder restored nodes are embedded with, or none after a warning: the
/// rebuild still succeeds and reports the nodes left pending.
fn store_embedder(
    config: &Config,
    err: &mut dyn Write,
) -> anyhow::Result<Option<Arc<dyn ContentEmbedder>>> {
    match models::build_models(config) {
        Ok((embedder, _reranker)) => Ok(Some(Arc::new(StoreEmbedder(embedder)))),
        Err(error) => {
            writeln!(
                err,
                "warning: no embedder ({error:#}); restored nodes stay without a vector until a \
                 later pass embeds them"
            )?;
            Ok(None)
        }
    }
}

/// The database with no log attached: `rebuild_from_log` attaches it only once
/// the rebuild has been checked.
async fn open_graph(
    args: &RebuildArgs,
    config: &Config,
    embedder: Option<&Arc<dyn ContentEmbedder>>,
) -> anyhow::Result<DefaultGraph> {
    let graph = DefaultGraph::open(
        args.database
            .to_str()
            .context("the database path is not UTF-8")?,
        GraphConfig::new(config.embedding_dims),
    )
    .await?;
    Ok(match embedder {
        Some(embedder) => graph.with_embedder(Arc::clone(embedder)),
        None => graph,
    })
}

/// Refusals made before anything was deleted, over rows only the database
/// holds. `--force` may answer them with a replacing rebuild.
fn is_forceable(error: &StoreError) -> bool {
    matches!(
        error,
        StoreError::ProjectionNotEmpty
            | StoreError::RebuildMismatch {
                against: MismatchSource::PriorProjection,
                ..
            }
    )
}

/// Names what a forced rebuild is about to delete.
fn force_warning(refusal: &StoreError) -> String {
    match refusal {
        StoreError::RebuildMismatch {
            found, unexpected, ..
        } => format!(
            "warning: deleting {unexpected} of {found} live rows the log does not record (--force)"
        ),
        _ => "warning: replacing superseded rows the log may not record (--force)".to_string(),
    }
}

/// Puts a refusal or failure of the store in terms of the command line.
fn explain_refusal(error: StoreError) -> anyhow::Error {
    match error {
        StoreError::RebuildMismatch {
            against: MismatchSource::PriorProjection,
            found,
            missing,
            unexpected,
            ..
        } => anyhow::anyhow!(
            "hash mismatch between the database and the log: {unexpected} of the {found} live \
             rows (nodes and edges) are not in the log, and {missing} rows the log records are \
             not in the database. Rows written before the log existed are the usual cause. \
             Nothing was changed. Back up the database, then pass --force to rebuild from the \
             log anyway; the rows the log does not record are deleted"
        ),
        StoreError::ProjectionNotEmpty => anyhow::anyhow!(
            "the database has no live rows but holds superseded ones the log may not record. \
             Nothing was changed. Back up the database, then pass --force to rebuild from the \
             log anyway"
        ),
        refused @ (StoreError::EmptyLogWouldWipe { .. }
        | StoreError::ForeignLogWouldWipe { .. }) => {
            anyhow::Error::new(refused).context("the rebuild was refused and nothing was changed")
        }
        other => anyhow::Error::new(other).context(
            "the rebuild did not finish cleanly and the database may hold a partial rebuild; \
             run it again with --force to start over from the log, or restore a backup",
        ),
    }
}

fn print_report(
    out: &mut dyn Write,
    database: &Path,
    log_dir: &Path,
    report: &RebuildReport,
) -> std::io::Result<()> {
    let (replayed, embedded) = (&report.replayed, &report.replayed.reembedded);
    writeln!(
        out,
        "rebuilt {} from {}",
        database.display(),
        log_dir.display()
    )?;
    writeln!(out, "nodes: {}", report.live_nodes)?;
    writeln!(out, "edges: {}", report.live_edges)?;
    writeln!(
        out,
        "events: {} applied, {} voided, {} quarantined",
        replayed.applied, replayed.skipped_voided, replayed.quarantined
    )?;
    writeln!(
        out,
        "embeddings: {} restored, {} failed, {} pending",
        embedded.re_embedded, embedded.failed, embedded.pending
    )
}

/// Fetches the weights the config asks for, through the daemon's own loaders.
///
/// Downloading is only half of it: each model is also LOADED. A truncated or
/// corrupt file downloads perfectly happily and only fails when something
/// tries to use it, which without this would be the first `recall` a user
/// ever runs, days later, with no obvious connection to the install. Paying
/// a few minutes and some memory once, here, is what makes the guarantee
/// worth stating: if this command succeeds, `liamd` will start.
fn fetch_models(config: &Config, config_path: &Path) -> anyhow::Result<()> {
    println!("config: {}", config_path.display());

    let wants_embedder = config.embedder.provider == "local";
    let wants_llm = config.llm.provider == "llama-cpp";

    if !wants_embedder && !wants_llm {
        // `Config::load` falls back to built-in defaults when the file is
        // absent, and both of those defaults are mock. So a mistyped
        // `--config` arrives here looking exactly like a deliberately mock
        // config. Saying which one it was is the difference between a
        // one-character fix and an afternoon.
        let note = if config_path.exists() {
            String::new()
        } else {
            format!(
                " Note that {} does not exist, so the built-in defaults were used; \
                 check the path if that was not what you intended.",
                config_path.display()
            )
        };
        anyhow::bail!(
            "nothing to fetch: embedder.provider = {:?} and llm.provider = {:?} are both \
             mock providers, and mocks load no weights. Mock embeddings are random and the \
             mock LLM invents answers, so set embedder.provider = \"local\" and \
             llm.provider = \"llama-cpp\" before fetching.{note}",
            config.embedder.provider,
            config.llm.provider
        );
    }

    let home = std::env::var("HOME").unwrap_or_default();

    if wants_embedder {
        let cache_dir = models::resolve_path_with_home(
            "embedder.cache_dir",
            &config.embedder.cache_dir,
            &home,
        )?;
        println!("embedder: {}", config.embedder.model);
        println!("  cache_dir -> {cache_dir}");
        // The same single-threaded requirement `liamd` documents in `main`:
        // fastembed reads this out of the environment, and mutating the
        // environment once other threads exist is a data race on POSIX.
        // Nothing above has spawned one, and this binary starts no runtime.
        std::env::set_var("FASTEMBED_CACHE_DIR", &cache_dir);
        // Pulls the reranker too, which is a second download the daemon
        // needs and nobody would think to ask for by name.
        let (_embedder, _reranker) = models::build_models(config)?;
        println!("embedder: ready, with its reranker");
    }

    if wants_llm {
        let cache_dir =
            models::resolve_path_with_home("llm.cache_dir", &config.llm.cache_dir, &home)?;
        println!(
            "llm: {} ({}) -> {cache_dir}",
            config.llm.model, config.llm.gguf_file
        );
        // Goes through `build_llm`, not the loader underneath it, so the
        // macOS check that a backend actually resolved to Metal runs here as
        // well. A fetch that quietly verified a CPU-only load would promise
        // a start the daemon then refuses.
        let _llm = models::build_llm(config)?;
        println!("llm: ready");
    }

    println!("done: liamd can now start without downloading anything.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use liam_log::dedup::{BloomConfig, HashBloom};
    use liam_log::reader::SequentialScanReader;
    use liam_log::wal::{WalConfig, WalWriter};
    use liam_store::{DefaultGraph, EventLog, GraphConfig, NewNode, SharedLog};

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("arguments must parse")
    }

    #[test]
    fn fetch_models_is_selected_by_its_subcommand() {
        assert_eq!(
            parse(&["liam", "fetch-models"]).command,
            Command::FetchModels
        );
    }

    /// A bare `liam` must not do anything. `liamd` treats no subcommand as
    /// "serve stdio" for backwards compatibility with existing MCP client
    /// configs; the CLI has no such history and no safe default, so it shows
    /// help instead of guessing.
    #[test]
    fn a_bare_invocation_shows_help_rather_than_guessing() {
        let error =
            Cli::try_parse_from(["liam"]).expect_err("a bare invocation must not select an action");
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn a_mistyped_subcommand_is_a_usage_error_with_exit_code_two() {
        let error = Cli::try_parse_from(["liam", "fetch-model"])
            .expect_err("a mistyped subcommand must not parse");
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn version_is_available_and_exits_zero() {
        // Packaging and bug reports both need this, and the install script
        // uses it as the smoke check that the binary runs at all.
        let error = Cli::try_parse_from(["liam", "--version"])
            .expect_err("--version short-circuits parsing");
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
        assert_eq!(error.exit_code(), 0);
        assert!(
            error.to_string().contains(env!("CARGO_PKG_VERSION")),
            "--version must print the crate version, got: {error}"
        );
    }

    /// `--config` is global, so it has to work on the far side of the
    /// subcommand as well as before it.
    #[test]
    fn the_config_flag_is_accepted_after_the_subcommand() {
        let cli = parse(&["liam", "fetch-models", "--config", "/explicit.toml"]);
        assert_eq!(cli.config.as_deref(), Some(Path::new("/explicit.toml")));
    }

    /// Pins that the CLI reads the same file the daemon would. The
    /// precedence itself is tested in `config`; this checks the CLI is
    /// actually wired to it rather than having grown its own copy.
    #[test]
    fn the_config_flag_beats_the_environment() {
        // Before the subcommand this time, which is the other half of the
        // `global = true` contract the test above pins.
        let cli = parse(&["liam", "--config", "/explicit.toml", "fetch-models"]);
        assert_eq!(
            resolve_config_source(cli.config.as_deref(), Some("/from-env.toml")),
            PathBuf::from("/explicit.toml")
        );
    }

    fn rebuild_args(cli: Cli) -> RebuildArgs {
        match cli.command {
            Command::Rebuild(args) => args,
            other => panic!("expected the rebuild subcommand, got {other:?}"),
        }
    }

    #[test]
    fn rebuild_requires_a_database_argument() {
        let error = Cli::try_parse_from(["liam", "rebuild"])
            .expect_err("a rebuild with no database must not parse");
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn rebuild_takes_the_database_and_an_optional_log_directory() {
        // Act
        let bare = rebuild_args(parse(&["liam", "rebuild", "--database", "/data/liam.db"]));
        let with_log = rebuild_args(parse(&[
            "liam",
            "rebuild",
            "--database",
            "/data/liam.db",
            "--log-dir",
            "/elsewhere/log",
        ]));

        // Assert
        assert_eq!(bare.database, PathBuf::from("/data/liam.db"));
        assert_eq!(bare.log_dir, None);
        assert!(!bare.force);
        assert_eq!(with_log.log_dir, Some(PathBuf::from("/elsewhere/log")));
    }

    #[test]
    fn rebuild_defaults_the_log_directory_to_a_sibling_named_after_the_database() {
        assert_eq!(
            default_log_dir(Path::new("/data/liam.db")),
            PathBuf::from("/data/liam.log")
        );
    }

    // ---- running a rebuild against a real store and log ----

    const DIMS: usize = 8;

    fn config() -> Config {
        Config {
            embedding_dims: DIMS,
            ..Config::default()
        }
    }

    /// A store with two nodes and an edge, written through a real log, and the
    /// multiset of live row hashes it holds.
    async fn logged_store(database: &Path, log_dir: &Path) -> Vec<[u8; 32]> {
        let writer = WalWriter::open_with_system_clock(
            log_dir,
            WalConfig {
                segment_max_bytes: 1 << 20,
                rotate_interval_secs: 3600,
            },
        )
        .expect("open the wal");
        let reader = SequentialScanReader::local(log_dir).expect("open the reader");
        let log: SharedLog = std::sync::Arc::new(tokio::sync::Mutex::new(EventLog::new(
            Box::new(writer),
            std::sync::Arc::new(reader),
            HashBloom::new(BloomConfig::default()),
        )));
        let store = DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .expect("open the store")
            .with_log(log)
            .await
            .expect("attach the log");
        let alpha = store
            .insert(NewNode::now("fact", "label", "alpha"))
            .await
            .unwrap();
        let beta = store
            .insert(NewNode::now("fact", "label", "beta"))
            .await
            .unwrap();
        store.relate(&alpha, &beta, "mentions").await.unwrap();
        store.projection_hash_multiset().await.unwrap()
    }

    /// The database's live row hashes, read without a log.
    async fn stored_multiset(database: &Path) -> Vec<[u8; 32]> {
        DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .expect("open the store")
            .projection_hash_multiset()
            .await
            .unwrap()
    }

    /// Deletes the database file and its sidecar files, leaving the log.
    fn delete_database(dir: &Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let is_database = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("liam.db"));
            if is_database && path.is_file() {
                std::fs::remove_file(path).unwrap();
            }
        }
    }

    async fn run_rebuild(args: &RebuildArgs) -> (i32, String, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = rebuild(args, &config(), &mut out, &mut err).await;
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    fn rebuild_of(database: &Path, log_dir: Option<&Path>) -> RebuildArgs {
        RebuildArgs {
            database: database.to_path_buf(),
            log_dir: log_dir.map(Path::to_path_buf),
            force: false,
        }
    }

    #[tokio::test]
    async fn rebuild_after_the_database_is_deleted_restores_it_from_the_log() {
        // Arrange: the log sits at the default place beside the database
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let before = logged_store(&database, &dir.path().join("liam.log")).await;
        delete_database(dir.path());

        // Act
        let (code, out, err) = run_rebuild(&rebuild_of(&database, None)).await;

        // Assert
        assert_eq!(code, 0, "stderr: {err}");
        assert!(out.contains("nodes: 2"), "{out}");
        assert!(out.contains("edges: 1"), "{out}");
        assert_eq!(before.len(), 3, "two nodes and an edge");
        assert_eq!(stored_multiset(&database).await, before);
    }

    #[tokio::test]
    async fn rebuild_exits_non_zero_when_the_rebuilt_hashes_do_not_match_the_store() {
        // Arrange: a row only the store knows about, written without the log
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .unwrap()
            .insert(NewNode::now("fact", "label", "unlogged"))
            .await
            .unwrap();

        // Act
        let (code, _out, err) = run_rebuild(&rebuild_of(&database, Some(&log_dir))).await;

        // Assert
        assert_ne!(code, 0);
        assert!(err.to_lowercase().contains("mismatch"), "{err}");
    }

    #[test]
    fn rebuild_takes_a_force_flag() {
        // Act
        let args = rebuild_args(parse(&[
            "liam",
            "rebuild",
            "--database",
            "/data/liam.db",
            "--force",
        ]));

        // Assert
        assert!(args.force);
    }

    /// A store whose database also holds a row written without the log.
    async fn store_with_an_unlogged_row(dir: &Path) -> (PathBuf, PathBuf, Vec<[u8; 32]>) {
        let (database, log_dir) = (dir.join("liam.db"), dir.join("wal"));
        let logged = logged_store(&database, &log_dir).await;
        DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .unwrap()
            .insert(NewNode::now("fact", "label", "unlogged"))
            .await
            .unwrap();
        (database, log_dir, logged)
    }

    #[tokio::test]
    async fn rebuild_names_the_rows_the_log_does_not_record_and_deletes_nothing_without_force() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir, _) = store_with_an_unlogged_row(dir.path()).await;
        let held = stored_multiset(&database).await;

        // Act
        let (code, out, err) = run_rebuild(&rebuild_of(&database, Some(&log_dir))).await;

        // Assert: one of the four live rows is unlogged, and the database is as it was
        assert_ne!(code, 0);
        assert!(err.contains("1 of the 4 live rows"), "{err}");
        assert!(err.contains("--force"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert_eq!(stored_multiset(&database).await, held);
    }

    #[tokio::test]
    async fn rebuild_with_force_deletes_the_unlogged_rows_and_keeps_what_the_log_holds() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir, logged) = store_with_an_unlogged_row(dir.path()).await;
        let args = RebuildArgs {
            force: true,
            ..rebuild_of(&database, Some(&log_dir))
        };

        // Act
        let (code, out, err) = run_rebuild(&args).await;

        // Assert
        assert_eq!(code, 0, "stderr: {err}");
        assert!(err.contains("deleting 1 of 4"), "{err}");
        assert!(out.contains("nodes: 2"), "{out}");
        assert_eq!(stored_multiset(&database).await, logged);
    }

    #[tokio::test]
    async fn rebuild_warns_that_gc_deletions_are_not_in_the_log() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;

        // Act
        let (code, _out, err) = run_rebuild(&rebuild_of(&database, Some(&log_dir))).await;

        // Assert
        assert_eq!(code, 0, "stderr: {err}");
        assert!(err.contains("GC"), "{err}");
        assert!(err.contains("Back up"), "{err}");
    }

    #[tokio::test]
    async fn rebuild_refuses_a_store_a_live_daemon_holds_and_deletes_nothing() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        let before = logged_store(&database, &log_dir).await;
        let daemon = liam_daemon::storelock::StoreLock::acquire(&database).expect("take the lock");

        // Act
        let (code, out, err) = run_rebuild(&rebuild_of(&database, Some(&log_dir))).await;

        // Assert
        assert_ne!(code, 0);
        assert!(err.contains("lock"), "{err}");
        assert!(out.is_empty(), "{out}");
        drop(daemon);
        assert_eq!(stored_multiset(&database).await, before);
    }

    #[tokio::test]
    async fn rebuild_refuses_a_missing_log_directory_instead_of_rebuilding_from_nothing() {
        // Arrange: a populated store and a log path that does not exist
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        let before = logged_store(&database, &log_dir).await;
        let missing = dir.path().join("no-such-log");

        // Act
        let (code, _out, err) = run_rebuild(&rebuild_of(&database, Some(&missing))).await;

        // Assert: it fails, creates nothing, and leaves the database alone
        assert_ne!(code, 0);
        assert!(err.contains("no-such-log"), "{err}");
        assert!(!missing.exists());
        assert_eq!(stored_multiset(&database).await, before);
    }

    #[tokio::test]
    async fn rebuild_with_force_still_refuses_to_wipe_a_database_for_another_logs_directory() {
        // Arrange: a populated store and the directory of a different, empty log
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        let before = logged_store(&database, &log_dir).await;
        let empty = dir.path().join("empty-log");
        std::fs::create_dir(&empty).unwrap();
        let args = RebuildArgs {
            force: true,
            ..rebuild_of(&database, Some(&empty))
        };

        // Act
        let (code, out, err) = run_rebuild(&args).await;

        // Assert: the replacing rebuild is refused and the database is as it was
        assert_ne!(code, 0);
        assert!(err.contains("belongs to log"), "{err}");
        assert!(err.contains("nothing was changed"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert_eq!(stored_multiset(&database).await, before);
    }
}
