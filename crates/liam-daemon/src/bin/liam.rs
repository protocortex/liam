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
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use async_trait::async_trait;
use clap::{Args, Parser, Subcommand};
use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{WalConfig, WalWriter, MANIFEST_NAME};
use liam_log::LogWriter;
use liam_store::{
    ContentEmbedder, DefaultGraph, EmbedError, Error as StoreError, EventLog, GraphConfig,
    MismatchSource, RebuildMode, RebuildReport, SharedLog,
};

use liam_daemon::config::{resolve_config_source, Config};
use liam_daemon::models::{self, StoreEmbedder};
use liam_daemon::storelock::StoreLock;
use liam_daemon::telemetry;

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
    // Without a subscriber the store's own warnings, such as why a node could
    // not be embedded, are dropped.
    telemetry::init("warn");

    match cli.command {
        Command::FetchModels => fetch_models(&config, &config_path),
        Command::Rebuild(args) => {
            models::export_fastembed_cache_dir(&config)?;
            let runtime = tokio::runtime::Runtime::new()?;
            let code = runtime.block_on(rebuild(
                &args,
                &config,
                &config_path,
                store_embedder,
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

/// GC records what it deletes now, but a sweep made before that is in no log, so
/// a rebuild cannot know it.
const GC_WARNING: &str = "warning: only GC sweeps made before the log recorded deletions are \
     missing from it, so a store that ran GC back then can get the rows those sweeps removed back \
     from the log. Back up the database before relying on a rebuild.";

/// `<database stem>.log` beside the database, until the config names a log
/// directory.
fn default_log_dir(database: &Path) -> PathBuf {
    let stem = database.file_stem().unwrap_or(database.as_os_str());
    let mut name = stem.to_owned();
    name.push(".log");
    database.with_file_name(name)
}

/// Expands `~` in a path flag the way the daemon does for its configured paths.
fn expand_home(flag: &str, path: &Path) -> anyhow::Result<PathBuf> {
    let text = path
        .to_str()
        .with_context(|| format!("{flag} is not valid UTF-8"))?;
    models::resolve_config_path(flag, text)
}

/// Only the embedder: whether the reranker loads has no bearing on a rebuild.
fn store_embedder(config: &Config) -> anyhow::Result<Arc<dyn ContentEmbedder>> {
    Ok(Arc::new(StoreEmbedder(models::build_embedder(config)?)))
}

/// Runs `liam rebuild`: the report goes to `out`, a refusal or a failure to
/// `err`. Returns the process exit code, non zero on any failure.
/// `embedder_for` builds the embedder once the store lock is held.
async fn rebuild(
    args: &RebuildArgs,
    config: &Config,
    config_path: &Path,
    embedder_for: impl FnOnce(&Config) -> anyhow::Result<Arc<dyn ContentEmbedder>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    match try_rebuild(args, config, config_path, embedder_for, out, err).await {
        Ok(()) => 0,
        Err(error) => {
            let _ = writeln!(err, "liam rebuild: {error:#}");
            1
        }
    }
}

/// The steps in order: every check that needs no change comes first, the store
/// lock is ours before the log or the database is opened, and the embedder is
/// built before the store deletes anything. A missing config, a path that is not
/// a log and a log with no records are refused here because each would otherwise
/// rebuild an empty or wrongly embedded store and report success.
async fn try_rebuild(
    args: &RebuildArgs,
    config: &Config,
    config_path: &Path,
    embedder_for: impl FnOnce(&Config) -> anyhow::Result<Arc<dyn ContentEmbedder>>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        config_path.exists(),
        "the config file {} does not exist, so the built-in defaults would embed every \
         restored node with the mock embedder; pass --config or set LIAM_CONFIG",
        config_path.display()
    );
    let database = expand_home("--database", &args.database)?;
    let log_dir = match &args.log_dir {
        Some(dir) => expand_home("--log-dir", dir)?,
        None => default_log_dir(&database),
    };
    anyhow::ensure!(
        log_dir.join(MANIFEST_NAME).is_file(),
        "{} is not an event log, it has no {MANIFEST_NAME} file; pass --log-dir if the log \
         lives elsewhere",
        log_dir.display()
    );
    let _lock = StoreLock::acquire(&database)
        .context("liam rebuild needs the store to itself: stop liamd, then rerun")?;
    tracing::debug!("rebuild: store lock taken");

    let embedder = Arc::new(FirstFailure::new(embedder_for(config).context(
        "the embedder could not be built: fix the model setup or run `liam fetch-models`, then \
         rerun. Nothing was changed",
    )?));
    let writer = open_writer(&log_dir)?;
    // An empty log would rebuild an empty store and report success, which is
    // not what a lost database needs.
    anyhow::ensure!(
        writer.head().is_some(),
        "the log in {} holds no records, so there is nothing to rebuild from",
        log_dir.display()
    );
    let log = shared_log(writer, &log_dir)?;
    let report = replay_log(
        &database,
        config.embedding_dims,
        log,
        embedder.clone(),
        args.force,
        err,
    )
    .await?;
    print_report(
        out,
        config_path,
        &config.embedder.provider,
        &database,
        &log_dir,
        &report,
    )?;

    let failed = report.replayed.reembedded.failed;
    anyhow::ensure!(
        failed == 0,
        "{failed} node embeddings failed ({}); the rebuild is incomplete, rerun it once the \
         cause is fixed",
        embedder
            .first_failure()
            .unwrap_or("reasons are logged above")
    );
    Ok(())
}

fn open_writer(log_dir: &Path) -> anyhow::Result<impl LogWriter + 'static> {
    WalWriter::open_with_system_clock(log_dir, REBUILD_WAL)
        .with_context(|| format!("could not open the log in {}", log_dir.display()))
}

fn shared_log(writer: impl LogWriter + 'static, log_dir: &Path) -> anyhow::Result<SharedLog> {
    let reader = SequentialScanReader::local(log_dir)
        .with_context(|| format!("could not read the log in {}", log_dir.display()))?;
    let log = EventLog::new(
        Box::new(writer),
        Arc::new(reader),
        HashBloom::new(BloomConfig::default()),
    );
    Ok(Arc::new(tokio::sync::Mutex::new(log)))
}

/// Remembers the first embed failure, because the store only counts them.
struct FirstFailure {
    inner: Arc<dyn ContentEmbedder>,
    first: OnceLock<String>,
}

impl FirstFailure {
    fn new(inner: Arc<dyn ContentEmbedder>) -> Self {
        Self {
            inner,
            first: OnceLock::new(),
        }
    }

    fn first_failure(&self) -> Option<&str> {
        self.first.get().map(String::as_str)
    }
}

#[async_trait]
impl ContentEmbedder for FirstFailure {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let embedded = self.inner.embed(text).await;
        if let Err(error) = &embedded {
            let _ = self.first.set(error.to_string());
        }
        embedded
    }
}

/// Rebuilds the database from `log`. `--force` only widens the mode after the
/// store refused the safe one, so a refusal is never skipped unseen.
async fn replay_log(
    database: &Path,
    dims: usize,
    log: SharedLog,
    embedder: Arc<dyn ContentEmbedder>,
    force: bool,
    err: &mut dyn Write,
) -> anyhow::Result<RebuildReport> {
    let graph = open_graph(database, dims, &embedder).await?;
    let mode = if graph.projection_hash_multiset().await?.is_empty() {
        RebuildMode::RequireEmpty
    } else {
        RebuildMode::ResetChecked
    };

    writeln!(err, "{GC_WARNING}")?;
    let rebuilt = match graph.rebuild_from_log(log.clone(), mode).await {
        Err(refusal) if force && is_forceable(&refusal) => {
            writeln!(err, "{}", force_warning(&refusal))?;
            // The refused store was consumed, so the replacing run opens its own.
            open_graph(database, dims, &embedder)
                .await?
                .rebuild_from_log(log, RebuildMode::Replace)
                .await
        }
        other => other,
    };
    let (_graph, report) = rebuilt.map_err(explain_refusal)?;
    tracing::debug!("rebuild: done");
    Ok(report)
}

/// The database with no log attached: `rebuild_from_log` attaches it only once
/// the rebuild has been checked.
async fn open_graph(
    database: &Path,
    dims: usize,
    embedder: &Arc<dyn ContentEmbedder>,
) -> anyhow::Result<DefaultGraph> {
    let graph = DefaultGraph::open(
        database
            .to_str()
            .context("the database path is not UTF-8")?,
        GraphConfig::new(dims),
    )
    .await?;
    Ok(graph.with_embedder(Arc::clone(embedder)))
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

/// Puts a refusal or failure of the store in terms of the command line. Only
/// the errors that can arise before the projection is cleared claim that
/// nothing was changed; a backend or log read failure may come from either side
/// of it, so it cannot.
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
        | StoreError::ForeignLogWouldWipe { .. }
        | StoreError::CorruptLogState(_)) => {
            anyhow::Error::new(refused).context("the rebuild was refused and nothing was changed")
        }
        // Raised after the replay, so the database holds its result.
        unverified @ StoreError::RebuildMismatch {
            against: MismatchSource::Log,
            ..
        } => anyhow::Error::new(unverified)
            .context("the rebuild finished but its result does not match the log"),
        other => anyhow::Error::new(other).context(
            "the rebuild did not finish cleanly and the database may hold a partial rebuild; \
             run it again with --force to start over from the log, or restore a backup",
        ),
    }
}

fn print_report(
    out: &mut dyn Write,
    config_path: &Path,
    provider: &str,
    database: &Path,
    log_dir: &Path,
    report: &RebuildReport,
) -> std::io::Result<()> {
    let (replayed, embedded) = (&report.replayed, &report.replayed.reembedded);
    writeln!(out, "config: {}", config_path.display())?;
    writeln!(out, "embedder: {provider}")?;
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
        println!("embedder: {}", config.embedder.model);
        if let Some(cache_dir) = models::export_fastembed_cache_dir(config)? {
            println!("  cache_dir -> {}", cache_dir.display());
        }
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

    use std::sync::atomic::{AtomicUsize, Ordering};

    use liam_store::{Backend, DefaultBackend, NewNode, FOREVER};

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
    fn the_default_log_directory_is_a_sibling_named_after_the_database_stem() {
        for (database, log_dir) in [
            ("/data/work.db", "/data/work.log"),
            ("liam.db", "liam.log"),
            ("/data/store", "/data/store.log"),
        ] {
            assert_eq!(
                default_log_dir(Path::new(database)),
                PathBuf::from(log_dir),
                "{database}"
            );
        }
    }

    #[test]
    fn a_mistyped_rebuild_flag_is_a_usage_error_with_exit_code_two() {
        for flag in ["--logdir", "--forse"] {
            let error = Cli::try_parse_from(["liam", "rebuild", "--database", "liam.db", flag])
                .expect_err("an unknown flag must not parse");
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::UnknownArgument,
                "{flag}"
            );
            assert_eq!(error.exit_code(), 2, "{flag}");
        }
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

    // ---- running a rebuild against a real store and log ----

    const DIMS: usize = 8;

    fn config() -> Config {
        Config {
            embedding_dims: DIMS,
            ..Config::default()
        }
    }

    fn log_at(log_dir: &Path) -> SharedLog {
        shared_log(open_writer(log_dir).unwrap(), log_dir).unwrap()
    }

    /// A store written through a real log: two nodes, an edge, and a supersede
    /// of the first node, which closes it and adds a `supersedes` edge.
    async fn logged_store(database: &Path, log_dir: &Path) {
        let store = DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .expect("open the store")
            .with_log(log_at(log_dir))
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
        store
            .supersede(&alpha, NewNode::now("fact", "label", "alpha two"))
            .await
            .unwrap();
    }

    /// Every column of every node and edge row, in id order, so a restore that
    /// differs in anything but the hashed fields shows.
    async fn dump(database: &Path) -> String {
        let backend = DefaultBackend::open(database.to_str().unwrap(), 1)
            .await
            .expect("open the database");
        let mut dump = String::new();
        for table in ["nodes", "edges"] {
            let rows = backend
                .query(&format!("SELECT * FROM {table} ORDER BY id"), &[])
                .await
                .unwrap();
            dump.push_str(&format!("{rows:?}\n"));
        }
        dump
    }

    /// How many live nodes have no vector.
    async fn nodes_without_vectors(database: &Path) -> usize {
        DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .expect("open the store")
            .reembed_missing()
            .await
            .unwrap()
            .pending
    }

    async fn insert_without_the_log(database: &Path, content: &str) {
        DefaultGraph::open(database.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .unwrap()
            .insert(NewNode::now("fact", "label", content))
            .await
            .unwrap();
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

    /// `rebuild` needs a config file that exists; its content is not read here.
    fn config_file(dir: &Path) -> PathBuf {
        let path = dir.join("liam.toml");
        std::fs::write(&path, "").unwrap();
        path
    }

    async fn run_with(
        args: &RebuildArgs,
        config_path: &Path,
        embedder_for: impl FnOnce(&Config) -> anyhow::Result<Arc<dyn ContentEmbedder>>,
    ) -> (i32, String, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = rebuild(
            args,
            &config(),
            config_path,
            embedder_for,
            &mut out,
            &mut err,
        )
        .await;
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    /// A rebuild with the embedder the config asks for, configured in `dir`.
    async fn run_rebuild(dir: &Path, args: &RebuildArgs) -> (i32, String, String) {
        run_with(args, &config_file(dir), store_embedder).await
    }

    fn rebuild_of(database: &Path, log_dir: Option<&Path>) -> RebuildArgs {
        RebuildArgs {
            database: database.to_path_buf(),
            log_dir: log_dir.map(Path::to_path_buf),
            force: false,
        }
    }

    fn assert_lock_released(database: &Path) {
        StoreLock::acquire(database).expect("the rebuild must release the store lock");
    }

    #[tokio::test]
    async fn rebuild_after_the_database_is_deleted_restores_every_row_from_the_log() {
        // Arrange: the log sits at the default place beside the database
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        logged_store(&database, &dir.path().join("liam.log")).await;
        let before = dump(&database).await;
        delete_database(dir.path());

        // Act
        let (code, out, err) = run_rebuild(dir.path(), &rebuild_of(&database, None)).await;

        // Assert: the report says what was done and every column is back
        assert_eq!(code, 0, "stderr: {err}");
        let lines: Vec<_> = out.lines().collect();
        assert_eq!(
            lines[0],
            format!("config: {}", dir.path().join("liam.toml").display())
        );
        assert_eq!(lines[1], "embedder: mock");
        assert!(lines[2].starts_with("rebuilt "), "{out}");
        assert_eq!(
            lines[3..],
            [
                "nodes: 2",
                "edges: 2",
                "events: 4 applied, 0 voided, 0 quarantined",
                "embeddings: 2 restored, 0 failed, 0 pending",
            ],
            "{out}"
        );
        assert_eq!(dump(&database).await, before);
        assert_eq!(nodes_without_vectors(&database).await, 0);
        assert_lock_released(&database);
    }

    #[tokio::test]
    async fn rebuild_of_a_caught_up_store_changes_nothing_and_repeats_identically() {
        // Arrange: a store that is exactly what its log says
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let before = dump(&database).await;
        let args = rebuild_of(&database, Some(&log_dir));

        // Act
        let (first_code, first_out, first_err) = run_rebuild(dir.path(), &args).await;
        let after_first = dump(&database).await;
        let (second_code, second_out, _) = run_rebuild(dir.path(), &args).await;

        // Assert
        assert_eq!(first_code, 0, "stderr: {first_err}");
        assert_eq!(after_first, before);
        assert_eq!(second_code, 0);
        assert_eq!(second_out, first_out);
        assert_eq!(dump(&database).await, before);
    }

    #[tokio::test]
    async fn rebuild_names_the_rows_the_log_does_not_record_and_deletes_nothing_without_force() {
        // Arrange: one of the live rows is unlogged
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        insert_without_the_log(&database, "unlogged").await;
        let held = dump(&database).await;

        // Act
        let (code, out, err) =
            run_rebuild(dir.path(), &rebuild_of(&database, Some(&log_dir))).await;

        // Assert: refused, said so, and the database is as it was
        assert_ne!(code, 0);
        assert!(err.to_lowercase().contains("mismatch"), "{err}");
        assert!(err.contains("1 of the 5 live rows"), "{err}");
        assert!(err.contains("--force"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert_eq!(dump(&database).await, held);
        assert_lock_released(&database);
    }

    #[tokio::test]
    async fn rebuild_with_force_deletes_the_unlogged_rows_and_keeps_what_the_log_holds() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let logged = dump(&database).await;
        insert_without_the_log(&database, "unlogged").await;
        let args = RebuildArgs {
            force: true,
            ..rebuild_of(&database, Some(&log_dir))
        };

        // Act
        let (code, out, err) = run_rebuild(dir.path(), &args).await;

        // Assert
        assert_eq!(code, 0, "stderr: {err}");
        assert!(err.contains("deleting 1 of 5"), "{err}");
        assert!(out.contains("nodes: 2"), "{out}");
        assert_eq!(dump(&database).await, logged);
    }

    /// A store none of whose rows is live, which the log may not fully record.
    async fn store_with_only_superseded_rows(dir: &Path) -> (PathBuf, PathBuf, String) {
        let (database, log_dir) = (dir.join("liam.db"), dir.join("wal"));
        logged_store(&database, &log_dir).await;
        let backend = DefaultBackend::open(database.to_str().unwrap(), 1)
            .await
            .unwrap();
        for table in ["nodes", "edges"] {
            backend
                .execute(
                    &format!("UPDATE {table} SET tx_to = 5 WHERE tx_to = {}", FOREVER.0),
                    &[],
                )
                .await
                .unwrap();
        }
        let held = dump(&database).await;
        (database, log_dir, held)
    }

    #[tokio::test]
    async fn rebuild_refuses_a_store_with_only_superseded_rows_without_force() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir, held) = store_with_only_superseded_rows(dir.path()).await;

        // Act
        let (code, out, err) =
            run_rebuild(dir.path(), &rebuild_of(&database, Some(&log_dir))).await;

        // Assert
        assert_ne!(code, 0);
        assert!(err.contains("superseded"), "{err}");
        assert!(err.contains("Nothing was changed"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert_eq!(dump(&database).await, held);
    }

    #[tokio::test]
    async fn rebuild_with_force_replaces_the_superseded_rows_from_the_log() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir, held) = store_with_only_superseded_rows(dir.path()).await;
        let args = RebuildArgs {
            force: true,
            ..rebuild_of(&database, Some(&log_dir))
        };

        // Act
        let (code, out, err) = run_rebuild(dir.path(), &args).await;

        // Assert
        assert_eq!(code, 0, "stderr: {err}");
        assert!(err.contains("replacing superseded rows"), "{err}");
        assert!(out.contains("nodes: 2"), "{out}");
        assert_ne!(dump(&database).await, held);
    }

    #[test]
    fn a_superseded_only_store_is_forceable_and_explained_in_its_own_terms() {
        assert!(is_forceable(&StoreError::ProjectionNotEmpty));
        assert!(force_warning(&StoreError::ProjectionNotEmpty).contains("superseded"));
        assert!(explain_refusal(StoreError::ProjectionNotEmpty)
            .to_string()
            .contains("superseded"));
    }

    #[tokio::test]
    async fn rebuild_warns_that_gc_deletions_made_before_logging_are_not_in_the_log() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;

        // Act
        let (code, _out, err) =
            run_rebuild(dir.path(), &rebuild_of(&database, Some(&log_dir))).await;

        // Assert
        assert_eq!(code, 0, "stderr: {err}");
        assert!(
            err.contains("GC sweeps made before the log recorded"),
            "{err}"
        );
        assert!(err.contains("Back up"), "{err}");
    }

    #[tokio::test]
    async fn rebuild_refuses_a_store_another_process_holds_and_deletes_nothing() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let before = dump(&database).await;
        let holder = StoreLock::acquire(&database).expect("take the lock");

        // Act
        let (code, out, err) =
            run_rebuild(dir.path(), &rebuild_of(&database, Some(&log_dir))).await;

        // Assert
        assert_ne!(code, 0);
        assert!(err.contains("lock"), "{err}");
        assert!(err.contains("stop liamd, then rerun"), "{err}");
        assert!(out.is_empty(), "{out}");
        drop(holder);
        assert_eq!(dump(&database).await, before);
    }

    /// Records, while it embeds, whether the store lock was still free.
    struct LockProbe {
        database: PathBuf,
        embedded: AtomicUsize,
        lock_was_free: AtomicUsize,
    }

    #[async_trait]
    impl ContentEmbedder for LockProbe {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
            self.embedded.fetch_add(1, Ordering::SeqCst);
            if StoreLock::acquire(&self.database).is_ok() {
                self.lock_was_free.fetch_add(1, Ordering::SeqCst);
            }
            Ok(vec![0.5; DIMS])
        }
    }

    #[tokio::test]
    async fn rebuild_holds_the_store_lock_while_it_embeds() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let probe = Arc::new(LockProbe {
            database: database.clone(),
            embedded: AtomicUsize::new(0),
            lock_was_free: AtomicUsize::new(0),
        });
        let injected = Arc::clone(&probe);

        // Act
        let (code, _out, err) = run_with(
            &rebuild_of(&database, Some(&log_dir)),
            &config_file(dir.path()),
            move |_| Ok(injected),
        )
        .await;

        // Assert
        assert_eq!(code, 0, "stderr: {err}");
        assert_eq!(probe.embedded.load(Ordering::SeqCst), 2);
        assert_eq!(probe.lock_was_free.load(Ordering::SeqCst), 0);
    }

    struct FailingEmbedder;

    #[async_trait]
    impl ContentEmbedder for FailingEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
            Err("no model loaded".into())
        }
    }

    #[tokio::test]
    async fn rebuild_exits_non_zero_and_names_the_first_reason_when_embeddings_fail() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;

        // Act
        let (code, out, err) = run_with(
            &rebuild_of(&database, Some(&log_dir)),
            &config_file(dir.path()),
            |_| Ok(Arc::new(FailingEmbedder)),
        )
        .await;

        // Assert: the rows are restored and reported, and the run says it is incomplete
        assert_eq!(code, 1);
        assert!(
            out.contains("embeddings: 0 restored, 2 failed, 0 pending"),
            "{out}"
        );
        assert!(
            err.contains("2 node embeddings failed (no model loaded)"),
            "{err}"
        );
        assert!(err.contains("rerun"), "{err}");
    }

    #[tokio::test]
    async fn rebuild_refuses_before_any_change_when_the_embedder_cannot_be_built() {
        // Arrange: a store that already has its vectors
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let args = rebuild_of(&database, Some(&log_dir));
        assert_eq!(run_rebuild(dir.path(), &args).await.0, 0);
        let before = dump(&database).await;

        // Act
        let (code, out, err) = run_with(&args, &config_file(dir.path()), |_| {
            Err(anyhow::anyhow!("model files are missing"))
        })
        .await;

        // Assert: nothing was deleted, vectors included
        assert_ne!(code, 0);
        assert!(err.contains("model files are missing"), "{err}");
        assert!(err.contains("liam fetch-models"), "{err}");
        assert!(err.contains("Nothing was changed"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert_eq!(dump(&database).await, before);
        assert_eq!(nodes_without_vectors(&database).await, 0);
    }

    #[cfg(not(feature = "local"))]
    #[test]
    fn a_local_embedder_this_build_lacks_is_an_error_not_a_mock() {
        let mut config = config();
        config.embedder.provider = "local".to_string();

        assert!(store_embedder(&config).is_err());
        assert!(store_embedder(&Config::default()).is_ok());
    }

    #[tokio::test]
    async fn rebuild_refuses_a_missing_config_file_before_it_touches_anything() {
        // Arrange: a config path that does not exist, so the built-in mock
        // embedder would be used
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let before = dump(&database).await;
        let missing = dir.path().join("no-such.toml");

        // Act
        let (code, out, err) = run_with(
            &rebuild_of(&database, Some(&log_dir)),
            &missing,
            store_embedder,
        )
        .await;

        // Assert
        assert_ne!(code, 0);
        assert!(err.contains("no-such.toml"), "{err}");
        assert!(err.contains("--config"), "{err}");
        assert!(err.contains("LIAM_CONFIG"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert!(!dir.path().join("liam.db.lock").exists());
        assert_eq!(dump(&database).await, before);
    }

    #[tokio::test]
    async fn rebuild_refuses_a_missing_log_directory_and_creates_nothing() {
        // Arrange: a database path in a directory that does not exist either
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("fresh").join("liam.db");
        let missing = dir.path().join("no-such-log");

        // Act
        let (code, _out, err) =
            run_rebuild(dir.path(), &rebuild_of(&database, Some(&missing))).await;

        // Assert
        assert_ne!(code, 0);
        assert!(err.contains("no-such-log"), "{err}");
        assert!(!missing.exists());
        assert!(!dir.path().join("fresh").exists());
    }

    #[tokio::test]
    async fn rebuild_refuses_a_directory_that_is_not_a_log_and_leaves_it_as_it_was() {
        // Arrange: a directory with something in it and no log manifest
        let dir = tempfile::tempdir().unwrap();
        let (database, plain) = (dir.path().join("liam.db"), dir.path().join("plain"));
        std::fs::create_dir(&plain).unwrap();
        std::fs::write(plain.join("notes.txt"), "keep").unwrap();

        // Act
        let (code, _out, err) = run_rebuild(dir.path(), &rebuild_of(&database, Some(&plain))).await;

        // Assert: refused, and neither the directory nor the database was created into
        assert_ne!(code, 0);
        assert!(err.contains("not an event log"), "{err}");
        let names: Vec<_> = std::fs::read_dir(&plain)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, ["notes.txt"]);
        assert!(!database.exists());
    }

    #[tokio::test]
    async fn rebuild_refuses_a_log_with_no_records_instead_of_rebuilding_an_empty_store() {
        // Arrange: a log that was opened and never written to
        let dir = tempfile::tempdir().unwrap();
        let (database, empty) = (dir.path().join("liam.db"), dir.path().join("empty-log"));
        drop(log_at(&empty));

        // Act
        let (code, out, err) = run_rebuild(dir.path(), &rebuild_of(&database, Some(&empty))).await;

        // Assert
        assert_ne!(code, 0);
        assert!(err.contains("no records"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert!(!database.exists());
    }

    #[tokio::test]
    async fn rebuild_with_the_store_locked_and_no_database_creates_no_database() {
        // Arrange: a log to rebuild from, a database that does not exist, and a lock on it
        let dir = tempfile::tempdir().unwrap();
        let log_dir = dir.path().join("wal");
        logged_store(&dir.path().join("other.db"), &log_dir).await;
        let database = dir.path().join("liam.db");
        let _holder = StoreLock::acquire(&database).expect("take the lock");

        // Act
        let (code, _out, _err) =
            run_rebuild(dir.path(), &rebuild_of(&database, Some(&log_dir))).await;

        // Assert
        assert_ne!(code, 0);
        assert!(!database.exists());
    }

    /// The command line refuses an empty log before the store sees it, so this
    /// drives the core to pin what the store's own refusal turns into.
    #[tokio::test]
    async fn a_forced_rebuild_from_an_empty_log_deletes_nothing() {
        // Arrange: rows no log records, and a log with no records
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        insert_without_the_log(&database, "unlogged").await;
        let before = dump(&database).await;
        let empty = log_at(&dir.path().join("empty-log"));

        // Act
        let result = replay_log(
            &database,
            DIMS,
            empty,
            store_embedder(&config()).unwrap(),
            true,
            &mut Vec::new(),
        )
        .await;

        // Assert
        let message = format!("{:#}", result.expect_err("the wipe must be refused"));
        assert!(message.contains("nothing was changed"), "{message}");
        assert!(message.contains("log is empty"), "{message}");
        assert_eq!(dump(&database).await, before);
    }

    #[tokio::test]
    async fn rebuild_with_force_still_refuses_to_wipe_a_database_for_another_log() {
        // Arrange: a populated store and the directory of a different, populated log
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let before = dump(&database).await;
        let other = dir.path().join("other-log");
        logged_store(&dir.path().join("other.db"), &other).await;
        let args = RebuildArgs {
            force: true,
            ..rebuild_of(&database, Some(&other))
        };

        // Act
        let (code, out, err) = run_rebuild(dir.path(), &args).await;

        // Assert: the replacing rebuild is refused and the database is as it was
        assert_ne!(code, 0);
        assert!(err.contains("belongs to log"), "{err}");
        assert!(err.contains("nothing was changed"), "{err}");
        assert!(out.is_empty(), "{out}");
        assert_eq!(dump(&database).await, before);
    }

    #[test]
    fn only_a_failure_after_the_reset_says_the_database_may_hold_a_partial_rebuild() {
        // Arrange
        let before_the_reset = [
            StoreError::EmptyLogWouldWipe {
                rows_unknown_to_log: 1,
            },
            StoreError::CorruptLogState("log_cursor holds the offset -1".into()),
        ];

        // Act
        let after_replay = format!(
            "{:#}",
            explain_refusal(StoreError::Backend("disk full".into()))
        );
        let unverified = format!(
            "{:#}",
            explain_refusal(StoreError::RebuildMismatch {
                against: MismatchSource::Log,
                expected: 3,
                found: 2,
                missing: 1,
                unexpected: 0,
            })
        );

        // Assert
        for refused in before_the_reset {
            let message = format!("{:#}", explain_refusal(refused));
            assert!(message.contains("nothing was changed"), "{message}");
            assert!(!message.contains("partial"), "{message}");
        }
        assert!(after_replay.contains("partial rebuild"), "{after_replay}");
        assert!(
            unverified.contains("restore the missing log segments"),
            "{unverified}"
        );
        assert!(!unverified.contains("nothing was changed"), "{unverified}");
        assert!(!unverified.contains("--force"), "{unverified}");
    }

    /// A torn final record was never acknowledged, so the log's own open drops it
    /// and the command has nothing to report: no quarantine, no failure.
    #[tokio::test]
    async fn rebuild_drops_a_torn_last_record_without_reporting_it() {
        // Arrange: the last record of the last segment is cut short
        let dir = tempfile::tempdir().unwrap();
        let (database, log_dir) = (dir.path().join("liam.db"), dir.path().join("wal"));
        logged_store(&database, &log_dir).await;
        let segment = log_dir.join("00000000000000000000.wal");
        let length = segment.metadata().unwrap().len();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&segment)
            .unwrap();
        file.set_len(length - 5).unwrap();
        delete_database(dir.path());

        // Act
        let (code, out, _err) =
            run_rebuild(dir.path(), &rebuild_of(&database, Some(&log_dir))).await;

        // Assert: the supersede in the torn record is gone, and the run says nothing of it
        assert_eq!(code, 0);
        assert!(out.contains("nodes: 2\nedges: 1\n"), "{out}");
        assert!(
            out.contains("events: 3 applied, 0 voided, 0 quarantined"),
            "{out}"
        );
    }
}
