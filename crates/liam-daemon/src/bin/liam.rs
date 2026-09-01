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

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use liam_daemon::config::{resolve_config_source, Config};
use liam_daemon::models;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Subcommand)]
enum Command {
    /// Download and load every model liam.toml asks for, so the first daemon
    /// start is not also the first download.
    FetchModels,
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
    }
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
}
