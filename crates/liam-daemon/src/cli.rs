// SPDX-License-Identifier: AGPL-3.0-only
//! Argument parsing, and the mode it selects.
//!
//! Kept in its own module, with the mode as a plain enum, so `main` stays a
//! thin dispatch and the argument-to-mode mapping is unit-testable without
//! spawning a process.
//!
//! Why clap rather than matching on `argv` by hand: the argument decides
//! whether this process opens the store at all. `serve` and the default
//! stdio mode do; `proxy` must not, because a proxy running alongside the
//! daemon that owns the database would trip the per-process store lock. A
//! hand-rolled match would let `liamd serv` fall through to the
//! store-opening default, so a typo would take the lock and fail the real
//! daemon's next start. clap turns that into a usage error and exit code 2
//! for free.
//!
//! clap does arrive transitively through leiden-rs, and always will now that
//! ADR-0002 made clustering unconditional. The direct dependency stays anyway:
//! a transitive edge can disappear on any upgrade of a crate that has no
//! obligation to keep it, and this binary's argument parsing is not something
//! to have break from someone else's minor release.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use liam_daemon::config::resolve_config_source;

#[derive(Debug, Parser)]
#[command(
    name = "liamd",
    version,
    about = "LIAM memory daemon: serves durable agent memory over MCP.",
    long_about = None
)]
pub struct Cli {
    /// Path to liam.toml. Overrides the LIAM_CONFIG environment variable.
    #[arg(long, value_name = "PATH", global = true)]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve MCP to many local clients over the Unix socket.
    Serve,
    /// Forward this process's stdio to a running daemon's socket.
    Proxy,
}

/// What this process should do. `Stdio` is the no-subcommand default, kept
/// first and kept the default so existing MCP client configs that just run
/// `liamd` behave exactly as they did before modes existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// One client, one process, MCP over this process's stdin and stdout.
    Stdio,
    /// The socket daemon: many clients, one store.
    Serve,
    /// A shuttle between this process's stdio and the daemon's socket.
    /// Opens no store and loads no model.
    Proxy,
}

impl Cli {
    /// Which mode the parsed arguments select.
    pub fn mode(&self) -> Mode {
        match self.command {
            None => Mode::Stdio,
            Some(Command::Serve) => Mode::Serve,
            Some(Command::Proxy) => Mode::Proxy,
        }
    }

    /// Where to read config from: `--config`, else `LIAM_CONFIG`, else
    /// `liam.toml`.
    ///
    /// Delegates so `liamd` and `liam` cannot drift apart on which file they
    /// read; the tests below stay here because they pin this binary's
    /// behaviour, which is what an existing MCP client config depends on.
    pub fn config_path(&self, env_config: Option<&str>) -> PathBuf {
        resolve_config_source(self.config.as_deref(), env_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_daemon::config::DEFAULT_CONFIG;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("arguments must parse")
    }

    #[test]
    fn no_subcommand_is_the_stdio_server() {
        // Given a bare invocation, which is what every existing MCP client
        // config runs
        // Then the mode is unchanged from before modes existed
        assert_eq!(parse(&["liamd"]).mode(), Mode::Stdio);
    }

    #[test]
    fn serve_selects_the_socket_daemon() {
        assert_eq!(parse(&["liamd", "serve"]).mode(), Mode::Serve);
    }

    #[test]
    fn proxy_selects_the_shuttle() {
        assert_eq!(parse(&["liamd", "proxy"]).mode(), Mode::Proxy);
    }

    #[test]
    fn a_mistyped_subcommand_is_a_usage_error_with_exit_code_two() {
        // Given a typo one character away from a real subcommand
        let error = Cli::try_parse_from(["liamd", "serv"])
            .expect_err("a mistyped subcommand must not parse");

        // Then it is a usage error that exits 2, never a silent fall
        // through to the store-opening stdio default. Falling through would
        // take the per-process store lock and break the real daemon's next
        // start.
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn an_unknown_flag_is_a_usage_error_with_exit_code_two() {
        let error =
            Cli::try_parse_from(["liamd", "--nope"]).expect_err("an unknown flag must not parse");
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn version_is_available_and_exits_zero() {
        // Given `--version`, which packaging and bug reports both need
        let error = Cli::try_parse_from(["liamd", "--version"])
            .expect_err("--version short-circuits parsing");

        // Then clap reports it as a successful exit carrying the crate
        // version, not a usage failure.
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
        assert_eq!(error.exit_code(), 0);
        assert!(
            error.to_string().contains(env!("CARGO_PKG_VERSION")),
            "--version must print the crate version, got: {error}"
        );
    }

    #[test]
    fn the_config_flag_beats_the_environment() {
        // Given both --config and LIAM_CONFIG
        let cli = parse(&["liamd", "--config", "/explicit.toml"]);

        // Then the flag wins
        assert_eq!(
            cli.config_path(Some("/from-env.toml")),
            PathBuf::from("/explicit.toml")
        );
    }

    #[test]
    fn the_environment_is_used_when_the_flag_is_absent() {
        assert_eq!(
            parse(&["liamd"]).config_path(Some("/from-env.toml")),
            PathBuf::from("/from-env.toml")
        );
    }

    #[test]
    fn the_default_config_is_used_when_neither_is_set() {
        assert_eq!(
            parse(&["liamd"]).config_path(None),
            PathBuf::from(DEFAULT_CONFIG)
        );
    }

    #[test]
    fn an_empty_environment_value_falls_back_to_the_default() {
        // Given LIAM_CONFIG set but blank, which is what an unset shell
        // variable expands to in a launchd plist or a wrapper script
        // Then it is treated as absent rather than as a path to ""
        assert_eq!(
            parse(&["liamd"]).config_path(Some("")),
            PathBuf::from(DEFAULT_CONFIG)
        );
    }

    #[test]
    fn the_config_flag_works_alongside_a_subcommand() {
        // Given --config after a subcommand, which `global = true` allows
        let cli = parse(&["liamd", "serve", "--config", "/explicit.toml"]);

        // Then both are honoured
        assert_eq!(cli.mode(), Mode::Serve);
        assert_eq!(cli.config_path(None), PathBuf::from("/explicit.toml"));
    }
}
