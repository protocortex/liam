// SPDX-License-Identifier: Apache-2.0
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

use clap::{ArgGroup, Args, Parser, Subcommand};

use liam_daemon::config::resolve_config_source;

use crate::deletion::{DeleteAction, DeleteTarget};

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
    /// Seal the plaintext events of the log in place. Needs `[encryption]` and
    /// a stopped daemon.
    EncryptLog,
    /// Count what a delete would destroy and run it, or finish, cancel or list
    /// delete jobs. Needs a stopped daemon, except for `--status`.
    Delete(DeleteCommand),
}

/// The flags of `liamd delete`: exactly one of the five that choose what to do.
#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("action")
        .required(true)
        .args(["subject", "event_id", "resume", "cancel", "status"])
))]
pub struct DeleteCommand {
    /// Delete everything held about this subject.
    #[arg(long)]
    subject: Option<String>,
    /// Only the subject in this scope.
    #[arg(long, conflicts_with_all = ["event_id", "resume", "cancel", "status"])]
    scope: Option<String>,
    /// Also the subject in every scope below `--scope`.
    #[arg(long, conflicts_with_all = ["event_id", "resume", "cancel", "status"])]
    include_children: bool,
    /// Delete one logged event instead of a subject.
    #[arg(long, value_name = "EVENT_ID")]
    event_id: Option<String>,
    /// Start the job. Without it only the counts are printed.
    #[arg(long, conflicts_with_all = ["resume", "cancel", "status"])]
    yes: bool,
    /// Finish job ID, or every open job when no ID is given.
    #[arg(long, value_name = "ID", num_args = 0..=1)]
    resume: Option<Option<String>>,
    /// Cancel job ID, which must not have destroyed a key yet.
    #[arg(long, value_name = "ID")]
    cancel: Option<String>,
    /// List the delete jobs by id and state.
    #[arg(long)]
    status: bool,
}

impl DeleteCommand {
    fn action(&self) -> DeleteAction {
        if let Some(subject) = &self.subject {
            return DeleteAction::Request {
                target: DeleteTarget::Subject {
                    subject: subject.clone(),
                    scope: self.scope.clone(),
                    include_children: self.include_children,
                },
                yes: self.yes,
            };
        }
        if let Some(event_id) = &self.event_id {
            return DeleteAction::Request {
                target: DeleteTarget::Event {
                    event_id: event_id.clone(),
                },
                yes: self.yes,
            };
        }
        match (&self.resume, &self.cancel) {
            (Some(id), _) => DeleteAction::Resume(id.clone()),
            (None, Some(id)) => DeleteAction::Cancel(id.clone()),
            (None, None) => DeleteAction::Status,
        }
    }
}

/// What this process should do. `Stdio` is the no-subcommand default, kept
/// first and kept the default so existing MCP client configs that just run
/// `liamd` behave exactly as they did before modes existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// One client, one process, MCP over this process's stdin and stdout.
    Stdio,
    /// The socket daemon: many clients, one store.
    Serve,
    /// A shuttle between this process's stdio and the daemon's socket.
    /// Opens no store and loads no model.
    Proxy,
    /// Seals the plaintext events of the log in place, then exits.
    EncryptLog,
    /// Deletes, or manages delete jobs, then exits.
    Delete(DeleteAction),
}

impl Mode {
    /// Whether this mode loads the embedder and reranker. The proxy forwards
    /// bytes, and a delete only reads the log and destroys keys.
    pub fn loads_models(&self) -> bool {
        !matches!(self, Self::Proxy | Self::Delete(_))
    }
}

impl Cli {
    /// Which mode the parsed arguments select.
    pub fn mode(&self) -> Mode {
        match &self.command {
            None => Mode::Stdio,
            Some(Command::Serve) => Mode::Serve,
            Some(Command::Proxy) => Mode::Proxy,
            Some(Command::EncryptLog) => Mode::EncryptLog,
            Some(Command::Delete(command)) => Mode::Delete(command.action()),
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
    fn encrypt_log_selects_the_migration() {
        assert_eq!(parse(&["liamd", "encrypt-log"]).mode(), Mode::EncryptLog);
    }

    fn delete_action(args: &[&str]) -> DeleteAction {
        match parse(args).mode() {
            Mode::Delete(action) => action,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn delete_with_a_subject_counts_unless_yes_is_given() {
        // Arrange
        let target = DeleteTarget::Subject {
            subject: "alice".into(),
            scope: Some("work".into()),
            include_children: true,
        };

        // Act
        let count = delete_action(&[
            "liamd",
            "delete",
            "--subject",
            "alice",
            "--scope",
            "work",
            "--include-children",
        ]);
        let run = delete_action(&[
            "liamd",
            "delete",
            "--subject",
            "alice",
            "--scope",
            "work",
            "--include-children",
            "--yes",
        ]);

        // Assert
        assert_eq!(
            count,
            DeleteAction::Request {
                target: target.clone(),
                yes: false
            }
        );
        assert_eq!(run, DeleteAction::Request { target, yes: true });
    }

    #[test]
    fn delete_resume_takes_an_optional_job_id() {
        assert_eq!(
            delete_action(&["liamd", "delete", "--resume"]),
            DeleteAction::Resume(None)
        );
        assert_eq!(
            delete_action(&["liamd", "delete", "--resume", "job-1"]),
            DeleteAction::Resume(Some("job-1".into()))
        );
    }

    #[test]
    fn delete_cancel_status_and_event_select_their_actions() {
        assert_eq!(
            delete_action(&["liamd", "delete", "--cancel", "job-1"]),
            DeleteAction::Cancel("job-1".into())
        );
        assert_eq!(
            delete_action(&["liamd", "delete", "--status"]),
            DeleteAction::Status
        );
        assert_eq!(
            delete_action(&["liamd", "delete", "--event-id", "e1", "--yes"]),
            DeleteAction::Request {
                target: DeleteTarget::Event {
                    event_id: "e1".into()
                },
                yes: true
            }
        );
    }

    #[test]
    fn delete_wants_exactly_one_action_and_scope_only_with_a_subject() {
        for args in [
            &["liamd", "delete"][..],
            &["liamd", "delete", "--status", "--resume"],
            &["liamd", "delete", "--subject", "a", "--event-id", "e"],
            &["liamd", "delete", "--scope", "work", "--event-id", "e"],
            &["liamd", "delete", "--status", "--yes"],
        ] {
            let error = Cli::try_parse_from(args).expect_err("must be a usage error");
            assert_eq!(error.exit_code(), 2, "{args:?}");
        }
    }

    #[test]
    fn only_the_proxy_and_delete_skip_the_models() {
        assert!(!Mode::Proxy.loads_models());
        assert!(!Mode::Delete(DeleteAction::Status).loads_models());
        for mode in [Mode::Stdio, Mode::Serve, Mode::EncryptLog] {
            assert!(mode.loads_models(), "{mode:?}");
        }
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
