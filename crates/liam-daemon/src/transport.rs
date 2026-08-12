// SPDX-License-Identifier: AGPL-3.0-only
//! `liamd serve`'s transport: how a listener is obtained and served.
//!
//! `socket` is the only submodule today: it owns the listener, the path
//! handling around it, and the accept loop. launchd activation, the stdio
//! proxy, and signal-driven shutdown are later Work Units (WU-6b, WU-9,
//! WU-6c respectively); their files do not exist yet, so this module does
//! not declare them, only `socket`.

pub mod socket;

use std::path::PathBuf;

use tokio::net::UnixListener;

/// Where the accept loop's listener came from.
///
/// Only one arm exists today: `Bound`, a socket this process bound itself
/// and therefore owns, so it is the one that gets to set the file's
/// permissions after bind and, later, decide on shutdown (WU-6c) whether
/// to unlink it. WU-6b adds `Activated`, a listener launchd handed us
/// because launchd owns the socket before this process even starts.
///
/// Any code that touches the socket FILE (chmod, unlink) must match on
/// this enum and act only in the `Bound` arm: doing it in the `Activated`
/// case would fight the supervisor that owns the path and can break its
/// restart behaviour. Modelling this as an enum from the start, rather
/// than retrofitting it once `Activated` exists, is what keeps that later
/// Work Unit purely additive instead of a rewrite.
///
/// Mode dispatch (WU-9) is what will call `socket::bind` and wire this
/// into `main`; until then nothing in the production build constructs one
/// outside tests, so it is `#[allow(dead_code)]` rather than actually
/// unused. Exercised by this Work Unit's own tests and, later, WU-8's
/// integration test.
#[allow(dead_code)]
pub enum ListenerSource {
    /// A socket this process bound itself, at `path`, and therefore owns.
    Bound {
        listener: UnixListener,
        path: PathBuf,
    },
}
