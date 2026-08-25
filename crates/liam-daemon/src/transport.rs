// SPDX-License-Identifier: Apache-2.0
//! `liamd serve`'s transport: how a listener is obtained and served.
//!
//! `socket` owns the listener, the path handling around it, and the accept
//! loop. `activation` decides where that listener comes from at startup.
//! `shutdown` owns the ordered stop: signals, cancellation, drain, unlink.
//! The stdio proxy is a later Work Unit (WU-9), so its file does not exist
//! yet and this module does not declare it.

pub mod activation;
pub mod proxy;
pub mod shutdown;
pub mod socket;

use std::path::PathBuf;

use tokio::net::UnixListener;

/// Where the accept loop's listener came from, and therefore who owns the
/// socket FILE.
///
/// `Bound` is a socket this process bound itself, so it is the one that
/// sets the file's permissions after bind and unlinks it on shutdown.
/// `Activated` is a listener launchd handed over, because launchd bound the
/// socket before this process even started and still owns the path.
///
/// Any code that touches the socket file (chmod, unlink) MUST match on this
/// enum and act only in the `Bound` arm. Doing it in the `Activated` case
/// fights the supervisor that owns the path: unlinking it would leave
/// launchd holding a descriptor for a file that no longer exists at that
/// name, so the next client would connect to nothing and on-demand start
/// would quietly stop working until the job was reloaded.
pub enum ListenerSource {
    /// A socket this process bound itself, at `path`, and therefore owns.
    Bound {
        listener: UnixListener,
        path: PathBuf,
    },
    /// A listening socket launchd bound and handed over. No path is carried
    /// on purpose: this process must not act on the file, and holding the
    /// path would make it far too easy to.
    ///
    /// Constructed only by `activation::resolve`.
    Activated(UnixListener),
}

impl ListenerSource {
    /// The listener to accept on, whichever way it was obtained.
    pub fn listener(&self) -> &UnixListener {
        match self {
            ListenerSource::Bound { listener, .. } => listener,
            ListenerSource::Activated(listener) => listener,
        }
    }

    /// The socket path this process owns and must clean up, or `None` when
    /// the supervisor owns it. The single place that decides whether an
    /// unlink is ours to perform.
    pub fn owned_path(&self) -> Option<&std::path::Path> {
        match self {
            ListenerSource::Bound { path, .. } => Some(path),
            ListenerSource::Activated(_) => None,
        }
    }
}
