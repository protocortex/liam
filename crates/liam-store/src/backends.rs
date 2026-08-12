// SPDX-License-Identifier: AGPL-3.0-only
//! Backend selection. Enable exactly one backend feature; `DefaultBackend`
//! resolves to it, and `DefaultGraph` (in the crate root) is `Graph` over it.

#[cfg(all(feature = "backend-libsql", feature = "backend-rusqlite"))]
compile_error!("enable exactly one backend: `backend-libsql` or `backend-rusqlite`");

#[cfg(not(any(feature = "backend-libsql", feature = "backend-rusqlite")))]
compile_error!("enable one backend: `backend-libsql` (default) or `backend-rusqlite`");

#[cfg(feature = "backend-libsql")]
mod libsql;
#[cfg(feature = "backend-libsql")]
pub use libsql::LibsqlBackend as DefaultBackend;

#[cfg(feature = "backend-rusqlite")]
mod rusqlite;
#[cfg(feature = "backend-rusqlite")]
pub use rusqlite::RusqliteBackend as DefaultBackend;
