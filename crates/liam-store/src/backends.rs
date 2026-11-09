// SPDX-License-Identifier: Apache-2.0
//! Backend selection. `DefaultBackend` resolves to the enabled backend, and
//! `DefaultGraph` (in the crate root) is `Graph` over it.
//!
//! One backend ships today, libSQL. The [`crate::Backend`] trait stays
//! regardless: it is the real insurance against being tied to one engine,
//! and it costs nothing to keep, whereas removing it would mean rewriting
//! `Graph<B: Backend>` throughout `graph.rs`.
//!
//! A rusqlite backend used to sit alongside it as a scaffold whose method
//! bodies were all `todo!()`. It was deleted rather than finished: no build
//! ever compiled it, so it could only rot; enabling it panicked at runtime;
//! and it cost three optional dependencies (`rusqlite` with a bundled
//! SQLite, `sqlite-vec`, `zerocopy`) that nothing used. It was not the
//! Linux escape hatch either, since both aarch64 blockers come from the
//! model crates rather than the store.
//!
//! Adding a second backend later means adding its module and a `cfg` here,
//! plus whatever selects it. The trait is already shaped for it.

mod libsql;
pub use libsql::LibsqlBackend as DefaultBackend;
