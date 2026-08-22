# CI/CD with GitHub Actions

**Date:** 2026-08-01
**Status:** Approved design, pending implementation plan (writing-plans)
**Baseline:** `main` at squash `fbaf6f1` (M1 merged); candle 0.11 bump in flight on PR #2.

## Purpose

The repo has no CI. It now uses PR-based, squash-only merges (branch protection
rejects merge-commit and rebase merges), so automated checks that gate PRs are
the missing safety net. This adds GitHub Actions CI (fmt, clippy, test, a heavy
`--features local` build, and a cargo-deny security/license gate) plus a
tag-triggered release workflow that builds and attaches `liamd` binaries.

## Context (from the repo)

- Cargo workspace, 3 crates: `liam-store`, `liam-model`, `liam-daemon`. Rust
  stable, edition 2021, no MSRV declared.
- Feature topology:
  - `liam-store`: `default = ["backend-libsql", "cluster"]`; `backend-rusqlite`
    is a scaffold that panics if enabled (exclude from CI).
  - `liam-model`: `default = []`; `local` is heavy (fastembed/ONNX + candle 0.11,
    downloads models at runtime, not build time).
  - `liam-daemon`: `local = ["liam-model/local"]`.
- Tests: `cargo test --workspace` (11 tests; graph tests gated on
  `backend-libsql`). No test requires network/model downloads.
- No `rustfmt.toml`, `clippy.toml`, `deny.toml`, or toolchain file today.

## Decisions (approved)

- **Scope:** CI + tag-triggered release binaries. No crates.io publish, no build
  provenance/attestation, no MSRV job this cut.
- **Checks:** fmt + clippy + test + `--features local` build + cargo-deny.
- **OS strategy:** metadata jobs (fmt, clippy, deny) on `ubuntu-latest`
  (platform-neutral, ~10x cheaper minutes). Build/test jobs run a matrix of
  `ubuntu-latest` + `macos-latest`. `windows-latest` is scaffolded (commented)
  for a one-line future add.

## Design

### `.github/workflows/ci.yml`

- **Triggers:** `pull_request` targeting `main`; `push` to `main`.
- **Concurrency:** group by workflow + ref, `cancel-in-progress: true`.
- **Permissions:** `contents: read`.
- **Caching:** `Swatinem/rust-cache@v2` on every job (candle compiles are slow).
- **Toolchain:** `dtolnay/rust-toolchain@stable` with `rustfmt`, `clippy`
  components (consistent with the added `rust-toolchain.toml`).

Jobs:
1. **lint** (`ubuntu-latest`):
   - `cargo fmt --all --check`
   - `cargo clippy --workspace --all-targets -- -D warnings` on **default
     features** (fast). The `local` path is not clippy-linted here to avoid a
     second heavy candle compile on the lint job; the `build-local` job covers
     that it compiles. (Revisit if we want lint coverage of `local` code.)
2. **test** (`strategy.matrix.os: [ubuntu-latest, macos-latest]`,
   fail-fast: false):
   - `cargo test --workspace` (default features).
3. **build-local** (`strategy.matrix.os: [ubuntu-latest, macos-latest]`):
   - `cargo build -p liam-daemon --features local` (build-only; no run, so no
     model download). This is the job that catches candle/fastembed API breakage.
4. **deny** (`ubuntu-latest`):
   - `EmbarkStudios/cargo-deny-action@v2` running advisories, licenses, bans,
     and sources checks against `deny.toml`.

### `.github/workflows/release.yml`

- **Trigger:** push of a tag matching `v*.*.*`.
- **Permissions:** `contents: write` (to create the Release and upload assets).
- **Matrix:** `[ubuntu-latest, macos-latest]` (Linux x86_64 + macOS Apple
  Silicon), scaffolded for more targets.
- **Steps:** checkout, stable toolchain, rust-cache, `cargo build --release -p
  liam-daemon --features local`, package the `liamd` binary as
  `liamd-<os>-<arch>.tar.gz`, then `softprops/action-gh-release@v2` attaches the
  archive to the Release for that tag.
- Our existing `v0.1.0` tag was never pushed, so nothing fires retroactively;
  the next pushed `v*` tag triggers it.

### Config files

- **`deny.toml`:** advisories (deny vulnerabilities/unmaintained), licenses with
  an allowlist covering our dependency tree (MIT, Apache-2.0, BSD-2-Clause,
  BSD-3-Clause, ISC, Zlib, Unicode-3.0, MPL-2.0 if present), bans (warn on
  duplicate versions), sources (crates.io only). The allowlist will be tuned on
  the first `cargo deny check` run (candle/ONNX deps may surface a license that
  must be added explicitly).
- **`rustfmt.toml`:** minimal (pins current formatting as canonical; empty or a
  couple of conservative settings).
- **`rust-toolchain.toml`:** `channel = "stable"`, components `rustfmt`,
  `clippy`, so local and CI toolchains agree.

## Testing / verification

- CI is validated by the PR that introduces it: the workflow must run green on
  its own PR (fmt, clippy, test on both OSes, build-local on both OSes, deny).
- Release workflow can't be fully exercised without pushing a tag; verify its
  YAML with `actionlint` locally and confirm the build/package steps are correct
  by reasoning, then prove it on the first real `v*` tag (e.g. cutting `v0.2.0`
  after the candle bump lands).
- `deny.toml` verified locally with `cargo deny check` before merge; iterate the
  license allowlist until clean.

## Non-goals

- No crates.io publishing (crates aren't API-stable; rusqlite backend panics).
- No Windows in the matrix yet (scaffolded only).
- No cross-compilation beyond the runner's native target.
- No build provenance/attestation, no MSRV/toolchain matrix.
- `backend-rusqlite` is excluded from all build/test jobs (scaffold panics).

## Open risks

- **build-local cost/time.** Building candle + ONNX on every PR across two OSes
  is the slowest part; `rust-cache` mitigates but cold caches will be slow.
  Acceptable given it's the exact breakage class we want to catch.
- **cargo-deny license churn.** The first run will likely fail on an
  unanticipated transitive license; the allowlist needs a tuning pass. Treat the
  first deny run as iterative, not a one-shot.
- **macOS runner minutes.** The two-OS build/test + macOS release build consume
  more minutes; if it becomes costly, drop `build-local` to Linux-only or make
  it `push`-to-main-only rather than per-PR.

## Next step

Invoke writing-plans for a step-by-step plan (workflows + config files, each
validated with `actionlint`/`cargo deny` where possible), then implement on a
branch off `main` and open a PR whose own run is the acceptance test.
