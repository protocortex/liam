# CI/CD GitHub Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GitHub Actions CI (fmt, clippy, test, `--features local` build, cargo-deny) and a tag-triggered release-binary workflow to the LIAM Rust workspace.

**Architecture:** Two workflow files under `.github/workflows/` plus three repo-root config files (`rust-toolchain.toml`, `rustfmt.toml`, `deny.toml`). CI gates PRs; release builds `liamd` on version tags. Verification is by running the same tools locally (`cargo fmt`, `cargo clippy`, `cargo deny`, `actionlint`) — the introducing PR's own green run is the final acceptance test.

**Tech Stack:** GitHub Actions, `dtolnay/rust-toolchain`, `Swatinem/rust-cache@v2`, `EmbarkStudios/cargo-deny-action@v2`, `softprops/action-gh-release@v2`, `actionlint`.

## Global Constraints

- Rust **stable**, edition 2021, no MSRV. Pin the toolchain via `rust-toolchain.toml`.
- OS strategy: metadata jobs (fmt, clippy, deny) on `ubuntu-latest`; `test` and `build-local` on matrix `[ubuntu-latest, macos-latest]`; `windows-latest` scaffolded (commented) only.
- `backend-rusqlite` is a panicking scaffold — never build/test it in CI. Use default features (`backend-libsql` + `cluster`).
- `local` feature is heavy (candle 0.11 + fastembed/ONNX); build-only, never run in CI (no model downloads).
- Release binaries build `liam-daemon --features local`.
- Commits via `/commit-and-push` (no raw `git commit` with identity overrides). Conventional Commits, no em/en dashes, no AI/Claude attribution.
- Pin third-party actions to a major tag as shown (`@v4`, `@v2`).

---

### Task 1: Toolchain, rustfmt config, and a fmt+clippy-clean tree

CI's `lint` job fails immediately if the existing tree isn't fmt/clippy clean. This task adds the toolchain/format config and brings the tree to green.

**Files:**
- Create: `rust-toolchain.toml`
- Create: `rustfmt.toml`
- Modify (only if clippy/fmt require): any source file with a warning/format diff.

**Interfaces:**
- Produces: a repo that passes `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` on default features. CI Task 3 relies on this.

- [ ] **Step 1: Create `rust-toolchain.toml`**

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

- [ ] **Step 2: Create `rustfmt.toml`**

```toml
edition = "2021"
```

- [ ] **Step 3: Check formatting**

Run: `cargo fmt --all --check`
Expected: either clean (exit 0) or a diff. If it reports a diff, continue to Step 4; if clean, skip Step 4.

- [ ] **Step 4: Apply formatting if needed**

Run: `cargo fmt --all`
Then re-run `cargo fmt --all --check` and confirm exit 0.

- [ ] **Step 5: Run clippy as CI will**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: exit 0. If warnings appear, fix each minimally (idiomatic clippy fixes; do not `#[allow]` broadly — prefer the suggested fix). Re-run until exit 0.

- [ ] **Step 6: Confirm tests still pass**

Run: `cargo test --workspace`
Expected: `11 passed` across the suites, 0 failed.

- [ ] **Step 7: Commit** via `/commit-and-push`:

`chore(ci): pin stable toolchain and rustfmt config, format tree`

---

### Task 2: cargo-deny policy

**Files:**
- Create: `deny.toml`

**Interfaces:**
- Produces: a `deny.toml` that passes `cargo deny check` locally; CI Task 3's `deny` job runs the same.

- [ ] **Step 1: Create `deny.toml`**

```toml
# cargo-deny policy. Run `cargo deny check` to validate.

[advisories]
version = 2
yanked = "deny"
# ignore = []  # add specific RUSTSEC ids here only with a justification comment

[licenses]
version = 2
# Allowlist tuned to the current dependency tree; extend as `cargo deny check`
# reports new licenses (candle/ONNX deps may surface additional ones).
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "Unicode-3.0",
    "Unicode-DFS-2016",
    "MPL-2.0",
    "CC0-1.0",
]
confidence-threshold = 0.9

[bans]
multiple-versions = "warn"
wildcards = "warn"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

- [ ] **Step 2: Install cargo-deny if absent**

Run: `cargo deny --version || cargo install cargo-deny --locked`

- [ ] **Step 3: Run the check and tune**

Run: `cargo deny check`
Expected: it may FAIL on a license not in the allowlist or a duplicate-version ban. For each **license** failure, verify the license is acceptable (permissive) and add its SPDX id to `[licenses].allow` with no other change. `multiple-versions` and `wildcards` are `warn` (non-fatal). Re-run until `advisories`, `licenses`, `sources` all report `ok`. Do NOT silence advisories without a written justification.

- [ ] **Step 4: Record the final result**

Run: `cargo deny check 2>&1 | tail -5`
Expected: no `error[...]`; summary shows checks passed (bans may warn).

- [ ] **Step 5: Commit** via `/commit-and-push`:

`chore(ci): add cargo-deny policy`

---

### Task 3: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `rust-toolchain.toml`, `rustfmt.toml` (Task 1), `deny.toml` (Task 2).

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
      - name: Format
        run: cargo fmt --all --check
      - name: Clippy
        run: cargo clippy --workspace --all-targets -- -D warnings

  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        # add windows-latest to extend the matrix
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Test
        run: cargo test --workspace

  build-local:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build (local feature)
        run: cargo build -p liam-daemon --features local

  deny:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: EmbarkStudios/cargo-deny-action@v2
        with:
          command: check
```

- [ ] **Step 2: Validate the workflow YAML**

Run: `actionlint .github/workflows/ci.yml 2>/dev/null || docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:latest -color .github/workflows/ci.yml`
If neither `actionlint` nor Docker is available, instead run `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` to confirm the YAML parses, and note that lint was skipped.
Expected: no errors.

- [ ] **Step 3: Commit** via `/commit-and-push`:

`ci: add CI workflow (fmt, clippy, test, local build, cargo-deny)`

---

### Task 4: Release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create `.github/workflows/release.yml`**

```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

permissions:
  contents: write

jobs:
  release:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
          - os: macos-latest
            target: aarch64-apple-darwin
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build release
        run: cargo build --release -p liam-daemon --features local
      - name: Package
        run: tar -czf liamd-${{ matrix.target }}.tar.gz -C target/release liamd
      - name: Upload to release
        uses: softprops/action-gh-release@v2
        with:
          files: liamd-${{ matrix.target }}.tar.gz
```

- [ ] **Step 2: Validate the workflow YAML**

Run: `actionlint .github/workflows/release.yml 2>/dev/null || python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))"`
Expected: no errors / YAML parses.

- [ ] **Step 3: Commit** via `/commit-and-push`:

`ci: add release workflow building liamd binaries on version tags`

---

## Self-Review

**Spec coverage:**
- ci.yml lint/test/build-local/deny jobs, OS strategy, caching, concurrency, permissions → Task 3. ✓
- release.yml tag-triggered, `--features local`, matrix, tar.gz, gh-release → Task 4. ✓
- `deny.toml` with allowlist + tuning pass → Task 2. ✓
- `rustfmt.toml`, `rust-toolchain.toml` → Task 1. ✓
- Ensuring the existing tree is fmt/clippy clean so CI is green on introduction → Task 1 (Steps 3-6). ✓
- Non-goals respected: no publish, no Windows job (scaffolded comment), no attestation, no MSRV, no rusqlite. ✓
- Branch-protection required-checks is called out in the spec as a UI follow-up, not a file — intentionally not a task.

**Placeholder scan:** No TBD/TODO. The deny-allowlist "tune until green" and the fmt/clippy "fix warnings" steps are concrete run-and-fix loops with exact commands, not deferred work.

**Consistency:** Feature flags (`--features local`, default features), action versions (`@v4`/`@v2`), and OS lists match between spec and every task and between ci.yml and release.yml.

## Execution Handoff

Plan complete and saved to `docs/plans/2026-08-01-cicd-github-actions.md`.
