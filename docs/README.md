# LIAM documentation

[Back to the project README](../README.md)

This is the index of every page. Pick a section, or use the table of contents
below.

## Table of contents

1. [Getting started](#getting-started)
2. [Reference](#reference)
3. [Development](#development)
4. [Design history](#design-history)

## Getting started

| Page | What it covers |
|---|---|
| [Install](getting-started/install.md) | Install with Homebrew, and the netinstall tarball that `install.sh` unpacks. |
| [Running LIAM](getting-started/running.md) | The three `liamd` modes, one daemon for many agents, the socket, and a real embedder. |

## Reference

| Page | What it covers |
|---|---|
| [Tools](reference/tools.md) | The twelve MCP tools, with the field tables for `remember`, `recall`, `timeline` and `relate`. |
| [Configuration](reference/configuration.md) | Every `liam.toml` key, its default and its meaning. |
| [Feature flags](reference/feature-flags.md) | The Cargo features of `liam-store` and `liam-daemon`. |

## Development

| Page | What it covers |
|---|---|
| [Workspace](development/workspace.md) | The four crates and how they fit together. |
| [Build and test](development/build-and-test.md) | Building and testing the workspace, and the opt-in model stack. |
| [Evaluating retrieval](development/evaluating-retrieval.md) | `liam eval retrieval`, its golden set and how to read the report. |
| [Status and roadmap](development/status-and-roadmap.md) | What is built, what is checked, and what comes next. |

## Design history

These folders hold the decisions and designs behind LIAM. They are a record, not
a guide.

- [Architecture decision records](adr/): one file per decision.
- [Specs](specs/): design documents and evaluations.
- [Plans](plans/): implementation plans.
