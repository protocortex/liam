# LIAM documentation

[Back to the project README](../README.md)

This is the index of every page. Pick a section, or use the table of contents
below.

## Table of contents

1. [Getting started](#getting-started)
2. [Reference](#reference)
3. [Concepts](#concepts)
4. [Operations](#operations)
5. [Development](#development)
6. [Design history](#design-history)

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

## Concepts

| Page | What it covers |
|---|---|
| [Why LIAM](concepts/why.md) | Why an agent memory needs history and contradiction, and how retrieval blends its signals. |
| [The event log](concepts/event-log.md) | The write-ahead log the database is built from, rebuilds, quarantined events and upgrades. |
| [The embedder fingerprint](concepts/embedder-fingerprint.md) | How a store records which embedder wrote its vectors, and how to change it. |
| [Reading modes and provenance](concepts/reading-modes.md) | Strict and infer reads, and the provenance line on every item. |
| [Summaries](concepts/summaries.md) | Model-written summary trees, their keys, and what the exit criteria prove. |
| [Tiering](concepts/tiering.md) | Hot and cold memories, the utility score, and the tiering keys. |
| [Extraction](concepts/extraction.md) | Facts derived from the log, trust and arbitration, predicates and the model server. |
| [Consolidation](concepts/consolidation.md) | The maintenance run, its steps, rollback and exit codes. |
| [Hosted providers](concepts/hosted-providers.md) | Sending extraction and summary requests to Anthropic, OpenAI or Gemini, and the privacy gate. |

## Operations

| Page | What it covers |
|---|---|
| [Garbage collection](operations/garbage-collection.md) | How a sweep is logged and what it blocks. |
| [Deleting a subject](operations/deleting-a-subject.md) | Encryption, `liamd delete`, its limits, and the log archive. |
| [Metrics and alerts](operations/metrics-and-alerts.md) | The `metrics` tool, every metric, and the alert rules. |

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
