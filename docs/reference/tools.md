# Tools

[Back to the docs index](../README.md)

`remember` and `recall` are the core pair, and ten more tools cover reading,
review and admin.

Point a client at `liamd` and it exposes twelve tools. This page has the field
tables for `remember`, `recall`, `timeline` and `relate`. The other eight are
documented in the pages linked here:

| Tool | What it does | Documented under |
|---|---|---|
| `ask` | Answers a question from memory, synthesized and cited. | [Reading modes and provenance](../concepts/reading-modes.md) |
| `clusters` | Lists memory clusters, largest first. | [Reading modes and provenance](../concepts/reading-modes.md) |
| `summaries` | Lists or shows the summaries LIAM wrote. | [Summaries](../concepts/summaries.md) |
| `facts` | Lists the facts held about a subject. | [Reading facts](../concepts/extraction.md#reading-facts-the-facts-tool) |
| `feedback` | Says whether a `recall` or `ask` result was right. | [Tiering](../concepts/tiering.md) |
| `relate` | Links two memories by handle. | below |
| `metrics` | Shows LIAM's own metrics and active alerts. | [Metrics and alerts](../operations/metrics-and-alerts.md) |
| `delete` | Previews what a deletion would destroy. Admin only. | [Deleting a subject](../operations/deleting-a-subject.md) |
| `consolidate` | Runs one consolidation now. Admin only. | [Consolidation](../concepts/consolidation.md) |

`remember` records a fact.

| Field | Required | Notes |
|---|---|---|
| `kind` | yes | Opaque label: `decision`, `fact`, `symbol`, `episode`. |
| `label` | yes | Short title. |
| `content` | yes | The text to embed and store. |
| `scope` | no | Partition by project or agent. Trimmed; ASCII letters, digits, `-`, `_`, `/`; up to 200 characters; no leading/trailing `/` or empty segment (`//`). |
| `subject` | no | Identity. A new value with the same subject supersedes the old. |

`recall` retrieves and reranks.

| Field | Required | Notes |
|---|---|---|
| `query` | yes | Text; embedded for the vector channel too. Up to 4000 characters; empty, whitespace-only and NUL-containing queries are rejected. Only the first 64 distinct words (case-insensitive) reach the lexical channel; words past the 64th distinct one reach only the vector arm. |
| `kind` | no | Restrict to one kind. |
| `scope` | no | Restrict to one partition. Same format as `remember`'s `scope`. |
| `k` | no | How many hits to return (default 8, range 1 to 32; larger values are clamped to 32). Recall reranks a pool of twice `k`, at least 16 and at most 32, then returns the top `k`. |

`ask` applies the same limits to its `question`: up to 4000 characters, empty and NUL
rejected, and words past the 64th distinct one reach only the vector arm.

`timeline` shows an entity's compiled page and the mentions behind it, most
recent first.

| Field | Required | Notes |
|---|---|---|
| `entity` | yes | Handle or unambiguous prefix, as shown by `recall`. |
| `limit` | no | How many mentions to show (default 50, max 200). A capped result ends with an "(more mentions exist)" trailer instead of claiming an exact total. |

`relate` records a relationship between two memories.

| Field | Required | Notes |
|---|---|---|
| `from` | yes | Handle of the source memory, as shown by `recall`. A full id works too. |
| `to` | yes | Handle of the target memory. |
| `type` | yes | Relation type, for example `mentions`. Lower-cased on write. `supersedes` is reserved for version history and refused. |
