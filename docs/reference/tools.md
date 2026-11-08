# Tools

[Back to the docs index](../README.md)

`remember` and `recall` are the core pair, and ten more tools cover reading,
review and admin.

Point a client at `liamd` and it exposes twelve tools. This page has the field
tables for `remember`, `recall`, `timeline` and `relate`. The other eight are
documented in the README sections linked here:

| Tool | What it does | Documented under |
|---|---|---|
| `ask` | Answers a question from memory, synthesized and cited. | [Reading modes and provenance](../../README.md#reading-modes-and-provenance) |
| `clusters` | Lists memory clusters, largest first. | [Reading modes and provenance](../../README.md#reading-modes-and-provenance) |
| `summaries` | Lists or shows the summaries LIAM wrote. | [Summaries](../../README.md#summaries) |
| `facts` | Lists the facts held about a subject. | [Reading facts](../../README.md#reading-facts-the-facts-tool) |
| `feedback` | Says whether a `recall` or `ask` result was right. | [Tiering](../../README.md#tiering) |
| `relate` | Links two memories by handle. | below |
| `metrics` | Shows LIAM's own metrics and active alerts. | [Metrics and alerts](../../README.md#metrics-and-alerts) |
| `delete` | Previews what a deletion would destroy. Admin only. | [Deleting a subject](../../README.md#deleting-a-subject) |
| `consolidate` | Runs one consolidation now. Admin only. | [Consolidation](../../README.md#consolidation) |

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
| `query` | yes | Text; embedded for the vector channel too. |
| `kind` | no | Restrict to one kind. |
| `scope` | no | Restrict to one partition. Same format as `remember`'s `scope`. |
| `k` | no | How many hits to return (default 8). |

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
