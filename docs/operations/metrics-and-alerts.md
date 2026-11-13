# Metrics and alerts

[Back to the docs index](../README.md)

The `metrics` tool shows the latest reading of each metric and the active
alerts. Pass `metric` for one, and `limit` to cap the readings (200 at most).
Readings carry only a small fixed set of labels (tier, retrieval arm, provider,
model) and never a subject, a producer or a lineage key, so keeping them cannot
defeat a delete.

| Metric | Meaning |
|---|---|
| `probe_recall_at_10` | Mean recall at 10 over the consolidation probes. |
| `arm_latency_p50_ms`, `arm_latency_p95_ms` | Nearest-rank percentiles of the last 50 timings of each retrieval arm. |
| `recall_latency_*`, `ask_latency_*` | The same for the `recall` and `ask` tools. |
| `degraded_replies` | Count of `recall` and `ask` replies that used a fallback, per tool and reason. See [Degraded replies](../reference/tools.md#degraded-replies). |
| `tier_bytes` | Bytes of vectors and signatures in the hot and the cold tier. |
| `log_bytes`, `archive_bytes` | Size of the log and archive directories. |
| `peak_rss_bytes` | Resident memory of the daemon and the processes it started. |
| `consolidation_seconds`, `rollback_count` | Duration of the last run, and rolled-back runs since start. |
| `items_promoted`, `items_reembedded` | What the last tier step moved, and the vectors it computed. |
| `bloom_false_positive_rate` | Filter passes the log did not confirm, over those plus rejections. |
| `ece` | Expected calibration error of item utility against feedback verdicts, in ten bins. Absent below `min_samples` verdicts. |
| `rules_extraction_backlog`, `model_extraction_backlog` | Events past each extraction cursor. |
| `model_calls`, `model_cost_micro_dollars` | Running totals per provider and model. |
| `cache_hit_rate` | Hits over lookups in the extraction response cache. |
| `deletion_backlog_count`, `deletion_backlog_age_days` | Open delete jobs, and the age of the oldest. |

After each consolidation run the readings are compared with the previous run's.
Three kinds of rule raise an alert, each inclusive of its bound: a relative
move of `delta_pct` percent (recall, latency, `ece`, Bloom rate), a ceiling
(rollbacks, extraction backlog, delete job age), and a rate of growth an hour
(model calls). A first run raises only ceiling alerts. A run that ends in a
failed restore is neither judged nor remembered. The previous values live in
`<database>.metrics-state.json`.
