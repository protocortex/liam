## Consolidation

[Back to the docs index](../README.md)

Consolidation is the maintenance job that runs on the `gc.interval_hours`
schedule, and on demand with `liam consolidate`. It is not the log's compactor,
which only turns closed WAL segments into Parquet and never removes data.

A run goes through these steps, each recorded in `<database>.consolidation.json`
as it finishes:

1. Snapshot the database into `consolidation.snapshot_dir`.
2. Draw up to `consolidation.probe_count` live nodes as probes and score them.
3. Link near-duplicates, those whose embeddings reach
   `consolidation.dedup_similarity_threshold`. Both rows stay; the merge adds an edge.
4. Run the maintenance steps: replay the log, sweep, repair provenance, refresh
   clusters, and (in the daemon only) resynthesize stale entities.
5. Score the probes again. A probe whose recall drops by `consolidation.probe_recall_margin`
   or more fails the run. A probe whose node the run's own sweep removed on purpose is
   not counted, but a probe that lost its node any other way fails the run, and so does a
   sweep that leaves no probe to check against.

A passing run commits. A failing one is rolled back: a marker in the log cancels
the run's own writes, the snapshot is restored and the log is replayed, so writes
clients made meanwhile survive. A run that stops for any reason resumes at its
next step, and a rollback that was cut short is finished when `liamd` starts,
before it serves.

```
liam consolidate [--database PATH]          # run, or resume an unfinished run
liam consolidate status [--database PATH]   # show the last run
```

`liam consolidate` needs the store to itself, so stop `liamd` first. It runs no
model, so resynthesis waits for the daemon's next scheduled run. `status` reads
one file and works while the daemon runs.

| Exit code | Meaning |
|---|---|
| `0` | The run committed, or `status` printed. |
| `1` | The run was rolled back and the store is as it was, or the command failed before or during the run. |
| `2` | The restore failed. The store needs manual inspection and its snapshot is kept. |

After a failed restore no new run starts, from the command or from the daemon,
until the manifest `<database>.consolidation.json` is removed. Check the store
and the kept snapshot first.
