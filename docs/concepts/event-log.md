# The event log

[Back to the docs index](../README.md)

Every write goes into an event log before it reaches the database, so the
database is a projection of the log. `liamd` opens the log at startup, before
it serves anything, and replays any event the database has not applied yet, such
as one left by a crash between the two. The maintenance tick does the same
before each sweep.

The log lives in `log.dir`, which defaults to `<database stem>.log` beside the
database (`~/.liam/liam.db` logs to `~/.liam/liam.log`). Back it up with the
database: it is the source of truth that `liam rebuild` reads.

`liamd` compacts each closed WAL segment into a Parquet file in the background
and finishes the ones a crash left behind at startup. A segment is closed by the
first write after it reaches `log.wal_segment_max_bytes` or
`log.wal_rotate_interval_secs`, so an idle daemon keeps its last segment open.

One writer owns the log directory at a time. A second `liamd`, or a `liam
rebuild` while `liamd` runs, fails at once with an error naming the directory.
The lock needs a filesystem with advisory locks, so keep the log on a local disk.

If the log is poisoned, which a failed disk write does, `liamd` logs `the event
log is poisoned; restart liamd to replay it` at error level and exits with a
failure. Restarting replays the log into the database.

`liam rebuild --database PATH [--log-dir DIR] [--force]` recreates the nodes and
edges from the log and checks the result against it, row by row. It needs the
store to itself, so stop `liamd` first. The log directory is `--log-dir`, else
`log.dir` from the config, else the default beside `--database`.

Vectors stored before documents and queries were embedded separately carry the
`query: ` prefix on documents too. After upgrading, run `liam rebuild --database
PATH` once: it clears the stored vectors and re-embeds every live node as a
document, which makes the store consistent.

## Upgrading a database that predates the log

The first start with the log copies every row the database already holds into
the log, nodes first, then edges. Writes are refused with `BackfillRequired`
until the copy is done, and since the daemon runs it before serving, clients see
nothing but a slower start on a large database. The cost grows with the number
of rows: the copy is one log write per row, so expect minutes for millions of
rows. A client that connects while it runs can time out, so retry once the
daemon logs `backfill completed`. The log reports `logged/total` at every saved
resume point. An interrupted copy resumes where it stopped, and a database that
is already logged is not copied again.

## Quarantined events

When replay refuses an event, the daemon appends a `Quarantined` record that
cancels it and carries the reason, and keeps the reason in the store's
`log_quarantine` table. A store restored from a snapshot, or rebuilt from the
log, gets the row back from that record. A log written before this record
existed holds a plain `Voided` record instead, which has no reason, so a replay
skips the event and recreates no row. Schema version 7 adds the record; a build
that reads only version 6 refuses a log that holds one.

## Refused at open

The database remembers which log it belongs to. If you delete the log directory
and keep the database, `liamd` refuses to start with an error naming both ids,
because a fresh log would not hold the rows the database has. Restore the log
directory, point `log.dir` at it, or rebuild from a log that holds the rows.
