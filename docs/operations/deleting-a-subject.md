# Deleting a subject

[Back to the docs index](../README.md)

LIAM can forget a subject. Set `[encryption]` and the log is sealed: each row
is encrypted under a key for its subject, and the keys are wrapped by a master
key. Deleting a subject destroys its keys, removes what was derived from its
rows (summaries, facts, cache entries, tier and usage rows, snapshots), and
scrubs the database. The data stays unreadable even where a copy of the log
remains.

Only `liamd delete` destroys data, and it needs the daemon stopped. The `delete`
tool is a dry run: it counts what a delete would destroy, store by store, and
changes nothing.

```sh
liamd delete --subject alice           # the counts, nothing changes
liamd delete --subject alice --yes     # start the job
liamd delete --status                  # job ids and states, never subjects
liamd delete --resume                  # finish every open job
liamd delete --cancel JOB              # only before a key was destroyed
```

A job has `deletion.deadline_days` to finish. It is warned about at 80 percent
of that time and alerted at the deadline (the `deletion_backlog_age_days`
metric). The daemon finishes open jobs when it starts, and refuses to start if
one cannot finish, because rows of a half-deleted subject would otherwise be
served.

What to know before you rely on it:

- Backups and cloud-synced copies of the database are out of reach. A copy made
  before a delete still holds the old pages.
- Losing the master key or the key registry (`keys.registry` in the log
  directory) loses all log content. There is no escrow.
- A client name is a claim, not an authenticated identity. `[admin].producers`
  guards against mistakes, and any client that claims an admin name can read the
  dry-run counts.
- An edge body is sealed under its source subject's key. Deleting only the
  destination subject does not make it unreadable.
- On Linux there is no keychain. A `0600` key file beside the data gives little
  protection against a local reader.
- Dedup applies within a key group. A subject that changed names across versions
  needs each name deleted.
- `secure_delete` lowers the risk that freed pages keep text, and does not
  remove it on an SSD or a copy-on-write file system.
- Nothing is deleted by age unless the owner sets a retention policy.

`liamd encrypt-log` seals an existing plaintext log in place, with the daemon
stopped.

## Archive

With `[archive]` enabled, a closed log partition whose newest event is older
than `after_days` is copied to `archive.dir` as zstd, checked, and removed from
the log directory. Nothing is deleted. A rebuild or a catch-up reads archived
partitions, and one that was archived and then lost fails with an error naming
the partition instead of rebuilding a store with a hole in it. Sealed bodies
barely compress, so zstd saves little on an encrypted log. The 90 days and
level 3 are starting values, not measured.
