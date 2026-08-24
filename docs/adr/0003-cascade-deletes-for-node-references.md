# ADR-0003: Cascade deletes for node references

- **Status:** Accepted
- **Date created:** 2026-08-23
- **Date accepted:** 2026-08-24

## Context

`gc` failed on any store holding a relationship, aborting the whole sweep with
`FOREIGN KEY constraint failed`. It deleted `nodes` rows before the `edges` and
`node_community` rows referencing them, and `sweep` in the daemon logs the error and carries
on, so retention silently stopped running and the store grew without bound. Fixed by deleting
the referencing rows first, per retention rule, recorded as ADR-0001 Amendment 4.

The premise that let it hide is worth restating, because it was written down in more than one
place and used as a decision driver. **libSQL enforces the declared `REFERENCES` by default**,
which stock SQLite does not. No `PRAGMA foreign_keys = ON` exists anywhere in the crate, and
that absence was read as "the constraints are inert". It is not.

So the guarantee now depends on every caller that deletes a node remembering to delete its
children first. There is exactly one such caller today, `Graph::gc`, and it does it correctly.
Nothing stops the next one getting it wrong, and the failure mode is not a crash at the call
site: it is a swept-under warning and retention that quietly stops.

Three declarations are involved, all of them children of `nodes(id)`:

- `edges.src` and `edges.dst`
- `node_community.node_id`

`nodes` itself references nothing, which matters more than it looks: it carries the FTS5
shadow table and three triggers, and it is the one table a rebuild would be genuinely risky on.

## Decision Drivers

- **The invariant belongs where it cannot be forgotten.** A rule enforced by convention in one
  function is a rule that holds until someone adds a second deleter. The database can enforce
  it instead.
- **The failure is silent.** `sweep` logs and continues, so a regression here does not surface
  as a test failure or an alert. It surfaces as a store that grew for months.
- **Existing databases are the whole problem.** Any change that only helps fresh databases
  would look correct in every test, because tests build fresh in-memory databases, while real
  stores stayed broken. That asymmetry is what makes this a migration decision rather than a
  DDL edit.
- **A second backend will not inherit the assumption.** The stubbed rusqlite backend follows
  stock SQLite and defaults enforcement off, so anything that relies on the database to cascade
  is relying on a backend-specific default.

## Considered Alternatives

### Keep the explicit deletes only (effort: none, already shipped)

- `gc` deletes children per retention rule before the nodes. Works on every database, new and
  old, with no migration.
- Trade-offs: correct today and free, since it is already landed. But it is a per-caller
  obligation with no enforcement, and the thing it guards against fails silently. It also
  spreads: every future table that references `nodes(id)` has to be added to `gc` by hand, and
  forgetting is invisible until a store stops sweeping.

### `ON DELETE CASCADE` on the DDL only, no migration (effort: S)

- Add the clause to `schema.rs` and stop.
- Trade-offs: rejected, and it is worth recording why because it is the obvious move. Every
  statement in `schema.rs` is `CREATE TABLE IF NOT EXISTS`, so an existing table keeps the
  constraint it was created with, and SQLite cannot `ALTER` one. Confirmed by running it: after
  re-creating a table with the clause, `pragma_foreign_key_list` still reports `NO ACTION` and
  the delete still fails. This would fix fresh databases, leave every real store broken, and
  pass every test, because tests build fresh databases. It produces two behaviours in the wild
  distinguishable only by when the database was created.

### Turn enforcement off and make the declarations decorative (effort: S)

- `PRAGMA foreign_keys = OFF` on every connection, restoring the behaviour the codebase
  believed it had all along.
- Trade-offs: it would have prevented the `gc` bug, and it makes the two backends agree. But it
  throws away a real integrity guarantee to avoid a migration, and it does so at exactly the
  moment the guarantee proved useful: enforcement is what surfaced the ordering bug at all. It
  also leaves the declarations in the schema lying about what they do, which is the state that
  produced the wrong memory of this system in the first place.

### Drop the `REFERENCES` declarations (effort: S)

- Same effect as above, honestly expressed, since a constraint that is not enforced should not
  be declared.
- Trade-offs: it needs the same table rebuild as the chosen option, so it costs what cascading
  costs and delivers less. Rejected on that basis rather than on principle.

## Decision

Adopt **`ON DELETE CASCADE` on both child tables, with a detected one-time rebuild**, and keep
the explicit deletes in `gc`.

The clause goes on `edges.src`, `edges.dst` and `node_community.node_id` in `schema.rs`, so
fresh databases get it directly. Existing databases get it through a migration that runs on
open, next to `migrate::add_column_if_missing`, which already exists for the analogous
column-level problem.

**Detection is a pragma, not a version number.** `pragma_foreign_key_list('edges')` reports
`on_delete` per declared constraint. A database needing the rebuild reports `NO ACTION`; one
already migrated reports `CASCADE`. That makes the migration idempotent and self-describing,
with no schema-version table to keep in step.

**The rebuild is the standard SQLite table-rebuild, on the child tables only.** Verified end to
end before writing this:

```sql
PRAGMA foreign_keys=OFF;              -- a no-op inside a transaction, so it goes outside
BEGIN;
CREATE TABLE edges_new (... ON DELETE CASCADE);
INSERT INTO edges_new SELECT * FROM edges;
DROP TABLE edges;
ALTER TABLE edges_new RENAME TO edges;
CREATE INDEX edges_out ON edges (src, type, tx_to);
CREATE INDEX edges_in  ON edges (dst, type, tx_to);
COMMIT;
PRAGMA foreign_key_check;             -- expect no rows
PRAGMA foreign_keys=ON;
```

Measured on the probe: `on_delete` moves from `NO ACTION` to `CASCADE`, `foreign_key_check`
returns nothing, rows are preserved, both indexes come back, and a subsequent node delete
removes its edges.

**`nodes` is never rebuilt**, and that is the main reason this is affordable. Cascade is
declared on the child, so only `edges` and `node_community` change. `nodes` keeps its FTS5
shadow table and its three triggers untouched, which is where a rebuild would actually be
dangerous.

**The explicit deletes in `gc` stay.** They become redundant on libSQL, and that is the point:
they are the guard on any backend that does not enforce foreign keys, which includes the
stubbed rusqlite backend and stock SQLite generally. Removing them would make correctness
depend on a backend-specific default, which is the mistake this record exists to correct, in
the other direction.

## Consequences

**Positive**

- A future caller that deletes a node cannot leave dangling children, whatever it forgets.
- A new table referencing `nodes(id)` declares its own cascade and needs no change to `gc`.
- The declared constraints and their real behaviour agree, so reading the schema stops
  producing the wrong mental model.

**Negative**

- **A migration that drops and recreates two tables holding real data.** It is one transaction
  and it is reversible only by restoring a backup, so it needs the review a schema change gets
  rather than the review a bugfix gets. This is the whole reason it is not folded into
  ADR-0001 Amendment 4.
- **Foreign keys are off for the duration of the rebuild.** That is required, since the drop
  and rename would otherwise trip the very constraints being rebuilt. `PRAGMA foreign_key_check`
  after `COMMIT` is what converts that window from an assumption into a check.
- **The first open of a large existing store pays a full copy of both tables.** Detection is
  cheap and the rebuild runs once, but the cost lands on a daemon start rather than on a
  maintenance window, and it is unmeasured at any real store size.
- **Cascade is silent by design.** A delete that removes twenty edges reports nothing about
  them, so `GcReport::edges_removed` stops seeing rows the database removed on its own. The
  explicit deletes in `gc` run first and still count, which keeps the report meaningful, but
  anything added later that deletes nodes outside `gc` will under-report.
- The two backends will behave differently until rusqlite either enables enforcement or keeps
  relying on the explicit deletes. That divergence is now recorded rather than assumed.

**Follow-up**

- Whether to enable `PRAGMA foreign_keys` explicitly rather than depend on the libSQL default
  is left open. Doing so would make the behaviour identical across backends and independent of
  a default that could change, and it is a one-line change per connection, but it turns every
  declared constraint on at once for the rusqlite path and that deserves its own measurement.
- `GcReport` could grow a note that cascade-removed rows are not counted, if anything ever
  reads the number for more than logging.
