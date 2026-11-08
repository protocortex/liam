## Garbage collection

[Back to the docs index](../README.md)

A sweep is recorded in the log as tombstones before the rows are deleted, so a
rebuild does not bring swept rows back. The log lock is held for the whole
sweep, so a very large sweep blocks every writer until it finishes. Chunking
bounds the size of each event and transaction, not the wait.
