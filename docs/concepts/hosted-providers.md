# Hosted providers

[Back to the docs index](../README.md)

LIAM can send extraction and summary requests to Anthropic, OpenAI or Gemini
instead of the local model. It is off by default, the default build links no
TLS code for it, and nothing is sent until you build with `--features hosted`
and set `extraction.provider` or `summary.provider` along with every part of
`[hosted]`.

A request goes out only when all of this holds:

- `hosted.enabled` is true and the request's host is in `hosted.allowed_hosts`.
  The host must match as written: a port, a trailing dot, capitals, a user name
  or an IP address is a different string and is refused. Redirects are never
  followed, and only https is used.
- The item passes the privacy gate. The default is to deny. An item is sent
  only when its scope (or an ancestor) is in `hosted.privacy.allowed_scopes`,
  none of its sources is in `local_only_sources`, and the provider's declared
  region is attested for that scope. A summary is judged on everything behind
  it: one local-only source keeps the whole summary at home. A cached answer is
  used only while the item still passes.
- The provider has a dated model id in `hosted.models` and a price in
  `hosted.prices`. The reply must name the pinned model, or it is refused and
  nothing is cached. A provider that no longer has the model is an error, never
  a fallback.

Keys never go in `liam.toml`. Put one in a file only you can read and name it in
`hosted.key_files`, or on macOS in a Keychain item called `liam-<provider>`. A
key does not appear in logs, errors or the cache.

Extraction asks hosted providers through their batch APIs, which cost less and
answer later. A run sends the requests the cache cannot answer in one batch,
records them in `<database>.hosted-batches.json`, and a later run collects the
answers into the response cache before it writes any fact. Events whose request
is still with the provider are not passed, so the cursor waits for them. A
request that expires is sent once more, and a second expiry fails it. Summaries
are asked directly. `ask` and entity synthesis stay on the local model whatever
the setting.

Every request, batch or direct, is first noted in `<database>.hosted-sends.jsonl`
with the provider, an opaque request id, the events it was built from and the
time. The file holds no text. If a run stops where it cannot tell whether a
request reached the provider, the next run sends it again and logs a warning
that it may be billed twice.

What a delete does about it: `liamd delete` cancels the batches that hold a
deleted request and asks the provider to delete their files where it has an
endpoint for that. A provider that keeps a copy anyway, one that cannot be
reached with the key and pin you set, and every direct request, leave a notice in
the job result and in `liamd delete --status`. A provider's own retention terms
decide how long its copy lasts, and LIAM cannot reach it. Treat anything you
allow into a hosted prompt as leaving your control.

Costs are counted per provider and model in the `model_calls` and
`model_cost_micro_dollars` metrics, using the price table you set. A model with
no price counts as free and is logged.

The endpoint paths and field names of the three adapters follow each
provider's public documentation and have not been checked against a live
account. Hosted R10, the extraction quality check, runs in two tiers. The
recorded tier replays answers shaped like each provider's through the real
adapters in normal CI. The live tier is ignored, needs a key, and has not been
run yet, so `r10_results.toml` has no rows.
