# The devices command

[Back to the docs](../README.md) | Related: [Configuration](configuration.md)

`liam devices` shows what the machine and this build can compute on, and what
the `auto` setting of `llm.device` comes to under your config. The embedder
always runs on the CPU and has no device setting. Run it first when a model is slower than expected or when you report a
hardware problem.

```sh
liam devices
liam devices --json
liam devices --config /path/to/liam.toml
```

It reads the config the same way as the other `liam` commands, and it never
writes anything.

## What it prints

| Line | Meaning |
|---|---|
| `platform` | The operating system and the CPU architecture of this binary. |
| `cpu cores` | On Apple Silicon, the performance and efficiency cores from `hw.perflevel0.logicalcpu` and `hw.perflevel1.logicalcpu`, with the logical total. Elsewhere only the logical cores. |
| `metal` | Whether llama.cpp found a Metal device it can offload to. Always not usable off macOS. |
| `built backends` | The compute backends compiled into this binary: Metal on macOS and the CPU always. |
| `llama.cpp devices` | The devices in llama.cpp's own backend registry, with name, description, backend and memory in MB. It says `not reported` when llama.cpp cannot start. |
| `llm.device` | What the configured value resolves to for the LLM, and the engine that runs it, `llama.cpp`. |

A missing config file gives the built-in defaults, which are mock providers, so
the device line shows `not used`. An invalid config makes the command exit with
the load error.

## How `auto` resolves

`auto` is Metal on macOS when a Metal device opens, and the CPU everywhere else.
When no Metal device opens on a Mac, `auto` falls back to the CPU and the line
says so. A value that cannot work, such as `metal` on Linux or `cuda`, which is
never built, shows as `unavailable` with the reason. A mock provider shows
`not used`.

## JSON

`--json` prints one object with the same facts: `os`, `arch`,
`performance_cores`, `efficiency_cores`, `logical_cores`, `metal_usable`,
`built_backends`, `llama_devices` (`null` when not reported), and `llm`, with
`configured`, `resolved`, `engine` and `note`.
