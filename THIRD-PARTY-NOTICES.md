# Third-Party Notices

This file lists third-party code that ships as part of LIAM binaries: the
vendored native code compiled in, and every Rust crate in the dependency tree
for the default build plus the `local` and `llama` features (the shipped
optional surface). It names each dependency and the license it is under. It
does not reproduce full license texts; see "Full license texts" below for
where to find those.

LIAM's own licensing terms are in this repository's LICENSE file.

## llama.cpp and ggml

LIAM links llama.cpp and its ggml tensor library directly into the binary.
They are vendored C++ sources built by the `llama-cpp-sys-2` crate, not a
system library and not downloaded at runtime. Both llama.cpp and ggml are
MIT licensed. This is the only bundled native (non-Rust) code in LIAM, and
the entry most readers checking this file are looking for.

## Rust dependencies, by license

Every crate below was confirmed present in the resolved dependency tree for
`cargo build --workspace` and for `cargo build -p liam-daemon --features
local,llama`, and every license name comes straight from that crate's
`license` field in `cargo metadata`, not from assumption. 371 Rust crates
ship in total.

### MIT OR Apache-2.0 (203 crates)

accelerate-src, ahash, aligned, allocator-api2, anstream, anstyle,
anstyle-parse, anstyle-query, anyhow, arrayvec, as-slice, async-trait,
atomic-waker, autocfg, base64, base64ct, bit_field, bitflags, bitstream-io,
candle-core, candle-metal-kernels, candle-nn, cc, cexpr, cfg-if, chacha20,
chrono, clap, clap_builder, clap_derive, clap_lex, cmake, colorchoice,
core-foundation, core-foundation-sys, crc32fast, crossbeam-deque,
crossbeam-epoch, crossbeam-utils, dary_heap, der, deranged, derive_builder,
derive_builder_core, derive_builder_macro, dirs, dirs-sys, displaydoc,
dyn-clone, either, enum-as-inner, enumflags2, enumflags2_derive, equivalent,
errno, fastrand, fdeflate, find-msvc-tools, find_cuda_helper, fixedbitset,
flate2, fnv, form_urlencoded, futures, futures-channel, futures-core,
futures-executor, futures-io, futures-macro, futures-sink, futures-task,
futures-util, getrandom, gif, glob, half, hashbrown, heck, home, http,
httparse, hyper-tls, ident_case, idna, idna_adapter, image, image-webp,
indexmap, ipnet, is_terminal_polyfill, itertools, itoa, jobserver,
lazy_static, lazycell, leiden-rs, libc, llama-cpp-2, llama-cpp-sys-2,
lock_api, log, matrixmultiply, memmap2, mime, minimal-lexical, monostate,
monostate-impl, native-tls, ndarray, no_std_io2, num-bigint, num-complex,
num-conv, num-derive, num-integer, num-rational, num-traits, num_cpus,
once_cell, ort, ort-sys, parking_lot, parking_lot_core, paste, pastey,
peeking_take_while, pem-rfc7468, percent-encoding, pin-project-lite,
pkg-config, png, portable-atomic, powerfmt, ppv-lite86, prettyplease,
proc-macro2, profiling, profiling-procmacros, qoi, quick-error, quote, rand,
rand_chacha, rand_core, rand_distr, rawpointer, rayon, rayon-cond,
rayon-core, ref-cast, ref-cast-impl, regex, regex-automata, regex-syntax,
reqwest, rustc-hash, rustls-pki-types, rustversion, scopeguard,
security-framework, security-framework-sys, seq-macro, serde, serde_core,
serde_derive, serde_derive_internals, serde_json, serde_spanned,
serde_urlencoded, shlex, signal-hook-registry, smallvec, socket2, socks,
stable_deref_trait, static_assertions, syn, system-configuration,
system-configuration-sys, tempfile, thiserror, thiserror-impl, thread_local,
time, time-core, toml, toml_datetime, toml_parser, toml_writer, typed-path,
unicode-normalization-alignments, unicode-segmentation, unicode-width,
unicode_categories, ureq, ureq-proto, url, utf8-zero, utf8_iter, utf8parse,
version_check, weezl, zeroize.

### MIT (91 crates)

aligned-vec, arg_enum_proc_macro, av-scenechange, bitvec, block2, built,
bytes, castaway, color_quant, compact_str, console, darling, darling_core,
darling_macro, dyn-stack, dyn-stack-macros, equator, equator-macro, fax,
float8, funty, gemm, gemm-c32, gemm-c64, gemm-common, gemm-f16, gemm-f32,
gemm-f64, h2, http-body, http-body-util, hyper, hyper-util, indicatif,
libm, libsql, libsql-ffi, libsql-sys, loop9, matchers, maybe-rayon, mio,
new_debug_unreachable, nom, noop_proc_macro, nu-ansi-term, objc2,
objc2-encode, objc2-foundation, onig, onig_sys, pulp, pulp-wasm-simd-flag,
radium, raw-cpuid, reborrow, rgb, schemars, schemars_derive, sharded-slab,
simd-adler32, simd_helpers, slab, strsim, synstructure, sysctl, tap, tiff,
tokio, tokio-macros, tokio-native-tls, tokio-util, tower, tower-http,
tower-layer, tower-service, tracing, tracing-attributes, tracing-core,
tracing-log, tracing-subscriber, try-lock, ulid, unit-prefix, want, which,
winnow, wyz, y4m, zip, zmij.

### Apache-2.0 (11 crates)

clang-sys, esaxx-rs, fastembed, hf-hub, lzma-rust2, rmcp, rmcp-macros,
safetensors, spm_precompiled, sync_wrapper, tokenizers.

### Unicode-3.0 (18 crates)

icu_collections, icu_locale_core, icu_normalizer, icu_normalizer_data,
icu_properties, icu_properties_data, icu_provider, litemap, potential_utf,
tinystr, writeable, yoke, yoke-derive, zerofrom, zerofrom-derive, zerotrie,
zerovec, zerovec-derive.

### Other permissive licenses (48 crates)

- MIT OR Apache-2.0 OR Zlib: bytemuck, bytemuck_derive, dispatch2,
  macro_rules_attribute, macro_rules_attribute-proc_macro, miniz_oxide,
  objc2-core-foundation, objc2-metal, zune-core, zune-inflate, zune-jpeg.
- MIT OR Unlicense: aho-corasick, byteorder, byteorder-lite, gryf,
  gryf-derive, memchr, same-file, walkdir.
- BSD-3-Clause: avif-serialize, bindgen, exr, lebe, ravif, subtle.
- ISC: hmac-sha256, libloading, rustls-webpki, untrusted.
- BSD-2-Clause: av1-grain, rav1e, v_frame.
- Apache-2.0 OR BSD-2-Clause OR MIT: zerocopy, zerocopy-derive.
- Apache-2.0 OR BSD-3-Clause: moxcms, pxfm.
- CDLA-Permissive-2.0: webpki-root-certs, webpki-roots.
- Zlib: foldhash.
- MPL-2.0: option-ext.
- 0BSD OR Apache-2.0 OR MIT: adler2.
- Apache-2.0 (with LLVM exception) OR Apache-2.0 OR MIT: rustix.
- Apache-2.0 OR BSL-1.0 (Boost): ryu.
- Apache-2.0 OR CC0-1.0: imgref.
- Apache-2.0 OR ISC OR MIT: rustls.
- (Apache-2.0 OR MIT) AND BSD-3-Clause, both apply: encoding_rs.
- (MIT OR Apache-2.0) AND Unicode-3.0, both apply: unicode-ident.
- Apache-2.0 AND ISC, both apply: ring.

## Named coverage

The plan for this migration calls out a few dependencies by name. Here is
where each one landed, confirmed against the current tree rather than
assumed:

- **llama.cpp**: vendored C++, MIT. See the callout above.
- **candle**: `candle-core`, `candle-nn`, and `candle-metal-kernels` are
  still in the tree, all MIT OR Apache-2.0. Candle's text-generation code
  path was removed when LIAM moved generation to llama.cpp; candle remains
  only as the runtime behind the local embedder.
- **fastembed and ort**: `fastembed` (Apache-2.0) and `ort` plus `ort-sys`
  (MIT OR Apache-2.0) are still in the tree; they run the local embedding
  and reranking models.
- **tokenizers**: `tokenizers` (Apache-2.0) is still in the tree, used by
  fastembed for text tokenization. It was not pruned.

No package in the tree is missing a `license` field, and none carries a
GPL, LGPL, or AGPL license.

## Full license texts

The full text of each Rust crate's license is available in that crate's own
repository (linked from its crates.io page) or inside the crate's source,
which `cargo vendor` or `cargo package --list` can retrieve locally. The
full text of llama.cpp's and ggml's MIT licenses is available in their
upstream repositories. For LIAM's own license terms, see this repository's
LICENSE file.

## Regenerating this list

Generated on 2026-08-11. To refresh after a dependency change, run:

```bash
export PATH="$HOME/.cargo/bin:$PATH"

# every crate that ships in the default build
cargo tree -e no-dev --prefix none | sort -u

# and the shipped optional surface
cargo tree -e no-dev --prefix none -p liam-daemon --features local,llama | sort -u

# license fields straight from the resolved metadata
cargo metadata --format-version 1 --filter-platform aarch64-apple-darwin --features local,llama \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);[print(f"{p[\"name\"]}\t{p.get(\"license\") or p.get(\"license_file\")}") for p in sorted(d["packages"],key=lambda p:p["name"])]'
```

Cross-reference the crate names from the two `cargo tree` runs against the
license fields from `cargo metadata`, then update the groups above. Watch
for any crate whose `license` field is empty (list it separately and mark
it for manual review) and for any GPL, LGPL, or AGPL license appearing
anywhere in the output; either would need attention before merging.
