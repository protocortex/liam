#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The default build of liam-daemon links no TLS. This fails when a normal
# dependency of the default feature set brings in a TLS stack, and prints the
# crates the `hosted` feature adds so a reviewer sees what turning it on costs.
#
# It guards only the default `cargo build`: `local` and `llama` already carry a
# TLS-capable ureq through hf-hub, so the feature flag is not the egress control.
# The control is the privacy gate and the host allowlist.
set -euo pipefail

TLS_CRATES='rustls|ring|webpki-roots|native-tls'

tls_crates() {
    cargo tree -e normal -p liam-daemon --prefix none "$@" \
        | awk '{print $1}' | sort -u | grep -xE "${TLS_CRATES}" || true
}

default_tls="$(tls_crates)"
if [ -n "${default_tls}" ]; then
    echo "the default build of liam-daemon links TLS crates:" >&2
    echo "${default_tls}" >&2
    exit 1
fi
echo "default build: no TLS crates"

echo "hosted feature adds:"
tls_crates --features hosted | sed 's/^/  /'
