#!/bin/bash
# usage: run1.sh MODEL DEVICE OUTPREFIX (OUTPREFIX must be an absolute path)
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${LIAM_REPO:?set LIAM_REPO to a checkout with target/release/liam built}"
/usr/bin/time -l ./target/release/liam --config "$HERE/bench.toml" eval embedder --model "$1" --dims 384 --device "$2" --json --vectors-out "$3.vec.json" > "$3.json" 2> "$3.err"
