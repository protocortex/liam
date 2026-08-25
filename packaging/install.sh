#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Installs LIAM from an unpacked netinstall tarball.
#
#   tar -xzf liam-vX.Y.Z-aarch64-apple-darwin-netinstall.tar.gz -C liam
#   cd liam && ./install.sh
#
# Order matters here and is not arbitrary. Models are fetched BEFORE the
# launchd job is installed, because a daemon whose weights are missing does
# not fail visibly: launchd retries it, and a client that connects just waits.
# Failing during an install the user is watching is the only version of that
# failure anyone can act on.
#
# Options:
#   --prefix DIR    where the binaries go (default ~/.local/bin)
#   --skip-models   install everything but do not download weights
#   --no-launchd    install everything but do not register the daemon
set -eu

PREFIX="${HOME}/.local/bin"
SKIP_MODELS=0
NO_LAUNCHD=0

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)
            # Rejecting a leading dash matters more than the arity check: with
            # only the count tested, `--prefix --skip-models` consumes both
            # tokens, sets PREFIX to "--skip-models", and then downloads the
            # weights the user just asked to skip.
            case "${2-}" in
                ''|-*) echo "install.sh: --prefix needs a directory" >&2; exit 2 ;;
            esac
            PREFIX="$2"
            shift 2
            ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --no-launchd)  NO_LAUNCHD=1; shift ;;
        -h|--help)
            # Prints the header block above: skip the shebang and the SPDX
            # line, strip the comment marker, stop at the first line of real
            # code. A hardcoded line range would silently start printing `set
            # -eu` the first time someone added a sentence up there.
            awk 'NR>2 && /^#/ { sub(/^# ?/, ""); print; next } NR>2 { exit }' "$0"
            exit 0
            ;;
        *)
            echo "install.sh: unknown option $1" >&2
            exit 2
            ;;
    esac
done

# A relative prefix has to become absolute HERE, before it reaches the plist.
# launchd resolves a relative ProgramArguments path against WorkingDirectory,
# which the plist pins to ~/.liam, so `--prefix bin` would install to
# ./bin/liamd and then tell launchd to run ~/.liam/bin/liamd. With RunAtLoad
# false there is no symptom at install time: this script prints "done", and the
# daemon simply never spawns when a client finally connects.
case "$PREFIX" in
    /*) ;;
    *)  PREFIX="$(pwd)/${PREFIX}" ;;
esac

SRC="$(cd "$(dirname "$0")" && pwd)"
LIAM_HOME="${HOME}/.liam"
CONFIG="${LIAM_HOME}/liam.toml"
PLIST_LABEL="dev.protocortex.liamd"
PLIST="${HOME}/Library/LaunchAgents/${PLIST_LABEL}.plist"

# v1 ships one target. Refusing here beats letting the user discover it from
# a "bad CPU type in executable" three steps later.
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) ;;
    *)
        echo "install.sh: this build is macOS on Apple Silicon only, found $(uname -s)/$(uname -m)" >&2
        exit 1
        ;;
esac

for binary in liamd liam; do
    [ -f "${SRC}/${binary}" ] || {
        echo "install.sh: ${binary} is missing from ${SRC}; is this the unpacked tarball?" >&2
        exit 1
    }
done

# `mkdir -p ~/.liam` is not tidiness: it has to exist before launchd binds the
# socket inside it and before the job's WorkingDirectory resolves.
mkdir -p "$LIAM_HOME" "$PREFIX"

# Printed resolved rather than as typed, so a relative --prefix shows where it
# actually landed.
echo "installing liamd and liam into ${PREFIX}"
for binary in liamd liam; do
    install -m 0755 "${SRC}/${binary}" "${PREFIX}/${binary}"
    # A tarball downloaded through a browser carries com.apple.quarantine, and
    # Archive Utility propagates it to what it extracts, so an unsigned binary
    # gets killed on first run with a dialog that says nothing useful. curl
    # sets no such attribute, so this is a no-op for most people; `|| true`
    # because `xattr -d` fails when the attribute was never there.
    xattr -d com.apple.quarantine "${PREFIX}/${binary}" 2>/dev/null || true
done

# Never clobber a config someone has edited. An upgrade that silently reset
# model choices, retention, or the socket path would be far worse than an
# upgrade that leaves a stale key behind.
if [ -f "$CONFIG" ]; then
    echo "keeping your existing ${CONFIG}"
    echo "  (compare it against ${SRC}/liam.toml if this is an upgrade)"
else
    install -m 0644 "${SRC}/liam.toml" "$CONFIG"
    echo "wrote ${CONFIG}"
fi

if [ "$SKIP_MODELS" -eq 1 ]; then
    echo "skipping model download; run '${PREFIX}/liam fetch-models --config ${CONFIG}' before starting the daemon"
else
    echo "fetching models; the first run downloads several gigabytes and loads each one to verify it"
    "${PREFIX}/liam" fetch-models --config "$CONFIG"
fi

if [ "$NO_LAUNCHD" -eq 1 ]; then
    echo "skipping launchd registration"
    echo "done. Start the daemon yourself with '${PREFIX}/liamd --config ${CONFIG} serve'."
    exit 0
fi

mkdir -p "$(dirname "$PLIST")"
# Two substitutions, most specific first. launchd expands neither `~` nor
# environment variables inside SockPathName or ProgramArguments, so every path
# has to be literal by the time launchd reads the file. The first expression
# also redirects the binary path when --prefix moved it.
sed -e "s|__HOME__/.local/bin/liamd|${PREFIX}/liamd|g" \
    -e "s|__HOME__|${HOME}|g" \
    "${SRC}/${PLIST_LABEL}.plist" > "$PLIST"

# Reinstalling over a registered job: launchd refuses to bootstrap a label it
# already holds, and would leave the old plist live. Booting it out first is
# what makes this script safe to run twice. It fails when nothing is loaded,
# which is the normal first-install case, hence `|| true`.
launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo
echo "done. The daemon starts on demand, on the first client connection."
echo "  socket: ${LIAM_HOME}/liamd.sock"
echo "  log:    ${LIAM_HOME}/liamd.log"
echo
echo "Point an MCP client at it with:  ${PREFIX}/liamd --config ${CONFIG} proxy"
