#!/usr/bin/env bash
# Download the published netinstall archive again and check it the way a user
# and the Homebrew tap read it. Run after the release job, on any OS with curl,
# tar and sha256sum or shasum.
#
# Usage: verify-release.sh <vX.Y.Z>
#
# It fails unless all of these hold:
#   - SHA256SUMS has exactly one line for the archive, with a 64 digit checksum
#   - the downloaded archive matches that checksum
#   - the per-archive .sha256 file names the same checksum
#   - the archive holds the files install.sh expects
#
# No build attestation is checked, because the release workflow does not
# produce one.
#
# DOWNLOAD_BASE reads the files from another place, such as a file:// folder in
# a local test.
set -euo pipefail

tag="${1:-}"
if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]]; then
  echo "usage: verify-release.sh <vX.Y.Z>" >&2
  exit 2
fi
asset="liam-${tag}-aarch64-apple-darwin-netinstall.tar.gz"
base="${DOWNLOAD_BASE:-https://github.com/protocortex/liam/releases/download/${tag}}"

fail() { echo "::error::$*" >&2; exit 1; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT
for f in "$asset" SHA256SUMS "${asset}.sha256"; do
  curl -fsSL --retry 3 --retry-delay 5 -o "${dir}/${f}" "${base}/${f}" || fail "could not download ${f} of ${tag}"
done

lines="$(awk -v f="$asset" '{ n = $2; sub(/^\*/, "", n) } n == f { print $1 }' "${dir}/SHA256SUMS")"
[[ "$(wc -l <<<"$lines")" -eq 1 && "$lines" =~ ^[0-9a-f]{64}$ ]] \
  || fail "SHA256SUMS of ${tag} needs exactly one 64 digit line for ${asset}"

actual="$(sha256 "${dir}/${asset}")"
[[ "$actual" = "$lines" ]] || fail "${asset} hashes to ${actual}, but SHA256SUMS says ${lines}"

other="$(cut -d' ' -f1 <"${dir}/${asset}.sha256")"
[[ "$other" = "$lines" ]] || fail "${asset}.sha256 says ${other}, but SHA256SUMS says ${lines}"

listing="$(tar -tzf "${dir}/${asset}")"
for entry in ./liamd ./liam ./install.sh ./liam.toml ./dev.protocortex.liamd.plist; do
  grep -Fxq "$entry" <<<"$listing" || fail "${asset} is missing ${entry}"
done
echo "${asset}: checksum ${actual} matches SHA256SUMS and ${asset}.sha256, and the archive is complete."
