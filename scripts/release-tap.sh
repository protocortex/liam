#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Move the Homebrew tap to a liam release with a pull request, and wait for the
# tap to tag and publish its own release.
#
# Usage: release-tap.sh <X.Y.Z> <tap-checkout>
#
# <tap-checkout> is a checkout of protocortex/homebrew-tap whose Formula/liam.rb
# the `formulas` job of the release workflow rewrote. Nothing in that checkout
# runs here: the script only reads the file and sends it to the API. Because
# the tap's own script wrote that file, this script first checks that exactly
# the stable url and sha256 changed, and that the sha256 is the one in the
# release's SHA256SUMS.
#
# Environment:
#   GH_TOKEN       a token that can write contents and pull requests in the tap,
#                  and read its Actions runs. Not needed for a dry run.
#   DRY_RUN        "true" validates, prints the diff and opens nothing.
#   TAP            the tap repository, default protocortex/homebrew-tap.
#   DOWNLOAD_BASE  reads SHA256SUMS from another place, such as a file:// folder
#                  in a local test. The expected url keeps the real address.
set -euo pipefail

version="${1:-}"
tap_dir="${2:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ || ! -d "${tap_dir}/.git" ]]; then
  echo "usage: release-tap.sh <X.Y.Z> <tap-checkout>" >&2
  exit 2
fi
tag="v${version}"
branch="release/${tag}"
tap="${TAP:-protocortex/homebrew-tap}"
dry="${DRY_RUN:-false}"
formula="Formula/liam.rb"
asset="liam-${tag}-aarch64-apple-darwin-netinstall.tar.gz"
url_base="https://github.com/protocortex/liam/releases/download/${tag}"
title="chore(release): liam ${version}"

say() { echo "$*" >&2; if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "$*" >>"$GITHUB_STEP_SUMMARY"; fi; }
fail() { echo "::error::$*" >&2; exit 1; }

# 1. Only the formula may differ from the tap's main.
status="$(git -C "$tap_dir" status --porcelain)"
case "$status" in
  "") changed=false ;;
  " M ${formula}") changed=true ;;
  *) fail "the tap checkout differs in more than ${formula}: ${status}" ;;
esac

# 2. Exactly the stable url and sha256 changed, to the values of this release.
sha=""
if sums="$(curl -fsSL --retry 3 --retry-delay 5 "${DOWNLOAD_BASE:-$url_base}/SHA256SUMS")"; then
  sha="$(awk -v f="$asset" '{ n = $2; sub(/^\*/, "", n) } n == f { print $1 }' <<<"$sums")"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || fail "SHA256SUMS of ${tag} has no valid line for ${asset}"
elif [ "$dry" != true ]; then
  fail "could not read SHA256SUMS of ${tag}"
else
  say "SHA256SUMS of ${tag} is not published, so this dry run checks the shape of the change only."
fi
if [ "$changed" = true ]; then
  diff="$(git -C "$tap_dir" diff -U0 -- "$formula")"
  # One or two lines changed: the url, the sha256, or both. A formula already at
  # this version changes only the sha256, which is the case of a dry run that
  # reads a placeholder checksum.
  mapfile -t added < <(grep -E '^\+[^+]' <<<"$diff" || true)
  mapfile -t removed < <(grep -E '^-[^-]' <<<"$diff" || true)
  if [[ "${#added[@]}" -lt 1 || "${#added[@]}" -gt 2 || "${#added[@]}" -ne "${#removed[@]}" ]]; then
    fail "the change touches ${#added[@]} added and ${#removed[@]} removed lines, not the stable url and sha256"
  fi
  want_url="+  url \"${url_base}/${asset}\""
  for line in "${added[@]}"; do
    if [[ "$line" = "$want_url" ]]; then continue; fi
    [[ "$line" =~ ^\+\ \ sha256\ \"[0-9a-f]{64}\"$ ]] || fail "unexpected added line: ${line}"
    if [ -n "$sha" ] && [[ "$line" != "+  sha256 \"${sha}\"" ]]; then
      fail "the sha256 in the formula is not the one in SHA256SUMS (${sha})"
    fi
  done
  for line in "${removed[@]}"; do
    [[ "$line" == '-  url "https://github.com/protocortex/liam/releases/download/'* || "$line" == '-  sha256 "'* ]] \
      || fail "unexpected removed line: ${line}"
  done
  # Whatever the diff looks like, the file must now hold the release's url and,
  # when SHA256SUMS is known, its checksum, each once.
  file="${tap_dir}/${formula}"
  [[ "$(grep -cxF "  url \"${url_base}/${asset}\"" "$file")" -eq 1 ]] || fail "${formula} does not hold the url of ${tag} exactly once"
  if [ -n "$sha" ]; then
    [[ "$(grep -cxF "  sha256 \"${sha}\"" "$file")" -eq 1 ]] || fail "${formula} does not hold the checksum of ${tag} exactly once"
  fi
  say "Formula change for ${tag} (${tap} on ${branch}):"
  say '~~~diff'
  say "$diff"
  say '~~~'
else
  say "${tap} already carries ${tag}, so no pull request is needed."
fi

if [ "$dry" = true ]; then
  say "Dry run: nothing was opened."
  exit 0
fi

wait_release() {
  local _
  for _ in $(seq 1 30); do
    if gh release view "$tag" --repo "$tap" >/dev/null 2>&1; then
      say "Tap release: https://github.com/${tap}/releases/tag/${tag}"
      return 0
    fi
    sleep 10
  done
  fail "${tap} has no release ${tag} five minutes after the merge. Check its release-tag run, then run this workflow again for ${tag}."
}
if [ "$changed" = false ]; then
  wait_release
  exit 0
fi

# 3. Open the pull request, or reuse the open one for this release.
pr="$(gh pr list --repo "$tap" --head "$branch" --state open --json number --jq '.[0].number // empty')"
if [ -z "$pr" ]; then
  # A branch left by a closed pull request or a failed run.
  if gh api "repos/${tap}/git/ref/heads/${branch}" >/dev/null 2>&1; then
    gh api -X DELETE "repos/${tap}/git/refs/heads/${branch}"
  fi
  base_sha="$(git -C "$tap_dir" rev-parse HEAD)"
  gh api -X POST "repos/${tap}/git/refs" -f "ref=refs/heads/${branch}" -f "sha=${base_sha}" >/dev/null
  # GitHub signs commits made through createCommitOnBranch.
  payload="$(mktemp)"
  trap 'rm -f "$payload"' EXIT
  jq -n --arg repo "$tap" --arg branch "$branch" --arg head "$base_sha" --arg msg "$title" \
    --arg path "$formula" --arg content "$(base64 <"${tap_dir}/${formula}" | tr -d '\n')" \
    '{query: "mutation($input: CreateCommitOnBranchInput!) { createCommitOnBranch(input: $input) { commit { oid } } }",
      variables: {input: {
        branch: {repositoryNameWithOwner: $repo, branchName: $branch},
        message: {headline: $msg},
        expectedHeadOid: $head,
        fileChanges: {additions: [{path: $path, contents: $content}]}
      }}}' >"$payload"
  oid="$(gh api graphql --input "$payload" --jq '.data.createCommitOnBranch.commit.oid // empty')"
  [ -n "$oid" ] || fail "the API did not create the commit on ${branch}"
  pr="$(gh api -X POST "repos/${tap}/pulls" -f "title=${title}" -f "head=${branch}" -f base=main \
    -f "body=Moves the stable formula to liam ${version}, with the url and checksum from the SHA256SUMS of https://github.com/protocortex/liam/releases/tag/${tag}. Opened by the liam release workflow. It merges once the checks pass, and release-tag then tags this repository ${tag}." \
    --jq .number)"
  say "Opened ${tap}#${pr} with commit ${oid}."
else
  oid="$(gh pr view "$pr" --repo "$tap" --json headRefOid --jq .headRefOid)"
  say "Reusing ${tap}#${pr} at ${oid}."
fi

# 4. Wait for the pull request checks.
run_id=""
for _ in $(seq 1 36); do
  run_id="$(gh run list --repo "$tap" --workflow tests.yml --commit "$oid" --event pull_request --json databaseId --jq '.[0].databaseId // empty')"
  [ -z "$run_id" ] || break
  sleep 5
done
[ -n "$run_id" ] || fail "no tests run appeared for ${oid} in ${tap}#${pr}"
say "Checks: https://github.com/${tap}/actions/runs/${run_id}"
if [ "$(gh run view "$run_id" --repo "$tap" --json conclusion --jq .conclusion)" = action_required ]; then
  fail "the tests run is waiting for approval. Approve it in ${tap}, then run this workflow again for ${tag}."
fi
gh run watch "$run_id" --repo "$tap" --exit-status --interval 20 >/dev/null

# 5. Squash merge. The merge can answer 405 for a moment after the checks end.
merged=false
for attempt in $(seq 1 20); do
  if gh api -X PUT "repos/${tap}/pulls/${pr}/merge" -f merge_method=squash \
    -f "commit_title=${title} (#${pr})" -f commit_message= >/dev/null; then
    merged=true
    break
  fi
  echo "Merge attempt ${attempt} was refused, retrying in 15s." >&2
  sleep 15
done
[ "$merged" = true ] || fail "${tap}#${pr} could not be merged"
say "Merged ${tap}#${pr}."

# 6. The tap tags the merge commit and publishes its release by itself.
wait_release
