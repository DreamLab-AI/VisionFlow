#!/usr/bin/env bash
set -euo pipefail
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
export GIT_AUTHOR_NAME=Audit GIT_COMMITTER_NAME=Audit
export GIT_AUTHOR_EMAIL=audit@example.invalid GIT_COMMITTER_EMAIL=audit@example.invalid
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
up="$root/up"
stage="$root/private-stage.git"
served="$root/served.git"
legacy="$root/legacy.git"
git init --quiet -b main "$up"
printf 'synthetic base\n' > "$up/public.txt"
git -C "$up" add public.txt
git -C "$up" commit --quiet -m base
base=$(git -C "$up" rev-parse HEAD)
git -C "$up" switch --quiet -c held
printf 'synthetic held\n' > "$up/held.txt"
git -C "$up" add held.txt
git -C "$up" commit --quiet -m held
held=$(git -C "$up" rev-parse HEAD)
git -C "$up" switch --quiet -c approved "$base"
printf 'synthetic approved\n' > "$up/approved.txt"
git -C "$up" add approved.txt
git -C "$up" commit --quiet -m approved
approved=$(git -C "$up" rev-parse HEAD)
git -C "$up" gc --quiet
for dir in "$stage" "$served" "$legacy"; do git init --bare --quiet "$dir"; done
git -C "$stage" -c transfer.fsckObjects=true fetch --quiet --no-tags "file://$up" '+refs/heads/*:refs/mirror/*'
git -C "$stage" cat-file -e "$held:held.txt"
git -C "$stage" -c receive.fsckObjects=true push --quiet --force "$served" "$approved:refs/heads/main"
test "$(git -C "$served" show "$approved:approved.txt")" = 'synthetic approved'
if git -C "$served" cat-file -e "$held:held.txt" 2>/dev/null; then
  printf 'FAIL: unrelated held object reached fresh served store\n'; exit 1
fi
printf 'PASS: staging contains held bytes; fresh approved-only publication does not\n'
# Simulate an old mirror that already fetched a held branch.
git -C "$legacy" fetch --quiet --no-tags "file://$up" '+refs/heads/*:refs/mirror/*'
git -C "$legacy" update-ref refs/heads/main "$approved"
while IFS= read -r ref; do git -C "$legacy" update-ref -d "$ref"; done < <(git -C "$legacy" for-each-ref '--format=%(refname)' refs/mirror)
git -C "$stage" push --quiet --force "$legacy" "$approved:refs/heads/main"
test "$(git -C "$legacy" show "$held:held.txt")" = 'synthetic held'
printf 'PASS: deleting old hidden refs and adopting approved-only pushes leaves legacy held bytes readable\n'
printf 'Scope: local Git subprocess design control, not a Forge repair or concurrent-read acceptance\n'
