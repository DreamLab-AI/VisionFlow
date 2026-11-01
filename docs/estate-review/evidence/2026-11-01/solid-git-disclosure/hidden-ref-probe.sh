#!/usr/bin/env bash
set -euo pipefail
probe_root=$(mktemp -d /tmp/vf-git-hidden-XXXXXX)
trap 'rm -rf "$probe_root"' EXIT
repo="$probe_root/repo"
mkdir "$repo"
git -C "$repo" init -q -b main
git -C "$repo" config user.name 'Audit Fixture'
git -C "$repo" config user.email 'audit@example.invalid'
printf 'synthetic-public\n' > "$repo/public.txt"
git -C "$repo" add public.txt
git -C "$repo" commit -qm public
public_commit=$(git -C "$repo" rev-parse HEAD)
printf 'synthetic-held-back\n' > "$repo/held.txt"
git -C "$repo" add held.txt
git -C "$repo" commit -qm held
held_commit=$(git -C "$repo" rev-parse HEAD)
git -C "$repo" update-ref refs/mirror/held "$held_commit"
git -C "$repo" reset -q --hard "$public_commit"
git -C "$repo" config uploadpack.hideRefs refs/mirror
git -C "$repo" config transfer.hideRefs refs/mirror
printf -v want 'want %s\n' "$held_commit"
printf '%04x%s00000009done\n' "$(( ${#want} + 4 ))" "$want" > "$probe_root/request"
# Hidden tips are absent from the advertisement under both settings.
git -C "$repo" upload-pack --stateless-rpc --advertise-refs . > "$probe_root/advert"
if rg -a -q "$held_commit" "$probe_root/advert"; then exit 10; fi
for setting in false true; do
  set +e
  GIT_CONFIG_PARAMETERS="'uploadpack.allowTipSHA1InWant=$setting' 'core.symlinks=false'" git -C "$repo" upload-pack --stateless-rpc . < "$probe_root/request" > "$probe_root/response-$setting" 2> "$probe_root/error-$setting"
  result=$?
  set -e
  printf 'allowTipSHA1InWant=%s exit=%s hidden_tip_advertised=false\n' "$setting" "$result"
  if [[ "$setting" == false ]]; then
    test "$result" -ne 0
    sed -n '1p' "$probe_root/error-$setting"
  else
    test "$result" -eq 0
    # The v0 response is NAK pkt-line followed by a standard packfile.
    test "$(head -c 8 "$probe_root/response-$setting")" = '0008NAK'
    mkdir "$probe_root/receiver"
    git -C "$probe_root/receiver" init -q
    tail -c +9 "$probe_root/response-$setting" | git -C "$probe_root/receiver" index-pack --stdin > /dev/null
    actual=$(git -C "$probe_root/receiver" show "$held_commit:held.txt")
    test "$actual" = synthetic-held-back
    printf 'Decoded held-back synthetic bytes from received pack.\n'
  fi
done
