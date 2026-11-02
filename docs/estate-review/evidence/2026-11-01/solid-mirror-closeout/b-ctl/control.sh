#!/usr/bin/env bash
# Commit B scratch controls (git 2.56). Run from anywhere; writes under $ROOT/run.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/run"
rm -rf "$ROOT"; mkdir -p "$ROOT"; cd "$ROOT"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null HOME="$ROOT"
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
q() { git "$@" >/dev/null 2>&1; }
has() { git -C "$1" cat-file -e "$2" 2>/dev/null && echo present || echo absent; }
RP="git -c core.hooksPath=/dev/null -c receive.fsckObjects=true receive-pack"

# upstream: main (approved) and side (held, off lineage, unique blob)
git init -q -b main up; cd up
echo a > a.txt; q add a.txt; q commit -m approved; APPROVED=$(git rev-parse HEAD)
q checkout -b side; echo SECRET > held.txt; q add held.txt; q commit -m held; HELD=$(git rev-parse HEAD)
HELD_BLOB=$(git rev-parse HEAD:held.txt); q checkout main; cd ..

# legacy served store (pre-release layout)
L=repos/$(printf 'a%.0s' {1..64})/sample.git
mkdir -p "$L"; q init --bare -b main "$L"
git -C "$L" config --add uploadpack.hideRefs refs/mirror
git -C "$L" config --add transfer.hideRefs refs/mirror
q -C "$L" fetch --no-tags "$ROOT/up" '+refs/heads/*:refs/mirror/*'
q -C "$L" update-ref refs/heads/main "$APPROVED"
# owner tag (annotated) on approved, a note, a PR head from a "fork", a ref reaching held
q -C "$L" tag -a v1 -m rel "$APPROVED"
git -C "$L" notes --ref=commits add -m "note on approved" "$APPROVED" >/dev/null 2>&1
git init -q -b feat forkup; ( cd forkup; echo f > f.txt; q add f.txt; q commit -m foreign ); FOREIGN=$(git -C forkup rev-parse HEAD)
q -C "$L" fetch --no-tags "$ROOT/forkup" "+refs/heads/feat:refs/forge/heads/bob/feat"
q -C "$L" update-ref refs/misc/stash-held "$HELD"
q -C "$L" repack -a -d   # one pack holding approved AND held, as a legacy gc would leave it
echo "== legacy store"; echo "held commit: $(has "$L" "$HELD")  held blob: $(has "$L" "$HELD_BLOB")"

# fork the way build_fork does it (before the move)
F=repos/bob/sample.git; mkdir -p repos/bob
q clone --local --bare -- "$L" "$F"; git -C "$F" config forge.parent "aaaa/sample"
PACK=$(ls "$L"/objects/pack/*.pack | head -1); FPACK=$(ls "$F"/objects/pack/*.pack | head -1); PACK_INODE=$(stat -c %i "$PACK")
# the fork owner also points a branch of their own at the held commit (possible: the fork holds it)
git -C "$F" update-ref refs/heads/republished "$HELD"
echo "== fork via clone --local --bare"
echo "fork held commit: $(has "$F" "$HELD")  held blob: $(has "$F" "$HELD_BLOB")  pack links: $(stat -c %h "$FPACK") same-inode: $([ "$(stat -c %i "$PACK")" = "$(stat -c %i "$FPACK")" ] && echo yes || echo no)"
echo "fork refs: $(git -C "$F" for-each-ref --format='%(refname)' | tr '\n' ' ')"

# 1. move out
mkdir -p mirror-recovery/aaaa; R=mirror-recovery/aaaa/sample.git.1; mv "$L" "$R"

# held set: commits reachable from refs/mirror and not from branches/tags
HSET=$(git -C "$R" rev-list $(git -C "$R" for-each-ref --format='%(refname)' refs/mirror) --not --branches --tags)
echo "held set: $HSET"
CONTAINS=$(for c in $HSET; do printf -- '--contains=%s ' "$c"; done)
echo "refs reaching held: $(git -C "$R" for-each-ref --format='%(refname)' $CONTAINS | tr '\n' ' ')"

rebuild() { # src dst [skip-regex]
  local src=$1 dst=$2 skip=$3; rm -rf "$dst"; mkdir -p "$(dirname "$dst")"; q init --bare -b main "$dst"
  local bad=$(git -C "$src" for-each-ref --format='%(refname)' $CONTAINS)
  local specs=()
  while read -r sha ref; do
    [[ $ref =~ $skip ]] && continue
    grep -qxF "$ref" <<<"$bad" && { echo "  dropped (reaches held): $ref"; continue; }
    specs+=("$sha:$ref")
  done < <(git -C "$src" for-each-ref --format='%(objectname) %(refname)')
  git -C "$src" push --porcelain --atomic --no-verify --receive-pack="$RP" -- "$(pwd)/$dst" "${specs[@]}" >/dev/null 2>"$dst.err" || { echo "  push failed: $(head -1 "$dst.err")"; }
}

echo "== A. fresh store + push of chosen refs (mirror: skip refs/mirror and refs/forge/heads)"
S=mirror-work/staging/aaaa/sample.git.x
rebuild "$R" "$S" '^refs/(mirror|forge/heads)/'
echo "held commit: $(has "$S" "$HELD")  held blob: $(has "$S" "$HELD_BLOB")  approved: $(has "$S" "$APPROVED")  foreign PR head: $(has "$S" "$FOREIGN")"
echo "refs: $(git -C "$S" for-each-ref --format='%(refname)' | tr '\n' ' ')"
echo "tag kept as annotated: $(git -C "$S" cat-file -t refs/tags/v1)  note: $(git -C "$S" notes --ref=commits show "$APPROVED" 2>&1)"
echo "alternates: $(ls "$S"/objects/info/alternates 2>/dev/null || echo none)  fsck: $(git -C "$S" fsck --unreachable --no-reflogs --connectivity-only 2>&1 | wc -l) unreachable lines"

echo "== B. negative controls on the recovery store"
q clone --local --bare -- "$R" negA.git; echo "clone --local --bare: held blob $(has negA.git "$HELD_BLOB") links $(stat -c %h $(ls negA.git/objects/pack/*.pack|head -1))"
q clone --bare -- "$R" negB.git; echo "clone --bare (path, implicit local): held blob $(has negB.git "$HELD_BLOB")"
q clone --mirror --no-local -- "$R" negC.git; echo "clone --mirror --no-local: held blob $(has negC.git "$HELD_BLOB")"
q clone --bare --no-local -- "$R" negD.git; echo "clone --bare --no-local: held blob $(has negD.git "$HELD_BLOB") (but refs: $(git -C negD.git for-each-ref --format='%(refname)'|tr '\n' ' '))"
q clone --bare --shared -- "$R" negE.git; echo "clone --shared: held blob $(has negE.git "$HELD_BLOB") alternates: $(cat negE.git/objects/info/alternates 2>/dev/null)"
cp -r "$R" negF.git; git -C negF.git for-each-ref --format='delete %(refname)' refs/mirror | git -C negF.git update-ref --stdin; git -C negF.git update-ref -d refs/misc/stash-held; q -C negF.git gc --prune=now; echo "ref deletion + gc --prune=now on a copy: held blob $(has negF.git "$HELD_BLOB") (reflogs: $(ls negF.git/logs 2>/dev/null | wc -l))"

echo "== C. fork rebuild the same way (all refs except refs/mirror, held check)"
mkdir -p mirror-recovery/bob; FR=mirror-recovery/bob/sample.git.1; mv "$F" "$FR"
FS=mirror-work/staging/bob/sample.git.y
rebuild "$FR" "$FS" '^refs/mirror/'
git -C "$FS" config forge.parent "$(git -C "$FR" config forge.parent)"
echo "fork rebuilt: held commit $(has "$FS" "$HELD") held blob $(has "$FS" "$HELD_BLOB") approved $(has "$FS" "$APPROVED") refs: $(git -C "$FS" for-each-ref --format='%(refname)'|tr '\n' ' ')"
echo "rebuilt fork objects sharing the legacy pack inode $PACK_INODE: $(find "$FS/objects" -type f -inum "$PACK_INODE" | wc -l)"
echo "== D. fork-swept-in-place alternative (repack -a -d + prune) on a copy of the fork"
cp -a "$FR" fswept.git; q -C fswept.git repack -a -d; q -C fswept.git prune --expire=now
echo "repack -a -d + prune: held blob $(has fswept.git "$HELD_BLOB")  (fork ref refs reach held? none here)"

echo "== E. hidden source refs: fork with uploadpack/transfer.hideRefs=refs/forge; push still copies them"
git -C "$FR" config --add uploadpack.hideRefs refs/forge; git -C "$FR" config --add transfer.hideRefs refs/forge
git -C "$FR" update-ref refs/forge/heads/carol/x "$APPROVED"
rebuild "$FR" fs2.git '^refs/mirror/'
echo "refs/forge/heads/carol/x copied: $(git -C fs2.git rev-parse -q --verify refs/forge/heads/carol/x >/dev/null && echo yes || echo no)"

echo "== F. hook planted in staging is not run with the constant receive-pack"
rm -rf hk.git; q init --bare -b main hk.git; printf '#!/bin/sh\ntouch %s/SENTINEL\n' "$ROOT" > hk.git/hooks/pre-receive; chmod +x hk.git/hooks/pre-receive
q -C "$R" push --no-verify --receive-pack="$RP" -- "$ROOT/hk.git" "$APPROVED:refs/heads/main"; echo "sentinel: $([ -e SENTINEL ] && echo RAN || echo not-run)"
