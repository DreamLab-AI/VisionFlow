#!/usr/bin/env bash
# Follow-up controls: transitive forks, blob refs, .keep packs, missing --contains.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/run2"; rm -rf "$ROOT"; mkdir -p "$ROOT"; cd "$ROOT"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null HOME="$ROOT"
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
q() { git "$@" >/dev/null 2>&1; }
has() { git -C "$1" cat-file -e "$2" 2>/dev/null && echo present || echo absent; }
RP="git -c core.hooksPath=/dev/null -c receive.fsckObjects=true receive-pack"
git init -q -b main up; cd up; echo a > a.txt; q add a.txt; q commit -m ok; A=$(git rev-parse HEAD)
q checkout -b side; echo SECRET > held.txt; q add held.txt; q commit -m held; H=$(git rev-parse HEAD); HB=$(git rev-parse HEAD:held.txt); HT=$(git rev-parse HEAD^{tree}); q checkout main; cd ..
L=legacy.git; q init --bare -b main $L; git -C $L config --add uploadpack.hideRefs refs/mirror
q -C $L fetch --no-tags "$ROOT/up" '+refs/heads/*:refs/mirror/*'; q -C $L update-ref refs/heads/main $A; q -C $L repack -a -d

echo "== 1. fork of a fork (clone --local of a clone --local) hardlinks held objects"
q clone --local --bare -- $L f1.git; q clone --local --bare -- f1.git f2.git
P=$(ls $L/objects/pack/*.pack); echo "f2 held blob: $(has f2.git $HB)  f2 pack inode == legacy pack inode: $([ "$(stat -c %i $P)" = "$(stat -c %i $(ls f2.git/objects/pack/*.pack))" ] && echo yes || echo no)"

echo "== 2. a ref pointing straight at a held blob or tree escapes --contains"
q -C $L update-ref refs/misc/blob $HB; q -C $L update-ref refs/misc/tree $HT
echo "for-each-ref --contains=$H: $(git -C $L for-each-ref --format='%(refname)' --contains=$H | tr '\n' ' ')"
q init --bare -b main s.git
q -C $L push --no-verify --receive-pack="$RP" -- "$ROOT/s.git" "$A:refs/heads/main" "$HB:refs/misc/blob" "$HT:refs/misc/tree"
echo "after copying refs/misc/*: held blob in staging: $(has s.git $HB)"
HELDOBJ=$(git -C $L rev-list --objects $(git -C $L for-each-ref --format='%(refname)' refs/mirror) --not --branches --tags | cut -d' ' -f1)
echo "held object set (rev-list --objects refs/mirror --not branches tags): $(echo $HELDOBJ | wc -w) objects"
echo "batch-check of held set in staging, hits: $(printf '%s\n' $HELDOBJ | git -C s.git cat-file --batch-check='%(objectname)' 2>/dev/null | grep -vc missing)"

echo "== 3. --contains with a commit the store lacks"
q init --bare -b main empty.git; q -C $L push --no-verify --receive-pack="$RP" -- "$ROOT/empty.git" "$A:refs/heads/main"
git -C empty.git for-each-ref --contains=$H >/dev/null 2>err.txt; echo "exit $? : $(head -1 err.txt)"

echo "== 4. mutant: delete refs/mirror + gc --prune=now in place, with a .keep pack"
cp -a $L m.git; git -C m.git update-ref -d refs/misc/blob; git -C m.git update-ref -d refs/misc/tree
touch $(ls m.git/objects/pack/*.pack | sed 's/\.pack$/.keep/')
git -C m.git for-each-ref --format='delete %(refname)' refs/mirror | git -C m.git update-ref --stdin
q -C m.git gc --prune=now; echo "held blob after gc with .keep: $(has m.git $HB)"
cp -a $L m2.git; q -C m2.git update-ref refs/misc/stash $H
git -C m2.git for-each-ref --format='delete %(refname)' refs/mirror | git -C m2.git update-ref --stdin
q -C m2.git gc --prune=now; echo "held blob after gc with a non-mirror ref reaching held: $(has m2.git $HB)"
