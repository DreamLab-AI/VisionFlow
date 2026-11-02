#!/usr/bin/env bash
# C16: approved blob reappearing in held commit; C17: fork then upstream force-push of held branch.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/run3"; rm -rf "$ROOT"; mkdir -p "$ROOT"; cd "$ROOT"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null HOME="$ROOT"
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
q() { git "$@" >/dev/null 2>&1; }
has() { git -C "$1" cat-file -e "$2" 2>/dev/null && echo present || echo absent; }
git init -q -b main up; cd up
echo old > f; q add f; q commit -m M1; OLDBLOB=$(git rev-parse HEAD:f)
echo new > f; q commit -am M2; M2=$(git rev-parse HEAD)
q checkout -b side; echo old > f; echo X1SECRET > x; q add f x; q commit -m X1; X1B=$(git rev-parse HEAD:x); q checkout main; cd ..
L=legacy.git; q init --bare -b main $L; git -C $L config --add uploadpack.hideRefs refs/mirror
q -C $L fetch --no-tags "$ROOT/up" '+refs/heads/*:refs/mirror/*'; q -C $L update-ref refs/heads/main $M2
echo "== C16 approximate H contains the approved M1 blob?"
HAPPROX=$(git -C $L rev-list --objects refs/mirror/side --not --branches --tags | cut -d' ' -f1)
echo "$HAPPROX" | grep -qx $OLDBLOB && echo "yes: approved blob $OLDBLOB is in H_approx" || echo "no"
ROOTS=$(git -C $L rev-list --objects --no-object-names --branches --tags)
HEXACT=$(comm -23 <(echo "$HAPPROX" | sort) <(echo "$ROOTS" | sort))
echo "$HEXACT" | grep -qx $OLDBLOB && echo "exact H still contains it" || echo "exact H (minus full root closure) excludes it; |H_approx|=$(echo "$HAPPROX"|wc -l) |H|=$(echo "$HEXACT"|wc -l)"
echo "== C17 fork while side=X1, then upstream force-pushes side to X2, legacy refetches"
q clone --local --bare -- $L fork.git
( cd up; q checkout side; q reset --hard main; echo X2 > y; q add y; q commit -m X2; q checkout main )
q -C $L fetch --no-tags "$ROOT/up" '+refs/heads/*:refs/mirror/*'
H2=$(comm -23 <(git -C $L rev-list --objects refs/mirror/side --not --branches --tags | cut -d' ' -f1 | sort) <(git -C $L rev-list --objects --no-object-names --branches --tags | sort))
echo "X1 blob in fork: $(has fork.git $X1B); X1 blob in parent's current H: $(echo "$H2" | grep -qx $X1B && echo yes || echo no)"
HITS=$(echo "$H2" | git -C fork.git cat-file --batch-check='%(objectname)' | grep -vc missing)
echo "H-trigger hits in fork: $HITS  (0 = fork would be skipped while serving X1)"
# dropped set D: all objects in parent minus objects reachable from copied roots
q init --bare -b main s.git; q -C $L push --no-verify -- "$ROOT/s.git" "$M2:refs/heads/main"
D=$(comm -23 <(git -C $L cat-file --batch-all-objects --batch-check='%(objectname)' | sort) <(git -C s.git cat-file --batch-all-objects --batch-check='%(objectname)' | sort))
echo "X1 blob in D: $(echo "$D" | grep -qx $X1B && echo yes || echo no); D-trigger hits in fork: $(echo "$D" | git -C fork.git cat-file --batch-check='%(objectname)' | grep -vc missing)"
