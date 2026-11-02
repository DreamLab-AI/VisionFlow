#!/usr/bin/env bash
# Commit B2 on fix/mirror-fork-sweep, then merge to local main. Usage: commit-b2.sh
set -eu
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
P=/home/devuser/workspace/solid-pod-rs
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
GATED=1997038475c93f31
cd "$W"
test "$(git branch --show-current)" = fix/mirror-fork-sweep
test -z "$(git status --porcelain --ignored | grep -v '^ M \|^A  \|^ A ')"
now=$(git diff | sha256sum | cut -c1-16)
test "$now" = "$GATED" || { echo "tree moved: $now"; exit 1; }
git add -- CHANGELOG.md crates/solid-pod-rs-forge/src/forks.rs crates/solid-pod-rs-forge/src/lib.rs \
  crates/solid-pod-rs-forge/src/migrate.rs crates/solid-pod-rs-forge/src/migrate_fork.rs \
  crates/solid-pod-rs-forge/src/migrate_fork_tests.rs crates/solid-pod-rs-forge/src/migrate_tests.rs
staged=$(git diff --cached | sha256sum | cut -c1-16)
echo "staged=$staged gated=$GATED"
test "$staged" = "$GATED"
git commit -q -F "$SP/commit-b2-msg.txt"
test -z "$(git status --porcelain)"
git show --stat --format='%h %s' HEAD | tail -9
cd "$P"
test "$(git branch --show-current)" = main
test -z "$(git status --porcelain)"
git log --oneline 8ba3372..main
git merge-tree --write-tree main fix/mirror-fork-sweep >/dev/null && echo merge-clean
git merge --no-ff -q fix/mirror-fork-sweep -m "merge: fork migration, SP-GIT-02 commit B2

Not pushed or deployed.

Co-Authored-By: jjohare <github@thedreamlab.uk>"
git log -1 --format='%h %s'
git -C "$W" switch -q --detach main
git branch -d fix/mirror-fork-sweep
