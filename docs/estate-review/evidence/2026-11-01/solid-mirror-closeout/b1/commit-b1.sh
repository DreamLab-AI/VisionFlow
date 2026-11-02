#!/usr/bin/env bash
# Commit B1 on fix/mirror-migration. Usage: commit-b1.sh
set -eu
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
cd "$W"
git add -- CHANGELOG.md crates/solid-pod-rs-forge/src/forks.rs crates/solid-pod-rs-forge/src/lib.rs \
  crates/solid-pod-rs-forge/src/migrate.rs crates/solid-pod-rs-forge/src/migrate_tests.rs \
  crates/solid-pod-rs-forge/src/mirror.rs crates/solid-pod-rs-forge/src/mirror_record.rs \
  crates/solid-pod-rs-forge/src/router.rs crates/solid-pod-rs-server/src/lib.rs \
  crates/solid-pod-rs-server/tests/forge_migration_runtime.rs
git diff --cached > "$EV/b1/final-b1.patch"
echo "staged sha256: $(sha256sum "$EV/b1/final-b1.patch" | cut -c1-16)"
echo "== files differing from the gated fix3 tree (expect only CHANGELOG.md):"
git diff --no-index "$EV/b1/fix3/after-fix3.patch" "$EV/b1/final-b1.patch" | grep '^[-+]+++ b/' | sort -u || true
git diff --no-index "$EV/b1/fix3/after-fix3.patch" "$EV/b1/final-b1.patch" | grep '^[-+]diff --git' | sort -u || true
git commit -q -F "$SP/commit-b1-msg.txt"
echo "== commit: $(git log -1 --format='%H %s')"
echo "== status after (expect empty):"; git status --porcelain
git show --stat HEAD | tail -12
