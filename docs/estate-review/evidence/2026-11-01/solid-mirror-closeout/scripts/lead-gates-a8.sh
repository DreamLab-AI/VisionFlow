#!/usr/bin/env bash
# Lead-run gates on the round-4 Commit A tree. Usage: lead-gates-a8.sh
set -u
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
OUT=$EV/lead-gates-a8.txt
export CARGO_HOME=/home/devuser/workspace/.cargo
export CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target
export CARGO_BUILD_JOBS=4
cd "$W" || exit 2
git diff > "$EV/final-a8.patch"
{
  echo "# $(date -u +%FT%TZ) HEAD $(git rev-parse --short HEAD) + final-a8.patch sha256 $(sha256sum "$EV/final-a8.patch" | cut -c1-16)"
  echo "# git diff --stat"; git diff --stat | tail -1
  echo "# lead edits since final-a7.patch (test + CHANGELOG):"; git diff --no-index "$EV/final-a7.patch" "$EV/final-a8.patch" | grep "^[-+][-+ ]" | grep -v "^[-+][-+][-+] " | head -80
  echo "# removed lines in the round-4 interdiff that contain assert/expect/panic:"
  git diff --no-index "$EV/final-a7.patch" "$EV/final-a8.patch" | grep "^-[-+].*\(assert\|expect\|panic\)"
  echo "# end removed lines"
} > "$OUT"

run() { echo "===== \$ $*" >> "$OUT"; "$@" >> "$OUT" 2>&1; echo "===== exit=$?" >> "$OUT"; }
run cargo fmt --all --check
run cargo test --locked --offline -p solid-pod-rs-forge --features mirror
run cargo test --locked --offline -p solid-pod-rs-forge
run cargo test --locked --offline -p solid-pod-rs-forge --features mirror,anchoring,announce
run cargo clippy --locked --offline -p solid-pod-rs-forge --features mirror,anchoring,announce --all-targets -- -D warnings
run cargo test --locked --offline -p solid-pod-rs-server --features forge-full forge
run cargo clippy --locked --offline -p solid-pod-rs-forge --features mirror --all-targets -- -D warnings
run cargo clippy --locked --offline -p solid-pod-rs-server --features forge-full --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" run cargo doc --locked --offline --no-deps -p solid-pod-rs-forge --features mirror
echo "# done" >> "$OUT"
