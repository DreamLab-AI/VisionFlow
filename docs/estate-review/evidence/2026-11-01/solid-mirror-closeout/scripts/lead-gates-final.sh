#!/usr/bin/env bash
# Lead-run gates on the final Commit A tree. Usage: lead-gates-final.sh
set -u
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
OUT=$EV/lead-gates-final.txt
export CARGO_HOME=/home/devuser/workspace/.cargo
export CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target
export CARGO_BUILD_JOBS=4
cd "$W" || exit 2
git diff > "$EV/final-a.patch"
{
  echo "# $(date -u +%FT%TZ) HEAD $(git rev-parse --short HEAD) + final-a.patch sha256 $(sha256sum "$EV/final-a.patch" | cut -c1-16)"
  echo "# git diff --stat"; git diff --stat | tail -1
} > "$OUT"

# Test baseline: lines removed from the test module relative to fix-a.patch's tree.
TB=$(mktemp -d -p /home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad)
git -C "$W" show HEAD:crates/solid-pod-rs-forge/src/mirror.rs > "$TB/base.rs"
(cd "$TB" && git init -q . && cp base.rs mirror.rs && git add mirror.rs && git -c user.name=t -c user.email=t@t commit -qm b)
mkdir -p "$TB/crates/solid-pod-rs-forge/src"
cp "$TB/base.rs" "$TB/crates/solid-pod-rs-forge/src/mirror.rs"
(cd "$TB" && git apply --include='crates/solid-pod-rs-forge/src/mirror.rs' "$EV/fix-a/fix-a.patch" 2>>"$OUT")
awk '/^mod tests \{|^#\[cfg\(test\)\]/{f=1} f' "$TB/crates/solid-pod-rs-forge/src/mirror.rs" > "$TB/prev-tests.rs"
awk '/^mod tests \{|^#\[cfg\(test\)\]/{f=1} f' "$W/crates/solid-pod-rs-forge/src/mirror.rs" > "$TB/now-tests.rs"
echo "# test lines removed since fix-a.patch (whitespace-insensitive):" >> "$OUT"
git diff --no-index -w "$TB/prev-tests.rs" "$TB/now-tests.rs" | grep '^-[^-]' >> "$OUT"
echo "# end removed lines" >> "$OUT"
rm -rf "$TB"

run() { echo "===== \$ $*" >> "$OUT"; "$@" >> "$OUT" 2>&1; echo "===== exit=$?" >> "$OUT"; }
run cargo fmt --all --check
run cargo test --locked --offline -p solid-pod-rs-forge --features mirror
run cargo test --locked --offline -p solid-pod-rs-forge
run cargo test --locked --offline -p solid-pod-rs-forge --features mirror,anchoring,announce
run cargo test --locked --offline -p solid-pod-rs-server --features forge-full forge
run cargo clippy --locked --offline -p solid-pod-rs-forge --features mirror --all-targets -- -D warnings
run cargo clippy --locked --offline -p solid-pod-rs-server --features forge-full --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" run cargo doc --locked --offline --no-deps -p solid-pod-rs-forge --features mirror
echo "# done" >> "$OUT"
