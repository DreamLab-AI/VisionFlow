#!/usr/bin/env bash
# Run the new per-state timeout test, then a mutation that drops the reset. Usage: run-newer-state-test.sh
set -u
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
M=$W/crates/solid-pod-rs-forge/src/mirror.rs
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
O=$EV/fix-a7/lead-newer-state-test.txt
export CARGO_HOME=/home/devuser/workspace/.cargo CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target CARGO_BUILD_JOBS=4
cd "$W" || exit 2
T=a_newer_state_is_retried_after_an_older_one_gave_up
{
  echo "# green run"
  cargo test --locked --offline -p solid-pod-rs-forge --features mirror "$T" 2>&1 | grep -E '^test |test result|panicked|error'
  echo "exit=${PIPESTATUS[0]}"
  cp "$M" "$M.bak"
  # Mutation: never reset the count for a newer state.
  sed -i 's/^\(\s*\)rec\.timeouts = 0;$/\1\/\/ mutated/' "$M"
  # Restore the applied-state reset (only the newer-state one stays mutated).
  echo "# mutated lines:"; grep -n '// mutated' "$M"
  echo "# mutation run (expect FAIL)"
  cargo test --locked --offline -p solid-pod-rs-forge --features mirror "$T" 2>&1 | grep -E '^test |test result|panicked|left|right'
  echo "exit=${PIPESTATUS[0]}"
  mv "$M.bak" "$M"
  grep -c '// mutated' "$M"
} > "$O" 2>&1
echo done
