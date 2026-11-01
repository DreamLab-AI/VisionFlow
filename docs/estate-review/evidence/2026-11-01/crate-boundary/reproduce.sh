#!/usr/bin/env bash
set -euo pipefail
repo=${1:-/home/devuser/workspace/sidestr-rs}
audit_dir=$(cd -- "$(dirname -- "$0")" && pwd)
scratch=${2:-/home/devuser/workspace/.tmp/vf-boundary-reproduction}
mkdir -p "$scratch"
export CARGO_TARGET_DIR="$scratch/target"
export CARGO_BUILD_JOBS=3 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cd "$repo"
cargo test --locked --offline -p sidestr-core --features consensus-oracle --test channel_oracle -- --nocapture
cargo test --locked --offline -p sidestr-agent --no-default-features --doc hitch::store -- --nocapture
cargo test --locked --offline -p sidestr-agent --no-default-features --lib hitch::follow::tests -- --nocapture
cargo test --locked --offline -p sidestr-agent --test hitch_binding_kat -- --nocapture
cargo test --locked --offline -p sidestr-agent --lib hitch::host::tests::nothing_is_sent_before_its_locks -- --exact --nocapture
node "$audit_dir/run-probe.cjs" "$repo" "$CARGO_TARGET_DIR" "$audit_dir/follower-probe.rs" "$scratch/follower-probe.json"
