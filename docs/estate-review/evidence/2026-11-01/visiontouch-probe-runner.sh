#!/usr/bin/env bash
# Software-only tests. The archive is private scratch; no device is opened.
set -euo pipefail
repo=${1:-/home/devuser/workspace/visiontouch-rs}
assets=$(cd -- "$(dirname -- "$0")" && pwd)
scratch=$(mktemp -d /tmp/visiontouch-audit.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
git -C "$repo" archive 7b95fb20d50a9d06c64fb6d6cc393b16edbbc35d | tar -x -C "$scratch"
cat "$assets/visiontouch-device-probes.rs" >> "$scratch/crates/device/src/lib.rs"
cat "$assets/visiontouch-effects-probes.rs" >> "$scratch/crates/effects/src/lib.rs"
cat "$assets/visiontouch-worker-probes.rs" >> "$scratch/apps/calibration/src/worker.rs"
VT_AUDIT_FOREIGN_CWD=$(git -C "$assets" rev-parse --show-toplevel) CARGO_TARGET_DIR="$repo/target" cargo test --locked --offline --manifest-path "$scratch/Cargo.toml" -p visiontouch-device -p visiontouch-effects -p visiontouch-studio estate_audit -- --nocapture
