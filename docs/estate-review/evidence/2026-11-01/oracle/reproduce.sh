#!/usr/bin/env bash
set -euo pipefail
# Read-only source checkout; all build and empty-reference data stays outside it.
repo=${1:-/home/devuser/workspace/sidestr-rs}
scratch=${2:-/home/devuser/workspace/.tmp/vf-oracle-reproduction}
mkdir -p "$scratch/empty-reference"
export CARGO_TARGET_DIR="$scratch/target"
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=3
export SIDESTR_SIDING="$scratch/empty-reference"
export SCHEMA="$scratch/empty-reference" BLAKETESTNODE="$scratch/empty-reference"
cd "$repo"
cargo test --locked --offline -p sidestr-round --all-features --test interop_round --test interop_pegout -- --nocapture
cargo test --locked --offline -p sidestr-core --test pegtweak pegtweak_mjs_agrees -- --exact --nocapture
