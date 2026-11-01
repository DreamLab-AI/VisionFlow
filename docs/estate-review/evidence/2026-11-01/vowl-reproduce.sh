#!/usr/bin/env bash
# Requires locked dependencies in Cargo's cache and wasm-bindgen CLI 0.2.128.
# Scratch must be on an executable filesystem. Nothing in the source repo is written.
set -euo pipefail
repo=${1:?source vowl-wasm repository}
scratch=${2:?new scratch directory on executable filesystem}
bindgen=${3:?path to wasm-bindgen 0.2.128}
consumer=${4:?visionGraph publishing-tools/WasmVOWL/modern directory}
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test ! -e "$scratch"
test "$("$bindgen" --version)" = 'wasm-bindgen 0.2.128'
mkdir -p "$scratch/source"
git -C "$repo" archive 28af4365a68d9aacfd7c04f63299cc96ad717ce3 | tar -x -C "$scratch/source"
export CARGO_TARGET_DIR="$scratch/target"
cargo build --manifest-path "$scratch/source/Cargo.toml" --target wasm32-unknown-unknown --release --features ngg1,interaction --locked --offline
"$bindgen" "$scratch/target/wasm32-unknown-unknown/release/vowl_wasm.wasm" --target web --out-dir "$scratch/source/pkg"
# Node-only metadata; this is not a release package.
cat > "$scratch/source/pkg/package.json" <<'JSON'
{"name":"@dreamlab-ai/vowl-wasm","version":"0.2.2","type":"module"}
JSON
VOWL_BUNDLE_GATE=1 cargo test --manifest-path "$scratch/source/Cargo.toml" --locked --offline --features ngg1,interaction
node --test "$scratch/source/tests/ngg1_wasm.test.mjs"
export VOWL_AUDIT_ROOT="$scratch/source"
export VOWL_CONSUMER_ROOT="$consumer"
node --test "$probe_dir/vowl-boundary-probes.mjs"
node --test "$probe_dir/vowl-worker-probes.mjs"
