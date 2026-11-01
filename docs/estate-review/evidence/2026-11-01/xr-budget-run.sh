#!/usr/bin/env bash
# Prepare/run isolated Rust tests. Optional third argument is a Godot binary
# on this machine; never points at the live XR project.
set -euo pipefail
source_repo=${1:?VisionClaw source checkout}
audit_dir=${2:?new disposable directory}
engine=${3:-}
evidence=$(cd "$(dirname "$0")" && pwd)
mkdir "$audit_dir"
audit_dir=$(cd "$audit_dir" && pwd)
mkdir -p "$audit_dir/xr-client/rust/src" "$audit_dir/godot/tests/unit"
for module in frame_budget hulls lod memory_cloud memory_route domain_palette; do
 cp "$source_repo/xr-client/rust/src/$module.rs" "$audit_dir/xr-client/rust/src/"
 printf 'pub mod %s;\n' "$module" >> "$audit_dir/xr-client/rust/src/lib.rs"
done
cp "$evidence/xr-budget-Cargo.toml" "$audit_dir/xr-client/rust/Cargo.toml"
cp "$evidence/xr-budget-Cargo.lock" "$audit_dir/xr-client/rust/Cargo.lock"
# Original parity tests read these source paths relative to the Rust crate.
mkdir -p "$audit_dir/client/src/features/visualisation"
cp -a "$source_repo/client/src/features/visualisation/memoryCloud" "$audit_dir/client/src/features/visualisation/"
cp -a "$source_repo/client/src/features/visualisation/memoryTrajectory" "$audit_dir/client/src/features/visualisation/"
mkdir -p "$audit_dir/client/src/features/visualisation/components"
cp "$source_repo/client/src/features/visualisation/components/EmbeddingCloudLayer.tsx" "$audit_dir/client/src/features/visualisation/components/"
cp -a "$source_repo/xr-client/materials" "$audit_dir/xr-client/materials"
mkdir -p "$audit_dir/xr-client/scripts"
for script in graph_scene radial_menu hud; do cp "$source_repo/xr-client/scripts/$script.gd" "$audit_dir/xr-client/scripts/"; done
export CARGO_TARGET_DIR="$audit_dir/target"
cargo test --offline --locked --manifest-path "$audit_dir/xr-client/rust/Cargo.toml" --lib > "$audit_dir/rust-suite.txt" 2>&1
cp "$evidence/xr-budget-audit.rs" "$audit_dir/xr-client/rust/src/audit.rs"
printf '\n#[cfg(test)]\nmod audit;\n' >> "$audit_dir/xr-client/rust/src/lib.rs"
XR_BUDGET_AUDIT_OUT="$audit_dir/rust-audit.json" cargo test --offline --locked --manifest-path "$audit_dir/xr-client/rust/Cargo.toml" --lib audit:: > "$audit_dir/rust-audit.txt" 2>&1
mkdir -p "$audit_dir/godot/scripts" "$audit_dir/godot/scenes" "$audit_dir/godot/addons"
cp -a "$source_repo/xr-client/addons/gut" "$audit_dir/godot/addons/"
cp -a "$source_repo/xr-client/materials" "$audit_dir/godot/materials"
for script in eye_candy emulated_volume spatial_volume frame_budget_pass node_lod; do
 cp "$source_repo/xr-client/scripts/$script.gd" "$audit_dir/godot/scripts/"
done
cp "$source_repo/xr-client/scenes/EmulatedVolume.tscn" "$audit_dir/godot/scenes/"
cp "$source_repo/xr-client/tests/gut_guard.sh" "$audit_dir/godot/tests/"
cp "$evidence/xr-budget-isolated-tests.gd" "$audit_dir/godot/tests/unit/test_eye_candy.gd"
cp "$evidence/xr-budget-scene-tick.gd" "$audit_dir/godot/scene_tick.gd"
cp "$evidence/xr-budget-probe.gd" "$audit_dir/godot/probe.gd"
cat > "$audit_dir/godot/project.godot" <<'PROJECT'
config_version=5
[application]
config/name="VisionFlow XR budget audit"
[rendering]
renderer/rendering_method="gl_compatibility"
PROJECT
if [ -n "$engine" ]; then
 cd "$audit_dir/godot"
 export XDG_DATA_HOME="$audit_dir/data"
 timeout 60 "$engine" --headless --editor --import --quit > import.txt 2>&1
 timeout 60 "$engine" --headless --path . --script addons/gut/gut_cmdln.gd -gdir=res://tests/unit -gexit -glog=1 > gut.txt 2>&1
 sed -E 's/\x1B\[[0-9;]*[mK]//g' gut.txt > gut-plain.txt
 bash tests/gut_guard.sh gut-plain.txt import.txt
 timeout 30 "$engine" --headless --path . --script probe.gd > probe.txt 2>&1
fi
# Review and retain results before removing audit_dir.
