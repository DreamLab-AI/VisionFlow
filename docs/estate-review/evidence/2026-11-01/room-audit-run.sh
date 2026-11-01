#!/usr/bin/env bash
set -euo pipefail
project=${1:?usage: room-audit-run.sh PROJECT OUTPUT_DIR}
output=${2:?usage: room-audit-run.sh PROJECT OUTPUT_DIR}
fixture_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output"
output=$(cd "$output" && pwd)
scratch=$(mktemp -d -t vf-room-reproduce-XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
cp -a "$project/crates/visionclaw-xr-presence" "$scratch/presence"
rm -rf "$scratch/presence/target"
mkdir -p "$scratch/actor/src"
cp "$project/src/actors/presence_actor.rs" "$scratch/actor/src/"
cp "$project/xr-client/rust/src/webrtc_audio.rs" "$scratch/actor/src/"
cp "$fixture_dir/room-audit.rs" "$scratch/actor/src/audit.rs"
cp "$fixture_dir/room-actor-Cargo.toml" "$scratch/actor/Cargo.toml"
cp "$fixture_dir/room-actor-Cargo.lock" "$scratch/actor/Cargo.lock"
cp "$fixture_dir/room-presence-Cargo.lock" "$scratch/presence/Cargo.lock"
printf '%s\n' '#![allow(dead_code)]' '#[cfg(test)] mod presence_actor;' '#[cfg(test)] mod webrtc_audio;' '#[cfg(test)] mod audit;' > "$scratch/actor/src/lib.rs"
python3 - "$project" "$scratch" <<'PYVERIFY'
from pathlib import Path
import sys
project, scratch = map(Path, sys.argv[1:])
for source, copied in [('src/actors/presence_actor.rs', 'actor/src/presence_actor.rs'), ('xr-client/rust/src/webrtc_audio.rs', 'actor/src/webrtc_audio.rs')]:
    assert (project/source).read_bytes() == (scratch/copied).read_bytes()
PYVERIFY
# An optional dedicated build cache avoids the production workspace's target lock.
audit_target=${ROOM_AUDIT_TARGET:-$scratch/target}
cargo test --offline --locked --manifest-path "$scratch/presence/Cargo.toml" --target-dir "$audit_target" > "$output/room-presence-suite.txt" 2>&1
printf '%s\n' 'presence_suite_exit=0' > "$output/room-test-exits.txt"
cargo test --offline --locked --manifest-path "$scratch/actor/Cargo.toml" --target-dir "$audit_target" -- --nocapture > "$output/room-actor-voice-suite.txt" 2>&1
printf '%s\n' 'actor_voice_suite_exit=0' >> "$output/room-test-exits.txt"
