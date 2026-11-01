#!/usr/bin/env bash
# Reproduce in a fresh directory; no production services or weights are used.
set -euo pipefail
source_repo=${1:?agentbox checkout at the recorded revision}
media_work=${2:?new disposable scratch directory}
evidence=$(cd "$(dirname "$0")" && pwd)
python3 - "$source_repo" "$evidence/media-source-receipt.json" <<'PY_CHECK'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for item in json.loads(pathlib.Path(sys.argv[2]).read_text())["files"]:
    assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
PY_CHECK
mkdir "$media_work"
media_work=$(cd "$media_work" && pwd)
mkdir -p "$media_work/skills/video-production" "$media_work/comfyui-sidecar/nodes" "$media_work/upstream" "$media_work/results"
cp -a "$source_repo/skills/video-production/runtime" "$media_work/skills/video-production/runtime"
cp -a "$source_repo/comfyui-sidecar/nodes/agentbox_video" "$media_work/comfyui-sidecar/nodes/agentbox_video"
cp -a "$source_repo/skills/comfyui" "$media_work/comfyui"
# These are isolated test dependencies, not a replacement sidecar lockfile.
uv venv "$media_work/venv"
uv pip install --python "$media_work/venv/bin/python" numpy==2.3.5 opencv-python-headless==5.0.0.93 safetensors==0.6.2 Pillow==12.0.0
media_cpp_lib=$(dirname "$(c++ -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="$media_cpp_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CARGO_TARGET_DIR="$media_work/target"
cargo test --offline --locked --manifest-path "$media_work/skills/video-production/runtime/Cargo.toml" > "$media_work/results/runtime.txt" 2>&1
"$media_work/venv/bin/python" -I -m unittest discover -s "$media_work/comfyui-sidecar/nodes/agentbox_video/tests" -t "$media_work/comfyui-sidecar/nodes" > "$media_work/results/nodes.txt" 2>&1
python3 -I -m unittest discover -s "$media_work/comfyui/scripts/tests" -p test_comfy_client.py > "$media_work/results/client.txt" 2>&1
npm ci --prefix "$media_work/comfyui/mcp-server" --ignore-scripts --no-audit --no-fund
node --test "$media_work/comfyui/mcp-server/server.test.js" > "$media_work/results/mcp.txt" 2>&1
curl -fsSL https://raw.githubusercontent.com/Comfy-Org/ComfyUI/faa7e494f3a0c9e45e1c3db8d20ee48fd5fe1b52/server.py -o "$media_work/upstream/server.py"
python3 - "$media_work/upstream/server.py" <<'PY'
import hashlib, pathlib, sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == 'b2ff22d5464bdc7fde71f2e7ed349363458c2ce1aad92ecced446e25a99d5440'
PY
python3 "$evidence/media-cancel-probe.py" "$media_work/comfyui/scripts/comfy_client.py" "$media_work/upstream/server.py" "$media_work/results/cancel.json"
mkdir -p "$media_work/skills/video-production/runtime/tests"
cp "$evidence/media-bounds-audit.rs" "$media_work/skills/video-production/runtime/tests/media_bounds_audit.rs"
MEDIA_AUDIT_ROOT="$media_work/bounds" cargo test --offline --locked --manifest-path "$media_work/skills/video-production/runtime/Cargo.toml" --test media_bounds_audit -- --nocapture > "$media_work/results/bounds.txt" 2>&1
"$media_work/venv/bin/python" -I "$evidence/media-node-probe.py" "$media_work/comfyui-sidecar/nodes" "$source_repo/comfyui-sidecar/verify-spede.sh" "$media_work/results/node-probe.json"
# Results remain in the directory supplied by the caller; remove it after review.
