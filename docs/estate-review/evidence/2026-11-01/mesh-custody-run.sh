#!/usr/bin/env bash
set -euo pipefail
# Usage: mesh-custody-run.sh AGENTBOX FORUM SCRATCH
# SCRATCH is retained for inspection; remove it after saving the evidence.
evidence=$(cd -- "$(dirname -- "$0")" && pwd)
agentbox=$(realpath "$1")
forum=$(realpath "$2")
scratch=$(realpath "$3")
mkdir -p "$scratch/src/identity_port"
python3 - "$evidence" "$agentbox" "$forum" "$scratch" <<'PY'
from pathlib import Path
import sys,shutil,json,hashlib
E,A,F,S=map(Path,sys.argv[1:])
receipt=json.loads((E/'mesh-source-receipt.json').read_text())
for record in receipt['repos']:
 repo=A if record['repo'].endswith('/agentbox') else F if record['repo'].endswith('/nostr-rust-forum') else None
 if repo is not None:
  for source in record['sources']:
   assert hashlib.sha256((repo/source['path']).read_bytes()).hexdigest()==source['sha256'], source['path']
c=(E/'mesh-custody-Cargo.toml').read_text().replace('/home/devuser/workspace/nostr-rust-forum',str(F))
(S/'Cargo.toml').write_text(c)
if (E/'mesh-custody-Cargo.lock').exists(): shutil.copy2(E/'mesh-custody-Cargo.lock',S/'Cargo.lock')
shutil.copy2(E/'mesh-custody-audit.rs',S/'src/lib.rs')
base=A/'services/nostr-pod-bridge/src'
for f in ['envmap.rs','identity.rs','mirror_key.rs','pyjson.rs','role_secret.rs','identity_port/acl.rs','identity_port/port.rs','identity_port/receipts.rs','identity_port/server.rs']:
 shutil.copy2(base/f,S/'src'/f)
(S/'src/backup.rs').write_text((A/'services/secret-backup/src/main.rs').read_text()+'\n'+(E/'mesh-backup-audit.rs').read_text())
PY
export AUDIT_AGENTBOX_MANIFEST="$agentbox/agentbox.toml"
export CARGO_TARGET_DIR="$scratch/target"
if [[ ! -f "$scratch/Cargo.lock" ]]; then cargo generate-lockfile --offline --manifest-path "$scratch/Cargo.toml"; fi
cargo test --locked --offline --manifest-path "$scratch/Cargo.toml" --lib -- --nocapture
