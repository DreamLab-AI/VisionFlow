#!/usr/bin/env bash
set -euo pipefail
forum=${1:?usage: forum-spaces-run.sh FORUM OUTPUT_DIR}
output=${2:?usage: forum-spaces-run.sh FORUM OUTPUT_DIR}
fixture=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output"
output=$(cd "$output" && pwd)
scratch=$(mktemp -d -t vf-forum-spaces-reproduce-XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$forum" "$fixture" "$scratch" <<'PY'
from pathlib import Path
import sys,re,json,shutil
forum,fixture,scratch=map(Path,sys.argv[1:])
(scratch/'src').mkdir()
for name in ['nip42.rs','filter.rs']:
    source=forum/'crates/nostr-bbs-relay-worker/src/relay_do'/name
    shutil.copyfile(source,scratch/'src'/name)
    assert source.read_bytes()==(scratch/'src'/name).read_bytes()
source=(forum/'crates/nostr-bbs-relay-worker/src/relay_do/storage.rs').read_text()
lib=(forum/'crates/nostr-bbs-relay-worker/src/lib.rs').read_text()
def literal(pattern,text):
    m=re.search(pattern,text,re.S)
    assert m, pattern
    return re.sub(r'\\\n\s*','',m.group(1))
values={
 'SCHEMA':literal(r'pub const EVENTS_CREATE_SQL: &str = "(.*?)";',lib),
 'INSERT':literal(r'let insert_stmt = db.prepare\(\s*"(.*?)",',source),
 'DELETE_REPLACEABLE':literal(r'EventTreatment::Replaceable =>.*?db.prepare\(\s*"(.*?)",',source),
 'DELETE_ADDRESSABLE':literal(r'EventTreatment::ParameterizedReplaceable =>.*?db.prepare\(\s*"(.*?)",',source),
 'READ_UNTAGGED':literal(r'None => format!\(\s*"(.*?)"',source)
}
assert values==json.loads((fixture/'forum-storage-sql.json').read_text())['sql'], 'SQL source has drifted; review before repeating'
(scratch/'src/sql.rs').write_text('\n'.join('pub const '+k+': &str = r#"'+v+'"#;' for k,v in values.items())+'\n')
(scratch/'src/lib.rs').write_text('#![allow(dead_code)]\nstruct NostrRelayDO;\nmod filter;\nmod nip42;\nmod sql;\n#[cfg(test)] mod audit;\n')
shutil.copyfile(fixture/'forum-spaces-audit.rs',scratch/'src/audit.rs')
shutil.copyfile(fixture/'forum-spaces-Cargo.lock',scratch/'Cargo.lock')
manifest=(fixture/'forum-spaces-Cargo.toml').read_text().replace('"/home/devuser/workspace/nostr-rust-forum/crates/nostr-bbs-core"',json.dumps(str((forum/'crates/nostr-bbs-core').resolve())))
(scratch/'Cargo.toml').write_text(manifest)
PY
audit_target=${FORUM_AUDIT_TARGET:-$scratch/target}
cargo test --locked --offline --manifest-path "$scratch/Cargo.toml" --target-dir "$audit_target" -- --nocapture --test-threads=1 > "$output/forum-spaces-suite.txt" 2>&1
printf '%s\n' 'forum_spaces_suite_exit=0' > "$output/forum-test-exits.txt"
