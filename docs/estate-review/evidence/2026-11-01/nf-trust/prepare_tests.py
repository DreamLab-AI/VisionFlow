"""Append audit cases to the pinned archive; never mutate a source checkout."""
from pathlib import Path
import argparse
import hashlib
import json

p = argparse.ArgumentParser()
p.add_argument('--kit', type=Path, required=True)
a = p.parse_args()
e = Path(__file__).resolve().parent
assert not (a.kit / '.git').exists(), 'Use a disposable Git archive'
name = 'crates/nostr-bbs-relay-worker/src/trust_sweep.rs'
target = a.kit / name
original = (e / 'source' / name).read_bytes()
assert target.read_bytes() == original, 'Source differs or tests already appended'
source = original.decode()
assert source.rstrip().endswith('}')
offset = source.rfind('}')
addition = (e / 'audit-tests.rs').read_text()
target.write_text(source[:offset] + addition + '\n' + source[offset:])
print(json.dumps({'original_sha256': hashlib.sha256(original).hexdigest(),
                  'audit_sha256': hashlib.sha256(addition.encode()).hexdigest(),
                  'insertion': 'inside existing trust_sweep::tests module before final brace'}))
