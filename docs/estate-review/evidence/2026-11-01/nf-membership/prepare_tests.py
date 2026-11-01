"""Insert audit cases inside existing modules in a disposable pinned archive."""
from pathlib import Path
import argparse
import hashlib
import json

p = argparse.ArgumentParser()
p.add_argument('--kit', type=Path, required=True)
a = p.parse_args()
e = Path(__file__).resolve().parent
assert not (a.kit / '.git').exists(), 'Use a disposable Git archive'
rows = []
for name, addition in [
    ('crates/nostr-bbs-core/src/whitelist_sql.rs', 'core-audit-tests.rs'),
    ('crates/nostr-bbs-forum-client/src/admin/membership.rs', 'client-audit-tests.rs'),
]:
    original = (e / 'source' / name).read_bytes()
    target = a.kit / name
    assert target.read_bytes() == original, name
    source = original.decode()
    assert source.rstrip().endswith('}')
    offset = source.rfind('}')
    tests = (e / addition).read_text()
    target.write_text(source[:offset] + tests + '\n' + source[offset:])
    rows.append({'path': name, 'original_sha256': hashlib.sha256(original).hexdigest(),
                 'addition_sha256': hashlib.sha256(tests.encode()).hexdigest(),
                 'insertion': 'before final brace of existing test module'})
print(json.dumps(rows, indent=2))
