#!/usr/bin/env python3
"""Revalidate explicitly mapped prior source receipts, without replaying old tests."""
import collections
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
workspace = base.parents[3]
records = []
for file in sorted(base.glob('*.json')):
    if not (file.name.startswith('ruview-') or file.name == 'explorer-snapshot.json'):
        continue
    data = json.loads(file.read_text())
    pairs = []
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                pairs.append((value['path'], value['sha256']))
            for key, child in value.items():
                if key == 'sources' and isinstance(child, dict):
                    for path, sha in child.items():
                        if isinstance(sha, str) and len(sha) == 64:
                            pairs.append((path, sha))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(data)
    for source, expected in pairs:
        path = Path(source)
        rule = 'absolute receipt path'
        if not path.is_absolute():
            if file.name == 'explorer-snapshot.json':
                root = workspace / 'WasmVOWL'
            elif file.name == 'ruview-mobile-snapshot.json':
                root = workspace / 'RuView/ui/mobile'
            elif file.name in {'ruview-workspace-snapshot.json', 'ruview-migration-snapshot.json'}:
                root = workspace / 'RuView/rust-port/wifi-densepose-rs'
            else:
                root = workspace / 'RuView'
            path = root / path
            rule = str(root.relative_to(workspace))
        if file.name == 'ruview-workspace-snapshot.json' and path.is_dir():
            path = path / 'Cargo.toml'
            rule += '; workspace member hash is Cargo.toml'
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        records.append({'receipt': file.name, 'path': str(path), 'resolution': rule, 'recorded': expected, 'actual': actual, 'state': 'unchanged' if actual == expected else 'missing' if actual is None else 'changed'})
result = {'date': '2026-09-07', 'method': 'Explicit receipt-relative roots; unchanged source hashes preserve earlier source evidence scope, not runtime test freshness. ADR annex changes are separate from implementation changes.', 'counts': dict(collections.Counter(r['state'] for r in records)), 'records': records}
Path(__file__).with_name('prior-evidence-revalidation.json').write_text(json.dumps(result, indent=2) + '\n')
print(result['counts'])
