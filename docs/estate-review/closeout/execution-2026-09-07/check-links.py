#!/usr/bin/env python3
"""Local Markdown target existence; not anchors, HTTP links or semantic validation."""
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

vf = Path(__file__).resolve().parents[4]
roots = [vf / 'docs' / p for p in ['estate-review', 'architecture', 'adr', 'diagrams']]
files = sorted({p for root in roots for p in root.rglob('*.md') if 'rendered' not in p.parts})
missing = []
checked = 0
excluded = 0
for file in files:
    source = file.read_text()
    source = re.sub(r'(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$', '', source)
    source = re.sub(r'`[^`\n]*`', '', source)
    for match in re.finditer(r'!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\n]+)\)', source):
        target = match[1].strip()
        if target.startswith('<'):
            target = target[1:-1]
        else:
            target = re.sub(r'\s+[\"\'][^\"\']*[\"\']$', '', target)
        if urlsplit(target).scheme or target.startswith(('#', '//')):
            continue
        target = unquote(target.split('#', 1)[0].split('?', 1)[0])
        if not target:
            continue
        if 'RuView' in Path(target).parts:
            excluded += 1
            continue
        checked += 1
        resolved = (file.parent / target).resolve()
        if not resolved.exists():
            missing.append({'source': str(file.relative_to(vf)), 'target': target})
report = {'method': __doc__, 'files': len(files), 'checked': checked, 'excluded_RuView_targets': excluded, 'missing': missing}
out = vf / 'docs/estate-review/evidence/execution-2026-09-07/link-check.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({**report, 'missing': len(missing)}))
raise SystemExit(bool(missing))
