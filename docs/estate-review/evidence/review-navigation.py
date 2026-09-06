#!/usr/bin/env python3
"""Check inline local links within this review; not a rendered-site validator."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(ROOT.rglob('*.md'))
FILE_SET = set(FILES)

def anchors(path):
    counts = Counter()
    result = set()
    fence = None
    for line in path.read_text().splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            char = marker[1][0]
            if fence is None:
                fence = char
            elif fence == char:
                fence = None
            continue
        if fence:
            continue
        heading = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if heading:
            slug = re.sub(r'[^\w\- ]', '', heading[1].lower()).replace(' ', '-')
            suffix = counts[slug]
            counts[slug] += 1
            result.add(slug + (f'-{suffix}' if suffix else ''))
        result.update(re.findall(r'\bid=["\']([^"\']+)["\']', line))
    return result

anchor_sets = {p: anchors(p) for p in FILES}
incoming = Counter()
missing_files = []
missing_anchors = []
checked_links = 0
checked_anchors = 0
for path in FILES:
    for link in re.findall(r'\]\(<?([^\n)>]+)>?\)', path.read_text()):
        if ':' in link or link.startswith('/'):
            continue
        filename, _, fragment = link.partition('#')
        target = (path.parent / filename).resolve() if filename else path
        checked_links += 1
        item = {'document': str(path.relative_to(ROOT)), 'target': link}
        if not target.exists():
            missing_files.append(item)
        if target in FILE_SET:
            if path != target:
                incoming[target] += 1
            if fragment:
                checked_anchors += 1
                if fragment not in anchor_sets[target]:
                    missing_anchors.append(item)
orphans = [str(p.relative_to(ROOT)) for p in FILES if p != ROOT / 'README.md' and not incoming[p]]
report = {
    'scope': 'Inline Markdown local file targets, ATX-heading/HTML-id anchors inside the review, and incoming review links. Excludes external URLs, absolute links, reference-style links, renderer behaviour and semantic completeness.',
    'markdown_files': len(FILES), 'checked_local_links': checked_links,
    'checked_internal_anchors': checked_anchors,
    'missing_files': missing_files, 'missing_internal_anchors': missing_anchors,
    'pages_without_incoming_review_links': orphans,
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES},
}
(ROOT / 'evidence/review-navigation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'scope')}))
raise SystemExit(bool(missing_files or missing_anchors or orphans))
