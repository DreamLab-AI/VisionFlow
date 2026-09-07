#!/usr/bin/env python3
"""Check that Mermaid render inputs match the current topic bytes; no semantic claim."""
import argparse
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('--report', type=Path)
args = parser.parse_args()
records = []
for topic in sorted(args.root.glob('*/*.md')):
    text = topic.read_text()
    if not text.startswith('---\n'):
        continue
    heading = None
    language = None
    buffer = []
    for line in text.splitlines():
        if language is None:
            match = re.match(r'^## ([A-Z]{2,3}-\d{2,3}\.\d+) ', line)
            if match:
                heading = match[1]
            if line.startswith('```'):
                language = line[3:].strip()
                buffer = []
        elif line.startswith('```'):
            if language == 'mermaid':
                if not heading:
                    raise SystemExit(f'Missing diagram heading in {topic}')
                expected = '\n'.join(buffer) + '\n'
                mmd = args.root / 'rendered' / topic.relative_to(args.root).with_suffix('') / (heading + '.mmd')
                svg = mmd.with_suffix('.svg')
                actual = mmd.read_text() if mmd.exists() else None
                svg_text = svg.read_text() if svg.exists() else ''
                viewbox = re.search(r'viewBox="[\d.\-]+ [\d.\-]+ ([\d.]+) ([\d.]+)"', svg_text)
                width = float(viewbox[1]) if viewbox else None
                records.append({'topic': str(topic.relative_to(args.root)), 'id': heading, 'source_sha256': hashlib.sha256(expected.encode()).hexdigest(), 'render_input_matches': actual == expected, 'svg_sha256': hashlib.sha256(svg_text.encode()).hexdigest() if svg_text else None, 'width': width, 'ok': actual == expected and width is not None and width <= 4500})
            language = None
        else:
            buffer.append(line)
failures = [r for r in records if not r['ok']]
result = {'method': 'Exact current Mermaid input bytes compared to mmdc input; SVG presence/width/hash recorded. Successful render execution is separately evidenced by command logs. This does not establish semantic correctness.', 'diagrams': len(records), 'passing': len(records)-len(failures), 'failing': len(failures), 'records': records}
if args.report:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'diagrams': len(records), 'passing': result['passing'], 'failing': len(failures), 'topics_needing_render': sorted(set(r['topic'] for r in failures))}))
raise SystemExit(1 if failures or not records else 0)
