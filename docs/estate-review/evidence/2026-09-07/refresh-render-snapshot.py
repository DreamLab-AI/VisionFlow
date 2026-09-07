#!/usr/bin/env python3
"""Render only changed Mermaid blocks identified by the render-source checker."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
diagrams = root / 'docs/diagrams'
report = Path(sys.argv[1])
snapshot = json.loads(report.read_text())
work = []
for row in snapshot['records']:
    if row['ok']:
        continue
    topic = diagrams / row['topic']
    heading = None
    language = None
    buffer = []
    for line in topic.read_text().splitlines():
        if language is None:
            if line.startswith('## '):
                heading = line.split()[1]
            if line.startswith('```'):
                language = line[3:].strip(); buffer = []
        elif line.startswith('```'):
            if language == 'mermaid' and heading == row['id']:
                source = '\n'.join(buffer) + '\n'
                if hashlib.sha256(source.encode()).hexdigest() != row['source_sha256']:
                    raise SystemExit('Source changed since snapshot: ' + row['id'])
                dest = diagrams / 'rendered' / Path(row['topic']).with_suffix('') / (heading + '.mmd')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(source)
                work.append((row, dest))
            language = None
        else:
            buffer.append(line)
def render(item):
    row, dest = item
    command = ['mmdc','-i',str(dest),'-o',str(dest.with_suffix('.svg')),'-q']
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return {'id':row['id'],'topic':row['topic'],'source_sha256':row['source_sha256'],'command':command,'exit_code':result.returncode,'output':result.stdout+result.stderr}
with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(render,work))
report.with_name(report.stem + '-refresh.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'rendered':len(results),'failed':sum(r['exit_code']!=0 for r in results)}))
raise SystemExit(1 if any(r['exit_code'] for r in results) else 0)
