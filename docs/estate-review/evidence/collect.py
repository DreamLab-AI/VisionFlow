#!/usr/bin/env python3
"""Collect local estate identities and isolated VisionFlow evaluator receipts.

Run from any directory. Writes snapshot.json beside this file; no service calls,
source edits, build, deployment, or changes to sibling repositories.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
CANON = HERE.parents[2]
WORKSPACE = CANON.parent
REPOSITORIES = {
    'VisionFlow': 'VisionFlow', 'VisionClaw': 'project',
    'agentbox': 'project/agentbox', 'solid-pod-rs': 'solid-pod-rs',
    'nostr-rust-forum': 'nostr-rust-forum',
    'dreamlab-ai-website': 'dreamlab-ai-website', 'loom': 'loom',
    'knowledgeGraph': 'knowledgeGraph', 'WasmVOWL': 'WasmVOWL',
    'dream-engine': 'dream-machine', 'RuView': 'RuView', 'ruvector': 'ruvector',
}
SOURCES = [
    'VisionFlow/README.md', 'VisionFlow/docs/README.md',
    'VisionFlow/docs/BASELINE-visionflow.md',
    'VisionFlow/docs/architecture/repository-map.md',
    'VisionFlow/docs/ecosystem-map.md',
    'VisionFlow/docs/registers/gap-register-v1.2.md',
    'VisionFlow/presentation/report/chapters/13-visionflow-canon.tex',
    'VisionFlow/presentation/report/chapters/15-open-questions.tex',
    'VisionFlow/website/build.sh', 'VisionFlow/package.json',
    'VisionFlow/.github/workflows/deploy.yml',
    'VisionFlow/.github/workflows/fixture-drift.yml',
    'VisionFlow/.github/workflows/drift-counter.yml',
    'VisionFlow/scripts/drift-counter/drift-counter.mjs',
    'VisionFlow/scripts/drift-counter/allowlist.json',
    'VisionFlow/scripts/generate-release-manifest.sh',
    'VisionFlow/dream.config.json', 'VisionFlow/docs/dream-cycle/LEDGER.md',
    'VisionFlow/scripts/dream-link-check.sh',
    'VisionFlow/scripts/dream-build-check.sh',
    'VisionFlow/scripts/dream-meta-tags-scan.sh',
    'VisionFlow/scripts/dream-structured-data-scan.sh',
    'project/agentbox/scripts/skill-count-check.js',
    'project/agentbox/mcp/servers/ontology-bridge.js',
    'loom/README.md', 'loom/Cargo.toml', 'knowledgeGraph/README.md',
    'WasmVOWL/README.md', 'dream-machine/README.md', 'ruvector/Cargo.toml',
    'RuView/README.md',
]

def run(argv, cwd):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                            timeout=60)
    return dict(command=argv, exit_code=result.returncode,
                stdout=result.stdout, stderr=result.stderr)

def git(path, *args):
    r = run(['git', *args], path)
    return r['stdout'].strip() if r['exit_code'] == 0 else None

snapshot = {
    'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Local checkout identity, source hashes, canon drift check and isolated evaluator probes; no federation or production verification.',
    'repositories': [], 'source_sha256': {}, 'evaluator_probes': [],
}
for name, rel in REPOSITORIES.items():
    p = WORKSPACE / rel
    status = git(p, 'status', '--porcelain', '--untracked-files=no')
    snapshot['repositories'].append(dict(
        name=name, workspace_path=rel, resolved_path=str(p.resolve()),
        head=git(p, 'rev-parse', 'HEAD'),
        branch=git(p, 'branch', '--show-current'),
        tracked_change_count=len(status.splitlines()) if status else 0,
        tracked_status_available=status is not None,
        untracked_files_counted=False,
    ))
for rel in SOURCES:
    p = WORKSPACE / rel
    snapshot['source_sha256'][rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
snapshot['drift_counter'] = run(
    ['node', 'scripts/drift-counter/drift-counter.mjs', '--json'], CANON)
with tempfile.TemporaryDirectory(prefix='visionflow-estate-probe-') as tmp:
    root = Path(tmp)
    (root / 'scripts').mkdir()
    (root / 'website/dist').mkdir(parents=True)
    (root / 'website/dist/index.html').write_text(
        '<html><img src="missing.png"><script type="application/ld+json">{broken}</script></html>')
    for name in ['dream-link-check.sh', 'dream-meta-tags-scan.sh',
                 'dream-structured-data-scan.sh', 'dream-build-check.sh']:
        shutil.copy2(CANON / 'scripts' / name, root / 'scripts' / name)
        if name == 'dream-build-check.sh':
            (root / 'website/dist/index.html').unlink()
        receipt = run(['bash', 'scripts/' + name], root)
        receipt['fixture'] = ('missing index.html' if name == 'dream-build-check.sh'
                              else 'HTML with missing.png, no required metadata and malformed JSON-LD')
        snapshot['evaluator_probes'].append(receipt)
(HERE / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
print('Wrote', HERE / 'snapshot.json')
print('Drift exit:', snapshot['drift_counter']['exit_code'])
for receipt in snapshot['evaluator_probes']:
    print(receipt['command'][1], 'exit:', receipt['exit_code'])
