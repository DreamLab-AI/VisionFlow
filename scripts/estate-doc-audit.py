#!/usr/bin/env python3
"""Read-only workspace census and ADR provenance; never infer semantic verification."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def git(root, *args):
    p = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
    if p.returncode:
        raise RuntimeError(f'git {args[0]} failed in {root}: {p.stderr.decode(errors="replace").strip()}')
    return p.stdout.decode(errors='replace')


def frontmatter(text):
    match = re.match(r'^---\r?\n(.*?)\r?\n---', text, re.S)
    return dict(re.findall(r'^([a-z_]+):\s*([^\n]*)', match[1], re.M)) if match else {}


def candidate(rel):
    p = Path(rel)
    return p.suffix.lower() in {'.md', '.mdx', '.rst'} and (bool(re.search(r'(^|[-_])adr[-_]\w', p.name, re.I)) or any(x.lower() in {'adr', 'adrs'} for x in p.parts))


def classify(repo, rel, text, scope):
    p = Path(rel)
    if scope == 'outside-declared-estate':
        return 'outside-estate', 'Discovered neighbour; no adoption inferred from workspace co-location.'
    if rel.startswith('docs/estate-review/') or p.name.lower() in {'readme.md', 'index.md', 'preamble.md', 'template.md', 'adr-inventory.md', 'adr-lineage.md', 'adr-candidates.md', 'adr-history-closeout.md'} or 'template' in p.stem.lower() or '/examples/' in rel or '/templates/' in rel or '/.claude/agents/' in '/' + rel:
        return 'support', 'Index, template, example or agent instructions; not an adopted decision.'
    if 'cross-link stub, not the canonical document' in text:
        return 'support', 'Cross-link stub; resolve canonical owner before evaluating.'
    if '```json-ld' in text and any(x in rel for x in ('knowledge/pages/', 'KnowledgeGraph/pages/', 'ontology/pages/')):
        return 'ontology-content', 'Ontology content; not an operative architecture decision.'
    if repo == 'project4' or (repo == 'nostr-rust-forum' and rel.startswith('docs/sprint/')) or any(x in p.parts for x in ('archive', 'archived')):
        return 'historical', 'Frozen lineage; current obligations follow successor records and historical companions.'
    if repo in {'ruvector', 'RuView'} and ('/ruvector/' in '/' + rel or repo == 'ruvector'):
        return 'imported', 'Upstream/component intent; estate applicability is limited to the consumed code path.'
    if '/skills/' in '/' + rel:
        return 'skill-local', 'Skill package decision; does not govern the estate runtime unless a consumer adopts it.'
    return 'operative-candidate', 'Compare the record with its source assessment; declared axes are not verification.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    w = args.workspace.resolve()
    selected = {'VisionFlow', 'project', 'project/agentbox', 'solid-pod-rs', 'nostr-rust-forum', 'dreamlab-ai-website', 'loom', 'knowledgeGraph', 'project4', 'visionGraph', 'WasmVOWL', 'vowl-wasm', 'dream-machine', 'ruvector', 'RuView', 'prose-sanitiser', 'diagram-ir'}
    # Top-level checkouts plus Git submodules declared by those checkouts.
    pending = [p for p in w.iterdir() if not p.name.startswith('.') and p.is_dir() and (p / '.git').exists()]
    pending.append(w / 'project/agentbox')
    roots = {}
    while pending:
        root = pending.pop().resolve()
        if root in roots or not root.is_relative_to(w):
            continue
        roots[root] = str(root.relative_to(w))
        gm = root / '.gitmodules'
        if gm.exists():
            for rel in re.findall(r'^\s*path\s*=\s*(.+?)\s*$', gm.read_text(), re.M):
                child = root / rel
                if (child / '.git').exists():
                    pending.append(child)
    records, repos, errors = [], [], []
    for root, repo in sorted(roots.items(), key=lambda x: x[1]):
        scope = 'estate-or-consumed-dependency' if repo in selected else 'outside-declared-estate'
        try:
            head = git(root, 'rev-parse', 'HEAD').strip()
            tracked = set(filter(None, git(root, 'ls-files', '-z').split('\0')))
            files = sorted(tracked | set(filter(None, git(root, 'ls-files', '--others', '--exclude-standard', '-z').split('\0'))))
            dirty = git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=no')
        except RuntimeError as exc:
            errors.append(str(exc)); continue
        own = []
        for rel in files:
            if not candidate(rel):
                continue
            file = root / rel
            if not file.is_file() or file.is_symlink():
                continue
            # A parent checkout must not count a nested repository twice.
            if any(file.is_relative_to(other) for other in roots if other != root and other.is_relative_to(root)):
                continue
            data = file.read_bytes(); text = data.decode(errors='replace'); fm = frontmatter(text)
            kind, disposition = classify(repo, rel, text, scope)
            match = re.search(r'ADR[-_][A-Za-z0-9]+(?:[-_]\d+)?', file.stem, re.I)
            ident = fm.get('id', match.group() if match else file.stem).strip('"\'')
            row = {'key': repo + ':' + rel, 'repo': repo, 'path': rel, 'id': ident, 'kind': kind, 'scope_disposition': disposition, 'sha256': hashlib.sha256(data).hexdigest(), 'tracked': rel in tracked, 'declared': {k: fm.get(k) for k in ('decision_status', 'implementation_status', 'activation_status', 'verified_commit', 'owner', 'review_trigger')}}
            own.append(row); records.append(row)
        repos.append({'path': repo, 'head': head, 'scope': scope, 'tracked_worktree_dirty': bool(dirty), 'tracked_status_sha256': hashlib.sha256(dirty.encode()).hexdigest(), 'adr_candidates': len(own), 'kinds': dict(collections.Counter(x['kind'] for x in own))})
    # Source bytes, rather than HEAD alone, bind the diagram audit in dirty trees.
    source_hashes = {}
    for topic in sorted((w / 'VisionFlow/docs/diagrams').glob('*/*.md')):
        fm = re.match(r'^---\n(.*?)\n---', topic.read_text(), re.S)
        if not fm:
            continue
        for rel in re.findall(r'^\s+-\s+([^\n]+)', fm[1], re.M):
            rel = rel.strip().strip('"\'').split('#')[0]
            p = w / 'VisionFlow' / rel
            if p.is_file():
                source_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    counts = dict(collections.Counter(r['kind'] for r in records))
    payload = {'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'method': 'Git tracked/nonignored ADR candidates in top-level checkouts and declared available submodules; symlink aliases deduplicated. Classification is scope routing, not semantic acceptance. Source hashes describe working-tree bytes. No credentials or deployment queried.', 'limits': ['Unregistered nested repositories and ignored/untracked ADRs are outside discovery.', 'Filename/frontmatter classification is provisional; see per-record review tables.', 'Historical and imported records are retained, not silently promoted to operative estate decisions.'], 'counts': counts, 'repos': repos, 'records': records, 'diagram_source_sha256': source_hashes, 'errors': errors}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'estate-inventory.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Workspace and ADR census', '', 'Generated by `scripts/estate-doc-audit.py`. Discovery and scope classification are separate from the source assessments in the dated audit. [Machine-readable records and source hashes](estate-inventory.json).', '', '| Repository | Scope | HEAD | Dirty tracked tree | ADR candidates |', '|---|---|---|---|---:|']
    for r in repos:
        lines.append(f'| `{r["path"]}` | {r["scope"]} | `{r["head"][:12]}` | {r["tracked_worktree_dirty"]} | {r["adr_candidates"]} |')
    lines += ['', '## Estate record dispositions', '', 'Every key contains the repository and full path: repeated ADR numbers are not the same decision. These are scope dispositions; source and runtime evidence is recorded separately.', '', '| Record | Kind | Declared decision / implementation / activation |', '|---|---|---|']
    for r in records:
        if r['kind'] == 'outside-estate':
            continue
        target = os.path.relpath(w / r['repo'] / r['path'], args.output.resolve())
        axes = ' / '.join(str(r['declared'].get(k) or 'unstated').replace('|', '/') for k in ('decision_status', 'implementation_status', 'activation_status'))
        label = r['key'].replace('|', '\\|')
        lines.append(f'| [{label}](<{target}>) | {r["kind"]} | {axes} |')
    (args.output / 'estate-inventory.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'repos': len(repos), 'candidates': len(records), 'counts': counts, 'diagram_sources': len(source_hashes), 'errors': errors}))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
