#!/usr/bin/env node
'use strict';
// Bind a passing current-source citation check to the exact input bytes.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync, execFileSync } = require('node:child_process');
const yaml = require('js-yaml');
const root = path.resolve(__dirname, '..');
const output = process.argv[2];
if (!output) throw new Error('usage: capture-diagram-source-snapshot.cjs OUTPUT.json');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
function capture() {
  const topics = {}, sources = {}, excluded = [];
  for (const area of fs.readdirSync(path.join(root, 'docs/diagrams')).sort()) {
    const folder = path.join(root, 'docs/diagrams', area);
    if (!fs.statSync(folder).isDirectory()) continue;
    for (const name of fs.readdirSync(folder).filter(n => n.endsWith('.md')).sort()) {
      const relative = `docs/diagrams/${area}/${name}`;
      const bytes = fs.readFileSync(path.join(root, relative));
      const match = bytes.toString().match(/^---\r?\n([\s\S]*?)\r?\n---/);
      if (!match) continue;
      topics[relative] = hash(bytes);
      const meta = yaml.load(match[1]);
      for (const source of meta.sources || []) {
        const file = source.split(':')[0];
        if (/(^|\/)ruview(\/|$)/i.test(file)) {
          if (!excluded.includes(file)) excluded.push(file);
          continue;
        }
        const absolute = path.resolve(root, file);
        if (fs.statSync(absolute).isDirectory()) {
          // Only the declared directory is represented; no recursive read claim.
          sources[file] = { type: 'directory' };
        } else sources[file] = { type: 'file', sha256: hash(fs.readFileSync(absolute)) };
      }
    }
  }
  return { topics, sources, excluded };
}
const before = capture();
const result = spawnSync(process.execPath, ['scripts/diagram-index-gen.cjs', 'docs/diagrams',
  '--check', '--strict-citations', '--worktree-citations'], { cwd: root, encoding: 'utf8' });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
if (result.status !== 0) process.exit(result.status || 1);
const after = capture();
if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('Diagram/source bytes changed during validation; retry after reconciliation.');
const report = { method: 'Current file SHA256 before and after a passing strict worktree citation check. Directories are markers, not recursive content attestations. RuView source reads excluded. Render execution and semantic audit are separate.',
  base_head: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
  generated_at: new Date().toISOString(), ...after };
fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ topics: Object.keys(after.topics).length, sources: Object.keys(after.sources).length, excluded: after.excluded.length }));
