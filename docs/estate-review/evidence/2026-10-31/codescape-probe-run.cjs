#!/usr/bin/env node
// Minimal build/rlib selection glue for the Rust audit probe, not a product adapter.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
if (process.argv.length !== 4) {
  throw new Error('Usage: node codescape-probe-run.cjs <codescape-root> <output.json>');
}
const root = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
const rows = execFileSync('cargo', [
  'build', '--offline', '--locked', '-p', 'codescape-server', '--lib', '--message-format=json',
], { cwd: root, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024, stdio: ['ignore', 'pipe', 'inherit'] })
  .trim().split('\n').map(JSON.parse);
const scratch = fs.mkdtempSync(path.join(path.dirname(output), '.codescape-probe-'));
try {
  const binary = path.join(scratch, 'probe');
  const args = ['--edition=2021', path.join(__dirname, 'codescape-command-probe.rs'), '-o', binary];
  for (const name of ['codescape_server', 'codescape_collab', 'codescape_layout', 'tokio', 'serde_json']) {
    const matches = rows.filter(r => r.reason === 'compiler-artifact' && r.target.name === name
      && r.filenames.some(f => f.endsWith('.rlib'))
      && (name !== 'serde_json' || r.features.includes('arbitrary_precision')));
    if (matches.length !== 1) throw new Error(`Ambiguous or missing runtime artefact ${name}: ${matches.length}`);
    const rlib = matches[0].filenames.find(f => f.endsWith('.rlib'));
    args.push('--extern', `${name}=${rlib}`, '-L', `dependency=${path.dirname(rlib)}`);
  }
  execFileSync('rustc', args, { stdio: 'inherit' });
  const result = execFileSync(binary, [root], { encoding: 'utf8', timeout: 30000 });
  const parsed = JSON.parse(result);
  if (parsed.cases.length !== 11) throw new Error('Unexpected probe case count');
  fs.writeFileSync(output, result);
  process.stdout.write('11 audit cases passed their assertions.\n');
} finally {
  fs.rmSync(scratch, { recursive: true, force: true });
}
