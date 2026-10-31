#!/usr/bin/env node
// Minimal build/rlib selection glue for the Rust audit probe, not a product adapter.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
if (process.argv.length !== 5) {
  throw new Error('Usage: node xr-links-run.cjs <codescape-root> <visionclaw-root> <output.json>');
}
const root = path.resolve(process.argv[2]);
const visionclaw = path.resolve(process.argv[3]);
const output = path.resolve(process.argv[4]);
const rows = execFileSync('cargo', [
  'build', '--offline', '--locked', '-p', 'codescape-server', '--lib', '--message-format=json',
], { cwd: root, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024, stdio: ['ignore', 'pipe', 'inherit'] })
  .trim().split('\n').map(JSON.parse);
const scratch = fs.mkdtempSync(path.join(path.dirname(output), '.codescape-probe-'));
try {
  const binary = path.join(scratch, 'probe');
  fs.copyFileSync(path.join(visionclaw, 'xr-client/rust/src/link_index.rs'), path.join(scratch, 'link_index.rs'));
  fs.copyFileSync(path.join(__dirname, 'xr-links-probe.rs'), path.join(scratch, 'probe.rs'));
  const args = ['--edition=2021', path.join(scratch, 'probe.rs'), '-o', binary];
  for (const name of ['codescape_server', 'codescape_ingest', 'codescape_model', 'serde', 'serde_json']) {
    const matches = rows.filter(r => r.reason === 'compiler-artifact' && r.target.name === name
      && r.filenames.some(f => f.endsWith('.rlib'))
      && (name !== 'serde' || r.features.includes('derive'))
      && (name !== 'serde_json' || r.features.includes('arbitrary_precision')));
    if (matches.length !== 1) throw new Error(`Ambiguous or missing runtime artefact ${name}: ${matches.length}`);
    const rlib = matches[0].filenames.find(f => f.endsWith('.rlib'));
    args.push('--extern', `${name}=${rlib}`, '-L', `dependency=${path.dirname(rlib)}`);
  }
  execFileSync('rustc', args, { stdio: 'inherit' });
  const result = execFileSync(binary, [scratch], { encoding: 'utf8', timeout: 30000 });
  const parsed = JSON.parse(result);
  if (parsed.cases.length !== 8) throw new Error('Unexpected probe case count');
  fs.writeFileSync(path.join(scratch, 'raw_string.rs'), parsed.cases[0].fixture);
  execFileSync('rustc', ['--crate-type=lib', '--emit=metadata', path.join(scratch, 'raw_string.rs'), '-o', path.join(scratch, 'raw_string.rmeta')], { stdio: 'inherit' });
  parsed.raw_string_fixture_compiles_as_rust = true;
  const unitArgs = ['--edition=2021', '--test', path.join(scratch, 'link_index.rs'), '-o', path.join(scratch, 'unit')];
  unitArgs.push(...args.slice(4));
  execFileSync('rustc', unitArgs, { stdio: 'inherit' });
  const suite = execFileSync(path.join(scratch, 'unit'), [], { encoding: 'utf8', timeout: 30000 });
  fs.writeFileSync(output.replace(/\.json$/, '-unit.txt'), suite.trim() + '\n');
  fs.writeFileSync(output, JSON.stringify(parsed, null, 2) + '\n');
  process.stdout.write('8 audit cases passed their assertions.\n');
} finally {
  fs.rmSync(scratch, { recursive: true, force: true });
}
