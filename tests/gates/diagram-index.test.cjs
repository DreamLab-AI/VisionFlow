'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const base = fs.mkdtempSync(path.join(os.tmpdir(), 'visionflow-diagram-index-'));
try {
  const root = path.join(base, 'docs/diagrams');
  for (const [area, id, adrs] of [['visionflow', 'VF-01', 'ADR-2002'], ['agentbox', 'AB-01', 'ADR-2002'], ['estate', 'ES-01', 'ADR-2002, agentbox:ADR-2003']]) {
    const dir = path.join(root, area);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, '01-fixture.md'), `---\nid: ${id}\ntitle: Test\narea: ${area}\ngoverning: []\nadrs: [${adrs}]\nsources: []\nverified_commit: abc1234\n---\n## ${id}.1 Test\n\x60\x60\x60mermaid\nflowchart TB\n    A["Source"] --> B["Evidence"]\n\x60\x60\x60\n`);
  }
  const result = spawnSync(process.execPath, ['scripts/diagram-index-gen.cjs', root, '--no-source-paths'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  const coverage = fs.readFileSync(path.join(root, 'COVERAGE.md'), 'utf8');
  for (const key of ['visionflow:ADR-2002', 'agentbox:ADR-2002', 'estate-unresolved:ADR-2002', 'agentbox:ADR-2003']) assert.ok(coverage.includes(`| ${key} |`), key);
  assert.ok(!coverage.includes('| ADR-2002 |'), 'must not collapse unrelated decisions');
  assert.ok(coverage.includes('declared source revisions:'));
  assert.ok(!coverage.includes('verified against commits:'));
  assert.ok(fs.readFileSync(path.join(root, 'README.md'), 'utf8').includes('diagram-index-gen.cjs'));
  const fixture = path.join(root, 'visionflow/01-fixture.md');
  fs.writeFileSync(path.join(base, 'fixture.js'), 'const answer = 42;\n');
  const source = fs.readFileSync(fixture, 'utf8').replace('sources: []', 'sources: [fixture.js]');
  fs.writeFileSync(fixture, source.replace('Source', 'fixture.js:999'));
  const run = (...flags) => spawnSync(process.execPath, ['scripts/diagram-index-gen.cjs', root, '--check', ...flags], { encoding: 'utf8' });
  assert.equal(run('--cite-check').status, 0, 'legacy advisory mode remains advisory');
  assert.equal(run('--strict-citations').status, 1, 'strict gate must reject a past-EOF citation');
  assert.equal(run('--strict-citations', '--no-source-paths').status, 2, 'cannot waive source access in strict mode');
  fs.writeFileSync(fixture, source.replace('Source', 'fixture.js:1'));
  assert.equal(run('--strict-citations').status, 0, 'a resolving citation passes');
  const git = (...args) => {
    const result = spawnSync('git', ['-C', base, ...args], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
  git('init', '-q');
  git('add', 'fixture.js');
  git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'reference bytes');
  const revision = git('rev-parse', 'HEAD');
  fs.writeFileSync(fixture, source.replace('abc1234', revision).replace('Source', 'fixture.js:1'));
  fs.writeFileSync(path.join(base, 'fixture.js'), '\n');
  assert.equal(run('--strict-citations').status, 0, 'declared revision mode retains historical source semantics');
  assert.equal(run('--strict-citations', '--worktree-citations').status, 1, 'current-byte mode must detect drift from the declared revision');
  console.log('DIAGRAM-INDEX-TEST-OK');
} finally {
  fs.rmSync(base, { recursive: true, force: true });
}
