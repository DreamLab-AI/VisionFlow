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
  // A citation whose declared revision cannot be resolved is READ FROM THE
  // WORKING TREE. It used to pass silently, which is how a corpus could report
  // zero warnings while a slice of it had never been checked against any commit.
  // Strict mode must now refuse it, and say why.
  const unresolved = run('--strict-citations');
  assert.equal(unresolved.status, 1, 'an unresolvable declared revision must not pass strict mode');
  assert.ok(/unverified/.test(unresolved.stdout + unresolved.stderr), 'the refusal must name the citation as unverified');
  assert.ok(/citations: \d+ verified at the declared revision/.test(unresolved.stdout), 'a run must state what it verified');
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

  // ── the citation allowlist ─────────────────────────────────────────────────
  // Some citations can never resolve at a declared revision because the file
  // exists in no commit (a gitignored build artefact is the live case). Those are
  // waived BY NAME, counted apart from what was verified, and printed as
  // allowlisted. Anything unverifiable and not named must still fail, or the
  // waiver has quietly become a blanket one.
  fs.writeFileSync(path.join(base, 'fixture.js'), 'const answer = 42;\n');
  const allowlistFile = path.join(root, 'citation-allowlist.json');
  const writeAllow = (allow) => fs.writeFileSync(allowlistFile, JSON.stringify({ allow }, null, 2));
  const reason = 'fixture: gitignored artefact, exists in no commit';
  // Both topics now cite fixture.js at a revision that does not contain it: VF-01
  // declares a sha git cannot resolve, AB-01 a sha that does not cover its repo.
  const abFixture = path.join(root, 'agentbox/01-fixture.md');
  fs.writeFileSync(fixture, source.replace('Source', 'fixture.js:1'));
  fs.writeFileSync(abFixture, fs.readFileSync(abFixture, 'utf8').replace('sources: []', 'sources: [fixture.js]').replace('Source', 'fixture.js:1'));

  writeAllow([{ topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture.js', line: 1, reason }]);
  const partly = run('--strict-citations');
  const partlyOut = partly.stdout + partly.stderr;
  assert.equal(partly.status, 1, 'an unverifiable citation that is not allowlisted must still fail strict mode');
  assert.ok(/AB-01\.1/.test(partlyOut), 'the refusal must name the citation that is not allowlisted');
  assert.ok(!/VF-01\.1 — fixture\.js:1 was read from the working tree/.test(partlyOut), 'an allowlisted citation must not be reported as a failure');

  writeAllow([
    { topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture.js', line: 1, reason },
    { topic: 'AB-01', diagram: 'AB-01.1', path: 'fixture.js', line: 1, reason },
  ]);
  const allowed = run('--strict-citations');
  assert.equal(allowed.status, 0, 'an allowlisted unverifiable citation must pass strict mode');
  assert.ok(/citations: \d+ verified at the declared revision, \d+ unverified \(working-tree fallback\), \d+ unresolvable, 2 allowlisted \(unverifiable\)/.test(allowed.stdout),
    'the counts line must carry the allowlisted total, apart from the verified one');
  assert.ok(/fixture\.js:1 allowlisted \(unverifiable\): fixture: gitignored artefact/.test(allowed.stdout),
    'each allowlisted citation must be printed with its reason');

  // A waiver that matches nothing has outlived the citation it was written for.
  writeAllow([
    { topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture.js', line: 1, reason },
    { topic: 'AB-01', diagram: 'AB-01.1', path: 'fixture.js', line: 1, reason },
    { topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture.js', line: 42, reason },
  ]);
  const stale = run('--strict-citations');
  assert.equal(stale.status, 1, 'a stale allowlist entry must fail the run rather than broaden silently');
  assert.ok(/matched no citation — it is stale/.test(stale.stdout + stale.stderr), 'the refusal must say the entry is stale');

  // An entry covers exactly the citation it names. Naming the wrong line must not
  // waive the right one, or `line` is decoration and the entry is a file-wide waiver.
  writeAllow([
    { topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture.js', line: 42, reason },
    { topic: 'AB-01', diagram: 'AB-01.1', path: 'fixture.js', line: 1, reason },
  ]);
  const wrongLine = run('--strict-citations');
  assert.equal(wrongLine.status, 1, 'an entry for another line must not waive this citation');
  assert.ok(/VF-01\.1 — fixture\.js:1 was read from the working tree/.test(wrongLine.stdout + wrongLine.stderr),
    'the citation the allowlist did not name must still be refused as unverified');

  // An entry may omit diagram and line to cover a whole topic, but a topic-wide
  // entry must never reach into another topic that happens to cite the same path.
  writeAllow([{ topic: 'VF-01', path: 'fixture.js', reason }]);
  const topicWide = run('--strict-citations');
  const topicWideOut = topicWide.stdout + topicWide.stderr;
  assert.equal(topicWide.status, 1, 'a topic-wide entry must not waive a citation in another topic');
  assert.ok(/AB-01\.1 — fixture\.js:1 was read from the working tree/.test(topicWideOut), 'the other topic must still be refused');
  assert.ok(!/VF-01\.1 — fixture\.js:1 was read from the working tree/.test(topicWideOut), 'a topic-wide entry must waive its own topic');

  // Granularity below the topic needs a second diagram citing a second file, or
  // `diagram` and `path` are decoration: with one diagram and one path in the
  // tree, every entry that matches the topic matches the only citation there is.
  fs.writeFileSync(path.join(base, 'fixture2.js'), 'const other = 7;\n');
  fs.writeFileSync(fixture, source.replace('Source', 'fixture.js:1').replace('sources: [fixture.js]', 'sources: [fixture.js, fixture2.js]')
    + `\n## VF-01.2 Second\n\x60\x60\x60mermaid\nflowchart TB\n    C["fixture2.js:1"] --> D["Evidence"]\n\x60\x60\x60\n`);

  writeAllow([
    { topic: 'VF-01', path: 'fixture.js', reason },
    { topic: 'AB-01', diagram: 'AB-01.1', path: 'fixture.js', line: 1, reason },
  ]);
  const otherPath = run('--strict-citations');
  assert.equal(otherPath.status, 1, 'an entry naming one file must not waive a citation of another');
  assert.ok(/VF-01\.2 — fixture2\.js:1 was read from the working tree/.test(otherPath.stdout + otherPath.stderr),
    'the file the allowlist did not name must still be refused');

  writeAllow([
    { topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture.js', line: 1, reason },
    { topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture2.js', line: 1, reason },
    { topic: 'AB-01', diagram: 'AB-01.1', path: 'fixture.js', line: 1, reason },
  ]);
  const wrongDiagram = run('--strict-citations');
  assert.equal(wrongDiagram.status, 1, 'an entry naming the wrong diagram must not waive the citation');
  assert.ok(/allow\[1\][^\n]*matched no citation — it is stale/.test(wrongDiagram.stdout + wrongDiagram.stderr),
    'the misdirected entry must be reported as stale, not silently applied elsewhere');

  writeAllow([{ topic: 'ZZ-99', path: 'fixture.js', reason }]);
  assert.equal(run('--strict-citations').status, 1, 'an allowlist entry naming a topic outside the tree must fail');
  fs.writeFileSync(allowlistFile, '{ not json');
  assert.equal(run('--no-source-paths').status, 1, 'a malformed allowlist must fail even without source access');

  // ── the hosted-runner freshness gate ───────────────────────────────────────
  // The runner cannot resolve cross-repository revisions, so it cannot repeat the
  // citation check. It asserts instead that VERIFICATION.md was written against
  // the revisions the tree declares NOW — the case a re-stamp without a re-resolve
  // would otherwise slip through.
  writeAllow([
    { topic: 'VF-01', diagram: 'VF-01.1', path: 'fixture.js', line: 1, reason },
    { topic: 'VF-01', diagram: 'VF-01.2', path: 'fixture2.js', line: 1, reason },
    { topic: 'AB-01', diagram: 'AB-01.1', path: 'fixture.js', line: 1, reason },
  ]);
  assert.equal(run('--strict-citations').status, 0, 'the allowlisted tree must be green before freshness is judged');
  const fresh = run('--no-source-paths', '--check-verification');
  assert.equal(fresh.status, 0, 'a VERIFICATION.md just written must read as fresh');
  assert.ok(/verification: fresh/.test(fresh.stdout), 'a fresh run must say so');
  assert.ok(/allowlist: 3 well-formed entries/.test(fresh.stdout), 'the runner must report the allowlist it validated');
  fs.writeFileSync(fixture, fs.readFileSync(fixture, 'utf8').replace('verified_commit: abc1234', 'verified_commit: def5678'));
  const restamped = run('--no-source-paths', '--check-verification');
  assert.equal(restamped.status, 1, 'a re-stamped topic must invalidate VERIFICATION.md');
  assert.ok(/VERIFICATION\.md is stale/.test(restamped.stdout + restamped.stderr), 'the refusal must name VERIFICATION.md as stale');

  console.log('DIAGRAM-INDEX-TEST-OK');
} finally {
  fs.rmSync(base, { recursive: true, force: true });
}
