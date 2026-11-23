'use strict';
// Regression suite for the diagrams corpus repair convention and panel-ID aliases
// (scripts/lib/diagram-convention.cjs, wired into scripts/diagram-index-gen.cjs).
//
// Every rule is driven through the real CLI against a throwaway corpus, passing and
// failing, so a gate that quietly stops failing is caught here.
//
//   node --test tests/gates/diagram-convention.test.cjs      (or: node tests/gates/diagram-convention.test.cjs)
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const REPO = path.resolve(__dirname, '../..');
const GEN = path.join(REPO, 'scripts/diagram-index-gen.cjs');
const SHA = 'abc1234';
let base;
let n = 0;

before(() => { base = fs.mkdtempSync(path.join(os.tmpdir(), 'visionflow-diagram-convention-')); });
after(() => { fs.rmSync(base, { recursive: true, force: true }); });

const DIAGRAM = '```mermaid\nflowchart TB\n    A["Source"] --> B["Evidence"]\n```\n';

/** A panel: heading, optional lines straight under it, then a diagram. */
function panel(id, { title = 'Panel', under = [], body = '' } = {}) {
  return `## ${id} ${title}\n${under.map((l) => l + '\n').join('')}${DIAGRAM}${body ? body + '\n' : ''}`;
}
/** A topic file. `aliases` is a string of raw frontmatter lines, `panels` an array of panel texts. */
function topic(id, panels, { aliases = '', extra = '' } = {}) {
  return `---\nid: ${id}\ntitle: Test ${id}\narea: visionflow\ngoverning: []\nadrs: []\nsources: []\nverified_commit: ${SHA}\n${aliases}---\n${extra}${panels.join('')}`;
}
/** Build a corpus under a fresh root; files maps `visionflow/01-x.md` to text. Returns the diagrams dir. */
function corpus(files, baseline) {
  const root = path.join(base, `c${++n}`, 'docs/diagrams');
  for (const [rel, text] of Object.entries(files)) {
    const f = path.join(root, rel);
    fs.mkdirSync(path.dirname(f), { recursive: true });
    fs.writeFileSync(f, text);
  }
  if (baseline) fs.writeFileSync(path.join(root, 'resolved-ratchet.json'), JSON.stringify({ panels: baseline }));
  return root;
}
function run(root, ...flags) {
  const r = spawnSync(process.execPath, [GEN, root, '--check', '--no-source-paths', ...flags], { encoding: 'utf8' });
  return { status: r.status, out: r.stdout + r.stderr };
}
function report(root, ...flags) {
  const file = path.join(root, '..', 'report.json');
  const r = run(root, '--report', file, ...flags);
  return { ...r, json: fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null };
}
const ok = (r, msg) => assert.equal(r.status, 0, `${msg || 'expected a pass'}\n${r.out}`);
const bad = (r, re, msg) => { assert.equal(r.status, 1, `${msg || 'expected a failure'}\n${r.out}`); assert.match(r.out, re); };

// ------------------------------------------------------------------ baseline behaviour

test('a corpus with no RESOLVED marker and no baseline file passes', () => {
  ok(run(corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')]) })));
});

test('(a) RESOLVED in an unlisted panel fails, naming the panel', () => {
  const r = run(corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1'), panel('VF-01.2', { body: 'RESOLVED at abc1234: fixed.' })]) }));
  bad(r, /VF-01\.2:\d+ — RESOLVED marker/);
  assert.doesNotMatch(r.out, /VF-01\.1:\d+ — RESOLVED/);
});

test('(a) every form found in the corpus counts: in a node label, a Note, a heading', () => {
  const label = '```mermaid\nflowchart TB\n    A["RESOLVED at forum 6064622: now sends the header"]\n```\n';
  const note = '```mermaid\nsequenceDiagram\n    Note over A,B: RESOLVED ADR-2053: try_send result\n```\n';
  const files = { 'visionflow/01-a.md': topic('VF-01', [`## VF-01.1 Label\n${label}`, `## VF-01.2 Note\n${note}`, panel('VF-01.3', { title: 'RESOLVED ADR-2045 - removed' }), panel('VF-01.4', { body: '**Debt:** RESOLVED at x abc1234' })]) };
  const r = run(corpus(files));
  bad(r, /VF-01\.1:/);
  for (const id of ['VF-01.2', 'VF-01.3', 'VF-01.4']) assert.match(r.out, new RegExp(`${id.replace('.', '\\.')}:\\d+ — RESOLVED marker`));
});

test('lowercase "resolved" and RESOLVED outside a panel are not markers', () => {
  const files = { 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body: 'The name is resolved at boot; resolved N times.' })], { extra: '## Audit notes\nRESOLVED in an audit paragraph, not a panel.\n' }) };
  ok(run(corpus(files)));
});

test('(a) a listed panel passes', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body: 'RESOLVED at abc1234' })]) }, ['VF-01.1']);
  ok(run(root));
});

test('ratchet growth: a new RESOLVED panel fails even when the baseline exists and passes today', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body: 'RESOLVED at abc1234' }), panel('VF-01.2')]) }, ['VF-01.1']);
  ok(run(root));
  fs.appendFileSync(path.join(root, 'visionflow/01-a.md'), panel('VF-01.3', { body: 'RESOLVED at abc1234' }));
  bad(run(root), /VF-01\.3:\d+ — RESOLVED marker/);
});

test('ratchet staleness: an entry for a panel that no longer exists fails', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')]) }, ['VF-01.9']);
  bad(run(root), /resolved-ratchet\.json: stale entry — VF-01\.9 no longer exists/);
});

test('ratchet staleness: an entry whose panel no longer carries RESOLVED fails', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body: 'RESOLVED at abc1234' })]) }, ['VF-01.1']);
  ok(run(root));
  fs.writeFileSync(path.join(root, 'visionflow/01-a.md'), topic('VF-01', [panel('VF-01.1')]));
  bad(run(root), /stale entry — VF-01\.1 no longer carries a RESOLVED marker/);
});

test('ratchet staleness: an entry for a renumbered panel says where it went', () => {
  const files = {
    'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')]),
    'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1', { body: 'RESOLVED at abc1234' })], { aliases: 'aliases:\n  VF-01.5: VF-02.1\n' }),
  };
  bad(run(corpus(files, ['VF-01.5'])), /stale entry — VF-01\.5 no longer exists; it was renumbered to VF-02\.1/);
});

test('a malformed baseline is an error, not an empty one', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')]) });
  fs.writeFileSync(path.join(root, 'resolved-ratchet.json'), '{"panels": ["nope", "VF-01.1", "VF-01.1"]}');
  const r = run(root);
  bad(r, /'nope' is not a panel ID/);
  assert.match(r.out, /duplicate entry VF-01\.1/);
});

test('--update-resolved-baseline creates the file once, then only ever prunes', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body: 'RESOLVED at abc1234' }), panel('VF-01.2', { body: 'RESOLVED at abc1234' })]) });
  ok(run(root, '--update-resolved-baseline'));
  const read = () => JSON.parse(fs.readFileSync(path.join(root, 'resolved-ratchet.json'), 'utf8')).panels;
  assert.deepEqual(read(), ['VF-01.1', 'VF-01.2']);
  // a panel that newly carries RESOLVED is never added; the gate still fails on it
  fs.appendFileSync(path.join(root, 'visionflow/01-a.md'), panel('VF-01.3', { body: 'RESOLVED at abc1234' }));
  bad(run(root, '--update-resolved-baseline'), /VF-01\.3:\d+ — RESOLVED marker/);
  assert.deepEqual(read(), ['VF-01.1', 'VF-01.2']);
  // clearing a marker lets the update prune it
  fs.writeFileSync(path.join(root, 'visionflow/01-a.md'), topic('VF-01', [panel('VF-01.1', { body: 'RESOLVED at abc1234' }), panel('VF-01.2')]));
  ok(run(root, '--update-resolved-baseline'));
  assert.deepEqual(read(), ['VF-01.1']);
});

// ------------------------------------------------------------------ banners

const banner = (to, sha = SHA) => `> **Superseded** by ${to} at ${sha}.`;

test('(b) a banner naming an existing panel passes and the panel is reported frozen', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: ['', banner('VF-01.2')] }), panel('VF-01.2')]) });
  const r = report(root);
  ok(r);
  assert.deepEqual(r.json.convention.frozen.map((f) => [f.panel, f.successor, f.sha]), [['VF-01.1', 'VF-01.2', SHA]]);
});

test('(b) a banner naming no panel and no alias fails', () => {
  bad(run(corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [banner('VF-01.7')] })]) })), /VF-01\.1:\d+ — Superseded banner names VF-01\.7, which is neither a panel nor a resolvable alias/);
});

test('(b) a banner naming an aliased old ID resolves through the alias', () => {
  const files = {
    'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [banner('VF-01.5')] })]),
    'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1')], { aliases: 'aliases:\n  VF-01.5: VF-02.1\n' }),
  };
  const r = report(corpus(files));
  ok(r);
  const [f] = r.json.convention.frozen;
  assert.deepEqual([f.declared, f.via, f.successor, f.current], ['VF-01.5', 'VF-01.5', 'VF-02.1', 'VF-02.1']);
});

test('(b) a malformed banner fails: no bold, wrong sha, missing full stop, uppercase sha', () => {
  for (const line of ['> Superseded by VF-01.2 at abc1234.', `> **Superseded** by VF-01.2 at ${SHA}`, '> **Superseded** by VF-01.2 at xyz', '> **Superseded** by VF-01.2 at ABC1234.', '> **Superseded** by VF-01.2 at abc12.']) {
    const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [line] }), panel('VF-01.2')]) });
    bad(run(root), /malformed Superseded banner/, line);
  }
});

test('(b) a banner that is not directly under the heading fails', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body: banner('VF-01.2') }), panel('VF-01.2')]) });
  bad(run(root), /a Superseded line must sit directly under the panel heading/);
  const second = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: ['Some prose first.', banner('VF-01.2')] }), panel('VF-01.2')]) });
  bad(run(second), /must sit directly under the panel heading/);
});

test('a panel cannot supersede itself, and a supersession cycle fails', () => {
  bad(run(corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [banner('VF-01.1')] })]) })), /cannot be superseded by itself/);
  const cyc = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [banner('VF-01.2')] }), panel('VF-01.2', { under: [banner('VF-01.1')] })]) });
  bad(run(cyc), /supersession cycle/);
});

test('the banner is not counted against the three-prose-lines-per-diagram cap', () => {
  // one diagram + the banner + three prose lines = at the cap only because the banner is exempt
  const p = panel('VF-01.1', { under: [banner('VF-01.2')], body: 'one\ntwo\nthree' });
  ok(run(corpus({ 'visionflow/01-a.md': topic('VF-01', [p, panel('VF-01.2')]) })));
});

test('(c) a frozen panel carrying RESOLVED fails, baselined or not; clearing it then leaves a stale entry', () => {
  const frozenResolved = topic('VF-01', [panel('VF-01.1', { under: [banner('VF-01.2')], body: 'RESOLVED at abc1234' }), panel('VF-01.2')]);
  bad(run(corpus({ 'visionflow/01-a.md': frozenResolved })), /VF-01\.1:\d+ — a frozen panel may not carry a RESOLVED marker/);
  const baselined = corpus({ 'visionflow/01-a.md': frozenResolved }, ['VF-01.1']);
  bad(run(baselined), /a frozen panel may not carry a RESOLVED marker/);
  // the marker removed: the panel is clean but the baseline entry is now unneeded
  fs.writeFileSync(path.join(baselined, 'visionflow/01-a.md'), topic('VF-01', [panel('VF-01.1', { under: [banner('VF-01.2')] }), panel('VF-01.2')]));
  bad(run(baselined), /stale entry — VF-01\.1 is frozen and no longer needs it/);
  fs.writeFileSync(path.join(baselined, 'resolved-ratchet.json'), '{"panels": []}');
  ok(run(baselined));
});

// ------------------------------------------------------------------ aliases

const withAlias = (aliases, extraFiles = {}) => corpus({
  'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')]),
  'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1'), panel('VF-02.2')], { aliases }),
  ...extraFiles,
});

test('aliases: block and inline forms both parse and are reported', () => {
  const block = report(withAlias('aliases:\n  VF-01.9: VF-02.1   # moved in the split\n  VF-01.8: VF-02.2\n'));
  ok(block);
  assert.deepEqual(block.json.convention.aliases.map((a) => [a.from, a.to, a.topic]), [['VF-01.8', 'VF-02.2', 'VF-02'], ['VF-01.9', 'VF-02.1', 'VF-02']]);
  const inline = report(withAlias('aliases: {VF-01.9: VF-02.1, VF-01.8: VF-02.2}\n'));
  ok(inline);
  assert.equal(inline.json.convention.aliases.length, 2);
});

test('aliases: an empty map and an absent key are fine', () => {
  ok(run(withAlias('aliases:\n')));
  ok(run(withAlias('')));
});

test('aliases: a target that does not exist fails', () => {
  bad(run(withAlias('aliases:\n  VF-01.9: VF-02.7\n')), /alias VF-01\.9: VF-02\.7 — target panel VF-02\.7 does not exist/);
});

test('aliases: a source that is also a live panel fails', () => {
  bad(run(withAlias('aliases:\n  VF-01.1: VF-02.1\n')), /alias source VF-01\.1 is also a live panel ID/);
});

test('aliases: a chain fails and names the straight target', () => {
  bad(run(withAlias('aliases:\n  VF-01.9: VF-01.8\n  VF-01.8: VF-02.1\n')), /alias VF-01\.9: VF-01\.8 chains \(VF-01\.8 is itself aliased to VF-02\.1\); point VF-01\.9 straight at VF-02\.1/);
});

test('aliases: loops fail, including a self loop', () => {
  bad(run(withAlias('aliases:\n  VF-01.9: VF-01.8\n  VF-01.8: VF-01.9\n')), /loops \(/);
  bad(run(withAlias('aliases:\n  VF-01.9: VF-01.9\n')), /loops onto itself/);
});

test('aliases: two topics aliasing the same old ID fails', () => {
  const files = {
    'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')]),
    'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1')], { aliases: 'aliases:\n  VF-01.9: VF-02.1\n' }),
    'visionflow/03-c.md': topic('VF-03', [panel('VF-03.1')], { aliases: 'aliases:\n  VF-01.9: VF-03.1\n' }),
  };
  bad(run(corpus(files)), /alias source VF-01\.9 is also aliased by visionflow\/0[23]-[bc]\.md; two topics may not alias the same old ID/);
});

test('aliases: a value that is not a panel ID, or a list, fails', () => {
  bad(run(withAlias('aliases:\n  VF-01.9: VF-02\n')), /'VF-02' is not a panel ID/);
  bad(run(withAlias('aliases:\n  - VF-01.9\n')), /aliases must be a map/);
});

test('aliases resolve a `see` reference, a bare mention and a panel link, and the report lists them', () => {
  const files = {
    'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body: 'See VF-01.9 and also VF-01.8, linked [here](02-b.md#vf-019-old-name).' })]),
    'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1')], { aliases: 'aliases:\n  VF-01.9: VF-02.1\n  VF-01.8: VF-02.1\n' }),
  };
  const r = report(corpus(files));
  ok(r);
  const via = r.json.convention.references.viaAlias.map((x) => x.ref).sort();
  assert.deepEqual(via, ['02-b.md#vf-019-old-name', 'VF-01.8', 'VF-01.9'].sort());
  assert.equal(r.json.convention.references.unresolved.length, 0);
});

test('an unresolved `see` reference and an unresolved panel link fail; a bare unresolved mention only warns', () => {
  const files = (body) => ({
    'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { body })]),
    'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1')]),
  });
  bad(run(corpus(files('see VF-02.9'))), /"see VF-02\.9" names no panel and no alias/);
  bad(run(corpus(files('[x](02-b.md#vf-029-gone)'))), /link 02-b\.md#vf-029-gone matches no panel heading and no alias/);
  const mention = report(corpus(files('VF-02.9 was retired.')));
  ok(mention);
  assert.ok(mention.json.warnings.some((w) => /VF-02\.9 is mentioned but is neither a panel nor an alias/.test(w)));
  ok(run(corpus(files('[x](02-b.md#vf-021-panel) and see VF-02.1'))));
});

test('register and review-ledger references follow aliases; unresolved ones only warn', () => {
  const root = withAlias('aliases:\n  VF-01.9: VF-02.1\n');
  fs.writeFileSync(path.join(root, 'REGISTER-test.md'), '- [VF-01.9 old](visionflow/02-b.md#vf-019-old) tension, see VF-01.9\n- VF-02.8 mentioned\n');
  fs.writeFileSync(path.join(root, 'review-ledger.jsonl'), JSON.stringify({ kind: 'triage', skipped: 'VF-01.9 reviewed, VF-02.7 too' }) + '\n');
  const r = report(root);
  ok(r);
  const via = r.json.convention.references.viaAlias.map((x) => `${x.where}:${x.ref}`);
  assert.ok(via.includes('REGISTER-test.md:VF-01.9') && via.includes('review-ledger.jsonl:VF-01.9'), via.join(','));
  assert.ok(r.json.convention.references.unresolved.some((x) => x.where === 'review-ledger.jsonl' && x.ref === 'VF-02.7'));
  assert.ok(r.json.warnings.some((w) => /review-ledger\.jsonl — VF-02\.7/.test(w)));
});

// ------------------------------------------------------------------ indexes

test('the generated indexes mark frozen panels with their successor and list the aliases', () => {
  const files = {
    'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { title: 'Old', under: [banner('VF-01.5')] })]),
    'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1', { title: 'New' })], { aliases: 'aliases:\n  VF-01.5: VF-02.1\n' }),
  };
  const root = corpus(files);
  const r = spawnSync(process.execPath, [GEN, root, '--no-source-paths'], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stdout + r.stderr);
  const cov = fs.readFileSync(path.join(root, 'COVERAGE.md'), 'utf8');
  assert.match(cov, /\| \[VF-01\.1 Old\]\(visionflow\/01-a\.md#vf-011-old\) — superseded by \[VF-02\.1\]\(visionflow\/02-b\.md#vf-021-new\) \|/);
  assert.match(cov, /## Superseded panels \(frozen\)/);
  assert.match(cov, /\| \[VF-01\.1\]\(visionflow\/01-a\.md#vf-011-old\) \| \[VF-02\.1\]\(visionflow\/02-b\.md#vf-021-new\) \| `VF-01\.5` \| `abc1234` \|/);
  assert.match(cov, /## Panel aliases/);
  assert.match(cov, /\| `VF-01\.5` \| \[VF-02\.1\]\(visionflow\/02-b\.md#vf-021-new\) \| VF-02 \|/);
  const readme = fs.readFileSync(path.join(root, 'README.md'), 'utf8');
  assert.match(readme, /### Superseded panels \(frozen\)/);
  assert.match(readme, /### Panel aliases/);
  assert.match(readme, /\| `VF-01\.5` \| \[VF-02\.1\]\(visionflow\/02-b\.md#vf-021-new\) \| VF-02 \|/);
});

test('a corpus with neither frozen panels nor aliases gets index sections of neither kind', () => {
  const root = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')]) });
  assert.equal(spawnSync(process.execPath, [GEN, root, '--no-source-paths'], { encoding: 'utf8' }).status, 0);
  const cov = fs.readFileSync(path.join(root, 'COVERAGE.md'), 'utf8');
  assert.doesNotMatch(cov, /Superseded panels|Panel aliases|superseded by/);
});

// ------------------------------------------------------------------ size cap

test('size cap: more than 15 panels warns in the console and the JSON report, and does not fail', () => {
  const panels = Array.from({ length: 16 }, (_, i) => panel(`VF-01.${i + 1}`));
  const r = report(corpus({ 'visionflow/01-a.md': topic('VF-01', panels) }));
  ok(r);
  assert.match(r.out, /! visionflow\/01-a\.md: size — 16 panels \(cap 15\)/);
  assert.equal(r.json.totals.sizeWarnings, 1);
  assert.deepEqual(r.json.convention.size.warnings.map((w) => [w.topic, w.panels, w.reasons.length]), [['VF-01', 16, 1]]);
  assert.ok(r.json.warnings.some((w) => /size — 16 panels/.test(w)));
});

test('size cap: more than 600 lines warns; both reasons are reported together', () => {
  const pad = Array.from({ length: 610 }, () => '').join('\n');
  const r = report(corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1')], { extra: pad + '\n' }) }));
  ok(r);
  assert.match(r.out, /size — 6\d\d lines \(cap 600\)/);
  const both = report(corpus({ 'visionflow/01-a.md': topic('VF-01', Array.from({ length: 16 }, (_, i) => panel(`VF-01.${i + 1}`)), { extra: pad + '\n' }) }));
  ok(both);
  assert.equal(both.json.convention.size.warnings[0].reasons.length, 2);
});

test('size cap: exactly 15 panels and exactly 600 lines are within the cap', () => {
  const panels = Array.from({ length: 15 }, (_, i) => panel(`VF-01.${i + 1}`));
  let text = topic('VF-01', panels);
  const have = text.split('\n').length - 1;
  text = topic('VF-01', panels, { extra: '\n'.repeat(600 - have) });
  assert.equal(text.split('\n').length - 1, 600);
  const r = report(corpus({ 'visionflow/01-a.md': text }));
  ok(r);
  assert.equal(r.json.totals.sizeWarnings, 0);
  assert.doesNotMatch(r.out, /size —/);
});

// ------------------------------------------------------------------ frozen revision

test('the extended banner "(frozen at <sha>)" parses; the plain form parses; both are reported', () => {
  const files = { 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [`> **Superseded** by VF-01.2 at 3f66d51 (frozen at 59b1aca).`] }), panel('VF-01.2', { under: [banner('VF-01.3', '3f66d51')] }), panel('VF-01.3')]) };
  const r = report(corpus(files));
  ok(r);
  assert.deepEqual(r.json.convention.frozen.map((f) => [f.panel, f.sha, f.frozenAt]), [['VF-01.1', '3f66d51', '59b1aca'], ['VF-01.2', '3f66d51', null]]);
  for (const bad1 of ['(frozen at xyz)', '(frozen 59b1aca)', '(frozen at 59b1aca) extra']) {
    bad(run(corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [`> **Superseded** by VF-01.2 at 3f66d51 ${bad1}.`] }), panel('VF-01.2')]) })), /malformed Superseded banner/, bad1);
  }
});

test('a frozen panel\'s citations are read at its own "frozen at" revision, not the topic\'s verified_commit', () => {
  const repo = path.join(base, 'cite-repo');
  const git = (...a) => { const r = spawnSync('git', ['-C', repo, '-c', 'user.email=t@example.invalid', '-c', 'user.name=t', '-c', 'commit.gpgsign=false', ...a], { encoding: 'utf8' }); assert.equal(r.status, 0, r.stderr); return r.stdout.trim(); };
  fs.mkdirSync(path.join(repo, 'docs/diagrams/visionflow'), { recursive: true });
  git('init', '-q');
  fs.writeFileSync(path.join(repo, 'a.txt'), 'one\ntwo\nthree\n');
  git('add', 'a.txt'); git('commit', '-qm', 'old');
  const old = git('rev-parse', '--short=9', 'HEAD');
  fs.writeFileSync(path.join(repo, 'a.txt'), 'one\n');          // the file shrank: line 3 no longer exists
  git('add', 'a.txt'); git('commit', '-qm', 'new');
  const cur = git('rev-parse', '--short=9', 'HEAD');
  const label = '```mermaid\nflowchart TB\n    A["a.txt:3"] --> B["a.txt:1"]\n```\n';
  const topicText = (under) => `---\nid: VF-01\ntitle: T\narea: visionflow\ngoverning: []\nadrs: []\nsources: [a.txt]\nverified_commit: ${cur}\n---\n## VF-01.1 Old\n${under}${label}## VF-01.2 New\n\`\`\`mermaid\nflowchart TB\n    A["a.txt:1"]\n\`\`\`\n`;
  const dir = path.join(repo, 'docs/diagrams');
  const file = path.join(dir, 'visionflow/01-a.md');
  const cite = () => spawnSync(process.execPath, [GEN, dir, '--check', '--cite-check'], { encoding: 'utf8' });
  fs.writeFileSync(file, topicText(''));
  assert.match(cite().stdout + cite().stderr, /a\.txt:3 past EOF/, 'unfrozen: read at the topic revision, line 3 is gone');
  fs.writeFileSync(file, topicText(`> **Superseded** by VF-01.2 at ${cur}.\n`));
  assert.match(cite().stdout + cite().stderr, /a\.txt:3 past EOF/, 'frozen without "frozen at": still the topic revision');
  fs.writeFileSync(file, topicText(`> **Superseded** by VF-01.2 at ${cur} (frozen at ${old}).\n`));
  const r = cite();
  assert.doesNotMatch(r.stdout + r.stderr, /past EOF/, 'frozen at the old revision: line 3 exists there');
  assert.equal(r.status, 0, r.stdout + r.stderr);
  // the other diagram in the topic is still judged at the topic revision
  fs.writeFileSync(file, topicText(`> **Superseded** by VF-01.2 at ${cur} (frozen at ${old}).\n`).replace('A["a.txt:1"]\n```\n', 'A["a.txt:3"]\n```\n'));
  { const c = cite(); assert.match(c.stdout + c.stderr, /VF-01\.2 — a\.txt:3 past EOF/); }
});

// ------------------------------------------------------------------ --only and the real corpus

test('--only still resolves banner and alias targets outside the filter, and skips ratchet staleness', () => {
  const files = {
    'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [banner('VF-02.1')] })]),
    'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1')]),
  };
  const root = corpus(files, ['VF-02.9']);
  ok(run(root, '--only', '01-a'));          // VF-02.1 lives outside the filter; VF-02.9 is stale but whole-tree only
  bad(run(root), /stale entry — VF-02\.9/);
  const gone = corpus({ 'visionflow/01-a.md': topic('VF-01', [panel('VF-01.1', { under: [banner('VF-02.4')] })]), 'visionflow/02-b.md': topic('VF-02', [panel('VF-02.1')]) });
  bad(run(gone, '--only', '01-a'), /VF-02\.4, which is neither a panel nor a resolvable alias/);
  assert.equal(run(root, '--only', '01-a', '--update-resolved-baseline').status, 2);
});

test('the real corpus passes today, with a non-empty ratchet that is exactly its RESOLVED panels', () => {
  const dir = path.join(REPO, 'docs/diagrams');
  const out = path.join(base, 'real-report.json');
  const r = spawnSync(process.execPath, [GEN, dir, '--check', '--no-source-paths', '--report', out], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stdout + r.stderr);
  const j = JSON.parse(fs.readFileSync(out, 'utf8'));
  assert.ok(j.convention.resolved.baseline.length > 0);
  assert.deepEqual(j.convention.resolved.marked, j.convention.resolved.baseline);
  assert.deepEqual(j.convention.resolved.stale, []);
});
