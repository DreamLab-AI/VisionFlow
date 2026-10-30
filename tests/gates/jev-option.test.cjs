#!/usr/bin/env node
// The Jev option (docs/jev-option.md), VF-01/03/06 + ES-09. Mocked transport only:
// no live calls. Three groups:
//   A. adapter: endpoint resolution, every fallback trigger, wire shape, no content in logs
//   B. fallback equivalence: the REAL pre-change scripts (git show of the base commit) vs
//      the changed scripts, flag off and every fallback mode -> identical stdout, exit
//      code and files written (stderr differs only by the single `jev:` log line)
//   C. accept path: a confident mock yields the advisory annotation and the verdict,
//      exit code and the deterministic output are unchanged
'use strict';
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const repo = path.resolve(__dirname, '..', '..');
const jev = require(path.join(repo, 'scripts/lib/jev.cjs'));
const Q = require(path.join(repo, 'scripts/lib/jev-questions.cjs'));
const MOCK = path.join(__dirname, 'jev-mock-fetch.cjs');
const BASE = '094e164'; // the commit this option was cut from: the pre-change scripts
let passed = 0;
const ok = (name, fn) => Promise.resolve().then(fn).then(() => { passed++; console.log(`ok   ${name}`); },
  (e) => { console.log(`FAIL ${name}\n     ${e.message.split('\n')[0]}`); process.exitCode = 1; });

const resp = (status, obj, headers = {}) => ({ status, headers: { get: (k) => headers[k.toLowerCase()] ?? null }, text: async () => (typeof obj === 'string' ? obj : JSON.stringify(obj)) });
const ENV = { VISIONFLOW_JEV_ENDPOINT: 'http://127.0.0.1:1/v1/systemone' };
const qs = { pick: { type: 'choice', instructions: 'x', criteria: { a: 'A', b: 'B' } }, rate: { type: 'score', instructions: 'x', criteria: ['lo', 'mid', 'hi'] }, yes: { type: 'noul', instructions: 'x' } };
const good = { model: 'm', usage: {}, answers: { pick: { type: 'choice', choice: 'a', confidence: 0.9, probabilities: {} }, rate: { type: 'score', score: 1.5, confidence: 0.8, legend: {}, probabilities: {} }, yes: { type: 'noul', noul: 0.95 } } };
const call = (extra) => jev.decide({ version: 'test/v1', state: { secret: 'TOP-SECRET-CONTENT' }, questions: qs, env: ENV, log: () => {}, ...extra });
const withAnswers = (patch) => ({ ...good, answers: { ...good.answers, ...patch } });

function oldScript(rel) { // pre-change source, placed at the same relative path in a scratch tree
  const r = spawnSync('git', ['-C', repo, 'show', `${BASE}:${rel}`], { encoding: 'utf8', maxBuffer: 1 << 26 });
  return r.status === 0 ? r.stdout : null;
}
const sh = (script, args, env = {}, cwd = repo) => {
  const r = spawnSync(process.execPath, [...(env.JEV_MOCK ? ['--require', MOCK] : []), script, ...args], { encoding: 'utf8', cwd, env: { PATH: process.env.PATH, HOME: process.env.HOME, ...env } });
  return { status: r.status, stdout: r.stdout, stderr: r.stderr.split('\n').filter((l) => !l.startsWith('jev:') && !l.startsWith('JEV-ADVISORY')).join('\n') };
};
const tree = (dir) => { const o = {}; (function w(d) { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); if (e.isDirectory()) w(p); else o[path.relative(dir, p)] = fs.readFileSync(p, 'utf8'); } })(dir); return o; };
const FALLBACKS = [{}, { VISIONFLOW_JEV: '1' }, { VISIONFLOW_JEV: '1', ...ENV, JEV_MOCK: 'error' }, { VISIONFLOW_JEV: '1', ...ENV, JEV_MOCK: 'timeout', VISIONFLOW_JEV_TIMEOUT_MS: '150' },
  { VISIONFLOW_JEV: '1', ...ENV, JEV_MOCK: 'malformed' }, { VISIONFLOW_JEV: '1', ...ENV, JEV_MOCK: 'lowconf' }, { VISIONFLOW_JEV: '1', ...ENV, JEV_MOCK: 'http500' },
  { VISIONFLOW_JEV: '1', ...ENV, JEV_MOCK: 'missing' }, { VISIONFLOW_JEV: 'triage,claim,diagram', ...ENV, JEV_MOCK: 'badchoice' }, { ...ENV, JEV_MOCK: 'accept' } /* flag off + endpoint set */];
const FB_NAMES = ['flag off', 'flag on, no endpoint', 'transport error', 'timeout', 'malformed', 'low confidence', 'http 500', 'missing answer', 'choice outside set', 'endpoint set, flag off'];

(async () => {
  // ── A. adapter ──────────────────────────────────────────────────────────
  await ok('flag off: no request is built, no log', async () => { let hit = 0; const logs = [];
    const r = await call({ enabled: false, fetchImpl: async () => { hit++; return resp(200, good); }, log: (l) => logs.push(l) });
    assert.deepEqual([r.outcome, r.reason, hit, logs.length], ['fallback', 'flag-off', 0, 0]); });
  await ok('enabled(): off unless explicitly on; list selects items', () => {
    assert.equal(jev.enabled('health', { env: {} }), false); assert.equal(jev.enabled('health', { env: { VISIONFLOW_JEV: '0' } }), false);
    assert.equal(jev.enabled('health', { env: { VISIONFLOW_JEV: '1' } }), true); assert.equal(jev.enabled('health', { env: { VISIONFLOW_JEV: 'claim,triage' } }), false);
    assert.equal(jev.enabled('triage', { env: { VISIONFLOW_JEV: 'claim,triage' } }), true); assert.equal(jev.enabled('claim', { env: {}, cli: 'claim' }), true);
    assert.equal(jev.cliFlag(['check', '--jev=health']), 'health'); assert.equal(jev.cliFlag(['check']), undefined); });
  await ok('no endpoint resolved -> no-endpoint', async () => assert.equal((await call({ env: {}, fetchImpl: async () => resp(200, good) })).reason, 'no-endpoint'));
  await ok('resolution order: explicit config > agentbox proxy (no auth) > typesafe bearer', () => {
    const e1 = jev.resolveEndpoint({ ...ENV, VISIONFLOW_JEV_KEY_ENV: 'K', K: 'k1', AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p', AGENTBOX_SYSTEM_ONE_MODEL: 'pm', TYPESAFE_API_KEY: 't' });
    assert.equal(e1.url, ENV.VISIONFLOW_JEV_ENDPOINT); assert.equal(e1.headers.Authorization, 'Bearer k1');
    const e2 = jev.resolveEndpoint({ AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p', AGENTBOX_SYSTEM_ONE_MODEL: 'pm', TYPESAFE_API_KEY: 't' });
    assert.deepEqual([e2.url, e2.model, 'Authorization' in e2.headers], ['http://p', 'pm', false]);
    const e3 = jev.resolveEndpoint({ TYPESAFE_API_KEY: 't' });
    assert.deepEqual([e3.url, e3.model, e3.headers.Authorization], [jev.DEFAULT_API, 'jev-latest', 'Bearer t']);
    assert.equal(jev.resolveEndpoint({ AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p' }).reason, 'no-endpoint'); });
  await ok('an agentbox-projected local route serves local-only questions; public ones go to the proxy', () => {
    const PX = { AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p', AGENTBOX_SYSTEM_ONE_MODEL: 'pm' };
    const LOCAL = { VISIONFLOW_JEV_ENDPOINT: 'http://laya', VISIONFLOW_JEV_MODEL: 'lm', VISIONFLOW_JEV_LOCAL: '1' };
    assert.deepEqual([jev.resolveEndpoint({ ...PX, ...LOCAL }, 'public').url, jev.resolveEndpoint({ ...PX, ...LOCAL }, 'public').model], ['http://p', 'pm']);
    assert.equal(jev.resolveEndpoint({ ...PX, ...LOCAL }, 'local-only').url, 'http://laya');
    assert.equal(jev.resolveEndpoint(LOCAL, 'public').url, 'http://laya', 'no proxy: the local route still serves');
    const manual = { ...PX, VISIONFLOW_JEV_ENDPOINT: 'http://mine' };
    assert.equal(jev.resolveEndpoint(manual, 'public').url, 'http://mine', 'an endpoint named by hand always wins'); });
  await ok('proxy aliases: AGENTBOX_SKILL_ROUTE_API / _MODEL accepted, primary names win, no Authorization', () => {
    const a = jev.resolveEndpoint({ AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SKILL_ROUTE_API: 'http://alias', AGENTBOX_SKILL_ROUTE_MODEL: 'am', TYPESAFE_API_KEY: 't' });
    assert.deepEqual([a.url, a.model, 'Authorization' in a.headers], ['http://alias', 'am', false]);
    const m = jev.resolveEndpoint({ AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p', AGENTBOX_SKILL_ROUTE_MODEL: 'am' });
    assert.deepEqual([m.url, m.model], ['http://p', 'am']);
    const w = jev.resolveEndpoint({ AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p', AGENTBOX_SYSTEM_ONE_MODEL: 'pm', AGENTBOX_SKILL_ROUTE_API: 'http://alias', AGENTBOX_SKILL_ROUTE_MODEL: 'am' });
    assert.deepEqual([w.url, w.model], ['http://p', 'pm']);
    assert.equal(jev.resolveEndpoint({ AGENTBOX_SKILL_ROUTE_API: 'http://alias', AGENTBOX_SKILL_ROUTE_MODEL: 'am' }).reason, 'no-endpoint', 'alias needs PROXY=1');
    assert.equal(jev.resolveEndpoint({ AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SKILL_ROUTE_API: 'http://alias', AGENTBOX_SKILL_ROUTE_MODEL: 'am' }, 'local-only').reason, 'local-only-refused'); });
  await ok('dataClassFor: local-only only when the consumer flag is 1 (ADR-2139 amendment, 2026-10-30)', () => {
    assert.equal(jev.dataClassFor({}, 'VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY'), 'public');
    assert.equal(jev.dataClassFor({ VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY: '0' }, 'VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY'), 'public');
    assert.equal(jev.dataClassFor({ VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY: '1' }, 'VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY'), 'local-only');
    const PX = { AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p/v1/systemone', AGENTBOX_SYSTEM_ONE_MODEL: 'pm', AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL: '0' };
    assert.equal(jev.resolveEndpoint(PX, jev.dataClassFor(PX, 'VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY')).url, 'http://p/v1/systemone');
    const pinned = { ...PX, VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY: '1' };
    assert.equal(jev.resolveEndpoint(pinned, jev.dataClassFor(pinned, 'VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY')).reason, 'local-only-refused'); });
  await ok('local-only refuses proxy and typesafe unless declared local', () => {
    const proxy = { AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p', AGENTBOX_SYSTEM_ONE_MODEL: 'pm' };
    assert.equal(jev.resolveEndpoint(proxy, 'local-only').reason, 'local-only-refused');
    assert.equal(jev.resolveEndpoint({ TYPESAFE_API_KEY: 't' }, 'local-only').reason, 'local-only-refused');
    assert.equal(jev.resolveEndpoint({ ...proxy, AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL: '1' }, 'local-only').url, 'http://p');
    assert.equal(jev.resolveEndpoint(ENV, 'local-only').url, ENV.VISIONFLOW_JEV_ENDPOINT); });
  await ok('fallback triggers each map to their reason', async () => {
    const cases = [
      ['transport', async () => { throw new TypeError('fetch failed'); }], ['http-500', async () => resp(500, {})], ['http-401', async () => resp(401, {})],
      ['http-429', async () => resp(429, {}, { 'retry-after': '0' })], ['http-529', async () => resp(529, {}, { 'retry-after': '0' })],
      ['malformed', async () => resp(200, 'nope')], ['malformed', async () => resp(200, { model: 'm' })],
      ['malformed', async () => resp(200, withAnswers({ yes: undefined }))], ['malformed', async () => resp(200, withAnswers({ pick: { type: 'choice', choice: 'zzz', confidence: 0.9 } }))],
      ['malformed', async () => resp(200, withAnswers({ rate: { type: 'score', score: 9, confidence: 0.9 } }))], ['malformed', async () => resp(200, withAnswers({ yes: { type: 'noul', noul: 1.5 } }))],
      ['malformed', async () => resp(200, withAnswers({ yes: { type: 'choice', choice: 'a', confidence: 1 } }))],
      ['low-confidence', async () => resp(200, withAnswers({ pick: { type: 'choice', choice: 'a', confidence: 0.1 } }))], ['low-confidence', async () => resp(200, withAnswers({ yes: { type: 'noul', noul: 0.55 } }))],
    ];
    for (const [reason, f] of cases) assert.equal((await call({ fetchImpl: f, retries: 1 })).reason, reason, reason); });
  await ok('timeout: hard deadline aborts a hanging transport', async () => {
    const t0 = Date.now();
    const r = await call({ deadlineMs: 80, fetchImpl: (u, i) => new Promise((_, rej) => i.signal.addEventListener('abort', () => rej(Object.assign(new Error('x'), { name: 'AbortError' })))) });
    assert.equal(r.reason, 'timeout'); assert.ok(Date.now() - t0 < 2000); });
  await ok('429 retried within the deadline, then accepted', async () => { let n = 0;
    const r = await call({ retries: 1, fetchImpl: async () => (n++ ? resp(200, good) : resp(429, {}, { 'retry-after': '0' })) });
    assert.deepEqual([r.outcome, n], ['accepted', 2]); });
  await ok('accept path returns validated answers', async () => { const r = await call({ fetchImpl: async () => resp(200, good) }); assert.equal(r.outcome, 'accepted'); assert.equal(r.answers.pick.choice, 'a'); });
  // System One (Typesafe and the local façade) does not repeat the question's type on each answer.
  await ok('untagged answers are judged by the question type', async () => {
    const untagged = { model: 'jev-1.14.0', usage: {}, answers: { pick: { choice: 'a', confidence: 0.9, probabilities: {} }, rate: { score: 1.5, confidence: 0.8, distribution: [] }, yes: { noul: 0.95 } } };
    const r = await call({ fetchImpl: async () => resp(200, untagged) });
    assert.equal(r.outcome, 'accepted'); assert.equal(r.answers.pick.choice, 'a');
    const weak = { ...untagged, answers: { ...untagged.answers, yes: { noul: 0.55 } } };
    assert.equal((await call({ fetchImpl: async () => resp(200, weak) })).reason, 'low-confidence'); });
  await ok('wire shape: model, state, questions; proxy sends no Authorization; typesafe bearer', async () => {
    const seen = [];
    const f = async (url, init) => { seen.push({ url, init, body: JSON.parse(init.body) }); return resp(200, good); };
    await call({ env: { AGENTBOX_SYSTEM_ONE_PROXY: '1', AGENTBOX_SYSTEM_ONE_ENDPOINT: 'http://p/v1/systemone', AGENTBOX_SYSTEM_ONE_MODEL: 'pm', AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL: '1' }, fetchImpl: f, dataClass: 'local-only' });
    await call({ env: { TYPESAFE_API_KEY: 'sk-test' }, fetchImpl: f });
    assert.equal(seen[0].init.method, 'POST'); assert.equal(seen[0].body.model, 'pm'); assert.equal('Authorization' in seen[0].init.headers, false);
    assert.equal(seen[1].url, jev.DEFAULT_API); assert.equal(seen[1].init.headers.Authorization, 'Bearer sk-test'); assert.equal(seen[1].body.model, 'jev-latest');
    assert.deepEqual(Object.keys(seen[1].body).sort(), ['model', 'questions', 'state']); });
  await ok('logs carry reason + set version only: never state, body or key', async () => { const logs = [];
    await call({ env: { TYPESAFE_API_KEY: 'sk-SECRETKEY' }, fetchImpl: async () => resp(500, 'TOP-SECRET-CONTENT'), log: (l) => logs.push(l) });
    await call({ fetchImpl: async () => resp(200, good), log: (l) => logs.push(l) });
    assert.equal(logs.length, 2); assert.equal(logs[0], 'jev: fallback reason=http-500 set=test/v1'); assert.match(logs[1], /^jev: accepted set=test\/v1/);
    assert.ok(!logs.join('').match(/SECRET/)); });

  // question-set wire shape: Choice criteria = map, Score criteria = ordered list, Noul none required
  await ok('question sets: choice criteria map, score criteria list, versions, instructions carry meaning', () => {
    const sets = [Q.claimCheck([{ key: 'C1/x', text: 't', status: 'partial', excerpts: [{ ref: 'a:1', text: '1: x' }] }]), Q.citationCheck([{ claim: 'A --> B', file: 'x/a.rs', lines: [{ n: 1, t: 'fn a()' }] }]),
      Q.healthPriority([{ kind: 'ci', repo: 'r', workflow: 'w', outcome: 'failure' }]), Q.failureTriage([{ kind: 'ci', repo: 'r', workflow: 'w', outcome: 'failure' }])];
    for (const s of sets) { assert.match(s.version, /\/v\d+$/); for (const q of Object.values(s.questions)) {
      assert.ok(q.instructions.length > 80); if (q.type === 'choice') assert.ok(q.criteria && !Array.isArray(q.criteria) && Object.keys(q.criteria).length >= 3);
      if (q.type === 'score') assert.ok(Array.isArray(q.criteria) && q.criteria.length >= 2); } }
    assert.ok('insufficient' in Q.failureTriage([{}]).questions.triage_0.criteria); assert.ok('insufficient' in Q.claimCheck([{ key: 'k', excerpts: [] }]).questions.claim_0.criteria);
    assert.ok(!('diagramReview' in Q), 'the whole-diagram question measured no signal and is retired'); });

  // VF-06-S1-02: one atomic Noul per citation. The wording is pinned because the
  // 0.9 flag thres…[2489 chars]

  // ── B/C. estate-health check (VF-03, ES-09) ─────────────────────────────
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'jev-vf-'));
  const snap = path.join(tmp, 'snap.json');
  const run = (id, wf, c) => ({ id, workflow: wf, status: 'completed', conclusion: c, url: `https://example.invalid/runs/${id}` });
  fs.writeFileSync(snap, JSON.stringify({ schema: 'visionflow.estate-health/1', generated_at: new Date().toISOString(),
    repos: [{ name: 'alpha', url: 'u', readable: true, ci: { state: 'red', runs: [run(1, 'Build', 'failure'), run(2, 'Lint', 'success')] } },
      { name: 'beta', url: 'u2', readable: true, ci: { state: 'red', runs: [run(3, 'Deploy', 'failure')] } }, { name: 'gamma', url: 'u3', readable: false, ci: { state: 'unknown', runs: [] } }],
    surfaces: [{ name: 'site', ok: false, status: 404 }] }));
  const oldEH = path.join(tmp, 'old', 'scripts', 'estate-health.mjs');
  fs.mkdirSync(path.dirname(oldEH), { recursive: true });
  const ehSrc = oldScript('scripts/estate-health.mjs');
  if (!ehSrc) console.log(`SKIP equivalence: ${BASE} is not in this clone (shallow?)`);
  const EH = path.join(repo, 'scripts/estate-health.mjs');
  let ehRef = null;
  if (ehSrc) {
    fs.writeFileSync(oldEH, ehSrc);
    const ref = ehRef = sh(oldEH, ['check', '--file', snap, '--max-age-hours', '1000']);
    await ok('estate-health: pre-change reference is RED with the sentinel last', () => { assert.equal(ref.status, 1); assert.equal(ref.stdout.trimEnd().split('\n').pop(), 'ESTATE-HEALTH-RED'); });
    FALLBACKS.forEach((env, i) => ok(`estate-health check == pre-change: ${FB_NAMES[i]}`, () => {
      const now = sh(EH, ['check', '--file', snap, '--max-age-hours', '1000'], env);
      assert.deepEqual(now, ref); }));
    await ok('estate-health check == pre-change: every item via --jev flag, mock errors', () => {
      for (const mode of ['error', 'malformed', 'lowconf']) assert.deepEqual(sh(EH, ['check', '--file', snap, '--max-age-hours', '1000', '--jev'], { ...ENV, JEV_MOCK: mode }), ref); });
  }
  const logf = path.join(tmp, 'eh.log');
  await ok('VF-03 accept: advisory ranking inserted, deterministic lines/exit/sentinel unchanged, Score wire shape', () => {
    const r = sh(EH, ['check', '--file', snap, '--max-age-hours', '1000', '--jev=health'], { ...ENV, JEV_MOCK: 'accept', JEV_MOCK_LOG: logf });
    const lines = r.stdout.trimEnd().split('\n');
    assert.equal(r.status, 1); assert.equal(lines.pop(), 'ESTATE-HEALTH-RED');
    const det = lines.filter((l) => !l.startsWith('JEV ')); const adv = lines.filter((l) => l.startsWith('JEV priority'));
    if (ehRef) assert.equal(det.join('\n') + '\n', ehRef.stdout.replace(/ESTATE-HEALTH-RED\n$/, ''));
    assert.equal(adv.length, 4); assert.match(adv[0], /^JEV priority 1 site /); assert.ok(adv.every((l) => /advisory; unmeasured/.test(l)) && !adv.some((l) => /triage/.test(l)));
    const req = JSON.parse(fs.readFileSync(logf, 'utf8').trim().split('\n')[0]);
    assert.ok(Object.values(req.body.questions).every((q) => q.type === 'score' && Array.isArray(q.criteria))); assert.equal(Object.keys(req.body.questions).length, 4);
    assert.equal(Object.keys(req.body.state.items).length, 4); });
  await ok('ES-09 accept: triage lines for failed runs only, choice criteria map incl. insufficient; verdict unchanged', () => {
    fs.rmSync(logf, { force: true });
    const r = sh(EH, ['check', '--file', snap, '--max-age-hours', '1000', '--jev=triage'], { ...ENV, JEV_MOCK: 'accept', JEV_MOCK_LOG: logf });
    const adv = r.stdout.split('\n').filter((l) => l.startsWith('JEV triage'));
    assert.equal(r.status, 1); assert.equal(r.stdout.trimEnd().split('\n').pop(), 'ESTATE-HEALTH-RED');
    assert.equal(adv.length, 2); assert.match(adv[0], /^JEV triage alpha Build -> source-build conf=0\.90 \(advisory; not a verdict\)$/);
    const req = JSON.parse(fs.readFileSync(logf, 'utf8').trim().split('\n')[0]);
    assert.ok(Object.values(req.body.questions).every((q) => q.type === 'choice' && 'insufficient' in q.criteria));
    assert.ok(!JSON.stringify(req.body).includes('example.invalid'), 'URLs are not sent'); });
  await ok('estate-health: green snapshot sends nothing even with the flag', () => {
    const g = path.join(tmp, 'green.json'); fs.writeFileSync(g, JSON.stringify({ schema: 'visionflow.estate-health/1', generated_at: new Date().toISOString(), repos: [], surfaces: [] }));
    fs.rmSync(logf, { force: true });
    const r = sh(EH, ['check', '--file', g, '--jev'], { ...ENV, JEV_MOCK: 'accept', JEV_MOCK_LOG: logf });
    assert.equal(r.status, 0); assert.ok(!fs.existsSync(logf)); });

  // ── VF-01: check-augmentation-citations.cjs ─────────────────────────────
  const matrixRel = 'docs/architecture/compatibility-matrix.md';
  const matrix = fs.readFileSync(path.join(repo, matrixRel), 'utf8');
  const estate = path.join(tmp, 'estate');
  for (const d of ['nostr-rust-forum', 'agentbox', 'project']) fs.mkdirSync(path.join(estate, d), { recursive: true });
  for (const p of new Set([...matrix.matchAll(/`([^`\s]+?):\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*`/g)].map((m) => m[1]))) for (const d of ['nostr-rust-forum', 'agentbox', 'project']) {
    const f = path.join(estate, d, p); fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, Array.from({ length: 5000 }, (_, i) => `// line ${i + 1} of ${p}`).join('\n')); }
  const acSrc = oldScript('scripts/check-augmentation-citations.cjs');
  const AC = path.join(repo, 'scripts/check-augmentation-citations.cjs');
  if (acSrc) {
    const oldAC = path.join(tmp, 'oldac', 'scripts', 'check-augmentation-citations.cjs');
    fs.mkdirSync(path.dirname(oldAC), { recursive: true }); fs.mkdirSync(path.join(tmp, 'oldac', path.dirname(matrixRel)), { recursive: true });
    fs.writeFileSync(oldAC, acSrc); fs.writeFileSync(path.join(tmp, 'oldac', matrixRel), matrix);
    for (const root of [estate, path.join(tmp, 'no-such-estate')]) {
      const ref = sh(oldAC, ['--root', root]);
      FALLBACKS.forEach((env, i) => ok(`augmentation gate == pre-change (${root === estate ? 'passing' : 'failing'} estate): ${FB_NAMES[i]}`, () => assert.deepEqual(sh(AC, ['--root', root], { ...env, AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL: '1' }), ref)));
    }
  }
  await ok('VF-01 accept: JEV claim annotation after the summary; gate exit unchanged; pinned local-only refuses typesafe, unpinned uses it', () => {
    fs.rmSync(logf, { force: true });
    const base = sh(AC, ['--root', estate]);
    const r = sh(AC, ['--root', estate, '--jev'], { ...ENV, JEV_MOCK: 'accept', JEV_MOCK_LOG: logf });
    assert.equal(r.status, 0); assert.ok(r.stdout.startsWith(base.stdout)); const adv = r.stdout.slice(base.stdout.length).trimEnd().split('\n');
    assert.equal(adv.length, 1); assert.match(adv[0], /^JEV claim C1 Durable net value\/nostr-rust-forum: overstated conf=0\.90 \(advisory; reviewer disposition/);
    const req = JSON.parse(fs.readFileSync(logf, 'utf8').trim().split('\n')[0]); const c0 = req.body.state.claims.claim_0;
    assert.ok(c0.cited_excerpts[0].ref.includes(':') && c0.cited_excerpts[0].text.length > 0 && c0.assigned_status);
    fs.rmSync(logf, { force: true });
    const priv = sh(AC, ['--root', estate, '--jev'], { TYPESAFE_API_KEY: 'k', VISIONFLOW_JEV_CLAIMS_LOCAL_ONLY: '1', JEV_MOCK: 'accept', JEV_MOCK_LOG: logf });
    assert.deepEqual(priv, base); assert.ok(!fs.existsSync(logf), 'no request when pinned local-only is not satisfied');
    const pub = sh(AC, ['--root', estate, '--jev'], { TYPESAFE_API_KEY: 'k', JEV_MOCK: 'accept', JEV_MOCK_LOG: logf });
    assert.equal(pub.status, 0); assert.ok(fs.existsSync(logf), 'unpinned: the TypeSafe step is used'); });
  await ok('VF-01: a failing gate stays failing (exit 1, same FAIL lines) when Jev says everything is supported', () => {
    const r = sh(AC, ['--root', path.join(tmp, 'no-such-estate'), '--jev'], { ...ENV, JEV_MOCK: 'accept' }); assert.equal(r.status, 1); });

  // ── VF-06: diagram-index-gen.cjs ────────────────────────────────────────
  const dgSrc = oldScript('scripts/diagram-index-gen.cjs');
  const DG = path.join(repo, 'scripts/diagram-index-gen.cjs');
  const mkTree = (name) => { const base = path.join(tmp, name); const root = path.join(base, 'docs/diagrams'); fs.mkdirSync(path.join(root, 'visionflow'), { recursive: true });
    fs.writeFileSync(path.join(base, 'fixture.js'), 'function route(x) {\n  return x ? send(x) : drop();\n}\n');
    fs.writeFileSync(path.join(root, 'visionflow/01-fixture.md'), '---\nid: VF-01\ntitle: Test\narea: visionflow\ngoverning: []\nadrs: []\nsources: [fixture.js]\nverified_commit: abc1234\n---\n## VF-01.1 Test\n```mermaid\nflowchart TB\n    A["route fixture.js:1"] --> B["drop fixture.js:2"]\n```\n');
    return { base, root }; };
  if (dgSrc) {
    const oldDG = path.join(tmp, 'olddg', 'scripts', 'diagram-index-gen.cjs'); fs.mkdirSync(path.dirname(oldDG), { recursive: true }); fs.writeFileSync(oldDG, dgSrc);
    const refT = mkTree('dg-ref'); const ref = sh(oldDG, [refT.root, '--check', '--cite-check']); const refFiles = tree(refT.root);
    FALLBACKS.forEach((env, i) => ok(`diagram gen == pre-change (stdout, exit, files): ${FB_NAMES[i]}`, () => {
      const t = mkTree(`dg-${i}`); const now = sh(DG, [t.root, '--check', '--cite-check', '--jev', ...(env.VISIONFLOW_JEV ? [] : [])].filter((a) => a !== '--jev' || env.VISIONFLOW_JEV), { ...env, AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL: '1' });
      assert.deepEqual(now, { ...ref, stdout: now.stdout.split(t.root).join(refT.root), stderr: now.stderr.split(t.root).join(refT.root) }); assert.deepEqual(tree(t.root), refFiles); }));
  }
  await ok('flag off + endpoint set: no request is made by any of the three scripts', () => {
    const t = mkTree('nolog').root; const lf = path.join(tmp, 'nolog.log'); fs.rmSync(lf, { force: true });
    const env = { ...ENV, JEV_MOCK: 'accept', JEV_MOCK_LOG: lf, AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL: '1' };
    sh(EH, ['check', '--file', snap, '--max-age-hours', '1000'], env); sh(AC, ['--root', estate], env); sh(DG, [t, '--check', '--cite-check'], env);
    assert.ok(!fs.existsSync(lf)); });

  await ok('VF-06 accept: one Noul per resolved citation; flagged ones on stderr + report; stdout, exit and indexes untouched', () => {
    const a = mkTree('dg-acc-off'); const b = mkTree('dg-acc-on'); fs.rmSync(logf, { force: true });
    const off = sh(DG, [a.root, '--check', '--cite-check']); const reportPath = path.join(tmp, 'dg-report.json');
    const raw = spawnSync(process.execPath, ['--require', MOCK, DG, b.root, '--check', '--cite-check', '--jev', '--jev-report', reportPath], { encoding: 'utf8', env: { PATH: process.env.PATH, ...ENV, JEV_MOCK: 'accept', JEV_MOCK_LOG: logf } });
    assert.equal(raw.status, off.status); assert.equal(raw.stdout.split(b.root).join(a.root), off.stdout);
    assert.match(raw.stderr, /JEV-ADVISORY .*01-fixture\.md:VF-01\.1 fixture\.js:1 p_defect=0\.90 \(cited lines may not hold this label; check by hand\)/);
    assert.match(raw.stderr, /JEV-ADVISORY .*01-fixture\.md:VF-01\.1 fixture\.js:2 p_defect=0\.90/);
    assert.deepEqual(tree(b.root), tree(a.root), 'no index/VERIFICATION/frontmatter change');
    const rep = JSON.parse(fs.readFileSync(reportPath, 'utf8')); assert.equal(rep.advisory, true); assert.equal(rep.flag_at, 0.9);
    assert.deepEqual(rep.results.map((x) => [x.cite, x.flagged]), [['fixture.js:1', true], ['fixture.js:2', true]]);
    const reqs = fs.readFileSync(logf, 'utf8').trim().split('\n'); assert.equal(reqs.length, 1, 'both citations in one batched request');
    const req = JSON.parse(reqs[0]); assert.deepEqual(Object.keys(req.body.questions), ['q0', 'q1']);
    assert.equal(req.body.questions.q0.type, 'noul'); assert.deepEqual(Object.keys(req.body.questions.q0.criteria), ['true', 'false']);
    assert.equal(req.body.state.items.q0.cited_lines, '1: function route(x) {'); assert.match(req.body.state.items.q1.cited_lines, /^2: .*drop\(\)/);
    assert.match(req.body.state.items.q0.claim, /route fixture\.js:1/); });
  await ok('VF-06: the 0.9 flag applies only to the model family it was calibrated on', () => {
    const items = [{ cite: 'a.rs:1' }];
    assert.equal(Q.citationFlags({ q0: { noul: 0.05 } }, items, 0.9, 'jev-1.14.0')[0].flagged, true);
    const other = Q.citationFlags({ q0: { noul: 0.05 } }, items, 0.9, 'laya-typed-decisions')[0];
    assert.deepEqual([other.flagged, other.calibrated, Number(other.p_defect.toFixed(2))], [false, false, 0.95]); });
  await ok('VF-06: --jev without --cite-check asks nothing and says why', () => {
    const t = mkTree('dg-nocite'); fs.rmSync(logf, { force: true });
    const raw = spawnSync(process.execPath, ['--require', MOCK, DG, t.root, '--check', '--jev'], { encoding: 'utf8', env: { PATH: process.env.PATH, ...ENV, JEV_MOCK: 'accept', JEV_MOCK_LOG: logf } });
    assert.equal(raw.status, 0); assert.ok(!fs.existsSync(logf), 'no request'); assert.match(raw.stderr, /needs --cite-check/); });

  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`\n${passed} passed${process.exitCode ? ', FAILURES above' : ''}`);
  if (!process.exitCode) console.log('JEV-OPTION-TEST-OK');
})();
