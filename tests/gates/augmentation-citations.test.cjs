#!/usr/bin/env node
// Drives deliberate defects through scripts/check-augmentation-citations.cjs and
// asserts the gate goes red; then asserts the real matrix passes (EXP-AC-001).
'use strict';
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const repo = path.resolve(__dirname, '..', '..');
const checker = path.join(repo, 'scripts', 'check-augmentation-citations.cjs');
const matrixPath = path.join(repo, 'docs/architecture/compatibility-matrix.md');
const original = fs.readFileSync(matrixPath, 'utf8');
let failed = 0;
const expect = (name, cond, detail = '') => { console.log(`${cond ? 'ok' : 'FAIL'} - ${name}${detail ? ': ' + detail : ''}`); if (!cond) failed++; };
const run = (root) => spawnSync('node', [checker, '--root', root], { encoding: 'utf8' });

// Fixture estate: a fake sibling root where only the cited files we create exist.
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ac-gate-'));
for (const dir of ['nostr-rust-forum', 'agentbox', 'project']) fs.mkdirSync(path.join(root, dir), { recursive: true });
const cited = [...original.matchAll(/`([^`\s]+?):\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*`/g)].map(m => m[1]);
const map = { 'nostr-rust-forum': 'nostr-rust-forum', agentbox: 'agentbox', VisionClaw: 'project' };
// Materialise every path the real table cites, under every substrate dir (cheap, exact-match not needed for the gate).
const FIXTURE_LINES = '\n'.repeat(4999); // 5000-line stand-ins so every real citation's line exists
for (const p of new Set(cited)) for (const d of Object.values(map)) { const f = path.join(root, d, p); fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, FIXTURE_LINES); }

// 1. Real matrix against the materialised fixture passes.
let r = run(root);
expect('real matrix passes against fixture estate', r.status === 0, r.stdout.trim() || r.stderr.trim());

// 2. A cited path that does not exist turns the gate red.
try {
  fs.writeFileSync(matrixPath, original.replace('`management-api/lib/authority.js:395-404`', '`management-api/lib/does-not-exist.js:395-404`'));
  r = run(root);
  expect('missing cited path → exit 1', r.status === 1 && /missing/.test(r.stderr), r.stderr.trim().split('\n')[0]);
  // 3. An elided path turns the gate red.
  fs.writeFileSync(matrixPath, original.replace('`src/actors/elevation_actor.rs:68,93,1099-1132`', '`src/.../elevation_actor.rs:68,93,1099-1132`'));
  r = run(root);
  expect('elided path → exit 1', r.status === 1 && /elided/.test(r.stderr));
  // 4. A `measured` cell citing a document turns the gate red.
  fs.writeFileSync(matrixPath, original.replace('| C1 Durable net value | `partial`', '| C1 Durable net value | `measured` — (`docs/README.md:1`)'));
  fs.mkdirSync(path.join(root, 'nostr-rust-forum/docs'), { recursive: true }); fs.writeFileSync(path.join(root, 'nostr-rust-forum/docs/README.md'), '');
  r = run(root);
  expect('measured citing a document → exit 1', r.status === 1 && /must cite code\/config\/receipt/.test(r.stderr));
  // 5. Fewer than six rows turns the gate red.
  fs.writeFileSync(matrixPath, original.replace(/^\| C6 .*\n/m, ''));
  r = run(root);
  expect('five rows → exit 1', r.status === 1 && /expected 6/.test(r.stderr));
  // 6. (auditor gap 1) a line number beyond the file's length turns the gate red.
  fs.writeFileSync(matrixPath, original.replace('`management-api/lib/authority.js:395-404`', '`management-api/lib/authority.js:99999`'));
  r = run(root);
  expect('line past EOF → exit 1', r.status === 1 && /cites line 99999/.test(r.stderr));
  // 7. (auditor gap 2) a header aliased to another substrate turns the gate red.
  fs.writeFileSync(matrixPath, original.replace('| Condition | nostr-rust-forum | agentbox | VisionClaw |', '| Condition | nostr-rust-forum | agentbox | agentbox |'));
  r = run(root);
  expect('duplicate substrate header → exit 1', r.status === 1 && /duplicate substrate column/.test(r.stderr));
  fs.writeFileSync(matrixPath, original.replace('| Condition | nostr-rust-forum | agentbox | VisionClaw |', '| Condition | nostr-rust-forum | agentbox | Loom |'));
  r = run(root);
  expect('unknown substrate header → exit 1', r.status === 1 && /unknown substrate column/.test(r.stderr));
  // 8. (auditor gap 3) a citation to a directory turns the gate red.
  fs.writeFileSync(matrixPath, original.replace('`management-api/lib/authority.js:395-404`', '`management-api/lib:1`'));
  r = run(root);
  expect('directory citation → exit 1', r.status === 1 && /not a regular file/.test(r.stderr));
  // 9. (auditor gap 4) a C7 row with fabricated citations turns the gate red.
  fs.writeFileSync(matrixPath, original.replace(/^(\| C6 .*\n)/m, '$1| C7 Invented | `measured` — (`nope/x.rs:1`) | `measured` — (`nope/y.rs:1`) | `measured` — (`nope/z.rs:1`) |\n'));
  r = run(root);
  expect('C7 row → exit 1', r.status === 1 && /row label "C7"/.test(r.stderr) && /expected 6 condition rows, found 7/.test(r.stderr));
  // 10. (auditor gap 5) `measured` citing an extensionless document turns the gate red.
  fs.mkdirSync(path.join(root, 'nostr-rust-forum'), { recursive: true }); fs.writeFileSync(path.join(root, 'nostr-rust-forum/README'), FIXTURE_LINES);
  fs.writeFileSync(matrixPath, original.replace('| C1 Durable net value | `partial`', '| C1 Durable net value | `measured` — (`README:1`)'));
  r = run(root);
  expect('measured citing extensionless doc → exit 1', r.status === 1 && /must cite code\/config\/receipt/.test(r.stderr));
} finally { fs.writeFileSync(matrixPath, original); fs.rmSync(root, { recursive: true, force: true }); }

// 6. Real matrix against the real sibling checkouts (skipped if siblings are absent).
if (fs.existsSync(path.resolve(repo, '..', 'nostr-rust-forum'))) { r = run(path.resolve(repo, '..')); expect('real matrix passes against sibling checkouts', r.status === 0, r.stdout.trim()); }
else console.log('skip - sibling checkouts absent');

if (failed) { console.log(`FAILED (${failed})`); process.exit(1); }
console.log('augmentation-citations gate suite OK');
