'use strict';
// Regression suite for scripts/diagram-register-gen.cjs.
//
// The VisionClaw register (docs/diagrams/REGISTER-visionclaw.md) was committed
// for months with no generator in the repo. The script that now produces it
// must reproduce a known historical register byte for byte from that
// revision's topic files, keep the current one in sync, and its --check mode
// must go red when the committed file drifts.
//
// Needs full history (8475f10 is the reference revision), so it is not part of
// tests/gates/run-all.sh: the hosted gate-suites job checks out with depth 1.
//
//   node tests/gates/diagram-register.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const ROOT = path.resolve(__dirname, '../..');
const SCRIPT = path.join(ROOT, 'scripts/diagram-register-gen.cjs');
const REF = '8475f10';
const REGISTER = 'docs/diagrams/REGISTER-visionclaw.md';

const node = (args, opts = {}) => spawnSync(process.execPath, [SCRIPT, ...args], { encoding: 'utf8', maxBuffer: 1 << 28, ...opts });
const git = (...args) => spawnSync('git', ['-C', ROOT, ...args], { encoding: 'utf8', maxBuffer: 1 << 28 });

// 1. Byte equality with the register committed at the reference revision.
assert.equal(git('cat-file', '-e', `${REF}^{commit}`).status, 0, `reference revision ${REF} is not in this clone (fetch full history)`);
const expected = git('show', `${REF}:${REGISTER}`);
assert.equal(expected.status, 0, expected.stderr);
const regenerated = node(['--rev', REF, '--stdout']);
assert.equal(regenerated.status, 0, regenerated.stderr);
assert.ok(expected.stdout.length > 1000, 'reference register unexpectedly small');
assert.ok(regenerated.stdout === expected.stdout, `register regenerated from ${REF} differs from the one committed at ${REF}`);
console.log(`ok - ${REF} register reproduced byte for byte (${Buffer.byteLength(regenerated.stdout)} bytes)`);

// 2. The committed register on this tree is in sync.
const check = node(['--check']);
assert.equal(check.status, 0, check.stdout + check.stderr);
console.log('ok - --check passes on the working tree');

// 3. --check fails on drift, and the write mode repairs it (on a scratch copy).
const base = fs.mkdtempSync(path.join(os.tmpdir(), 'visionflow-diagram-register-'));
try {
  const area = path.join(base, 'docs/diagrams/visionclaw');
  fs.mkdirSync(area, { recursive: true });
  for (const f of fs.readdirSync(path.join(ROOT, 'docs/diagrams/visionclaw'))) {
    const src = path.join(ROOT, 'docs/diagrams/visionclaw', f);
    if (fs.statSync(src).isFile()) fs.copyFileSync(src, path.join(area, f));
  }
  const target = path.join(base, REGISTER);
  fs.copyFileSync(path.join(ROOT, REGISTER), target);
  assert.equal(node(['--root', base, '--check']).status, 0, 'scratch copy starts in sync');
  fs.appendFileSync(target, '- stale line\n');
  const stale = node(['--root', base, '--check']);
  assert.equal(stale.status, 1, '--check must fail on a drifted register');
  assert.match(stale.stderr, /stale/);
  assert.equal(node(['--root', base]).status, 0);
  assert.equal(node(['--root', base, '--check']).status, 0, 'write mode must restore sync');
  console.log('ok - --check goes red on drift and write mode repairs it');
} finally {
  fs.rmSync(base, { recursive: true, force: true });
}

// 4. The skill-format registers are refused, never overwritten.
for (const area of ['diagram-ir', 'prose-sanitiser']) {
  const refused = node(['--area', area, '--check']);
  assert.equal(refused.status, 2, `${area} must be refused`);
}
console.log('ok - skill-format areas refused');
console.log('DIAGRAM-REGISTER-OK');
