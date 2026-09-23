// estate-health CI state: judgement call 7 (the state is HEAD's). Drives the
// knowledgeGraph false red of 2026-09-23 through the collector's own functions:
// a retired workflow's last run failed on an older commit, HEAD moved on with no
// runs, and the repository must read "none", not "red". A red run ON HEAD must
// still read red, so the fix cannot hide a real failure.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { ciStateAtHead, ciStateFrom, latestRunPerWorkflow } from '../../scripts/estate-health.mjs';

const OLD = '721d8074c0000000000000000000000000000000';
const HEAD = '4ed9ac1590000000000000000000000000000000';
const run = (id, sha, conclusion, created) => ({
  id, workflow_id: 320196274, name: 'Build and verify', event: 'push', status: 'completed',
  conclusion, head_sha: sha, created_at: created, run_number: id,
});

function stateOf(runs, headSha) {
  const latest = [...latestRunPerWorkflow(runs, null).values()];
  const folded = ciStateFrom(latest.map((r) => ({ status: r.status, conclusion: r.conclusion })));
  return ciStateAtHead(folded, latest.map((r) => r.head_sha), headSha);
}

// 1. the false red: latest run failed on an older commit, HEAD has none
assert.deepEqual(stateOf([run(2, OLD, 'failure', '2026-09-23T07:42:48Z'), run(1, 'aaa', 'success', '2026-09-07T13:21:47Z')], HEAD),
  { state: 'none', stale: true }, 'a verdict on an older commit is not HEAD\'s state');

// 2. a real red on HEAD stays red
assert.deepEqual(stateOf([run(3, HEAD, 'failure', '2026-09-23T16:00:00Z')], HEAD), { state: 'red', stale: false });

// 3. one workflow on HEAD is enough: the fold stands (red from the other workflow counts)
const other = { ...run(4, OLD, 'failure', '2026-09-23T08:00:00Z'), workflow_id: 1, name: 'Other' };
assert.deepEqual(stateOf([run(5, HEAD, 'success', '2026-09-23T16:00:00Z'), other], HEAD), { state: 'red', stale: false });

// 4. no runs at all, or HEAD unknown: unchanged
assert.deepEqual(ciStateAtHead('none', [], HEAD), { state: 'none', stale: false });
assert.deepEqual(ciStateAtHead('red', [OLD], null), { state: 'red', stale: false });

// 5. importing the collector must not run its CLI; running it bare still prints usage and exits 2
const bare = spawnSync(process.execPath, ['scripts/estate-health.mjs'], { encoding: 'utf8' });
assert.equal(bare.status, 2, bare.stderr);
assert.match(bare.stderr, /usage: estate-health\.mjs collect/);

console.log('ESTATE-HEALTH-CI-TEST-OK');
