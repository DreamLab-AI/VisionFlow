'use strict';
// Run with an exported agentbox root as argv[2]. Only temporary disk and Fastify injection.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(process.argv[2]);
const api = path.join(root, 'management-api');
const Fastify = require(path.join(api, 'node_modules/fastify'));
const route = require(path.join(api, 'routes/exec-record'));
const { ExecutionJournal } = require(path.join(api, 'lib/execution-journal'));
const { LocalJsonlEventsAdapter } = require(path.join(api, 'adapters/events/local-jsonl'));
const chain = require(path.join(api, 'lib/audit-chain'));
const silent = { info() {}, warn() {}, error() {}, debug() {} };
(async () => {
  const dir = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'audit-journal-'));
  const adapter = new LocalJsonlEventsAdapter({ eventsDir: dir });
  let journal = new ExecutionJournal({ eventsAdapter: adapter });
  const app = Fastify({ logger: false });
  await app.register(route, { logger: silent, getPlane: () => ({ ready: true, journal }) });
  await app.ready();
  const post = async payload => {
    const r = await app.inject({ method: 'POST', url: '/v1/exec/record', payload });
    assert.equal(r.statusCode, 201); return r.json();
  };
  const base = { session: 'same-attempt', harness: 'dream-engine', type: 'tool.called', key: 'step-1-called' };
  const first = await post({ ...base, payload: { operation: 'A' } });
  const second = await post({ ...base, payload: { operation: 'B' } });
  assert.equal(second.duplicate, true); assert.equal(first.event_id, second.event_id);
  const rows = () => fs.readdirSync(dir).filter(f => f.endsWith('.jsonl')).flatMap(f => fs.readFileSync(path.join(dir,f),'utf8').trim().split('\n').map(JSON.parse));
  assert.equal(rows().length, 1); assert.equal(rows()[0].payload.payload.operation, 'A');
  console.log('PASS: same key with changed payload returns prior success; B is absent');
  journal = new ExecutionJournal({ eventsAdapter: adapter });
  const afterRestart = await post({ ...base, payload: { operation: 'C' } });
  assert.equal(afterRestart.seq, 0); assert.equal(afterRestart.duplicate, false);
  assert.equal(afterRestart.event_id, first.event_id); assert.equal(rows().length, 2);
  console.log('PASS: fresh unhydrated journal reuses seq 0 and event ID, appends C');
  // Actual adapter; direct public append calls overlap before local sequence state commits.
  const event = {session_urn: 'urn:agentbox:meta:session-race', harness:'audit',type:'tool.called'};
  const concurrent = await Promise.all([journal.append({...event,event_id:'event-a'}), journal.append({...event,event_id:'event-b'})]);
  assert.deepEqual(concurrent.map(r=>r.envelope.seq),[0,0]);
  console.log('PASS: two concurrent append calls in one process receive the same session sequence');
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.jsonl')).map(f=>path.join(dir,f));
  assert.equal(chain.verifyFiles(files).ok,true);
  console.log('PASS: the real on-disk hash chain still verifies across all four records');
  await app.close();
  console.log(JSON.stringify({eventsDir:dir,records:rows().length,concurrentSequences:concurrent.map(r=>r.envelope.seq)}));
})().catch(e=>{console.error(e);process.exitCode=1;});
