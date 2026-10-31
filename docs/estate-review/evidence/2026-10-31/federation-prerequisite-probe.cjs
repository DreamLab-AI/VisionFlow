#!/usr/bin/env node
'use strict';
// Read-only production bridge and route invocation; no HTTP listener or live data.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(process.argv[2]);
const bridge = require(path.join(root, 'management-api/lib/bc20-provenance-bridge.js'));
const routes = require(path.join(root, 'management-api/routes/uri-resolver.js'));
(async () => {
  const key = 'a'.repeat(64);
  const drops = [];
  const data = `urn:agentbox:dataspace:${key}:audit-space`;
  const converted = bridge.toVisionclaw(data, { onDrop: (reason) => drops.push(reason) });
  assert.equal(converted, null);
  const bead = bridge.toVisionclaw(`urn:agentbox:bead:${key}:sha256-12-123456abcdef`);
  assert.equal(bead.visionclaw_id, `urn:visionclaw:bead:${key}:sha256-12-123456abcdef`);
  const handlers = new Map();
  await routes({ get: (url, handler) => handlers.set(url, handler) }, { manifest: {}, logger: {} });
  async function resolve(urn) {
    let status = 200, payload;
    const reply = { code: (value) => { status = value; return reply; }, send: (value) => { payload = value; },
      redirect: (url, code) => { status = code; payload = { url }; } };
    await handlers.get('/v1/uri/:urn')({ params: { urn }, query: {} }, reply);
    return { status, payload };
  }
  const unresolved = await resolve(data);
  assert.equal(unresolved.status, 404);
  const advertised = await (async () => {
    let value;
    await handlers.get('/v1/uri')({}, { send: (v) => { value = v; } });
    return value;
  })();
  assert.equal(Object.hasOwn(advertised.grammar.kinds, 'dataspace'), false);
  const result = { cases: [
    { case: 'dataspace_crossing_not_implemented', converted, drops },
    { case: 'existing_bead_crossing_control', converted: bead.visionclaw_id },
    { case: 'dataspace_resolver_not_implemented', ...unresolved },
    { case: 'resolver_contract_advertises_no_dataspace_or_410', contract: advertised.contract,
      dataspace_in_grammar: false },
  ] };
  fs.writeFileSync(process.argv[3], JSON.stringify(result, null, 2) + '\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
