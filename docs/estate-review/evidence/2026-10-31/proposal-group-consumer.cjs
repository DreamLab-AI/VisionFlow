// Read the real CLI's saved dry-runs and call the production apply adapter.
// No CLI, Git or HTTP invocation is permitted by this probe.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { applyOntologyDecision } = require(path.join(path.resolve(process.argv[2]), 'management-api/lib/ontology-apply.js'));
const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
(async () => {
  const results = [];
  for (const name of ['grouped_amend', 'mixed_grouped_create']) {
    const proposal = input[name].proposal;
    let calls = 0;
    const refuse = async () => { calls++; throw new Error('External call forbidden by audit fixture'); };
    let message;
    try {
      await applyOntologyDecision({ outcome: 'promote', iri: proposal.iri, proposal,
        signerNpub: 'fixture', at: '2026-10-31T00:00:00Z', caseId: proposal.digest.slice(7), digest: proposal.digest },
      { runVault: refuse, runGit: refuse, fetchFn: refuse });
    } catch (e) { message = e.message; }
    assert(message && message.startsWith('ontology-apply:'));
    assert.equal(calls, 0);
    results.push({ name, kind: proposal.kind, pages: proposal.pages, error: message, externalCalls: calls });
  }
  console.log(JSON.stringify(results, null, 2));
})().catch(e => { console.error(e); process.exitCode = 1; });
