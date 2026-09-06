---
title: Shared memory durability, retrieval and expiry
status: in-progress
date: 2026-09-04
type: explanation
---

# Shared memory durability, retrieval and expiry

The shared-memory plane lets agents recover context across sessions. Its intended value depends on retaining a searchable representation of the value actually stored, respecting retention rules, and measuring useful recall. Agentbox's MCP server provides a concrete common path, but its failure behaviour is weaker than the blanket searchable-write guarantee in ADR-2014. The [receipt](evidence/memory-snapshot.json) identifies source hashes and local checks.

## Write acknowledgement and semantic visibility

The [MCP entry point](../../../project/agentbox/mcp/servers/ruvector-mcp.cjs) requires the Postgres module and does not silently select sql.js. The [shared factory](../../../project/agentbox/mcp/servers/lib/memory-tools.js) returns a failed result when the pool is unavailable. Embedding failure is different: it logs a warning, inserts with `embedding = NULL`, and returns `success: true`, `stored: true`, `embedded: false`.

On conflict, the update uses `COALESCE(EXCLUDED.embedding, memory_entries.embedding)`. Thus a replacement value can retain an embedding of the previous value when embedding fails. These are two distinct recovery obligations: embed previously unembedded rows and repair value/vector version mismatch. No automatic repair guarantee was established in this pass.

The [isolated probe](evidence/memory-store-probes.cjs) executes the real factory with an embedding exception and a mock database. Its [receipt](evidence/memory-store-probes.json) verifies successful unembedded-write acknowledgement and captures the replacement expression. It does not insert into any real store. The existing ten factory tests pass and explicitly expect `embedded: false` when Xinference is unavailable; the current test contract therefore confirms, rather than prevents, this degraded behaviour.

## Model and search contracts

Store embedding uses at most the first 2,000 characters of the value. The entry point defaults to `bge-small-en-v1.5` and checks returned vector length against 384. An environment override can change the model name while the column dimension stays fixed; dimension agreement alone does not prove compatible embedding geometry. Long-value recall also depends on whether the relevant information lies in the embedded prefix.

Scoped searches materialise the namespace/source subset and rank by exact vector distance, while unfiltered search retains the index-oriented query. Both return `method: hnsw-xinference`, so the response label does not identify the actual execution plan. Vector failure falls back to substring search with an explicit `degraded: true`, warning and `ilike-fallback` method. Those degradation signals should survive every consumer.

The manifest keeps SONA learning/application and attention reranking off. Historical measurements explain that decision, but were not rerun here. The recall harness defines a frozen median-of-three gate; its existence is not evidence that the current deployed corpus, model and filters pass it. Closeout needs receipts bound to those exact identities and representative namespace sizes.

## Retention and namespace boundaries

With typed metadata enabled, a positive TTL creates `expires_at`, but [metadata construction](../../../project/agentbox/mcp/servers/lib/memory-metadata.js) defaults `memory_type` to `semantic`. The sweep deletes only expired **episodic** rows. A caller supplying TTL alone therefore gets an expiry timestamp that this sweep will not delete. Retrieve/list and the inspected ordinary search queries do not filter expired entries; even episodic records remain readable until deletion occurs. TTL here is not an immediate read-denial or universal erasure guarantee.

Protected namespaces are an explicit write/delete/sweep guard, with an administrative environment override. Ordinary reads can specify a namespace or `*`; these selectors are retrieval scopes, not per-caller authorisation in this factory. Any confidentiality claim needs the surrounding caller/access boundary to be traced separately. This review follows the MCP-only rule and does not use direct database operations.

## Closeout conditions

CP-03/07/08 must decide whether embedding failure rejects a write or enters an explicit, durable repair state. Bind stored value and vector versions, test replacement failure and recovery, expose pending/unsearchable rows, and verify recall after repair. Freeze the actual model identity and preprocessing alongside dimensions; cover long values and namespace filters in the recall fixture.

CP-04/07 must specify whether TTL means scheduled cleanup or immediate visibility expiry, which memory types it applies to, and how protected namespaces and backups participate. Test TTL-only writes, expired reads, missed sweeps and recovery. Preserve namespace selection while establishing the real caller-authority boundary.

ADR-2014 is amended to partial implementation of its full searchable-write guarantee; its MCP/Postgres boundary remains implemented. ADR-2018/2019 gain scoped evidence and remaining acceptance conditions. The [estate roadmap](closeout/README.md) retains wider learning, privacy, operational recovery and consumed RuVector implementation work as open.

## Execution journal durability and reconstruction

Historical agentbox ADR-057 remains proposed, but current code implements ExecutionJournal, a canonical event vocabulary/schema, model-request citation checks and projection helpers. This replaces an unassessed adoption question with evidence of implemented primitives and an unresolved integration contract. A bounded search of management-api JavaScript did not establish production construction/hydration or model-call enforcement sites; it does not rule out external callers.

The [actual composition fixture](evidence/journal-durability-probe.cjs) connects ExecutionJournal to LocalJsonlEventsAdapter with a temporary regular file where the log directory should be. Directory creation fails. [The result](evidence/journal-durability-probe.json) shows dispatch still notifies one subscriber and resolves; the journal returns sequence zero and records the event in its in-memory idempotency map. Retrying the same event returns duplicate:true although no durable event was written. The adapter keeps its hash-chain position unchanged, but that does not preserve the journal's separate sequence and acknowledgement invariant.

The same fixture submits changed model-visible text citing sequence zero. assertModelRequestTraceable returns ok because it checks integer citations against the in-memory sequence ceiling. It does not compare message content to an immutable event payload or establish that the cited event reached storage. The fixture neither invokes a model nor claims a deployed unjournalled request occurred. It demonstrates that the helper's successful result is weaker than ADR-057's reconstructable-provenance promise.

These are composition boundaries: an events channel designed to remain available despite telemetry write failure cannot automatically serve as a mandatory durable journal. The proposal's status is preserved, with primitive implementation credited and complete journal acceptance withheld. Existing mirrors, transcripts and aggregate logs remain separate records until actual projection provenance is demonstrated.

CP-01/03/07/08/09 requires a durable append acknowledgement contract, propagated storage failure, retry after failed append, and hydration from verified persisted events. Bind model-visible content to immutable payloads or explicit redacted hash receipts, not merely valid sequence numbers. Test crash-tail recovery, concurrency, duplicate IDs with different bodies, projection failure/rebuild and strict pre-model-call enforcement through real adapters. Publish per-harness coverage and degradation; no reconstructed completion may exceed durable evidence. No production storage or model call was touched.
