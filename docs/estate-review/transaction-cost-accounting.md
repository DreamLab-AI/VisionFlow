---
title: Contextual transaction cost from capture to measurement
status: partial-implementation-assessment
date: 2026-09-05
type: explanation
---

# Contextual transaction cost from capture to measurement

The book's third commitment asks for handoffs, tokens, compression and verification accounting per workflow, with a CTC figure per DAG beside Mesh Velocity. Current code carries cost-related fields through an implemented emitter and receiver, but this review does not establish the promised per-DAG measurement or display. The producer and receiver also describe different token semantics. [Evidence receipt](evidence/transaction-cost-snapshot.json) records eleven source hashes, seventeen passing existing tests and five synthetic helper assertions.

## Producer accounting unit

Agentbox's [transcript recorder](../../../project/agentbox/config/hooks/trajectory-recorder.cjs) reads assistant usage and attaches that record's token total to each tracked Bash tool use in the record. [tokenCountOf](../../../project/agentbox/config/hooks/lib/trajectory-util.cjs) sums positive finite input, output, cache-creation and cache-read values, returning null for zero or missing totals. This is a turn-associated token count, not an independently measured cost for each Bash command or a cumulative DAG total.

For a turn issuing two tracked Bash calls, the source assigns the same turn count to both. An actual helper probe with a 150-token synthetic turn produced two 150-token emit bodies, summing to 300. The scanner assignment was inspected, not run in that probe. A naive sum would double count that turn; the result does not establish that a production dashboard currently makes that error. The emitted bodies do not carry a distinct assistant-turn usage identity for deduplicating this allocation.

The scanner counts tool uses named `Task` across the whole transcript as handoffs and stores that count on the trajectory rollup. That bounded definition does not census other delegation APIs or reconstruct DAG edges. An environment chain ID takes precedence over a trajectory fallback; inherited ID propagation across actual agents remains an acceptance obligation. Only tracked, gradeable Bash results enter these steps, so their coverage is not total model activity, compression loss, verification effort or recovery cost.

## Delivery and meaning across the boundary

The recorder persists steps/rollup and then attempts event emission, capped at 200 bodies per Stop invocation. HTTP completion resolves without checking response status; timeouts and network failures are non-fatal, and the reviewed path has no durable emission retry queue. A complete local trajectory therefore does not establish complete downstream accounting. Existing [learning capture findings](learning-evidence.md) also apply to watermark and persistence recovery.

The mapper forwards `token_count` and `handoff_id`; its success/failure appears in metadata. It does not populate the typed `verification` field. Chain identity alone can produce an event without any token count. The [route and publisher](../../../project/agentbox/management-api/routes/agent-events.js) support forwarding an explicitly supplied verification value, but that capability does not supply one for this mapper.

VisionClaw's [schema](../../../project/src/agent_events/schema.rs) describes `token_count` as cumulative model tokens to an action and verification as a DAG verdict. That differs from the recorder's turn-per-step value and absent typed verification. `has_ctc()` tests whether any of three options is present; it does not require measured cost, a complete workflow or a verified verdict. The ingest's one-shot canary latches before asynchronous observation, so an observation failure does not retry within that process. Its evidence is field presence.

## Persistence, aggregation and display

The [KPI capture task](../../../project/src/services/kpi_compute.rs) copies the three fields into SQLite. Broadcast lag skips events and insertion failures are logged; no replay appears in this capture loop. The repository casts the unsigned token count to signed SQLite storage without a checked conversion. This is an additional numeric-domain acceptance obligation, not a reproduced production overflow.

A bounded source search found no CTC aggregation/display reference in VisionClaw's TS/TSX client and no calculation in its KPI compute/handler path beyond capture. The agentbox search found the recorder, utility, publisher and tests, without establishing a CTC dashboard consumer. The [joined trace](joined-provenance-trace.md) also drops token counts from its normalised response. These results do not prove absence in every language or an external deployment; they mean the historical “reporting a CTC figure per DAG” statement lacks a verified current consumer in this assessment.

The seventeen passing tests exercise utilities and the real emit route through in-process Fastify injection into the publisher. They prove forwarding for supplied inputs. They do not run the actual Stop hook's HTTP transport, a live database, VisionClaw WebSocket, complete DAG or display. The additional helper assertions confirm turn-total mapping, absent typed verification and chain-only emission under synthetic inputs.

## Closeout requirements

CP-01/07/08 must define the accounting unit before presenting a scalar: distinct model turns, token categories, parent/child DAG edges, completion/retry identities and missing-data status. Reconcile producer and receiver semantics, deduplicate multi-tool turns, and distinguish unknown cost from measured zero. Preserve token counts separately from priced spend and independently evaluated verification/recovery burden.

Trace one multi-agent workflow with multi-tool turns, retries, partial failure and an unfinished branch through capture, persisted rollup, delivery, aggregation and actual display. Include non-Bash work and alternate delegation mechanisms according to the agreed scope. Test capture gaps, more than 200 emits, HTTP denial, restart and numeric limits, with reconciliation or explicit incompleteness. Compression and verification measurements need explicit producers and definitions rather than inferred presence of a correlation ID.

Link the resulting receipt to the [book commitment](book-roadmap-reconciliation.md), [KPI definitions](kpi-outcomes.md) and [ordered closeout](closeout/execution-sequence.md). A populated envelope or recent canary fire alone cannot close per-DAG production measurement.
