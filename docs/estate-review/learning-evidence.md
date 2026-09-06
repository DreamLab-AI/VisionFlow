---
title: Learning evidence, privacy and recovery
status: in-progress
date: 2026-09-04
type: explanation
---

# Learning evidence, privacy and recovery

The learning pipeline intends to replace guessed success with observed outcomes and conservative promotion. Its transcript-driven producer and Wilson sample floor are implemented safeguards. They still depend on what the outcome measures, whether captured commands are safe to retain, and whether failed persistence can be recovered. The [receipt](evidence/learning-snapshot.json) identifies the inspected source and 27 passing helper tests.

## Outcome meaning

[gradeResult](../../../project/agentbox/config/hooks/lib/trajectory-util.cjs) skips interrupted or indeterminate calls. Explicit `is_error: false` produces quality 1.0, or 0.85 with stderr noise; an explicit error produces 0.0. This avoids inventing success for absent signals. It measures tool execution status, however, not whether a command achieved the user's intended outcome. A command can exit successfully while producing the wrong artefact or printing failure text, as the separate dream-evaluator probes demonstrate.

[Aggregation](../../../project/agentbox/mcp/servers/lib/aggregate-effectiveness.js) groups action patterns, uses recency-weighted quality >= 0.5 as success input, computes a Wilson lower bound, and gates eligibility on raw count. The floor prevents one observation becoming an eligible pattern. It does not make repeated correlated calls independent, establish task value, or correct a systematically misleading outcome signal. Promotion needs both sound statistics and representative evidence.

## Redaction guarantee is narrower than the ADR

The utility returns null for non-string input or a thrown redaction error, and the recorder skips those commands. For other strings it applies regular expressions and truncates to 4,000 characters. Successful execution of this routine is not a proof of safe redaction.

The [synthetic probe receipt](evidence/trajectory-redaction-probes.json) uses only invented values. A quoted, space-containing `--password` loses its first token but retains the remaining words. A short JSON `"password":"..."` value remains unchanged. The [recorder](../../../project/agentbox/config/hooks/trajectory-recorder.cjs) persists the returned command in the step result, so those forms could reach durable trajectory storage. No real secret or transcript was read, and no actual disclosure is asserted.

Closeout needs a defined command-retention policy, structured handling or conservative rejection of unsupported secret-bearing forms, and a representative privacy test corpus. Avoid relying on increasingly broad regular expressions as proof that arbitrary shell text is safe. Raw stderr is not persisted by this step path; a redacted failure hint is used for in-memory classification.

## Persistence and acknowledgement

The recorder reads a per-session line watermark, scans new results, and assigns the new line count before persistence. If the Postgres module cannot be loaded, it writes that advanced watermark and returns successfully, skipping those lines on a later invocation. This differs from a connection/query exception: the catch path logs the failure without writing the advanced stash, permitting a later retry.

Step inserts are individually idempotent by tool-use-derived ID, but the sequence is not wrapped in one transaction with the rollup and local stash. A partial database success followed by failure needs reconciliation of durable steps, counters and the completion summary. The subsequent best-effort event emit is outside the persistence try/catch, so an event receipt must not be assumed to prove the entire rollup committed. These are source-level recovery obligations, not reproduced production failures.

## Closeout

CP-04/07/08 requires explicit privacy coverage, outcome semantics beyond execution success, representative promotion evidence, and restart/partial-write tests. Exercise absent database module, connection failure, failure after one step insert, repeated Stop events and interrupted stash writes. Compare persisted steps, rollups and emitted events by common IDs. Preserve the raw-count floor and indeterminate-outcome rejection while documenting their limits.

ADR-2015 is amended to partial implementation of its full privacy guarantee; the narrower skip-on-error behaviour remains implemented. ADR-2016 retains its scoped aggregation decision and gains evidence-quality acceptance conditions. The [roadmap](closeout/README.md) keeps full learning-loop and operational validation open.

## Producer ordering is advisory

[ADR-2017](../../../project/agentbox/docs/adr/ADR-2017-consumer-behind-producer-w066.md) prohibits enabling retrieval/routing consumers ahead of trajectory recording. The validator emits W066 as an advisory warning; its success path checks errors only. The [actual-validator fixtures](evidence/learning-order-probe.json) exercise five minimal manifests: all exit zero, with W066 only for the three consumer-before-producer combinations. The [reproducer](evidence/learning-order-probe.cjs) does not modify the real manifest.

The gates module reads each environment variable independently. The recorder checks both master and recording flags, while the hybrid factory's effectiveness bonus checks the retrieval flag. An injected pool fixture supplies an invented existing aggregate and raises score 0.5 to 0.58 with recording off, whether the master flag is off or on. No SQL executes: the pool and embedding transport are stubs. This establishes the helper's behaviour; outer route/registration gates and deployed process environments still need an admission trace. Routing likewise checks its own flag in the inspected helper, but the routing journey was not executed.

Stopping a producer does not erase its prior corpus. The current warning's claim that consumers necessarily have no data is therefore too broad. The intended system needs to distinguish capture availability, aggregate freshness, sample quality and permitted reuse. CP-01/07/08 requires a policy choice between active capture and a qualified retained corpus, validation/runtime enforcement, and a restart/override matrix. ADR-2017 is partial against its prohibition; no live activation status is re-certified.
