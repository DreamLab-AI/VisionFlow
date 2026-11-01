---
title: Dream-engine night recovery and source-supply audit
date: 2026-11-01
status: assessed-with-open-findings
scope: ADR-DE-007, ADR-DE-008 and selected agentbox consumer paths
---

# Dream-engine night recovery and source-supply audit

Process-crash recovery works at the two tested journal boundaries. Source-request completion is less reliable: an empty line window is marked honoured, duplicate ranges disappear from accounting, and requests that all fail produce no returned receipt. The adapter then treats any read of a path as satisfying every requested range for that path. These behaviours can turn missing context into a misleading “already read” alert.

This pass assesses ADR-DE-007 and ADR-DE-008 section by section, updates AB-23.7 and AB-23.21, and adds three DE-03 panels. It extends the [candidate and evidence review](2026-11-01-dream-candidate-and-evidence.md). Other AB-23 panels, the remaining local decisions and the wider estate audit stay open. Owner decision statuses and production code are unchanged.

## Evidence and limits

The [corpus gate](evidence/2026-11-01/dream-recovery-corpus-regeneration.log) covers 178 topics and 1,591 panels, with 24,527 verified citations, zero unresolved citations and nine documented unversioned-artefact exceptions. [Final verification](evidence/2026-11-01/dream-recovery-final-corpus-check.log) records all 40 declared revisions. These structural checks do not close the remaining semantic review.

Dream-engine remains at `815f963b1cbd6999e78144ce5475005ae1dd1d8e`; agentbox remains at `9dfc27b16e82e9786535a55e790710493dd51fdf`. [Input hashes and revision comparisons](evidence/2026-11-01/dream-recovery-source-manifest.json) bind the reviewed files to those commits and compare the changed adapter panels' inputs with their retained diagram pins. Unrelated dirty files were preserved.

[Ten focused tests pass](evidence/2026-11-01/dream-recovery-source-probes.log), using the [committed probe source](evidence/2026-11-01/dream-recovery-source-probes.rs) against an isolated clean export. Existing [24 journal tests](evidence/2026-11-01/dream-recovery-existing-tests.log) and [29 source tests](evidence/2026-11-01/dream-source-existing-tests.log) also pass. The earlier unchanged-source workspace run had 555 passing tests; this pass did not rerun unrelated suites. All new model replies are local fixed responses. No model service, forge, live request store, real night directory or daemon was used.

For each crash case, a child acquired the actual night lock and wrote the journal, then signalled readiness. The parent confirmed competing acquisition was refused, killed and reaped that child, and acquired the lock again. After ModelCalled, the same run resumed at attempt two. After Persisted, a changed run ID returned AlreadyComplete with the recorded verdict. This proves those local process-death paths; it does not test power loss, network filesystem semantics, forge deduplication or reset interruption. Passing defect assertions below means a shortcoming was reproduced, not repaired.

## Findings and acceptance work

Paths without an agentbox prefix are relative to dream-engine.

| Finding | Evidence and user consequence | Acceptance required |
|---|---|---|
| DE-R1 / medium: empty or partial coverage can settle a request | `crates/dream-core/src/source.rs:644-651,835-870,1162-1167`: asking for `a.rs:99-100` in a one-line file returns zero body bytes, displays `lines 99-1 of 1` and lists the request as honoured. Agentbox `wanted.rs:328-338` ignores line ranges when settling keys and treats a directory as met by any descendant file. This consumer behaviour is source-verified, not a whole-night experiment. | Define completion as supplied line coverage, distinguish empty and clipped ranges, and retain unmet windows. A range outside EOF must be refused with a reason. Test that reading one range does not retire another. |
| DE-R2 / medium: refusal accounting disappears | `source.rs:1101` silently drops a second range for the same path; the probe receives neither an honoured nor refused entry for it. At `1141-1143` and `1183-1184`, all-invalid or all-unreadable selections return None, losing the accumulated refusal reasons. | Give each original request a stable outcome, including directory expansion and duplicate ranges. Return diagnostic records independently of whether any source text can be supplied. |
| DE-R3 / medium: reservation and byte limits do not mean what the ADR says | `source.rs:1091` counts input positions before validation: with one reserved slot, a missing first request prevents a valid second one from using it. `clip_file` adds a marker after clipping payload; `render_section` applies the caller's redactor afterwards (`850-854`). At default limits the probe records 30,134 requested body bytes against 30,000 and 60,134 total against 60,000. With a 100-byte budget, the first requested body exceeds the whole allowance and excludes the planner's file. | Reserve successful selections, or document positional allocation. Account for markers and redaction before final clipping; define separately whether section headers are included. A supplied redactor can expand output arbitrarily, so enforce the limit after it. |
| DE-R4 / medium: tracked filenames can be unavailable to the planner | `source.rs:138-153` parses line-based `git ls-tree` output without decoding Git's quoted paths. With `core.quotepath=true`, tracked `café.rs` is missed even though `read_blob` reads it successfully. | Parse NUL-delimited tree output and preserve path bytes under an explicit encoding policy. Include non-ASCII, quotes, tabs and newlines in fixtures; this probe directly covers non-ASCII only. |
| DE-R5 / medium: “three nights” is three offers, and completion evidence is best effort | Agentbox `wanted.rs:276-289` increments on every `take_for` call and accepts no night ID. Same-night retries can spend another offer. `engine.rs:1295-1299` ignores source-receipt write errors and still settles; `wanted.rs:309-313` ignores settlement write errors. The shared store uses a fixed temporary path and unprotected read-modify-write (`wanted.rs:70-98,231-271`). | Count unique night IDs; preserve pending state when evidence cannot be recorded; serialise store updates across different nights/repos. A per-night lock does not protect this shared file. Concurrency loss is an identified source-level risk, not a reproduced live event. |
| DE-R6 / qualification: crash safety depends on the storage fault model | `crates/dream-gate/src/manifest.rs:328-335` writes a temporary file then renames it without file or directory synchronisation. `runstate.rs:466-484` resets two files through separate link/unlink operations and ignores rollback errors. Neither is a multi-file crash transaction. | State process-crash and machine/storage-failure guarantees separately. Test a kill at each reset operation and persistence under the chosen storage failure model before claiming durable at-most-once publication. No power-loss duplication or interrupted-reset loss was experimentally induced here. |
| DE-R7 / API boundary: preliminary identity and budget checks are incomplete | `finished_night` returns None for another night (`runstate.rs:411`), while `begin` later rejects it. Agentbox freezes a manifest at `engine.rs:763` before `begin` at `867`; the early check therefore does not universally preserve a mismatched directory. A first `begin(..., max_attempts=0)` still starts attempt one (`runstate.rs:537-540`). | Check directory identity before any manifest mutation and define zero-budget handling. The adapter uses the fixed default of two, so the zero-budget case is a public API limitation, not its observed configuration. |
| AB-R1 / diagram drift: question shape is guidance, not an invariant | Agentbox `inbox.rs:278-281` requires only Decision and If approved; `plain_decision` returns unstructured questions unchanged (`289-290`). A parsed four-field source-supply request is treated as a decision by `wanted.rs:194`, and `engine.rs:1669` can emit another alert on each later inactive-key report. | Describe the actual compatibility path and alert cadence. If all questions must have four fields or each request only one lifetime alert, enforce that policy and test it. No external messages were sent during this audit. |

Source screening for key material is pattern-based (`source.rs:761-790`), not a general secret detector. This pass does not certify it against all credential formats. Filename and content checks are applied before the tested source text is offered, but that does not make arbitrary repository contents safe or relevant.

## ADR-DE-007 section dispositions

| Section or decision clause | Assessment |
|---|---|
| Context: ledger commits move run identity | Consistent with `manifest::run_id` including baseline revision. A matching terminal night ignores a changed run ID; the killed-child probe confirms this. |
| Decision: terminal rule and finalising Persisted | Implemented for matching repo/date/night. Finalisation writes are best effort; failure does not permit rerun. A verdict does not establish that a draft exists. |
| Decision: different run below Persisted | Implemented as Fresh at Initialised with the previous attempt count incremented. Existing tests cover changed-run exhaustion. |
| Decision: persistence before external effects | Adapter refuses accepted-candidate publication if the persistence write fails. It does not reconcile a forge result. This ordering concerns forge/ledger publication; inbox and request-store writes already occur earlier. |
| Decision: explicit reset archives both files or neither | Unique hard-link names prevent replacement on successful reset; existing tests cover rapid resets and a failed second archive. The absolute all-or-neither wording excludes neither process death between operations nor failure of rollback itself. DE-R6 remains open. |
| Decision: run ID stable only while HEAD is stable | Correctly qualified in the record; completion belongs to the night. |
| Consequences: early check and manual recovery | Matching terminal nights skip before freezing. Wrong-night identity is checked later, so the broad skip/preservation wording needs DE-R7's qualification. Manual reset is the documented path for a lost publish. |
| Hardening: one engine per night | Local process exclusion and release after forced death are verified. Holding the lock remains a host contract; the standalone journal functions do not acquire it. |
| Hardening: absence versus damage and old schemas | Existing tests cover corrupt, unknown-schema and historical journal fixtures. `begin` rejects WrongNight; `finished_night` alone does not. |
| Hardening: attempt budget belongs to the night | Existing changed-ID and abandonment tests pass. Zero-budget first admission is an API edge case. |
| Known limit: lost publish and missing outcome record | Accurately describes the unreconciled process-crash window. The probe verifies terminal classification, not a real forge interaction. |
| Known limit: duplicate impossible, manual reset, future night key | “Impossible by construction” needs the retained-journal/locking/storage assumptions. The night key is a plausible reconciliation key, not an implemented exactly-once guarantee. |

## ADR-DE-008 section dispositions

| Section or decision clause | Assessment |
|---|---|
| Context: repeated file requests reach a person | Historical motivation is explicit. Current operator workload was not measured. |
| Decision: GatherInput constructor and non-exhaustive shape | Present, including `with_requested`; the integration probes use that public API. |
| Decision: requested first, half-slot and half-byte reservation | Requested paths precede planner/evidence selections. Positional reservation and marker accounting contradict successful-file and strict-byte interpretations; DE-R3 is reproduced. With one file slot the at-least-one rule also leaves no slot for the planner. |
| Decision: tracked directory expansion, shallow first | Implemented by `expand_directory`; it expands to file requests and discards the original directory/range identity. Consumer settlement treats any descendant as completion. |
| Decision: equal validation, no key material | Path, denylist, binary, size and content checks exist; quoted-path parsing loses valid names. Content screening is a finite heuristic, not complete credential recognition. |
| Decision: planner told what is already being read | Prompt lists provisionally accepted requests before actual blob reading/rendering. Binary content, secret screening, budget or an empty window can still prevent meaningful delivery. |
| Decision: honoured/refused note and SourceRecord | Present when at least one file is marked inlined; incomplete for duplicate ranges and absent for all-refused reads. Empty windows count as honoured. |
| Decision: parsing, expiry and alerts owned by consumer | Separation is implemented. The actual consumer counts offers and matches paths, with the qualifications in DE-R1 and DE-R5. |
| Consequences: planner keeps half, human sees only loop alerts, constructor migration | Half-byte protection is not strict and one-slot configurations cannot reserve half a slot. Human-action classification has compatibility exceptions. Constructor-based use is current; no runtime deployment claim follows. |

ADR-2143 was read in full for consumer context. Decision 1 and the parser portion of Decision 2 were source-assessed here. Full prompt/governance rendering, migration records and deployment evidence remain outside this pass; the 17 October activation and later re-verification are historical records. Its section status is therefore partial, not complete.

## Diagram dispositions and useful comparisons

AB-23.7 now orders manifest freezing before `begin`, distinguishes same-run and changed-run retries, and qualifies the storage guarantee. AB-23.21 replaces strict-budget, exact-night and universal-question-shape claims with the actual selection, offer and settlement rules. Three new DE-03 panels expose these boundaries. The nested dream-engine topic NIT-01 was read and has an owner finding: its business narrative overstates reboot safety, and NIT-01.3's “any failure” restoration note ignores rollback errors and interruption. That corpus was not edited or marked complete in passing.

[Aider's repository map](https://aider.chat/docs/repomap.html) supplies definitions and signatures to help a coding model navigate the repository, while dream-engine here supplies selected baseline blobs and remembered requests. That suggests a useful comparison: evaluate dependency-aware context selection against the current planner at equal input budgets. Measure relevant-line coverage, valid-patch rate, repeated requests and operator interruptions; neither documentation nor these tests establishes which approach performs better.

[Temporal's activity guidance](https://docs.temporal.io/activity-definition#idempotency) separates retryable execution from idempotent external effects. Dream-engine is a smaller host-driven nightly decision component with a deliberate no-retry boundary after Persisted. For maintainers who need every accepted draft delivered, add a durable outcome record and reconciliation keyed to the night before making delivery guarantees. This is an architectural inference from the compared contracts, not a recommendation to replace the engine or a measured market ranking.

[Temporary-file persistence documentation](https://docs.rs/tempfile/latest/tempfile/struct.NamedTempFile.html#method.persist) likewise distinguishes atomic replacement from synchronised storage. It is a useful design reference for DE-R6, not a claim that dream-engine uses that API.

[Reproduction notes](evidence/2026-11-01/dream-recovery-reproduction.md), probe logs, source hashes and [retained artefact hashes](evidence/2026-11-01/dream-recovery-retention.json) accompany this report. Remaining dream-engine work includes DE-001 host parity, DE-002 provenance/dependencies, DE-005 seam failures, DE-006 forge behaviour, the other twenty AB-23 panels, nested diagrams and the unresolved acceptance work above.
