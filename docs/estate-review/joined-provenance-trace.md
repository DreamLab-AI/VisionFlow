---
title: Joined provenance trace and acceptance boundaries
status: source-reviewed-open
date: 2026-09-05
type: explanation
---

# Joined provenance trace and acceptance boundaries

VisionClaw implements `GET /api/trace` as a read-time projection over agent-event and enrichment-decision repositories. It groups records sharing an identity and reports absent pod marks. This fulfils a useful part of the book's eleventh commitment, but does not establish task-level causality, complete capture or resource-scoped access. [Source receipt](evidence/joined-trace-snapshot.json) records six hashes; this assessment did not run a server, HTTP request, database fixture or the existing Rust tests.

## What the endpoint joins

The [handler](../../../project/src/handlers/trace_handler.rs) constructs the [service](../../../project/src/services/provenance_trace.rs) using the two SQLite repositories. `query` reads trajectories and then decisions. Its route always passes an empty pod-mark list and `pod_source_available=false`. The pure builder accepts supplied pod marks, but that capability is not a configured pod fetch in this HTTP path; enabling a pod's git feature alone does not connect it here.

The builder groups every non-null `agent_did` string by exact equality, emitting a join when at least two distinct source kinds share it. It does not require matching handoff, activity, proposal, case or resource identifiers. Different tasks by the same identity anywhere in the selected window can satisfy the predicate. At this boundary the string is not independently verified as a signed identity; producer validation must be established separately. Anonymous records remain visible but cannot join.

Normalisation retains a source reference, action/outcome, timestamp and selected detail. It does not expose the full trajectory's source/target URNs or token count, nor the decision's proposal URN as separate fields. This is an identity-based timeline; reconstructing a complete action requires a stronger correlation contract and access to the original records.

## Window, completeness and source state

Both [trajectory SQL](../../../project/src/adapters/sqlite_kpi_repository.rs) and [decision SQL](../../../project/src/adapters/sqlite_enrichment_repository.rs) filter only on timestamp greater than or equal to the cutoff. There is no upper time bound, pagination or row limit in these methods. The service clamps a negative window to zero, permits arbitrarily large positive windows and applies the optional identity filter after loading both result sets. Thus the documented `[now-window, now]` is not an enforced upper-bound contract; future-dated rows are eligible. No shared snapshot spans the two sequential reads.

`sources_present` identifies the queried stores even if they returned zero rows. `distinct_source_kinds` separately counts kinds contributing records. Neither field proves an upstream producer is healthy or capture is complete. A read failure returns HTTP 500 rather than an explicitly partial trace. Pod absence is explicit, but there is no per-source capture watermark or lag/failure status in this response.

## Authority and canary meaning

The route is mounted under the API's [RBAC gate](../../../project/src/middleware/rbac_gate.rs). Safe reads require `ReadOnly` by default; `RBAC_PUBLIC_READS=1` or `true` permits anonymous safe reads, and acknowledged report mode can bypass denials. The trace handler accepts an optional arbitrary identity filter and contains no caller-to-identity or caller-to-resource restriction. Authentication at the gate therefore does not demonstrate permission to read every returned provenance detail. These are source-level behaviours; the deployed environment and actual data exposure were not tested.

A successful read calls the liveness observer when `max_join_span >= 2`. That proves the builder found an identity spanning source kinds in the loaded rows, not that their tasks correlate or that a new action completed. Re-reading the same eligible records can invoke observation again. Observation failure is logged and the trace still returns successfully. Apply the [liveness evidence distinctions](liveness-observation.md) before using this fire to promote a milestone.

## Closeout contract

CP-01/04/08 should agree whether this endpoint promises an identity timeline or a causally linked action trace, then align its name, book acceptance and canary predicate with that promise. Preserve the useful timeline while making the following acceptance evidence explicit:

- Bind source identity and authority to signed producer records, and test cross-user/resource reads in enforced, public-read and report configurations.
- Correlate one actual task across the required sources; unrelated tasks sharing a DID must not satisfy a task-completion predicate.
- Define upper/lower time bounds, pagination, input limits and consistent-read expectations; exercise future timestamps, empty sources and concurrent writes.
- Report capture watermarks, missing producers and partial failures separately from queried/empty stores. Connect and authorise pod retrieval before claiming a three-source HTTP trace.
- Follow the same task from original records to returned fields and its acceptance receipt; repeated reads must not be mistaken for independent completed work.

These obligations extend [ADR-2016](../../../project/docs/adr/ADR-2016-provenance-append-only.md)'s consumer contract without changing its historical append-only decision. The [book map](book-roadmap-reconciliation.md) and [execution sequence](closeout/execution-sequence.md) retain complete-system acceptance as open.
