---
title: KPI definitions, lineage and outcome evidence
status: source-reviewed
date: 2026-09-04
type: explanation
---

# KPI definitions, lineage and outcome evidence

Historical ADR-130 D5 has an implemented path: an agent-event tap, enrichment-decision reads, SQLite snapshots with lineage, summary/lineage HTTP routes and a mounted client KPI panel. The [source receipt](evidence/kpi-outcome-snapshot.json) traces those components. Three metrics are now computed; Mesh Velocity alone remains awaiting_data_source. This is partial delivery of the historical programme, not zero implementation and not a four-metric acceptance result.

**HITL Precision is no longer awaiting a data source (2026-09-14, VisionClaw ADR-2110, PRD-augmentation-conditions FR5.3).** Its source is decided enrichment cases joined to the proposal that requested them, one row per case at its terminal decision, with the agent-event trajectory rows that name the case supplying the intent comparison. An escalation counts as *warranted* when the human's outcome family differs from the requested action's, when the outcome is `amend` or `delegate`, or when a trajectory for that case recorded `intent_match == false`. The metric reports its denominator alongside the value, and reports **no value at all** — not zero — when no human-decided case falls in the window. Cases resolved by the reserved non-DID actor `system:whelk-gate` (the EL++ consistency gate) are excluded from both terms and from the Trust Variance human-outcome series, so a reasoner rejection can no longer read as a human override. The computation is unit-tested and persists a snapshot with lineage on the same terms as the other live metrics; it has **not** been observed against live traffic, and no deployment carries it.

## What the numbers measure

Augmentation Ratio divides observed agent-event envelopes by the number of recorded enrichment decisions in the window. The denominator is decided records, not necessarily all opened escalations; the numerator is activity volume, not independently verified completed work. Zero denominator produces value zero with confidence zero. The confidence function is a linear sample-count score capped at 30 observations. It does not estimate statistical uncertainty, independence, outcome quality or capture completeness.

Trust Variance computes normalised Gini-Simpson dispersion across observed outcome categories. It measures category mixture, not variance in an independently measured trust score or correctness of human judgement. Normalisation depends on the categories present in the sample. Interpretation requires an explicit product decision about what high or low dispersion means; neither is intrinsically better.

The event tap records received envelopes and logs storage failures or broadcast lag. It resumes on subsequent frames but does not recover the skipped events in that loop. No measured completeness bound supports the source comment that undercount is only slight. The ratio's confidence does not encode lost-capture evidence. Metric definitions and observation quality therefore need separate fields.

## Read-time snapshots and provenance limits

GET /api/kpi/summary recomputes and persists both metrics. Each snapshot and its own lineage are transactional, but the two metric insertions occur sequentially; a later failure does not roll back the earlier snapshot in this service. The event count and decision list are also separate reads, not one demonstrated cross-store snapshot boundary.

Augmentation lineage stores two window-count descriptors. Trust Variance retains category counts and a capped set of contributing decision activity URNs. These are useful provenance records, but they do not establish complete event-level replay for every metric/window. Optional Oxigraph lineage is not required to credit the implemented SQLite path; the remaining question is whether its evidence is sufficient for the intended audit and comparison.

After both writes, the service attempts to fire the KPI canary without requiring a nonzero source count. A successful summary request over empty inputs can therefore record that computation/persistence occurred. This does not prove that the underlying work loop carried traffic. The distinction connects directly to the [liveness evidence contract](liveness-observation.md).

The client polls every 60 seconds and retains prior tiles when refresh fails. Its hook returns tiles and loading state without exposing the refresh error or a last-success field. Source wiring establishes a dashboard path, but not that stale values are visibly distinguished in the rendered experience. No browser or failed-refresh case ran here.

## Closeout requirements

CP-01/03/05/07/08/09 requires approved operational definitions, capture-completeness indicators, and an explicit distinction between sample sufficiency and statistical confidence. Define the expected relationship between agent activity, opened cases, decisions and independently validated outcomes before interpreting the ratio as augmentation.

Bind window boundaries, source revisions, capture health and formula version to an auditable metric run. Test no data, missing denominator, delayed decisions, duplicate/replayed events, tap loss, storage failure between metric writes and repeat reads. Decide whether reads should create history, and define retention and partial-run recovery. Supply complete or explicitly bounded lineage and visible stale/error states. Add the two deferred metrics only when their source/outcome contracts are evidenced; do not fill their tiles with proxy numbers merely to finish the dashboard.
