---
title: Failure telemetry coverage and consumer meaning
status: partial-census-open
date: 2026-09-05
type: explanation
---

# Failure telemetry coverage and consumer meaning

Agentbox has a shared fourteen-mode taxonomy, an `unmapped` fallback and classification at selected recorder, publisher and route boundaries. This supports structured failure reporting. It does not establish the book's fifth commitment that every agent failure through the pipeline and QE fleet is tagged, retained and available for analysis. [Evidence](evidence/failure-telemetry-snapshot.json) includes twelve source hashes, sixteen passing existing tests and six additional in-process route assertions. No production failure, live database, WebSocket or rendered dashboard was exercised.

## Coverage by failure source

| Source or boundary | Current evidence | Remaining acceptance |
|---|---|---|
| Gradeable Bash transcript results | Recorder classifies failures using a redacted, capped stderr hint; persists the tag in step result | Include skipped/ungradeable results in capture diagnostics; define non-Bash coverage |
| Transcript absent, crash before Stop, redaction rejection or incomplete tool result | These paths cannot be assumed to yield a graded step; earlier capture/recovery findings remain | Independent expected-work/failure census, restart and missing-telemetry records |
| Explicit publisher failures | Outcome, failure context or top-level mode triggers classification | Enumerate all callers and alternative failure representations; establish who may assert a mode |
| Four explicit emit/batch auth or identity rejections | Handler returns a structured tag; existing tests cover selected branches with mocked auth | Durable rejection audit and real signed negative requests; a response tag is not an emitted event |
| Emit schema validation | Actual malformed-payload injection returned 400 without a failure tag and emitted no event | Classify or explicitly exclude transport/schema failures; make exclusion visible in the denominator |
| CTC recorder delivery failure | Previous review establishes cap, unchecked HTTP status and non-fatal delivery errors | Recover or expose missing failure records without recursively relying on the failed channel |
| VisionClaw ingest/capture/storage failures | Existing ingest/capture code logs or skips failures; the reviewed SQLite projection has no failure-mode column | Define capture health and durable failure projection separately from traffic volume |
| QE fleet and independent runtimes | A current producer-by-producer census is not established in this pass | Trace fleet adapters, process exits, timeouts, cancellation and queue failures to a durable consumer |

The [classifier](../../../project/agentbox/management-api/lib/failure-taxonomy.js) accepts explicit canonical IDs, maps named reasons and uses two stderr heuristics before falling back to `unmapped`. `signal` and `action` are reserved rather than classification inputs today. Permission-denied text maps to a role-specification failure even though that text alone does not distinguish a policy violation from an environmental permission problem. Keep heuristic attribution separate from independently adjudicated cause. This pass inspects the local implementation; it does not validate the external research taxonomy or claim classification accuracy on real failures.

## The same event can carry conflicting tags

The [trajectory mapper](../../../project/agentbox/config/hooks/lib/trajectory-util.cjs) places its classified mode in `metadata.failure_mode`. The [publisher](../../../project/agentbox/management-api/utils/agent-event-publisher.js) detects `metadata.outcome === 'failure'`, but its classification context reads a top-level mode or failure object, not that metadata mode.

An actual mapper-to-route-to-publisher injection with `FM-1.2` therefore produced `metadata.failure_mode = FM-1.2` and top-level `failure_mode = unmapped`. Both reach the canonical notification. The specific classification is retained in metadata, but consumers choosing different locations disagree. This is a reproduced in-process boundary mismatch, not a demonstrated production miscount.

The sixteen existing taxonomy/route tests passed. They establish classifier behaviour and selected handler branches; auth is mocked in the route branch suite. The six additional assertions use the real route with auth explicitly off, confirm the conflicting tags, and show a malformed request produces neither a tag nor a publisher event. None proves estate-wide failure coverage.

## Receiver and display are separate paths

VisionClaw's [AgentActionEnvelope](../../../project/src/agent_events/schema.rs) declares no top-level `failure_mode` field. Serde's normal unknown-field handling does not promote it into the envelope; the generic metadata survives and becomes the binary payload. Thus a metadata mode can survive this projection while a top-level-only mode has no declared slot. The [KPI capture](../../../project/src/services/kpi_compute.rs) stores selected identity/cost/verification fields, not a failure taxonomy history.

The [swarm panel](../../../project/client/src/features/bots/components/SwarmObservabilityPanel.tsx) obtains failure counts from `mastFailureTags` in polled bot data. Its [extractor](../../../project/client/src/features/bots/swarmObservability.ts) merges top-level, metrics-level and per-agent bags, accepting positive numeric/coercible counts and arbitrary tag names. There is no event-to-count aggregation in this helper. If the same aggregate is supplied in multiple bags, it is summed repeatedly; producer ownership must prevent overlap. Empty or missing bags hide the section, which does not establish zero failures.

A bounded search of agentbox management-api, config, mcp, lib and scripts found no `mastFailureTags` producer reference. This does not rule out external metrics producers; it leaves their binding to this consumer unverified. The panel's canary fires on live agent data independently of whether a failure-count bag exists, so it cannot certify this telemetry path.

## Closeout requirements

CP-01/06/07/08 needs a failure-source inventory with an expected-event denominator and explicit exclusions. Distinguish task failure, policy denial, transport/storage failure, cancellation and absent observation. Retain `unmapped` without presenting it as a diagnosed cause, and record taxonomy/version, evidence source and classifier confidence or method.

Choose one authoritative wire field and test source-to-receiver-to-durable-record consistency. Derive metrics from deduplicated event identities with declared windows and disjoint aggregation ownership. Show no failures, missing telemetry and unavailable aggregation as different states. Reconcile route rejections and failed delivery through a durable recovery path.

For complete-system acceptance, run a workflow that includes a classified failure, an ambiguous failure, malformed input, auth denial, process loss and capture outage; account for every expected result after restart. Add the QE fleet and every named producer before closing the full census. Preserve the [book commitment](book-roadmap-reconciliation.md) and [trajectory capture obligations](learning-evidence.md), with remaining work routed through the [closeout sequence](closeout/execution-sequence.md).
