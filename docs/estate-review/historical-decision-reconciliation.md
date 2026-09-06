---
title: Historical decisions and surviving obligations
status: in-progress
date: 2026-09-04
type: explanation
---

# Historical decisions and surviving obligations

An operative ADR can distil a narrow choice from a much larger historical programme. Its implementation status applies to that choice, not every commitment in its predecessors. Estate closeout therefore requires section-level disposition: carried into an operative decision, deliberately retired, deferred with an owner and trigger, or still awaiting assessment. A reference alone establishes none of those outcomes.

## VisionClaw routing coverage

The [generated historical map](closeout/visionclaw-history.md) covers all 137 inventory-classified VisionClaw historical candidates. Forty-three have an exact-number mention in an operative lineage field; 94 do not. The [machine-readable result](evidence/visionclaw-history-map.json) records historical source hashes and exact lineage text. The method does not expand numeric ranges or resolve aliases and cross-repository number reuse. These counts measure navigation coverage, not the number of superseded decisions.

Frozen records remain unchanged. The companion map supplies current routes while preserving their original context and claims. No missing mention is treated as permission to delete an implementation or retire a requirement. Historical links can themselves be stale after archival; linking directly to each surviving file avoids relying on their internal relative paths.

## Compound predecessor: ADR-130

[ADR-130](../../../project/docs/archive/adr/ADR-130-gap-close-visionclaw-decisions.md) records six decisions. Treating it as wholly closed by the ACSP successor would discard independent obligations.

| Historical decision | Current assessment route | Remaining disposition work |
|---|---|---|
| D1: Godot mixed reality, WebXR deprecation | [XR configuration and controls](rendered-state.md#xr-control-coverage-and-hierarchy-semantics), ADR-2032 | Separate deleted legacy defect loci from gaze, mixed-reality and headset acceptance; historical deletion does not prove the replacement journey |
| D2: Replace crashbug broker transport, retain domain kernel | [ACSP consumption](forum-decisions.md#visionclaw-acsp-consumption-and-recovery), ADR-2006 | Domain-kernel integration, durable state and response recovery remain open despite transport replacement |
| D3: Central live-traffic LivenessHarness | [Observer semantics](liveness-observation.md) | Durable observer exists; evidence predicates, producer revision and standing-health acceptance remain open |
| D4: Minimal avatars and gaze-primary copresence | [Wire and rendered state](rendered-state.md), ADR-2020 | Separate staged presence framing from actual room, avatar and interaction execution; verify each target client |
| D5: Restore KPI lineage using SQLite/Oxigraph | [KPI source-to-panel review](kpi-outcomes.md) | Two implemented metrics with semantic/capture/lineage gaps; two await source data |
| D6: Schnorr challenge and payload DID match | [Naming and signed proof](federation-identifiers.md#mint-site-coverage-and-proof-of-identity), ADR-2022 | Freshness, reuse, audience and authority remain separate acceptance requirements |

This table is a section-level routing assessment. It does not newly ratify the historical Proposed record or assert that its incompletely assessed telemetry/KPI sections are implemented.

## Two bounded historical reconciliations

[ADR-090](../../../project/docs/archive/adr/ADR-090-hexagonal-crate-modularisation.md) contains a dated amendment to its planned crate list. Current ADR-2005 preserves the amended choice to use the root for startup, but extraction remains partial. The [current manifest and source review](vision-and-architecture.md#server-extraction-and-enforceable-boundaries) supersedes historical counts as an inventory observation. Historic build-time claims still need representative measurements; changed counts do not prove dependency-direction enforcement.

[ADR-110](../../../project/docs/archive/adr/ADR-110-agentic-actors-acsp-control-surfaces.md) joins an actor producer, forum schema and elevation workflow. Its successor ADR-2006 now explicitly limits the stateless claim. The [current consumer review](forum-decisions.md#visionclaw-acsp-consumption-and-recovery) identifies pending state, event-correlation loss and recovery requirements. Preserve the signed forum surface decision while keeping producer/consumer delivery and durable outcome acceptance open.

## Closeout sequence

CP-01/09 first reviews compound records and commitments absent from operative lineage. For each section, record its accountable maintainer, intended outcome, current source/consumer evidence and explicit disposition. CP-02–08 then supplies the relevant implementation or runtime acceptance criteria. Reconcile formal supersession and generated indices only after semantic review. Extend this method to agentbox, other first-party histories and the upstream/vendor packs; matching copied text establishes provenance, not local adoption.

## Agentbox routing and compound obligations

The [agentbox map](closeout/agentbox-history.md) covers all 72 inventory-classified historical records. Twenty-six have exact-number mentions in operative lineage; 46 do not. The [receipt](evidence/agentbox-history-map.json) preserves source hashes and exact mentions with the same limits as the VisionClaw map. Missing or coincident numeric references remain semantic-review work.

Archived [ADR-037](../../../project/agentbox/docs/archive/adr/ADR-037-gap-close-agentbox-decisions.md) contains eight independent decisions. Current reviews provide routes, not blanket closure:

| Section | Assessment route | Surviving obligation |
|---|---|---|
| D1 failure taxonomy | [Learning evidence](learning-evidence.md) | Verify producer coverage and unmapped handling; failure classification is not task-quality evaluation |
| D2 action authority | [Agent governance](agent-grounding-and-governance.md) | Carry signed case authority through waiting, decision recovery and actual action enforcement |
| D3 trajectory learning | [Learning ordering](learning-evidence.md) | Separate collection, stored corpus, retrieval, ranking and routing activation |
| D4 cross-model consultation | [Selector and envelope review](agent-grounding-and-governance.md#cross-model-consultation-and-acceptance) | Unknown-family acceptance and warning-only envelopes require production dispatch/admission evidence |
| D5 live mirror provenance | [Runtime egress](runtime-egress-and-profiles.md) | Test actual content, recipient and resolvable provenance through the live mirror separately from digest output |
| D6 spawn identity | [Federation identifiers](federation-identifiers.md) | Separate historical Multikey representation, current canonical naming, proof and persistence |
| D7 voice authority | [Voice mandate review](agent-grounding-and-governance.md#voice-speaker-target-and-mandate-scope) | Issuer/resource/revocation and downstream application remain open beyond tested producer dispatch |
| D8 skill count | [Capability instructions](capability-instructions-and-enforcement.md) | A count/lint gate establishes inventory properties, not executed capability or runtime liveness |

The historical D2 response-kind wording also requires reconciliation with the current ACSP request/response contract. Do not copy it into a new protocol specification merely because it appears in a predecessor.

## Journal and scheduling proposals remain distinct

Archived [ADR-057](../../../project/agentbox/docs/archive/adr/ADR-057-replayable-agent-execution-journal.md) is proposed. Its five decisions require one append-only execution journal, reconstruction of model-visible inputs, derived projections, separation from live control, and adapter-by-adapter coverage. Its acceptance explicitly includes duplicate handling, crash-tail recovery, replayed transcript/cost/mirror equivalence and projection watermarks. The [reviewed mirrors](runtime-egress-and-profiles.md), [memory](shared-memory.md) and [learning records](learning-evidence.md) do not establish that harness-neutral journal contract. The [current journal composition review](shared-memory.md#execution-journal-durability-and-reconstruction) credits implemented primitives but reproduces a failed-append acknowledgement gap. Retain proposed status and require durable adapter/model-call integration; do not relabel telemetry as completion.

Archived [ADR-072](../../../project/agentbox/docs/archive/adr/ADR-072-evaluator-before-schedule.md) is also proposed. Its context distinguishes an absent or un-runnable evaluator from a candidate failing evaluation. The [existing dream review](self-improvement.md) establishes additional candidate-ordering and verdict gaps, but not enforcement of evaluator readiness before scheduling. The [current admission trace and four extracted cases](self-improvement.md#evaluator-readiness-before-scheduling) show that empty, inline and missing-script configurations pass the inspected validator. Per-deep readiness enforcement remains open. A syntax-success receipt from one historical night cannot certify every future deep or the candidate patch.

CP-01/07/08/09 keeps these proposals and compound sections visible until accepted, deliberately deferred or retired with rationale. Preserve the archived source and attach current evidence in companions. Continue with consultation, voice and journal/scheduling consumers rather than treating operative-pack coverage as the end of the estate assessment.

## RuView strategy supersession and surviving domains

[RuView ADR-002](../../../RuView/docs/adr/ADR-002-ruvector-rvf-integration-strategy.md) is Superseded by ADR-016/017. Keep that lineage, but its “fully realised” and “no decision dropped” notes are too broad. The successors focus on training replacements and seven signal/MAT integration points. The current `wifi-densepose-ruvector` library exposes signal, MAT, viewpoint and optional CRV modules; it is not the proposed `wifi-densepose-rvf` crate containing all storage, learning, audit, consensus, crypto and runtime adapters. The [six-source receipt](evidence/ruview-strategy-lineage.json) records current manifests, public modules and predecessor/successor hashes. This is scope reconciliation, not package-publication or runtime verification.

| Original rollout domain | Evidence route | Surviving closeout obligation |
|---|---|---|
| ADR-003 cognitive containers | [Dedicated container assessment](sensing-extension.md#cognitive-container-contract-and-durable-lifecycle) | Establish container schema, writer/reader compatibility, consumed model/index/runtime segments and durable lifecycle; do not infer every RVF capability from a container name |
| ADR-004 HNSW fingerprints | [Fingerprint retrieval](sensing-extension.md#fingerprint-retrieval-and-hnsw-acceptance) | Actual selected search implementation, dimensional validity, persistence and recall/scale evidence; the server helper is a linear scan |
| ADR-005 SONA | [Adaptation lifecycle](sensing-extension.md#sona-feedback-admission-and-profile-lifecycle) | Admit real feedback, validate profile/model compatibility and demonstrate recoverable adaptation |
| ADR-006 GNN | [Three-mode coverage](sensing-extension.md#gnn-mode-coverage-and-learning-evidence) | Separate spatial forward inference from query refinement, temporal reasoning and online weight updates |
| ADR-007 secure sensing | [Model admission](sensing-extension.md#secure-sensing-claims-and-model-admission) | Select actual cryptographic primitives and enforcement at each load/transit/storage boundary; library presence does not establish PQ protection |
| ADR-008 consensus | [Multi-device agreement](sensing-extension.md#multi-device-agreement-and-replicated-state) | Replicated state, partition/rejoin, clocks and accepted updates; multistatic fusion does not establish consensus |
| ADR-009 portable edge | [Edge runtime](sensing-extension.md#edge-runtime-implementation-and-isolation-boundaries) | Boot/sense/sync/sleep and persistence across supported profiles, actual runtime limits and model admission |
| ADR-010 witness audit | [Audit completeness](sensing-extension.md#witness-segments-and-audit-completeness) | Signed event-chain completeness, durable append/recovery and verification policy; JSON witness metadata is narrower |

The prose says nine integration domains, while the rollout table and ADR-003–010 range contain eight. Attention is separately named in the proposed adapter tree and capability list; treat its exact scope as a reconciliation item rather than inventing a ninth completed phase. [Training](sensing-extension.md#training-integration-and-model-evidence) and [signal/MAT](sensing-extension.md#signal-and-mat-integration-helper-availability-versus-execution) assessments supply the evidence for ADR-016/017's narrower integration work. Copy-on-write branching, also promised in the original benefits, needs an explicit owner/implementation disposition and branch-isolation/space-cost evidence; it is not closed by those algorithm integrations.

The six original limitations map to unresolved obligations across this table: persistent vectors to container/search lifecycle; static inference to admitted adaptation; threshold matching to evaluated retrieval; missing audit to witness/security; limited edge to portable runtime; single-node state to consensus. This preserves the intended system scope while crediting the successor work. It does not treat those obligations as already delivered or automatically required in every estate deployment: consumption and any deliberate deferral/retirement must be ratified.

### Dependency and acceptance reconciliation

The original dependency correction should remain historical. Current workspace declarations include the five 2.0.4-named dependency constraints, but publication availability and the broad claim that every other capability is accessible through them were not verified here. The [upstream scope map](upstream-decision-scope.md) distinguishes registry locks, local source and copied ADRs. Use that consumed-revision evidence before selecting an API.

The example `mincut-matching`, `solver-interpolation`, `attn-mincut`, `temporal-compress`, `attention` and `full` flags are not the current integration crate's feature matrix: it has empty default features and optional `crv`; the five base RuVector dependencies are declared directly. The proposed separate signal/NN/MAT adapter files likewise require an explicit mapping to actual modules/callers. Do not use historical feature names as verified build instructions.

All five original risk mitigations remain acceptance requirements: version/API compatibility at a consumed revision; measured abstraction overhead; actual feature/target matrix; independent value and scope for each adopted increment; measured edge artifact size. The 10–100× lookup, sub-millisecond adaptation, compression/branch savings, 5.5 KB runtime, boot timing, added binary size and complete self-contained deployment claims remain unmeasured for the selected RuView path. Original week ranges are unratified estimates, not current deadlines.

CP-01/06/07/08/09 should first assign a current decision route to every domain and branching/attention obligation, then approve adoption/deferral/retirement, followed by dependency-bound implementation and acceptance. Integration and individual domain maintainers are proposed accountable roles. Supersession may settle where a decision lives; it does not erase its unmet requirements.
