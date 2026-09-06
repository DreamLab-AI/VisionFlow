---
title: Complete-system closeout roadmap
status: in-progress
date: 2026-09-04
type: explanation
---

# Complete-system closeout roadmap

This roadmap extends the estate review into the ADR corpus at the user's request. Closeout means that the intended system has an explicit decision record, its implementation and activation agree with that record, and the critical journeys have reproducible evidence. A passing component test or an ADR marked complete is insufficient by itself.

**Current state:** inventory and initial evidence-backed work packages are available. Evidence-backed extensions span multiple repository packs; the dated amendment entries below and generated inventory identify their current scope. Affected governing documents are updated alongside operative records. Record-by-record source verification, further amendments across the repository packs, accountable-owner confirmation and full-system evidence remain unfinished. [Every discovered ADR candidate](adr-inventory.md) is listed; the [machine-readable inventory](../evidence/adr-inventory.json) preserves source hashes and declared status. The inventory includes historical and support documents so they cannot disappear from scope by omission. Its provisional classifications need review, especially imported material and ontology pages named after ADRs.

The [assessment completion requirements](assessment-requirements.md) distinguish finishing this requested review from implementing every repair on the roadmap. Current record coverage and remaining assessment work are explicit there.

## Amendment policy

Preserve historical decisions and their dates. For operative records, retain the existing decision / implementation / activation axes and add a closeout section with: current evidence, unresolved gaps, a work-package identifier, dependencies, accountable role, verification commands or receipt requirements, and an explicit acceptance condition. Where evidence contradicts a completion claim, amend that claim with the reason and revision; do not silently erase its history.

Consolidated packs treat their living governing document as normative. Amend that document and its operative ADR together, and regenerate indices using the repository's own generator. Frozen archives receive lineage/disposition links through companion indices rather than having obsolete designs rewritten as current policy. Proposed changes remain proposed until their normal decision process adopts them. Existing owner fields are preserved; roles below identify responsibility needed, not personal assignments already accepted.

Start delivery planning with the [ordered execution sequence](execution-sequence.md). Check the [refreshed assessment audit](assessment-requirements.md#pass-65-assessment-audit--2026-09-04) before interpreting corpus coverage as completion.

## Work packages and dependency order

| Package | Scope and repositories | Required closeout result | Accountable role | Dependencies |
|---|---|---|---|---|
| CP-01 Decision and release identity | All 14 inventoried repositories; expand when links establish more estate-owned repos | Classify every record, resolve governing-document lineage, distinguish upstream/imported scope, bind source/build/deployment revisions and features | Estate architecture steward with repository maintainers | None |
| CP-02 Corpus and semantic publication | visionGraph, knowledgeGraph, historical Logseq, VisionClaw, WasmVOWL | Strict input/publication validation; deliberate inference visibility; consistent RDF identity; one versioned consumer bundle; explain explorer lineage | Knowledge and ontology maintainers | CP-01 |
| CP-03 Grounded execution | Loom, agentbox, VisionClaw | Preserve generation, provenance, domain and budget contracts across backend choice/cache/fallback; measure actual consumer paths with negative controls | Grounding and runtime maintainers | CP-02 |
| CP-04 Identity, authority and storage | solid-pod-rs, forum, agentbox, VisionClaw | Signed scoped grant → mutation → receipt → revocation → denied retry across tiers, caches and restart; explicit partial-provenance repair | Identity/storage maintainers | CP-01 |
| CP-05 Human judgement and governance | Forum, agentbox, VisionClaw, commercial website | Human intent and rationale bound to exact request; atomic/recoverable decision projection; applied/rejected acknowledgement; supersession and appeal propagated safely | Governance and product maintainers | CP-03, CP-04 |
| CP-06 Runtime and embodied interaction | VisionClaw, agentbox, WasmVOWL, RuView | Trace ingest/query/persistence/render/XR/voice paths; test authenticated actions and binary contracts; distinguish simulated sensing from hardware evidence | Runtime/rendering maintainers | CP-02, CP-04 |
| CP-07 Memory and improvement | RuVector consumed modules, agentbox, dream-engine | Compatible embedding/persistence and privacy; repeatable recall; tested candidate bound to acceptance; fair scheduling and restart recovery | Memory/evaluation maintainers | CP-01, CP-03, CP-04 |
| CP-08 Delivery and recovery | All repositories, including VisionFlow and dreamlab-ai-website | Reproducible release manifest, meaningful CI gates, private/public delivery policy, observable failure and tested restoration | Release/operations maintainers | CP-02–07 |
| CP-09 Complete-system acceptance | All repositories and governing ADR packs | Demonstrate the journeys below; reconcile every operative ADR and historical disposition; publish a dated evidence index with remaining limitations | Estate acceptance owner | CP-01–08 |

Work within packages can proceed independently when it does not depend on an unverified upstream contract. This is a dependency roadmap, not a schedule or staffing commitment. Dates require owner and capacity decisions; invented delivery dates would weaken it.

## Evidence-backed initial backlog

| Package | Established gap | Required change or evidence | Current evidence |
|---|---|---|---|
| CP-01 / CP-08 | Old repository/count gates and fixture success-skips do not cover the current estate | Define authoritative roster and denominators; require an actual compared revision set; inject a mismatch and verify CI rejects it | [Canon](../canon-and-verification.md), ER-001–005 |
| CP-02 | Malformed pages disappear before validation; non-boolean public flags publish; private ancestors enter derived public outputs | Input census diagnostics, strict flags, inference-visibility policy tested on each export | [Knowledge](../knowledge-production.md), [current vault](../authored-vault-transition.md), ER-007–011, ER-018–019 |
| CP-03 | Cache loses domain/budget overrides; expansion degradation hidden; serving generation can differ from loaded content | Full effective-request cache contract, returned degradation, atomic bundle activation with served identity | [Grounding](../grounding-delivery.md), [agent boundaries](../agent-grounding-and-governance.md), ER-012–014 |
| CP-04 | Forced-local mutation has a different gate; malformed policy can inherit broader access; private responses advertise public caching | Explicit local-authoring authority, typed policy errors, visibility-aware cache policy, cross-tier deny probes | [Agent boundaries](../agent-grounding-and-governance.md), [storage](../storage-and-authority.md), ER-015, ER-024–027 |
| CP-05 | Early approval can miss its waiter; relay OK precedes projection and application | Register/replay durable decisions; transactional or reconciled projection; separate sent/committed/applied receipts | [Agent boundaries](../agent-grounding-and-governance.md), [forum](../forum-decisions.md), ER-016–017 |
| CP-07 | Dream baseline is evaluated before emitted patch; free-text verdict can accept failure; revision reads and run identity diverge | Bounded candidate rerun, typed required-check gate, frozen experiment manifest and durable fair scheduling | [Self-improvement](../self-improvement.md), ER-020–023 |

CP-06 now has source-backed rendering and sensing conditions in the later sections. Remaining CP-04/07/08 features still require investigation before a complete implementation backlog can be claimed. They stay in scope rather than being marked complete because other packages already have findings.

## System acceptance journeys

Each journey must name exact component revisions, configuration, input fixture, identity/authority, expected result, observed result and persisted receipt. Negative controls and recovery steps are part of acceptance.

1. **Author to grounded answer:** publish a validated public corpus; exclude private source and private inferred material; activate one generation; retrieve through both agent tools and chat; demonstrate domain/budget/provenance limits and deliberate degradation.
2. **Request to human decision to applied change:** a registered agent submits a correlated proposal; the authorised human sees its evidence and signs a decision; relay and broker persist it; exactly one intended mutation is applied or explicitly rejected; replay, timeout, opposite decision and revoked signer do not create ambiguous success.
3. **Owned data through revocation and recovery:** grant least authority, write/read a resource, verify provenance state, revoke access, test inherited policy and caches, restart the relevant tier and restore from a backup while preserving explainable ownership/history.
4. **Action to rendered state:** ingest an authenticated action, preserve graph identity and persisted semantics, transmit the correct wire representation, render it in intended desktop/XR surfaces, and show denied/stale/disconnected states honestly. Hardware sensing claims require hardware evidence; simulation must remain labelled.
5. **Learning to reviewed improvement:** store and retrieve compatible memory, nominate an experiment, freeze its baseline/candidate, execute required evaluators, reject a broken candidate, retain raw receipts, open a reviewable result and recover after interrupted persistence without silently repeating or losing the run.
6. **Release to operation:** build the declared estate revisions, run effective gates, deploy through the actual consumer paths, observe a seeded failure and restore service/data. Public claims, ADR activation status and the deployment receipt must agree.

## Exit criteria for ADR work

Every decision candidate must receive one of: operative and verified; operative with an explicit remaining closeout condition; proposed/deferred with activation prohibited or bounded; superseded with a valid successor; historical with mapped lineage; imported and outside estate authority; or false-positive/support material with an explanation. “Needs review” is an intermediate state, never final closeout.

Every operative record needs an accountable owner, review trigger, evidence revision, implementation and activation status, and completion condition. Every affected governing document and generated index must agree. Every critical journey must have a passing receipt or a named, accepted limitation that genuinely bounds the intended product; an unexamined feature is not such a limitation.

The programme remains open until these conditions hold across the complete estate. The [investigation ledger](../investigation-ledger.md) tracks evidence acquisition; this roadmap tracks the decisions and implementation needed to close the system.

## Current ADR validation receipts

VisionFlow's seven-record pack and solid-pod-rs' seven-record pack pass their local index validators. The initial agentbox validator reported **11 stale governed-path declarations**; [the full receipt](../evidence/agentbox-adr-validation.json) records the affected records and paths. These need source re-verification before advancing `verified_commit`. The existing index was not regenerated after that failed validation. Passing metadata validation in any pack is not evidence that its system acceptance conditions have passed.

## Source re-verification progress — 2026-09-04

Agentbox ADR-2002, 2009, 2010 and 2011 now have refreshed source evidence and scoped closeout conditions. Its proxy suite passes 45 assertions with no skips when using the installed runtime dependency. ADR-2022 now records **partial** implementation because forced-local authoring precedes the remote direct-load guard; its governing document was amended with it. The [current validator receipt](../evidence/agentbox-adr-validation-pass7.json) retains the remaining stale declarations. These amendments do not establish deployed activation or complete the other repository packs.

## VisionClaw persistence amendments — 2026-09-04

ADR-2004, 2015, 2016 and 2017 now carry source-backed closeout extensions; their architecture/data-authority governing documents were amended alongside them. ADR-2016 changes to partial implementation because individual quad inserts do not guarantee its complete-record claim. The temporary backup probe restores WAL data but also demonstrates partial-set success and an in-source default destination. [Current validation](../evidence/visionclaw-adr-validation.json) records unresolved pack issues; these are not bypassed to regenerate an index.

## Commercial overlay amendments — 2026-09-04

All eight operative website ADRs now have closeout extensions covering CP-01/03/04/05/06/08. Their governing documents distinguish release-gate coverage, pin resolution, mirror coverage and chat authority/correlation. [Evidence](../commercial-surfaces.md): 97 local tests, pin parity and eight valid ADRs. Frozen website archives still need lineage disposition. No live-service acceptance is inferred.

## XR and protocol amendments — 2026-09-04

VisionClaw ADR-2018/2019/2020/2034/2036 and their governing documents now carry CP-01/04/06/08 closeout conditions. The [review](../rendered-state.md) distinguishes implemented hover/query work from missing ordering, attribution and headset evidence. All 218 local XR library tests pass; this does not certify Godot runtime or deployment.

## Sensing extension amendments — 2026-09-04

RuView ADR-018/023/028/035 now carry CP-01/06/08 conditions and scoped historical claims. The [sensing review](../sensing-extension.md) records 100 passing hardware-crate tests but a reproduced firmware/server parser mismatch. Identify an actual estate consumer and accepted ownership before treating this evaluated extension as an integrated sensor. Hardware and trained-model acceptance remain open.

## Shared memory amendments — 2026-09-04

Agentbox ADR-2014/2018/2019 now link to the [memory review](../shared-memory.md). CP-03/04/07/08 require coherent value/vector recovery, explicit expiry semantics, effective-model identity and current recall evidence. Ten mock factory tests pass; embedding failure still allows degraded writes, so ADR-2014's full guarantee is partial.

## Coverage and lineage accounting — 2026-09-04

The current 853-candidate inventory contains 540 decision candidates, 278 historical records and 35 support/ontology candidates. Of the decision candidates, 31 carry the dated evidence-backed closeout marker; the remaining 509 still require review under that marker-based accounting. This is not a quality or completion percentage: the coordination ADR, historical dispositions and imported material need different treatment.

[Vendored lineage](adr-lineage.md) identifies 176 RuView/RuVector candidates: 161 byte-identical to standalone paths and 15 divergent. They stay in the inventory as reference lineage, with integration relevance and ownership unresolved. Each divergent copy must be traced before choosing a governing successor; copying an amendment into both trees would not by itself resolve authority.

## Explorer consumer proposal — 2026-09-04

WasmVOWL now has proposed ADR-001 for CP-01/02/06/08. The [explorer review](../ontology-explorer.md) documents failing frontend state initialisation and schema mismatch despite passing native Rust tests. Publisher variants need independent validation; no browser/performance acceptance is inferred. Inventory totals must be read from the regenerated inventory after this new record.

## Learning evidence amendments — 2026-09-04

Agentbox ADR-2015/2016 now carry CP-04/07/08 privacy, outcome and recovery conditions. [Evidence](../learning-evidence.md) includes 27 passing helper tests and synthetic redaction counterexamples. Skip-on-error and a Wilson floor remain valuable but do not prove secret-free retention or task achievement.

## Loom decision amendments — 2026-09-04

ADR-135–138 and RUST-ARCHITECTURE now carry CP-01/02/03/08 acceptance conditions. The [grounding review](../grounding-delivery.md) and [current receipt](../evidence/loom-closeout-snapshot.json) distinguish library/generation tests from serving activation and profile parity. The agentbox ADR-051 support stub now resolves archived and operative lineage without becoming a duplicate decision.

## Extracted producer amendments — 2026-09-04

knowledgeGraph ADR-2001–2004 now carry CP-01/02/06/08 identity, publication and consumer conditions. The [producer review](../knowledge-production.md) preserves useful count/validation gates while requiring input census and equal-count identity/visibility probes. Two ADR-named ontology pages are now correctly classified as corpus content.

## Dream programme amendments — 2026-09-04

The three toolkit/fork ADRs and agentbox ADR-2024 now link CP-01/07/08 to frozen candidate evaluation, typed evaluator vetoes, durable run identity and human review. The [review](../self-improvement.md) preserves the proposed status of outer-loop optimisation and makes its dependency on a trustworthy inner loop explicit.

## Forum decision amendments — 2026-09-04

All nine operative forum ADRs now carry CP-01/04/05/06/08 acceptance requirements. ADR-2006 is partial because sweep pagination and write-outcome reporting do not establish committed coverage. [Forum evidence](../forum-decisions.md#identity-and-trust-decision-closeout) includes 22 native key tests and a synthetic query reproduction. A governance receipt decision, archive dispositions and browser/deployment journeys remain open.

## Forum governance and history routing — 2026-09-04

Proposed [ADR-2010](../../../../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md) defines CP-05/08 receipt and recovery acceptance. The [history companion](../../../../nostr-rust-forum/docs/adr-history-closeout.md) routes all 24 frozen entries and three sprint canonical documents to current governing surfaces and work packages. Routing is established; adoption, implementation and historical source verification remain open.

## Current authored vault decisions — 2026-09-04

Proposed [visionGraph decisions](../../../../visionGraph/docs/adr/README.md) now define CP-01/02/03/04/06/08 publication and generation acceptance. Its governing contract separates public-site inclusion from local consumers and keeps adopted implementation distinct from the proposed complete contract. Historical Logseq decisions still require lineage review.

## Historical authoring decisions — 2026-09-04

The [Logseq companion](logseq-decision-lineage.md) maps six historical designs to CP-01/02/03/06/07/08 and current consumers. It preserves archival declarations and distinguishes two rendered ontology pages from decision records. This establishes routing and continuing acceptance requirements; historical implementation and browser/reload proof remain open.

## Shared pod decision pack — 2026-09-04

All seven operative solid-pod-rs ADRs now carry CP-01/04/08 acceptance requirements, with the living baseline updated. [Evidence](../storage-and-authority.md#operative-storage-decision-pack) distinguishes parser and resolver behaviour, bounded replay retention, feature-dependent provenance and separate anchor confirmation. Frozen-history disposition and full consumer/deployment evidence remain open.

## Consumed semantic artefact requirements — 2026-09-04

CP-01/03/07/08 now include effective database geometry/model validation and sidecar binding. [Actual adapter probes](../consumed-vector-storage.md) accept a wrong metric and mark a wrong width ready; Loom ADR-137/138 and RUST-ARCHITECTURE carry these acceptance conditions. Full semantic recall and deployed generation evidence remain open.

## Runtime profiles and session egress — 2026-09-04

Agentbox ADR-2007 is partial for substring-based endpoint validation; ADR-2026 is proposed/partial after its first source review. [Isolated probes](../runtime-egress-and-profiles.md) establish wrong-host acceptance and raw sentinel composition without any external send. CP-04/07/08 require explicit per-path egress, recipient and log-retention policies.

## Configuration projection acceptance — 2026-09-04

Agentbox ADR-2003/2008/2031 now carry CP-01/04/08 state-binding and recovery requirements. [Actual projector fixtures](../configuration-projection.md) retain deleted/unreadable-registry entries; ADR-2008 becomes partial. Nix/boot evidence and loaded-process convergence remain open.

## Adapter lifecycle and middleware — 2026-09-04

Agentbox ADR-2004/2005 now carry CP-03/04/08 acceptance for timeout/replacement and method/field-specific privacy coverage. Both are partial against their broad guarantees. [Dispatch evidence](../adapter-dispatch.md) distinguishes source lifecycle findings from isolated privacy probes; actual storage and startup recovery remain open.

## Capability instruction and lint acceptance — 2026-09-04

Agentbox ADR-2020/2021 are partial for unestablished runtime limits and insufficient frontmatter/context checks. [Evidence](../capability-instructions-and-enforcement.md) adds CP-01/07/08 executor, off-state and baked-revision acceptance requirements. No research performance, provider or build guarantee is inferred.

## Federation identifier acceptance — 2026-09-04

Agentbox ADR-2025 and VisionClaw ADR-2023/2025 now carry CP-01/02/04/05 byte, grammar and mapping requirements. [Paired helper evidence](../federation-identifiers.md) confirms tested hashes, exposes bead coverage divergence and prefix-only address acceptance. Two-repository CI, exposed-route validation and persisted recovery remain open.

## Vault path and Notes acceptance — 2026-09-04

Agentbox ADR-2028/2029 now carry CP-01/02/06/08 precedence, off-state, relocation and editor recovery requirements. [Resolver evidence](../authored-vault-transition.md#runtime-path-overrides-and-notes-launch) retains a legacy path while reporting disabled; Notes launch is binary-driven. Staged image/editor acceptance remains open.

## Learning ordering acceptance — 2026-09-04

Agentbox ADR-2017 is partial because W066 warns without rejecting consumer-before-producer configurations. [Scoped fixtures](../learning-evidence.md#producer-ordering-is-advisory) also distinguish stopped capture from an absent corpus. CP-01/07/08 require explicit retained-aggregate policy, validation/runtime admission, freshness and restart/override acceptance. Deployed route evidence remains open.

## Process identity and shutdown acceptance — 2026-09-04

Agentbox ADR-2032 is partial for coverage across signalling tools. [Process evidence](../process-lifecycle.md) distinguishes lexical discovery, registry-based eligibility, process identity, signal delivery and observed exit. CP-01/04/08 require caller coverage, stale/reused PID and workspace reconciliation, failure recovery and release-bound operational receipts. Four helper tests do not certify live shutdown.

## GPU wrapper and graphics acceptance — 2026-09-04

Agentbox ADR-2006 now records the reached graphics review trigger while preserving historical CUDA evidence. [Current source review](../rendered-state.md#gpu-packaging-and-runtime-boundary) adds CP-01/06/08 acceptance for wrapper/supervisor environment differences, locked evaluation and separate compute/presentation paths. Existing backend tests skip without Nix and need interface maintenance; current hardware and deployment acceptance remain open.

## Compose exposure gate acceptance — 2026-09-04

Agentbox ADR-2013 is partial after [actual scanner fixtures](../runtime-ingress.md#port-gate-syntax-and-exposure-coverage) accepted nested flow public mappings. CP-01/04/08 require structured effective-configuration coverage, complete deployment inputs and service-specific listener/authority receipts. Current tree gate success is retained with its exact scope; no present unsanctioned exposure is asserted.

## Relay admission acceptance — 2026-09-04

Agentbox ADR-2012 is partial: [source and helper evidence](../runtime-ingress.md#relay-admission-versus-inbox-authorisation) place its allowlist at inbox consumption after relay admission. CP-01/04/08 require backend-specific publisher/subscriber policy, empty-list verification, key removal/restart and durable delivery/replay receipts. Three helper tests do not certify the complete ingress journey.

## Decision-register navigation acceptance — 2026-09-04

Agentbox ADR-2001 retains partial/staged status. [Navigation and generator evidence](../canon-and-verification.md#decision-register-and-reader-navigation) adds CP-01/08 acceptance for historical routes, semantic baseline review and generated-index comparison. Six stale records currently prevent index regeneration; check-mode success alone does not prove index freshness.

## Custody and revocation acceptance — 2026-09-04

Agentbox ADR-2027 now links a [provisional custody register](../runtime-ingress.md#custody-and-revocation-acceptance). CP-01/04/07/08 require accepted custodians, complete deployed inventory, explicit revocation windows and protected recovery receipts. Seven proposed rows do not establish implementation; none/inactive remains for the full lifecycle policy.

## Service package acceptance — 2026-09-04

Agentbox ADR-2030 is partial for missing promised package texts/READMEs. [Current metadata inventory](../configuration-projection.md#service-package-and-release-metadata) adds CP-01/08 ownership, archive/dependency and release identity receipts. All 32 operative agentbox records now have closeout extensions; that coverage does not resolve stale baselines, historical lineage or full-system acceptance.

## VisionClaw replay acceptance — 2026-09-04

VisionClaw ADR-2002 now carries CP-04/05/08 [clock, body-binding and retry acceptance](../runtime-ingress.md#visionclaw-replay-and-operation-boundaries). Six helper assertions establish cache behaviour; full-route and deployment evidence remain open. Replay claims and application idempotency require separate receipts.

## VisionClaw visibility acceptance — 2026-09-04

VisionClaw ADR-2003 now carries CP-02/04/06/08 [metadata and output acceptance](../rendered-state.md#visibility-defaults-and-output-coverage). Six domain tests support the filter, and current source covers initial/position paths. Metadata authority, alternate outputs and client state after visibility changes remain open.

## Role authority acceptance — 2026-09-04

VisionClaw ADR-2010/2011 now carry CP-01/04/05/08 [caller-authority, revocation and mode requirements](../role-authority.md). Target transactions do not refresh caller roles; removal restores fallback and report mode forwards denials. Concurrency, composed-route and durable audit evidence remain open.

## Development and production artefact acceptance — 2026-09-04

VisionClaw ADR-2037/2038/2039 now distinguish implemented helper behaviour from proposed release/profile assertions. [Nine build cases](../role-authority.md#development-bypass-and-release-identity) add CP-01/04/06/08 artefact-feature, pre-listener and full-transport requirements. Production acceptance cannot be inferred from --release alone.

## Effective security-profile acceptance — 2026-09-04

VisionClaw ADR-2012/2026/2027 now align with the proposed release controls and current [effective-policy evidence](../role-authority.md#profile-claims-and-effective-policy). CP-01/04/08 requires a combined feature, bypass, route and role-fallback matrix. Report-mode expiry and release exclusion are not inferred from a dated acknowledgement.

## Vault converter acceptance — 2026-09-04

VisionClaw ADR-2042 is partial after [actual collision/report fixtures](../authored-vault-transition.md#converter-collision-and-dry-run-boundaries). CP-01/02/08 require unique destination planning, full input/output accounting and consumer/recovery validation. Existing 86-test success is retained with its scope; no real vault was changed.

## Vault inclusion acceptance — 2026-09-04

VisionClaw ADR-2014/2040 now distinguish [typed metadata, local fallback and public publication](../authored-vault-transition.md#inclusion-typing-and-local-fallback). CP-01/02/04/08 requires a validated formal-data exception and reader/fallback/migration receipts. Fifty-six domain tests do not certify every consumer path.

## Knowledge settings compatibility acceptance — 2026-09-04

VisionClaw ADR-2041 now carries CP-01/02/06/08 [persisted and mixed-version compatibility requirements](../configuration-projection.md#knowledge-settings-migration). Eleven helper/migration tests pass. Actual save/reload, binary peer identity and named alias retirement remain open.

## Identifier mint-site acceptance — 2026-09-04

VisionClaw ADR-2021/2022 now distinguish [constructor coverage, signed proof and persistence](../federation-identifiers.md#mint-site-coverage-and-proof-of-identity). CP-01/04/05/08 requires a complete mint/lookup inventory, approved legacy exceptions and governed migration receipts. Prior helper parity does not prove universal constructor adoption.

## Request realm and delegation acceptance — 2026-09-04

VisionClaw ADR-2009/2013 now distinguish [current client signing, legacy server acceptance and deferred delegation](../role-authority.md#request-realms-and-deferred-delegation). CP-01/04/05/08 requires a deployed consumer census, session lifecycle/retirement receipts and an explicit scoped delegation decision. Eleven mocked interceptor tests pass; complete authentication and revocation journeys remain open.

## Simulation and compact-wire acceptance — 2026-09-04

VisionClaw ADR-2024/2028/2029 now carry [layout, force and overflow acceptance](../rendered-state.md#simulation-layout-and-force-authority). CP-01/02/06/08 requires field-level actual-toolchain compatibility, loaded module identity, residency transitions and all-class capacity/generation tests. All 53 extracted host field offsets match, but a same-size swap passes the existing guard. Device and deployed-client evidence remain open.

## PTX provenance and compatibility acceptance — 2026-09-04

VisionClaw ADR-2030 is partial after [six isolated PTX-phase cases](../rendered-state.md#ptx-build-acceptance-and-loaded-artefact-identity). CP-01/06/08 requires compiler-failure distinctions, selected-module provenance, actual ABI/symbol/driver checks and rollback. Nonempty files and rewrite warnings do not establish runtime compatibility; native linking and device execution remain untested in this pass.

## XR controls and hierarchy acceptance — 2026-09-04

VisionClaw ADR-2032/2033/2035 now carry [renderer, control and hierarchy criteria](../rendered-state.md#xr-control-coverage-and-hierarchy-semantics). CP-01/02/06/08 requires full control coverage, producer-label reconciliation and exported headset/runtime receipts. ADR-2033 is partial; an existing extracted hierarchy test fails against current acceptance. ADR-2031 is explicitly disposed as a preserved tombstone, with no feature delivery obligation.

## ACSP consumer acceptance — 2026-09-04

VisionClaw ADR-2006 is partial after [consumer/state/correlation review](../forum-decisions.md#visionclaw-acsp-consumption-and-recovery). CP-01/03/04/05/08 requires a durable case authority, retained signed-event correlation, explicit action semantics and early-response/lag/restart/persistence-failure receipts through mutation outcomes. Removal of the old broker transport does not make the full workflow stateless.

## Crate boundaries and supervision acceptance — 2026-09-04

VisionClaw ADR-2005/2007 now carry [responsibility boundaries](../vision-and-architecture.md#server-extraction-and-enforceable-boundaries) and [acknowledged context/recovery criteria](../rendered-state.md#gpu-supervision-and-context-delivery). CP-01/03/06/08 requires actual dependency/build evidence and child/device failure recovery. Twelve manifest members and four supervisors establish structure, not completed extraction or isolation.

## Development inputs and operative-pack acceptance — 2026-09-04

VisionClaw ADR-2001/2008 complete dated extension coverage of the 42-number operative pack. [Development fixtures](../configuration-projection.md#development-restart-and-build-input-coverage) expose missed crate CUDA/manifest inputs; [corpus validation](../canon-and-verification.md#visionclaw-operative-pack-coverage-and-baseline-debt) still reports four stale baselines. CP-01/06/08/09 requires actual-image build identity and semantic/index/history reconciliation. Coverage is not full-system closeout.

## Historical obligation reconciliation — 2026-09-04

The [VisionClaw historical map](visionclaw-history.md) covers 137 candidates: 43 explicit operative-lineage mentions and 94 without one. [Section-level review](../historical-decision-reconciliation.md) separates ADR-130's six decisions and retains unassessed liveness/KPI obligations. CP-01/09 requires semantic disposition before supersession; the map does not certify historical closure or upstream adoption.

## Liveness evidence acceptance — 2026-09-04

Historical VisionClaw ADR-130 D3 now has a [current observer assessment](../liveness-observation.md). CP-01/03/07/08/09 requires typed outcome/predicate and producer-revision evidence plus standing-loop continuity. Durable fires and optional relay wiring exist; a recent fire, including a failure transition, is not sufficient evidence of successful operation or wave promotion.

## KPI outcome acceptance — 2026-09-04

Historical VisionClaw ADR-130 D5 now has a [source-to-panel assessment](../kpi-outcomes.md). CP-01/03/05/07/08/09 requires accepted metric definitions, capture health, auditable run boundaries and stale-display evidence. Two metrics exist and two await source data. Summary persistence/canary firing does not prove nonempty traffic or improved outcomes.

## Canary consumer and promotion acceptance — 2026-09-04

[Three mocked D1 checker cases](../liveness-observation.md#consumer-evidence-and-promotion-scope) distinguish roster/transport success from rendered-beam and durable-receipt evidence. CP-01/06/08/09 requires journey-matched predicates, acknowledgement/retry, visible freshness and explicit canon ratification. The bounded search did not establish an automatic wave-promoter, and no real promotion occurred.

## Agentbox historical obligation acceptance — 2026-09-04

The [agentbox routing map](agentbox-history.md) covers 72 archived candidates, with 26 exact operative-lineage mentions. [Section-level assessment](../historical-decision-reconciliation.md#agentbox-routing-and-compound-obligations) separates ADR-037's eight decisions and preserves journal/evaluator-readiness proposals. CP-01/07/08/09 requires explicit adoption/disposition and consumer evidence; mirrors and available scripts do not establish those stronger contracts.

## Independent consultation acceptance — 2026-09-04

Historical agentbox ADR-037 D4 now has [helper and envelope evidence](../agent-grounding-and-governance.md#cross-model-consultation-and-acceptance). CP-01/03/05/07/09 requires known executed-model identity, production selector/admission wiring and candidate-bound outcomes. Three helper assertions pass; warning-only consultation success does not establish independent closure verification.

## Voice authority and application acceptance — 2026-09-04

Historical agentbox ADR-037 D7 now has [producer tests and authority review](../agent-grounding-and-governance.md#voice-speaker-target-and-mandate-scope). CP-01/03/04/05/06/08 requires issuer/target/action/revocation binding and applied/rejected receipts. Nineteen existing mocked tests pass; signed request dispatch does not prove authorised target execution.

## Execution journal durability acceptance — 2026-09-04

Historical agentbox ADR-057 now has [actual journal/adapter composition evidence](../shared-memory.md#execution-journal-durability-and-reconstruction). CP-01/03/07/08/09 requires durable acknowledgements, retry/hydration and content-bound pre-model provenance. The temporary failed-write fixture still advances journal state and accepts citations; implemented helpers do not establish a complete replayable execution record.

## Evaluator admission acceptance — 2026-09-04

Historical agentbox ADR-072 now has [four extracted validation cases](../self-improvement.md#evaluator-readiness-before-scheduling). CP-01/07/08/09 requires per-deep, target-bound evaluator readiness before scheduling, separate from candidate evaluation and promotion. The narrow Darwin guard is implemented; empty and missing-script cases remain admitted by the inspected validator.


## VisionFlow operative acceptance — 2026-09-04

The [seven operative canon extensions](../canon-and-verification.md#visionflow-operative-closeout-scope) preserve implemented build/deploy/diagram mechanisms while assigning publication quality, real parity checks, release identity and historical reconciliation to CP-01/08/09. The [latest assessment audit](assessment-requirements.md#pass-66-coverage-update--2026-09-04) records remaining coverage; local validator and baseline-diagram success do not complete those obligations.

## Catalog recommendation acceptance — 2026-09-05

The [local catalog assessment](../catalog-decisions.md) extends nine agentbox skill ADRs and classifies four BHIL example/template records as support. CP-01/03/07/08/09 requires an amended loading/ranking contract, authoritative data, separate interface scope tests and reproducible benchmark receipts. Current source has useful search and proposal mechanisms, but those do not establish model recommendation quality or upstream technology adoption.

## Forum navigation and message lifecycle acceptance — 2026-09-05

CP-01/06/08/09 now includes the [sprint consumer review](../forum-decisions.md#forum-navigation-counts-and-cold-entry): deployment-base navigation, tombstones on every replay path, store-to-page deletion reconciliation and cold-entry recovery. The three canonical sprint ADRs have dated extensions; source inspection does not complete browser/relay acceptance.

## Engineering governance acceptance — 2026-09-05

The [17 section dispositions](../engineering-governance.md) distinguish accepted harness mechanisms, deferred validation and speculative mandate governance. CP-01/04/05/07/08/09 requires distinct source-verified control coverage and real authority/recovery evidence. Audit-script fixtures pass missing sources and duplicate edges; declared pairing is not runtime acceptance.

## Upstream source and adoption acceptance — 2026-09-05

Use the [393-record pack map](upstream-packs.md) with the [consumer identity assessment](../upstream-decision-scope.md). CP-01/03/06/07/08/09 must bind claims to the selected source/package/features, distinguish copied review histories and record explicit adoption. Loom’s sibling path dependency and RuView’s registry lock entries are different source authorities. Grouping and byte equality do not complete semantic review.

## RuView signal integration acceptance — 2026-09-05

ADR-017 now has a [seven-point consumer assessment](../sensing-extension.md#signal-and-mat-integration-helper-availability-versus-execution). CP-01/06/07/08/09 requires caller selection, fallback visibility, bounded retention and measured numerical/performance outcomes. Existing helper code does not establish the claimed runtime improvements.

## RuView training integration acceptance — 2026-09-05

ADR-016 now has [five separate integration dispositions](../sensing-extension.md#training-integration-and-model-evidence). CP-01/06/07/08/09 requires default-path selection, assignment/gradient validity, actual dataset compression and held-out model/resource evidence. The earlier blanket completion table does not establish these outcomes.

## RuView dataset and metric acceptance — 2026-09-05

The [ADR-015 assessment](../sensing-extension.md#dataset-separation-and-evaluation-meaning) assigns subject/environment/device separation, teacher-label provenance and observed-versus-imputed metrics to CP-01/06/07/08/09. Current real-data CLI validation is synthetic; six native evaluator assertions verify metric defaults, not model generalisation.

## MERIDIAN adaptation acceptance — 2026-09-05

The [seven-phase ADR-027 assessment](../sensing-extension.md#cross-environment-adaptation-and-readiness) requires readiness enforcement and calibration-to-installed-model evidence under CP-01/06/07/08/09. Four native assertions show successful adaptation below the configured frame minimum; helper output is not a measured model improvement.

## Book claim reconciliation — 2026-09-05

The [canon chapter assessment](../canon-claim-reconciliation.md) maps eleven assertions to current implementation evidence and CP-01/04/05/06/07/08/09 obligations. Retain dated history, credit implemented primitives and qualify universal enforcement language. Other book chapters and external-study transfer claims remain unreviewed at this scope.

## Published milestone reconciliation — 2026-09-05

The [eighteen-commitment map](../book-roadmap-reconciliation.md) links the book roadmap to CP-01–09 and identifies unexamined CTC, failure-census, pocket-provenance and disclosure consumers. Preserve historical dates and falsification criteria; only target-client and complete-journey receipts can close its published milestones.

## Joined provenance acceptance — 2026-09-05

CP-01/04/08: the [trace review](../joined-provenance-trace.md) establishes an implemented identity timeline with two SQLite sources, not a causally complete action receipt. Require task correlation, resource-authorised reads, capture health, bounded windows and a connected pod source before closing the broader milestone. Repeated reads and shared identity alone must not establish independent completed work.

## Transaction-cost acceptance — 2026-09-05

CP-01/07/08: [CTC accounting](../transaction-cost-accounting.md) has tested forwarding but no verified complete per-DAG measurement consumer. Reconcile turn/per-step/cumulative units, deduplicate usage, recover or disclose missing events and validate the actual aggregation/display on a multi-agent workflow. Token or chain field presence cannot close this milestone.

## Failure telemetry acceptance — 2026-09-05

CP-01/06/07/08: [failure telemetry](../failure-telemetry.md) needs an explicit source census, consistent wire field and durable count producer. Test malformed requests, policy denials, ambiguous/classified outcomes, process loss and capture outage. The QE fleet remains a separate unverified producer obligation; a taxonomy library and live-agent dashboard canary do not close it.

## Decision-history acceptance — 2026-09-05

CP-04/05/08/09: [history consumers](../decision-history.md) need ratified read scope, stable traversal, retained signed request context and projection/application reconciliation. The local “Current” badge and a successful D1 query establish different evidence; neither is a complete applied-action receipt.

## Agent-disclosure acceptance — 2026-09-05

CP-04/06/09: [disclosure source review](../agent-disclosure.md) credits public access and sixteen mounts. Close freshness, lookup-failure and historical-principal semantics, then verify the complete author-surface census with signed-out/mobile readers. Keep registration, authorship and per-action authority distinct.

## Pocket-provenance acceptance — 2026-09-05

CP-04/05/08: [pocket provenance](../pocket-provenance.md) requires durable session/decision binding beyond the current in-memory latest-match resolver. Validate resource-authorised phone lookup after eviction and restart, with original uncertainty/evidence and an applied/rejected receipt. Reference syntax alone does not close this journey.

## Sensing decision navigation — 2026-09-05

The [local RuView map](sensing-decisions.md) separates 45 local decisions from vendored copies and links their existing assessments. Its generated review counts distinguish extended records from decisions still requiring dedicated review. The [sensing reading routes](../sensing-extension.md#reading-the-evidence) now cover retrieval, model evaluation, adaptation, integrity, coordination and edge lifecycle. Coverage is generated from the inventory, not inferred from chapter length.

### Sensing UI provenance acceptance — 2026-09-05

CP-01/06/09 must cover [ADR-019 source and freshness transitions](../sensing-extension.md#sensing-ui-source-and-freshness-contract). Require distinct unknown, stale, hardware and simulation states; qualified inference/geometry meaning; backend-absent startup and selected-server routing; and browser reconnect/status-race coverage. Four isolated service assertions establish current behaviour only.

### Survivor lifecycle acceptance — 2026-09-05

CP-01/05/06/09 require [ADR-026 integration and transition evidence](../sensing-extension.md#survivor-tracking-and-operational-integration): authoritative confirmed counts, assignment objective, identity across loss, observation time/zone boundaries, persisted five-event lifecycle and authorised rescue. Ten helper tests and two assignment assertions establish bounded source behaviour; selected MAT and estate journeys remain open.

### Mobile sensing acceptance — 2026-09-05

CP-01/05/06/08/09 must resolve the [mobile companion's 31 criteria](../mobile-sensing.md#acceptance-register), including authoritative MAT records, schema/source labels, retry/hydration, actual native/web rendering and meaningful test gates. Placeholder suites and empty Maestro files do not establish platform acceptance. Future offline inference, push, watch, BLE and multi-server increments remain separately scoped.

### Training UI acceptance — 2026-09-05

CP-01/03/06/07/08/09 must resolve [ADR-036's fourteen implementation items](../sensing-extension.md#training-ui-and-model-operation-contracts): mounted target and API contracts, dataset/teacher identity, truthful objectives/metrics, job lifecycle, profile installation and browser recording-to-inference evidence. Existing panels and standalone handler files do not close that journey.

### macOS acquisition acceptance — 2026-09-05

CP-01/06/08/09 require [ADR-025 helper/protocol and platform evidence](../sensing-extension.md#macos-helper-and-runtime-contract): versioned executable modes, parser/identity semantics, observation timing, selected-runtime dispatch and actual scan-to-UI tests. Separate Python connected-AP polling does not close the proposed Rust multi-AP path.

### Python proof acceptance — 2026-09-05

CP-01/03/06/08/09 require [ADR-011's separated replay/acquisition/inference claims](../sensing-extension.md#python-proof-replay-and-mock-boundaries), actual mock entrypoint gates, a pinned build with rejecting CI and captured-data provenance. Current synthetic replay metadata is explicit; the local replay did not pass because NumPy was absent.

### Roadmap planner acceptance — 2026-09-05

CP-01/06/07/08/09 require [ADR-038's observation, catalogue and correctness contracts](../sensing-extension.md#roadmap-planning-design-and-evidence-state) before automated dispatch. Predicted metric effects are not evidence. The deferred planner is not a prerequisite for executing this roadmap manually.

### Rust backend migration acceptance — 2026-09-05

CP-01/03/06/08/09 require [ADR-020's selected binary, feature and parity contracts](../sensing-extension.md#rust-primary-backend-migration). Library builds and ONNX declarations do not prove the documented executable, unified API or cross-target runtime. Measure the actual release and define Python retirement/rollback after parity.

### CRV facade acceptance — 2026-09-05

CP-01/03/06/07/09 require [ADR-033's 37 stage and integration criteria](../sensing-extension.md#crv-stage-facade-and-identity-evidence), selected upstream versions, real caller wiring and labelled identity evidence. A six-stage facade and a two-stage frame method do not establish the complete sensing journey.

### RuView strategy lineage acceptance — 2026-09-05

CP-01/06/07/08/09 must retain [ADR-002's surviving domains](../historical-decision-reconciliation.md#ruview-strategy-supersession-and-surviving-domains). ADR-016/017 do not establish complete storage, online learning, security, consensus, portable runtime, audit or branching delivery. Ratify explicit current adoption/deferral/retirement and acceptance for each obligation.

### Cognitive container acceptance — 2026-09-05

CP-01/04/06/07/08/09 require [ADR-003 format and durable lifecycle contracts](../sensing-extension.md#cognitive-container-contract-and-durable-lifecycle). Pin actual writer/consumer formats and establish recovery, branch/replay, authority and cross-target evidence before accepting container packaging as complete operational persistence.

### Commodity observation acceptance — 2026-09-05

CP-01/03/06/08/09 require [ADR-013 readiness, source and capability evidence](../sensing-extension.md#commodity-sensing-capabilities-and-observation-readiness). Empty/stale input must not stand in for measured absence; settle classifier semantics and deliver a captured proof bundle plus an executable installation journey before accepting commodity sensing.

### Advanced signal algorithm acceptance — 2026-09-05

CP-01/03/06/09 require [ADR-014 consumed algorithms and semantic evidence](../sensing-extension.md#signal-algorithm-semantics-and-runtime-adoption). Wire and validate selected adapters, finite inputs, timing, geometry and directional contracts before claiming research-grade or cross-environment results.

### Contrastive objective acceptance — 2026-09-05

CP-01/03/07/09 require [ADR-024 objective and consolidation evidence](../sensing-extension.md#contrastive-training-objective-and-consolidation-boundary). Settle the accepted candidate set, projection, temperature, regularisation and batch policies, then verify gradients and real fine-tuning consumers. This focused finding leaves the broader ADR assessment pending.

### Embedding state acceptance — 2026-09-05

CP-01/03/07/08/09 require [augmentation identity and projection-state invariants](../sensing-extension.md#embedding-augmentation-and-projection-state). Verify nonzero adapter output through merge/unmerge, training and export/load, and bind artifacts to actual data and augmentation provenance. ADR-024 remains under review.

### Embedding deployment acceptance — 2026-09-05

CP-01/03/06/08/09 require [quantised-model fidelity and scoped target budgets](../sensing-extension.md#embedding-quantisation-and-deployment-evidence). Vector rounding, empty rank comparisons and one-byte parameter arithmetic do not establish deployable model behaviour. Correct the metric and optional-encoder budget before target acceptance.

### Cumulative AETHER acceptance — 2026-09-05

CP-01/03/04/06/07/08/09 require the [full seven-phase ADR-024 contract](../sensing-extension.md#aether-cumulative-closeout-contract), including attainable test criteria, actual index/adaptation orchestration and explicit optional/future dispositions. Documentation review completion leaves implementation acceptance open.

## Mesh execution — 2026-09-05

A twelve-node Opus mesh executed the scaffolded acceptance conditions across ten repositories (dream-machine upstream excluded). Every node worked in its repository's working tree; nothing was committed. Receipts live in each repository under `docs/estate-closeout/2026-09-05/` (solid-pod-rs: `crates/solid-pod-rs/docs/estate-closeout/2026-09-05/`; prose-sanitiser also `docs/release-receipts/2026-09-05.*`). Each advanced record carries a dated `Acceptance progress — 2026-09-05` (or `Re-verification 2026-09-05`) section; ledger indexes were regenerated and validate in all seven ledger repositories.

| Repository | Outcome | Verification | Browser |
|---|---|---|---|
| VisionClaw | 34 operative records advanced (data/auth/vault lane 21, GPU/wire/XR lane 13); 41 new records ADR-2043–2091 for decisions taken during closeout (dead-code removal, auth-header-only sockets, compile-time wire locks, boot profile assertion); atomic provenance, in-transaction caller-role recheck, backup membership, typed mints, 0x44 co-presence wired, release-mode overflow remap, PTX version-token fix, all 11 HUD sites | server 1225 lib + xr 310 + gpu 71 + protocol 31 + 1254 data-lane; 0 failures; real nvcc 12.9 PTX check | not run (dev stack not running; forbidden in-container) |
| agentbox | 31 records advanced + ADR-2033 (deepsec Security gate) + 23 new records ADR-2034–2062; relay admission before ack, YAML ports gate, per-slot adapter deadline, privacy coverage, URL-parsed profile redirects, projector ownership ledger, cost-cap limiter, real skill-lint frontmatter, Loom cache key, vault precedence, crate licence texts, Hermes argv identity, dream-engine typed required-check gate with frozen manifest and restart-safe runs | nip98 selftest 120/0 skipped, agentbox-ops 200, nostr-pod-bridge 117, dream-engine 150, 60+39+56 config/security/contract specs, 126-skill lint, licensing 8/8 | n/a |
| solid-pod-rs | ADR-2002–2007 closed in-repo: typed `PolicyOutcome` (invalid never inherits; drifted second resolver copy removed), replay store never evicts unexpired entries, provenance receipt stages, audience-keyed cache control, mempool manifest, OIDC matrix | 1801 tests, clippy/fmt clean, core feature surface builds | n/a |
| VisionFlow | ADR-2001–2007 + engineering ADR-004/005: asset inventory + build receipt, six blocking deploy gates, drift counter repaired and pinned (ADR-2005 → complete), release roster 6 → 14 with provenance, harness audit de-duplicated (82.5 % → honest 79.5 %) | 103 gate assertions in 4 suites | 26/26 CDP checks, real Chrome diagram gate 10/10 zero drift, 9 screenshots |
| dreamlab-ai-website | ADR-2002–2008 advanced: exact kit pins + lockfile/checksum parity, 36 Actions to full SHAs, real gate aggregation, Vitest gates deploy, 7 config mirrors / 20 sites, live zone-config drift fixed, Talk-to-AI request-id binding | Vitest 191, cargo 21, eslint 0 errors | 7 checks / 8 verdicts pass, 7 screenshots (serviceWorker stub documented) |
| loom | ADR-135/137 acceptance met in-repo, 136/138 advanced: immutable bundle activation with one serving identity, artefact qualification from effective RuVector settings (Euclidean/wrong-width rejected), 14-field grounding contract on all six statuses, build receipt | 278 tests, clippy clean | local façade via sidecar + curl, drift injection receipt |
| nostr-rust-forum | ADR-2003/2004/2006/090/092 closed, 2010 advanced, 091 partial: keyset trust-sweep pagination fixed and proven atomic against real local D1, receipt stages, shared identity vector set, badge freshness states | 1823 tests, ~140 new; worker-build + trunk build | wrangler dev + Trunk client via sidecar, 5 screenshots |
| knowledgeGraph | ADR-2001–2004 closed: census, strict flags, inference-visibility policy, blocking build, 8,138-IRI identity tripwire, export manifest, explorer fixture, ADR-NG-001 disposition | 85 tests; strict build reproduces 8,138 / 265,796 / 101,321 | SPA served via sidecar, 4 screenshots; EXP-01 consumer stall recorded for WasmVOWL |
| prose-sanitiser | ADR-2030 crate side: `missing_docs` on all 7 crates, dead licence links in every published archive fixed, metadata complete, release receipt with reproduced digests and proven extraction lineage | 767 tests, clippy, doc -D warnings, cargo deny | n/a |
| dream-engine | ADR-2024 all six conditions implemented (see agentbox row) | 150 tests | n/a |

**Findings escalated for owner decision** (each recorded in the repository receipt): solid-pod-rs does not validate token `aud` and silently substitutes `sub` for a malformed `webid`; nostr-rust-forum's Rust validates derived subkeys as scalars while agentbox JS does not, and the badge panel is defeated by the relay client firing EOSE while disconnected; the production Loom corpus lacks an `embeddingModel` stamp and HP serves split lexical/semantic generations; opening the RVDB artefact rewrites its metadata; VisionClaw's browser decoder auto-detected V2 on foreign opcodes (guarded); WasmVOWL `/graph` requests neither WASM nor the binary tier; agentbox's baked dream evaluators now all read as `required`.

**Ledger discipline.** Records whose governed paths changed in the uncommitted tree were re-verified and anchored at HEAD with their `verified_paths` kept (agentbox ADR-2058, integration amendment): the validator will report them STALE at the landing commit until `verified_commit` is moved to that commit. ADR-2033's `nodeModulesHash` is resolved by the first host image build.

**Still open after this pass:** everything that needs hardware, a headset, hosted CI, production deployments or a human signer; the WasmVOWL consumer stall; the RuView records classified evidence-review-required; publication of prose-sanitiser 0.1.2.

### Learning / memory lane addendum — 2026-09-05

agentbox ADR-2014–2019 and 2025–2027 advanced: fail-closed memory store (no `COALESCE` vector retention, value↔vector digests, TTL on every read path), two-phase trajectory redactor with a stated retention policy and a durable 200-event overflow queue, canonical top-level `failure_mode`, attributable-outcome promotion, the producer-before-consumer invariant chosen as *qualified retained corpus* and enforced at runtime (W066 → E066, new E067), recall runs bound to model fingerprint/corpus revision with a consuming gate, effective-model fingerprint pinned at boot, versioned two-language federation fixture (bead crossing ratified), shared egress policy with redaction before egress and a durable provenance archive, break-glass expiry/scope/audit, and a new `services/secret-backup` crate (tar-in-age, restore exercised on synthetic data; not yet in `flake.nix`). Tests: sovereign jest 646 pass / 3 skip, nostr-pod-bridge 126, nip98 selftest 134/0/0, federation 35.

**Escalated:** a protocol-conformant median-of-three recall run against the deployed RuVector corpus gives self-recall 164/200 (band ≥175) and true-recall 96/120 (band ≥102); the new gate refuses (exit 3). This is the current retrieval geometry, previously unmeasured; one production row awaits embedding repair; index degradation under bulk churn is the documented suspect (non-concurrent HNSW rebuild is the recorded remedy).
