---
title: Catalog recommendations and decision scope
status: source-reviewed
date: 2026-09-05
type: explanation
---

# Catalog recommendations and decision scope

Agentbox's RuVector catalog is an implemented recommendation aid with a lean instructional entry point and a separate TypeScript search path. Its nine ADRs govern that local skill, not the entire RuVector library or estate adoption of every advertised capability. The useful architectural idea is to organise capabilities around user problems. Closeout must distinguish an available description, a ranked recommendation and an integration demonstrated to solve the user's problem.

The [source snapshot](evidence/catalog-decision-snapshot.json) records current files and package omissions. This pass inspected code and documents; it ran no search, model, swarm, build or benchmark. Historical scores and timings remain historical claims.

## The entry point has changed

The current SKILL.md is 72 lines and 3,448 bytes. It explicitly calls itself a lean guide, links the problem map and audience guidance in references/capability-map.md, and points to five domain overlays. This differs from ADR-001/002's proposed full-context, single-file catalogue and BENCHMARK-RESULTS.md's 30.2KB/21-section interface description. Splitting references is a reasonable design choice, but the load path and context budget need an explicit amendment. The presence of linked material does not establish which material an agent actually read.

The inspected entry point and capability map do not contain the proposed prominent “What RuVector Does NOT Do” section. The TypeScript classifier does implement scope checks, with a matched problem section able to prevent an out-of-scope rejection, and meta-query handling preceding that check. These are separate interfaces: the CLI guard does not automatically govern an agent reading Markdown. Its confidence values are branch-assigned constants and heuristics, not calibrated probability estimates.

Five industry overlay files exist. Audience guidance exists in the reference map, and CLI/proposal code conditionally includes plain descriptions. These are implemented supporting mechanisms. They do not establish recommendation accuracy, readability or validated applicability of the described technology to a particular industry.

## Search is sparse, but the ranking contract differs

DiscoveryService builds full-vocabulary sparse TF-IDF vectors and scans its documents directly. It does not use the HNSW stage still named in ADR-007. It scores documents before the vertical filtering step rather than restricting retrieval to matched capability domains first.

The threshold is 0.15, not the ADR's 0.25. Field weighting repeats text before sublinear term-frequency calculation; a crate weight of 0.5 rounds up to one repetition. This differs from the ADR's weighted sum of separately embedded fields. Reranking adds 0.1 for a primary crate, 0.05 for production status and 0.05 for keyword overlap. The ADR instead specifies multiplicative primary/status boosts and an exact-term bonus.

Vertical handling can insert previously unscored technologies with a minimum base score of 0.01, then apply bonuses and the final threshold. Thus returned scores include policy boosts; they are not raw cosine similarities or calibrated confidence. Deduplication follows score sorting, so a comment claiming preference for technology documents does not establish an explicit type-priority rule. These are source observations, not measured ranking failures.

The declared enriched Technology and Capability fields exist. CatalogRepository imports TypeScript data and builds maps; this checkout has no catalog.json or scripts/build-catalog.ts despite package references to them. The proposed shared JSON/regeneration contract is therefore not established by the current package. Field presence also does not establish data freshness or exhaustive enrichment.

## Proposals and benchmarks need separate evidence

ProposalService constructs an RVBP from supplied search results and catalogue data. The inspected skill entry point and proposal generator do not establish ADR-006's multi-agent escalation executor, trigger enforcement, independent source review or budget accounting. An external orchestrator could provide those capabilities; this bounded assessment does not claim none exists anywhere.

ADR-009 distinguishes predicted V3 outcomes from earlier measurements and acknowledges that Q5 was not run in those earlier benchmarks. BENCHMARK-RESULTS.md later labels all earlier comparison values measured and claims 168 passing tests. This checkout has no referenced tests directory, and Bun is unavailable on PATH. The report cannot serve as a current reproducible test receipt. CLI internal search timing, process startup and an LLM's full response latency are different measurements; eliminating a search subprocess does not establish the proposed under-10ms conversational result.

## Record-by-record disposition

All nine records remain locally scoped. Preserve ADR-001–008's proposed status and ADR-009's accepted benchmark decision; neither status establishes implementation completion or estate ratification.

| Local record | Current disposition | Closeout requirement |
|---|---|---|
| ADR-001 CAG architecture | Partial; current entry point delegates to references | Amend loading contract and measure actual context/recommendation path |
| ADR-002 intent headers | Problem map exists in a reference; generation contract unverified | Reconcile section coverage, source data and regeneration |
| ADR-003 scope boundary | CLI guard exists; proposed prominent Markdown section absent | Define and test scope behaviour separately for each interface, including mixed queries |
| ADR-004 vertical overlays | Five overlays and structured mappings exist | Verify mapping freshness, domain coverage and recommendation evidence |
| ADR-005 audience adaptation | Instructional and conditional output support exists | Evaluate complete responses with intended readers and retain technical traceability |
| ADR-006 escalation | Proposal construction exists; local executor not established | Supply executor/trigger/source/budget receipts or explicitly defer orchestration |
| ADR-007 CLI search | Sparse search exists with changed threshold, weighting and routing | Ratify actual algorithm or amend implementation; calibrate held-out positive and negative queries |
| ADR-008 data model | Enriched types/data exist; shared JSON generation path absent here | Establish authoritative data source, validation, completeness and reproducible generation |
| ADR-009 benchmark comparison | Accepted evaluation design; current reproduction incomplete | Restore identified fixtures/harness and revision-bound results; separate predictions and distinct latency scopes |

CP-01/03/07/08/09 requires owner disposition of these differences and a reproducible recommendation-to-integration evidence chain. The catalog's recommendations cannot substitute for the [consumed vector implementation review](consumed-vector-storage.md) or [shared-memory review](shared-memory.md).

## BHIL material is reusable support

The three ADR files under agentbox skills/bhil-methodology/templates contain placeholder IDs, dates and titles. The model-selection record under examples is a worked teaching example in that skill's example corpus. Classify these four records as support, preserving their contents. Its accepted example status and numerical model evaluation claims do not constitute an estate production model-selection decision. Any adoption needs a separately identified project decision and evidence for the actual candidate, workload and revision. This classification makes no recommendation about current model choice or pricing.
