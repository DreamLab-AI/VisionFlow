---
id: ADR-2022
title: Hold one bi-temporal contract for ontology history across the estate, and show history in visualisation through as-of projections, never a versioned reasoning graph
date: 2026-11-03
decision_status: proposed
implementation_status: none
activation_status: inactive
supersedes: []
superseded_by: []
depends_on: [ADR-2013, ADR-2014, ADR-2015]
verified_commit: 307cdc5
verified_paths: []
owner: jjohare
review_trigger: Reopen when any of these happens — ADR-2015's typed model fixes its temporal fields; federation (VisionClaw VC-48) starts implementing leases or tombstones; a graph client asks for a time view; or VisionClaw ADR-2102 (erasure) is decided
repo: visionflow
domain: BASELINE-visionflow.md
evidence_class: source
acceptance_boundary: A source reading of VisionClaw afe3cb574 and of adrianco/the-goodies (README as of 2026-11-02). It establishes what the bi-temporal provenance code provides and what it does not cover. It does not establish that any proposal below is wanted, sized or safe under erasure. Every proposal here is an idea to come back to, not a commitment.
---

# ADR-2022 — Hold one bi-temporal contract for ontology history across the estate, and show history in visualisation through as-of projections, never a versioned reasoning graph

## Context

This review was prompted by [adrianco/the-goodies](https://github.com/adrianco/the-goodies). It is a knowledge graph for a house, and it never overwrites anything. Entities are stored as immutable `(id, version)` rows. Edges are half-open `[valid_from, valid_to)` intervals. Valid time (when an edit was made) is kept separate from transaction time (`server_seq`, when the server learned of it). A write that loses a conflict is still stored and acknowledged, and replicas compare a sha256 digest to prove they hold the same data. Its `snapshot(T)` and `diff(T1,T2)` are marked "designed, not built".

VisionClaw already has the core of that model. `provenance_writer.rs:15-22` makes `urn:ngm:graph:provenance` the append-only historical source of truth, and makes the asserted graph its `state_at(now)` projection. Retraction closes `[validFrom, validTo)` and deletes nothing. Recorded time is held separately as `prov:generatedAtTime`. `GET /api/ontology/state-at?t=&recorded_as_of=` answers both questions (`handlers/api_handler/ontology/mod.rs:1370-1420`, `:1686`).

There are gaps. The only production writer of assertion-versions is `decision_service.rs:595`. The authored corpus reaches `:assert` through a `CLEAR+INSERT` rebuild (`github_sync_service.rs:969`) that writes no versions, so `state-at` sees governed decisions but not the history of the corpus itself. No graph client uses `state-at`. And neither ontomark nor federation has adopted the contract.

## Decision

**Proposed, not yet adopted.** Five ideas, recorded so they can be picked up one at a time.

**P1: one temporal contract.** Make VisionClaw's model (legacy ADR-049, PRD-022 WS-2) the estate contract for any time-qualified ontology fact:
- valid time is `dl:validFrom` / `dl:validTo`, read as the half-open interval `[from, to)`;
- recorded time is `prov:generatedAtTime`;
- queries in valid time never read recorded time, and replication never reads valid time.

ADR-2015's typed model and ontomark's graph contracts would reuse these names and meanings. Federation (VC-48 leases, retraction tombstones) would express expiry and retraction as `validTo` on the same model, not as a second scheme.

**P2: corpus history in the as-of view.** Let `state-at` answer for authored content as well as governed decisions. Two candidate ways, not yet chosen:
- (a) project the visionGraph vault at the last commit at or before T, using commit time as recorded time;
- (b) have the sync emit assertion-versions from the diff between the previous and the new `:assert` graph.

Option (a) adds no stored state. Option (b) puts corpus history in the same graph as decisions.

**P3: a time view in visualisation.** Add a time slider to the graph clients, served by `state-at` (P2 for content, provenance for decisions). Reasoning runs only on a single-instant projection. A reasoned view at T means rebuilding the projection at T and saturating that. Whelk is never run over the union of versions, because OWL has no time semantics: a class that moved from parent A to parent B would end up under both.

**P4: convergence digests.** Check agreement instead of assuming it. Each projection (vault → Oxigraph `:assert`, server → graph client, and later each federation peer) publishes a sha256 over the sorted current `(IRI, content hash)` set. A mismatch triggers a full rebuild and is reported. visionGraph's publish marker already carries content and ontology digests, which is the starting point.

**P5: losing writes are kept.** If federation or offline clients ever allow concurrent writers, a write that loses resolution is stored as a non-current version and acknowledged, never dropped. This is the-goodies' "a concurrent write can lose prominence; it can never lose existence".

**Out of scope.** Versioning the Oxigraph `:assert` graph, the GPU graph state or RuVector memory. These stay disposable projections that can be rebuilt. Authored Markdown in git, plus the provenance graph, remains where history lives.

## Consequences

- P1 costs nothing until a second consumer needs time. It stops ontomark and federation from inventing incompatible interval or tombstone semantics, the drift pattern ADR-2014 exists to remove.
- P2 widens what the append-only store holds, and so widens the erasure surface. Under option (b), any personal data in authored content inherits the problem VisionClaw ADR-2102 (crypto-shred or declare out of scope) has not yet resolved. Option (a) relies on git history, which has its own erasure limits (history rewrite).
- P3 needs P2 to be useful for content. Rebuilding a projection at T costs a sync and a saturation per distinct T, so it suits stepping through a slider, not continuous scrubbing, unless snapshots are cached.
- P4 would have caught the PATH B / PATH A materialiser drift (VC-20.4) mechanically rather than by review.
- P5 only matters once there are multiple writers; today, governed proposals serialise writes.
- Supersedes nothing. Changes none of the invariants in `BASELINE-visionflow.md`. Implementations would belong to VisionClaw, ontomark and visionGraph (ADR-2006).

## Verification

Nothing is implemented. The facts in Context were read at VisionClaw `afe3cb574`:
- `src/services/provenance_writer.rs:15-60`: interval, two clocks, `state_at` at `:506-518`;
- `src/handlers/api_handler/ontology/mod.rs:1370-1420`, `:1686`: the `state-at` route;
- `grep -rn build_assertion_version src crates`: the only caller outside the writer is `src/services/decision_service.rs:595`;
- `src/services/github_sync_service.rs:969`: `rebuild_assert_graph`.

the-goodies features and their status come from its README, fetched 2026-11-03.
