---
id: ADR-2022
title: Hold one bi-temporal contract for ontology history across the estate, and show history in visualisation through as-of projections, never a versioned reasoning graph
date: 2026-11-03
decision_status: accepted
implementation_status: partial
activation_status: staged
supersedes: []
superseded_by: []
depends_on: [ADR-2013, ADR-2014, ADR-2015]
verified_commit: ab1e636ff
verified_paths: [visionclaw:src/services/corpus_history.rs, visionclaw:src/services/temporal_contract_tests.rs, visionclaw:src/services/github_sync_service.rs, visionclaw:src/handlers/api_handler/ontology/mod.rs, visionclaw:crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs, visionclaw:client/src/store/timelineStore.ts, visionclaw:client/src/features/graph/components/GemNodes.tsx, visionclaw:client/src/features/graph/utils/timelineTint.ts, ontomark:conformance/temporal/README.md, ontomark:conformance/temporal/vectors.json]
owner: jjohare
review_trigger: Activation — a deployed VisionClaw built from ab1e636ff or later, with the panel key present, runs an operator force_full sync, and `GET /api/ontology/state-at?t=<now>` returns versions owned by the corpus writer (an activity with the legacy `urn:visionclaw:execution:corpus-sync-` prefix, or one carrying `dl:writer "corpus-sync"`), with no assert_digest or corpus_history errors in that sync. Reopen P5 on the first multi-writer path (federation write-back or offline clients). Reopen the erasure boundary when VisionClaw ADR-2102 is decided. estate-temporal-v2 (Amendment 2026-11-25) moves implementation_status back to complete when ontomark's v2 contract files and VisionClaw's v2 writers, reader and refit tool are merged, and activation_status to live when the operator's refit of the deployed store passes its verify step.
repo: visionflow
domain: BASELINE-visionflow.md
evidence_class: source+test
acceptance_boundary: Established at VisionClaw ab1e636ff and ontomark 28330ce by unit and repository-level tests in in-memory Oxigraph stores and by client vitest. Not established — no deployed instance has run the new sync, so production volume, state-at latency over the full corpus and the panel key's presence in the deployed environment are unmeasured. P1 adoption by ADR-2015 and VC-48 is not established; those records are unimplemented. estate-temporal-v2 (Amendment 2026-11-25) is decided and not established: its ontomark and VisionClaw implementations are on branches, not merged, so the boundary above describes estate-temporal-v1 only.
---

# ADR-2022 — Hold one bi-temporal contract for ontology history across the estate, and show history in visualisation through as-of projections, never a versioned reasoning graph

## Context

This decision was prompted by [adrianco/the-goodies](https://github.com/adrianco/the-goodies), a knowledge graph for a house that never overwrites anything:
- entities are immutable `(id, version)` rows, and edges are half-open `[valid_from, valid_to)` intervals;
- valid time is kept apart from transaction time;
- a write that loses a conflict is kept;
- replicas compare a sha256 digest to prove they hold the same data.

VisionClaw already had the core of that model. The append-only `urn:ngm:graph:provenance` holds assertion-versions with `[dl:validFrom, dl:validTo)` and a separate `prov:generatedAtTime` (`provenance_writer.rs`, legacy ADR-049). `GET /api/ontology/state-at` answers both "true at T" and "known at T". A client `TimelineScrubber` called it.

The gaps were:
- only governed decisions wrote versions;
- the authored corpus reached `:assert` through a `CLEAR+INSERT` rebuild with no history;
- the scrubber's overlay had no render caller, and matched node ids against IRI subjects, so it could never fire;
- nothing checked that the rebuild landed;
- no other consumer shared the contract.

## Decision

**P1: one temporal contract.** *(Amended 2026-11-25 for `estate-temporal-v2`; the v1 text is quoted in that amendment below. v2 is decided and being implemented; what VisionClaw and ontomark run today is `estate-temporal-v1`.)* `estate-temporal-v2` is the estate's meaning of a time-qualified ontology fact. It has one form:
- a **version** is a `prov:Entity` with `rdf:subject` / `rdf:predicate` / `rdf:object`, `dl:validFrom`, `prov:generatedAtTime` (recorded time), `prov:wasGeneratedBy` and `prov:wasAttributedTo`. A version never carries an end; `dl:validTo` is not used in the provenance graph;
- an end is a separate **closure record**, `dl:Closure`, with `dl:closesVersion`, `dl:closes` naming one or both clocks (`memory:ValidTime`, `memory:RecordedTime`, from eljeffeg/memory-ontology, CC BY 4.0), `dl:validUntil` if and only if it closes valid time, its own `prov:generatedAtTime`, `prov:wasGeneratedBy` and `prov:wasAttributedTo`, and an optional `dl:reason`. A closure closes at least one clock, is not a version, and is never closed;
- the **state rule**: a version is in the state at valid time t as known at recorded time r if and only if it was generated at or before r, its `validFrom` is at or before t, no closure on it closes recorded time with a recorded time at or before r, and no closure on it closes valid time with a recorded time at or before r and a `validUntil` at or before t. Without r, every closure applies. Several valid-time closures each exclude independently, so the effective end is the earliest, and that is the `validTo` an API row reports;
- **version identity** hashes the recorded time as well as subject, predicate, object, `validFrom` and agent, so a re-assertion or a reinstating version never collides with an earlier one. A wrong correction is undone by asserting a new version. A closure's IRI is content-addressed over its version, clocks, `validUntil` and recorded time, under the agent;
- nothing is deleted at run time, as-of queries never read the replication order, and a reasoned view at T is a projection at T that is then saturated.

ontomark owns the spec and the vectors (`conformance/temporal/`; the v2 spec and vectors are being added on an ontomark branch, not yet on main). Consumers vendor the vectors, pinned by SHA-256. VisionClaw is the reference implementation (its v2 record is VisionClaw ADR-2219, planned). A record that adds time to ontology facts (ADR-2015's typed model) cites this contract instead of defining its own. Federation tombstones (VC-48) are not temporal facts: they evict received copies from a cache, and are outside the contract. Erasure is VisionClaw ADR-2102's.

**P2: corpus history, recorded from the sync's diff (option b).** On each full sync, the public corpus IRI triples are diffed against the corpus writer's *own* open versions: those generated by an activity carrying `dl:writer "corpus-sync"` *(corrected 2026-11-25: this line named the `corpus-sync-…` activity prefix, which stopped identifying the writer at VisionClaw `adf382ea4`; see the 2026-11-07 amendment. The legacy prefix branch still exists in the reader at VisionClaw main, matches no activity in the production store, and is removed under v2)*. The change is appended to the provenance graph in one transaction, attributed to VisionClaw's panel identity (`src/services/corpus_history.rs`). The rules:
- Diffing against the writer's own versions, not against `:assert`, means decision instances and governed mutations are never closed or duplicated, and an unchanged re-sync writes nothing.
- `save_ontology_graph` and the history writer share one IRI-triple projection, `ontology_graph_iri_triples`, so they cannot drift.
- Only pages marked `public: true` are recorded (9,151 of the 9,163 class pages in visionGraph at 6267d0dd0), and only IRI triples. Nothing private enters a store that cannot yet delete.
- Neither corpus source supplies a commit time, so valid time and recorded time are both the sync instant. History starts at the first full sync with the writer enabled, at full-sync granularity.
- A missing panel key is reported as "history NOT recorded" with its reason. A misconfigured key is a sync error. `VISIONCLAW_CORPUS_HISTORY=0` stops the writer.
- *Under v2 (decided 2026-11-25, being implemented):* the sync ends a version by writing a closure record linked to its own sync activity and agent, "open" means a version with no closure, and the writer guards against two syncs in the same second.
- *Planned amendment (2026-11-25; not implemented):* when a page's `supersedes` entry or withdrawal explains why its public triples left the corpus, the corpus sync applies the authored clock instead of its default valid-time close. `corrected` or `withdrawn` closes recorded time, and `merged` or `renamed` records no close. `world-changed`, and any disappearance with no authored reason, closes valid time. A page left `public: true` with `status: deprecated` gets no close. The page vocabulary is visionGraph ADR-VG-010 §3 and §7 (proposed).

Option (a), rebuilding from the vault's git history, was not chosen. The deployed corpus source is not guaranteed to be a git checkout, and option (b) puts corpus history in the same graph and query path as decisions.

**P3: a time view in visualisation.** The existing scrubber now reaches the render:
- `GemNodes` tints each node by `nodeTimelineStatus`: faded when not valid at T, green when added, red when retracted.
- Nodes now match on class IRI as well as node id.
- The store fetches `state-at?view=subjects`, one row per subject rather than one per version.

The scrubber stays opt-in through `provenance.enableTimeline`. The overlay works per node: a class whose parent moves keeps its type triples, so it reads as valid throughout. An edge move is visible through `state-at` but is not drawn on the graph. Reasoning only ever runs on a single-instant projection.

**P4: convergence digests between source and projection.** After each assert rebuild:
- the SHA-256 of the corpus IRI triples the rebuild meant to write is compared with that of the triples `:assert` holds for those subjects and predicates;
- after history is recorded, the open corpus versions are compared with the public set (under v2, a version is open when it has no closure).

A mismatch is a sync error (`assert_digest_mismatch`, `corpus_history_digest_mismatch`), and an operator `force_full` is the repair. `SyncStatistics.assert_digest` records the matching digest. Server-to-client and federation-peer digests are not built; they become relevant when those copies hold triples rather than render nodes.

**P5: keep losing concurrent writes. Withdrawn.** No path today has concurrent writers: governed proposals serialise writes. It reopens with the first multi-writer path.

**Out of scope.** Versioning the `:assert` or `:inferred` graphs, GPU graph state or RuVector memory. These stay disposable projections that can be rebuilt.

## Consequences

- Every full sync now appends to a graph that cannot shrink. On the first enabled sync this is roughly one version per public class type triple plus one per class relation, about 9 quads each. After that it is proportional to what changed. Volume and `state-at` latency over the full corpus have not been measured on a deployed store.
- The erasure surface grows by public class IRIs and relations only. VisionClaw ADR-2102 still owns erasure, and `docs/DATA-authority-erasure.md` (VisionClaw) records the boundary.
- `/api/ontology/state-at` gains an optional `view=subjects`. The default response is unchanged.
- P4 checks the asserted projection only. Drift between the inferred-edge materialisers (VC-20.4 PATH A and PATH B) is outside it.
- Diagram drift for the next re-stamp: VisionClaw VC-22.4 says "two producers" for the provenance graph, and there are now three. Line citations into `github_sync_service.rs` after the rebuild function have shifted. Both diagrams are pinned to a revision, so nothing is broken yet.

## Verification

At VisionClaw `ab1e636ff` (merged to main), built and tested in a worktree with `CARGO_TARGET_DIR` shared:

- `cargo test --lib -- corpus_history temporal_contract github_sync_service provenance_writer api_handler::ontology`: all pass. That includes 10 corpus-history tests (first sync, idempotent re-sync, one edge move closing one version and opening one with `state_at` on both sides, other writers untouched, duplicates collapsed, digests, kill switch, and three through `OxigraphOntologyRepository`: clean rebuild, stray triple gives a mismatch, no or bad identity) and 4 contract tests (pin, pure, SPARQL, subjects view).
- `cargo test -p visionclaw-adapters --lib`: 78 pass.
- `cargo test --lib`: 1,820 pass, 1 fails. The failure, `prod_lan_headset_wss_authenticate_validates_through_nginx`, parses `nginx.production.conf`, which this change does not touch, so it fails identically on the prior main.
- Client: `vitest run src/features/graph src/store` gives 38 files and 359 tests passing, including `timelineTint.test.ts` (6). `tsc --noEmit` reports only the existing `GlassEdgeMaterial.ts` error, also present on prior main. ESLint finds no new warnings in the changed files.

At ontomark `28330ce` (merged to main): `conformance/temporal/vectors.json` SHA-256 `6973aeef…6f75` matches VisionClaw's vendored pin.

Activation is `staged`. No host build or force_full sync has been run, because the first enabled full sync writes history that cannot be removed. The review trigger names the runtime check that moves this to `live`.

## Amendment (2026-11-07): corpus-writer ownership is prefix or marker

VisionClaw `adf382ea4` builds new corpus-sync activity ids through the canonical
`uri::execution` constructor, so they now parse as `Execution`. Every canonical
execution id begins `urn:visionclaw:execution:sha256-12-` whichever writer made
it, so the `corpus-sync-` prefix in P2 no longer identifies the corpus writer
on its own. Ownership is now: the legacy `corpus-sync-` prefix (versions already
in the append-only graph) **or** a `dl:writer "corpus-sync"` quad written with
each new activity (`src/services/corpus_history.rs`, writer lookup). P2's rule
("diffed against the corpus writer's own open versions") is unchanged; only the
test for "own" widened. The `review_trigger` is corrected to match. Tests:
`legacy_and_canonical_versions_are_owned_together`,
`a_canonical_activity_from_another_writer_is_not_owned`.

## Amendment (2026-11-25): estate-temporal-v2, single form

Decided 2026-11-25 by the operator's estate plan of that date; being implemented
on ontomark and VisionClaw branches, not merged. `implementation_status` moves
from `complete` to `partial` because P1 now states a contract no merged code
implements; the verification above remains the evidence for v1.

**Why.** v1 ends a fact by adding `dl:validTo` to the version. That added quad
has no recorded time of its own, so "what did we believe at r" cannot exclude a
close recorded after r, and v1 has no way to say a fact was never true (a
recorded-time correction). memory-ontology (eljeffeg, CC BY 4.0) separates the
two clocks the same way; v2 adopts its clock concepts by IRI and keeps its own
closure resource, because `memory:closes` is legal only on memory records.

**P1 as it read before this amendment:**

> `estate-temporal-v1` is the estate's meaning of a time-qualified ontology fact:
> valid time is `dl:validFrom` / `dl:validTo` as the half-open `[from, to)`;
> recorded time is `prov:generatedAtTime`; retraction appends a `validTo` and
> deletes nothing; as-of queries never read the replication order, and a reasoned
> view at T is a projection at T that is then saturated. [...] A record that adds
> time to ontology facts (ADR-2015's typed model, VC-48's leases and tombstones)
> cites this contract instead of defining its own.

**What changed in P1.** It cites v2: closure records, the state rule, one form
(no reader or writer of bare `dl:validTo` closes remains once VisionClaw's refit
has run), and version identity that includes recorded time. The tombstone clause
is removed: a federation tombstone evicts a received copy and is not a temporal
fact (VisionClaw ADR-2102 owns erasure). `prov:invalidatedAtTime` is not used.

**What changed in P2.** Line 50 named the `corpus-sync-…` activity prefix as the
test for the writer's own versions; since the 2026-11-07 amendment the test is the
`dl:writer "corpus-sync"` marker, and the line now says so. P2 also records how the
sync ends versions under v2, and the planned amendment under which the sync honours
an authored clock from the page. That part is planned and not implemented.

**P4.** "Open corpus versions" means versions with no closure under v2.

**Not changed.** P3 (the client keeps reading a reported `validTo` per row, now the
effective end), P5, the out-of-scope list, and the v1 verification record.

**Activation.** v2 goes live only after the operator runs VisionClaw's offline
refit on a checkpoint copy and then on the stopped live store, and its verify step
passes. A v2 binary disables corpus history while any `dl:validTo` quad remains in
the provenance graph (a read-only boot guard, part of the v2 work), and rolling back
after the refit is a checkpoint restore.
