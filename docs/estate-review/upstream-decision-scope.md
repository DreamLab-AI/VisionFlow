---
title: Upstream decisions and consumed implementation identity
status: source-and-lineage-reviewed
date: 2026-09-05
type: explanation
---

# Upstream decisions and consumed implementation identity

RuVector and RuView belong in the assessment because the estate consumes their capabilities. That does not make every upstream research proposal, example application or copied ADR an adopted estate decision. The [pack map](closeout/upstream-packs.md) routes 393 decision records across 17 families, including four previously extended RuView records. It preserves every record's current review state. Grouping alone does not complete semantic review.

## The consumer selects the relevant source

Loom's Cargo.toml explicitly depends on the sibling ruvector/crates/ruvector-core path with default features disabled and hnsw/storage/simd/parallel selected. The [consumed vector review](consumed-vector-storage.md) therefore has a direct source relationship to that checkout. Other RuVector crates and examples do not become Loom dependencies merely by sharing the repository.

RuView's Rust workspace declares published RuVector dependencies. Its checked-in lockfile records registry sources and checksums for nine package/version entries, including two attention versions. The integration crate uses the workspace dependencies; its CRV/GNN dependencies are optional under the crv feature. Signal processing directly declares mincut, attention and solver dependencies; MAT gates solver/temporal dependencies behind a ruvector feature enabled in its defaults. These declarations identify possible and default build paths, not proof that every runtime executes them.

The [identity receipt](evidence/upstream-pack-map.json) captures those manifests and lock entries. The inspected workspace manifests do not select vendor/ruvector as their dependency source. Consequently, reviewing that copied tree cannot by itself verify the published crate implementation RuView resolves. No cargo resolution, build, registry download, deployed inspection or hardware run occurred in this pass. Build-time configuration overrides and the actual release artefact require their own evidence.

The [shared-memory review](shared-memory.md) is a separate integration: its MCP wrapper and database/embedding contracts must not inherit guarantees from an unrelated RVF or quantum-engine ADR. The [catalog review](catalog-decisions.md) likewise distinguishes recommendations from actual adoption.

## Copies preserve different review histories

The refreshed lineage comparison still finds 176 vendored candidates: 161 byte-identical and 15 divergent. One candidate is an ADR-authoring agent definition, now classified as support in both locations. RuView's own copy of that agent definition is also support. These are role/instruction files with name, type, capabilities and hooks, not decisions.

The identity receipt retains full diffs for all 15 divergent pairs. Six changes disambiguate duplicate numbers 017, 029 and 031. Others change status or qualify claims: the stale security-debt tracker, cognitive-container wire-format scope, causal-atlas demonstrator, software-only TEE account, PostgreSQL extension scope, Lean integration, federation scope and solver routing limitations. These are meaningful differences in review history, not proof that the underlying standalone and vendored source implementations differ in the same ways.

Preserve both copies and qualify references by repository and full path. Do not propagate a July status correction into the vendor tree as if its code had been verified, and do not prefer an older unqualified copy as evidence of broader capability. Every adoption decision needs the consumed package/revision, selected features and consumer-path evidence.

## Review order for the remaining packs

| Priority | Families | Evidence needed to resolve estate relevance |
|---|---|---|
| 1 | RuView main and Rust-port decisions; RuVector core/RVF/storage decisions | Map each claim to the sensing, memory or Loom consumer and exact artefact; reconcile simulated, trained and hardware evidence |
| 2 | Mincut, solver research, temporal tensor and relevant top-level integration records | Trace called algorithms and feature gates in the registry packages RuView actually resolves; independent algorithm existence is insufficient |
| 3 | Coherence, delta behaviour and quantum-engine decisions | Establish an actual estate caller or retain explicitly unadopted upstream design scope; assess any public benefit claims separately |
| 4 | DNA, prime-radiant, delta-behaviour examples, vibecast, OSpipe and ruvbot | Separate demonstrations and independent applications from deployed estate functionality; document any real dependency or shared obligation |

This order does not remove lower-priority records from the user-requested corpus review. Their section-level dispositions remain open. An unadopted proposal can be fully assessed as such only after its scope and relevant claims have been checked; a directory label is not enough.

CP-01/03/06/07/08/09 requires a per-record adoption/disposition map, consumed-source identity, concrete acceptance predicates and retained limitations. The practical system vision benefits from reusable components, but reuse must follow the executable dependency and consumer boundary rather than the widest upstream catalogue of promises.
