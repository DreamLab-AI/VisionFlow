---
title: Consumed vector storage and semantic artefact contracts
status: source-and-local-probe-verified
date: 2026-09-04
type: explanation
---

# Consumed vector storage and semantic artefact contracts

Loom consumes the local `ruvector-core` path with `hnsw`, `storage`, `simd` and `parallel`, and disables default features. This is a concrete subset of the wider RuVector project. The shared-memory MCP's PostgreSQL path and the local vector artefact are different storage systems; evidence from one must not certify the other.

The [receipt](evidence/loom-vector-config-probe.json) pins the consumer and library source and records the synthetic local probe. Reference-memory search was used for orientation; conclusions below come from the checked-out source and executed consumer.

## Persisted configuration controls the opened database

[VectorDB::new](../../../ruvector/crates/ruvector-core/src/vector_db.rs) reads stored configuration and replaces the caller's dimensions, metric, HNSW configuration and quantisation settings with it. It then rebuilds the in-memory index from persisted vectors. That supports reopening an existing database under its own geometry, but consumers must check whether that geometry is acceptable for their model and score contract.

[Loom's adapter](../../../loom/crates/loom-vector-ruvector/src/hnsw.rs) supplies 384-dimensional cosine/HNSW defaults, opens the database and checks only that its length is nonzero. It does not validate the effective stored configuration. Query width is checked against 384, and returned distances are unconditionally converted using `clamp(1 - distance, 0, 1)` as cosine similarity. A stored flat index is also accepted by this adapter named HnswIndex.

The [probe source](evidence/loom-vector-config-probe.rs) builds three real temporary redb artefacts and queries the actual adapter, with no embedding service:

| Stored artefact | Adapter ready | Outcome |
|---|---|---|
| 384 dimensions, cosine, flat | true | Aligned vectors score 1.0 |
| 384 dimensions, Euclidean, flat | true | The same aligned directions score 0.5 under the adapter's cosine conversion |
| 3 dimensions, cosine, flat | true | A 384-dimensional query returns SemanticUnready for dimension mismatch |

This establishes a consumer configuration gap. It does not show that a deployed exporter has produced a wrong artefact, or that semantic fallback is active: that path defaults off and has additional generation/threshold checks. A readable nonempty database is weaker evidence than a compatible semantic index. Model identity also requires more than vector width.

## Local storage commits and live index updates are separate

[VectorStorage](../../../ruvector/crates/ruvector-core/src/storage.rs) uses redb transactions to store vector bytes and supplied metadata together. VectorDB inserts into storage before adding to its in-memory index; deletion likewise changes storage before the index. An index error can therefore follow a committed storage mutation. That is source ordering evidence, not an injected failure reproduction. Reopening rebuilds from storage, but does not by itself prove uninterrupted live-index consistency or crash recovery.

VectorDB searches the index for k candidates and then applies any metadata filter; it does not refill the result set after filtering. Loom currently passes `filter: None`, so this behaviour is not an established cause of lost results on its inspected path. This distinction matters when evaluating general RuVector claims against actual consumers.

The generation sidecar is read separately at open. Its timestamp and class count are not a cryptographic binding to the database, embedding model or effective metric. The [grounding review](grounding-delivery.md) documents the broader serving-generation boundary. A local vector file, sidecar and lexical bundle must identify the same accepted generation before semantic candidates are used.

## Closeout requirements

CP-01/03/07/08 require an explicit artefact contract: validate effective dimensions, metric, index configuration, quantisation, model/preprocessing identity and content digest before readiness. Reject incompatible artefacts with a diagnostic reason. Bind the sidecar to the actual database and common corpus generation. Verify read-only serving intent against the storage opener's behaviour; an existing empty path must not silently become an accepted published artefact.

Test correct and wrong configurations, missing/mismatched sidecars, changed model at equal width, interrupted export, index-update failure and reopen. Measure semantic recall through the actual fusion path with negative controls, retaining the distinction between lexical fallback and successful semantic retrieval. Library maintainers own storage/index semantics; Loom maintainers own compatibility checks and honest readiness/score reporting. No upstream adoption or full RuVector ADR closure is inferred from this bounded consumer review.
