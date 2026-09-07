---
title: Imported ADR packs and actual consumer boundaries
date: 2026-09-07
status: consumer-scope-audited-upstream-semantics-open
scope: RuVector and RuView imported decision families
---

# Imported ADR scope — 2026-09-07

Imported ADRs remain in the estate audit, but they do not all describe implemented estate capabilities. This review assigns a consumer-based disposition to each of the 17 families in the existing [upstream pack map](closeout/upstream-packs.md). It supplements the [earlier identity review](upstream-decision-scope.md), not its per-record semantic review. “Excluded” below means excluded from **implemented estate claims**, not removed from the user-requested review corpus.

## Three different implementation identities

| Consumer | Source selection verified in this pass | What it establishes | What remains unverified |
|---|---|---|---|
| Loom HNSW adapter | [workspace Cargo.toml](../../../loom/Cargo.toml) selects sibling `../ruvector/crates/ruvector-core`, defaults off, features `hnsw`, `storage`, `simd`, `parallel`; [adapter](../../../loom/crates/loom-vector-ruvector/src/hnsw.rs) imports and calls `ruvector_core::VectorDB` | Direct source relationship to the sibling checkout and a concrete semantic-index consumer | Loaded image, runtime semantic-fallback flag, current recall and build overrides |
| Loom Postgres build channel | [adapter manifest](../../../loom/crates/loom-vector-ruvector/Cargo.toml) has default features empty; `pg-write` gates tokio-postgres, pgvector, Xinference and the stage/export binaries | An optional build/off-turn channel; it is not the default serving hot path | Whether a particular build enabled or executed it |
| Loom attestation | [manifest](../../../loom/crates/loom-attest-proofgate/Cargo.toml) optionally enables only ruvector-core; [implementation](../../../loom/crates/loom-attest-proofgate/src/lib.rs) implements `ChainedLedger` with SHA-256 and a head checkpoint | A concrete local ledger, not adoption of design-named upstream `ProofGate<T>` / `MutationLedger` | Independent safety/durability proof or loaded runtime identity; those named types belong to another upstream crate |
| Agentbox shared memory | [manifest](../../../project/agentbox/agentbox.toml) and [compose](../../../project/agentbox/docker-compose.yml) select `ruvnet/ruvector-postgres:2.0.5@sha256:7fb09d439d82fccbe6e0035d09f91c9dce456bb2d70cb2e8d6afb12a6afa2051`; [MCP wrapper](../../../project/agentbox/mcp/servers/ruvector-mcp.cjs) owns the agent-facing embedding path | Declared image identity and implemented MCP/embedding contract | Actual running image digest and loaded extension build/version. Image tag 2.0.5 must not be silently equated to the extension's version or sibling Git HEAD |
| RuView sensing consumers | [workspace](../../../RuView/rust-port/wifi-densepose-rs/Cargo.toml), [lockfile](../../../RuView/rust-port/wifi-densepose-rs/Cargo.lock), [integration crate](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-ruvector/Cargo.toml) | Published registry dependencies with checksums, not the vendored RuVector tree. Lock has nine RuVector package/version entries, including two attention versions | Selected release features, registry implementation equivalence, actual executable/device use and hardware benefit |

The nine RuView lock entries are attention 0.1.32 and 2.0.4, attn-mincut 2.0.4, core 2.0.5, crv 0.1.1, gnn 2.0.5, mincut 2.0.4, solver 2.0.4 and temporal-tensor 2.0.4. The integration crate's `crv` feature gates CRV/GNN. Signal and training manifests directly select mincut, attention and solver families. MAT defaults include its `ruvector` feature, which enables solver/temporal dependencies. These are declared build paths; default selection is still not evidence that a deployed process exercised an algorithm.

No RuVector package reference was found in the inspected VisionClaw root Cargo.toml/Cargo.lock. Its observational memory-flash endpoint is not a database write client. This is a narrow root dependency observation, not a transitive all-language proof for every workspace repository.

## Per-pack disposition

Counts reproduce the existing pack-map inventory, not a newly asserted exhaustive file census. Standalone/vendored copies retain distinct lineage. A selected crate can make some decisions relevant without adopting an entire family.

| Family | Existing standalone / vendored count | Consumer-based disposition | Rationale and remaining work |
|---|---:|---|---|
| `RuView/docs/adr` | 42 / 0 | Relevant external product; no wholesale estate adoption | Sensing source and RuVector integration declarations exist. Per-record hardware/training/deployment assertions require the sensing reviews and exact runtime evidence; a source implementation is not device validation. |
| `RuView/rust-port/wifi-densepose-rs/docs/adr` | 3 / 0 | Relevant Rust consumer decisions | These are nearest the registry dependencies. Preserve the Rust-port scope; do not replace registry source identity with the vendored checkout. |
| `crates/ruQu/docs/adr` | 1 / 1 | Excluded from demonstrated estate implementation | No ruQu selection in the inspected Loom, VisionClaw or RuView manifests. A quantum-labelled upstream component is not supplied by the selected HNSW/solver paths. Reopen with a concrete caller and build identity. |
| `crates/ruvector-mincut/docs/adr` | 7 / 7 | Relevant capability; exact package semantics open | RuView selects published mincut/attn-mincut. This establishes relevance, not equality between local/vendored algorithm designs and the resolved registry crate. Map claims to checked dependency source and call sites. |
| `crates/rvf/docs/adr` | 1 / 1 | Design relevance; implementation adoption not established here | Cognitive-container/RVF design references do not make Agentbox memory Postgres or Loom's `.rvdb` file the same wire format. No RVF dependency was found in the inspected consumer manifests. Retain the earlier sensing-container audit separately. |
| `docs/adr` | 50 / 50 | Mixed: core and Postgres relevant; family-wide acceptance excluded | Loom's core adapter and Agentbox's image/MCP boundary are concrete. Other top-level engine, security, inference and federation claims need individual consumers. In particular Loom's local ledger is not the named upstream ProofGate implementation. |
| `docs/adr/coherence-engine` | 22 / 22 | Excluded from demonstrated estate implementation | No selected coherence-engine consumer in the inspected manifests. Do not transfer coherence/performance guarantees to the memory or ontology paths. Per-record conceptual relevance remains open. |
| `docs/adr/delta-behavior` | 10 / 10 | Excluded from demonstrated estate implementation | No delta-behaviour engine consumer identified in the inspected manifests. Learning trajectories and recency scoring are not evidence of this separate engine. |
| `docs/adr/quantum-engine` | 15 / 15 | Excluded from demonstrated estate implementation | No quantum-engine dependency/caller established. Solver or SIMD selection does not establish quantum-engine adoption. |
| `docs/adr/temporal-tensor-store` | 6 / 6 | Relevant capability through RuView; package claims open | RuView integration and default MAT feature select published temporal-tensor. Validate that package's actual storage/compression semantics; do not transfer them to Agentbox Postgres. |
| `docs/research/sublinear-time-solver/adr` | 12 / 12 | Relevant research lineage; asymptotic/quality claims unverified | RuView selects published solver. Research ADRs do not prove which algorithm a selected branch executes, nor its achieved latency/accuracy. Require call tracing and representative measurement. |
| `examples/OSpipe` | 1 / 1 | Independent example, excluded from estate implementation | No example application selected by inspected consumer manifests. Similar agent-tool terminology is insufficient. |
| `examples/delta-behavior/adr` | 4 / 4 | Independent example, excluded from estate implementation | Distinct from both Agentbox's learning pipeline and any adopted engine; require an explicit deployed caller before claiming reuse. |
| `examples/dna/adr` | 15 / 15 | Independent example, excluded from estate implementation | No DNA example consumer established. Shared vector infrastructure does not adopt this application's domain claims. |
| `examples/prime-radiant/docs/adr` | 6 / 6 | Independent example, excluded from estate implementation | No prime-radiant application dependency established; assess demonstrations as demonstrations. |
| `examples/vibecast-7sense/docs/adr` | 9 / 9 | Independent example, excluded from estate implementation | RuView sensing does not by itself adopt the separate vibecast application. Require an executable integration. |
| `npm/packages/ruvbot/docs/adr` | 15 / 15 | Separate application, excluded from inspected consumer implementation | Agentbox has agent/tool orchestration, but the inspected manifests do not establish a ruvbot dependency. Do not infer adoption from similarly named agents. |

These dispositions do not claim a global negative search of every executable or dynamic plugin. They are bounded to the selected consumer manifests, source seams and earlier pack lineage. Any newly demonstrated caller reopens the relevant row.

## Why an upstream benchmark cannot close the deployed recall issue

The RuvNet reference search returned a later/different corpus record for `docs/adr/ADR-258-hnsw-delete-repair.md`, explicitly describing a research PoC and deferred production integration. It is useful evidence that “accepted ADR” and “production integration” can differ, but it is **not** a source identity for the local checkout or loaded Postgres extension. Its synthetic deletion benchmark is not the Agentbox 384-dimensional production corpus. The [Agentbox audit](2026-09-07-agentbox-audit.md) therefore preserves the distinction between local HNSW tombstone code, the historical parallel-build incident, and the proposed cross-store erasure protocol.

## Evidence handling and closeout

The [prior-evidence revalidation](evidence/2026-09-07/prior-evidence-revalidation.json) belongs to the estate's source-hash check. A matching source hash preserves a previous source observation; it does not rerun a benchmark, prove a loaded binary or accept a proposal. This report independently reread the manifests and source paths above, including the nine lock entries and declared Postgres digest. It did not run Cargo resolution/builds, download registry packages, inspect a loaded extension, query SQL or change shared memory except through the authorised memory MCP.

Closeout should retain three independent fields for each relevant upstream claim: **adoption** (which consumer/build selects it), **implementation** (which exact package/source realises it), and **acceptance** (which source test/runtime measurement demonstrates the promised behaviour). Unadopted examples need an explicit scope disposition, not invented implementation work. Relevant but unverified registry or image claims need package/image identity and consumer tests, not a blanket ADR status upgrade.


## Diagram-reference corrections in this pass

Estate ES-01, ES-02, ES-09 and ES-10 now use repository-qualified ADR keys. ES-10 names both repositories where number collisions had hidden separate decisions: VisionClaw ADR-2012 is the dev-auth bypass, Agentbox ADR-2012 relay admission; VisionClaw ADR-2027 defines deployment profiles, Agentbox ADR-2027 custody. ES-02's ambiguous ADR-2015 was removed: its envelope provenance helper is not VisionClaw's derived-quad writeback fence, and the Agentbox trajectory recorder is not among that topic's drawn/cited sources. All 33 retained qualified references resolve to actual records.

ES-10.10 now draws implemented optional break-glass bounds and fingerprint logging, preserving the durable-audit gap. It distinguishes VisionClaw's surviving ordinary ZIP script from Agentbox's separate Rust age implementation, without claiming the latter replaced the former or was deployed. ES-09 has no ZIP/backup assertion to replace. Source changes were not made; existing external diagram edits were preserved.

Validation: all local links in this report resolve; qualified ADR paths resolve; `node scripts/diagram-index-gen.cjs docs/diagrams --check --render --jobs 2 --only estate/10` passed, rendering 10/10 diagrams. No production backup, secret access, SQL, build or deployment was performed.
