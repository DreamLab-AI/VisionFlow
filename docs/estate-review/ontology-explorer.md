---
title: Ontology explorer and publisher lineage
status: in-progress
date: 2026-09-04
type: explanation
---

# Ontology explorer and publisher lineage

WasmVOWL intends to make the corpus navigable through a React/WebGL interface backed by Rust force layout. The standalone checkout has distinct legacy and modern implementations, while knowledgeGraph, historical Logseq and current visionGraph carry publisher copies. A shared name does not establish shared behaviour. The [source/test receipt](evidence/explorer-snapshot.json) covers the standalone checkout; the earlier [lineage snapshot](evidence/knowledge-boundaries.json) already shows divergent package, Cargo and App hashes in publisher copies.

## Standalone input and state

The [file loader](../../../WasmVOWL/modern/src/components/UI/FileDropZone.tsx) accepts JSON only and checks for a class array. It does not parse arbitrary RDF despite a broader source comment. The [graph store](../../../WasmVOWL/modern/src/stores/useGraphStore.ts) creates class nodes and property edges, taking only the first domain/range value and omitting edges without either. It does not merge the separate classAttribute/propertyAttribute arrays that the loader preserves. Closeout must define the exact accepted ontology projection and show a representative publisher export retains labels, relation endpoints and datatypes.

The store mutates Maps through Immer middleware, but no `enableMapSet` call was found in the modern source. The existing graph-store suite fails all 32 cases at `state.nodes.clear()` with an Immer plugin error. Across the full frontend suite, **19 pass and 60 fail**. Four UI-store tests also fail state expectations; these failures must be diagnosed rather than assumed to be graphics errors. No browser was run in this pass.

## React-to-Rust contract

The [simulation hook](../../../WasmVOWL/modern/src/hooks/useWasmSimulation.ts) serialises `nodes` and `edges` and passes them to `loadOntology`. The [Rust binding](../../../WasmVOWL/rust-wasm/src/bindings/mod.rs) invokes StandardParser, whose [class parser](../../../WasmVOWL/rust-wasm/src/ontology/parser.rs) requires `class` or `classes`. This is a source-established schema mismatch. Rust's **47 passing native library tests** do not prove this JavaScript-to-WASM handoff works.

The hook also reloads the graph when the nodes Map changes, while each simulation frame calls `updateNodePosition` for individual nodes. That couples evolving position state to topology initialisation and warrants a convergence/reinitialisation test. It is not a measured frame-rate claim. Initialisation and load errors are logged to the console; no corresponding user-visible error is set by the hook.

The README describes React 18 and performance figures, while the inspected modern manifest requires React 19.2. Numerical speed/frame-rate claims remain unverified here. A native force-layout test cannot certify a WebGL scene, browser interaction or performance on an actual corpus.

## Publisher lineage and closeout

The knowledgeGraph explorer no longer contains the same hook/store paths as standalone WasmVOWL; current visionGraph also has its own publishing-tools/WasmVOWL tree. This prevents transferring the standalone failures directly to the deployed knowledge explorer. Conversely, a working publisher copy would not prove the standalone quick start works. A full tree and build-output lineage comparison remains open.

CP-01/02/06/08 needs an explicit owner for the standalone and each publisher variant, a build/revision manifest, one accepted schema at the React/Rust boundary, and a real-browser test loading the actual exported corpus. Require successful and malformed file handling, labels/edges/datatypes, search/filter/selection/export, visible runtime errors and accessible non-pointer interaction. Measure startup, stable layout, memory and interaction latency against a named dataset/device before adopting performance claims.

The new [proposed explorer ADR](../../../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md) records these acceptance conditions without declaring adoption or implementation completion. The [estate roadmap](closeout/README.md) retains publisher lineage and live-browser validation as open work.
