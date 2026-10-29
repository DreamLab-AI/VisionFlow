---
id: VW-04
title: JS API surface, examples and downstream consumers
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
adrs: [vowl-wasm:ADR-001]
sources:
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/src/bindings/ngg.rs
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
  - ../vowl-wasm/tests/ngg1_wasm.test.mjs
  - ../vowl-wasm/src/interaction/mod.rs
  - ../vowl-wasm/.github/workflows/ci.yml
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts
verified_commit: {vowl-wasm: 89d1a0a32eec16e79a7fa507eaa6f8fcb92e4f0b, visiongraph: fd5269a55e52b8b1d1e9d7d59cb5f0c7835a5a2e}
---

## VW-04.1 `WebVowl` — the frozen 0.1.0 JS surface
```mermaid
classDiagram
    class WebVowl {
        <<wasm_bindgen>>
        +new() src/bindings/mod.rs:45
        +loadOntology(json) src/bindings/mod.rs:56
        +initSimulation() src/bindings/mod.rs:76
        +runSimulation(n) src/bindings/mod.rs:91
        +tick() src/bindings/mod.rs:106
        +isFinished() bool src/bindings/mod.rs:121
        +getAlpha() f64 src/bindings/mod.rs:127
        +setCenter(x,y) src/bindings/mod.rs:140
        +setLinkDistance(d) src/bindings/mod.rs:146
        +setChargeStrength(s) src/bindings/mod.rs:152
        +reheat(alpha) src/bindings/mod.rs:134
        +getNodeCount() usize src/bindings/mod.rs:188
        +getEdgeCount() usize src/bindings/mod.rs:194
        +getNodeIds() Vec~String~ src/bindings/mod.rs:211
        +getNodePositions() Vec~f32~ src/bindings/mod.rs:258
        +writePositionsInto(buf) usize src/bindings/mod.rs:298
        +getStatistics() src/bindings/mod.rs:362
        +getMetadata() src/bindings/mod.rs:332
        +pinNode/unpinNode/isPinned/resetPins/getPinnedCount
        +filterByDegree/toggleDatatypes/toggleSetOperators/resetVisibility/filterHierarchy
    }
```
- INVARIANT (README.md:148-149): this is the frozen 0.1.0 surface — names/signatures do not change without a major bump; later releases only add (`getNodeIds`/`getNodePositions`/`writePositionsInto` arrived in 0.1.1, per src/lib.rs crate doc, and `reheat` is now on the binding, src/bindings/mod.rs:131-136).
- `getNodeIds()`/`getNodePositions()` pair is index-stable: ticking, pinning, filtering and folding never reorder or remove nodes (src/bindings/mod.rs:210-256 doc comments), so JS zips `ids[i] <-> xy[2i], xy[2i+1]` once per load.

## VW-04.2 `writePositionsInto` vs `getNodePositions` — allocation trade-off
```mermaid
flowchart LR
    subgraph Alloc["getNodePositions() — src/bindings/mod.rs:258"]
        A1["new Vec::with_capacity(n*2)<br/>per call"] --> A2["returned Vec<f32> →<br/>wasm-bindgen copies out"]
    end
    subgraph NoAlloc["writePositionsInto(buf) — src/bindings/mod.rs:298"]
        B1["reuse caller-owned buffer<br/>writes min(out.len(), n*2)"] --> B2["returns count written<br/>src/bindings/mod.rs:303-312"]
    end
    subgraph ZeroCopy["NggExplorer.positionsPtr()/positionsLen() — src/bindings/explorer.rs:82,89"]
        C1["raw pointer into<br/>wasm.memory.buffer"] --> C2["JS Float32Array view,<br/>no copy at all — needs ngg1 tier"]
    end
```
- The doc comment is explicit: `writePositionsInto` saves the per-frame *allocation* over `getNodePositions`, not the copy — `NggExplorer::positions_ptr` is the only genuinely copy-free path, and it requires the `ngg1` binary-tier format (src/bindings/mod.rs:291-296).

## VW-04.3 `checkNodeClick` — ray/sphere hit testing (feature `interaction`)
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS: checkNodeClick(origin, dir)
    participant WV as WebVowl::check_node_click<br/>src/bindings/mod.rs:402
    participant R as interaction::Ray::new<br/>src/interaction/mod.rs:35
    participant F as find_closest_node_hit<br/>src/interaction/mod.rs:167
    JS->>WV: ray_origin[3], ray_direction[3]
    WV->>WV: validate len == 3 each, else console.warn + None<br/>src/bindings/mod.rs:407-413
    WV->>R: Ray::new(origin, direction)
    WV->>WV: build NodeHitTest per node,<br/>radius=20.0 fixed<br/>src/bindings/mod.rs:423-435
    WV->>F: find_closest_node_hit(&ray, &nodes)
    F->>F: ray_sphere_intersection per candidate<br/>src/interaction/mod.rs:99
    F-->>WV: Option<(id, distance)>
    WV-->>JS: Option<String> node id, or null
```
- The 20.0 hit-test radius is a fixed constant in the binding, not read from per-node visual size (src/bindings/mod.rs:425-426 comment: "In a real implementation, this could be configurable or per-node").

## VW-04.5 Bundle contents contract, verified in CI
```mermaid
flowchart LR
    BUILD["wasm-pack build --target web --release<br/>--scope dreamlab-ai --features ngg1,interaction<br/>.github/workflows/ci.yml:176-204"] --> FILES["pkg/vowl_wasm.js<br/>pkg/vowl_wasm.d.ts<br/>pkg/vowl_wasm_bg.wasm<br/>pkg/package.json"]
    FILES --> CHECK1["name == @dreamlab-ai/vowl-wasm<br/>version == Cargo.toml version<br/>ci.yml Verify bundle contents step"]
    FILES --> CHECK2["d.ts must export:<br/>NggExplorer, positionsPtr, positionsLen,<br/>loadCsr, getMetadata, getNodePositions,<br/>writePositionsInto, getNodeIds<br/>ci.yml:224-228"]
    FILES --> CHECK3["bundle gate: ontomark_ngg1_migration test with VOWL_BUNDLE_GATE=1<br/>real wasm module, size budget, exact frozen JS API<br/>ci.yml:233-239"]
    FILES --> CHECK4["node --test tests/ngg1_wasm.test.mjs<br/>golden tier decodes and re-encodes byte for byte through WASM<br/>ci.yml:241-242"]
```
- INVARIANT: CI fails the build if any of those eight zero-copy-path symbols vanish from `pkg/vowl_wasm.d.ts` — "the explorer breaks at runtime rather than at build" (.github/workflows/ci.yml:222-223).

## VW-04.6 EXTERNAL: visionGraph's `physics.worker.ts` drives `NggExplorer`
```mermaid
sequenceDiagram
    autonumber
    participant W as physics.worker.ts<br/>visionGraph publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts:38 interface WasmPhysics
    participant PKG as "@dreamlab-ai/vowl-wasm" npm package<br/>(published pkg/, see VW-04.5, VW-05.3)
    participant WA as NggExplorer<br/>src/bindings/explorer.rs:17
    Note over W: EXTERNAL — visionGraph repo, area VG-* (authored concurrently)
    W->>PKG: import init, default glue<br/>await init() → InitOutput.memory
    W->>WA: loadCsr(bytes) / tick() / isFinished() / getAlpha()
    W->>WA: positionsPtr() / positionsLen()
    W->>W: new Float32Array(wasm.memory.buffer, ptr, len)<br/>SharedArrayBuffer or transferable double-buffer
    W->>WA: setParam(key, val) [optional live tuning]
```
- The worker resolves exported names "permissively (snake_case per Rust, with camelCase fallbacks)" (physics.worker.ts:33-35 comment) — it does not assume `wasm-bindgen`'s `js_name` renaming is exhaustive, a defensive stance against this crate's binding surface.
- Cross-repo dependency direction: visionGraph depends on the *published* `@dreamlab-ai/vowl-wasm` npm bundle (pinned in its `package.json`/`package-lock.json`, per VisionFlow estate map), never on this crate's source — see VW-05.3 for the publish path that produces that bundle.

## VW-04.7 `interaction` module — always-native benches vs the WASM binding
```mermaid
flowchart TB
    RS["ray_sphere_intersection(ray, center, radius)<br/>src/interaction/mod.rs:99"] --> USE1["checkNodeClick binding<br/>(feature interaction, WASM)"]
    RS --> USE2["interaction_bench.rs criterion bench<br/>(native, feature interaction)"]
    FC["find_closest_node_hit(ray, &[NodeHitTest])<br/>src/interaction/mod.rs:167"] --> USE1
    FC --> USE2
```
- Both entry points share the same pure functions; the WASM binding adds only input validation and `NodeHitTest` construction from `VowlGraph` (VW-04.3), so a change to hit-test geometry is exercised by the native bench before it ever reaches JS.

## VW-04.8 `ngg1` browser bindings — owned decode and encode beside the zero-copy explorer
```mermaid
flowchart LR
    BYTES["Uint8Array tier"] -->|"decodeNgg1(bytes)"| DEC["NggGraph::from_bytes, then validated owned snapshot<br/>src/bindings/ngg.rs:119-121 and src/bindings/ngg.rs:35"]
    DEC --> GET["accessors return copies: ids, positions, domains, categories,<br/>flags, degrees, labels, iris, rowPtr, colIdx, edgeType<br/>src/bindings/ngg.rs:57-112"]
    FIELDS["typed arrays plus string arrays"] -->|"encodeNgg1(...)"| ENC["encode_parts through ontomark's NGG1 writer,<br/>mismatched arrays and invalid CSR throw an NGG1 string<br/>src/bindings/ngg.rs:182-200"]
    CONST["ngg1Constants()"] --> OWN["flags, sentinels and edge codes read from the ontomark owner<br/>src/bindings/ngg.rs:236-244"]
    DEC -. "layout continues in a worker, no per-frame metadata" .-> EXP["NggExplorer: loadCsr, tick, positionsPtr, setParam, setTheta, reheat<br/>src/bindings/explorer.rs:40-118"]
    GATE["feature ngg1 gates both modules<br/>src/bindings/mod.rs:3-9"] --- DEC
    N1["INVARIANT: numeric domain and category ids come from the producer, the type maps them to no estate taxonomy<br/>src/bindings/ngg.rs:13-16"]
    N2["OPEN: the TypeScript reader consolidation and the Obsidian plugin are tracked by their own steps and are not claimed by ADR-001<br/>docs/adr/ADR-001-shared-ontology-rendering-contract.md:89-93"]
    DEC --- N1
    EXP --- N2
```

**What it shows.** Version 0.2.0 adds an owned, copying way to read and write NGG1 tiers in the browser, next to the existing zero-copy explorer: the codec is ontomark's, the binding only wraps it, and a worker keeps laying out through `NggExplorer` without serialising metadata per frame (src/bindings/ngg.rs:1-6, README.md:216-233).
**Why it is this way.** ADR-001 wants one owner per graph format, so `vowl-wasm` delegates every binary read and write to ontomark and keeps only the browser surface, with a JavaScript test that round-trips the golden fixture through the built wasm (docs/adr/ADR-001-shared-ontology-rendering-contract.md:26, tests/ngg1_wasm.test.mjs:1-3).
