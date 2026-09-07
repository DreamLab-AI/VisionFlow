---
id: VW-04
title: JS API surface, examples and downstream consumers
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
adrs: []
sources:
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/src/interaction/mod.rs
  - ../vowl-wasm/examples/barnes_hut_benchmark.rs
  - ../vowl-wasm/.github/workflows/ci.yml
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts
verified_commit: {vowl-wasm: 65e2d1e78, visionGraph: 9e308164c}
---

## VW-04.1 `WebVowl` — the frozen 0.1.0 JS surface
```mermaid
classDiagram
    class WebVowl {
        <<wasm_bindgen>>
        +new() src/bindings/mod.rs:46
        +loadOntology(json) src/bindings/mod.rs:57
        +initSimulation() src/bindings/mod.rs:161
        +runSimulation(n) src/bindings/mod.rs:176
        +tick() src/bindings/mod.rs:191
        +isFinished() bool src/bindings/mod.rs:206
        +getAlpha() f64 src/bindings/mod.rs:212
        +setCenter(x,y) src/bindings/mod.rs:218
        +setLinkDistance(d) src/bindings/mod.rs:224
        +setChargeStrength(s) src/bindings/mod.rs:230
        +getNodeCount() usize src/bindings/mod.rs:266
        +getEdgeCount() usize src/bindings/mod.rs:272
        +getNodeIds() Vec~String~ src/bindings/mod.rs:289
        +getNodePositions() Vec~f32~ src/bindings/mod.rs:336
        +writePositionsInto(buf) usize src/bindings/mod.rs:376
        +getStatistics() src/bindings/mod.rs:440
        +getMetadata() src/bindings/mod.rs:410
        +pinNode/unpinNode/isPinned/resetPins/getPinnedCount
        +filterByDegree/toggleDatatypes/toggleSetOperators/resetVisibility/filterHierarchy
    }
```
- INVARIANT (README.md:132-135): this is the frozen 0.1.0 surface — names/signatures do not change without a major bump; 0.1.x only adds (`getNodeIds`/`getNodePositions`/`writePositionsInto` arrived in 0.1.1, per src/lib.rs crate doc).
- `getNodeIds()`/`getNodePositions()` pair is index-stable: ticking, pinning, filtering and folding never reorder or remove nodes (src/bindings/mod.rs:288-334 doc comments), so JS zips `ids[i] <-> xy[2i], xy[2i+1]` once per load.

## VW-04.2 `writePositionsInto` vs `getNodePositions` — allocation trade-off
```mermaid
flowchart LR
    subgraph Alloc["getNodePositions() — src/bindings/mod.rs:336"]
        A1["new Vec::with_capacity(n*2)<br/>per call"] --> A2["returned Vec<f32> →<br/>wasm-bindgen copies out"]
    end
    subgraph NoAlloc["writePositionsInto(buf) — src/bindings/mod.rs:376"]
        B1["reuse caller-owned buffer<br/>writes min(out.len(), n*2)"] --> B2["returns count written<br/>src/bindings/mod.rs:395-407"]
    end
    subgraph ZeroCopy["NggExplorer.positionsPtr()/positionsLen() — src/bindings/explorer.rs:82,89"]
        C1["raw pointer into<br/>wasm.memory.buffer"] --> C2["JS Float32Array view,<br/>no copy at all — needs ngg1 tier"]
    end
```
- The doc comment is explicit: `writePositionsInto` saves the per-frame *allocation* over `getNodePositions`, not the copy — `NggExplorer::positions_ptr` is the only genuinely copy-free path, and it requires the `ngg1` binary-tier format (src/bindings/mod.rs:369-374).

## VW-04.3 `checkNodeClick` — ray/sphere hit testing (feature `interaction`)
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS: checkNodeClick(origin, dir)
    participant WV as WebVowl::check_node_click<br/>src/bindings/mod.rs:480
    participant R as interaction::Ray::new<br/>src/interaction/mod.rs:35
    participant F as find_closest_node_hit<br/>src/interaction/mod.rs:167
    JS->>WV: ray_origin[3], ray_direction[3]
    WV->>WV: validate len == 3 each, else console.warn + None<br/>src/bindings/mod.rs:485-491
    WV->>R: Ray::new(origin, direction)
    WV->>WV: build NodeHitTest per node,<br/>radius=20.0 fixed<br/>src/bindings/mod.rs:501-513
    WV->>F: find_closest_node_hit(&ray, &nodes)
    F->>F: ray_sphere_intersection per candidate<br/>src/interaction/mod.rs:99
    F-->>WV: Option<(id, distance)>
    WV-->>JS: Option<String> node id, or null
```
- The 20.0 hit-test radius is a fixed constant in the binding, not read from per-node visual size (src/bindings/mod.rs:503-504 comment: "In a real implementation, this could be configurable or per-node").

## VW-04.4 `examples/barnes_hut_benchmark.rs` — O(n²) vs Barnes-Hut comparison
```mermaid
flowchart TB
    GEN["create_test_graph(n)<br/>examples/barnes_hut_benchmark.rs:11<br/>nodes placed on a circle, radius 500"] --> RUN1["ForceSimulation with<br/>use_barnes_hut=false<br/>Instant::now() timed run"]
    GEN --> RUN2["ForceSimulation with<br/>use_barnes_hut=true"]
    RUN1 --> CMP["compare elapsed time,<br/>printed to stdout"]
    RUN2 --> CMP
```
- `cargo run --release --example barnes_hut_benchmark` (examples/barnes_hut_benchmark.rs:3) is a native binary, not a criterion bench — it prints timings rather than emitting `criterion` reports (contrast with VW-05.4).

## VW-04.5 Bundle contents contract, verified in CI
```mermaid
flowchart LR
    BUILD["wasm-pack build --target web --release<br/>--scope dreamlab-ai --features ngg1,interaction<br/>.github/workflows/ci.yml job wasm-pack"] --> FILES["pkg/vowl_wasm.js<br/>pkg/vowl_wasm.d.ts<br/>pkg/vowl_wasm_bg.wasm<br/>pkg/package.json"]
    FILES --> CHECK1["name == @dreamlab-ai/vowl-wasm<br/>version == Cargo.toml version<br/>ci.yml Verify bundle contents step"]
    FILES --> CHECK2["d.ts must export:<br/>NggExplorer, positionsPtr, positionsLen,<br/>loadCsr, getMetadata, getNodePositions,<br/>writePositionsInto, getNodeIds"]
```
- INVARIANT: CI fails the build if any of those seven zero-copy-path symbols vanish from `pkg/vowl_wasm.d.ts` — "the explorer breaks at runtime rather than at build" (ci.yml Verify-bundle-contents step comment).

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
