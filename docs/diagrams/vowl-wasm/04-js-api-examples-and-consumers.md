---
id: VW-04
title: JS API surface, README usage snippets and downstream consumers
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
adrs: [vowl-wasm:ADR-001]
sources:
  - ../vowl-wasm/src/graph/mod.rs
  - ../vowl-wasm/src/graph/pinning.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/layout/csr_sim.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/src/bindings/ngg.rs
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
  - ../vowl-wasm/tests/ngg1_wasm.test.mjs
  - ../vowl-wasm/examples/barnes_hut_benchmark.rs
  - ../vowl-wasm/Cargo.toml
  - ../WasmVOWL/modern/package.json
  - ../WasmVOWL/modern/package-lock.json
  - ../WasmVOWL/modern/src/hooks/useWasmSimulation.ts
  - ../WasmVOWL/modern/src/stores/useUIStore.ts
  - ../vowl-wasm/src/interaction/mod.rs
  - ../vowl-wasm/.github/workflows/ci.yml
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package-lock.json
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/protocol.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/positionTransport.ts
verified_commit: {vowl-wasm: 0779174612a3942b15331356422f58bf8379912b, visiongraph: dbc50f2f1075cab9fe108d4dcf917c4917d6f454, wasmvowl: 2bb54b3ee2dc3a7eda34b60d9f08fdb0fb535aab}
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](06-layout-and-release-repairs.md). Historical probe receipts retain their original revisions; source references were refreshed on 14 November 2026 after reviewing the plain-array loader, documentation and lint changes. The successor states the repaired contracts and acceptance boundaries.

## VW-04.1 `WebVowl` — the frozen 0.1.0 JS surface
```mermaid
flowchart TB
    WV["WebVowl constructor<br/>src/bindings/mod.rs:45"] --> LOAD["loadOntology(json)<br/>src/bindings/mod.rs:62"]
    WV --> STEP["initSimulation: src/bindings/mod.rs:87<br/>runSimulation: src/bindings/mod.rs:102<br/>tick: src/bindings/mod.rs:117"]
    WV --> STATE["isFinished: src/bindings/mod.rs:132<br/>getAlpha: src/bindings/mod.rs:138<br/>reheat: src/bindings/mod.rs:145"]
    WV --> TUNE["setCenter: src/bindings/mod.rs:151<br/>setLinkDistance: src/bindings/mod.rs:157<br/>setChargeStrength: src/bindings/mod.rs:163"]
    WV --> POSITION["getNodeIds: src/bindings/mod.rs:222<br/>getNodePositions: src/bindings/mod.rs:269<br/>writePositionsInto: src/bindings/mod.rs:309"]
    WV --> META["getNodeCount: src/bindings/mod.rs:199<br/>getEdgeCount: src/bindings/mod.rs:205<br/>getStatistics: src/bindings/mod.rs:373<br/>getMetadata: src/bindings/mod.rs:343"]
    WV --> PIN["pinNode, unpinNode, isPinned,<br/>resetPins, getPinnedCount<br/>src/graph/pinning.rs:82"]
    WV --> FILTER["filterByDegree, toggleDatatypes,<br/>toggleSetOperators, resetVisibility<br/>src/bindings/mod.rs:591"]
    FILTER --> HIER["filterHierarchy(depth)<br/>src/bindings/mod.rs:169"]
```
- INVARIANT (README.md:154-155): this is the frozen 0.1.0 surface — names/signatures do not change without a major bump; later releases only add (`getNodeIds`/`getNodePositions`/`writePositionsInto` arrived in 0.1.1, per src/lib.rs crate doc, and `reheat` is now on the binding, src/bindings/mod.rs:142-147).
- `getNodeIds()`/`getNodePositions()` pair is index-stable: ticking, pinning, filtering and folding never reorder or remove nodes (src/bindings/mod.rs:221-267 doc comments), so JS zips `ids[i] <-> xy[2i], xy[2i+1]` once per load.

- **Replacement boundary:** `loadOntology` replaces graph and metadata, installs an empty `PinManager` and calls `simulation.reset()`, which restarts alpha but keeps link distance, charge and centre; CHANGED at 0779174 (0.3.0), which stopped replacing the simulation with a default one (src/bindings/mod.rs:54-60, src/bindings/mod.rs:75-81). `pinNode` accepts non-finite positions through the legacy manager (src/graph/pinning.rs:82-102); CSR pinning rejects them.

## VW-04.2 `writePositionsInto` vs `getNodePositions` — allocation trade-off
```mermaid
flowchart LR
    subgraph Alloc["Owned result<br/>src/bindings/mod.rs:269"]
        A1["new Vec::with_capacity(n*2)<br/>per call"] --> A2["returned Vec<f32> →<br/>wasm-bindgen copies out"]
    end
    subgraph Reuse["Caller buffer<br/>src/bindings/mod.rs:309"]
        B1["reuse caller-owned buffer<br/>writes 2 * min(floor(out.len()/2), n)"] --> B2["returns even count; odd tail untouched<br/>src/bindings/mod.rs:314-323"]
    end
    subgraph ZeroCopy["Direct view<br/>src/bindings/explorer.rs:163,169"]
        C1["raw pointer into<br/>wasm.memory.buffer"] --> C2["JS Float32Array view,<br/>no copy at this boundary; needs ngg1 tier<br/>rebuild view after load or memory growth"]
    end
```
- The Rust method avoids creating a result vector, but the allocation claim in its doc comment is too broad (src/bindings/mod.rs:302-307): freshly generated wasm-bindgen 0.2.128 glue allocates a temporary WASM array for each `writePositionsInto` call and copies it back. Caller buffer reuse is not allocation-free across the ABI. `NggExplorer::positions_ptr` is the direct-view alternative (src/bindings/explorer.rs:157-170); the downstream worker still copies that view into its transport buffer. CSR `writePositionsInto` copies individual floats, including an odd last element, unlike the legacy pair-only method (src/layout/csr_sim.rs:836-840).

## VW-04.3 `checkNodeClick` — ray/sphere hit testing (feature `interaction`)
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS: checkNodeClick(origin, dir)
    participant WV as WebVowl::check_node_click<br/>src/bindings/mod.rs:413
    participant R as interaction::Ray::new<br/>src/interaction/mod.rs:35
    participant F as find_closest_node_hit<br/>src/interaction/mod.rs:167
    JS->>WV: ray_origin[3], ray_direction[3]
    WV->>WV: validate len == 3 each, else console.warn + None<br/>src/bindings/mod.rs:418-424
    WV->>R: Ray::new(origin, direction)
    WV->>WV: build NodeHitTest per node,<br/>radius=20.0 fixed, including hidden nodes<br/>src/bindings/mod.rs:434-446
    WV->>F: find_closest_node_hit(&ray, &nodes)
    F->>F: ray_sphere_intersection per candidate<br/>src/interaction/mod.rs:99
    F-->>WV: Option<(id, distance)>
    WV-->>JS: Option<String> node id, or null
```
- The 20.0 hit-test radius is a fixed constant in the binding, not read from per-node visual size (src/bindings/mod.rs:436-437 comment: "In a real implementation, this could be configurable or per-node").

## VW-04.5 Bundle contents contract, verified in CI
```mermaid
flowchart LR
    BUILD["wasm-pack build --target web --release<br/>--scope dreamlab-ai --features ngg1,interaction<br/>.github/workflows/ci.yml:176-204"] --> FILES["pkg/vowl_wasm.js<br/>pkg/vowl_wasm.d.ts<br/>pkg/vowl_wasm_bg.wasm<br/>pkg/package.json"]
    FILES --> CHECK1["name == @dreamlab-ai/vowl-wasm<br/>version == Cargo.toml version<br/>ci.yml Verify bundle contents step"]
    FILES --> CHECK2["d.ts must export:<br/>NggExplorer, positionsPtr, positionsLen,<br/>loadCsr, getMetadata, getNodePositions,<br/>writePositionsInto, getNodeIds<br/>ci.yml:224-228"]
    FILES --> CHECK3["bundle gate: ontomark_ngg1_migration test with VOWL_BUNDLE_GATE=1<br/>real wasm module, size budget, exact frozen JS API<br/>ci.yml:233-239"]
    FILES --> CHECK4["node --test tests/ngg1_wasm.test.mjs<br/>golden tier decodes and re-encodes byte for byte through WASM<br/>ci.yml:241-242"]
```
- CI checks that those eight API symbols exist, while the native bundle gate checks the exact declaration surface. They are not all zero-copy APIs. The build fails if a required symbol vanishes from `pkg/vowl_wasm.d.ts` (.github/workflows/ci.yml:222-239). A stale local `pkg/` failed the gate in this audit; current source was tested through a separately generated bundle.

## VW-04.6 EXTERNAL: visionGraph's `physics.worker.ts` drives `NggExplorer`
```mermaid
sequenceDiagram
    autonumber
    participant W as visionGraph worker<br/>physics.worker.ts:41
    participant PKG as vowl-wasm 0.4.0 package<br/>GitHub release tarball, see VW-05.3
    participant WA as NggExplorer<br/>explorer.rs:56
    Note over W: External consumer: visionGraph
    W->>PKG: import init, default glue<br/>await init() → InitOutput.memory
    W->>WA: loadCsr(bytes) / tick() / isFinished() / getAlpha()
    W->>WA: positionsPtr() / positionsLen()
    W->>W: rebuild Float32Array(wasm.memory.buffer, ptr, len)<br/>copy into shared or transferable transport buffer<br/>physics.worker.ts:179-205
    W->>WA: after every loadCsr, setParam for each sticky layout key<br/>physics.worker.ts:166-173 and physics.worker.ts:290-291
    Note over W,WA: CHANGED at dbc50f2: sticky keys start from EXPLORER_LAYOUT, collideRadius 21, collideStrength 0.7,<br/>collideIterations 3, componentGravity 0.1, protocol.ts:125-130. A live setParam other than reheat<br/>is kept for later scopes, physics.worker.ts:320-321
    W->>WA: optional drag: pinNode / unpinNode / reheat<br/>physics.worker.ts:54-56
```
- The worker resolves exported names "permissively (snake_case per Rust, with camelCase fallbacks)" (physics.worker.ts:35-39 comment) — it does not assume `wasm-bindgen`'s `js_name` renaming is exhaustive, a defensive stance against this crate's binding surface.
- Drag pinning is optional on the worker side: `pinNode`, `unpinNode` and `reheat` are bound with `optionalMethod` (physics.worker.ts:129-131) and the message handlers return silently when the loaded package lacks them. The package supplies them from 0.2.2 (the worker's own comment says `vowl-wasm ≥ 0.2.2`, physics.worker.ts:53); a pin that succeeds wakes the layout to `DRAG_ALPHA` 0.3 and an unpin to `RELEASE_ALPHA` 0.1 (physics.worker.ts:226-239, physics.worker.ts:325-337).
- Both known consumers now run 0.4.0 and turn the collision forces on; the second one, the standalone WasmVOWL viewer, is VW-04.11. visionGraph still installs the GitHub release tarball rather than the npm version (publishing-tools/WasmVOWL/modern/package.json:25, publishing-tools/WasmVOWL/modern/package-lock.json:598-603).

- **Failure boundary:** `loadScope` changes generation/count before validating bytes (physics.worker.ts:282-305). After a rejected tier, a later Start can publish old positions under the new generation. Caller `nodeCount` is not checked against the loaded tier. `isMainToWorker` only checks an object with a string `type` (protocol.ts:234-236), so unchecked non-finite parameter messages reach CSR configuration. These are host-adapter gaps, not guarantees supplied by the codec.

## VW-04.7 `interaction` module — native benches and the WASM binding
```mermaid
flowchart TB
    RS["ray_sphere_intersection(ray, center, radius)<br/>src/interaction/mod.rs:99"] --> USE1["checkNodeClick binding<br/>(feature interaction, WASM)"]
    RS --> USE2["interaction_bench.rs criterion bench<br/>(native, feature interaction)"]
    FC["find_closest_node_hit(ray, &[NodeHitTest])<br/>src/interaction/mod.rs:167"] --> USE1
    FC --> USE2
```
- Both entry points share pure geometry functions; the WASM binding additionally chooses candidates and validates array lengths (VW-04.3). A native benchmark measures the cases it runs; it does not establish that hidden-node selection or JavaScript argument handling is correct.

## VW-04.8 `ngg1` browser bindings — owned decode and encode beside the zero-copy explorer
```mermaid
flowchart LR
    BYTES["Uint8Array tier"] -->|"decodeNgg1(bytes)"| DEC["NggGraph::from_bytes, then validated owned snapshot<br/>src/bindings/ngg.rs:119-121 and src/bindings/ngg.rs:35"]
    DEC --> GET["accessors return copies: ids, positions, domains, categories,<br/>flags, degrees, labels, iris, rowPtr, colIdx, edgeType<br/>src/bindings/ngg.rs:57-112"]
    FIELDS["typed arrays plus string arrays"] -->|"encodeNgg1(...)"| ENC["encode_parts through ontomark's NGG1 writer,<br/>mismatched arrays and invalid CSR throw an NGG1 string<br/>src/bindings/ngg.rs:182-200"]
    CONST["ngg1Constants()"] --> OWN["flags, sentinels and edge codes read from the ontomark owner<br/>src/bindings/ngg.rs:236-244"]
    DEC -. "layout continues in a worker, no per-frame metadata" .-> EXP["NggExplorer: loadCsr, loadArrays, tick, positionsPtr, setParam, setTheta, pinNode, unpinNode, reheat<br/>src/bindings/explorer.rs:80-239"]
    GATE["feature ngg1 gates both modules<br/>src/bindings/mod.rs:3-9"] --- DEC
    N1["INVARIANT: numeric domain and category ids come from the producer, the type maps them to no estate taxonomy<br/>src/bindings/ngg.rs:13-16"]
    N2["OPEN: the TypeScript reader consolidation and the Obsidian plugin are tracked by their own steps and are not claimed by ADR-001<br/>docs/adr/ADR-001-shared-ontology-rendering-contract.md:99-103"]
    DEC --- N1
    EXP --- N2
```

**What it shows.** Version 0.2.0 adds an owned, copying way to read and write NGG1 tiers in the browser, next to the existing zero-copy explorer: the codec is ontomark's, the binding only wraps it, and a worker keeps laying out through `NggExplorer` without serialising metadata per frame (src/bindings/ngg.rs:1-6, README.md:222-239).
**Why it is this way.** ADR-001 wants one owner per graph format, so `vowl-wasm` delegates every binary read and write to ontomark and keeps only the browser surface, with a JavaScript test that round-trips the golden fixture through the built wasm (docs/adr/ADR-001-shared-ontology-rendering-contract.md:36, tests/ngg1_wasm.test.mjs:1-3).

## VW-04.9 Plain-array loading keeps the old graph on validation failure

```mermaid
flowchart TB
    IN["Interleaved positions and parallel edge arrays<br/>src/bindings/explorer.rs:110-116"] --> VALID{"Even position count, finite coordinates,<br/>parallel arrays, endpoints in range?<br/>src/bindings/explorer.rs:23-50"}
    VALID -->|no| ERR["Return a JavaScript error before changing the simulation<br/>src/bindings/explorer.rs:117"]
    VALID -->|yes| LOAD["Copy arrays into CsrSimulation<br/>src/bindings/explorer.rs:118-124"]
    LOAD --> KEEP["Stable node index order until the next load<br/>self-loops and duplicate edges accepted<br/>src/bindings/explorer.rs:92-101"]
```

**What it shows.** `loadArrays` accepts a graph already held in memory without NGG1 encoding. Invalid input leaves the preceding graph intact; the built-WASM regression checks both the error and retained counts (`tests/ngg1_wasm.test.mjs:100-130`).

**Why it is this way.** Validation happens before the array copies and simulation replacement, so malformed input produces a JavaScript exception instead of reaching a simulation assertion (`src/bindings/explorer.rs:105-124`). This source review does not establish which downstream deployment includes the new export.

## VW-04.10 Where the usage examples live — README snippets and one native Rust example
```mermaid
flowchart TB
    README["README.md usage blocks, the only JavaScript examples the repository carries"] --> RSU["Rust usage<br/>README.md:52-74"]
    README --> JSU["JavaScript: load, simulate, statistics<br/>README.md:78-89<br/>calls loadOntology, initSimulation, runSimulation, getStatistics, getMetadata, see VW-04.1"]
    README --> POS["Reading positions back: getNodeIds once, then getNodePositions per frame<br/>README.md:98-108<br/>or writePositionsInto into a caller buffer, README.md:114-118, see VW-04.2"]
    README --> WRK["NggExplorer for a rendering worker<br/>README.md:125-141, see VW-04.6"]
    README --> NGG["decodeNgg1, encodeNgg1, ngg1Constants<br/>README.md:229-241, see VW-04.8"]
    EX["examples/barnes_hut_benchmark.rs, native Rust, not JavaScript<br/>run with cargo run --release --example barnes_hut_benchmark<br/>examples/barnes_hut_benchmark.rs:3, Cargo.toml:145-146"]
    EX --> BENCH["times the exact O(n squared) force pass against Barnes-Hut at theta 0.5, 0.9 and 1.2, for 100 to 2000 nodes, ten iterations each<br/>examples/barnes_hut_benchmark.rs:60-67, examples/barnes_hut_benchmark.rs:89-90, examples/barnes_hut_benchmark.rs:110"]
    DEBT["DEBT: Cargo.toml:29 excludes /examples/*.html from the published package, but the repository tracks no HTML or JavaScript example,<br/>so a JS consumer's only worked code is the README snippets. No file under tests/ reads README.md, and the one built-WASM test covers NGG1 round trips, tests/ngg1_wasm.test.mjs:1-3"]
    README --- DEBT
    EX --- DEBT
```
**What it shows.** The topic's examples are README code blocks plus one Cargo example. The Cargo example is a native benchmark of the layout force pass and exercises none of the JavaScript surface drawn in VW-04.1.
**Why it is this way.** The crate documents its frozen JS surface through README snippets (README.md:154-155) and keeps its one `examples/` file as a layout benchmark. Why no JavaScript example was added is not recorded in the files cited here.

## VW-04.11 EXTERNAL: the two consumers on 0.4.0 and where their layout values live
```mermaid
flowchart TB
    REL["vowl-wasm 0.4.0, one tarball<br/>published to npm and attached to the GitHub release, see VW-05.3"]
    REL --> VG["visionGraph explorer installs the GitHub tarball<br/>publishing-tools/WasmVOWL/modern/package.json:25"]
    REL --> WVN["WasmVOWL modern resolves 0.4.0 from registry.npmjs.org<br/>../WasmVOWL/modern/package.json:20 and ../WasmVOWL/modern/package-lock.json:557-560"]
    VG --> VGL["own physics worker drives NggExplorer, EXPLORER_LAYOUT radius 21<br/>publishing-tools/WasmVOWL/modern/src/workers/protocol.ts:125-130, see VW-04.6"]
    WVN --> OG["through ontomark-graph 0.4.0 createLayout over vowlEngine<br/>../WasmVOWL/modern/package.json:19 and ../WasmVOWL/modern/src/hooks/useWasmSimulation.ts:77"]
    OG --> WVL["RECOMMENDED_LAYOUT_PARAMS with collideRadius raised to 23 for its 20-unit discs plus border<br/>../WasmVOWL/modern/src/stores/useUIStore.ts:45-51"]
    N1["INVARIANT: both lockfiles record the same sha512 integrity for 0.4.0, so the npm and GitHub artefacts match<br/>publishing-tools/WasmVOWL/modern/package-lock.json:601 and ../WasmVOWL/modern/package-lock.json:560"]
    VG --- N1
    WVN --- N1
```
**What it shows.** CHANGED since the earlier stamp, when WasmVOWL pinned the v0.1.1 tarball and visionGraph 0.2.3: both now take 0.4.0 and both set collision values close to the crate's recommendation (src/bindings/explorer.rs:202-206).
**Why it is this way.** Each consumer chooses the radius for the discs it draws; WasmVOWL's are larger, so it overrides one value of the shared preset. Its layout path is drawn in the wasmvowl area.
