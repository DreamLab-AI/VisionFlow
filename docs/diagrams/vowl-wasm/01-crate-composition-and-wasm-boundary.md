---
id: VW-01
title: Crate composition, modules and the WASM boundary
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
adrs: []
sources:
  - ../vowl-wasm/Cargo.toml
  - ../vowl-wasm/src/lib.rs
  - ../vowl-wasm/src/error.rs
  - ../vowl-wasm/src/debug.rs
  - ../vowl-wasm/src/render/mod.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/src/graph/mod.rs
  - ../vowl-wasm/src/ontology/mod.rs
  - ../vowl-wasm/src/layout/mod.rs
  - ../vowl-wasm/src/ngg1.rs
  - ../vowl-wasm/src/interaction/mod.rs
verified_commit: 65e2d1e78
---

## VW-01.1 Top-level module tree
```mermaid
flowchart TB
    LIB["lib.rs<br/>src/lib.rs:145 #!deny(missing_docs, unsafe_code)"]
    LIB --> BIND["bindings/<br/>src/bindings/mod.rs, explorer.rs"]
    LIB --> DEBUG["debug<br/>src/debug.rs"]
    LIB --> GRAPH["graph/<br/>src/graph/mod.rs:6"]
    LIB --> LAYOUT["layout/<br/>src/layout/mod.rs:12"]
    LIB --> ONT["ontology/<br/>src/ontology/mod.rs:6"]
    LIB --> RENDER["render<br/>src/render/mod.rs"]
    LIB -. "feature: ngg1" .-> NGG1["ngg1<br/>src/ngg1.rs:19 NGG1_MAGIC"]
    LIB -. "feature: interaction" .-> INTER["interaction<br/>src/interaction/mod.rs"]
    LIB --> ERR["error, private<br/>src/error.rs:6"]
```
- `error` is not re-exported as `pub mod`; only `Result`/`VowlError` are re-exported at the crate root (`src/lib.rs:154`).
- `render::SvgRenderer` (src/render/mod.rs:20) exists but is not wired to any `#[wasm_bindgen]` binding — a native-only diagnostic path.

## VW-01.2 Submodule breakdown — graph and layout
```mermaid
flowchart TB
    GRAPH["graph/"] --> GBUILD["builder<br/>src/graph/builder.rs:7"]
    GRAPH --> GNODE["node<br/>src/graph/node.rs"]
    GRAPH --> GEDGE["edge<br/>src/graph/edge.rs"]
    GRAPH --> GPIN["pinning<br/>src/graph/pinning.rs:16"]
    GRAPH --> GSTAT["statistics<br/>src/graph/statistics.rs:13"]
    LAYOUT["layout/"] --> LSIM["simulation::ForceSimulation<br/>src/layout/simulation.rs:20"]
    LAYOUT --> LQUAD["quadtree::QuadTree<br/>src/layout/quadtree.rs:90"]
    LAYOUT --> LFORCE["force fns<br/>src/layout/force.rs:13"]
    LAYOUT --> LSIMD["simd<br/>src/layout/simd.rs"]
    LAYOUT --> LCSR["csr_sim::CsrSimulation<br/>src/layout/csr_sim.rs:69"]
```
- All five `graph::` submodules and all five `layout::` submodules are always compiled — none carry a `#[cfg(feature = ...)]` gate, unlike `ontology::` and `bindings::` below (VW-01.3).

## VW-01.3 Submodule breakdown — ontology and bindings
```mermaid
flowchart TB
    ONT["ontology/"] --> OPARSE["parser::StandardParser<br/>src/ontology/parser.rs:10"]
    ONT --> OMODEL["model<br/>src/ontology/model.rs"]
    ONT --> OVAL["owl2_validator::OWL2Validator<br/>src/ontology/owl2_validator.rs:64"]
    ONT -. "feature: markdown-ontology" .-> OLOAD["loader<br/>src/ontology/loader.rs:132"]
    ONT -. "feature: markdown-ontology" .-> OMD["markdown_parser<br/>src/ontology/markdown_parser.rs:8"]
    BIND["bindings/"] --> WV["WebVowl<br/>src/bindings/mod.rs:28"]
    BIND -. "feature: ngg1" .-> NGGX["NggExplorer<br/>src/bindings/explorer.rs:17"]
```
- `ontology::loader`/`markdown_parser` and `bindings::explorer::NggExplorer` are the crate's only feature-gated modules (matching VW-01.4's feature table).

## VW-01.4 Feature flags and what they gate
```mermaid
flowchart LR
    subgraph Always["always compiled, default = []"]
        CORE["OWL/JSON parsing, graph model,<br/>force layout, OWL2 validation,<br/>pinning, statistics<br/>Cargo.toml:63-64"]
    end
    NGG1F["ngg1<br/>Cargo.toml:70"] --> NGGMOD["src/ngg1.rs + NggExplorer<br/>bindings/explorer.rs:17"]
    MDF["markdown-ontology<br/>Cargo.toml:76 dep:regex"] --> MDMOD["ontology::loader + markdown_parser<br/>+ parseMarkdownOntology/validateOWL2"]
    INTF["interaction<br/>Cargo.toml:80"] --> INTMOD["interaction module<br/>+ checkNodeClick binding"]
    DBGF["debug-serde<br/>Cargo.toml:84"] --> DBGMOD["WebVowl.getGraphData()<br/>bindings/mod.rs:254"]
    PARF["parallel<br/>Cargo.toml:89 dep:rayon, native only"] --> PARMOD["ontology loading data-parallelism"]
    SIMDF["simd<br/>Cargo.toml:92, needs simd128 target-feature"] --> SIMDMOD["layout/simd.rs SIMD128 kernels"]
    NGG1F -. "not in bundle alone" .-> BUNDLE
    INTF --> BUNDLE["published npm bundle<br/>README.md:150 --features ngg1,interaction"]
```
- INVARIANT: `default = []` (Cargo.toml:63) — nothing optional compiles unless asked for.
- `markdown-ontology` pulls `regex`, measured as 68% of the 0.1.0 compiled code section (Cargo.toml:44-47); deliberately excluded from the published bundle.

## VW-01.5 `init()` and `version()` — module entry, called once by wasm-bindgen
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS host<br/>await init()
    participant WB as wasm-bindgen glue
    participant M as vowl_wasm::init<br/>src/lib.rs:169
    participant V as vowl_wasm::version<br/>src/lib.rs:178
    JS->>WB: instantiate module
    WB->>M: #[wasm_bindgen(start)] init()
    Note right of M: no-op today, reserved for panic-hook install<br/>src/lib.rs:162
    JS->>V: version()
    V-->>JS: env!("CARGO_PKG_VERSION")<br/>src/lib.rs:179
```

## VW-01.6 `VowlError` to `JsValue` at every fallible binding
```mermaid
classDiagram
    class VowlError {
        <<enum, thiserror>>
        ParseError(String)
        InvalidData(String)
        GraphError(String)
        LayoutError(String)
        RenderError(String)
        BindingError(String)
        InteractionError(String)
    }
    class Result~T~ {
        <<alias>>
        std::result::Result~T, VowlError~
    }
    class JsValue {
        <<wasm_bindgen>>
    }
    VowlError --> JsValue : From impl src/error.rs 43
    Result~T~ ..> VowlError
    VowlError <-- SerdeJsonError : From impl src/error.rs 49
```
- Every `#[wasm_bindgen]` method on `WebVowl` returns `std::result::Result<T, JsValue>` (e.g. `load_ontology` src/bindings/mod.rs:57) — the crate's typed `VowlError` never crosses the boundary, only its rendered message.

## VW-01.7 Two entry points: `WebVowl` vs `NggExplorer`
```mermaid
flowchart TB
    subgraph WebVowl["WebVowl - general purpose, always available, src/bindings/mod.rs:28"]
        WV1["loadOntology(json) via StandardParser + GraphBuilder"]
        WV2["tick/run/init on layout::ForceSimulation"]
        WV3["getGraphData() behind debug-serde"]
        WV4["pinning, filtering, statistics"]
    end
    subgraph NggExplorer["NggExplorer - narrow, allocation-free, src/bindings/explorer.rs:17, feature ngg1"]
        NG1["loadCsr(bytes) via ngg1::Ngg1 parse"]
        NG2["tick() drives layout::csr_sim::CsrSimulation"]
        NG3["positionsPtr/positionsLen<br/>zero-copy Float32Array over wasm.memory.buffer"]
    end
    Worker["Rendering worker, JS"] -->|"per-frame, no per-frame alloc"| NggExplorer
    MainThread["Main-thread consumer, JS"] -->|"structured JSON in/out"| WebVowl
```
- INVARIANT: `NggExplorer` never allocates across the WASM boundary per frame — only the position pointer crosses (src/bindings/explorer.rs:9).

## VW-01.8 `DebugFlags` — opt-in console tracing
```mermaid
classDiagram
    class DebugFlags {
        +bool log_positions
        +bool log_forces
        +bool log_velocities
        +new() DebugFlags src/debug.rs:33
        +enable_all() src/debug.rs:47
        +should_log(iteration usize) bool src/debug.rs:58
    }
    class ForceSimulation {
        -DebugFlags debug_flags
        +enable_debug() src/layout/simulation.rs:50
        +set_debug_flags(flags) src/layout/simulation.rs:55
    }
    ForceSimulation --> DebugFlags
```
- Native builds log via `web_sys::console` only under `#[cfg(target_arch = "wasm32")]` (src/debug.rs:71); non-wasm32 builds compile the same call sites to no-ops (src/debug.rs:171-183).

## VW-01.9 `SvgRenderer` — native-only diagnostic path
```mermaid
sequenceDiagram
    autonumber
    participant C as caller, native, tests/benches
    participant R as SvgRenderer<br/>src/render/mod.rs:20
    participant G as VowlGraph
    C->>R: SvgRenderer::new(width, height)<br/>src/render/mod.rs:28
    C->>R: with_padding(padding)<br/>src/render/mod.rs:37
    C->>R: render(&graph), Renderer trait<br/>src/render/mod.rs:116
    R->>G: normalize_coords(x, y, graph)<br/>src/render/mod.rs:80
    R-->>C: SVG string
```
- `SvgRenderer` has no `#[wasm_bindgen]` surface — it is reachable only from Rust callers (tests, benches, native embedding), not from the published JS API (see VW-04).
