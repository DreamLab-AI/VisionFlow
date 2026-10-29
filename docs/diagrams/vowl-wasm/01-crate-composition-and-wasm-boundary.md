---
id: VW-01
title: Crate composition, modules and the WASM boundary
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
adrs: [vowl-wasm:ADR-001]
sources:
  - ../vowl-wasm/Cargo.toml
  - ../vowl-wasm/src/lib.rs
  - ../vowl-wasm/src/error.rs
  - ../vowl-wasm/src/debug.rs
  - ../vowl-wasm/src/render/mod.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/src/bindings/ngg.rs
  - ../vowl-wasm/src/graph/mod.rs
  - ../vowl-wasm/src/ontology/mod.rs
  - ../vowl-wasm/src/layout/mod.rs
  - ../vowl-wasm/src/ngg1.rs
  - ../vowl-wasm/src/interaction/mod.rs
  - ../vowl-wasm/src/layout/simulation.rs
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
verified_commit: 89d1a0a32eec16e79a7fa507eaa6f8fcb92e4f0b
---

## VW-01.1 Top-level module tree
```mermaid
flowchart TB
    LIB["lib.rs<br/>src/lib.rs:136-137 #!deny(missing_docs, unsafe_code)"]
    LIB --> BIND["bindings/<br/>src/bindings/mod.rs, explorer.rs, ngg.rs"]
    LIB --> DEBUG["debug<br/>src/debug.rs"]
    LIB --> GRAPH["graph/<br/>src/graph/mod.rs:6"]
    LIB --> LAYOUT["layout/<br/>src/layout/mod.rs:12"]
    LIB --> ONT["ontology/<br/>src/ontology/mod.rs:6"]
    LIB --> RENDER["render<br/>src/render/mod.rs"]
    LIB -. "feature: ngg1" .-> NGG1["ngg1, re-export of ontomark-ngg1<br/>src/ngg1.rs:5"]
    LIB -. "feature: interaction" .-> INTER["interaction<br/>src/interaction/mod.rs"]
    LIB --> ERR["error, private<br/>src/error.rs:6"]
```
- `error` is not re-exported as `pub mod`; only `Result`/`VowlError` are re-exported at the crate root (`src/lib.rs:157`).
- `render::SvgRenderer` (src/render/mod.rs:20) exists but is not wired to any `#[wasm_bindgen]` binding — a native-only diagnostic path, not referenced from `bindings/mod.rs`, so unreachable from the published JS API (see vowl-wasm/04-js-api-examples-and-consumers.md).
- All five `graph::` submodules (`src/graph/mod.rs`) and all five `layout::` submodules (`src/layout/mod.rs`) are always compiled — neither file carries a `#[cfg(feature = ...)]` gate, and `ontology::` is now always compiled too (`src/ontology/mod.rs:6-7`: only `model` and `parser`; the `markdown-ontology` feature with its parser, validator and loader was deleted, see `docs/adr/ADR-001-shared-ontology-rendering-contract.md:25`). Only `bindings::explorer` and `bindings::ngg` are gated, both on `ngg1` (`src/bindings/mod.rs:3-9`); the feature mapping is detailed in VW-01.4.

## VW-01.4 Feature flags and what they gate
```mermaid
flowchart LR
    subgraph Always["always compiled, default = []"]
        CORE["OWL/JSON parsing, graph model,<br/>force layout, pinning, statistics<br/>Cargo.toml:58-61"]
    end
    NGG1F["ngg1<br/>Cargo.toml:67 dep:ontomark-ngg1 at Cargo.toml:46"] --> NGGMOD["src/ngg1.rs re-export + NggExplorer<br/>bindings/explorer.rs:17 + NggGraph, decodeNgg1, encodeNgg1<br/>bindings/ngg.rs:17-18"]
    INTF["interaction<br/>Cargo.toml:75"] --> INTMOD["interaction module<br/>+ checkNodeClick binding"]
    DBGF["debug-serde<br/>Cargo.toml:79"] --> DBGMOD["WebVowl.getGraphData()<br/>bindings/mod.rs:176"]
    PARF["parallel<br/>Cargo.toml:84 dep:rayon, native only"] --> PARMOD["ontology loading data-parallelism"]
    SIMDF["simd<br/>Cargo.toml:87, needs simd128 target-feature"] --> SIMDMOD["layout/simd.rs SIMD128 kernels"]
    NGG1F -. "not in bundle alone" .-> BUNDLE
    INTF --> BUNDLE["published npm bundle<br/>README.md:201 --features ngg1,interaction"]
```
- INVARIANT: `default = []` (Cargo.toml:61) — nothing optional compiles unless asked for.
- The `markdown-ontology` feature and its `regex` dependency no longer exist: the manifest has no `regex` and no such feature (Cargo.toml:26-46, Cargo.toml:57-84), and the README records the removal, with Markdown parsing left to `ontomark-parse` (README.md:204-205). The NGG1 codec is `ontomark-ngg1` 0.1.0 from crates.io, optional behind `ngg1` (Cargo.toml:45-46).

## VW-01.5 `init()` and `version()` — module entry, called once by wasm-bindgen
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS host<br/>await init()
    participant WB as wasm-bindgen glue
    participant M as vowl_wasm::init<br/>src/lib.rs:166
    participant V as vowl_wasm::version<br/>src/lib.rs:177
    JS->>WB: instantiate module
    WB->>M: #[wasm_bindgen(start)] init()
    Note right of M: no-op today, reserved for panic-hook install<br/>src/lib.rs:157
    JS->>V: version()
    V-->>JS: env!("CARGO_PKG_VERSION")<br/>src/lib.rs:178
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
- Every `#[wasm_bindgen]` method on `WebVowl` returns `std::result::Result<T, JsValue>` (e.g. `load_ontology` src/bindings/mod.rs:56) — the crate's typed `VowlError` never crosses the boundary, only its rendered message.

## VW-01.7 Two entry points: `WebVowl` vs `NggExplorer`
```mermaid
flowchart TB
    subgraph WebVowl["WebVowl - general purpose, always available, src/bindings/mod.rs:27"]
        WV1["loadOntology(json) via StandardParser + GraphBuilder"]
        WV2["tick/run/init on layout::ForceSimulation"]
        WV3["getGraphData() behind debug-serde"]
        WV4["pinning, filtering, statistics"]
    end
    subgraph NggExplorer["NggExplorer - narrow, allocation-free, src/bindings/explorer.rs:17, feature ngg1"]
        NG1["loadCsr(bytes) via ngg1::Ngg1 parse"]
        NG2["tick() drives layout::csr_sim::CsrSimulation"]
        NG3["positionsPtr/positionsLen<br/>zero-copy Float32Array over wasm.memory.buffer"]
        NG4["setTheta and reheat, Barnes-Hut opening angle and re-heat of a settled scope<br/>src/bindings/explorer.rs:111-118"]
        NG5["decodeNgg1, encodeNgg1, ngg1Constants: owned snapshots through the ontomark codec<br/>src/bindings/ngg.rs:119 and src/bindings/ngg.rs:182 and src/bindings/ngg.rs:240"]
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

Audit qualification — 2026-09-07, re-read at 89d1a0a on 2026-10-29: the repository now has one local record, ADR-001 (accepted, implemented for NGG1 ownership, active: `docs/adr/ADR-001-shared-ontology-rendering-contract.md:4-6`), whose own caveat is that the plugin, the TypeScript reader consolidation and the wider acceptance evidence are not claimed. Its `Cargo.toml` and code are the implementation evidence; WasmVOWL and embedded explorer copies have separate governance and source identities. `cargo test --locked --offline --lib` passed 136 default-feature tests at the earlier stamp; that count was not re-run at 89d1a0a and the removed markdown and validator tests are no longer in it. That run covers native logic, not optional `ngg1`, `interaction`, browser loading or a deployed consumer.
