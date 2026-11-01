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
  - ../vowl-wasm/src/layout/csr_sim.rs
  - ../vowl-wasm/src/layout/simd.rs
  - ../vowl-wasm/src/layout/mod.rs
  - ../vowl-wasm/src/ngg1.rs
  - ../vowl-wasm/src/interaction/mod.rs
  - ../vowl-wasm/src/layout/simulation.rs
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
verified_commit: 28af4365a68d9aacfd7c04f63299cc96ad717ce3
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
- Crate-level `deny(unsafe_code)` is locally overridden for SIMD intrinsics (src/layout/simd.rs:24); it is not a guarantee of no unsafe code. `error` is not re-exported as `pub mod`; only `Result`/`VowlError` are re-exported at the crate root (`src/lib.rs:157`).
- `SvgRenderer::render` writes edge comments, raw node labels and coordinates; it does not call the fitting helper, filter hidden nodes or draw edge lines (src/render/mod.rs:116-173). `render::SvgRenderer` (src/render/mod.rs:20) exists but is not wired to any `#[wasm_bindgen]` binding — a Rust diagnostic path compiled for either target, not referenced from `bindings/mod.rs`, so unavailable through the published JS API (see vowl-wasm/04-js-api-examples-and-consumers.md).
- All five `graph::` submodules (`src/graph/mod.rs`) and all five `layout::` submodules (`src/layout/mod.rs`) are always compiled — neither file carries a `#[cfg(feature = ...)]` gate, and `ontology::` is now always compiled too (`src/ontology/mod.rs:6-7`: only `model` and `parser`; the `markdown-ontology` feature with its parser, validator and loader was deleted, see `docs/adr/ADR-001-shared-ontology-rendering-contract.md:25`). Only `bindings::explorer` and `bindings::ngg` are gated, both on `ngg1` (`src/bindings/mod.rs:3-9`); the feature mapping is detailed in VW-01.4.

## VW-01.4 Feature flags and what they gate
```mermaid
flowchart LR
    subgraph Always["always compiled, default = []"]
        CORE["OWL/JSON parsing, graph model,<br/>force layout, pinning, statistics<br/>Cargo.toml:58-61"]
    end
    NGG1F["ngg1<br/>Cargo.toml:67 dep:ontomark-ngg1 at Cargo.toml:46"] --> NGGMOD["src/ngg1.rs re-export + NggExplorer<br/>bindings/explorer.rs:18 + NggGraph, decodeNgg1, encodeNgg1<br/>bindings/ngg.rs:17-18"]
    INTF["interaction<br/>Cargo.toml:75"] --> INTMOD["interaction module<br/>+ checkNodeClick binding"]
    DBGF["debug-serde<br/>Cargo.toml:79"] --> DBGMOD["WebVowl.getGraphData()<br/>bindings/mod.rs:176"]
    PARF["parallel<br/>Cargo.toml:84 dep:rayon, native only"] --> PARMOD["enables rayon dependency;<br/>current source has no parallel loading implementation"]
    SIMDF["simd<br/>Cargo.toml:87, needs simd128 target-feature"] --> SIMDMOD["layout/simd.rs SIMD128 kernels"]
    NGG1F -. "not in bundle alone" .-> BUNDLE
    INTF --> BUNDLE["published npm bundle<br/>README.md:201 --features ngg1,interaction"]
```
- `default = []` (Cargo.toml:61) disables named optional features. The `parallel` feature currently adds rayon but no source call uses it; the manifest comment describes an absent loading implementation. Optional dependency selection alone is not parallel execution.
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
    Note right of M: no-op, reserved for panic hook<br/>src/lib.rs:166
    JS->>V: version()
    V-->>JS: env!("CARGO_PKG_VERSION")<br/>src/lib.rs:178
```

## VW-01.6 `VowlError` to `JsValue` at every fallible binding
```mermaid
flowchart TB
    RESULT["Result&lt;T&gt; alias<br/>std::result::Result with VowlError<br/>src/error.rs:7"] --> ERR["VowlError variants: ParseError, InvalidData,<br/>GraphError, LayoutError, RenderError,<br/>BindingError, InteractionError<br/>src/error.rs:11-38"]
    JSON["serde_json::Error<br/>From conversion: src/error.rs:49"] --> ERR
    ERR --> JS["JsValue containing rendered error text<br/>From conversion: src/error.rs:43"]
```
- Fallible `WebVowl` operations such as `loadOntology` return `std::result::Result<T, JsValue>` (e.g. `load_ontology` src/bindings/mod.rs:56) — the crate's typed `VowlError` never crosses the boundary, only its rendered message. Getters, booleans and setters also have non-Result signatures (src/bindings/mod.rs:121-152).

## VW-01.7 Two entry points: `WebVowl` vs `NggExplorer`
```mermaid
flowchart TB
    WORKER["JS rendering worker"] --> NG["NggExplorer, feature ngg1<br/>src/bindings/explorer.rs:18"]
    NG --> LOAD["loadCsr: NGG1 parse<br/>tick: CsrSimulation"]
    NG --> PTR["positionsPtr and positionsLen<br/>direct view over wasm.memory.buffer"]
    NG --> TUNE["setTheta, pinNode, unpinNode, reheat<br/>src/bindings/explorer.rs:112-137"]
    MAIN["JS main-thread consumer"] --> WV["WebVowl, always available<br/>src/bindings/mod.rs:27"]
    WV --> JSON["loadOntology: parser and builder<br/>tick/run/init: ForceSimulation"]
    WV --> STATE["Typed positions, metadata,<br/>pinning, filtering, statistics<br/>getGraphData needs debug-serde"]
    OWNED["Separate owned module APIs"] --> DECODE["decodeNgg1<br/>src/bindings/ngg.rs:119"]
    OWNED --> ENCODE["encodeNgg1<br/>src/bindings/ngg.rs:182"]
    OWNED --> CONST["ngg1Constants<br/>src/bindings/ngg.rs:240"]
```
- The pointer-view path avoids copying positions at that boundary (src/bindings/explorer.rs:75-90). This is not an allocation-free engine: ticks allocate force buffers and tree storage (src/layout/csr_sim.rs:386-387, src/layout/csr_sim.rs:186-187); `writePositionsInto` also uses allocating generated glue. Separate owned codec functions are not explorer methods.

## VW-01.8 `DebugFlags` — opt-in console tracing
```mermaid
flowchart TB
    SIM["ForceSimulation<br/>enable_debug and set_debug_flags<br/>src/layout/simulation.rs:50-55"] --> FLAGS["DebugFlags<br/>log_positions, log_forces, log_velocities<br/>new: src/debug.rs:33<br/>enable_all: src/debug.rs:47"]
    FLAGS --> LOG["should_log(iteration)<br/>iteration modulo log_interval<br/>zero interval panics<br/>src/debug.rs:58-59"]
```
- WASM builds log via `web_sys::console` only under `#[cfg(target_arch = "wasm32")]` (src/debug.rs:71); non-wasm32 logging helpers are no-ops (src/debug.rs:171-183). The public Rust `log_interval` is unchecked: zero panics in `should_log` (src/debug.rs:58-59).

Historical test qualification — 2026-09-07, re-read at 89d1a0a on 2026-10-29: the repository now has one local record, ADR-001 (accepted, implemented for NGG1 ownership, active: `docs/adr/ADR-001-shared-ontology-rendering-contract.md:4-6`), whose own caveat is that the plugin, the TypeScript reader consolidation and the wider acceptance evidence are not claimed. Its `Cargo.toml` and code are the implementation evidence; WasmVOWL and embedded explorer copies have separate governance and source identities. `cargo test --locked --offline --lib` passed 136 default-feature tests at the earlier stamp; that count was not re-run at 89d1a0a and the removed markdown and validator tests are no longer in it. That run covers native logic, not optional `ngg1`, `interaction`, browser loading or a deployed consumer.
