---
id: VC-51
title: XR visual budgets and engine qualification gates
area: visionclaw
governing:
  - ../project/docs/XR-client.md
adrs: [ADR-2167, ADR-2165, ADR-2159, ADR-2166]
sources:
  - ../project/docs/XR-client.md
  - ../project/xr-client/rust/src/frame_budget.rs
  - ../project/xr-client/scripts/frame_budget_pass.gd
  - ../project/xr-client/scripts/eye_candy.gd
  - ../project/xr-client/scripts/emulated_volume.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/perf/benchmark.gd
  - ../project/xr-client/project.godot
  - ../project/xr-client/rust/Cargo.toml
  - ../project/xr-client/visionclaw_xr_gdext.gdextension
  - ../project/xr-client/materials/edge_flow_common.gdshaderinc
  - ../project/xr-client/materials/edge_flow.tres
  - ../project/xr-client/scripts/node_lod.gd
verified_commit: {visionclaw: 86cdd7635582}
---

## For developers

The [XR budget audit](../../estate-review/2026-11-01-xr-budget-and-engine-qualification.md) traces ADR-2167 and ADR-2165 in full. Native arithmetic and isolated headless controller tests are separate from full-client, renderer and headset qualification.

## For the business

Visual polish trades some graph detail for frames, sky and motes. The controls need reliable timing and complete accounting to preserve a comfortable, readable workspace; a desktop or headless pass cannot establish both-eye headset behaviour.

## VC-51.1 Requested extras reserve a bounded share before near detail

```mermaid
flowchart TD
    HARD["Far tiers and hard triangle minimums<br/>frame_budget.rs:255,269"] --> PRIORITY["Route, cloud, bursts and hulls<br/>frame_budget.rs:274,285,289,297"]
    PRIORITY --> PRESS["Pressure includes triangle overrun and unmet layers<br/>frame_budget.rs:328,329<br/>frame_budget.rs:330,331,332"]
    PRESS --> ALLOW["Without pressure: buy requested tier before near detail<br/>frame_budget.rs:335,338,340"]
    PRESS --> LAST["Under pressure: extras buy only leftovers<br/>frame_budget.rs:356,359"]
    ALLOW --> NEAR["Gem then cylinder share<br/>frame_budget.rs:350,352"]
    NEAR --> SPLIT["Split per-volume caps; floors precede weights<br/>frame_budget.rs:506,524,527"]
    CALLS["Call guard permits zero-added-call tiers above limit<br/>frame_budget.rs:320"] --> ALLOW
```

**What it shows.** Extras have an approved allowance, with a last-claim fallback under triangle or layer pressure. **Why it is this way.** The owner accepted fewer near cylinders and some gems; existing call overrun is reported but does not withdraw a zero-added-call allowance.

## VC-51.2 Live demand and drawn-frame counts bound the guarantee

```mermaid
flowchart TD
    SCENE["Count named graph and volume calls<br/>graph_scene.gd:3161,3171"] --> CALLS["Add agent, link and hull calls<br/>graph_scene.gd:3177,3181,3183"]
    CALLS --> INPUT["Pass assembled call demand to allocator<br/>graph_scene.gd:3184,3187"]
    OTHER["Measured other triangles, but no other-call remainder<br/>frame_budget_pass.gd:102,125"] --> INPUT
    BENCH["Benchmark separately calibrates other calls<br/>benchmark.gd:230,572"]
    VISIBLE["Controller counts every visible frame<br/>eye_candy.gd:462,466,469"] --> PRICE["Cost clamps at eight frames<br/>frame_budget.rs:170"]
    VISIBLE --> DRAW["Style switch reaches all grouped frames<br/>emulated_volume.gd:138,141"]
```

**What it shows.** Whole-frame call compliance depends on complete inputs; nine constructed frames also exceed the clamped price by 24 triangles. **Why it is this way.** The live scene and benchmark build demand differently, and pricing alone does not enforce a visible-frame limit.

## VC-51.3 Sparse timing can delay automatic quality reduction

```mermaid
flowchart TD
    PROBE["Startup: majority of readings must be nonzero<br/>eye_candy.gd:356,357"] --> GPU["Feed valid GPU milliseconds and that frame delta<br/>eye_candy.gd:368,380"]
    GPU --> WINDOW["Governor accumulates sampled elapsed time<br/>eye_candy.gd:169,170,171"]
    ZERO["Missing readings increment consecutive-zero run<br/>eye_candy.gd:369,370"] --> FALLBACK["More than 180 zeros selects frame mode<br/>eye_candy.gd:371,373,374,375"]
    GPU --> RESET["Positive sample resets zero run<br/>eye_candy.gd:378"]
    WINDOW --> REQUEST["Automatic or manually pinned request<br/>eye_candy.gd:254,257,258,261"]
    REQUEST --> GRANT["Draw at most grant and request<br/>eye_candy.gd:334,335"]
    LOW["Low cost immediately withdraws demand<br/>eye_candy.gd:242,247"] --> REQUEST
```

**What it shows.** Forty seconds of sparse loaded samples can leave tier 4 requested while neither control window completes. **Why it is this way.** Zero samples do not advance the governor; a fresh-sample coverage rule is separate from the existing consecutive-zero fallback and manual quality pin.

## VC-51.4 Engine selection does not close the stereo gate

```mermaid
flowchart TD
    TARGET["Recorded target: Godot 4.8-beta2<br/>XR-client.md:97,98"] --> CONTROLLER["AUDIT: isolated headless tests on both engines<br/>18 retained tests per engine; no native extension"]
    ROLLBACK["4.6.1 rollback remains separate<br/>XR-client.md:131"]
    API["godot crate 0.2; minimum engine 4.3<br/>Cargo.toml:21<br/>visionclaw_xr_gdext.gdextension:4"] --> LOAD["Historical full extension-load evidence<br/>current audit does not rerun it"]
    RENDER["Compatibility renderer; MSAA disabled in project<br/>project.godot:56,59"] --> STEREO["Both-eye headset qualification still partial<br/>XR-client.md:122"]
    CONTROLLER -.-> STEREO
    LOAD -.-> STEREO
    STEREO -.-> ROLLBACK
```

**What it shows.** Target engine, API compatibility, headless behaviour and both-eye rendering have different evidence. **Why it is this way.** The engine change reopens the stereo invariant; an unchanged extension or a successful desktop mirror does not prove the right eye works.

## VC-51.5 The gold edge breathe is one shader term, mirrored to the ribbons

```mermaid
flowchart TD
    COL["Gold flow colour<br/>edge_flow_common.gdshaderinc:5"] --> UNI["Breathe period 36 s, depth 0.8<br/>edge_flow_common.gdshaderinc:16,17<br/>edge_flow.tres:14,15"]
    UNI --> SWELL["Cosine swell scales albedo and alpha, except inferred and subclass edges<br/>edge_flow_common.gdshaderinc:99,100,101,102"]
    SWELL --> BEAT["Beat pulse gain applied after the breathe<br/>edge_flow_common.gdshaderinc:104"]
    UNI --> MIRROR["Ribbon material copies breathe uniforms with the rest<br/>node_lod.gd:20,23,98,103"]
```

**What it shows.** The breathe is time-driven in the shader, so reduced motion leaves it on, and the trough is 0.04 of the peak. It multiplies with the separate beat pulse. **Why it is this way.** The owner asked on 2026-11-01 for a slow visible breathe. 36 s divides Godot's 3600 s TIME rollover exactly (`edge_flow_common.gdshaderinc:10`).
