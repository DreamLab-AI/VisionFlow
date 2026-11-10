---
id: VC-51
title: XR visual budgets and engine qualification gates
area: visionclaw
governing:
  - ../project/docs/XR-client.md
adrs: [ADR-2167, ADR-2165, ADR-2159, ADR-2166, ADR-2170, ADR-2188, ADR-2179, ADR-2181, ADR-2183]
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
  - ../project/docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md
  - ../project/xr-client/scripts/overview_lod.gd
  - ../project/xr-client/materials/node_impostor.gdshader
  - ../project/xr-client/materials/edge_ribbon.gdshader
  - ../project/xr-client/materials/node_halo_quad.gdshader
  - ../project/xr-client/scripts/connection_volume_source.gd
  - ../project/xr-client/scripts/hud.gd
  - ../project/xr-client/scripts/highlight_writer.gd
  - ../project/xr-client/rust/src/highlight.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/xr-client/rust/src/render_store.rs
  - ../project/xr-client/rust/src/render_store_pack.rs
  - ../project/xr-client/tests/unit/test_overview_lod.gd
  - ../project/xr-client/tests/unit/test_label_budget.gd
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

The [XR budget audit](../../estate-review/2026-11-01-xr-budget-and-engine-qualification.md) traces ADR-2167 and ADR-2165 in full. Native arithmetic and isolated headless controller tests are separate from full-client, renderer and headset qualification. VC-51.6 to VC-51.8 add the ADR-2188 overview tier (2026-11-06): pixel-floored impostors and screen-width ribbons, the per-volume edge limit with its HUD count, and the Rust highlight mask; their evidence is headless GUT and Rust units only.

## For the business

Visual polish trades some graph detail for frames, sky and motes. The controls need reliable timing and complete accounting to preserve a comfortable, readable workspace; a desktop or headless pass cannot establish both-eye headset behaviour.

## VC-51.1 Requested extras reserve a bounded share before near detail

```mermaid
flowchart TD
    HARD["Far tiers and hard triangle minimums<br/>frame_budget.rs:383,397"] --> PRIORITY["Route, cloud, bursts and hulls<br/>frame_budget.rs:402,413,417,425"]
    PRIORITY --> PRESS["Pressure includes triangle overrun and unmet layers<br/>frame_budget.rs:459,460<br/>frame_budget.rs:461,462,463"]
    PRESS --> ALLOW["Without pressure: buy requested tier before near detail<br/>frame_budget.rs:466,469,471"]
    PRESS --> LAST["Under pressure: extras buy only leftovers<br/>frame_budget.rs:487,490"]
    ALLOW --> NEAR["Gem then cylinder share<br/>frame_budget.rs:481,483"]
    NEAR --> SPLIT["Split per-volume caps; floors precede weights<br/>frame_budget.rs:643,661,664"]
    SPLAT["Splat draw calls reserved before extras, at most two volumes,<br/>splat triangles reported outside TRI_LIMIT<br/>frame_budget.rs:255,261,449,450,526"] --> CALLS
    CALLS["Call guard over base plus splat calls permits zero-added-call tiers above limit<br/>frame_budget.rs:451"] --> ALLOW
```

**What it shows.** Extras have an approved allowance, with a last-claim fallback under triangle or layer pressure; splat volumes now take their draw calls first. **Why it is this way.** The owner accepted fewer near cylinders and some gems; existing call overrun is reported but does not withdraw a zero-added-call allowance.
**Tension (ADR-2170 vs PRD-008):** splat triangles are never charged against the 100 000-triangle budget (`xr-client/rust/src/frame_budget.rs:38-42`), so an entered tier-3 splat volume adds 200 000 triangles beside it (`xr-client/rust/src/frame_budget.rs:69`, `xr-client/rust/src/frame_budget.rs:203-205`); the owner accepted this as ADR-2170 decision 7, and only a headset receipt can say whether 90 fps survives it.

## VC-51.2 Live demand and drawn-frame counts bound the guarantee

```mermaid
flowchart TD
    SCENE["Count named graph and volume calls<br/>graph_scene.gd:3586,3596"] --> CALLS["Add agent, link and hull calls<br/>graph_scene.gd:3606,3610,3612"]
    CALLS --> INPUT["Pass assembled call demand to allocator<br/>graph_scene.gd:3613,3616"]
    OTHER["Measured other triangles, but no other-call remainder<br/>frame_budget_pass.gd:127,151"] --> INPUT
    BENCH["Benchmark separately calibrates other calls<br/>benchmark.gd:230,572"]
    VISIBLE["Controller counts every visible frame<br/>eye_candy.gd:462,466,469"] --> PRICE["Cost clamps at eight frames<br/>frame_budget.rs:183"]
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
    TARGET["Recorded target: Godot 4.8-beta2<br/>XR-client.md:115,116"] --> CONTROLLER["AUDIT: isolated headless tests on both engines<br/>18 retained tests per engine; no native extension"]
    ROLLBACK["4.6.1 rollback remains separate<br/>XR-client.md:149"]
    API["godot crate 0.2; minimum engine 4.3<br/>Cargo.toml:21<br/>visionclaw_xr_gdext.gdextension:4"] --> LOAD["Historical full extension-load evidence<br/>current audit does not rerun it"]
    RENDER["Compatibility renderer; MSAA disabled in project<br/>project.godot:56,59"] --> STEREO["Both-eye headset qualification still partial<br/>XR-client.md:140"]
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

## VC-51.6 The overview tier is pixel-floored impostors and screen-width ribbons fed one px_per_rad (ADR-2188 D1, D2)

```mermaid
flowchart TD
    POLL["_update_px_per_rad, once a second<br/>graph_scene.gd:1820,1822,1125"] --> XR{"XR interface live?<br/>graph_scene.gd:1833"}
    XR -->|yes| EYE["eye 0 projection FOV against the per-eye render width<br/>graph_scene.gd:1834,1836,1837"]
    XR -->|no| DESK["camera vertical FOV against viewport height<br/>graph_scene.gd:1838,1839"]
    EYE --> PPR["px_per_rad = size / (2 tan(fov / 2)), 750 fallback<br/>overview_lod.gd:18,46,47,48,49"]
    DESK --> PPR
    PPR --> MAT["written only when it changes, to the impostor and ribbon materials<br/>graph_scene.gd:1840,1844,1845"]
    MAT --> SHARE["every volume's MultiMesh reuses the template material<br/>connection_volume_source.gd:263"]
    SHARE --> IMP["impostor: floor = min(3 px of depth, 5 cm), grow only<br/>node_impostor.gdshader:45,46,60,61,62"]
    SHARE --> RIB["ribbon: 1.5 px wide at each end, never thinner than the cylinder,<br/>alpha 1 to 0.25 over 50 to 150 px projected length<br/>edge_ribbon.gdshader:33,55,56,63,64"]
    IMP --> EMPH["emphasis is an opaque outline ring, not glow<br/>node_impostor.gdshader:92,93"]
    MIRROR["GDScript copies of the same maths<br/>overview_lod.gd:60,63,74,76,80,81"] -.-> IMP
    MIRROR -.-> RIB
    PIN["GUT pins the copies and the shader DEFAULTS, not the shader formulas<br/>test_overview_lod.gd:32,44,52,78,79,82"] --> MIRROR
```

**What it shows.** One per-second measurement drives both overview shaders through materials every volume shares, so a node never shrinks below 3 px and an edge is a 1.5 px line whose alpha falls with projected length. **Why it is this way.** ADR-2188 carries the desktop overview contract (ADR-2179, ADR-2181, ADR-2182) into the headset without bloom; headless Godot cannot compile shaders, so the maths is mirrored in GDScript for GUT (`xr-client/tests/unit/test_overview_lod.gd:79`).
**Debt (parallel):** `overview_lod.gd` re-implements the impostor floor, ribbon width and length fade the shaders compute (`xr-client/scripts/overview_lod.gd:6-8`, `xr-client/materials/node_impostor.gdshader:60-62`, `xr-client/materials/edge_ribbon.gdshader:55-65`); GUT pins the copy and the shader defaults (`xr-client/tests/unit/test_overview_lod.gd:78-91`) but a change to a shader formula would pass it.
**Open (risk):** `px_per_rad` is the centre-of-view value from eye 0's FOV (`xr-client/scripts/graph_scene.gd:1836-1837`), so on an asymmetric headset frustum the pixel floor and line width are approximate, as the doc concedes unverified (`docs/XR-client.md:825-826`).
**Tension (ADR-2188 D3 vs halo):** ADR-2188 says emphasis is "never glow" (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:33`), yet the halo quad still draws additively (`xr-client/materials/node_halo_quad.gdshader:18`); the doc reconciles this as emissive geometry rather than post-process bloom (`docs/XR-client.md:847-849`).

## VC-51.7 Each volume sizes its edge request from its own data and the HUD counts what it dropped (ADR-2188 D2, ADR-2179 P4)

```mermaid
flowchart TD
    PAIRS["content_pairs re-read per generation or refresh<br/>connection_volume_source.gd:124,130"] --> COUNT["edge_count_for_mask: edges with both ends in the class mask<br/>connection_volume_source.gd:131; binary_protocol.rs:2437,2438"]
    COUNT --> LIMIT["edge_draw_limit: total times the Look Edges share,<br/>at most EDGE_DRAW_CEILING 20 000<br/>overview_lod.gd:38,107,108,109"]
    LIMIT --> FETCH["edges_for_mask, heaviest first, and edge_total kept<br/>connection_volume_source.gd:132,133,134"]
    FETCH --> PACK["near cylinders plus far ribbons packed per frame<br/>connection_volume_source.gd:165,168"]
    PACK --> DRAWN["drawn_edges and dropped_edges<br/>connection_volume_source.gd:175,181"]
    DRAWN --> SUM["_update_edge_readout sums visible volumes,<br/>touches the HUD only when a count changes<br/>graph_scene.gd:1791,1797,1807,1808,1809,1814"]
    SUM --> TXT["Edges N of M drawn, empty when nothing dropped<br/>overview_lod.gd:118,119,121"]
    TXT --> BADGE["EdgeBadge in the HUD header<br/>hud.gd:466,2549"]
    OLD["NODE_SAFETY_CEILING and EDGE_SAFETY_CEILING 20 000<br/>graph_scene.gd:51,52"] --> BUD["_recompute_instance_budgets<br/>graph_scene.gd:4766,4767"]
    BUD --> STATUS["status line nodes drawn at most N<br/>graph_scene.gd:2513"]
    BUD --> RANK["_rerank_edges writes _edge_pairs, which no draw path reads<br/>graph_scene.gd:4798,4829,4853,4855"]
    RANK -.-> D1["DEBT: dead ranked-edge path and a misleading cap warning<br/>graph_scene.gd:4840"]
```

**What it shows.** The cap a viewer sees is the triangle-budget ceiling, applied per volume and reported in the header, not the old scene-wide safety constants. **Why it is this way.** ADR-2179 P4 forbids a silent cap; ribbons are 2 triangles, so 20 000 of them are 40 000 of the 100 000 budget (`xr-client/scripts/overview_lod.gd:33-38`).
**Debt (loose end):** `NODE_SAFETY_CEILING` and `EDGE_SAFETY_CEILING` (`xr-client/scripts/graph_scene.gd:51-52`) now feed only the status text (`xr-client/scripts/graph_scene.gd:2513`) and `_rerank_edges`, whose `_edge_pairs` output has no reader (`xr-client/scripts/graph_scene.gd:4853-4855`), so its "rendering N" warning (`xr-client/scripts/graph_scene.gd:4840`) describes nothing drawn.
**Drift:** the ceiling comment says it stops a payload blowing "the MultiMesh instance buffers" (`xr-client/scripts/graph_scene.gd:45-50`), while the overview rule says there is no fixed buffer capacity to clip against (`xr-client/scripts/overview_lod.gd:103-105`); `xr-client/perf/benchmark.gd:140-143` still sizes its default edge load from the scene constant.

## VC-51.8 The highlight mask is a Rust byte per slot, applied on the pack path (ADR-2188 D4, ADR-2183)

```mermaid
flowchart TD
    SCN["scene writers: search hits, hover held 250 ms, selection and up to 64 neighbours<br/>graph_scene.gd:2102,2126,2138,2139"] --> HW["HighlightWriter.write: dedup per connection and bit,<br/>never FILTERED_OUT<br/>highlight_writer.gd:23,37,41,45"]
    HW --> GD["gdext set_highlight_ids<br/>binary_protocol.rs:1646,1651"]
    GD --> RS["RenderStore.set_highlight_ids masks out FILTERED_OUT<br/>render_store.rs:2585,2586"]
    FILT["received nodeFilter writes FILTERED_OUT, greys instead of removing<br/>render_store.rs:2552,2576"] --> MASK
    RS --> MASK["HighlightMask: one byte per slot plus a version<br/>highlight.rs:21,29,116,117,118"]
    MASK --> LVL["level: pointer outranks filter, filter outranks search and neighbours<br/>highlight.rs:50,51,53,55"]
    LVL --> APPLY["apply on the per-frame pack path after the plan records the unmasked instance<br/>render_store_pack.rs:164,241"]
    APPLY --> NODE["grey: desaturate, no halo; emphasis: lift, x1.35, outline flag in custom.a<br/>highlight.rs:83,86,87,88,92,93,94"]
    MASK --> EDGE["edge custom.r = 1 when an end is grey, ribbons re-emitted once per mask version<br/>render_store_pack.rs:431,432,452"]
    EDGE --> SH["ribbon draws a grey edge at grey_alpha 0.3<br/>edge_ribbon.gdshader:37,65"]
    NODE --> HALO["halo quad discards at level 0<br/>node_halo_quad.gdshader:68"]
    PINB["GUT reads highlight.rs and pins the GDScript bit copies<br/>test_label_budget.gd:162,164,165"] --> HW
```

**What it shows.** Pointer and query state reach the renderer as one byte per node slot, folded to grey, normal or emphasised at pack time, so a hover never invalidates a pack plan. **Why it is this way.** ADR-2183 makes filters grey out rather than remove, and ADR-2188 D3 rules out glow, so emphasis is colour, size and an outline (`xr-client/rust/src/highlight.rs:1-18`).
**Debt (parallel):** the five bit values are copied into GDScript with a "must match" comment (`xr-client/scripts/highlight_writer.gd:8-21`, `xr-client/rust/src/highlight.rs:21-29`), held only by a source-text GUT check (`xr-client/tests/unit/test_label_budget.gd:162-170`); `HighlightWriter.neighbours_of` mirrors the Rust `neighbours_in` for clients without it (`xr-client/scripts/highlight_writer.gd:79-98`).
**Drift:** the ADR-2188 section of `docs/XR-client.md` places `_update_px_per_rad` at graph_scene line 1805 and `set_edge_readout` at hud line 2266 (`docs/XR-client.md:820-821`, `docs/XR-client.md:837-838`), but at this pin they sit at `xr-client/scripts/graph_scene.gd:1820` and `xr-client/scripts/hud.gd:2549`.
**Invariant (resolved Open):** the headset receipt is recorded. ADR-2188 notes OpenXR focused, 88-90 fps with all three volumes and zero script, shader, heartbeat or reconnect errors over 15 minutes on the HP (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:52`), with `activation_status: live`, corrected on 2026-11-07 from `active`, which the ledger does not accept (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:7`, `docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:56`) and the XR-client 0.1.35 changelog entry (`docs/XR-client.md:8`).
**Drift:** RESOLVED at visionclaw 46edc196 — ADR-2188's Verification section now opens "All satisfied (2026-11-06)" and lists the headset receipt as satisfied (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:44`, `docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:48`), matching the status section (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:52`).
**Open:** visual legibility of the rail and pages in the headset is still a human check (`docs/adr/ADR-2188-the-xr-client-adopts-the-overview-first-contract-without-bloom.md:52`); GL-gated GUT is not checkable from the container.
