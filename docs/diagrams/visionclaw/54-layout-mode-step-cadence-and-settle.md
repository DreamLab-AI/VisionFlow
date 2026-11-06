---
id: VC-54
title: Layout mode, step cadence and the three meanings of settle
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2140, ADR-2141, ADR-2169, ADR-2189]
sources:
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/crates/visionclaw-domain/src/models/simulation_params.rs
  - ../project/src/models/simulation_params.rs
  - ../project/src/adapters/sqlite_settings_repository.rs
  - ../project/src/actors/gpu/physics_restore.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/actors/position_stream.rs
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/actors/gpu/test_fixtures/live_physics_settings.json
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/client/src/features/control-center/macros/MacroBar.tsx
  - ../project/client/src/features/control-center/primitives/MacroDial.tsx
  - ../project/client/src/features/control-center/registry/macros.ts
  - ../project/client/src/features/control-center/registry/manifest.ts
  - ../project/client/src/features/control-center/state/settingWrites.ts
  - ../project/client/src/features/graph/components/GraphManager.tsx
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/hud.gd
  - ../project/docs/XR-client.md
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
---

## For developers

Drawn for the [2026-11-01 duplication audit](52-drag-holds-and-observation-gates.md). It covers ADR-2140's single `physics.layoutMode` write path, what bounds the step rate as opposed to the ack-paced position stream (ADR-2141), and the three separate mechanisms called "settle". `live_physics_settings.json` is a test fixture captured from the live store, not the store itself. Restamped on 2026-11-06 at `ec785afa9`: citations moved with the hold-owner map and slot-id changes, and VC-54.5 adds the three surfaces that write the one shared layout document.

## For the business

A layout choice is now stored once and every client sees the same one. How smoothly a dragged graph appears to move depends on two different rates, the physics step and each viewer's delivery, and the stored force style decides how strongly neighbours follow a held node.

## VC-54.1 The layout mode has one stored definition and one write path

```mermaid
flowchart TD
    POST["POST layout mode parses either vocabulary, unknown is 400<br/>layout_handler.rs:45,64,67"] --> SAME{"Mode already stored?<br/>layout_handler.rs:86"}
    SAME -->|yes| NOOP["Nothing written, no broadcast, no resume<br/>layout_handler.rs:87"]
    SAME -->|no| PATCH["apply_physics_patch with layoutMode only<br/>layout_handler.rs:92,95"]
    PATCH --> MERGE["Deep-merge onto the stored document, validate twice<br/>settings_routes.rs:651,664,674"]
    MERGE --> STORE["Write the settings snapshot<br/>settings_routes.rs:686"]
    STORE --> RESOLVE["Stored zero becomes the active mode's default here only<br/>settings_routes.rs:695,696<br/>visionclaw-domain/src/models/simulation_params.rs:450,453,470"]
    RESOLVE --> GPU["UpdateSimulationParams to the GPU actor<br/>settings_routes.rs:701,705"]
    GPU --> ACTOR["Actor neither preserves nor fills the mode<br/>force_compute_actor.rs:2618,2619,2620"]
    BOOT["Boot: legacy qualityGates string migrates once<br/>sqlite_settings_repository.rs:1071,1079<br/>physics_restore.rs:33,39,42"] --> STORE
```

**What it shows.** One path from request to device. The legacy quality-gate string is read once, at boot. **Why it is this way.** ADR-2140 item 1 (2026-10-31) removed the actor's preserve-override, `SetLayoutMode` and the propagation overlay so that no second copy of the mode exists.

**Debt (loose end):** the 2026-11-06 rendering-row insert landed between the ADR-2140 decision-1 doc comment and the function it documents, so that comment (`src/adapters/sqlite_settings_repository.rs:1046-1054`) now attaches to `RENDERING_ROW` (`src/adapters/sqlite_settings_repository.rs:1055-1057`) and `seed_physics_from_blob` (`src/adapters/sqlite_settings_repository.rs:1071`) has none.

## VC-54.2 The step is paced by completion, the stream by each client's acknowledgement

```mermaid
flowchart TD
    MODE["Cadence 16 ms in Continuous, 0 in FastSettle<br/>physics_orchestrator_actor.rs:113,272,276"] --> EFF["Any held node makes the cadence 0<br/>physics_orchestrator_actor.rs:310,314"]
    STEP["GPU step: gate, mutex, kernels, readback<br/>force_compute_actor.rs:1979,2013,2018,2025"] --> DONE["PhysicsStepCompleted back to the orchestrator<br/>force_compute_actor.rs:2407"]
    DONE --> NEXT["Next step after cadence minus step time<br/>physics_orchestrator_actor.rs:1892,1894"]
    EFF --> NEXT
    NEXT --> RUN["run_later then physics_step<br/>physics_orchestrator_actor.rs:289,296,304"]
    TIMING["StepTiming logs phases every 120 steps<br/>force_compute_actor.rs:4891"] -.-> STEP
    BENCH["Ignored stage-timing test at live size<br/>execution.rs:2133"] -.-> STEP
    STREAM["Per client: next chunk only on ack or timeout<br/>position_stream.rs:17,18,19"] --> RATE["Delivery rate set by that client's round trip<br/>position_stream.rs:19,20"]
    CFG["Fixture: chunk 2500 nodes, ack timeout 500 ms<br/>live_physics_settings.json:1"] --> STREAM
    DONE -.-> STREAM
```

**What it shows.** The step rate and the per-client stream rate are independent clocks. On the live graph the 16 ms cadence does nothing, because a step takes longer than that. **Why it is this way.** The sequential pipeline avoids queued steps. ADR-2141 made delivery ack-paced so that a slow client cannot stall the step.

**Open:** ADR-2169 records the step at 9–10 Hz and leaves ack-paced delivery unmeasured. A rate observed at a client is a stream rate, so the step rate and the per-client stream rate need separate measurements (`position_stream.rs:19`; `force_compute_actor.rs:4891`).

## VC-54.3 Three mechanisms are called settle

```mermaid
flowchart TD
    FS["FastSettle mode: damping override burst, then latch off<br/>visionclaw-domain/src/models/simulation_params.rs:88,90<br/>physics_orchestrator_actor.rs:359"] --> DEF["Default and always-defaulted mode is Continuous<br/>visionclaw-domain/src/models/simulation_params.rs:83,84<br/>project/src/models/simulation_params.rs:181"]
    WIN["Hold settle window: 2 s of forced steps after last release<br/>force_compute_actor.rs:4536"]
    TEL["Settlement telemetry: 180 ticks below KE epsilon<br/>force_compute_actor.rs:291,296"]
    EQ["Orchestrator equilibrium auto-pause, held guard<br/>physics_orchestrator_actor.rs:655,673,714"] --> OFF["Fixture: pauseOnEquilibrium false<br/>live_physics_settings.json:1"]
    FS --> FA2["FA2 branch sets velocity from force, never reads damping<br/>visionclaw_unified.cu:1059,1073"]
    FA2 -.-> D1["DEBT: FastSettle damping override is inert<br/>under the stored adaptive integrator<br/>visionclaw_unified.cu:1089"]
```

**What it shows.** A burst mode, a 2 s time box and a KE detector share one word, and a fourth judge (the auto-pause) is configured off. **Why it is this way.** Each was added for a different consumer: layout passes, ADR-2169 holds, and the REST `/api/layout/status`.

**Debt:** "settle" names three unrelated mechanisms (`physics_orchestrator_actor.rs:359`; `force_compute_actor.rs:4536,291`), and only the Verlet branch reads damping (`visionclaw_unified.cu:1089`). Since spring mode merged, that branch runs throughout every hold window (`execution.rs:1159,1160`), so damping is live there and inert only at rest under the stored adaptive integrator.

## VC-54.4 How strongly a held node's neighbours follow it: stored force style at rest, spring mode while held

```mermaid
flowchart TD
    PIN["Held node skips integration but stays in neighbours' spring loops<br/>visionclaw_unified.cu:1018"] --> ATT{"Stored linLogMode<br/>live_physics_settings.json:1"}
    ATT -->|true, fixture| LOG["Attraction log1p of distance, ignores spring_k<br/>visionclaw_unified.cu:434,437"]
    ATT -->|false| HOOKE["Hooke around one global rest length<br/>visionclaw_unified.cu:440,451"]
    LOG --> OVL["While held: plus spring_k times stretch from the grab snapshot<br/>visionclaw_unified.cu:454,455; execution.rs:718"]
    HOOKE --> OVL
    OVL --> CAP["Total force capped at max_force, raised with spring_k by the hold gain<br/>visionclaw_unified.cu:1044; execution.rs:1165,1166"]
    CAP --> INT{"Overlay on, or stored adaptiveSpeed false?<br/>execution.rs:1159,1160; live_physics_settings.json:1"}
    INT -->|no: at rest, fixture| AD["Speed is global_speed over 1 plus root swing<br/>visionclaw_unified.cu:1059,1073"]
    INT -->|yes: held or settling| VER["Verlet with dt and damping<br/>visionclaw_unified.cu:1088,1089"]
    FLOOR["Hold floor raises global_speed to 2.0<br/>force_compute_actor.rs:4552,4601"] -.-> GAP["DEBT: only the adaptive branch reads it, and the overlay<br/>turns that branch off for the same span<br/>visionclaw_unified.cu:1073"]
    SEL["Kernel chosen by stability_threshold<br/>execution.rs:661,662,664"] -.-> ONE["Both force kernels call one spring helper<br/>visionclaw_unified.cu:422,645,2400"]
```

**What it shows.** At rest, the fixture's LinLog plus adaptive pairing gives a weak, swing-damped pull. Turning LinLog off globally would move every edge to one rest length, and the resting layout with it. While a node is held, the merged ADR-2169 amendment adds a linear spring around each edge's length at the moment of grab. It strengthens that spring by the hold gain and integrates with Verlet, so the pull no longer depends on the stored force style or on swing. **Why it is this way.** LinLog is the modularity layout. The `HOLD_GLOBAL_SPEED` floor was the first attempt to answer the hand inside it. The spring overlay is the second attempt, and it replaced the first without removing it.

**Debt:** the speed floor is computed for exactly the span in which the overlay disables its only reader (`force_compute_actor.rs:4601`; `execution.rs:1160`; `visionclaw_unified.cu:1073`). The earlier debt that the spring force was written twice is closed: both kernels now call `edge_spring_scale` (`visionclaw_unified.cu:422`), although each kernel still has its own neighbour loop.

## VC-54.5 One shared layout document, written from three surfaces

```mermaid
flowchart TD
    subgraph DESK["desktop dock"]
        direction TB
        DIALS["Spread and Motion are the layout dials, scope shared because they write the physics bucket<br/>MacroBar.tsx:5 MacroBar.tsx:77 manifest.ts:55-57"]
        SPREAD["Spread writes repelK, restLength, centerGravityK<br/>macros.ts:41-43"]
        MOTION["Motion writes globalSpeed, damping, temperature<br/>macros.ts:72-74"]
        BADGE["affects-everyone marker and text on each shared dial<br/>MacroDial.tsx:41 MacroDial.tsx:197 manifest.ts:224"]
        DIALS --> SPREAD
        DIALS --> MOTION
        DIALS --> BADGE
    end
    subgraph MODE["desktop layout mode"]
        direction TB
        SW["registry write of layoutMode also POSTs the mode<br/>settingWrites.ts:65-66"]
        GM["GraphManager POSTs on every stored-mode change, to receive one-shot positions<br/>GraphManager.tsx:300-302 GraphManager.tsx:313"]
    end
    subgraph XRH["headset"]
        direction TB
        CYC["mode cycle POSTs after the volume capability gate, label flips on 2xx<br/>graph_scene.gd:2718-2722 graph_scene.gd:2737 graph_scene.gd:2739"]
        LBL["HUD Shared label, every viewer's layout changes<br/>hud.gd:563"]
        NOP["mode deliberately not persisted on device, re-applying at boot would move every viewer<br/>XR-client.md:1128-1130"]
        CYC --- LBL
        CYC --- NOP
    end
    SRV["POST /api/layout/mode, any authenticated user, 60 per minute<br/>layout_handler.rs:46 layout_handler.rs:370-372"]
    SAMEQ["already stored: nothing written<br/>layout_handler.rs:86-87"]
    ONE["Spectral or Temporal: placement over the caller's own view, returned in the response only<br/>layout_handler.rs:128 layout_handler.rs:139 layout_handler.rs:178 layout_handler.rs:191"]
    PHYS["settings physics PUT, the one physics document<br/>settings_routes.rs:651 settings_routes.rs:686"]
    SPREAD --> PHYS
    MOTION --> PHYS
    SW --> SRV
    GM --> SRV
    CYC --> SRV
    SRV --> SAMEQ
    SRV --> ONE
    MOTION -.-> INERT["DEBT: at rest under the stored adaptive integrator the damping share of Motion is not read<br/>visionclaw_unified.cu:1059 visionclaw_unified.cu:1073"]
```

**What it shows.** Layout mode and the two layout dials are one server-side document; the desktop marks the dials as shared and the headset refuses to persist the mode locally for the same reason. **Why it is this way.** ADR-2140 made `physics.layoutMode` the one stored definition, and ADR-2189 §2 labels every control that writes it.
**Debt (parallel):** Two desktop callers POST the mode: `settingWrites.ts:66` on a registry write and `GraphManager.tsx:313` on the resulting store change, so one edit sends two requests; the server's same-mode check (`layout_handler.rs:86`) makes the second a no-op. The shared-scope sentence is also defined twice (`MacroDial.tsx:41`, `manifest.ts:224`).
**Open (risk):** A Spectral or Temporal placement is computed over the caller's own view (`layout_handler.rs:139`) and exists only in that response (`layout_handler.rs:191`). Each desktop fetches its own on the store change (`GraphManager.tsx:315`), while the headset's completion handler leaves the body unread (`graph_scene.gd:2816`), so viewers can hold different placements under one shared mode.
