---
id: VC-54
title: Layout mode, step cadence and the three meanings of settle
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2140, ADR-2141, ADR-2169]
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
verified_commit: {visionclaw: d6c091f0ffb6d1af04ce011836c45ace6f4dafd9}
---

## For developers

Drawn for the [2026-11-01 duplication audit](52-drag-holds-and-observation-gates.md). It covers ADR-2140's single `physics.layoutMode` write path, what bounds the step rate as opposed to the ack-paced position stream (ADR-2141), and the three separate mechanisms called "settle". `live_physics_settings.json` is a test fixture captured from the live store, not the store itself. Code is identical at `531df5ef7`.

## For the business

A layout choice is now stored once and every client sees the same one. How smoothly a dragged graph appears to move depends on two different rates, the physics step and each viewer's delivery, and the stored force style decides how strongly neighbours follow a held node.

## VC-54.1 The layout mode has one stored definition and one write path

```mermaid
flowchart TD
    POST["POST layout mode parses either vocabulary, unknown is 400<br/>layout_handler.rs:45,64,67"] --> SAME{"Mode already stored?<br/>layout_handler.rs:86"}
    SAME -->|yes| NOOP["Nothing written, no broadcast, no resume<br/>layout_handler.rs:87"]
    SAME -->|no| PATCH["apply_physics_patch with layoutMode only<br/>layout_handler.rs:92,95"]
    PATCH --> MERGE["Deep-merge onto the stored document, validate twice<br/>settings_routes.rs:640,653,663"]
    MERGE --> STORE["Write the settings snapshot<br/>settings_routes.rs:675"]
    STORE --> RESOLVE["Stored zero becomes the active mode's default here only<br/>settings_routes.rs:684,685<br/>visionclaw-domain/src/models/simulation_params.rs:450,453,470"]
    RESOLVE --> GPU["UpdateSimulationParams to the GPU actor<br/>settings_routes.rs:690,694"]
    GPU --> ACTOR["Actor neither preserves nor fills the mode<br/>force_compute_actor.rs:2595,2596,2597"]
    BOOT["Boot: legacy qualityGates string migrates once<br/>sqlite_settings_repository.rs:1050,1058<br/>physics_restore.rs:33,39,42"] --> STORE
```

**What it shows.** One path from request to device. The legacy quality-gate string is read once, at boot. **Why it is this way.** ADR-2140 item 1 (2026-10-31) removed the actor's preserve-override, `SetLayoutMode` and the propagation overlay so that no second copy of the mode exists.

## VC-54.2 The step is paced by completion, the stream by each client's acknowledgement

```mermaid
flowchart TD
    MODE["Cadence 16 ms in Continuous, 0 in FastSettle<br/>physics_orchestrator_actor.rs:113,272,276"] --> EFF["Any held node makes the cadence 0<br/>physics_orchestrator_actor.rs:310,314"]
    STEP["GPU step: gate, mutex, kernels, readback<br/>force_compute_actor.rs:1958,1992,1995,2002"] --> DONE["PhysicsStepCompleted back to the orchestrator<br/>force_compute_actor.rs:2384"]
    DONE --> NEXT["Next step after cadence minus step time<br/>physics_orchestrator_actor.rs:1888,1890"]
    EFF --> NEXT
    NEXT --> RUN["run_later then physics_step<br/>physics_orchestrator_actor.rs:289,296,304"]
    TIMING["StepTiming logs phases every 120 steps<br/>force_compute_actor.rs:4784"] -.-> STEP
    BENCH["Ignored stage-timing test at live size<br/>execution.rs:1804"] -.-> STEP
    STREAM["Per client: next chunk only on ack or timeout<br/>position_stream.rs:17,18,19"] --> RATE["Delivery rate set by that client's round trip<br/>position_stream.rs:19,20"]
    CFG["Fixture: chunk 2500 nodes, ack timeout 500 ms<br/>live_physics_settings.json:1"] --> STREAM
    DONE -.-> STREAM
```

**What it shows.** The step rate and the per-client stream rate are independent clocks. On the live graph the 16 ms cadence does nothing, because a step takes longer than that. **Why it is this way.** The sequential pipeline avoids queued steps. ADR-2141 made delivery ack-paced so that a slow client cannot stall the step.

**Open:** ADR-2169 records the step at 9–10 Hz and leaves ack-paced delivery unmeasured. A rate observed at a client is a stream rate, so the step rate and the per-client stream rate need separate measurements (`position_stream.rs:19`; `force_compute_actor.rs:4784`).

## VC-54.3 Three mechanisms are called settle

```mermaid
flowchart TD
    FS["FastSettle mode: damping override burst, then latch off<br/>visionclaw-domain/src/models/simulation_params.rs:88,90<br/>physics_orchestrator_actor.rs:359"] --> DEF["Default and always-defaulted mode is Continuous<br/>visionclaw-domain/src/models/simulation_params.rs:83,84<br/>project/src/models/simulation_params.rs:181"]
    WIN["Hold settle window: 2 s of forced steps after last release<br/>force_compute_actor.rs:4439"]
    TEL["Settlement telemetry: 180 ticks below KE epsilon<br/>force_compute_actor.rs:287,292"]
    EQ["Orchestrator equilibrium auto-pause, held guard<br/>physics_orchestrator_actor.rs:655,673,714"] --> OFF["Fixture: pauseOnEquilibrium false<br/>live_physics_settings.json:1"]
    FS --> FA2["FA2 branch sets velocity from force, never reads damping<br/>visionclaw_unified.cu:983,997"]
    FA2 -.-> D1["DEBT: FastSettle damping override is inert<br/>under the stored adaptive integrator<br/>visionclaw_unified.cu:1013"]
```

**What it shows.** A burst mode, a 2 s time box and a KE detector share one word, and a fourth judge (the auto-pause) is configured off. **Why it is this way.** Each was added for a different consumer: layout passes, ADR-2169 holds, and the REST `/api/layout/status`.

**Debt:** "settle" names three unrelated mechanisms (`physics_orchestrator_actor.rs:359`; `force_compute_actor.rs:4439,287`), and only the Verlet branch reads damping (`visionclaw_unified.cu:1013`).

## VC-54.4 How strongly a held node's neighbours follow it is set by the stored force style

```mermaid
flowchart TD
    PIN["Held node skips integration but stays in neighbours' spring loops<br/>visionclaw_unified.cu:942"] --> ATT{"Stored linLogMode<br/>live_physics_settings.json:1"}
    ATT -->|true, fixture| LOG["Attraction log1p of distance, ignores spring_k<br/>visionclaw_unified.cu:574"]
    ATT -->|false| HOOKE["Hooke around one global rest length<br/>visionclaw_unified.cu:577,590"]
    LOG --> CAP["Total force capped at max_force<br/>visionclaw_unified.cu:968"]
    HOOKE --> CAP
    CAP --> INT{"Stored adaptiveSpeed<br/>live_physics_settings.json:1"}
    INT -->|true, fixture| AD["Speed is global_speed over 1 plus root swing<br/>visionclaw_unified.cu:983,997"]
    INT -->|false| VER["Verlet with dt and damping<br/>visionclaw_unified.cu:1012,1013"]
    FLOOR["Hold floor raises global_speed to 2.0<br/>force_compute_actor.rs:4455,4496"] --> AD
    FLOOR -.-> GAP["OPEN: a Verlet span while held would leave<br/>the floor with no reader<br/>visionclaw_unified.cu:1012"]
```

**What it shows.** The fixture's LinLog plus adaptive pairing gives a weak, swing-damped pull. Turning LinLog off globally would move every edge to one rest length, and the resting layout with it. **Why it is this way.** LinLog is the modularity layout, and the ADR-2169 floor was the first attempt to answer the hand inside it.

**Open:** an in-flight change (HS, not merged) adds a snapshot rest-length overlay and runs Verlet while held. That would make the `HOLD_GLOBAL_SPEED` floor unread over the same span (`force_compute_actor.rs:4455,4496`; `visionclaw_unified.cu:1012`).
