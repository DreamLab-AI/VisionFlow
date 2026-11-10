---
id: VT-07
title: Estate position and the future VisionClaw grab path
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/adr/ADR-0006-estate-membership-lab-evaluation.md
  - docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md
  - docs/architecture/repository-map.md
adrs: [visiontouch-rs:ADR-0006, visionflow:ADR-2017]
sources:
  - ../visiontouch-rs/docs/adr/ADR-0006-estate-membership-lab-evaluation.md
  - ../visiontouch-rs/crates/device/src/lib.rs
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
  - ../visiontouch-rs/Cargo.toml
  - docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md
  - docs/architecture/repository-map.md
  - ../project/client/src/features/graph/hooks/useGraphEventHandlers.ts
  - ../project/client/src/features/visualisation/controls/SpacePilotController.ts
  - ../project/client/src/features/graph/services/interactionModes/vrArMode.ts
  - ../project/xr-client/rust/src/selection.rs
  - ../project/src/handlers/socket_flow_handler/message_routing.rs
verified_commit: {visiontouch-rs: a9e52b6154b44fb7afc158230c4ec26d4213c015, visionflow: a4a6c20c34c0e298bc4a6725e71e58e00685f64c, visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

Source review and qualifications: [1 November audit](../../estate-review/2026-11-01-visiontouch-software-boundaries.md). The source pins are retained; this review changes claims, not device behaviour.

visiontouch-rs is a private supporting study in the estate, not a substrate: the canon records it in the repository map and ADR-2017, and the repository records the same posture in ADR-0006. The only planned integration is a pose and stylus button entering VisionClaw by the same two paths as its other input devices, the web-client pointer handlers that produce node drag messages and an XR-client selection arbiter that takes pointer input. None of that is built; no VisionClaw file mentions this device, and the repository has no bridge crate. This topic is a map of the intended seams, so every VisionClaw claim is EXTERNAL at the declared VisionClaw revision. The five cited VisionClaw files were byte-identical at 75b72416bac176ca8fde715cac3bd7c77657b3e7 during the 1 November review.

## For the business

The lab wants a physical stylus that can pick up and drag nodes in the VisionClaw knowledge graph, as a study for its VisionFlow installation. Today that is an intention with a documented route, not a feature: the repository provides lab software with simulation coverage and owner-host input traces, the VisionClaw side has not been touched, and nothing here is a product, a release or a service. Pushing back on the user's hand (force return) is explicitly out of scope. If the route is ever started, it needs its own decision record in both repositories; any delivered-force claim separately needs physical verification, and any step towards publishing or demonstrating outside the lab reopens the legal review before anything else.

## VT-07.1 Where the study sits in the estate

```mermaid
flowchart TB
    HR["visiontouch-rs, private lab evaluation, no economic use<br/>repository-map.md:40"]
    A6["repository ADR-0006: accepted, complete, live<br/>ADR-0006-estate-membership-lab-evaluation.md:5-7"]
    A17["canon ADR-2017: accepted, none, inactive<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:5-6"]
    MAP["canon repository map, dependency sketch node VT<br/>repository-map.md:56"]
    VC["VisionClaw graph: future consumer only<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:28"]
    OUT["outputs: an academic paper, later a pose and button input<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:22"]
    HR --> A6
    HR --> A17
    A17 --> MAP
    HR --> OUT
    OUT -. "EXTERNAL: not built" .-> VC
    N1["INVARIANT: the repository stays private and no canon text may call it a product, service, release or driver<br/>ADR-0006-estate-membership-lab-evaluation.md:35-41"]
    N2["OPEN: lifecycle labels have different scopes, lab membership is live while the proposed integration is inactive<br/>ADR-0006-estate-membership-lab-evaluation.md:5-7 vs ADR-2017-haptic-input-device-is-a-supporting-study.md:5-6"]
    N3["DOC-DRIFT: ADR-2017 verified_commit 4a8d4b5 predates the commit that added the ADR (1e3bdd0)<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:10"]
    N4["INVARIANT: ADR-0006 now carries a committed verified_commit and its verification paragraph names this diagrams area (VT-01 to VT-08)<br/>ADR-0006-estate-membership-lab-evaluation.md:10 and ADR-0006-estate-membership-lab-evaluation.md:54-57"]
    A6 --- N1
    A6 --- N2
    A17 --- N3
    A6 --- N4
```

Both records agree on the posture and on the integration being future work. Their lifecycle labels describe membership and future integration respectively; the labels alone do not establish a contradiction.

## VT-07.2 What the arm can supply today

```mermaid
flowchart TB
    DEV["Device::sample on a dedicated worker, feedback off<br/>crates/device/src/lib.rs:680"]
    ES["EncoderState: six counters and the stylus button<br/>crates/device/src/lib.rs:216-224"]
    REG["register_reset at the reset pose, per connection<br/>crates/device/src/lib.rs:583"]
    POS["position_m: three coordinates in metres from the reset pose<br/>crates/device/src/lib.rs:271-274"]
    BTN["button_pressed, active low<br/>crates/device/src/lib.rs:222-223"]
    BRG["local bridge on the device host to the client over loopback, never via the server<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:31"]
    CL["web or XR client input path, see VT-07.3 and VT-07.4"]
    DEV --> ES
    ES --> POS
    ES --> BTN
    REG --> POS
    POS --> BRG
    BTN --> BRG
    BRG -. "EXTERNAL: not built" .-> CL
    N1["DIVERGENCE: ADR-2017 says the lab wants a 6-DoF input device<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:23 but the SDK exposes a 3-coordinate position and the wrist counters stay uncalibrated counts<br/>12-force-feedback.md:14-15"]
    N2["DEBT: no bridge crate or stream exists, the workspace members are hal, device, effects, three tools and two apps<br/>Cargo.toml:2"]
    N3["INVARIANT: pose is relative to a per-connection reset pose, a reconnect clears it<br/>crates/device/src/lib.rs:569-575"]
    POS --- N1
    BRG --- N2
    REG --- N3
```

Orientation is not available from the SDK, so a world-space ray from the stylus cannot be built without new calibration work. A position-only mapping is the realistic first step.

## VT-07.3 Web client route, EXTERNAL and not built

```mermaid
sequenceDiagram
    autonumber
    participant SP as SpacePilotController<br/>SpacePilotController.ts:138
    participant HK as useGraphEventHandlers<br/>useGraphEventHandlers.ts:28
    participant RT as handle_text_message<br/>message_routing.rs:18
    SP->>SP: object mode edits position and quaternion directly<br/>(SpacePilotController.ts:458-479)
    Note over SP,HK: DRIFT: no direct SpacePilot-to-pointer-handler call is present in these<br/>files
    HK->>HK: R3F pointer events drive the drag state<br/>(useGraphEventHandlers.ts:87)
    HK->>RT: nodeDragStart with node id and start position<br/>(useGraphEventHandlers.ts:159)
    HK->>RT: nodeDragUpdate, throttled, with timestamp<br/>(useGraphEventHandlers.ts:65)
    RT->>RT: route to the position update handlers<br/>(message_routing.rs:51-59)
    HK->>RT: nodeDragEnd<br/>(useGraphEventHandlers.ts:278)
    Note over SP,RT: EXTERNAL: nothing in VisionClaw reads this device, the drag messages are<br/>produced by existing pointer input<br/>ADR-2017 anchors checked at the declared revision, see<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:29
    Note over HK: EXTERNAL: VisionClaw settings already use the word haptic for VR<br/>controller feedback, that is a different feature<br/>vrArMode.ts:113
    Note over SP,RT: OPEN: a loopback bridge in a browser must be authenticated like any<br/>other client input, which ADR-2017 leaves to a future VisionClaw record<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:39
```

A device that produced the same ray-and-drag events would need no new server message type. The pointer handlers and server routes exist. SpacePilot object manipulation is a separate path; sharing a future adapter requires explicit integration.

## VT-07.4 XR client route, EXTERNAL and not built

```mermaid
flowchart TB
    PI["PointerInput: hand slot, ray, trigger held<br/>selection.rs:76"]
    RES["Resolver variants ControllerRay, Pinch, GazeDwell<br/>selection.rs:37-41"]
    ARB["SelectionArbiter::tick takes controllers, gaze, candidates<br/>selection.rs:297"]
    EVT["SelectionEvent out<br/>selection.rs:305"]
    NEW["a new Resolver variant for the stylus<br/>ADR-2017-haptic-input-device-is-a-supporting-study.md:30"]
    STY["stylus button as the trigger, position as a ray origin<br/>crates/device/src/lib.rs:223"]
    STY -. "EXTERNAL: would map to trigger held" .-> PI
    PI --> ARB
    RES --> ARB
    ARB --> EVT
    NEW -. "EXTERNAL: not built" .-> RES
    N1["INVARIANT: explicit triggers act on a rising edge and outrank dwell, so a held stylus button would not re-select every frame<br/>selection.rs:314-315 and selection.rs:645"]
    N2["OPEN: PointerInput needs a ray, the arm gives only a position, so a ray direction must be chosen or calibrated<br/>selection.rs:76 vs crates/device/src/lib.rs:271"]
    ARB --- N1
    PI --- N2
```

The cited `selection.rs` anchors match the declared revision and are unchanged at the reviewed 75b72416 HEAD. The XR path would add a resolver, not a new server message.
