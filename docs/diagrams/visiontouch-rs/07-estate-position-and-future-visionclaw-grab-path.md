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
verified_commit: {visiontouch-rs: 3875bfd994d73c068a8dafa6ea0ec6f012063e1e, visionflow: 1e3bdd074f43400cb51200b20d5f41ea711af47d, visionclaw: 458b0ded5da35b968ebd7ad056eba8bc06da38a5}
---

## For developers

visiontouch-rs is a private supporting study in the estate, not a substrate: the canon records it in the repository map and ADR-2017, and the repository records the same posture in ADR-0006. The only planned integration is a pose and stylus button entering VisionClaw by the same two paths as its other input devices, a web-client controller that already produces node drag messages and an XR-client selection arbiter that takes pointer input. None of that is built; no VisionClaw file mentions this device, and the repository has no bridge crate. This topic is a map of the intended seams, so every VisionClaw claim is EXTERNAL and read at VisionClaw HEAD.

## For the business

The lab wants a physical stylus that can pick up and drag nodes in the VisionClaw knowledge graph, as a study for its VisionFlow installation. Today that is an intention with a documented route, not a feature: the arm works as a lab instrument, the VisionClaw side has not been touched, and nothing here is a product, a release or a service. Pushing back on the user's hand (force return) is explicitly out of scope. If the route is ever started, it needs its own decision record in both repositories and a physical force verification first, and any step towards publishing or demonstrating outside the lab reopens the legal review before anything else.

## VT-07.1 Where the study sits in the estate

```mermaid
flowchart TB
    HR["visiontouch-rs, private lab evaluation, no economic use<br/>docs/architecture/repository-map.md:36"]
    A6["repository ADR-0006: proposed, none, inactive<br/>../visiontouch-rs/docs/adr/ADR-0006-estate-membership-lab-evaluation.md:5-7"]
    A17["canon ADR-2017: accepted, none, inactive<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:5-6"]
    MAP["canon repository map, dependency sketch node HR<br/>docs/architecture/repository-map.md:52"]
    VC["VisionClaw graph: future consumer only<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:28"]
    OUT["outputs: an academic paper, later a pose and button input<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:22"]
    HR --> A6
    HR --> A17
    A17 --> MAP
    HR --> OUT
    OUT -. "EXTERNAL: not built" .-> VC
    N1["INVARIANT: the repository stays private and no canon text may call it a product, service, release or driver<br/>../visiontouch-rs/docs/adr/ADR-0006-estate-membership-lab-evaluation.md:35-41"]
    N2["DIVERGENCE: the same decision is proposed in the study repository and accepted in the canon<br/>../visiontouch-rs/docs/adr/ADR-0006-estate-membership-lab-evaluation.md:5 vs docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:5"]
    N3["DOC-DRIFT: ADR-2017 verified_commit 4a8d4b5 predates the commit that added the ADR (1e3bdd0)<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:10"]
    N4["DEBT: ADR-0006 verified_commit is blank until the study repository commits<br/>../visiontouch-rs/docs/adr/ADR-0006-estate-membership-lab-evaluation.md:10"]
    A6 --- N1
    A6 --- N2
    A17 --- N3
    A6 --- N4
```

Both records agree on the posture and on the integration being future work. They disagree only in their lifecycle labels.

## VT-07.2 What the arm can supply today

```mermaid
flowchart TB
    DEV["Device::sample on a dedicated worker, feedback off<br/>crates/device/src/lib.rs:529"]
    ES["EncoderState: six counters and the stylus button<br/>crates/device/src/lib.rs:186-194"]
    REG["register_reset at the reset pose, per connection<br/>crates/device/src/lib.rs:474"]
    POS["position_m: three coordinates in metres from the reset pose<br/>crates/device/src/lib.rs:241-244"]
    BTN["button_pressed, active low<br/>crates/device/src/lib.rs:192-193"]
    BRG["local bridge on the device host to the client over loopback, never via the server<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:31"]
    CL["web or XR client input path, see VT-07.3 and VT-07.4"]
    DEV --> ES
    ES --> POS
    ES --> BTN
    REG --> POS
    POS --> BRG
    BTN --> BRG
    BRG -. "EXTERNAL: not built" .-> CL
    N1["DIVERGENCE: ADR-2017 says the lab wants a 6-DoF input device<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:23 but the SDK exposes a 3-coordinate position and the wrist counters stay uncalibrated counts<br/>docs/spec/12-force-feedback.md:14-15"]
    N2["DEBT: no bridge crate or stream exists, the workspace members are hal, device, effects, three tools and two apps<br/>Cargo.toml:2"]
    N3["INVARIANT: pose is relative to a per-connection reset pose, a reconnect clears it<br/>crates/device/src/lib.rs:460-466"]
    POS --- N1
    BRG --- N2
    REG --- N3
```

Orientation is not available from the SDK, so a world-space ray from the stylus cannot be built without new calibration work. A position-only mapping is the realistic first step.

## VT-07.3 Web client route, EXTERNAL and not built

```mermaid
sequenceDiagram
    autonumber
    participant SP as SpacePilotController<br/>client/src/features/visualisation/controls/SpacePilotController.ts:138
    participant HK as useGraphEventHandlers<br/>client/src/features/graph/hooks/useGraphEventHandlers.ts:27
    participant RT as handle_text_message<br/>src/handlers/socket_flow_handler/message_routing.rs:16
    SP->>HK: R3F ray and drag, as pointer events (client/src/features/graph/hooks/useGraphEventHandlers.ts:78)
    HK->>RT: nodeDragStart with node id and start position (client/src/features/graph/hooks/useGraphEventHandlers.ts:150)
    HK->>RT: nodeDragUpdate, throttled, with timestamp (client/src/features/graph/hooks/useGraphEventHandlers.ts:64)
    RT->>RT: route to the position update handlers (src/handlers/socket_flow_handler/message_routing.rs:52-60)
    HK->>RT: nodeDragEnd (client/src/features/graph/hooks/useGraphEventHandlers.ts:255)
    Note over SP,RT: EXTERNAL: nothing in VisionClaw reads this device, the drag messages are produced by existing pointer input<br/>ADR-2017 anchors checked at VisionClaw HEAD, see docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:29
    Note over HK: EXTERNAL: VisionClaw settings already use the word haptic for VR controller feedback, that is a different feature<br/>client/src/features/graph/services/interactionModes/vrArMode.ts:113
    Note over SP,RT: OPEN: a loopback bridge in a browser must be authenticated like any other client input, which ADR-2017 leaves to a future VisionClaw record<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:39
```

A device that produced the same ray-and-drag events would need no new server message type. Everything shown here exists and is used by mouse and SpacePilot input today.

## VT-07.4 XR client route, EXTERNAL and not built

```mermaid
flowchart TB
    PI["PointerInput: hand slot, ray, trigger held<br/>xr-client/rust/src/selection.rs:76"]
    RES["Resolver variants ControllerRay, Pinch, GazeDwell<br/>xr-client/rust/src/selection.rs:37-41"]
    ARB["SelectionArbiter::tick takes controllers, gaze, candidates<br/>xr-client/rust/src/selection.rs:282"]
    EVT["SelectionEvent out<br/>xr-client/rust/src/selection.rs:290"]
    NEW["a new Resolver variant for the stylus<br/>docs/adr/ADR-2017-haptic-input-device-is-a-supporting-study.md:30"]
    STY["stylus button as the trigger, position as a ray origin<br/>crates/device/src/lib.rs:193"]
    STY -. "EXTERNAL: would map to trigger held" .-> PI
    PI --> ARB
    RES --> ARB
    ARB --> EVT
    NEW -. "EXTERNAL: not built" .-> RES
    N1["INVARIANT: explicit triggers act on a rising edge and outrank dwell, so a held stylus button would not re-select every frame<br/>xr-client/rust/src/selection.rs:299-300 and xr-client/rust/src/selection.rs:618"]
    N2["OPEN: PointerInput needs a ray, the arm gives only a position, so a ray direction must be chosen or calibrated<br/>xr-client/rust/src/selection.rs:76 vs crates/device/src/lib.rs:241"]
    ARB --- N1
    PI --- N2
```

Anchors in `selection.rs` were read at VisionClaw HEAD and match ADR-2017. The XR path would add a resolver, not a new server message.
