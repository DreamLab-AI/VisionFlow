---
id: VC-44
title: Concurrent XR content connections and separate room presence
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2159-the-xr-client-holds-one-backend-connection-per-content-source.md
adrs: [ADR-2143, ADR-2159]
sources:
  - ../project/xr-client/scripts/backend_connection.gd
  - ../project/xr-client/scripts/backend_caps.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/frame_budget_pass.gd
  - ../project/xr-client/tests/unit/test_backend_switch.gd
  - ../project/xr-client/rust/tests/two_clients.rs
verified_commit: {visionclaw: 75b72416bac176ca8fde715cac3bd7c77657b3e7}
---

## For developers

This is the current source arrangement replacing [VC-40's historical exclusive switch](40-xr-two-backends-capability-gating.md). The [audit](../../estate-review/2026-10-31-codescape-presence-and-xr.md) distinguishes Rust transport evidence, inspected GDScript and unmeasured headset acceptance.

The subsequent [headless volume/link audit](46-spatial-volume-fit-and-link-budget.md) qualifies fit cadence and highlight budgets with executed Godot fixtures; transport and headset acceptance remain separate.

## For the business

Knowledge, ontology and code can be fed by concurrent connections. Source separation and capability guards are implemented; their presence alone does not establish the three-volume headset frame-rate target.

## VC-44.1 Two content connections feed three volumes while presence stays separate

```mermaid
flowchart TD
    SC["GraphScene builds two BackendConnections<br/>graph_scene.gd:1282"] --> VC["VisionClaw client<br/>graph_scene.gd:1284"]
    SC --> CS["Codescape client with empty signing secret<br/>graph_scene.gd:1285"]
    VC --> K["Knowledge volume<br/>test_backend_switch.gd:26"]
    VC --> O["Ontology volume<br/>test_backend_switch.gd:27"]
    CS --> C["Codescape volume<br/>test_backend_switch.gd:28"]
    SC --> SRC["Each binding creates ConnectionVolumeSource<br/>graph_scene.gd:1313,1314"]
    SC --> PR["Presence URL derives from VisionClaw base<br/>graph_scene.gd:2424,2428"]
    CS --> OWN["Each instance owns its client, caps, pins, HTTP and backoff<br/>backend_connection.gd:36,47,50,51,57"]
    VC --> OWN
```

**What it shows.** The scene keeps both graph connections and one separately configured room-presence path. **Why it is this way.** Node ids, acknowledgements and request state belong to their content connection; changing focus no longer selects an exclusive graph backend.

## VC-44.2 Requests and drags gate through a volume's connection

```mermaid
flowchart TD
    FOCUS["Focused volume resolves connection<br/>graph_scene.gd:1753,1778"] --> CAP["Connection supports and gate<br/>backend_connection.gd:155,162"]
    REQ["Request helpers use gate_for_volume<br/>graph_scene.gd:1764"] --> CAP
    DRAG["Drag start, update and end call DRAG gate<br/>backend_connection.gd:247,256,265"] --> CAP
    CAP -->|unsupported| NO["Emit refused and return false<br/>backend_connection.gd:165,167"]
    CAP -->|supported drag| SEND["Call this client's drag method<br/>backend_connection.gd:252,261,270"]
    TEST["GUT cases: no exclusive switch, focused fold refusal, no Codescape drag<br/>test_backend_switch.gd:18,37,53"] --> CAP
```

**What it shows.** Codescape drag is now refused before sending, unlike ADR-2143 A6's historical limitation. **Why it is this way.** Capabilities follow each connection and the focused volume. These controls express client UX; they are not server authorisation for federated content.

## VC-44.3 Shared budget and transport preflight are distinct acceptance layers

```mermaid
flowchart TD
    VIS["Visible volume demand summed<br/>frame_budget_pass.gd:20"] --> BUD["Allocate once and split caps<br/>frame_budget_pass.gd:125,137"]
    BUD --> PACK["Pack each visible connected volume with its caps<br/>graph_scene.gd:1506,1524,1529"]
    PACK --> HIDE["Hidden or offline clears instances and overlay slot<br/>graph_scene.gd:1524,1525,1527"]
    PRE["Rust loopback preflight drives production graph transport<br/>two_clients.rs:410,419"] --> ACK["Independent frame and ack sequences, stall and pump timing checks<br/>two_clients.rs:506,511,512,523,524"]
    ACK --> LIMIT["Test thread substitutes for Godot pump<br/>no engine or render-store load measurement<br/>two_clients.rs:7,10,38"]
```

**What it shows.** Budget allocation, transport fairness and headset frame time are separate claims. **Why it is this way.** The preflight exercises real transport with simulated main-thread pumping; ADR-2159 still requires E8 on the headset before activation.

**Drift:** ADR-2159's verification says no BackendConnection exists; current construction is wired into the scene (`graph_scene.gd:1282`). Its inactive status is not overturned by this source observation.
