---
id: VC-44
title: Concurrent XR content connections and separate room presence
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2159-the-xr-client-holds-one-backend-connection-per-content-source.md
adrs: [ADR-2143, ADR-2159]
sources:
  - ../project/xr-client/scripts/backend_connection.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/frame_budget_pass.gd
  - ../project/xr-client/tests/unit/test_backend_switch.gd
  - ../project/xr-client/rust/tests/two_clients.rs
  - ../project/nginx.dev.conf
  - ../project/docs/XR-client.md
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

This is the current source arrangement that replaced the exclusive switch; [VC-40](40-xr-two-backends-capability-gating.md) draws the capability table, the retired switch and the palette at the same pin. The [audit](../../estate-review/2026-10-31-codescape-presence-and-xr.md) distinguishes Rust transport evidence, inspected GDScript and unmeasured headset acceptance.

The subsequent [headless volume/link audit](46-spatial-volume-fit-and-link-budget.md) qualifies fit cadence and highlight budgets with executed Godot fixtures; transport and headset acceptance remain separate. The [federation prerequisite review](48-dataspace-federation-prerequisites.md) traces why the current native transport does not satisfy release upgrade gates and why concurrent local sources do not establish dataspace permissions.

## For the business

Knowledge, ontology and code can be fed by concurrent connections. Source separation and capability guards are implemented; their presence alone does not establish the three-volume headset frame-rate target.

## VC-44.1 Two content connections feed three volumes while presence stays separate

```mermaid
flowchart TD
    SC["GraphScene builds two BackendConnections<br/>graph_scene.gd:1406"] --> VC["VisionClaw client<br/>graph_scene.gd:1408"]
    SC --> CS["Codescape client with empty signing secret<br/>graph_scene.gd:1409"]
    VC --> K["Knowledge volume<br/>test_backend_switch.gd:26"]
    VC --> O["Ontology volume<br/>test_backend_switch.gd:27"]
    CS --> C["Codescape volume<br/>test_backend_switch.gd:28"]
    SC --> SRC["Each binding creates ConnectionVolumeSource<br/>graph_scene.gd:1438,1439"]
    SC --> PR["Presence URL derives from VisionClaw base<br/>graph_scene.gd:2941,2945"]
    CS --> CSB["Codescape base: XR_CODESCAPE_WS, else XR_BACKEND_WS<br/>(default port 4000) with /codescape appended<br/>graph_scene.gd:2963,2964,2967 backend_connection.gd:77,78"]
    CSB --> NGX["Only the dev nginx on port 3001 routes /codescape/<br/>to codescape port 4100, the backend has no such route<br/>nginx.dev.conf:56,74,75"]
    CS --> OWN["Each instance owns its client, caps, pins, HTTP and backoff<br/>backend_connection.gd:36,47,52,53,59"]
    VC --> OWN
    SC -.-> D44["DRIFT: the doc section still headed accepted, not built says nothing exists<br/>and the client has one scene and one backend<br/>XR-client.md:1141,1143"]
```

**What it shows.** The scene keeps both graph connections and one separately configured room-presence path; graph and presence go straight to the backend on port 4000 while Codescape is reached through nginx on port 3001 (`docs/XR-client.md:304`). **Why it is this way.** Node ids, acknowledgements and request state belong to their content connection; changing focus no longer selects an exclusive graph backend.
**Open (risk):** with `XR_CODESCAPE_WS` unset, the Codescape connection dials `XR_BACKEND_WS` plus `/codescape` (`xr-client/scripts/graph_scene.gd:2967`), which on the default port 4000 reaches a backend with no `/codescape` route; only the nginx location at `nginx.dev.conf:74` serves it, so that volume retries under backoff until the operator sets the variable.
**Drift:** `docs/XR-client.md:1305-1306` still says to point `XR_BACKEND_WS` at Codescape or press the Session tab's Backend toggle, and `docs/XR-client.md:1316` launches with `XR_BACKEND_MODE=codescape`; the toggle is retired and no client script reads that variable, while the per-source base is `xr-client/scripts/graph_scene.gd:2963-2967`.

## VC-44.2 Requests and drags gate through a volume's connection

```mermaid
flowchart TD
    FOCUS["Focused volume resolves connection<br/>graph_scene.gd:2236,2261"] --> CAP["Connection supports and gate<br/>backend_connection.gd:157,164"]
    REQ["Request helpers use _gate_for_volume<br/>graph_scene.gd:2246"] --> CAP
    DRAG["Drag start, update and end call DRAG gate<br/>backend_connection.gd:282,291,300"] --> CAP
    CAP -->|unsupported| NO["Emit refused and return false<br/>backend_connection.gd:167,169"]
    CAP -->|supported drag| SEND["Call this client's drag method<br/>backend_connection.gd:286,295,304"]
    TEST["GUT cases: no exclusive switch, focused fold refusal, no Codescape drag<br/>test_backend_switch.gd:18,37,53"] --> CAP
```

**What it shows.** Codescape drag is now refused before sending, unlike ADR-2143 A6's historical limitation. **Why it is this way.** Capabilities follow each connection and the focused volume. These controls express client UX; they are not server authorisation for federated content.

## VC-44.3 Shared budget and transport preflight are distinct acceptance layers

```mermaid
flowchart TD
    VIS["Visible volume demand summed<br/>frame_budget_pass.gd:20"] --> BUD["Allocate once and split caps<br/>frame_budget_pass.gd:149,166"]
    BUD --> PACK["Pack each visible connected volume with its caps<br/>graph_scene.gd:1764,1782,1787"]
    PACK --> HIDE["Hidden or offline clears instances and overlay slot<br/>graph_scene.gd:1782,1783,1785"]
    PRE["Rust loopback preflight drives production graph transport<br/>two_clients.rs:410,419"] --> ACK["Independent frame and ack sequences, stall and pump timing checks<br/>two_clients.rs:506,511,512,523,524"]
    ACK --> LIMIT["Test thread substitutes for Godot pump<br/>no engine or render-store load measurement<br/>two_clients.rs:7,11,38"]
```

**What it shows.** Budget allocation, transport fairness and headset frame time are separate claims. **Why it is this way.** The preflight exercises real transport with simulated main-thread pumping; ADR-2159 still requires E8 on the headset before activation.

**Drift:** ADR-2159's verification says no BackendConnection exists; current construction is wired into the scene (`graph_scene.gd:1406`). Its inactive status is not overturned by this source observation.
