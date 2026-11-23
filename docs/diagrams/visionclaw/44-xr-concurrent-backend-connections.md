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
  - ../project/xr-client/rust/src/transport.rs
  - ../project/xr-client/rust/src/liveness.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/nginx.dev.conf
  - ../project/docs/XR-client.md
verified_commit: {visionclaw: 4900ec60bc1ba9075343fe848c62e64434126c7e}
---

## For developers

This is the current source arrangement that replaced the exclusive switch; [VC-40](40-xr-two-backends-capability-gating.md) draws the capability table, the retired switch and the palette at the same pin. The [audit](../../estate-review/2026-10-31-codescape-presence-and-xr.md) distinguishes Rust transport evidence, inspected GDScript and unmeasured headset acceptance.

The subsequent [headless volume/link audit](46-spatial-volume-fit-and-link-budget.md) qualifies fit cadence and highlight budgets with executed Godot fixtures; transport and headset acceptance remain separate. The [federation prerequisite review](48-dataspace-federation-prerequisites.md) records the signed release upgrade and the remaining dataspace permission requirements.

## For the business

Knowledge, ontology and code can be fed by concurrent connections. Source separation and capability guards are implemented; their presence alone does not establish the three-volume headset frame-rate target.

## VC-44.1 Two content connections feed three volumes while presence stays separate

```mermaid
flowchart TD
    SC["GraphScene builds two BackendConnections<br/>graph_scene.gd:1378"] --> VC["VisionClaw client<br/>graph_scene.gd:1380"]
    SC --> CS["Codescape client with empty signing secret<br/>graph_scene.gd:1381"]
    VC --> K["Knowledge volume<br/>test_backend_switch.gd:26"]
    VC --> O["Ontology volume<br/>test_backend_switch.gd:27"]
    CS --> C["Codescape volume<br/>test_backend_switch.gd:28"]
    SC --> SRC["Each binding creates ConnectionVolumeSource<br/>graph_scene.gd:1410,1411"]
    SC --> PR["Presence URL derives from VisionClaw base<br/>graph_scene.gd:2952,2956"]
    CS --> CSB["Codescape base: XR_CODESCAPE_WS, else XR_BACKEND_WS<br/>(default port 4000) with /codescape appended<br/>graph_scene.gd:2974,2975,2978 backend_connection.gd:87,88"]
    CSB --> NGX["Only the dev nginx on port 3001 routes /codescape/<br/>to codescape port 4100, the backend has no such route<br/>nginx.dev.conf:56,74,75"]
    CS --> OWN["Each instance owns its client, caps, pins, HTTP and backoff<br/>backend_connection.gd:39,50,61,62,68"]
    VC --> OWN
    SC -.-> D44["DRIFT: the doc section still headed accepted, not built says nothing exists<br/>and the client has one scene and one backend<br/>XR-client.md:1161,1163"]
```

**What it shows.** The scene keeps both graph connections and one separately configured room-presence path; graph and presence go straight to the backend on port 4000 while Codescape is reached through nginx on port 3001 (`docs/XR-client.md:307`). **Why it is this way.** Node ids, acknowledgements and request state belong to their content connection; changing focus no longer selects an exclusive graph backend.
**Open (risk):** with `XR_CODESCAPE_WS` unset, the Codescape connection dials `XR_BACKEND_WS` plus `/codescape` (`xr-client/scripts/graph_scene.gd:2978`), which on the default port 4000 reaches a backend with no `/codescape` route; only the nginx location at `nginx.dev.conf:74` serves it, so that volume retries under backoff until the operator sets the variable.
**Drift:** `docs/XR-client.md:1325-1326` still says to point `XR_BACKEND_WS` at Codescape or press the Session tab's Backend toggle, and `docs/XR-client.md:1336` launches with `XR_BACKEND_MODE=codescape`; the toggle is retired and no client script reads that variable, while the per-source base is `xr-client/scripts/graph_scene.gd:2974-2978`.

## VC-44.2 Requests and drags gate through a volume's connection

```mermaid
flowchart TD
    FOCUS["Focused volume resolves connection<br/>graph_scene.gd:2246,2271"] --> CAP["Connection supports and gate<br/>backend_connection.gd:196,203"]
    REQ["Request helpers use _gate_for_volume<br/>graph_scene.gd:2256"] --> CAP
    DRAG["Drag start, update and end call DRAG gate<br/>backend_connection.gd:331,340,349"] --> CAP
    CAP -->|unsupported| NO["Emit refused and return false<br/>backend_connection.gd:206,208"]
    CAP -->|supported drag| SEND["Call this client's drag method<br/>backend_connection.gd:335,344,353"]
    TEST["GUT cases: no exclusive switch, focused fold refusal, no Codescape drag<br/>test_backend_switch.gd:18,37,53"] --> CAP
```

**What it shows.** Codescape drag is now refused before sending, unlike ADR-2143 A6's historical limitation. **Why it is this way.** Capabilities follow each connection and the focused volume. These controls express client UX; they are not server authorisation for federated content.

## VC-44.3 Shared budget and transport preflight are distinct acceptance layers

```mermaid
flowchart TD
    VIS["Visible volume demand summed<br/>frame_budget_pass.gd:20"] --> BUD["Allocate once and split caps<br/>frame_budget_pass.gd:149,166"]
    BUD --> PACK["Pack each visible connected volume with its caps<br/>graph_scene.gd:1759,1777,1782"]
    PACK --> HIDE["Hidden or offline clears instances and overlay slot<br/>graph_scene.gd:1777,1778,1780"]
    PRE["Rust loopback preflight drives production graph transport<br/>two_clients.rs:442,451"] --> ACK["Independent frame and ack sequences, stall and pump timing checks<br/>two_clients.rs:551-555,564,565,581,582"]
    ACK --> LIMIT["Test thread substitutes for Godot pump<br/>no engine or render-store load measurement<br/>two_clients.rs:7,11,38"]
```

**What it shows.** Budget allocation, transport fairness and headset frame time are separate claims. **Why it is this way.** The preflight exercises real transport with simulated main-thread pumping; ADR-2159 still requires E8 on the headset before activation.

**Drift:** ADR-2159's verification says no BackendConnection exists; current construction is wired into the scene (`graph_scene.gd:1378`). Its inactive status is not overturned by this source observation.

## VC-44.4 Reconnect liveness tracks bytes and an applied position frame

```mermaid
flowchart TD
    READ["ReadClock records every TCP read that delivers bytes<br/>liveness.rs:149-160"] --> CLOCK["Pump consults liveness: ping every 3 s,<br/>expire after 10 s without read progress<br/>liveness.rs:46-47,88-100; transport.rs:255-275"]
    PARSE["After topology parsing, touch progress again<br/>transport.rs:283-286"] --> CLOCK
    CLOCK --> DOWN["Disconnected schedules exponential backoff,<br/>2 s to 60 s, shortened by up to 20 percent jitter<br/>backend_connection.gd:26-30,93-100,423-441"]
    DOWN --> UPGRADE["Socket upgrade alone keeps attempt count<br/>backend_connection.gd:428-436"]
    UPGRADE --> FRAME["First applied position frame emits stream_live once<br/>binary_protocol.rs:2902-2904"]
    FRAME --> RESET["Reset backoff attempts<br/>backend_connection.gd:446-447"]
    RESUME["App or XR session resumes or regains focus<br/>graph_scene.gd:1683-1685,1705-1719"] --> DIAL["Redial an open, disconnected connection now;<br/>leave connected or already dialling alone<br/>backend_connection.gd:187-194"]
    DIAL --> UPGRADE
```

**Boundary:** These are the current transport and scene mechanisms, not a fresh headset or network-fault acceptance run. Byte progress keeps a large message alive while it arrives, and the parsing reset avoids charging local parse time to the peer. The timing decision is consulted by the pump; it is not a general timeout around every awaited socket operation.
