---
id: VC-40
title: XR backend capability table, per-volume request gate, the retired exclusive switch and the Codescape palette (ADR-2143 A6, ADR-2159)
area: visionclaw
governing:
  - ../project/docs/XR-client.md
  - ../project/docs/adr/ADR-2143-codescape-section-separate-workspace-and-port.md
  - ../project/docs/adr/ADR-2159-the-xr-client-holds-one-backend-connection-per-content-source.md
adrs: [ADR-2143, ADR-2159, ADR-2172, ADR-2134, ADR-2108, ADR-2189]
sources:
  - ../project/docs/XR-client.md
  - ../project/docs/adr/ADR-2143-codescape-section-separate-workspace-and-port.md
  - ../project/docs/adr/ADR-2159-the-xr-client-holds-one-backend-connection-per-content-source.md
  - ../project/xr-client/scripts/backend_caps.gd
  - ../project/xr-client/scripts/backend_connection.gd
  - ../project/xr-client/scripts/connection_volume_source.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/hud.gd
  - ../project/xr-client/scripts/graph_parity.gd
  - ../project/xr-client/rust/src/domain_palette.rs
  - ../project/xr-client/rust/src/render_store.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/xr-client/tests/unit/test_backend_caps.gd
  - ../project/xr-client/tests/unit/test_hud_tabs.gd
  - ../project/xr-client/tests/unit/test_graph_parity.gd
  - ../project/xr-client/tests/unit/test_backend_switch.gd
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

ADR-2143 A6 (2026-10-21) introduced the capability table, the request gate, the HUD greying, an exclusive `_switch_backend` and the Codescape palette. ADR-2159 then replaced the exclusive switch with one `BackendConnection` per content source; [VC-44](44-xr-concurrent-backend-connections.md) maps that arrangement. This topic was restamped on 2026-11-02 at the VisionClaw pin and redrawn: VC-40.1 shows the table and gate as they now run per volume, VC-40.2 shows what replaced the switch's hygiene, and VC-40.3 shows the palette and a gap in how it reaches the Codescape volume. Evidence is source tracing; no headset or GUT run was made for this review.

## For the business

Knowledge and code each keep their own connection, and the headset greys out any control the focused source cannot serve rather than letting a request fail silently. The menu now follows the seven shared sections, and a whole section hides only when none of its pages applies. One gap remains visible to a viewer: the code volume does not get the code palette.

## VC-40.1 One capability table gates every server-bound request per volume and greys the matching controls

```mermaid
flowchart TB
    subgraph TABLE["backend_caps.gd: pure data, no I/O"]
        KEYS["thirteen surfaces: physics_write, layout_mode, radial, fold, cycles, demo, memory_cloud,<br/>pattern_query, canary, node_relations, intervention, pin, drag<br/>xr-client/scripts/backend_caps.gd:20 xr-client/scripts/backend_caps.gd:32 xr-client/scripts/backend_caps.gd:34"]
        MODE["for_mode: VisionClaw and any unknown mode grant everything; Codescape grants CODESCAPE_SUPPORTED, which is empty<br/>xr-client/scripts/backend_caps.gd:45 xr-client/scripts/backend_caps.gd:100"]
        AMAP["ACTION_CAPS and ACTION_PREFIX_CAPS: every server-bound HUD action to its surface,<br/>forces_step:, memory_preset:, memory_hit:, memory_typed: by prefix<br/>xr-client/scripts/backend_caps.gd:49 xr-client/scripts/backend_caps.gd:77 xr-client/scripts/backend_caps.gd:133"]
        TMAP["TAB_CAPS: forces, swarm and pins have no live control left on Codescape, so the whole tab hides<br/>xr-client/scripts/backend_caps.gd:87 xr-client/scripts/backend_caps.gd:115"]
        KEYS --> MODE
        MODE --> AMAP
        MODE --> TMAP
    end
    subgraph CONN["each BackendConnection holds its own caps"]
        CGATE["gate(cap, label, quiet): true when served, else emits refused and sends nothing<br/>xr-client/scripts/backend_connection.gd:164 xr-client/scripts/backend_connection.gd:168"]
        DRAG["drag start, update and end ask the DRAG gate before the client is called<br/>xr-client/scripts/backend_connection.gd:282,291,300"]
    end
    subgraph SCENE["graph_scene.gd: the gate sits at the request, routed by the focused volume"]
        GATE["_gate_for_volume resolves the volume's connection and asks its gate<br/>xr-client/scripts/graph_scene.gd:2246 xr-client/scripts/graph_scene.gd:2252"]
        CTRL["_on_hud_control refuses a gated action before graph_parity.handle_control sees it<br/>xr-client/scripts/graph_scene.gd:2374 xr-client/scripts/graph_scene.gd:2375 xr-client/scripts/graph_scene.gd:2377"]
        REQ["request helpers ask the gate: fold, cycles, layout mode, radial, unpin all (PIN), reset, physics PUT,<br/>canary (quiet), relations (quiet), expand, pattern count (quiet) and execute, demo<br/>xr-client/scripts/graph_scene.gd:2536 xr-client/scripts/graph_scene.gd:2643 xr-client/scripts/graph_scene.gd:2754 xr-client/scripts/graph_scene.gd:2777 xr-client/scripts/graph_scene.gd:2797 xr-client/scripts/graph_scene.gd:2875 xr-client/scripts/graph_scene.gd:2919 xr-client/scripts/graph_scene.gd:4530 xr-client/scripts/graph_scene.gd:5139 xr-client/scripts/graph_scene.gd:5205 xr-client/scripts/graph_scene.gd:5559 xr-client/scripts/graph_scene.gd:5629 xr-client/scripts/graph_scene.gd:5684"]
        PAR["graph_parity.gd asks backend_supports, the focused volume's connection, before its forces write and its demo poll<br/>xr-client/scripts/graph_parity.gd:355 xr-client/scripts/graph_parity.gd:407 xr-client/scripts/graph_scene.gd:2236"]
        SYNC["_sync_hud_backend pushes the focused connection's caps to the HUD<br/>xr-client/scripts/graph_scene.gd:2213 xr-client/scripts/graph_scene.gd:2221"]
        GATE --> CTRL
        GATE --> REQ
    end
    subgraph HUD["hud.gd: grey, never reflow"]
        REG["_action_btn keeps an action-to-buttons registry and the original hint<br/>xr-client/scripts/hud.gd:1779"]
        APPLY["_apply_cap_to_button: disabled plus UNSUPPORTED_HINT, restored when the capability returns<br/>xr-client/scripts/hud.gd:1797 xr-client/scripts/backend_caps.gd:94"]
        SET["set_backend_capabilities: every registered button, the memory type field, pattern Execute,<br/>a section tab shows while any of its pages is live (ADR-2189),<br/>a hidden active page falls back to the Layers page<br/>xr-client/scripts/hud.gd:1808 xr-client/scripts/hud.gd:1828 xr-client/scripts/hud.gd:1830 xr-client/scripts/hud.gd:1832 xr-client/scripts/hud.gd:1833 xr-client/scripts/hud.gd:278"]
        DEC["intervention decide POST refused with the same hint when the caps lack the broker route<br/>xr-client/scripts/hud.gd:2628 xr-client/scripts/hud.gd:2636"]
        REG --> APPLY --> SET
        SET --> DEC
    end
    MODE --> CGATE
    CGATE --> GATE
    CGATE --> DRAG
    SYNC -->|"set_backend_capabilities(conn.caps)"| SET
    INV["INVARIANT: the UI and the network cannot drift apart - hiding a tab is cosmetic, the gate in the request helper is the property<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:339"]
    T1["GUT: test_backend_caps pins the table, test_hud_tabs the greying, hidden tabs, fallback and restore<br/>xr-client/tests/unit/test_backend_caps.gd:8 xr-client/tests/unit/test_hud_tabs.gd:287 xr-client/tests/unit/test_hud_tabs.gd:317"]
    REQ --> INV
    SET --> T1
```

**What it shows.** Codescape serves the graph stream, presence, voice, a read-only physics view and settlement, and answers 404 (405 for the physics PUT) to every knowledge route by design. One table names each surface; each connection holds its own copy and refuses at its gate; the scene routes a request through the focused volume's connection; the HUD greys from the focused connection's table. Since the A6 text, the table has gained `drag` (ADR-2158 E9), and the drag-end pin record it listed is gone (ADR-2172 F3). **Why it is this way.** The advisor's objection to gating at the action name was that `graph_parity.handle_control` and its per-frame demo poll issue HTTP of their own, so only a gate at the point of sending covers every route out (ADR-2143-codescape-section-separate-workspace-and-port.md:333-340). ADR-2159 moved the table from a scene-wide `_caps` onto each connection.

**Drift:** ADR-2143 A6 item 2 still names `GraphScene._gate(cap, label)` and the drag-end pin record (`ADR-2143-codescape-section-separate-workspace-and-port.md:333,336`); the gate is now `_gate_for_volume` over `BackendConnection.gate` (`graph_scene.gd:2246`; `backend_connection.gd:164`).

## VC-40.2 The exclusive switch is retired; its hygiene now lives in focus sync and connection close

```mermaid
flowchart TD
    GONE["GUT asserts the scene has no _switch_backend<br/>xr-client/tests/unit/test_backend_switch.gd:32"]
    FOCUS["Focus change, portal entry and connection change all re-sync the HUD<br/>xr-client/scripts/graph_scene.gd:2316 xr-client/scripts/graph_scene.gd:1592 xr-client/scripts/graph_scene.gd:4654"] --> SYNC["_sync_hud_backend: focused connection's caps to the HUD<br/>xr-client/scripts/graph_scene.gd:2213 xr-client/scripts/graph_scene.gd:2221"]
    SYNC --> MEM["memory cloud and search follow the live base, folded away where the route is absent<br/>xr-client/scripts/graph_scene.gd:2226 xr-client/scripts/graph_scene.gd:2227 xr-client/scripts/graph_scene.gd:2231"]
    MEM --> BASE["the base is always the VisionClaw connection's origin<br/>xr-client/scripts/graph_scene.gd:2972 xr-client/scripts/graph_scene.gd:2974"]
    CLOSE["BackendConnection.close: cancel its own HTTPRequests, release its pins while the socket is up,<br/>close the client, bump the generation<br/>xr-client/scripts/backend_connection.gd:119 xr-client/scripts/backend_connection.gd:125 xr-client/scripts/backend_connection.gd:126 xr-client/scripts/backend_connection.gd:130"]
    PINS["release_pins iterates a set nothing fills since ADR-2172 F3<br/>xr-client/scripts/backend_connection.gd:329 xr-client/scripts/backend_connection.gd:52"]
    CLOSE --> PINS
    PAR["graph_parity.on_backend_switched still cancels polls and re-reads physics<br/>xr-client/scripts/graph_parity.gd:493"] -.-> DEBT["DEBT: no production caller; only test_graph_parity reaches it<br/>xr-client/tests/unit/test_graph_parity.gd:256"]
    DOC["A6 item 4 and the XR client doc still describe _switch_backend<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:346 docs/XR-client.md:1335"] -.-> GONE
```

**What it shows.** There is no switch to abandon a server any more: both connections stay open, and each owns its requests, pins and generation. What A6 did at a switch now happens in two places: focus changes re-sync the HUD's capability table, and a connection's `close()` cancels its own requests and releases its own pins. Memory always talks to VisionClaw, so it no longer needs re-pointing for correctness. **Why it is this way.** ADR-2159 D1 gives each connection its own client, acknowledgement sequence and request state, so ids from one source can no longer be applied to another by a late reply; the epoch problem that A6 item 4 solved is now solved by per-connection generations (VC-44, VC-53.2).

**Debt:** `graph_parity.on_backend_switched` (`graph_parity.gd:493`) survives with no production caller; delete it or wire it to a connection event.

**Drift:** `docs/XR-client.md:1335` ("Switch hygiene. `_switch_backend` cancels ...") and ADR-2143 A6 item 4 (`:347`) describe a function the scene no longer has, while `docs/XR-client.md:1219` correctly says `BackendConnection` replaced it.

## VC-40.3 Colour mode 2, the Codescape web client's palette, reaches only the VisionClaw client

```mermaid
flowchart LR
    ENUM["ColorMode: Domain 0 (default), Community 1, Codescape 2; for_backend picks 2 for codescape, 0 otherwise<br/>xr-client/rust/src/domain_palette.rs:48 xr-client/rust/src/domain_palette.rs:50 xr-client/rust/src/domain_palette.rs:64 xr-client/rust/src/domain_palette.rs:82"]
    GD["gdext: set_color_mode and default_color_mode_for_backend, per client<br/>xr-client/rust/src/binary_protocol.rs:1537 xr-client/rust/src/binary_protocol.rs:1550"]
    PAL["codescape_community_color: hue (community * 83) mod 360, CSS hsl() at 0.68 / 0.56, no per-node fallback,<br/>activity heat keeps the warm blend<br/>xr-client/rust/src/render_store.rs:279 xr-client/rust/src/render_store.rs:282"]
    BASE["base_color picks the palette per node slot when the node buffer is packed<br/>xr-client/rust/src/render_store.rs:2497 xr-client/rust/src/render_store.rs:2501 xr-client/rust/src/render_store.rs:2036"]
    CYC["Encode page Colour button cycles 0, 1, 2 on the parity client; face reads Domain, Community, Codescape<br/>xr-client/scripts/hud.gd:698 xr-client/scripts/hud.gd:703 xr-client/scripts/graph_parity.gd:121 xr-client/scripts/graph_parity.gd:125 xr-client/scripts/hud.gd:2147"]
    PREF["the VisionClaw mode persists in xr_prefs.cfg and is restored onto the VisionClaw client only<br/>xr-client/scripts/graph_scene.gd:2075 xr-client/scripts/graph_scene.gd:2076 xr-client/scripts/graph_scene.gd:2082"]
    CYC --> PREF
    PREF -.-> D2["DRIFT: the comment says a Codescape switch re-applies its palette,<br/>but apply_backend_palette runs only at boot with visionclaw<br/>xr-client/scripts/graph_scene.gd:2072 xr-client/scripts/graph_scene.gd:2073 xr-client/scripts/graph_scene.gd:1388"]
    DEF["apply_backend_palette runs once, at boot, with visionclaw, on the client parity holds: the VisionClaw client<br/>xr-client/scripts/graph_scene.gd:1387 xr-client/scripts/graph_scene.gd:1388 xr-client/scripts/graph_parity.gd:507"]
    CSV["Codescape volume packs from its own connection's client, whose mode no code sets<br/>xr-client/scripts/connection_volume_source.gd:150 xr-client/scripts/connection_volume_source.gd:158"]
    ENUM --> GD --> PAL --> BASE
    GD --> CYC
    GD --> DEF
    CSV --> BASE
    CSV -.-> GAP["DEBT: the Codescape client stays at Domain, whose fallback is one default swatch for code nodes;<br/>ADR-2159 D1 gives each connection its own palette<br/>ADR-2159-the-xr-client-holds-one-backend-connection-per-content-source.md:37"]
    N1["DIVERGENCE closed in the formula: the browser's swatches are ported exactly<br/>docs/XR-client.md:1342 docs/XR-client.md:1343"]
    PAL --> N1
```

**What it shows.** The palette formula matches the Codescape desktop client, and the Colour button cycles all three modes, but only on the VisionClaw client that `GraphParity` holds. Under ADR-2159 the Codescape volume packs from a second `BinaryProtocolClient` (`connection_volume_source.gd:150`), and nothing calls `set_color_mode` on it, so it keeps the `Domain` default (`domain_palette.rs:50`). **Why it is this way.** A6 selected the palette at each exclusive switch, on the one client there was. When the switch was deleted, `apply_backend_palette` kept its boot call (`graph_scene.gd:1388`) and lost its per-switch call. Nobody moved the call onto connection construction. Source tracing only: no headset image was taken to confirm the colours drawn.


**Drift:** `docs/XR-client.md:1346` still says `apply_backend_palette` selects the backend's default "at boot and on every switch".
