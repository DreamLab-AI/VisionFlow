---
id: VC-40
title: XR client on two backends - the capability table, the request gate, the switch and the Codescape palette (ADR-2143 A6)
area: visionclaw
governing:
  - ../project/docs/XR-client.md
  - ../project/docs/adr/ADR-2143-codescape-section-separate-workspace-and-port.md
adrs: [ADR-2143, ADR-2134, ADR-2108]
sources:
  - ../project/docs/XR-client.md
  - ../project/docs/adr/ADR-2143-codescape-section-separate-workspace-and-port.md
  - ../project/xr-client/scripts/backend_caps.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/hud.gd
  - ../project/xr-client/scripts/graph_parity.gd
  - ../project/xr-client/scripts/memory_search.gd
  - ../project/xr-client/rust/src/domain_palette.rs
  - ../project/xr-client/rust/src/render_store.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/xr-client/tests/unit/test_backend_caps.gd
  - ../project/xr-client/tests/unit/test_hud_tabs.gd
  - ../project/xr-client/tests/unit/test_graph_parity.gd
  - ../project/xr-client/tests/unit/test_backend_switch.gd
verified_commit: {visionclaw: e4ba5eb35ee6715cfd7f2daf1cd9af05e8a18cce}
---

## VC-40.1 One capability table gates every server-bound request and greys the matching controls

```mermaid
flowchart TB
    subgraph TABLE["backend_caps.gd: pure data, no I/O"]
        KEYS["twelve surfaces: physics_write, layout_mode, radial, fold, cycles, demo, memory_cloud,<br/>pattern_query, canary, node_relations, intervention, pin<br/>xr-client/scripts/backend_caps.gd:20 xr-client/scripts/backend_caps.gd:31 xr-client/scripts/backend_caps.gd:33"]
        MODE["for_mode: VisionClaw and any unknown mode grant everything; Codescape grants CODESCAPE_SUPPORTED, which is empty<br/>xr-client/scripts/backend_caps.gd:43 xr-client/scripts/backend_caps.gd:98"]
        AMAP["ACTION_CAPS and ACTION_PREFIX_CAPS: every server-bound HUD action to its surface,<br/>forces_step:, memory_preset:, memory_hit:, memory_typed: by prefix<br/>xr-client/scripts/backend_caps.gd:47 xr-client/scripts/backend_caps.gd:75 xr-client/scripts/backend_caps.gd:131"]
        TMAP["TAB_CAPS: forces, swarm and pins have no live control left on Codescape, so the whole tab hides<br/>xr-client/scripts/backend_caps.gd:85 xr-client/scripts/backend_caps.gd:113"]
        KEYS --> MODE
        MODE --> AMAP
        MODE --> TMAP
    end
    subgraph SCENE["graph_scene.gd: the gate sits at the request, not at the button"]
        CAPS["_caps, refreshed by _sync_hud_backend at boot and on every switch<br/>xr-client/scripts/graph_scene.gd:82 xr-client/scripts/graph_scene.gd:835"]
        GATE["_gate(cap, label, quiet): true when served, else a HUD strip notice 'Not on Codescape: ...' and false<br/>xr-client/scripts/graph_scene.gd:864"]
        CTRL["_on_hud_control refuses a gated action before graph_parity.handle_control sees it<br/>xr-client/scripts/graph_scene.gd:939 xr-client/scripts/graph_scene.gd:947"]
        REQ["every request helper asks the gate: physics PUT, reset, layout mode, radial, fold, cycles,<br/>canary (quiet), relations (quiet), expand, demo, pattern query count and execute<br/>xr-client/scripts/graph_scene.gd:1490 xr-client/scripts/graph_scene.gd:1448 xr-client/scripts/graph_scene.gd:1326 xr-client/scripts/graph_scene.gd:1346 xr-client/scripts/graph_scene.gd:1110 xr-client/scripts/graph_scene.gd:1215 xr-client/scripts/graph_scene.gd:3009 xr-client/scripts/graph_scene.gd:3591 xr-client/scripts/graph_scene.gd:3657 xr-client/scripts/graph_scene.gd:4136"]
        PINS["pins ride /wss as nodePin and nodeUnpin text frames that Codescape drops unacknowledged:<br/>the drag-end record and _unpin_all are gated on PIN<br/>xr-client/scripts/graph_scene.gd:2354 xr-client/scripts/graph_scene.gd:1368"]
        PAR["graph_parity.gd asks backend_supports before its forces write and its demo poll<br/>xr-client/scripts/graph_parity.gd:352 xr-client/scripts/graph_parity.gd:404 xr-client/scripts/graph_scene.gd:858"]
        CAPS --> GATE --> CTRL
        GATE --> REQ
        GATE --> PINS
        CAPS --> PAR
    end
    subgraph HUD["hud.gd: grey, never reflow"]
        REG["_action_btn keeps an action-to-buttons registry and the original hint<br/>xr-client/scripts/hud.gd:1423"]
        APPLY["_apply_cap_to_button: disabled plus UNSUPPORTED_HINT, restored when the capability returns<br/>xr-client/scripts/hud.gd:1441 xr-client/scripts/backend_caps.gd:92"]
        SET["set_backend_capabilities: every registered button, the memory type field, pattern Execute,<br/>the hidden tabs, fallback to Graph if the active tab hid<br/>xr-client/scripts/hud.gd:1452 xr-client/scripts/hud.gd:1476"]
        DEC["intervention decide POST refused with the same hint when the backend lacks the broker route<br/>xr-client/scripts/hud.gd:2088 xr-client/scripts/hud.gd:2096"]
        REG --> APPLY --> SET
        SET --> DEC
    end
    MODE --> CAPS
    CAPS -->|"set_backend_capabilities(_caps)"| SET
    INV["INVARIANT: the UI and the network cannot drift apart - hiding a tab is cosmetic, the gate in the request helper is the property<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:339"]
    T1["GUT: test_backend_caps pins the table, test_hud_tabs the greying, hidden tabs, fallback and restore<br/>xr-client/tests/unit/test_backend_caps.gd:8 xr-client/tests/unit/test_hud_tabs.gd:265"]
    REQ --> INV
    SET --> T1
```

**What it shows.** Codescape serves the graph stream, presence, voice, a read-only physics view and settlement, and answers 404 (405 for the physics PUT) to every knowledge route by design, so the client stopped asking: one table names each surface, the scene refuses at the request helper, and the HUD greys from the same table. **Why it is this way.** The advisor's objection to gating at the action name was that `graph_parity.handle_control` and its per-frame demo poll issue HTTP of their own, so only a gate at the point of sending covers every route out (ADR-2143-codescape-section-separate-workspace-and-port.md:333-340).

## VC-40.2 A backend switch abandons the old server before connecting to the new one

```mermaid
sequenceDiagram
    autonumber
    participant HUD as Session tab Backend button<br/>xr-client/scripts/hud.gd:1611
    participant GS as GraphScene._switch_backend<br/>xr-client/scripts/graph_scene.gd:909
    participant AB as _abandon_backend_requests<br/>xr-client/scripts/graph_scene.gd:877
    participant GPA as GraphParity<br/>xr-client/scripts/graph_parity.gd:493
    participant SYNC as _sync_hud_backend<br/>xr-client/scripts/graph_scene.gd:835
    participant MEM as MemoryCloud / MemorySearch<br/>xr-client/scripts/memory_search.gd:63
    HUD->>GS: control_pressed backend:codescape
    GS->>GS: same mode: return (:909)
    GS->>AB: cancel every HTTPRequest: physics, fold, cycles, query, exec, relations, expand, observe, demo (:877)
    AB->>AB: clear _physics_pending and _physics_staged, fold, cycles, query, exec, demo pending, the once-per-session canary latch stays
    AB->>AB: drop the applied fold plan, the cycle overlay and the query-result planes, all bound to the old epoch ids
    AB->>AB: send_node_unpin for every pinned id while the old socket is open, then clear the set
    AB->>GPA: on_backend_switched: cancel physics, settlement and demo polls, want physics again (:493)
    GS->>GS: re-point /wss and /ws/presence at the new base, reset backoff, connect
    GS->>SYNC: refresh _caps, push mode and capabilities to the HUD (:835)
    SYNC->>MEM: configure(_http_base) and set_http_base, fold the cloud away where its route is absent (:848 :853)
    GS->>GPA: apply_backend_palette: Codescape mode 2, VisionClaw mode 0 (xr-client/scripts/graph_parity.gd:507)
    Note over GS: Before A6 a physics PUT still in flight at the switch committed its staged values into the new session,<br/>and memory search kept posting to the boot-time origin<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:316-326
    Note over GPA: GUT test_backend_switch_drops_polls_and_rereads_physics<br/>xr-client/tests/unit/test_graph_parity.gd:252
    Note over AB: GUT test_switch_clears_applied_fold_pins_and_pending_state<br/>xr-client/tests/unit/test_backend_switch.gd:17 ADR-2143-codescape-section-separate-workspace-and-port.md:346
```

**What it shows.** The switch is a hygiene pass before a reconnect: every request addressed to the old server is cancelled, every "in flight" latch is reopened, and anything drawn from the old graph (cycle overlay, pins) is dropped so ids from the previous epoch are never applied to the new one. **Why it is this way.** Physics writes are commit-on-2xx (`_physics_staged` is applied only when the server accepts), so an in-flight reply arriving after the switch would have been the one path that mutated client state from the wrong server.

## VC-40.3 Colour mode 2: the Codescape web client's community palette, agreed swatch for swatch

```mermaid
flowchart LR
    ENUM["ColorMode: Domain 0, Community 1, Codescape 2; for_backend picks 2 for codescape, 0 otherwise<br/>xr-client/rust/src/domain_palette.rs:48 xr-client/rust/src/domain_palette.rs:57 xr-client/rust/src/domain_palette.rs:64 xr-client/rust/src/domain_palette.rs:82"]
    GD["gdext: set_color_mode and default_color_mode_for_backend<br/>xr-client/rust/src/binary_protocol.rs:1256 xr-client/rust/src/binary_protocol.rs:1269"]
    PAL["codescape_community_color: hue (community * 83) mod 360, CSS hsl() at 0.68 / 0.56, no per-node fallback,<br/>activity heat keeps the warm blend<br/>xr-client/rust/src/render_store.rs:274 xr-client/rust/src/render_store.rs:282"]
    BASE["base_color picks the palette per node slot at pack time<br/>xr-client/rust/src/render_store.rs:2265 xr-client/rust/src/render_store.rs:2269"]
    CYC["Graph tab Colour button cycles 0, 1, 2; face reads Domain, Community, Codescape<br/>xr-client/scripts/graph_parity.gd:121 xr-client/scripts/hud.gd:1684"]
    DEF["apply_backend_palette at boot and on every switch<br/>xr-client/scripts/graph_parity.gd:507"]
    ENUM --> GD --> PAL --> BASE
    GD --> CYC
    GD --> DEF
    N1["DIVERGENCE closed: the golden-ratio walk with a per-node fallback gave the headset and the browser different swatches for the same community<br/>docs/XR-client.md:978"]
    PAL --> N1
```

**What it shows.** The headset now reads a Codescape community exactly as the desktop client does, so a hull or a cluster has one colour across both views. **Why it is this way.** Code nodes carry no corpus domain, so the WP1 domain palette collapses to one default swatch on Codescape, and the old community walk disagreed with the browser; porting the browser's formula (not inventing a third) is what makes voice over a shared scene usable.
