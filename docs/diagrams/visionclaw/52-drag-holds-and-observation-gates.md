---
id: VC-52
title: Drag holds and observation gates
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
adrs: [ADR-2169, ADR-2140, ADR-2167, ADR-2172, ADR-2141]
sources:
  - ../project/src/actors/gpu/hold_registry.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/scripts/drag-physics-acceptance.mjs
  - ../project/xr-client/scripts/spatial_environment.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/client/src/features/graph/hooks/useGraphEventHandlers.ts
  - ../project/client/src/features/graph/hooks/dragHeartbeat.ts
  - ../project/client/src/features/graph/workers/graph.worker.ts
  - ../project/client/src/features/graph/workers/lib/binary-processor.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
verified_commit: {visionclaw: 531df5ef74453a2ed320bff968c203a9b91faed7}
---

## For developers

The [contract review](../../estate-review/2026-11-01-drag-holds-and-acceptance.md) tests the real registry and runs the unchanged acceptance script against an intentionally wrong local peer. Holds, explicit pins, actor notifications and observed movement have separate lifetimes. Historical [VC-16](16-interaction-drag-pin-layout.md) describes the previous implementation.

## For the business

Letting go should return a node to the graph; pinning should keep it anchored. A second person needs to see whose hand controls the node and whether their view is current. A passing test is useful only if it can reject a peer that remains pinned or stops delivering updates.

## VC-52.1 A hold and an explicit pin are distinct reasons to anchor a node

```mermaid
flowchart TD
    SOCKET["Socket permits an existing hold or fewer than two<br/>position_updates.rs:183-185"] --> HOLD["Hold writes owner, position and refresh time<br/>hold_registry.rs:59-64"]
    HOLD --> RELEASE["Release checks current owner<br/>hold_registry.rs:68-70"]
    RELEASE --> OTHER["Wrong owner: ignored<br/>hold_registry.rs:70"]
    RELEASE --> PIN["Explicit pin survives release<br/>hold_registry.rs:74-75"]
    RELEASE --> FREE["Otherwise remove pinned position<br/>hold_registry.rs:77-78"]
    EXPLICIT["Explicit pin has its own set<br/>hold_registry.rs:117-121"] --> PIN
    UNPIN["Unpin removes explicit reason only<br/>hold_registry.rs:126-132"] -->|no remaining hold| FREE
```

**What it shows.** Fourteen native tests cover the registry. Three additional probes check its lifetime/capacity boundaries and model the independently maintained orchestrator set. The registry itself accepts more than two holds; admission belongs to the socket.

**Why it is this way.** A release from an old holder must not free another person's node, and a temporary hold must not erase an explicit pin.

## VC-52.2 Release notifications can disagree about ownership

```mermaid
flowchart TD
    END["Release request<br/>position_updates.rs:248-253"] --> NOTIFY["Send node-only Released notification<br/>position_updates.rs:259-266"]
    NOTIFY --> ORCH["Orchestrator removes id without owner check<br/>physics_orchestrator_actor.rs:1189-1195"]
    END --> GPU["Send holder and node ids to GPU actor<br/>position_updates.rs:268-273"]
    GPU --> OWNER["Registry ignores a stale owner's release<br/>hold_registry.rs:68-70"]
    OWNER -.-> GAP["TENSION: GPU may still hold a node<br/>after orchestrator clears its held set<br/>physics_orchestrator_actor.rs:1194"]
    LATE["A later hold writes ownership again<br/>hold_registry.rs:59-64"] --> EXPIRE["Backstop only runs within compute handling<br/>force_compute_actor.rs:1900-1913"]
```

**What it shows.** This known ADR gap is supported by source and an actual-registry/model probe, not a fresh actor integration run. Expiry readmits then eventually clears a late hold when called; it does not reject closed holders.

**Why it is this way.** Two independent notifications carry different information. Ownership-aware state should come from accepted transitions rather than unqualified interaction intents.

## VC-52.3 The effective interaction speed differs from stored speed by decision

```mermaid
flowchart TD
    RELEASE["After last effective release, open settle window<br/>force_compute_actor.rs:375-390"] --> SPAN["Held or before settle deadline<br/>force_compute_actor.rs:4492-4495"]
    SPAN --> SPEED["Effective speed is max of stored and 2.0<br/>force_compute_actor.rs:4455,4496"]
    SPAN --> HEAT["Withhold reheat; keep stored damping<br/>force_compute_actor.rs:4500-4501"]
    SPAN --> BYPASS["Bypass stability, also during warmup<br/>force_compute_actor.rs:4494"]
    DONE["Outside hold and settle span<br/>force_compute_actor.rs:4497-4499"] --> STORED["Use stored speed again<br/>force_compute_actor.rs:4498"]
```

**What it shows.** ADR-2169 explicitly amends ADR-2140. The accepted temporary floor does not edit the stored document. Passing the stored damping through cannot establish a two-second physical settling time.

**Why it is this way.** Interaction responsiveness and residual motion are different outcomes; both need live measurements under the actual force set.

## VC-52.4 Cross-client samples and silence can produce a false acceptance

```mermaid
flowchart TD
    A["Snapshot dragger position at release<br/>drag-physics-acceptance.mjs:319-320"] --> DELTA["Subtract it from observer's latest position<br/>drag-physics-acceptance.mjs:359-364"]
    DELTA --> PASS["Positive difference passes release check<br/>drag-physics-acceptance.mjs:386-388"]
    QUIET["No motion after release returns release time<br/>drag-physics-acceptance.mjs:230"] --> REST["Zero settle milliseconds passes rest check<br/>drag-physics-acceptance.mjs:353-355,381-384"]
    PASS -.-> GAP["TENSION: neither check requires a fresh<br/>post-release position observation<br/>drag-physics-acceptance.mjs:355,388"]
    REST -.-> GAP
```

**What it shows.** The negative control leaves the node pinned at 150 and sends zero post-release positions. All five checks pass; the reported travel comes from different clients' earlier samples.

**Why it is this way.** The script confuses observation differences with elapsed-time motion. A release receipt and one observer's ordered samples must establish when an observation was made; silence alone cannot prove rest.

## VC-52.5 Selecting a volume transform does not select its coordinate source

```mermaid
flowchart TD
    HAND["Choose first held hand<br/>graph_scene.gd:171-172"] --> FRAME["Select held or targeted volume transform<br/>graph_scene.gd:203-209"]
    PRIMARY["Read position from primary binary client<br/>spatial_environment.gd:92,107"] --> WORLD["Apply selected transform to that position<br/>spatial_environment.gd:108-111"]
    FRAME --> WORLD
    GRAB["Grab uses the chosen connection's position<br/>graph_scene.gd:3415"] -.-> GAP["TENSION: differing connection positions<br/>need an end-to-end marker test<br/>spatial_environment.gd:107"]
    WORLD -.-> GAP
```

**What it shows.** The new focus-frame helper is source-confirmed. A marker test with distinct positions per connection is still needed; existing transform assertions do not cover that supplier boundary.

**Why it is this way.** Node ids and transforms belong to streams/volumes. Carrying the correct frame without the matching point can place an overlay incorrectly when those streams differ.

## VC-52.6 One desktop hold from press to rejoin, across socket, orchestrator, registry and GPU

```mermaid
sequenceDiagram
    autonumber
    participant UI as useGraphEventHandlers<br/>useGraphEventHandlers.ts:159
    participant WS as handle_node_drag_start<br/>position_updates.rs:287
    participant GS as GraphServiceSupervisor<br/>graph_service_supervisor.rs:2056
    participant OR as PhysicsOrchestratorActor<br/>physics_orchestrator_actor.rs:1186
    participant FC as ForceComputeActor<br/>force_compute_actor.rs:3004
    participant HR as HoldRegistry<br/>hold_registry.rs:59
    UI->>WS: nodeDragStart with position, useGraphEventHandlers.ts:159
    WS->>WS: may_hold, then dragged_nodes and drag_last_update, position_updates.rs:309,322,323
    WS->>GS: NodeInteractionMessage Dragged, position_updates.rs:231,233
    GS->>OR: forward unchanged, graph_service_supervisor.rs:2069
    OR->>OR: held_nodes.insert, physics_orchestrator_actor.rs:1191
    WS->>FC: HoldNodes holder and position, position_updates.rs:239
    FC->>HR: hold writes owner, position, touched, hold_registry.rs:59,61,62,63
    FC->>FC: clear reheat, impulse and settle window, force_compute_actor.rs:3018,3019,3020
    loop every 500 ms while the button is down
        UI->>WS: nodeDragUpdate heartbeat at the current position, useGraphEventHandlers.ts:172,176
    end
    Note over UI: heartbeat interval dragHeartbeat.ts:9,15 is below the 2 s socket timeout
    FC->>FC: next step, mask uploaded then pins applied, force_compute_actor.rs:2124,2125
    UI->>WS: nodeDragEnd, useGraphEventHandlers.ts:278
    WS->>WS: remove from dragged_nodes and drag_last_update, position_updates.rs:485,486
    WS->>GS: NodeInteractionMessage Released, position_updates.rs:263
    GS->>OR: forward, graph_service_supervisor.rs:2069
    OR->>OR: held_nodes.remove with no owner, physics_orchestrator_actor.rs:1194
    WS->>FC: ReleaseNodes holder and nodes, position_updates.rs:271
    FC->>HR: release by current owner only, hold_registry.rs:68,69
    FC->>FC: after_release, rejoin impulse and 2 s settle window, force_compute_actor.rs:383,386,375
    Note over WS,FC: backstops, socket 2 s timeout position_updates.rs:615,639 and registry 5 s expiry force_compute_actor.rs:1903,4443
```

**What it shows.** One gesture fans out into two independent messages, one to the GPU actor and one through the graph service to the orchestrator, and each keeps its own held set. **Why it is this way.** ADR-2169 (2026-11-01) put ownership in the GPU actor's registry. The orchestrator's intent channel predates it and was kept for the auto-pause and wake behaviour.

## VC-52.7 Where the fact "this node is held or pinned" is stored

```mermaid
flowchart TD
    REG["AUTHORITATIVE: HoldRegistry held, explicit, positions, touched<br/>hold_registry.rs:39,41,43,48"] --> MASK["Derived GPU pinned_mask from registry positions<br/>force_compute_actor.rs:396,541"]
    SOCK["Socket dragged_nodes for the two-hand admission<br/>types.rs:73<br/>position_updates.rs:183,184"] -.-> D1["DEBT: same per-holder set the registry keeps<br/>hold_registry.rs:83"]
    TIME["Socket drag_last_update, 2 s timeout<br/>types.rs:75<br/>position_updates.rs:615,628"] --> LAYER["Layered with registry touched and 5 s expiry<br/>hold_registry.rs:99<br/>force_compute_actor.rs:4441,4443"]
    ORCH["Orchestrator held_nodes, node id only<br/>physics_orchestrator_actor.rs:117,1191,1194"] -.-> T1["See VC-52.2 tension"]
    DESK["Desktop worker pinnedNodeIds still read<br/>graph.worker.ts:31<br/>binary-processor.ts:74"] -.-> D2["DEBT: no writer, proxy pin calls are deprecated no-ops<br/>graphWorkerProxy.ts:410,416"]
    STALE["Doc comment for removed per-frame pin re-upload<br/>force_compute_actor.rs:365,366"] -.-> D3["DEBT: orphaned above begin_release_settle<br/>force_compute_actor.rs:375"]
```

**What it shows.** Of the server-side copies, the registry is the only one that carries ownership and the hold-versus-pin distinction. The GPU mask is derived from it, and the timeouts are deliberately layered. **Why it is this way.** ADR-2169 added the registry beside the existing socket and orchestrator state rather than replacing either.

**Debt:** the socket's `dragged_nodes` duplicates the registry's per-holder set (`types.rs:73`; `hold_registry.rs:83`), and the desktop `pinnedNodeIds` is read but never written (`graph.worker.ts:31`; `graphWorkerProxy.ts:410`).
