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
  - ../project/scripts/lib/drag-release-evidence.mjs
  - ../project/xr-client/scripts/spatial_environment.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/client/src/features/graph/hooks/useGraphEventHandlers.ts
  - ../project/client/src/features/graph/hooks/dragHeartbeat.ts
  - ../project/client/src/features/graph/workers/graph.worker.ts
  - ../project/client/src/features/graph/workers/lib/binary-processor.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/src/utils/unified_gpu_compute/execution.rs
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

The [contract review](../../estate-review/2026-11-01-drag-holds-and-acceptance.md) tests the real registry and runs the unchanged acceptance script against an intentionally wrong local peer. Holds, explicit pins, actor notifications and observed movement have separate lifetimes. Historical [VC-16](16-interaction-drag-pin-layout.md) describes the previous implementation.

## For the business

Letting go should return a node to the graph; pinning should keep it anchored. A second person needs to see whose hand controls the node and whether their view is current. A passing test is useful only if it can reject a peer that remains pinned or stops delivering updates.

## VC-52.1 A hold and an explicit pin are distinct reasons to anchor a node

```mermaid
flowchart TD
    SOCKET["Socket permits an existing hold or fewer than two<br/>position_updates.rs:87-89"] --> HOLD["Hold writes owner, position and refresh time<br/>hold_registry.rs:59-64"]
    HOLD --> RELEASE["Release checks current owner<br/>hold_registry.rs:68-70"]
    RELEASE --> OTHER["Wrong owner: ignored<br/>hold_registry.rs:70"]
    RELEASE --> PIN["Explicit pin survives release<br/>hold_registry.rs:74-75"]
    RELEASE --> FREE["Otherwise remove pinned position<br/>hold_registry.rs:77-78"]
    EXPLICIT["Explicit pin has its own set<br/>hold_registry.rs:117-121"] --> PIN
    UNPIN["Unpin removes explicit reason only<br/>hold_registry.rs:126-132"] -->|no remaining hold| FREE
```

**What it shows.** Fourteen native tests cover the registry. Three additional probes check its lifetime/capacity boundaries and model the independently maintained orchestrator set. The registry itself accepts more than two holds; admission belongs to the socket.

**Why it is this way.** A release from an old holder must not free another person's node, and a temporary hold must not erase an explicit pin.

## VC-52.2 Release notifications now agree on ownership, but expiry reaches only one of the two held sets

```mermaid
flowchart TD
    END["Release request<br/>position_updates.rs:152-157"] --> NOTIFY["Released notification now carries the holder<br/>position_updates.rs:164-172"]
    NOTIFY --> ORCH["Orchestrator removes the id only when the holder matches<br/>physics_orchestrator_actor.rs:1183 physics_orchestrator_actor.rs:1188-1189"]
    END --> GPU["Send holder and node ids to GPU actor<br/>position_updates.rs:174-179"]
    GPU --> OWNER["Registry ignores a stale owner's release<br/>hold_registry.rs:68-70"]
    LATE["A later hold writes ownership again in both copies<br/>hold_registry.rs:59-64 physics_orchestrator_actor.rs:1198"] --> EXPIRE["Registry backstop expires a silent hold after 5 s, inside compute handling<br/>force_compute_actor.rs:1919-1932 force_compute_actor.rs:4540"]
    EXPIRE -.-> GAP["OPEN: expiry calls after_release only, the orchestrator is not told,<br/>so its held entry stays until a matching Released arrives<br/>force_compute_actor.rs:1932 physics_orchestrator_actor.rs:117"]
```

**What it shows.** The stale-holder tension of the 2026-11-01 review is closed in source: `NodeInteractionMessage` now carries `holder` and the orchestrator keeps a node-to-holder map. **Why it is this way.** A second owner's grab must not be ended by the first owner's late release (`physics_orchestrator_actor.rs:1183`).
**Open (risk):** A Released lost in transit is exactly the case the registry's 5 s backstop exists for (`force_compute_actor.rs:1917-1919`), yet the orchestrator has no matching expiry, so its cadence stays at zero and equilibrium auto-pause stays blocked (`physics_orchestrator_actor.rs:115-117`).

## VC-52.3 While held and settling, spring mode switches the integrator and strands the speed floor

```mermaid
flowchart TD
    RELEASE["After last effective release, open settle window<br/>force_compute_actor.rs:379-394"] --> SPAN["Held or before settle deadline<br/>force_compute_actor.rs:4597,4599"]
    SPAN --> SPEED["Effective speed is max of stored and 2.0<br/>force_compute_actor.rs:4552,4601"]
    SPAN --> HEAT["Withhold reheat; keep stored damping<br/>force_compute_actor.rs:4605,4606"]
    SPAN --> BYPASS["Bypass stability, also during warmup<br/>force_compute_actor.rs:4599"]
    SPAN --> OVER["Elastic overlay for hold and settle window; snapshot springs for the hold only<br/>force_compute_actor.rs:4607,4608,2015,2016"]
    OVER --> VERLET["Overlay forces adaptive_speed to 0 for the step<br/>execution.rs:1159,1160"]
    OVER --> GAIN["Springs: rest lengths snapshotted at grab; spring_k and max_force times the hold gain<br/>execution.rs:718,1164,1165,1166"]
    VERLET --> INT["Verlet reads dt and damping, not global_speed<br/>visionclaw_unified.cu:1086,1088,1089"]
    SPEED -.-> DEAD["DEBT: only the adaptive branch reads global_speed,<br/>so the floor has no reader in the span it applies to<br/>visionclaw_unified.cu:1059,1073"]
    DONE["Outside hold and settle span<br/>force_compute_actor.rs:4602-4604"] --> STORED["Stored speed, no overlay, stored integrator again<br/>force_compute_actor.rs:4603"]
```

**What it shows.** ADR-2169 explicitly amends ADR-2140. Since the 2026-11-01 review, the amendment's spring mode has merged. A hold and its settle window now turn on an elastic overlay, which sends integration down the Verlet path. Stored `dt` and damping therefore govern residual motion in the window. The `HOLD_GLOBAL_SPEED` floor is still computed, but no kernel reads it while it applies. **Why it is this way.** The floor was the first answer to a weak pull under LinLog plus adaptive speed (its doc comment still describes the FA2 integrator, `force_compute_actor.rs:4542-4551`). The spring overlay replaced that mechanism without removing the floor. Interaction response and a two-second settling time both still need live measurement under the actual force set.

**Debt:** `HOLD_GLOBAL_SPEED` and `StepOverrides::global_speed` (`force_compute_actor.rs:4552,4568`) are dead while the overlay is on; delete them or fold them into the overlay's gain, and refresh the doc comment.

## VC-52.4 Release evidence now comes from one observer's ordered post-release samples

```mermaid
flowchart TD
    ORD["Observer drops any frame whose sequence is not newer<br/>drag-physics-acceptance.mjs:178-179"] --> LOG["Each frame logged with sequence and full positions<br/>drag-physics-acceptance.mjs:189"]
    REL["Release time and the observer's last sequence before release<br/>drag-physics-acceptance.mjs:306-307"] --> EV["releaseEvidence over the observer log only<br/>drag-physics-acceptance.mjs:340 drag-release-evidence.mjs:2"]
    LOG --> EV
    EV --> SKIP["Frames at or before release, or not newer than the release sequence, are skipped<br/>drag-release-evidence.mjs:9"]
    SKIP --> TRAV["Travel sums distances between successive post-release samples of each held node<br/>drag-release-evidence.mjs:13-22"]
    SKIP --> SET["settle is null when nothing moved after release, so silence fails the rest check<br/>drag-release-evidence.mjs:28"]
    TRAV --> PASS["Every released node must travel beyond moveEps<br/>drag-physics-acceptance.mjs:368"]
    SET --> REST["Settle must exist and fit the budget<br/>drag-physics-acceptance.mjs:363"]
    TRAV -.-> OPEN1["OPEN: the first post-release sample is only a baseline, so a node that moves once and then rests reports zero travel<br/>drag-release-evidence.mjs:16-22"]
```

**What it shows.** Both earlier false-acceptance routes are closed: travel no longer subtracts another client's pre-release snapshot, and an empty post-release stream yields no settle time. **Why it is this way.** The 2026-11-03 external fixes required sequence-ordered post-release observer samples (`drag-release-evidence.mjs:1`).

## VC-52.5 Selecting a volume transform does not select its coordinate source

```mermaid
flowchart TD
    HAND["Choose first held hand<br/>graph_scene.gd:174-175"] --> FRAME["Select held or targeted volume transform<br/>graph_scene.gd:206-212"]
    PRIMARY["Read position from primary binary client<br/>spatial_environment.gd:102,113"] --> WORLD["Apply selected transform to that position<br/>spatial_environment.gd:114-117"]
    FRAME --> WORLD
    GRAB["Grab uses the chosen connection's position<br/>graph_scene.gd:3848"] -.-> GAP["TENSION: differing connection positions<br/>need an end-to-end marker test<br/>spatial_environment.gd:113"]
    WORLD -.-> GAP
```

**What it shows.** The new focus-frame helper is source-confirmed. A marker test with distinct positions per connection is still needed; existing transform assertions do not cover that supplier boundary.

**Why it is this way.** Node ids and transforms belong to streams/volumes. Carrying the correct frame without the matching point can place an overlay incorrectly when those streams differ.

## VC-52.6 One desktop hold from press to rejoin, across socket, orchestrator, registry and GPU

```mermaid
sequenceDiagram
    autonumber
    participant UI as useGraphEventHandlers<br/>useGraphEventHandlers.ts:159
    participant WS as handle_node_drag_start<br/>position_updates.rs:193
    participant GS as GraphServiceSupervisor<br/>graph_service_supervisor.rs:2067
    participant OR as PhysicsOrchestratorActor<br/>physics_orchestrator_actor.rs:1198
    participant FC as ForceComputeActor<br/>force_compute_actor.rs:3027
    participant HR as HoldRegistry<br/>hold_registry.rs:59
    UI->>WS: nodeDragStart with position, useGraphEventHandlers.ts:159
    WS->>WS: may_hold, then dragged_nodes and drag_last_update, position_updates.rs:215,228,229
    WS->>GS: NodeInteractionMessage Dragged, position_updates.rs:135,138
    GS->>OR: forward unchanged, graph_service_supervisor.rs:2080
    OR->>OR: track_hold_owner records node and holder, physics_orchestrator_actor.rs:1183,1186
    WS->>FC: HoldNodes holder and position, position_updates.rs:144
    FC->>HR: hold writes owner, position, touched, hold_registry.rs:59,61,62,63
    FC->>FC: clear reheat, impulse and settle window, force_compute_actor.rs:3041,3042,3043
    loop every 500 ms while the button is down
        UI->>WS: nodeDragUpdate heartbeat at the current position, useGraphEventHandlers.ts:172,176
    end
    Note over UI: heartbeat interval dragHeartbeat.ts:9,15 is below the 2 s socket timeout
    FC->>FC: next step, mask uploaded then pins applied, force_compute_actor.rs:2147,2148
    UI->>WS: nodeDragEnd, useGraphEventHandlers.ts:278
    WS->>WS: remove from dragged_nodes and drag_last_update, position_updates.rs:391,392
    WS->>GS: NodeInteractionMessage Released, position_updates.rs:169
    GS->>OR: forward, graph_service_supervisor.rs:2080
    OR->>OR: removes only when the holder matches, physics_orchestrator_actor.rs:1188,1189
    WS->>FC: ReleaseNodes holder and nodes, position_updates.rs:177
    FC->>HR: release by current owner only, hold_registry.rs:68,69
    FC->>FC: after_release, rejoin impulse and 2 s settle window, force_compute_actor.rs:387,390,379
    Note over WS,FC: backstops, socket 2 s timeout position_updates.rs:521,545 and registry 5 s expiry force_compute_actor.rs:1922,4540
```

**What it shows.** One gesture fans out into two independent messages, one to the GPU actor and one through the graph service to the orchestrator, and each keeps its own held set. **Why it is this way.** ADR-2169 (2026-11-01) put ownership in the GPU actor's registry. The orchestrator's intent channel predates it and was kept for the auto-pause and wake behaviour.

## VC-52.7 Where the fact "this node is held or pinned" is stored

```mermaid
flowchart TD
    REG["AUTHORITATIVE: HoldRegistry held, explicit, positions, touched<br/>hold_registry.rs:39,41,43,48"] --> MASK["Derived GPU pinned_mask from registry positions<br/>force_compute_actor.rs:400,545"]
    SOCK["Socket dragged_nodes for the two-hand admission<br/>types.rs:73<br/>position_updates.rs:87,88"] -.-> D1["DEBT: same per-holder set the registry keeps<br/>hold_registry.rs:83"]
    TIME["Socket drag_last_update, 2 s timeout<br/>types.rs:75<br/>position_updates.rs:521,534"] --> LAYER["Layered with registry touched and 5 s expiry<br/>hold_registry.rs:99<br/>force_compute_actor.rs:4538,4540"]
    ORCH["Orchestrator held_nodes, node id to holder<br/>physics_orchestrator_actor.rs:117,1186,1189"] -.-> T1["owner-checked, but not expired with the registry, see VC-52.2"]
    DESK["Desktop worker pinnedNodeIds still read<br/>graph.worker.ts:31<br/>binary-processor.ts:74"] -.-> D2["DEBT: no writer, proxy pin calls are deprecated no-ops<br/>graphWorkerProxy.ts:454,460"]
    STALE["Doc comment for removed per-frame pin re-upload<br/>force_compute_actor.rs:369,370"] -.-> D3["DEBT: orphaned above begin_release_settle<br/>force_compute_actor.rs:379"]
```

**What it shows.** Of the server-side copies, the registry is the only one that carries ownership and the hold-versus-pin distinction. The GPU mask is derived from it, and the timeouts are deliberately layered. **Why it is this way.** ADR-2169 added the registry beside the existing socket and orchestrator state rather than replacing either.

**Drift:** the desktop drag handlers cite "ADR-03 D7: pin state is server-authoritative", but no ADR-03 or ADR-003 exists in `docs/adr` or `docs/archive` (`useGraphEventHandlers.ts:151,269`).

**Debt:** the socket's `dragged_nodes` duplicates the registry's per-holder set (`types.rs:73`; `hold_registry.rs:83`), and the desktop `pinnedNodeIds` is read but never written (`graph.worker.ts:31`; `graphWorkerProxy.ts:454`).
