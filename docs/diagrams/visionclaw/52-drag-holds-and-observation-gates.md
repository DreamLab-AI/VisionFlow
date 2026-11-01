---
id: VC-52
title: Drag holds and observation gates
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
adrs: [ADR-2169, ADR-2140, ADR-2167, ADR-2172]
sources:
  - ../project/src/actors/gpu/hold_registry.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/scripts/drag-physics-acceptance.mjs
  - ../project/xr-client/scripts/spatial_environment.gd
  - ../project/xr-client/scripts/graph_scene.gd
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
