---
id: VC-16
title: Interaction — drag, pin, layout, constraints and agent beams
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/PROTOCOL-registry.md
adrs: [ADR-2020, ADR-2029, ADR-2035, ADR-2055, ADR-2141]
sources:
  - ../project/src/handlers/socket_flow_handler/message_routing.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/actors/agent_beam_actor.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/constraint_actor.rs
  - ../project/src/models/force_channels.rs
  - ../project/src/utils/binary_protocol.rs
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/src/adapters/oxigraph_graph_repository.rs
  - ../project/crates/vault-core/src/domains.rs
  - ../project/docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md
  - ../project/crates/visionclaw-domain/src/models/edge.rs
  - ../project/src/services/inferred_edge_materialiser.rs
  - ../project/src/main.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/src/app_state.rs
verified_commit: 86cdd76355823a51bb64d503c005fe277726e724
---

These panels are drawn at `86cdd7635582`, where a drag is a hold (ADR-2169): the hand's hold is released separately from an explicit pin. The [current contract audit](../../estate-review/2026-11-01-drag-holds-and-acceptance.md) and [VC-52](52-drag-holds-and-observation-gates.md) trace the hold lifecycle at their own pin. The ADR-2140 layout mode (`physics.layoutMode`), the step cadence and the three settle mechanisms are drawn at `d6c091f0f` in [VC-54](54-layout-mode-step-cadence-and-settle.md).


## VC-16.1 Node drag start — the connection takes hold

```mermaid
sequenceDiagram
    autonumber
    participant C as Browser / XR client
    participant MR as message_routing<br/>src/handlers/socket_flow_handler/message_routing.rs:51
    participant H as handle_node_drag_start<br/>position_updates.rs:192
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3027

    C->>MR: JSON text frame type nodeDragStart
    Note over C: payload shape { "type": "nodeDragStart", "data": { "nodeId": 42, "position": { "x", "y", "z" } } } documented at position_updates.rs:190
    MR->>H: dispatch message_routing.rs:52
    alt act.pubkey is None position_updates.rs:198
        Note over H: REJECTED - drag requires an authenticated session pubkey
    else authenticated
        alt data, nodeId or a finite in-range position missing
            Note over H: warn and drop (position_updates.rs:202-211, sanitize_position position_updates.rs:22)
        else this connection already holds two other nodes
            Note over H: may_hold position_updates.rs:88 refuses a third hand, MAX_DRAGGED_NODES_PER_CLIENT 2 (position_updates.rs:15, check :214-220)
        else valid
            H->>H: dragged_nodes.insert, drag_last_update.insert (position_updates.rs:227-229)
            H->>FCA: send_holds - NodeInteractionMessage Dragged to graph_service_addr wakes the orchestrator, then HoldNodes{holder, holds} (position_updates.rs:132-144, call :231)
            FCA->>FCA: holds.hold(holder, id, pos) marks pinned_mask_dirty (force_compute_actor.rs:3030-3037)
            FCA->>FCA: reheat_factor, pending_impulse and the settle window are all cleared (force_compute_actor.rs:3041-3043)
            H->>C: nodeDragStartAck position_updates.rs:234-241
            H->>H: run_later(drag_timeout_ms + 100) check_drag_timeout (position_updates.rs:243-249)
        end
    end
    Note over FCA: held nodes skip integration but still exert forces on neighbours - see VC-11.2
    Note over MR,FCA: INVARIANT every drag and pin opcode requires a pubkey - the five opcodes are routed at message_routing.rs:51-65
```

## VC-16.2 Drag update and drag end

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant MR as message_routing<br/>message_routing.rs:57
    participant HU as handle_node_drag_update<br/>position_updates.rs:262
    participant HE as handle_node_drag_end<br/>position_updates.rs:366
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3027

    loop while the pointer is held
        C->>MR: nodeDragUpdate with nodeId, position, timestamp (position_updates.rs:260)
        MR->>HU: dispatch message_routing.rs:58
        opt node not yet held by this connection
            HU->>HU: implicit start, may_hold check and timeout scheduled (position_updates.rs:283-304)
        end
        alt inside MIN_DRAG_INTERVAL_MS 16 since the last update (position_updates.rs:18, :308-309)
            HU->>HU: keep only the newest position in drag_pending, flush_pending_drag after 16ms (position_updates.rs:310-317, fn :339)
            Note over HU: coalesced, not dropped - the hand's last position always lands
        else accepted
            HU->>HU: drag_last_update.insert(node_id, now) position_updates.rs:323
            HU->>FCA: HoldNodes with the new position (send_holds position_updates.rs:329-335)
            Note over HU,FCA: the GPU is the only physics - an update runs no CPU SimulationStep and no side-channel broadcast, other clients see the entitled GPU snapshot stream (position_updates.rs:325-328)
        end
    end
    C->>MR: nodeDragEnd message_routing.rs:54
    MR->>HE: dispatch message_routing.rs:55
    HE->>HE: take drag_pending, remove from dragged_nodes and drag_last_update (position_updates.rs:389-391)
    HE->>FCA: last pending position as HoldNodes, then release_holds (position_updates.rs:396-402)
    HE->>FCA: ReleaseNodes{holder, nodes} after NodeInteractionMessage Released (position_updates.rs:152-179)
    FCA->>FCA: holds.release per node then after_release (force_compute_actor.rs:3048-3058)
    FCA->>FCA: a rejoined node refreshes the mask and gets RELEASE_IMPULSE 0.05, the RELEASE_SETTLE_WINDOW 2s opens (force_compute_actor.rs:387-395, :4516, :4537)
    HE->>C: nodeDragEndAck position_updates.rs:404-411
    Note over HE,FCA: drag end lets go - an explicit nodePin is what anchors a node past the drag, see VC-16.3
```

## VC-16.3 Explicit pin and unpin, timeout and disconnect release

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant MR as message_routing<br/>message_routing.rs:60
    participant HP as handle_node_pin and handle_node_unpin<br/>position_updates.rs:422
    participant SESS as WebSocket session stopped<br/>types.rs:508
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:2993

    alt explicit pin
        C->>MR: nodePin with a position message_routing.rs:60-61
        MR->>HP: handle_node_pin position_updates.rs:422
        HP->>FCA: PinNodePositions{pins, unpin empty, reheat false} position_updates.rs:448-452
        FCA->>FCA: holds.pin, mask dirty (force_compute_actor.rs:2999-3009)
        HP->>C: nodePinAck position_updates.rs:457-464
    else explicit unpin
        C->>MR: nodeUnpin message_routing.rs:63-64
        MR->>HP: handle_node_unpin position_updates.rs:474
        HP->>FCA: PinNodePositions{pins empty, unpin, reheat true} position_updates.rs:498-502
        FCA->>FCA: holds.unpin, settle window opens, RELEASE_IMPULSE (force_compute_actor.rs:3004-3015)
        HP->>C: nodeUnpinAck position_updates.rs:507-514
    else hand went quiet
        HP->>HP: check_drag_timeout position_updates.rs:520 re-arms itself until no update for drag_timeout_ms (position_updates.rs:529-553)
        HP->>FCA: ReleaseNodes via release_holds (position_updates.rs:544)
    else client vanished mid-drag
        SESS->>SESS: Actor stopped, warn when nodes were still held (types.rs:514-520)
        SESS->>FCA: release_holds(disconnected=true) sends ReleaseHolder{holder} (types.rs:521, position_updates.rs:173-174)
        FCA->>FCA: release_holder releases every node this holder owns (force_compute_actor.rs:3062-3073)
        SESS->>SESS: drag_pending and drag_last_update cleared (types.rs:522-523)
    end
    Note over SESS,FCA: INVARIANT a dropped connection cannot leave a node HELD - release is by holder id (types.rs:82), not by the client's own list, so a late hold update cannot outlive it (types.rs:511-513)
    Note over HP,FCA: an explicit pin is not owned by a connection and survives disconnect by design - only nodeUnpin clears it (position_updates.rs:414-416)
```

## VC-16.4 Agent beam 0x23 fan-out

```mermaid
sequenceDiagram
    autonumber
    participant HUB as agent-action hub<br/>ingest side is ES-02
    participant ABA as AgentBeamActor<br/>src/actors/agent_beam_actor.rs:297 started
    participant BC as BeamCoalescer<br/>agent_beam_actor.rs:129
    participant ENC as encode_agent_actions<br/>src/utils/binary_protocol.rs
    participant CC as ClientCoordinatorActor
    participant WS as Subscribed clients

    HUB->>ABA: AgentActionEnvelope (subscribe agent_beam_actor.rs:299)
    ABA->>ABA: project_joined agent_beam_actor.rs:189 - project_action :180 maps the envelope onto the identity-blind 0x23 action, then joins it to the roster agent by taskId
    ABA->>ABA: stamp_agent_flag(source_agent_id) agent_beam_actor.rs:105 sets AGENT_NODE_FLAG 0x80000000 agent_beam_actor.rs:79
    ABA->>BC: push(event) agent_beam_actor.rs:136
    alt backlog already at MAX_PENDING_ACTIONS 256 agent_beam_actor.rs:86
        Note over BC: push evicts the oldest and returns true, dropped_total counts it (agent_beam_actor.rs:137-146), recency over completeness
        Note over ABA: backpressure warned at most every BACKPRESSURE_WARN_INTERVAL 10s agent_beam_actor.rs:101
    end
    ABA->>BC: drain already-queued events up to MAX_COALESCE_PER_FLUSH 256 (agent_beam_actor.rs:91, :373-389)
    ABA->>BC: encode_pending agent_beam_actor.rs:163
    BC->>ENC: encode_agent_actions(&self.pending) agent_beam_actor.rs:167
    Note over ENC: ONE multi-action 0x23 frame, see VC-14.5 for the header layout
    ABA->>CC: try_send BroadcastAgentActionFrame(frame) agent_beam_actor.rs:395
    alt coordinator mailbox full
        Note over ABA: backlog is NOT cleared - retried after FLUSH_RETRY_INTERVAL 20ms (agent_beam_actor.rs:96, :352, :404-418)
    else accepted
        ABA->>BC: clear() agent_beam_actor.rs:400
    end
    ABA->>CC: every ROSTER_TICK 1s the roster keep-alive frames, one try_send without coalescing (agent_beam_actor.rs:244, :303-309, :222-232)
    CC->>WS: fan out on /wss
    Note over ABA,WS: ADR-2020 the 0x23 frame is identity-blind by design - agent-id-space and KG-id-space numeric ids only (agent_beam_actor.rs:46-57)
```

## VC-16.5 Layout mode over REST — one stored field, written through the physics patch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant LH as layout_handler<br/>src/handlers/layout_handler.rs:362
    participant APP as apply_physics_patch<br/>src/settings/api/settings_routes.rs:606
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:2480

    Note over LH: scope /layout mounted by src/main.rs:1257
    C->>LH: GET /modes layout_handler.rs:365
    LH->>C: get_layout_modes layout_handler.rs:33 - current stored mode plus seven advertised (LIVE_LAYOUT_MODES layout_handler.rs:23-31)
    C->>LH: POST /mode, RateLimit 60 per minute (layout_handler.rs:368-371)
    LH->>LH: set_layout_mode layout_handler.rs:45 parses the LayoutMode
    alt parse fails
        LH->>C: 400 ErrorBadRequest naming the accepted values layout_handler.rs:64-72
    else already the stored mode
        Note over LH: nothing written, no broadcast, no resumed simulation (layout_handler.rs:86-90)
    else changed
        LH->>APP: patch layoutMode (layout_handler.rs:92-106) - store, GPU actors, SQLite, broadcast
    end
    alt GPU-resident mode
        LH->>C: success, or success=false when persisting failed (layout_handler.rs:112-125)
    else spectral or temporal
        LH->>LH: CPU one-shot placement over GetGraphData (layout_handler.rs:128-157)
    end
    C->>LH: POST /radial layout_handler.rs:373
    LH->>FCA: set_radial_layout layout_handler.rs:198 sends SetRadialLayout{mode, focus_node} layout_handler.rs:232
    C->>LH: GET status layout_handler.rs:374, POST and GET zones layout_handler.rs:375-376, POST reset layout_handler.rs:377
    Note over LH: DEBT POST /zones answers success and forwards nothing - a TODO (layout_handler.rs:313-321), GET /zones always returns an empty list (layout_handler.rs:324-328)
    LH->>FCA: reset_layout layout_handler.rs:330 sends ResetPositions layout_handler.rs:334, handler force_compute_actor.rs:3608 sets reheat_factor = 1.0 force_compute_actor.rs:3683
    Note over LH: clustered is accepted and stored but not advertised, it rides the stored clusterStrength knob (layout_handler.rs:20-22)
```

## VC-16.6 Constraint REST routes — ForceComputeActor cache, two record-only routes

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CH as constraints_handler<br/>src/handlers/constraints_handler.rs:11
    participant SET as Settings actor
    participant FCA as ForceComputeActor<br/>get_gpu_compute_addr app_state.rs:1672
    participant EX as execution.rs flag derivation

    C->>CH: POST /constraints/define constraints_handler.rs:14
    CH->>CH: define_constraints constraints_handler.rs:22 validates a ConstraintSystem constraints_handler.rs:33
    CH->>SET: merge physics computeMode 2 and UpdateSettings (constraints_handler.rs:37-77)
    CH->>FCA: UpdateConstraints constraints_handler.rs:90-94, a failure is only logged (constraints_handler.rs:99-104)
    FCA->>FCA: parse and replace_constraint_cache (force_compute_actor.rs:3376-3393)
    C->>CH: POST /constraints/apply constraints_handler.rs:15
    CH->>CH: apply_constraints constraints_handler.rs:126 validates constraintType constraints_handler.rs:143 and nodeIds constraints_handler.rs:148
    Note over CH: DEBT nothing is sent - the reply says recorded with a note about future integration (constraints_handler.rs:181-188)
    C->>CH: POST /constraints/remove constraints_handler.rs:16
    Note over CH: DEBT remove_constraints constraints_handler.rs:191 also only counts and replies (constraints_handler.rs:218-223)
    C->>CH: GET /constraints/list constraints_handler.rs:17
    CH->>FCA: GetConstraints constraints_handler.rs:236
    FCA-->>CH: always Err (force_compute_actor.rs:3368-3373)
    CH-->>C: empty list with data_source settings (constraints_handler.rs:258-273)
    C->>CH: POST /constraints/validate constraints_handler.rs:18
    CH->>CH: validate_constraint_definition constraints_handler.rs:282 - pure validation, no actor call
    FCA->>EX: next physics step with num_constraints
    alt num_constraints > 0
        Note over EX: ENABLE_CONSTRAINTS set by derive_dispatch_feature_flags force_channels.rs:504-506, called from execution.rs:1141
    else zero
        Note over EX: bit CLEARS - the kernel must not walk a stale buffer
    end
    Note over CH,EX: DIVERGENCE ADR-2029 enablement is RESIDENCY-owned - the force-channel registry marks Constraints read-only (is_read_only force_channels.rs:212)
    Note over CH,SET: TENSION ADR-2140 - define writes the physics computeMode through UpdateSettings, outside apply_physics_patch (constraints_handler.rs:37-77)
```

## VC-16.7 Interaction surface map

```mermaid
flowchart LR
    subgraph WSOPS["WebSocket JSON opcodes - /wss, pubkey required"]
        D1["nodeDragStart<br/>message_routing.rs:52 to position_updates.rs:192"]
        D2["nodeDragEnd<br/>message_routing.rs:55 to position_updates.rs:366"]
        D3["nodeDragUpdate<br/>message_routing.rs:58 to position_updates.rs:262"]
        D5["nodePin<br/>message_routing.rs:61 to position_updates.rs:422"]
        D4["nodeUnpin<br/>message_routing.rs:64 to position_updates.rs:474"]
    end
    subgraph REST["REST surfaces"]
        L1["GET /modes (layout_handler.rs:365)"]
        L2["POST /mode (layout_handler.rs:371)"]
        L3["POST /radial (layout_handler.rs:373)"]
        K1["POST /constraints/define (constraints_handler.rs:14)"]
        K2["POST /constraints/apply (constraints_handler.rs:15)"]
        K3["POST /constraints/remove (constraints_handler.rs:16)"]
        K4["GET /constraints/list (constraints_handler.rs:17)"]
        K5["POST /constraints/validate (constraints_handler.rs:18)"]
    end
    subgraph SRVPUSH["Server-initiated"]
        B1["0x23 AGENT_ACTION beam<br/>agent_beam_actor.rs:395"]
    end
    GPU["ForceComputeActor holds, pins and pinned_mask<br/>force_compute_actor.rs:2993"]
    STORE["apply_physics_patch<br/>settings_routes.rs:606"]
    CC["ClientCoordinatorActor fan-out"]

    D1 -->|"HoldNodes (position_updates.rs:144)"| GPU
    D3 -->|"HoldNodes (position_updates.rs:329)"| GPU
    D2 -->|"ReleaseNodes (position_updates.rs:176)"| GPU
    D5 -->|"PinNodePositions (position_updates.rs:448)"| GPU
    D4 -->|"PinNodePositions (position_updates.rs:498)"| GPU
    L2 -->|"layoutMode patch (layout_handler.rs:92)"| STORE
    STORE --> GPU
    L3 -->|"SetRadialLayout (layout_handler.rs:232)"| GPU
    K1 -->|"UpdateConstraints (constraints_handler.rs:91)"| GPU
    K4 -->|"GetConstraints, always Err"| GPU
    B1 --> CC
    CC --> WSOPS

    N1["Disconnect releases every hold by holder - types.rs:521"]
    N2["apply and remove reach no actor - constraints_handler.rs:181, :218"]
    N3["Ingest side of the agent beam is estate ES-02 - this file covers fan-out only"]
    GPU -.- N1
    K2 -.- N2
    B1 -.- N3
```

## VC-16.8 DAG rank layering — subClassOf provenance, domain spokes as domain_member

```mermaid
flowchart TB
    subgraph PROD["producers — what each edge carries"]
        P1["ensure_source_domain src/services/github_sync_service.rs:2538<br/>every page node gets a group, derive_source_domain<br/>falls back to infrastructure :2530"]
        P2["group round-trips through Oxigraph since 7b6330608<br/>vc:group written src/adapters/oxigraph_graph_repository.rs:239<br/>read back :1391, source_domain fallback :1441"]
        P3["materialise_domain_roots src/services/github_sync_service.rs:874<br/>applies plan_domain_roots :2282 — one derived root per populated<br/>domain (domain_root_node_id :2216), stale roots and spokes purged<br/>slugs at crates/vault-core/src/domains.rs:4"]
        P4["spoke = domain_membership_edge(root_id, member)<br/>src/services/github_sync_service.rs:2384, fn at :2396<br/>edge_type DOMAIN_MEMBER_EDGE_TYPE = domain_member<br/>crates/visionclaw-domain/src/models/edge.rs:12"]
        P5["is-a relation edges keep the folded predicate<br/>owl_property_iri Some(predicate) src/services/github_sync_service.rs:1844<br/>Oxigraph writes vc:owlProperty src/adapters/oxigraph_graph_repository.rs:551"]
        P6["reasoner-inferred child to parent edge, label hierarchical<br/>with rdfs:subClassOf provenance src/services/inferred_edge_materialiser.rs:71"]
        P1 --> P2 --> P3 --> P4
    end
    subgraph RANK["consumer — ForceComputeActor DAG ranks"]
        R1["Edge::asserts_subsumption crates/visionclaw-domain/src/models/edge.rs:239<br/>explicit subclass labels pass; hierarchical passes only when<br/>owl_property_iri is RDFS_SUBCLASS_OF_IRI (edge.rs:242-243)"]
        R2["hierarchy_pairs src/actors/gpu/force_compute_actor.rs:432<br/>filter asserts_subsumption :438, source is CHILD and target PARENT :444"]
        R3["compute_dag_ranks src/actors/gpu/force_compute_actor.rs:451<br/>roots are participants that are never a child :470-477"]
        R4["called from graph upload src/actors/gpu/force_compute_actor.rs:1144-1146<br/>ranks cached for SetRadialLayout DagRank :1151, uploaded :1152"]
        R1 --> R2 --> R3 --> R4
    end
    P4 -. "domain_member never passes asserts_subsumption" .-> R1
    P5 --> R1
    P6 --> R1
    I["INVARIANT: a rank layer exists only for a class-subsumption edge.<br/>The hierarchical label is a force category that also folds equivalentClass,<br/>sameAs, subPropertyOf and instance-of, so it ranks only with subClassOf<br/>provenance (crates/visionclaw-domain/src/models/edge.rs:239-247).<br/>ADR-2035 amended 2026-10-02, fixes 8bdece469 and 8a501fbbc."]
    R1 --- I
```

**Invariant (ADR-2035, amended 2026-10-02):** the DAG ranker layers an edge only when `Edge::asserts_subsumption` holds (`crates/visionclaw-domain/src/models/edge.rs:239-247`), applied in `hierarchy_pairs` (`src/actors/gpu/force_compute_actor.rs:438`). A `hierarchical` edge counts only with `rdfs:subClassOf` in `owl_property_iri`. Domain-root spokes carry `domain_member` (`src/services/github_sync_service.rs:2396`, `crates/visionclaw-domain/src/models/edge.rs:12`). This resolves an earlier drift: the ADR accepted the bare label on the premise that ingest never wrote membership under it, but `materialise_domain_roots` did, and every domain root ranked below its own members. The amendment records the observed census: 9,796 subclass edges and 6,400 spokes ranked together on the pre-fix binary (`docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:196-200`). A same-day follow-up (`8a501fbbc`, recorded at `docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:286-337`) made the roots themselves stable. `materialise_domain_roots` now applies the pure `plan_domain_roots` reconcile (`src/services/github_sync_service.rs:2282`), so each domain has one derived root id in every process, and the next sync purges stale counter-minted roots and their `hierarchical` spokes.

**Drift (ADR-2035 vs code):** the record's Context still says the ranker uses `is_directed_hierarchy_relation` (`docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:28-29`). Its Consequences still describe the accept as coupled to ingest labelling (`docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:61-62`). That function was deleted in 8bdece469 and replaced by `hierarchy_pairs` (`src/actors/gpu/force_compute_actor.rs:432`). The Decision section (`:37-48`) is current; the surrounding prose is not.

**Open:** the fix moves a dependency onto provenance. A producer that writes a `hierarchical` subclass edge without `rdfs:subClassOf` in `owl_property_iri` would silently stop ranking, and so would a store path that drops `vc:owlProperty` (`src/adapters/oxigraph_graph_repository.rs:551`). The ADR names this as its review trigger; nothing at boot detects it.

**Debt:** two constraint routes and one layout route answer success without effect. `POST /constraints/apply` and `/constraints/remove` reach no actor (`src/handlers/constraints_handler.rs:181-188`, `:218-223`), and `POST /layout/zones` is a TODO (`src/handlers/layout_handler.rs:317`). `GET /constraints/list` always falls back to an empty list, because `ForceComputeActor` refuses `GetConstraints` (`src/actors/gpu/force_compute_actor.rs:3368-3373`). `ConstraintActor` is not addressed by any of these routes.

**Open:** `POST /layout/radial` and `POST /layout/reset` take no `AuthenticatedUser` extractor, unlike `POST /layout/mode` (`src/handlers/layout_handler.rs:46`). Whether the enclosing `/api` scope authenticates them is outside this source trace.
