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
verified_commit: e74bcddc772d47a45b1971156a2bab540eaa027c
---

## VC-16.1 Node drag start — pin acquisition

```mermaid
sequenceDiagram
    autonumber
    participant C as Browser / XR client
    participant MR as message_routing<br/>src/handlers/socket_flow_handler/message_routing.rs:52
    participant H as handle_node_drag_start<br/>position_updates.rs:188
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3009

    C->>MR: JSON text frame type nodeDragStart
    Note over C: payload shape { "type": "nodeDragStart", "data": { "nodeId": 42, "position": { "x", "y", "z" } } } documented at position_updates.rs:186
    MR->>H: dispatch :52
    alt act.pubkey is None :194
        Note over H: REJECTED - drag requires an authenticated session pubkey, anonymous clients cannot pin
    else authenticated
        alt data field missing :202
            Note over H: warn "nodeDragStart missing 'data' field"
        else position not finite or out of range :237
            Note over H: warn "nodeDragStart - rejecting invalid position" and drop
        else valid
            H->>H: drag_last_update.insert(node_id, Instant::now()) :262
            H->>H: NodeInteractionMessage Dragged to graph_service_addr resumes an auto-paused simulation :268-275
    H->>FCA: do_send PinNodePositions{pins, unpin, reheat true} :283-287
            FCA->>FCA: set entries in pinned_mask :3009
            FCA->>FCA: reheat_factor = max(reheat_factor, 0.3) :3022
            H->>C: nodeDragStartAck :295
        end
    end
    Note over FCA: pinned nodes SKIP integration but STILL exert forces on neighbours - see VC-11.2
    Note over MR,FCA: INVARIANT every drag and pin opcode requires a pubkey - the four opcodes are routed at message_routing.rs:52,55,58,61
```

## VC-16.2 Drag update and drag end

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant MR as message_routing<br/>message_routing.rs:58
    participant HU as handle_node_drag_update<br/>position_updates.rs:323
    participant HE as handle_node_drag_end<br/>position_updates.rs:443
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3009

    loop while the pointer is held
        C->>MR: nodeDragUpdate with nodeId, position, timestamp (position_updates.rs:321)
        MR->>HU: dispatch :58
        HU->>HU: consult drag_last_update.get(&node_id) :349
        alt throttle window not elapsed
            Note over HU: update coalesced - the per-node timestamp map is the rate gate
        else accepted
            HU->>HU: drag_last_update.insert(node_id, now) :408
            HU->>FCA: do_send PinNodePositions :421 with the new position, reheat false :424
            Note over HU,FCA: the GPU is the only physics - an update runs no CPU SimulationStep settle and no side-channel broadcast, other clients see the entitled GPU snapshot stream (:411-417)
        end
    end
    C->>MR: nodeDragEnd :55
    MR->>HE: dispatch
    HE->>HE: drag_last_update.remove(&node_id) :475
    HE->>HE: NodeInteractionMessage Released to graph_service_addr :484-490, no settle loop and no PinNodePositions
    Note over HE,FCA: drag end releases the drag bookkeeping - an explicit nodeUnpin is what clears the GPU pin, see VC-16.3
```

## VC-16.3 Unpin and orphaned-drag cleanup on disconnect

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant MR as message_routing<br/>message_routing.rs:61
    participant HU as handle_node_unpin<br/>position_updates.rs:516
    participant SESS as WebSocket session stopped<br/>types.rs:489
    participant FCA as ForceComputeActor

    alt explicit unpin
        C->>MR: nodeUnpin :61
        MR->>HU: dispatch
        HU->>HU: drag_last_update.remove(&node_id) :548
        HU->>FCA: do_send PinNodePositions :554 with the node in the unpin list
    else client vanished mid-drag
        SESS->>SESS: Actor stopped with drags still open
        SESS->>FCA: do_send PinNodePositions{pins: Vec::new(), unpin: node_ids, reheat: false} :510-514
        Note over SESS: debug "Cleaned up N orphaned drags on disconnect" :516
        Note over SESS: warn "Client disconnected with N nodes still dragged, sending release" :518-521
        SESS->>SESS: drag_last_update.clear() :523
    end
    Note over SESS,FCA: INVARIANT a dropped connection can never leave a node pinned forever - cleanup is unconditional on stop
    Note over SESS: reheat is FALSE on the cleanup path so a disconnect does not re-energise the layout
    Note over HU,FCA: check_drag_timeout (:577) is a per-node timeout watchdog, not a bulk sweep - invoked from handle_node_drag_start (:309) and self-reschedules via ctx.run_later (:623) until the node stops timing out or times out and auto-unpins via PinNodePositions (:594-616)
```

## VC-16.4 Agent beam 0x23 fan-out

```mermaid
sequenceDiagram
    autonumber
    participant HUB as agent-action hub<br/>ingest side is ES-02
    participant ABA as AgentBeamActor<br/>src/actors/agent_beam_actor.rs:197 started
    participant BC as BeamCoalescer<br/>agent_beam_actor.rs:104
    participant ENC as encode_agent_actions<br/>src/utils/binary_protocol.rs
    participant CC as ClientCoordinatorActor
    participant WS as Subscribed clients

    HUB->>ABA: AgentActionEnvelope
    ABA->>ABA: project_action :175 maps the envelope onto the identity-blind 0x23 action
    ABA->>ABA: stamp_agent_flag(source_agent_id) :100 sets AGENT_NODE_FLAG 0x80000000 :74
    ABA->>BC: push(event) :131
    alt backlog already at MAX_PENDING_ACTIONS 256 :81
        Note over BC: push returns false - the oldest are dropped, dropped_total :152 counts them, recency is preferred over completeness :78
        Note over ABA: backpressure warned at most every BACKPRESSURE_WARN_INTERVAL 10s :96
    end
    ABA->>BC: encode_pending :158
    BC->>ENC: encode_agent_actions(&self.pending) :162
    Note over ENC: ONE multi-action 0x23 frame - up to MAX_COALESCE_PER_FLUSH 256 actions (agent_beam_actor.rs:86), see VC-14.5 for the 15-byte header layout
    ABA->>CC: try_send BroadcastAgentActionFrame(frame) :285
    alt coordinator mailbox full
        Note over ABA: backlog is NOT cleared - retried after FLUSH_RETRY_INTERVAL 20ms :91
    else accepted
        ABA->>BC: clear() :167
        Note over ABA: debug "AgentBeamActor - dispatched coalesced 0x23 frame (N action(s))" :288
    end
    CC->>WS: fan out on /wss
    Note over ABA,WS: ADR-2020 the 0x23 frame is identity-blind by design - it carries agent-id-space numeric ids only, never a pubkey or DID :46-52
```

## VC-16.5 Layout mode switch over REST — post-ADR-2055

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant LH as layout_handler<br/>src/handlers/layout_handler.rs:307
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:2404

    C->>LH: GET /modes :310
    LH->>C: get_layout_modes :30 - five modes advertised (LIVE_LAYOUT_MODES :22-27, ADR-2055)
    C->>LH: POST /mode :311
    LH->>LH: set_layout_mode :38 parses the LayoutMode
    alt parse fails
        LH->>C: RESOLVED ADR-2055 - 400 ErrorBadRequest naming the accepted values, no longer a silent coercion to ForceDirected :58-60
    else parsed
        LH->>FCA: addr.send(SetLayoutMode{mode}) :73
        FCA->>FCA: SetLayoutMode handler :2404 on the live UnifiedGPUCompute path
    end
    C->>LH: POST /radial :312
    LH->>LH: set_radial_layout :171
    LH->>FCA: SetRadialLayout :205 -> handler at force_compute_actor.rs:2462
    C->>LH: GET status :313 / POST zones :314 / GET zones :315 / reset :316
    LH->>FCA: reset_layout :275 triggers ResetPositions :279 -> handler at force_compute_actor.rs:3565 which sets reheat_factor = 1.0 :3640
    Note over LH: RESOLVED ADR-2055 - Clustered is no longer advertised. It had no dedicated arm in SetLayoutMode and was indistinguishable from ForceDirected, so advertising it promised behaviour the server could not deliver
    Note over LH,FCA: REMOVED ADR-2055 - the physics-v2 engine registry that this diagram previously showed as a compiled-out alternative path no longer exists
```

## VC-16.6 Constraint CRUD to GPU residency

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CH as constraints_handler<br/>src/handlers/constraints_handler.rs:11
    participant CA as ConstraintActor<br/>src/actors/gpu/constraint_actor.rs:199
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3354
    participant EX as execution.rs flag derivation

    C->>CH: POST /constraints/define :14
    CH->>CH: define_constraints :22 takes web::Json<ConstraintSystem>
    CH->>CA: UpdateConstraints (constraint_actor.rs:199)
    C->>CH: POST /constraints/apply :15
    CH->>CH: apply_constraints :126 over an untyped Value
    alt constraintType absent
        Note over CH: ErrorBadRequest "constraintType is required" :143
    else nodeIds absent
        Note over CH: ErrorBadRequest "nodeIds array is required" :148
    else valid
        CH->>CA: UpdateConstraints
    end
    C->>CH: POST /constraints/remove :16
    CH->>CH: remove_constraints :191
    CH->>CA: ClearConstraints (constraint_actor.rs:262)
    C->>CH: GET /constraints/list :17
    CH->>CA: GetConstraints (constraint_actor.rs:221) and GetConstraintStatistics (constraint_actor.rs:270)
    C->>CH: POST /constraints/validate :18
    CH->>CH: validate_constraint_definition :282 over LegacyConstraintData - pure validation, no actor call
    CA->>FCA: UploadConstraintsToGPU :230 then (force_compute_actor.rs:3354)
    FCA->>EX: next physics step with num_constraints
    alt num_constraints > 0
        Note over EX: ENABLE_CONSTRAINTS set by derive_dispatch_feature_flags force_channels.rs:504-506
    else zero
        Note over EX: bit CLEARS - the kernel must not walk a stale buffer
    end
    Note over CH,EX: DIVERGENCE ADR-2029 enablement is RESIDENCY-owned - the force-channel registry marks Constraints read-only (is_read_only force_channels.rs:212) so any UI toggle routed through the registry is inert by design
```

## VC-16.7 Interaction surface map

```mermaid
flowchart LR
    subgraph WSOPS["WebSocket JSON opcodes - /wss, pubkey required"]
        D1["nodeDragStart<br/>message_routing.rs:52 to position_updates.rs:188"]
        D2["nodeDragEnd<br/>message_routing.rs:55 to position_updates.rs:443"]
        D3["nodeDragUpdate<br/>message_routing.rs:58 to position_updates.rs:323"]
        D4["nodeUnpin<br/>message_routing.rs:61 to position_updates.rs:516"]
    end
    subgraph REST["REST surfaces"]
        L1["GET /modes (layout_handler.rs:310)"]
        L2["POST /mode (layout_handler.rs:311)"]
        L3["POST /radial (layout_handler.rs:312)"]
        K1["POST /constraints/define (constraints_handler.rs:14)"]
        K2["POST /constraints/apply (constraints_handler.rs:15)"]
        K3["POST /constraints/remove (constraints_handler.rs:16)"]
        K4["GET /constraints/list (constraints_handler.rs:17)"]
        K5["POST /constraints/validate (constraints_handler.rs:18)"]
    end
    subgraph SRVPUSH["Server-initiated"]
        B1["0x23 AGENT_ACTION beam<br/>agent_beam_actor.rs:285"]
    end
    GPU["ForceComputeActor pinned_mask and SimParams<br/>force_compute_actor.rs:3009"]
    CA["ConstraintActor<br/>constraint_actor.rs:199"]
    CC["ClientCoordinatorActor fan-out"]

    D1 -->|"PinNodePositions (position_updates.rs:283)"| GPU
    D3 -->|"PinNodePositions (position_updates.rs:421)"| GPU
    D4 -->|"PinNodePositions (position_updates.rs:554)"| GPU
    D2 -->|"drag_last_update.remove (position_updates.rs:475)"| GPU
    L2 -->|"SetLayoutMode (layout_handler.rs:73)"| GPU
    L3 -->|"SetRadialLayout"| GPU
    K1 --> CA
    K2 --> CA
    K3 --> CA
    CA -->|"UploadConstraintsToGPU (constraint_actor.rs:230)"| GPU
    B1 --> CC
    CC --> WSOPS

    N1["Disconnect cleanup releases every orphaned pin - types.rs:510-514"]
    N2["Ingest side of the agent beam is estate ES-02 - this file covers fan-out only"]
    GPU -.- N1
    B1 -.- N2
```

## VC-16.8 DAG rank layering — subClassOf provenance, domain spokes as domain_member

```mermaid
flowchart TB
    subgraph PROD["producers — what each edge carries"]
        P1["ensure_source_domain src/services/github_sync_service.rs:2538<br/>every page node gets a group, derive_source_domain<br/>falls back to infrastructure :2530"]
        P2["group round-trips through Oxigraph since 7b6330608<br/>vc:group written src/adapters/oxigraph_graph_repository.rs:239<br/>read back :1412, source_domain fallback :1462"]
        P3["materialise_domain_roots src/services/github_sync_service.rs:874<br/>applies plan_domain_roots :2282 — one derived root per populated<br/>domain (domain_root_node_id :2216), stale roots and spokes purged<br/>slugs at crates/vault-core/src/domains.rs:4"]
        P4["spoke = domain_membership_edge(root_id, member)<br/>src/services/github_sync_service.rs:2384, fn at :2396<br/>edge_type DOMAIN_MEMBER_EDGE_TYPE = domain_member<br/>crates/visionclaw-domain/src/models/edge.rs:12"]
        P5["is-a relation edges keep the folded predicate<br/>owl_property_iri Some(predicate) src/services/github_sync_service.rs:1844<br/>Oxigraph writes vc:owlProperty src/adapters/oxigraph_graph_repository.rs:551"]
        P6["reasoner-inferred child to parent edge, label hierarchical<br/>with rdfs:subClassOf provenance src/services/inferred_edge_materialiser.rs:71"]
        P1 --> P2 --> P3 --> P4
    end
    subgraph RANK["consumer — ForceComputeActor DAG ranks"]
        R1["Edge::asserts_subsumption crates/visionclaw-domain/src/models/edge.rs:239<br/>explicit subclass labels pass; hierarchical passes only when<br/>owl_property_iri is RDFS_SUBCLASS_OF_IRI (edge.rs:242-243)"]
        R2["hierarchy_pairs src/actors/gpu/force_compute_actor.rs:419<br/>filter asserts_subsumption :425, source is CHILD and target PARENT :431"]
        R3["compute_dag_ranks src/actors/gpu/force_compute_actor.rs:438<br/>roots are participants that are never a child :460"]
        R4["called from graph upload src/actors/gpu/force_compute_actor.rs:1125<br/>ranks cached for SetRadialLayout DagRank :1132, uploaded :1133"]
        R1 --> R2 --> R3 --> R4
    end
    P4 -. "domain_member never passes asserts_subsumption" .-> R1
    P5 --> R1
    P6 --> R1
    I["INVARIANT: a rank layer exists only for a class-subsumption edge.<br/>The hierarchical label is a force category that also folds equivalentClass,<br/>sameAs, subPropertyOf and instance-of, so it ranks only with subClassOf<br/>provenance (crates/visionclaw-domain/src/models/edge.rs:239-247).<br/>ADR-2035 amended 2026-10-02, fixes 8bdece469 and 8a501fbbc."]
    R1 --- I
```

**Invariant (ADR-2035, amended 2026-10-02):** the DAG ranker layers an edge only when `Edge::asserts_subsumption` holds (`crates/visionclaw-domain/src/models/edge.rs:239-247`), applied in `hierarchy_pairs` (`src/actors/gpu/force_compute_actor.rs:425`); a `hierarchical` edge counts only with `rdfs:subClassOf` in `owl_property_iri`, and domain-root spokes carry `domain_member` (`src/services/github_sync_service.rs:2396`, `crates/visionclaw-domain/src/models/edge.rs:12`). This resolves the earlier drift in which the ADR accepted the bare label on the premise that ingest never wrote membership under it, while `materialise_domain_roots` did, ranking every domain root below its own members. The amendment records the observed census: 9,796 subclass edges plus 6,400 spokes ranked together on the pre-fix binary (`docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:196`). A same-day follow-up (`8a501fbbc`, recorded at `docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:288-340`) made the roots themselves stable: `materialise_domain_roots` now applies the pure `plan_domain_roots` reconcile (`src/services/github_sync_service.rs:2282`), so each domain has one derived root id in every process and stale counter-minted roots and their `hierarchical` spokes are purged on the next sync.

**Drift (ADR-2035 vs code):** the record's Context still says the ranker uses `is_directed_hierarchy_relation` (`docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:28-29`), and its Consequences still describe the accept as coupled to ingest labelling (`docs/adr/ADR-2035-dag-rank-accepts-hierarchical-label.md:61-62`). That function was deleted in 8bdece469 and replaced by `hierarchy_pairs` (`src/actors/gpu/force_compute_actor.rs:419`). The Decision section (`:37-43`) is current; the surrounding prose is not.

**Open:** the fix moves a dependency onto provenance. A producer that writes a `hierarchical` subclass edge without `rdfs:subClassOf` in `owl_property_iri`, or a store path that drops `vc:owlProperty` (`src/adapters/oxigraph_graph_repository.rs:551`), would silently stop ranking. The ADR names this as its review trigger; nothing at boot detects it.
