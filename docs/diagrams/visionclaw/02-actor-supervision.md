---
id: VC-02
title: Actor supervision tree, GraphServiceSupervisor routing and peer actor surfaces
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2005, ADR-2007, ADR-2045, ADR-2141, ADR-2142]
sources:
  - ../project/docs/BASELINE-architecture.md
  - ../project/src/app_state.rs
  - ../project/src/main.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/event_coordination.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/src/actors/ontology_actor.rs
  - ../project/src/actors/semantic_processor_actor.rs
  - ../project/src/actors/optimized_settings_actor.rs
  - ../project/src/actors/elevation_actor.rs
  - ../project/src/actors/elevation_voice.rs
  - ../project/src/actors/decision_elevation_actor.rs
  - ../project/src/actors/voice_interface_actor.rs
  - ../project/src/actors/voice_commands.rs
  - ../project/src/actors/multi_mcp_visualization_actor.rs
  - ../project/src/actors/agent_monitor_actor.rs
  - ../project/src/actors/agent_beam_actor.rs
  - ../project/crates/visionclaw-actors/src/supervisor.rs
  - ../project/tests/orchestration_improvements_test.rs
  - ../project/src/actors/mod.rs
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

The graph side of the server is a tree of actix actors under `GraphServiceSupervisor`, the one live supervision path since ADR-2045 removed the generic supervisor (VC-02.4 to VC-02.7). It starts four children in a fixed order, routes every graph, physics and client message to the right one (VC-02.3) and restarts a failed child with backoff or escalates. The position path is GPU to `ForceComputeActor` to the supervisor to `ClientCoordinatorActor`, never through the physics orchestrator (ADR-2141, VC-02.12). A node interaction carries its holder, so only the connection that took a node can release it (VC-02.12). At 809a859aa `ProtectedSettingsActor` is gone from the tree (VC-02.1).

## For the business

These are the server's internal workers for the graph, physics and connected viewers, and the rules for restarting one that fails. A failed worker is restarted a bounded number of times and then escalated, rather than being retried forever. Moving a node now stays smooth when two people grab the same node: one person letting go no longer stops the layout while the other is still dragging.

## VC-02.1 Supervision-tree topology — AppState::new boot order
```mermaid
flowchart TB
    APP["AppState::new<br/>src/app_state.rs:368"]
    CC["ClientCoordinatorActor<br/>src/app_state.rs:662<br/>Addr~ClientCoordinatorActor~"]
    AB["AgentBeamActor<br/>src/app_state.rs:677<br/>Addr~AgentBeamActor~ kept and handed to GSS<br/>by SetAgentBeamAddr (src/app_state.rs:817)"]
    MD["MetadataActor<br/>src/app_state.rs:797<br/>Addr~MetadataActor~"]
    GSS["GraphServiceSupervisor<br/>src/app_state.rs:802<br/>Addr~GraphServiceSupervisor~"]
    DEMO["AgentDemoDirector (ADR-2142)<br/>src/app_state.rs:837<br/>Addr~AgentDemoDirector~ — stored as AppState.agent_demo_addr"]
    REBIND["SetClientCoordinatorAddr<br/>src/app_state.rs:812"]
    GPU["GPUManagerActor boundary<br/>src/app_state.rs:964<br/>Note: GPU internals see VC-10"]
    ANA["analytics actors — supervised, not started here<br/>src/app_state.rs:1284-1290<br/>ADR-2053: GraphAnalyticsSupervisor owns ShortestPathActor and<br/>ConnectedComponentsActor, AppState keeps an empty GraphSubsystem, see VC-10"]
    SET["OptimizedSettingsActor<br/>src/app_state.rs:1106-1115<br/>Addr~OptimizedSettingsActor~"]
    AM["AgentMonitorActor<br/>src/app_state.rs:1138<br/>Addr~AgentMonitorActor~"]
    WS["WorkspaceActor<br/>src/app_state.rs:1150<br/>Addr~WorkspaceActor~"]
    ONT["OntologyActor<br/>src/app_state.rs:1163<br/>Option~Addr~OntologyActor~~"]
    TO["TaskOrchestratorActor<br/>src/app_state.rs:1217<br/>Addr~TaskOrchestratorActor~"]
    EL["ElevationActor<br/>src/app_state.rs:1225<br/>anon start — no Addr retained"]
    VI["VoiceInterfaceActor<br/>src/app_state.rs:1248<br/>anon start — no Addr retained"]
    DE["DecisionElevationActor<br/>src/main.rs:576<br/>started OUTSIDE AppState::new"]

    APP --> CC
    CC --> AB
    CC -.->|"rebind"| REBIND
    AB --> MD
    MD --> GSS
    GSS -.->|"do_send"| REBIND
    GSS -.->|"SetAgentBeamAddr — roster owner, ADR-2142 amendment"| AB
    GSS --> DEMO
    DEMO --> GPU
    GPU --> ANA
    ANA --> SET
    SET --> AM
    AM --> WS
    WS --> ONT
    ONT --> TO
    TO --> EL
    EL --> VI
    VI -.->|"same process, later in main()"| DE

    N1["INVARIANT (BASELINE Invariants) — the live ClientCoordinatorActor clients register<br/>with must be the CC instance above, not GraphServiceSupervisor's own child.<br/>REBIND at src/app_state.rs:812 makes GSS forward broadcasts through CC's non-empty registry."]
    REBIND --- N1
    N2["DOC-DRIFT: EventCoordinator::initialize_event_coordinator (src/actors/event_coordination.rs:112)<br/>is re-exported at src/actors/mod.rs:87 but has no caller in app_state.rs or main.rs —<br/>the ADR-2007 event-bus coordination path is dead code; only direct Actix do_send/send is live."]
    APP --- N2
    N3["OntologyActor wiring (src/app_state.rs:1154-1163): set_gpu_manager_addr (src/app_state.rs:1157, only if<br/>gpu_manager_addr Some) then set_client_manager_addr (:1161, unconditional) BEFORE start() (:1163)."]
    ONT --- N3
    N4["ProtectedSettingsActor, its AppState field and get_api_keys / get_nostr_user are deleted (7a0674e79).<br/>API keys live in NostrService only, see VC-05.3 and VC-06.6"]
    AM --- N4
```

## VC-02.2 GraphServiceSupervisor::initialize_actors — child start order and wiring
```mermaid
sequenceDiagram
    autonumber
    participant S as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:645
    participant CC as ClientCoordinatorActor<br/>src/actors/graph_service_supervisor.rs:775
    participant PO as PhysicsOrchestratorActor<br/>src/actors/graph_service_supervisor.rs:763
    participant SP as SemanticProcessorActor<br/>src/actors/graph_service_supervisor.rs:771
    participant GS as GraphStateActor<br/>src/actors/graph_service_supervisor.rs:753

    Note over S: initialize_actors src/actors/graph_service_supervisor.rs:645 — 4 ActorInfo rows inserted first (:648-702)
    S->>S: start_actor(ClientCoordinator) (src/actors/graph_service_supervisor.rs:704, the call, start_actor itself :745)
    S->>CC: ClientCoordinatorActor::new().start() (:775)
    S->>S: wire_physics_and_client() pushes node-type arrays to the coordinator and wire_force_compute_subscribers() subscribes physics to ForceComputeActor replacement once gpu_manager is known — both fire on ClientCoordinator or PhysicsOrchestrator start (:780-786)
    S->>S: start_actor(PhysicsOrchestrator) (:705)
    S->>PO: PhysicsOrchestratorActor::new(SimulationParams::default(), None, None).start() (:763)
    S->>S: wire_physics_and_client() and wire_force_compute_subscribers() again
    S->>S: start_actor(SemanticProcessor) (:706)
    S->>SP: SemanticProcessorActor::new(config).start() (:771)
    S->>S: start_actor(GraphState) (:707)
    alt kg_repo present
        S->>GS: GraphStateActor::new(kg_repo.clone()).start() (:753)
    else kg_repo absent
        S->>S: error! "Cannot start GraphStateActor without kg_repo" (:757)
    end
    S->>S: ctx.run_interval(health_check_interval=30s, perform_health_check) (:710-712)
    S->>S: ctx.run_interval(15s, emit ActorHeartbeat per live child) (:718-738)
    S->>S: supervision_stats.actors_supervised = 4 (:741)
    Note over S: SetClientCoordinatorAddr src/actors/graph_service_supervisor.rs:1574-1582 —<br/>AppState sends this AFTER initialize_actors to overwrite the CC child above<br/>with the application's canonical ClientCoordinatorActor (see VC-02.1), then<br/>wire_physics_and_client and wire_force_compute_subscribers run again (:1579-1580)<br/>SetAgentBeamAddr (src/actors/graph_service_supervisor.rs:2020-2025) then fills self.agent_beam,<br/>the roster owner since the ADR-2142 amendment of 2026-11-01
```

## VC-02.3 GraphServiceSupervisor message routing table
```mermaid
flowchart LR
    subgraph SUP["GraphServiceSupervisor handlers — grep impl Handler< src/actors/graph_service_supervisor.rs"]
        H1["SupervisorMessage graph_service_supervisor.rs:1436"]
        H2["ActorHeartbeat graph_service_supervisor.rs:1444"]
        H3["GetSupervisorStatus graph_service_supervisor.rs:1459"]
        H4["RestartActor graph_service_supervisor.rs:1467"]
        H5["RestartAllActors graph_service_supervisor.rs:1476"]
        H6["RESOLVED ADR-2045 (2026-09-05)<br/>SetParentSupervisor handler removed<br/>with src/actors/supervisor.rs, see VC-02.7"]
        H7["msgs::GetGraphData graph_service_supervisor.rs:1490"]
        H8["NotifyGraphUpdated graph_service_supervisor.rs:1525"]
        H9["SetClientCoordinatorAddr graph_service_supervisor.rs:1574"]
        H10["msgs::ReloadGraphFromDatabase graph_service_supervisor.rs:1585"]
        H11["msgs::ComputeShortestPaths graph_service_supervisor.rs:1689"]
        H12["msgs::UpdateGraphData graph_service_supervisor.rs:1715"]
        H13["msgs::AddNodesFromMetadata graph_service_supervisor.rs:1745"]
        H14["msgs::StartSimulation graph_service_supervisor.rs:1771"]
        H15["msgs::SimulationStep graph_service_supervisor.rs:1788"]
        H16["msgs::GetBotsGraphData graph_service_supervisor.rs:1815"]
        H17["msgs::UpdateSimulationParams graph_service_supervisor.rs:1840"]
        H18["msgs::ForceResumePhysics graph_service_supervisor.rs:1868"]
        H19["msgs::InitializeGPUConnection graph_service_supervisor.rs:1897"]
        H20["msgs::SetAppGpuComputeAddr graph_service_supervisor.rs:2000"]
        H21["msgs::UpdateBotsGraph graph_service_supervisor.rs:2031"]
        H29["msgs::UpdateChainPayments graph_service_supervisor.rs:2043"]
        H30["SetAgentBeamAddr graph_service_supervisor.rs:2020"]
        H22["msgs::UpdateNodePositions graph_service_supervisor.rs:2058"]
        H23["msgs::NodeInteractionMessage graph_service_supervisor.rs:2067"]
        H24["msgs::GetGraphStateActor graph_service_supervisor.rs:2135"]
        H25["msgs::GetPhysicsOrchestratorActor graph_service_supervisor.rs:2144"]
        H26["msgs::GetNodeTypeArrays graph_service_supervisor.rs:2159"]
        H27["msgs::GetNodeIdMapping graph_service_supervisor.rs:2181"]
        H28["msgs::AddEdge graph_service_supervisor.rs:2203"]
    end
    GS["self.graph_state (GraphStateActor)"]
    PH["self.physics (PhysicsOrchestratorActor)"]
    CL["self.client (ClientCoordinatorActor)"]
    GM["self.gpu_manager (GPUManagerActor) — boundary, see VC-10"]
    SELF["self (in-place state)"]
    AB["self.agent_beam (AgentBeamActor) — real-agent roster owner,<br/>ADR-2142 amendment 2026-11-01, see VC-02.20"]

    H7 --> GS
    H10 --> GS
    H11 --> GS
    H12 --> GS
    H13 --> GS
    H16 --> AB
    H21 --> AB
    H29 --> AB
    H30 -->|"self.agent_beam = msg.addr"| SELF
    H26 --> GS
    H27 --> GS
    H28 --> GS
    H22 -->|"publish_positions"| GS
    H22 -->|"one PublishPositions, ADR-2141"| CL
    H14 --> PH
    H15 --> PH
    H17 --> PH
    H18 --> PH
    H23 --> PH
    H25 --> PH
    H19 -->|"query GetForceComputeActor, InitializeGPU"| GM
    H19 -->|"StoreGPUComputeAddress"| PH
    H1 --> SELF
    H2 --> SELF
    H3 --> SELF
    H4 --> SELF
    H5 --> SELF
    H8 --> SELF
    H9 -->|"self.client = msg.addr, rewire"| CL
    H20 --> SELF
    H24 --> SELF

    N1["H12/H13/H28 also call notify_graph_updated (debounced graphUpdated broadcast) before forwarding<br/>H24/H26/H27 default-value fallback when the target child is None (never propagate an error to caller)<br/>H22 UpdateNodePositions no longer reaches PhysicsOrchestratorActor: publish_positions (graph_service_supervisor.rs:612) sends PublishPositions to the client coordinator and forwards the original UpdateNodePositions to GraphStateActor<br/>which mutates its Arc~GraphData~ in-place (graph_state_actor.rs:807-808), so a later GetGraphData poll (H7) returns the GPU-computed layout<br/>notify_graph_updated (called by H12/H13/H28) now also pushes node-type arrays to the coordinator before its debounce (graph_service_supervisor.rs:555-560, push_node_type_arrays :1542)"]
    H12 --- N1
```

## VC-02.4 GraphServiceSupervisor restart and backoff lifecycle — the one live supervision path
```mermaid
stateDiagram-v2
    [*] --> Unknown
    Unknown --> Healthy: start_actor completes, health Healthy and last_heartbeat set (src/actors/graph_service_supervisor.rs:788-792)
    Healthy --> Degraded: perform_health_check finds a heartbeat older than 60s (src/actors/graph_service_supervisor.rs:979-983)
    Degraded --> Healthy: the 15s heartbeat emitter refreshes last_heartbeat (:718-738)
    Healthy --> Restarting: RestartActor handler calls restart_actor (:1467-1473, :795)
    Degraded --> Restarting: RestartActor handler calls restart_actor (:1467-1473, :795)
    Restarting --> Healthy: run_later(backoff) fires start_actor then replay_buffered_messages (:817-820)
    Restarting --> Escalated: restart_count exceeds restart_policy.max_restarts (:803-810)
    Escalated --> RestartingAll: strategy OneForAll, restart_all_actors clears all four Addr slots (:844-846, :871-882)
    Escalated --> Stopped: strategy Escalate, top of the tree, ctx.stop() (:848-860)
    Escalated --> Failed: any other strategy, ActorInfo health set Failed and the child stays down (:862-867)
    RestartingAll --> Healthy: each start_actor re-marks its ActorInfo Healthy (:788-792)
    note right of Restarting
        calculate_backoff src/actors/graph_service_supervisor.rs:825-838
        Exponential initial 1s doubled per restart_count, capped at 60s
        RestartPolicy::default :485-497 — max_restarts 5, within_time_period 300s
    end note
    note right of Stopped
        RESOLVED ADR-2045 (2026-09-05) — this section used to draw the generic
        SupervisorActor state machine from src/actors/supervisor.rs. Commit 346fff7af
        deleted that file with src/actors/lifecycle.rs. GraphServiceSupervisor is the
        only supervision path left in src/, and Escalate is the top of the tree.
    end note
```

## VC-02.5 GraphServiceSupervisor restart sequence — RestartActor to replay or escalate
```mermaid
sequenceDiagram
    autonumber
    participant OP as RestartActor sender<br/>src/actors/graph_service_supervisor.rs:1424-1428
    participant S as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:795
    participant AI as ActorInfo row<br/>src/actors/graph_service_supervisor.rs:321-331
    participant CH as supervised child<br/>src/actors/graph_service_supervisor.rs:745

    OP->>S: RestartActor { actor_type } (:1467-1473)
    S->>AI: health = Restarting, restart_count += 1, last_restart = now (:798-801)
    alt restart_count over restart_policy.max_restarts (:803-807)
        S->>S: escalate_failure(actor_type, ctx) (:808, :840)
        Note over S: OneForAll restarts every child (:844-846) — Escalate stops the supervisor (:848-860) — any other strategy marks the child Failed (:862-867)
    else inside the restart budget
        S->>S: backoff = calculate_backoff(actor_type) (:813, :825-838)
        S->>S: ctx.run_later(backoff) schedules the retry (:817)
        S->>CH: start_actor(actor_type) constructs and starts a fresh child (:818, :745-793)
        S->>AI: replay_buffered_messages drains message_buffer into the new Addr (:819, :885-969)
        S->>S: supervision_stats.total_restarts += 1 (:822)
    end
    Note over S,CH: the four supervised children are GraphState, PhysicsOrchestrator,<br/>SemanticProcessor and ClientCoordinator (ActorType :334-339) — agent-beam and<br/>presence are unsupervised peers started by AppState, see VC-02 and VC-02.20
    Note over OP,S: RESOLVED ADR-2045 (2026-09-05) — the generic SupervisorActor sequence<br/>drawn here before lived in src/actors/supervisor.rs, deleted by 346fff7af.<br/>The surviving copy is crates/visionclaw-actors/src/supervisor.rs:122, reached<br/>only from tests/orchestration_improvements_test.rs:282-284.
```

## VC-02.6 Drain and shutdown — GraphServiceSupervisor stop and the test-only crate drain
```mermaid
sequenceDiagram
    autonumber
    participant S as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:840
    participant Sys as actix runtime
    participant CH as supervised children<br/>src/actors/graph_service_supervisor.rs:745
    participant T as test caller<br/>tests/orchestration_improvements_test.rs:282-284
    participant SA as SupervisorActor crate copy<br/>crates/visionclaw-actors/src/supervisor.rs:122

    S->>S: escalate_failure with strategy Escalate — top of the tree (:848-860)
    S->>Sys: ctx.stop() (:860)
    Sys->>S: Actor stopping then stopped, logs GraphServiceSupervisor stopped (src/actors/graph_service_supervisor.rs:1365-1367)
    Note over S,CH: children hold no back-reference — each stops when its last Addr clone drops
    Note over S: this ctx.stop() is the only explicit stop in graph_service_supervisor.rs —<br/>there is no drain timer and no registration gate on the live path
    T->>SA: InitiateGracefulShutdown { timeout_secs } (crates/visionclaw-actors/src/supervisor.rs:115-120, sent at tests/orchestration_improvements_test.rs:302)
    SA->>SA: draining = true — RegisterActor now rejected (crates/visionclaw-actors/src/supervisor.rs:211, :225-233)
    loop ctx.run_later(timeout_secs) (crates/visionclaw-actors/src/supervisor.rs:213)
        SA->>SA: drain timeout elapsed, ctx.stop() (crates/visionclaw-actors/src/supervisor.rs:214-215)
    end
    Note over T,SA: RESOLVED ADR-2045 (2026-09-05) — the src/actors/supervisor.rs drain and the<br/>ActorLifecycleManager::shutdown_with_timeout path formerly drawn here are both<br/>deleted. The crate copy survives with no non-test sender of InitiateGracefulShutdown.
```

## VC-02.7 RESOLVED ADR-2045 — the removed supervision machinery and its survivor
```mermaid
flowchart TB
    ADR["ADR-2045 — remove the dead supervision machinery<br/>decision accepted, implementation complete, landed 346fff7af"]
    LIFE["DELETED src/actors/lifecycle.rs<br/>ActorLifecycleManager, static ACTOR_SYSTEM,<br/>initialize_actor_system, shutdown_actor_system"]
    SUP["DELETED src/actors/supervisor.rs<br/>generic SupervisorActor, ActorFactory,<br/>SupervisedActorTrait, ActorFailed"]
    COUP["DELETED coupling inside graph_service_supervisor.rs<br/>parent_supervisor field and the SetParentSupervisor message"]
    MOD["src/actors/mod.rs:88-107<br/>a comment block records both removals<br/>where the re-exports used to be"]
    ESC["src/actors/graph_service_supervisor.rs:848-860<br/>Escalate is now the top of the tree — log and ctx.stop()"]
    CRATE["crates/visionclaw-actors/src/supervisor.rs:122<br/>the only surviving SupervisorActor definition"]
    TEST["tests/orchestration_improvements_test.rs:282-284<br/>its only consumer, repointed at the crate copy"]
    LIVE["GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:645<br/>the one live supervision path, see VC-02.4 and VC-02.5"]

    ADR --> LIFE
    ADR --> SUP
    ADR --> COUP
    LIFE --> MOD
    SUP --> MOD
    SUP --> CRATE
    CRATE --> TEST
    COUP --> ESC
    MOD --> LIVE
    ESC --> LIVE

    N1["RESOLVED ADR-2045 (2026-09-05) — commit 346fff7af deleted both files and trimmed<br/>the actor set to graph-service, agent-beam and presence. No code in src/ constructs<br/>a generic SupervisorActor, and nothing outside tests sends InitiateGracefulShutdown."]
    LIVE --- N1
```

## VC-02.9 Message catalogue — client_messages, graph_messages, broadcast_messages
```mermaid
classDiagram
    class ClientCoordinatorActor {
      <<actor>>
      src/actors/client_coordinator_actor.rs
    }
    class RegisterClient {
      +ClientRecipients recipients
      result() Result~usize,String~
    }
    class ClientRecipients {
      +Recipient~SendToClientBinary~ binary
      +Recipient~SendToClientText~ text
      +Recipient~SendInitialGraphLoad~ initial_load
    }
    class UnregisterClient {
      +usize client_id
    }
    class BroadcastMessage {
      +String message
    }
    class GraphServiceSupervisor {
      <<actor>>
    }
    class GetGraphData {
      result() Result~Arc,String~
    }
    class UpdateGraphData {
      +Arc~ServiceGraphData~ graph_data
      result() Result~(),String~
    }
    class UpdateNodePositions {
      +List~u32,BinaryNodeData~ positions
      +Option~MessageId~ correlation_id
      result() Result~(),String~
    }
    class AddNodesFromMetadata {
      +MetadataStore metadata
      result() Result~(),String~
    }
    class AddEdge {
      +Edge edge
      result() Result~(),String~
    }
    class NodeInteractionMessage {
      +u32 node_id
      +u64 holder
      +NodeInteractionType interaction_type
      +Option~List~f32~~ position
      result() Result~(),VisionClawError~
    }
    RegisterClient *-- ClientRecipients
    ClientCoordinatorActor ..> RegisterClient
    ClientCoordinatorActor ..> UnregisterClient
    ClientCoordinatorActor ..> BroadcastMessage
    GraphServiceSupervisor ..> GetGraphData
    GraphServiceSupervisor ..> UpdateGraphData
    GraphServiceSupervisor ..> UpdateNodePositions
    GraphServiceSupervisor ..> AddNodesFromMetadata
    GraphServiceSupervisor ..> AddEdge
    GraphServiceSupervisor ..> NodeInteractionMessage
    note for GetGraphData "src/actors/messages/ — RegisterClient/UnregisterClient/BroadcastMessage\nowned by client_coordinator_actor.rs consumer; Result types confirmed from\nHandler impls at graph_service_supervisor.rs:1490,1715,2058,1745,2203,2067"
```

## VC-02.10 Message catalogue — supervision and heartbeat messages
```mermaid
classDiagram
    class GraphServiceSupervisor {
      <<actor>>
    }
    class SupervisorMessage {
      <<enum>>
      UpdateGraphData
      ReloadGraphFromDatabase
      StartSimulation
      StopSimulation
      SimulationStep
      UpdateSimulationParams
      UpdateNodePositions
      BroadcastMessage
      result() Result~(),VisionClawError~
    }
    class ActorHeartbeat {
      +ActorType actor_type
      +Instant timestamp
      +ActorHealth health
      +Option~ActorStats~ stats
    }
    class GetSupervisorStatus {
      result() SupervisorStatus
    }
    class RestartActor {
      +ActorType actor_type
      result() Result~(),VisionClawError~
    }
    class RestartAllActors {
      result() Result~(),VisionClawError~
    }
    class SetParentSupervisor {
      +Addr~SupervisorActor~ parent
    }
    class SetClientCoordinatorAddr {
      +Addr~ClientCoordinatorActor~ addr
    }
    class NotifyGraphUpdated {
      +str reason
    }
    class ActorFailed {
      +String actor_name
      +VisionClawError error
    }
    GraphServiceSupervisor ..> SupervisorMessage
    GraphServiceSupervisor ..> ActorHeartbeat
    GraphServiceSupervisor ..> GetSupervisorStatus
    GraphServiceSupervisor ..> RestartActor
    GraphServiceSupervisor ..> RestartAllActors
    GraphServiceSupervisor ..> SetClientCoordinatorAddr
    GraphServiceSupervisor ..> NotifyGraphUpdated
    note for SetClientCoordinatorAddr "src/actors/graph_service_supervisor.rs:1574-1582 —\nfield and handler both confirmed by direct read.\nRESOLVED ADR-2045 (2026-09-05) — SetParentSupervisor and ActorFailed\nwere removed from this actor with src/actors/supervisor.rs, see VC-02.7"
```

## VC-02.12 PhysicsOrchestratorActor — handled messages and the position path (ADR-2141)
```mermaid
sequenceDiagram
    autonumber
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:763
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs
    participant GPU as GPU boundary<br/>see VC-10
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:1472

    GSS->>PO: StartSimulation (src/actors/graph_service_supervisor.rs:1771)
    GSS->>PO: SimulationStep (:1788)
    GSS->>PO: UpdateSimulationParams (:1840, now boxed Box~UpdateSimulationParams~ in SupervisorMessage and BufferedMessage)
    GSS->>PO: ForceResumePhysics (:1868)
    GSS->>PO: NodeInteractionMessage (:2067), now carrying the holder connection id
    PO->>PO: track_hold_owner, Dragged records node to holder, Released clears only the same holder's hold (physics_orchestrator_actor.rs:1183-1194, :1201)
    Note over PO: any hold zeroes the step interval and blocks the equilibrium auto-pause (physics_orchestrator_actor.rs:309-314, :673-676),<br/>so a stale release from a previous holder can no longer put the orchestrator to sleep mid-drag (drag holds see VC-52)
    GSS->>PO: StoreGPUComputeAddress (:1897 handler, do_send at :1931, from the GPUManagerActor GetForceComputeActor reply)
    GSS->>PO: ForceComputeActorReplaced (subscribed by wire_force_compute_subscribers :598, so a restarted ForceComputeActor gets the graph again)
    GPU->>GSS: UpdateNodePositions from ForceComputeActor (handler :2058)
    GSS->>CC: PublishPositions via publish_positions (:612)
    CC->>CC: update_position_cache then pump, each client streamed its next chunk until acknowledged (client_coordinator_actor.rs:1472-1479)
    Note over PO,CC: INVARIANT (ADR-2141) — PhysicsOrchestratorActor has no UpdateNodePositions handler and<br/>no client_coordinator_addr: the position path is GPU to ForceComputeActor to GraphServiceSupervisor<br/>to ClientCoordinatorActor to WebSocket (physics_orchestrator_actor.rs:578-581)
    Note over GSS,CC: DRIFT: docs/BASELINE-architecture.md:436-438 still routes the position broadcast<br/>through PhysicsOrchestratorActor, and its :170-172 reads the same way
```

**Drift (BASELINE-architecture.md vs code):** `docs/BASELINE-architecture.md:436-438` and `:170-172` name `PhysicsOrchestratorActor` as the hop between `ForceComputeActor` and `ClientCoordinatorActor`; `src/actors/graph_service_supervisor.rs:2058-2064` hands `UpdateNodePositions` to `publish_positions` (`:612`), which sends `PublishPositions` straight to the coordinator, and the orchestrator has no handler for it (`src/actors/physics_orchestrator_actor.rs:578-581`).

## VC-02.15 SemanticProcessorActor and OptimizedSettingsActor — message surfaces
```mermaid
sequenceDiagram
    autonumber
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:767-773
    participant SP as SemanticProcessorActor<br/>src/actors/semantic_processor_actor.rs
    participant APP as AppState::new<br/>src/app_state.rs:1106
    participant SET as OptimizedSettingsActor<br/>src/actors/optimized_settings_actor.rs
    participant REDIS as REDIS_URL<br/>src/actors/optimized_settings_actor.rs:138

    GSS->>SP: SemanticProcessorActor::new(SemanticProcessorConfig::default()).start() (:771)
    APP->>SET: OptimizedSettingsActor::with_actors(sqlite_settings_repository, Some(graph_service_addr), None) (:1106-1110)
    SET->>SET: settings_actor.start() (src/app_state.rs:1115)
    alt REDIS_URL set (src/actors/optimized_settings_actor.rs:138)
        SET->>REDIS: connect for distributed settings cache
    else REDIS_URL unset
        SET->>SET: local-only settings path
    end
    Note over SET: settings internals are VC-06 — this file shows only the actor's message surface
```

## VC-02.18 ElevationActor (+ elevation_voice.rs) and DecisionElevationActor
```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:1225-1243
    participant EL as ElevationActor<br/>src/actors/elevation_actor.rs:191
    participant EV as elevation_voice<br/>src/actors/elevation_voice.rs
    participant MAIN as main<br/>src/main.rs:576
    participant DE as DecisionElevationActor<br/>src/actors/decision_elevation_actor.rs:175

    alt ELEVATION_ACTOR_ENABLED gate passes (src/actors/elevation_actor.rs:191)
        APP->>EL: ElevationActor::new(graph_adapter, sqlite_enrichment_repository, speech_service, Some(ontology_repository)).start() (:1225)
        Note over EL,EV: voice-guided path when local speech stack (Whisper STT, PocketTts TTS) is up (app_state.rs:1220-1222) — elevation_voice.rs
    else gate closed
        APP->>APP: log "ElevationActor disabled" (src/app_state.rs:1241)
    end
    Note over MAIN,DE: DecisionElevationActor is started in main() src/main.rs:576, NOT in AppState::new —<br/>a second, separately-gated ACSP actor family alongside ElevationActor
    alt DECISION_ELEVATION_ENABLED gate passes (src/actors/decision_elevation_actor.rs:175)
        MAIN->>DE: DecisionElevationActor::new() then actix::Actor::start(actor) (src/main.rs:576-578)
        MAIN->>MAIN: wrap in ActorElevationSink (src/main.rs:581), fed to DecisionService.with_elevation_sink (src/main.rs:598)
    else gate closed
        MAIN->>MAIN: log "DecisionElevationActor disabled" (src/main.rs:585)
    end
```

## VC-02.19 VoiceInterfaceActor (+ voice_commands.rs) and MultiMcpVisualizationActor
```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:1248-1259
    participant VI as VoiceInterfaceActor<br/>src/actors/voice_interface_actor.rs
    participant VC as voice_commands<br/>src/actors/voice_commands.rs
    participant MMV as MultiMcpVisualizationActor<br/>src/actors/multi_mcp_visualization_actor.rs

    alt speech_service Some
        APP->>VI: VoiceInterfaceActor::new(task_orchestrator_addr.clone(), speech_service.clone()).start() (:1248)
        VI->>VC: dispatch parsed spoken command to settings-assistant path
    else speech_service None
        APP->>APP: log "VoiceInterfaceActor disabled (no speech service)" (:1258)
    end
    Note over MMV: MultiMcpVisualizationActor has no start() call found in src/app_state.rs or src/main.rs —<br/>coverage gap, reported not diagrammed further (see report)
```

## VC-02.20 AgentMonitorActor and AgentBeamActor — message surfaces
```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:1123-1138
    participant AM as AgentMonitorActor<br/>src/actors/agent_monitor_actor.rs:491
    participant CFC as ClaudeFlowClient
    participant AB as AgentBeamActor<br/>src/actors/agent_beam_actor.rs:203
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:2031
    participant CC as ClientCoordinatorActor
    participant HUB as agent-events hub

    APP->>CFC: ClaudeFlowClient::new(mcp_host, mcp_port) (:1136)
    APP->>AM: AgentMonitorActor::new(claude_flow_client, graph_service_addr.clone()).start() (:1138)
    alt MOCK_AGENTS set (src/actors/agent_monitor_actor.rs:634)
        AM->>AM: synthesize mock agent roster, skip live MCP poll
    else unset
        AM->>CFC: poll live MCP agent roster
    end
    APP->>AB: let agent_beam_addr = AgentBeamActor::new(client_manager_addr.clone()).start() (src/app_state.rs:677)
    APP->>GSS: SetAgentBeamAddr { addr: agent_beam_addr } (src/app_state.rs:817)
    AM->>GSS: UpdateBotsGraph { source: ManagementApi, agents, targets } (src/actors/agent_monitor_actor.rs:714-718)
    GSS->>AB: forward UpdateBotsGraph, also GetBotsGraphData (:1815) and UpdateChainPayments (:2043)
    AB->>AB: RosterFeed.apply turns the roster into presence and state events (src/actors/agent_beam_actor.rs:250-268)
    HUB->>AB: process-global agent-event stream, each action joined to its roster agent by taskId (src/actors/agent_beam_actor.rs:186-197)
    AB->>CC: encoded 0x23 frames via BroadcastAgentActionFrame, plus a 1s roster keep-alive tick (src/actors/agent_beam_actor.rs:307-313)
    CC->>CC: broadcast_agent_actions, one frame per identity without actions whose target that caller may not see,<br/>nothing sent before a classification arrives (src/actors/client_coordinator_actor.rs:518-554, handler :1386-1400)
    Note over AB,GSS: Swarm agents are no longer graph nodes: GraphStateActor holds no bots graph<br/>(src/actors/graph_state_actor.rs:980-982) — /bots/data is the AgentBeamActor roster view (src/actors/agent_beam_actor.rs:271-287)
```

**Drift (RESOLVED in this restamp, 2026-11-02):** at 8692b12 this file routed `GetBotsGraphData`, `UpdateBotsGraph` and `UpdateChainPayments` to `GraphStateActor` and drew `AgentBeamActor` with no retained address. At 86cdd7635 all three go to `AgentBeamActor` (`src/actors/graph_service_supervisor.rs:1815`, `:2031`, `:2043`), registered by `SetAgentBeamAddr` (`src/app_state.rs:817`), and `GraphStateActor` holds no bots graph (`src/actors/graph_state_actor.rs:980-982`).
