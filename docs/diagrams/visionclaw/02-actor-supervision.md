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
verified_commit: e74bcddc772d47a45b1971156a2bab540eaa027c
---

## VC-02.1 Supervision-tree topology — AppState::new boot order
```mermaid
flowchart TB
    APP["AppState::new<br/>src/app_state.rs:393"]
    CC["ClientCoordinatorActor<br/>src/app_state.rs:683<br/>Addr~ClientCoordinatorActor~"]
    AB["AgentBeamActor<br/>src/app_state.rs:696<br/>no Addr held — hub-subscription lifetime"]
    MD["MetadataActor<br/>src/app_state.rs:778<br/>Addr~MetadataActor~"]
    GSS["GraphServiceSupervisor<br/>src/app_state.rs:783<br/>Addr~GraphServiceSupervisor~"]
    DEMO["AgentDemoDirector (ADR-2142)<br/>src/app_state.rs:815<br/>Addr~AgentDemoDirector~ — stored as AppState.agent_demo_addr"]
    REBIND["SetClientCoordinatorAddr<br/>src/app_state.rs:793"]
    GPU["GPUManagerActor boundary<br/>src/app_state.rs:943<br/>Note: GPU internals see VC-10"]
    ANA["analytics actors — supervised, not started here<br/>src/app_state.rs:945-951<br/>ADR-2053 removed the standalone ShortestPathActor and<br/>ConnectedComponentsActor pair, see VC-10"]
    SET["OptimizedSettingsActor<br/>src/app_state.rs:1099-1108<br/>Addr~OptimizedSettingsActor~"]
    AM["AgentMonitorActor<br/>src/app_state.rs:1131<br/>Addr~AgentMonitorActor~"]
    PS["ProtectedSettingsActor<br/>src/app_state.rs:1144<br/>Addr~ProtectedSettingsActor~"]
    WS["WorkspaceActor<br/>src/app_state.rs:1147<br/>Addr~WorkspaceActor~"]
    ONT["OntologyActor<br/>src/app_state.rs:1160<br/>Option~Addr~OntologyActor~~"]
    TO["TaskOrchestratorActor<br/>src/app_state.rs:1214<br/>Addr~TaskOrchestratorActor~"]
    EL["ElevationActor<br/>src/app_state.rs:1222<br/>anon start — no Addr retained"]
    VI["VoiceInterfaceActor<br/>src/app_state.rs:1245<br/>anon start — no Addr retained"]
    DE["DecisionElevationActor<br/>src/main.rs:575<br/>started OUTSIDE AppState::new"]

    APP --> CC
    CC --> AB
    CC -.->|"rebind"| REBIND
    AB --> MD
    MD --> GSS
    GSS -.->|"do_send"| REBIND
    GSS --> DEMO
    DEMO --> GPU
    GPU --> ANA
    ANA --> SET
    SET --> AM
    AM --> PS
    PS --> WS
    WS --> ONT
    ONT --> TO
    TO --> EL
    EL --> VI
    VI -.->|"same process, later in main()"| DE

    N1["INVARIANT (BASELINE Invariants) — the live ClientCoordinatorActor clients register<br/>with must be the CC instance above, not GraphServiceSupervisor's own child.<br/>REBIND at src/app_state.rs:793 makes GSS forward broadcasts through CC's non-empty registry."]
    REBIND --- N1
    N2["DOC-DRIFT: EventCoordinator::initialize_event_coordinator (src/actors/event_coordination.rs:112)<br/>is re-exported at src/actors/mod.rs:88 but has no caller in app_state.rs or main.rs —<br/>the ADR-2007 event-bus coordination path is dead code; only direct Actix do_send/send is live."]
    APP --- N2
    N3["OntologyActor wiring (src/app_state.rs:1151-1160): set_gpu_manager_addr (src/app_state.rs:1154, only if<br/>gpu_manager_addr Some) then set_client_manager_addr (:1158, unconditional) BEFORE start() (:1160)."]
    ONT --- N3
```

## VC-02.2 GraphServiceSupervisor::initialize_actors — child start order and wiring
```mermaid
sequenceDiagram
    autonumber
    participant S as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:640
    participant CC as ClientCoordinatorActor<br/>src/actors/graph_service_supervisor.rs:770
    participant PO as PhysicsOrchestratorActor<br/>src/actors/graph_service_supervisor.rs:758
    participant SP as SemanticProcessorActor<br/>src/actors/graph_service_supervisor.rs:766
    participant GS as GraphStateActor<br/>src/actors/graph_service_supervisor.rs:748

    Note over S: initialize_actors src/actors/graph_service_supervisor.rs:640 — 4 ActorInfo rows inserted first (:643-697)
    S->>S: start_actor(ClientCoordinator) (src/actors/graph_service_supervisor.rs:699, the call, start_actor itself :740)
    S->>CC: ClientCoordinatorActor::new().start() (:770)
    S->>S: wire_physics_and_client() pushes node-type arrays to the coordinator and wire_force_compute_subscribers() subscribes physics to ForceComputeActor replacement once gpu_manager is known — both fire on ClientCoordinator or PhysicsOrchestrator start (:775-781)
    S->>S: start_actor(PhysicsOrchestrator) (:700)
    S->>PO: PhysicsOrchestratorActor::new(SimulationParams::default(), None, None).start() (:758)
    S->>S: wire_physics_and_client() and wire_force_compute_subscribers() again
    S->>S: start_actor(SemanticProcessor) (:701)
    S->>SP: SemanticProcessorActor::new(config).start() (:766)
    S->>S: start_actor(GraphState) (:702)
    alt kg_repo present
        S->>GS: GraphStateActor::new(kg_repo.clone()).start() (:748)
    else kg_repo absent
        S->>S: error! "Cannot start GraphStateActor without kg_repo" (:752)
    end
    S->>S: ctx.run_interval(health_check_interval=30s, perform_health_check) (:705-707)
    S->>S: ctx.run_interval(15s, emit ActorHeartbeat per live child) (:713-733)
    S->>S: supervision_stats.actors_supervised = 4 (:736)
    Note over S: SetClientCoordinatorAddr src/actors/graph_service_supervisor.rs:1569-1577 —<br/>AppState sends this AFTER initialize_actors to overwrite the CC child above<br/>with the application's canonical ClientCoordinatorActor (see VC-02.1), then<br/>wire_physics_and_client and wire_force_compute_subscribers run again (:1574-1575)
```

## VC-02.3 GraphServiceSupervisor message routing table
```mermaid
flowchart LR
    subgraph SUP["GraphServiceSupervisor handlers — grep impl Handler< src/actors/graph_service_supervisor.rs"]
        H1["SupervisorMessage graph_service_supervisor.rs:1431"]
        H2["ActorHeartbeat graph_service_supervisor.rs:1439"]
        H3["GetSupervisorStatus graph_service_supervisor.rs:1454"]
        H4["RestartActor graph_service_supervisor.rs:1462"]
        H5["RestartAllActors graph_service_supervisor.rs:1471"]
        H6["RESOLVED ADR-2045 (2026-09-05)<br/>SetParentSupervisor handler removed<br/>with src/actors/supervisor.rs, see VC-02.7"]
        H7["msgs::GetGraphData graph_service_supervisor.rs:1485"]
        H8["NotifyGraphUpdated graph_service_supervisor.rs:1520"]
        H9["SetClientCoordinatorAddr graph_service_supervisor.rs:1569"]
        H10["msgs::ReloadGraphFromDatabase graph_service_supervisor.rs:1580"]
        H11["msgs::ComputeShortestPaths graph_service_supervisor.rs:1684"]
        H12["msgs::UpdateGraphData graph_service_supervisor.rs:1710"]
        H13["msgs::AddNodesFromMetadata graph_service_supervisor.rs:1740"]
        H14["msgs::StartSimulation graph_service_supervisor.rs:1766"]
        H15["msgs::SimulationStep graph_service_supervisor.rs:1783"]
        H16["msgs::GetBotsGraphData graph_service_supervisor.rs:1810"]
        H17["msgs::UpdateSimulationParams graph_service_supervisor.rs:1843"]
        H18["msgs::ForceResumePhysics graph_service_supervisor.rs:1871"]
        H19["msgs::InitializeGPUConnection graph_service_supervisor.rs:1900"]
        H20["msgs::SetAppGpuComputeAddr graph_service_supervisor.rs:2003"]
        H21["msgs::UpdateBotsGraph graph_service_supervisor.rs:2017"]
    H29["msgs::UpdateChainPayments graph_service_supervisor.rs:2031"]
        H22["msgs::UpdateNodePositions graph_service_supervisor.rs:2047"]
        H23["msgs::NodeInteractionMessage graph_service_supervisor.rs:2056"]
        H24["msgs::GetGraphStateActor graph_service_supervisor.rs:2124"]
        H25["msgs::GetPhysicsOrchestratorActor graph_service_supervisor.rs:2133"]
        H26["msgs::GetNodeTypeArrays graph_service_supervisor.rs:2148"]
        H27["msgs::GetNodeIdMapping graph_service_supervisor.rs:2170"]
        H28["msgs::AddEdge graph_service_supervisor.rs:2192"]
    end
    GS["self.graph_state (GraphStateActor)"]
    PH["self.physics (PhysicsOrchestratorActor)"]
    CL["self.client (ClientCoordinatorActor)"]
    GM["self.gpu_manager (GPUManagerActor) — boundary, see VC-10"]
    SELF["self (in-place state)"]

    H7 --> GS
    H10 --> GS
    H11 --> GS
    H12 --> GS
    H13 --> GS
    H16 --> GS
    H21 --> GS
    H29 --> GS
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

    N1["H12/H13/H28 also call notify_graph_updated (debounced graphUpdated broadcast) before forwarding<br/>H24/H26/H27 default-value fallback when the target child is None (never propagate an error to caller)<br/>H22 UpdateNodePositions no longer reaches PhysicsOrchestratorActor: publish_positions (graph_service_supervisor.rs:607) sends PublishPositions to the client coordinator and the same message to GraphStateActor<br/>which mutates its Arc~GraphData~ in-place (graph_state_actor.rs:905-906), so a later GetGraphData poll (H7) returns the GPU-computed layout<br/>notify_graph_updated (called by H12/H13/H28) now also pushes node-type arrays to the coordinator before its debounce (graph_service_supervisor.rs:550-555, push_node_type_arrays :1537)"]
    H12 --- N1
```

## VC-02.4 GraphServiceSupervisor restart and backoff lifecycle — the one live supervision path
```mermaid
stateDiagram-v2
    [*] --> Unknown
    Unknown --> Healthy: start_actor completes, health Healthy and last_heartbeat set (src/actors/graph_service_supervisor.rs:783-787)
    Healthy --> Degraded: perform_health_check finds a heartbeat older than 60s (src/actors/graph_service_supervisor.rs:974-978)
    Degraded --> Healthy: the 15s heartbeat emitter refreshes last_heartbeat (:713-733)
    Healthy --> Restarting: RestartActor handler calls restart_actor (:1462-1468, :790)
    Degraded --> Restarting: RestartActor handler calls restart_actor (:1462-1468, :790)
    Restarting --> Healthy: run_later(backoff) fires start_actor then replay_buffered_messages (:812-815)
    Restarting --> Escalated: restart_count exceeds restart_policy.max_restarts (:798-805)
    Escalated --> RestartingAll: strategy OneForAll, restart_all_actors clears all four Addr slots (:839-841, :866-877)
    Escalated --> Stopped: strategy Escalate, top of the tree, ctx.stop() (:843-855)
    Escalated --> Failed: any other strategy, ActorInfo health set Failed and the child stays down (:857-862)
    RestartingAll --> Healthy: each start_actor re-marks its ActorInfo Healthy (:783-787)
    note right of Restarting
        calculate_backoff src/actors/graph_service_supervisor.rs:820-833
        Exponential initial 1s doubled per restart_count, capped at 60s
        RestartPolicy::default :481-493 — max_restarts 5, within_time_period 300s
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
    participant OP as RestartActor sender<br/>src/actors/graph_service_supervisor.rs:1419-1423
    participant S as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:790
    participant AI as ActorInfo row<br/>src/actors/graph_service_supervisor.rs:321-331
    participant CH as supervised child<br/>src/actors/graph_service_supervisor.rs:740

    OP->>S: RestartActor { actor_type } (:1462-1468)
    S->>AI: health = Restarting, restart_count += 1, last_restart = now (:793-796)
    alt restart_count over restart_policy.max_restarts (:798-802)
        S->>S: escalate_failure(actor_type, ctx) (:803, :835)
        Note over S: OneForAll restarts every child (:839-841) — Escalate stops the supervisor (:843-855) — any other strategy marks the child Failed (:857-862)
    else inside the restart budget
        S->>S: backoff = calculate_backoff(actor_type) (:808, :820-833)
        S->>S: ctx.run_later(backoff) schedules the retry (:812)
        S->>CH: start_actor(actor_type) constructs and starts a fresh child (:813, :740-788)
        S->>AI: replay_buffered_messages drains message_buffer into the new Addr (:814, :880-964)
        S->>S: supervision_stats.total_restarts += 1 (:817)
    end
    Note over S,CH: the four supervised children are GraphState, PhysicsOrchestrator,<br/>SemanticProcessor and ClientCoordinator (ActorType :334-339) — agent-beam and<br/>presence are unsupervised peers started by AppState, see VC-02 and VC-02.20
    Note over OP,S: RESOLVED ADR-2045 (2026-09-05) — the generic SupervisorActor sequence<br/>drawn here before lived in src/actors/supervisor.rs, deleted by 346fff7af.<br/>The surviving copy is crates/visionclaw-actors/src/supervisor.rs:122, reached<br/>only from tests/orchestration_improvements_test.rs:282-284.
```

## VC-02.6 Drain and shutdown — GraphServiceSupervisor stop and the test-only crate drain
```mermaid
sequenceDiagram
    autonumber
    participant S as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:835
    participant Sys as actix runtime
    participant CH as supervised children<br/>src/actors/graph_service_supervisor.rs:740
    participant T as test caller<br/>tests/orchestration_improvements_test.rs:282-284
    participant SA as SupervisorActor crate copy<br/>crates/visionclaw-actors/src/supervisor.rs:122

    S->>S: escalate_failure with strategy Escalate — top of the tree (:843-855)
    S->>Sys: ctx.stop() (:855)
    Sys->>S: Actor stopping then stopped, logs GraphServiceSupervisor stopped (src/actors/graph_service_supervisor.rs:1360-1362)
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
    MOD["src/actors/mod.rs:89-108<br/>a comment block records both removals<br/>where the re-exports used to be"]
    ESC["src/actors/graph_service_supervisor.rs:843-855<br/>Escalate is now the top of the tree — log and ctx.stop()"]
    CRATE["crates/visionclaw-actors/src/supervisor.rs:122<br/>the only surviving SupervisorActor definition"]
    TEST["tests/orchestration_improvements_test.rs:282-284<br/>its only consumer, repointed at the crate copy"]
    LIVE["GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:640<br/>the one live supervision path, see VC-02.4 and VC-02.5"]

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
    note for GetGraphData "src/actors/messages/ — RegisterClient/UnregisterClient/BroadcastMessage\nowned by client_coordinator_actor.rs consumer; Result types confirmed from\nHandler impls at graph_service_supervisor.rs:1485,1710,2047,1740,2192,2056"
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
    note for SetClientCoordinatorAddr "src/actors/graph_service_supervisor.rs:1569-1577 —\nfield and handler both confirmed by direct read.\nRESOLVED ADR-2045 (2026-09-05) — SetParentSupervisor and ActorFailed\nwere removed from this actor with src/actors/supervisor.rs, see VC-02.7"
```

## VC-02.12 PhysicsOrchestratorActor — handled messages and the position path (ADR-2141)
```mermaid
sequenceDiagram
    autonumber
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:758
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs
    participant GPU as GPU boundary<br/>see VC-10
    participant CC as ClientCoordinatorActor<br/>src/actors/client_coordinator_actor.rs:1315

    GSS->>PO: StartSimulation (src/actors/graph_service_supervisor.rs:1766)
    GSS->>PO: SimulationStep (:1783)
    GSS->>PO: UpdateSimulationParams (:1843, now boxed Box~UpdateSimulationParams~ in SupervisorMessage and BufferedMessage)
    GSS->>PO: ForceResumePhysics (:1871)
    GSS->>PO: NodeInteractionMessage (:2056)
    GSS->>PO: StoreGPUComputeAddress (:1900 handler, do_send at :1934, from the GPUManagerActor GetForceComputeActor reply)
    GSS->>PO: ForceComputeActorReplaced (subscribed by wire_force_compute_subscribers :593, so a restarted ForceComputeActor gets the graph again)
    GPU->>GSS: UpdateNodePositions from ForceComputeActor (handler :2047)
    GSS->>CC: PublishPositions via publish_positions (:607)
    CC->>CC: update_position_cache then pump, each client streamed its next chunk until acknowledged (client_coordinator_actor.rs:1315-1323)
    Note over PO,CC: INVARIANT (ADR-2141) — PhysicsOrchestratorActor has no UpdateNodePositions handler and<br/>no client_coordinator_addr: the position path is GPU to ForceComputeActor to GraphServiceSupervisor<br/>to ClientCoordinatorActor to WebSocket (physics_orchestrator_actor.rs:563-566)
    Note over GSS,CC: DRIFT: docs/BASELINE-architecture.md:325-327 still routes the position broadcast<br/>through PhysicsOrchestratorActor, and its :152-154 reads the same way
```

**Drift (BASELINE-architecture.md vs code):** `docs/BASELINE-architecture.md:325-327` and `:152-154` name `PhysicsOrchestratorActor` as the hop between `ForceComputeActor` and `ClientCoordinatorActor`; `src/actors/graph_service_supervisor.rs:2047-2053` sends `UpdateNodePositions` straight to the coordinator and the orchestrator has no handler for it (`src/actors/physics_orchestrator_actor.rs:563-566`).

## VC-02.15 SemanticProcessorActor and OptimizedSettingsActor — message surfaces
```mermaid
sequenceDiagram
    autonumber
    participant GSS as GraphServiceSupervisor<br/>src/actors/graph_service_supervisor.rs:762-768
    participant SP as SemanticProcessorActor<br/>src/actors/semantic_processor_actor.rs
    participant APP as AppState::new<br/>src/app_state.rs:1099
    participant SET as OptimizedSettingsActor<br/>src/actors/optimized_settings_actor.rs
    participant REDIS as REDIS_URL<br/>src/actors/optimized_settings_actor.rs:138

    GSS->>SP: SemanticProcessorActor::new(SemanticProcessorConfig::default()).start() (:766)
    APP->>SET: OptimizedSettingsActor::with_actors(sqlite_settings_repository, Some(graph_service_addr), None) (:1099-1103)
    SET->>SET: settings_actor.start() (src/app_state.rs:1108)
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
    participant APP as AppState::new<br/>src/app_state.rs:1222-1240
    participant EL as ElevationActor<br/>src/actors/elevation_actor.rs:191
    participant EV as elevation_voice<br/>src/actors/elevation_voice.rs
    participant MAIN as main<br/>src/main.rs:575
    participant DE as DecisionElevationActor<br/>src/actors/decision_elevation_actor.rs:175

    alt ELEVATION_ACTOR_ENABLED gate passes (src/actors/elevation_actor.rs:191)
        APP->>EL: ElevationActor::new(graph_adapter, sqlite_enrichment_repository, speech_service, Some(ontology_repository)).start() (:1222)
        Note over EL,EV: voice-guided path when local speech stack (Whisper STT, PocketTts TTS) is up (app_state.rs:1217-1219) — elevation_voice.rs
    else gate closed
        APP->>APP: log "ElevationActor disabled" (src/app_state.rs:1238)
    end
    Note over MAIN,DE: DecisionElevationActor is started in main() src/main.rs:575, NOT in AppState::new —<br/>a second, separately-gated ACSP actor family alongside ElevationActor
    alt DECISION_ELEVATION_ENABLED gate passes (src/actors/decision_elevation_actor.rs:175)
        MAIN->>DE: DecisionElevationActor::new() then actix::Actor::start(actor) (src/main.rs:575-577)
        MAIN->>MAIN: wrap in ActorElevationSink (src/main.rs:580), fed to DecisionService.with_elevation_sink (src/main.rs:597)
    else gate closed
        MAIN->>MAIN: log "DecisionElevationActor disabled" (src/main.rs:584)
    end
```

## VC-02.19 VoiceInterfaceActor (+ voice_commands.rs) and MultiMcpVisualizationActor
```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:1245-1256
    participant VI as VoiceInterfaceActor<br/>src/actors/voice_interface_actor.rs
    participant VC as voice_commands<br/>src/actors/voice_commands.rs
    participant MMV as MultiMcpVisualizationActor<br/>src/actors/multi_mcp_visualization_actor.rs

    alt speech_service Some
        APP->>VI: VoiceInterfaceActor::new(task_orchestrator_addr.clone(), speech_service.clone()).start() (:1245)
        VI->>VC: dispatch parsed spoken command to settings-assistant path
    else speech_service None
        APP->>APP: log "VoiceInterfaceActor disabled (no speech service)" (:1255)
    end
    Note over MMV: MultiMcpVisualizationActor has no start() call found in src/app_state.rs or src/main.rs —<br/>coverage gap, reported not diagrammed further (see report)
```

## VC-02.20 AgentMonitorActor and AgentBeamActor — message surfaces
```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:1116-1131
    participant AM as AgentMonitorActor<br/>src/actors/agent_monitor_actor.rs:491
    participant CFC as ClaudeFlowClient
    participant AB as AgentBeamActor<br/>src/app_state.rs:696
    participant CC as ClientCoordinatorActor
    participant HUB as agent-events hub

    APP->>CFC: ClaudeFlowClient::new(mcp_host, mcp_port) (:1129)
    APP->>AM: AgentMonitorActor::new(claude_flow_client, graph_service_addr.clone()).start() (:1131)
    alt MOCK_AGENTS set (src/actors/agent_monitor_actor.rs:634)
        AM->>AM: synthesize mock agent roster, skip live MCP poll
    else unset
        AM->>CFC: poll live MCP agent roster
    end
    APP->>AB: AgentBeamActor::new(client_manager_addr.clone()).start() (:696)
    HUB->>AB: process-global agent-event stream (subscription keeps actor alive, no Addr retained)
    AB->>CC: encoded 0x23 frames via existing binary dispatch
```
