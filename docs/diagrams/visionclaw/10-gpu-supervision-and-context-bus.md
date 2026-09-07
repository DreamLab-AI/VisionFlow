---
id: VC-10
title: GPU supervision and context bus
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2007, ADR-2053]
sources:
  - ../project/src/actors/gpu/gpu_manager_actor.rs
  - ../project/src/actors/gpu/resource_supervisor.rs
  - ../project/src/actors/gpu/physics_supervisor.rs
  - ../project/src/actors/gpu/analytics_supervisor.rs
  - ../project/src/actors/gpu/graph_analytics_supervisor.rs
  - ../project/src/actors/gpu/context_bus.rs
  - ../project/src/actors/gpu/gpu_resource_actor.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/supervisor_messages.rs
  - ../project/src/actors/gpu/constraint_actor.rs
  - ../project/src/actors/gpu/ontology_constraint_actor.rs
  - ../project/src/actors/gpu/semantic_forces_actor.rs
  - ../project/src/actors/gpu/stress_majorization_actor.rs
  - ../project/src/actors/gpu/clustering_actor.rs
  - ../project/src/actors/gpu/anomaly_detection_actor.rs
  - ../project/src/actors/gpu/pagerank_actor.rs
  - ../project/src/actors/gpu/shortest_path_actor.rs
  - ../project/src/actors/gpu/connected_components_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/app_state.rs
verified_commit: 36bb64e1e
---

## VC-10.1 GPU supervision tree

```mermaid
flowchart TD
    APP["AppState::new<br/>src/app_state.rs:970 starts GPUManagerActor when GPU enabled"]
    GM["GPUManagerActor<br/>gpu_manager_actor.rs:57 struct, :140 impl Actor"]
    RS["ResourceSupervisor<br/>resource_supervisor.rs:40 struct, :347 impl Actor"]
    PS["PhysicsSupervisor<br/>physics_supervisor.rs"]
    AS["AnalyticsSupervisor<br/>analytics_supervisor.rs"]
    GAS["GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs"]
    GRA["GPUResourceActor<br/>gpu_resource_actor.rs:456 InitializeGPU"]
    FCA["ForceComputeActor<br/>force_compute_actor.rs:1867 ComputeForces"]
    SMA["StressMajorizationActor<br/>stress_majorization_actor.rs:315"]
    CA["ConstraintActor<br/>constraint_actor.rs:193"]
    OCA["OntologyConstraintActor<br/>ontology_constraint_actor.rs:451"]
    SFA["SemanticForcesActor<br/>semantic_forces_actor.rs:751"]
    CLA["ClusteringActor<br/>clustering_actor.rs:1172 RunKMeans"]
    ADA["AnomalyDetectionActor<br/>anomaly_detection_actor.rs:112"]
    PRA["PageRankActor<br/>pagerank_actor.rs:433"]
    SPA["ShortestPathActor<br/>shortest_path_actor.rs:214 ComputeSSP"]
    CCA["ConnectedComponentsActor<br/>connected_components_actor.rs:233"]

    APP -->|"InitializeGPU"| GM
    GM -->|"ResourceSupervisor::new().start() (gpu_manager_actor.rs:101)"| RS
    GM -->|"PhysicsSupervisor::new().start() (gpu_manager_actor.rs:91)"| PS
    GM -->|"AnalyticsSupervisor::new().start() (gpu_manager_actor.rs:94)"| AS
    GM -->|"GraphAnalyticsSupervisor::new().start() (gpu_manager_actor.rs:97)"| GAS
    GM -->|"SetSubsystemSupervisors try_send (gpu_manager_actor.rs:105)"| RS
    RS -->|"spawn_resource_actor (resource_supervisor.rs:352) in Actor::started (:350)"| GRA
    RS -->|"SetSharedGPUContext try_send (resource_supervisor.rs:151) then UpdateGPUGraphData (:173)"| PS
    RS -->|"SetSharedGPUContext try_send :196"| AS
    RS -->|"SetSharedGPUContext try_send :215"| GAS
    PS --> FCA
    PS --> SMA
    PS --> CA
    PS --> OCA
    PS --> SFA
    AS --> CLA
    AS --> ADA
    AS --> PRA
    GAS --> SPA
    GAS --> CCA

    N1["LAZY: Actor::started (gpu_manager_actor.rs:143) spawns nothing - spawn_supervisors (:82) runs on the first message via get_supervisors (:126), guarded by supervisors_spawned (:68)"]
    N3["ORDER: ResourceSupervisor is spawned LAST (gpu_manager_actor.rs:101) so the other three addresses already exist for SetSubsystemSupervisors (:105)"]
    GM -.- N3
    N2["ADR-2007 GPUManagerActor is a coordinator not a God Actor - it owns no CUDA handle, only supervisor addresses (gpu_manager_actor.rs:113-118)"]
    GM -.- N1
    GM -.- N2
```

## VC-10.2 Boot — GPU initialisation with total timeout

```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:970
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:253
    participant RS as ResourceSupervisor<br/>resource_supervisor.rs:300
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:456
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:768

    Note over GM: Actor::started (gpu_manager_actor.rs:143) spawns NO supervisors - they are created on first message via get_supervisors (:126)
    APP->>GM: InitializeGPU with graph + graph_service_addr
    GM->>GM: get_supervisors (gpu_manager_actor.rs:130) then spawn_supervisors (:82)
    GM->>RS: InitializeGPU forwarded
    RS->>RS: pending_graph_data = Some(graph) :395
    alt resource_actor is None
        RS->>RS: spawn_resource_actor :402
    end
    RS->>RS: init_state = InProgress :415
    Note over RS: timeout = self.timeouts.total = 60s<br/>supervisor_messages.rs:222 (device_init 10s :218, ptx_load 5s :219, graph_upload 30s :220, context_distribution 5s :221)
    RS->>GRA: tokio::time::timeout(60s, resource_addr.send(InitializeGPU)) (resource_supervisor.rs:424)
    Note over RS: the inner Ok(Err(mailbox))/Err(elapsed) split collapses to one Result before the outer<br/>match (resource_supervisor.rs:429-433) - the code below distinguishes them by string-matching<br/>"timed out" in the error text (:450), not by nested Result variants
    alt Ok(_) device + PTX ready
        GRA-->>RS: Ok
        RS->>RS: init_state = Completed :440, failure_count = 0 :441, current_delay = 1s :442
        opt shared_context.is_some() :445
            RS->>RS: distribute_context_to_supervisors :446
        end
    else Err(e), e contains "timed out"
        RS->>RS: init_state = TimedOut :451
        Note over RS: warn "GPU initialization timed out, system will continue in degraded mode" :452
        RS->>RS: handle_init_failure :454
    else Err(e), any other error (e.g. mailbox)
        RS->>RS: handle_init_failure :454 with the raw error string (e.g. "Mailbox error: ...", :431)
    end
    GRA->>RS: SetSharedGPUContext (resource_supervisor.rs:464, impl Handler) (context flows back up, not down)
    RS->>RS: init_state = Completed :472 then distribute_context_to_supervisors :475
    RS->>PS: SetSharedGPUContext try_send :151
```

## VC-10.3 SharedGPUContext distribution — direct messages, bus is additive

```mermaid
sequenceDiagram
    autonumber
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:456
    participant RS as ResourceSupervisor<br/>resource_supervisor.rs:123 distribute_context_to_supervisors
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:768
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:409
    participant GAS as GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs:336
    participant BUS as GPUContextBus<br/>context_bus.rs:66 tokio broadcast::Sender

    GRA->>RS: SetSharedGPUContext (gpu_resource_actor.rs:533, try_send to gpu_manager_addr - GM forwards, gpu_manager_actor.rs:599-609)
    RS->>RS: shared_context = Some(msg.context) (resource_supervisor.rs:470, impl Handler at :464)
    RS->>RS: distribute_context_to_supervisors (resource_supervisor.rs:129)
    alt shared_context is None
        Note over RS: warn "No context to distribute" and return (resource_supervisor.rs:133-134)
    else context present
        rect rgb(220,235,250)
            Note over RS,GAS: PRIMARY PATH - central point-to-point try_send from ResourceSupervisor
            opt physics_supervisor is Some (resource_supervisor.rs:150)
                RS->>PS: SetSharedGPUContext try_send :151
                opt pending_graph_data is Some (resource_supervisor.rs:168)
                    RS->>PS: UpdateGPUGraphData try_send :173 (relays graph to ForceComputeActor)
                end
            end
            opt analytics_supervisor is Some (resource_supervisor.rs:195)
                RS->>AS: SetSharedGPUContext try_send :196
            end
            opt graph_analytics_supervisor is Some (resource_supervisor.rs:214)
                RS->>GAS: SetSharedGPUContext try_send :215
            end
        end
        rect rgb(250,235,220)
            Note over RS,BUS: SECONDARY PATH - comment reads "the bus is a SUPPLEMENTARY broadcast" (resource_supervisor.rs:235-237)
            RS->>BUS: context_bus.publish(context) (resource_supervisor.rs:238)
            BUS-->>RS: receiver_count :92 (Err(_) maps to 0 when nobody subscribed :105)
        end
        RS->>RS: pending_graph_data = None (resource_supervisor.rs:252)
    end
    Note over RS,BUS: RESOLVED ADR-2053: direct point-to-point delivery is now the DECLARED authoritative mechanism - the bus is a supplementary broadcast for non-supervisor observers and a zero receiver count is normal
    Note over RS,BUS: RESOLVED ADR-2053 + ADR-2060: BASELINE said "not a central handle" - corrected to describe the code, which delivers direct-first and publishes to the bus as a supplement
    Note over RS: RESOLVED ADR-2053: try_send results are now inspected - a failure logs error!, is recorded in context_delivery_failures, and forces get_health to report Degraded instead of Healthy
```

## VC-10.4 GPU readiness lifecycle

```mermaid
stateDiagram-v2
    [*] --> NotStarted
    NotStarted: InitializationState NotStarted - resource_supervisor.rs:30 (enum at :29)
    InProgress: InitializationState InProgress - set at resource_supervisor.rs:415
    Completed: InitializationState Completed - set at resource_supervisor.rs:440 and :472
    Failed: InitializationState Failed(String) - set at resource_supervisor.rs:259
    TimedOut: InitializationState TimedOut - set at resource_supervisor.rs:451

    NotStarted --> InProgress: InitializeGPU handler (resource_supervisor.rs:382-385)
    InProgress --> Completed: Ok(_) within timeouts.total 60s (resource_supervisor.rs:438)
    InProgress --> TimedOut: Err(e), e contains "timed out" (resource_supervisor.rs:449-451)
    InProgress --> Failed: Err(e), any other error e.g. mailbox (resource_supervisor.rs:449-450,454)
    TimedOut --> Failed: handle_init_failure (resource_supervisor.rs:256-259)
    Failed --> NotStarted: ctx.run_later(delay) re-spawns GPUResourceActor (resource_supervisor.rs:286-290)
    Failed --> [*]: failure_count > policy.max_restarts - give up (resource_supervisor.rs:265-271)
    Completed --> Completed: SetSharedGPUContext re-distributes (resource_supervisor.rs:475)

    note right of Failed
        Backoff current_delay starts 1s (resource_supervisor.rs:101)
        multiplied by policy.backoff_multiplier (:276)
        clamped to policy.max_delay (:278)
        reset to 1s on success (:442)
    end note
    note right of Completed
        get_health (resource_supervisor.rs:294) maps state to SubsystemStatus
        Completed AND has_context to Healthy (:304-306)
        InProgress or NotStarted to Initializing (:308-309)
        Failed or TimedOut to Degraded (:310-312)
    end note
```

## VC-10.5 Initialisation failure, backoff and manual restart

```mermaid
sequenceDiagram
    autonumber
    participant CH as Child actor / init future
    participant RS as ResourceSupervisor<br/>resource_supervisor.rs:256 handle_init_failure
    participant CTX as Actix Context<br/>ctx.run_later
    participant GRA as GPUResourceActor<br/>spawned via resource_supervisor.rs:121 spawn_resource_actor

    CH->>RS: ActorFailure{actor_name, error} (resource_supervisor.rs:523, impl Handler at :520)
    alt actor_name == "GPUResourceActor" :524
        RS->>RS: handle_init_failure :525
    else other name
        Note over RS: ignored - the if has no else, ResourceSupervisor effectively supervises exactly one child (:524-526)
    end
    RS->>RS: init_state = Failed(error) :259, failure_count += 1 :260, last_error :261, last_attempt :262
    alt failure_count > policy.max_restarts :265
        Note over RS: error "Exceeded max initialization attempts, giving up" :266-269 then return - NO further retry is scheduled
    else within budget
        RS->>RS: delay = current_delay :274 then current_delay = min(current_delay * backoff_multiplier, max_delay) :275-278
        RS->>CTX: ctx.run_later(delay, ...) :286
        CTX->>RS: closure fires
        RS->>GRA: spawn_resource_actor :288
        RS->>RS: init_state = NotStarted :289
    end
    Note over RS,GRA: MANUAL override - RestartActor (resource_supervisor.rs:530-542) re-spawns and resets state to<br/>NotStarted (:536-537) WITHOUT consuming the backoff budget, and returns Err for any other actor_name (:540)
    Note over RS: GetSubsystemHealth (resource_supervisor.rs:373) reports restart_count = failure_count (:342) so backoff pressure is observable
```

## VC-10.6 ForceComputeActor self-initialisation and supersession

```mermaid
sequenceDiagram
    autonumber
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:755 self-init guard
    participant RS as ResourceSupervisor<br/>resource_supervisor.rs:138
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:768

    Note over FCA: Fields gpu_self_init_attempts :315, gpu_self_init_max_retries = 3 :425, gpu_self_init_last_attempt :319
    alt shared_context already present (force_compute_actor.rs:754)
        Note over FCA: trace "GPU context already present, skipping self-init" - supervisor-supplied context wins (:755)
    else attempts >= max_retries (force_compute_actor.rs:760)
        Note over FCA: trace "GPU self-init exhausted all 3 retries, skipping" (:761-764) - actor stays without a context permanently
    else backoff not elapsed (force_compute_actor.rs:770-779)
        Note over FCA: backoff_secs = 1u64 << (attempts - 1) (:772) giving 1s then 2s then 4s
    else proceed
        FCA->>FCA: gpu_self_init_attempts += 1 :783, gpu_self_init_last_attempt = now :784
        FCA->>FCA: create its own CUDA context
    end
    RS->>PS: SetSharedGPUContext :138
    PS->>FCA: SetSharedGPUContext (force_compute_actor.rs:3825, impl Handler)
    alt had_context true (force_compute_actor.rs:3829)
        Note over FCA: info "Received SharedGPUContext from supervisor chain (replacing self-initialized context)" (force_compute_actor.rs:3831)
    else first context
        Note over FCA: info "Received SharedGPUContext from supervisor chain" (force_compute_actor.rs:3833)
    end
    FCA->>FCA: shared_context = Some(msg.context) (force_compute_actor.rs:3840) then gpu_state.is_initialized = true (force_compute_actor.rs:3849)
    opt pending_graph_data is Some (force_compute_actor.rs:3854)
        FCA->>FCA: try_upload_pending_graph_data (force_compute_actor.rs:3856)
    end
    Note over FCA,PS: INVARIANT: the externally supplied context always replaces a self-created one so every GPU actor shares one CUDA device and stream :3836-3839
    Note over FCA: DIVERGENCE: self-init is a second, unsupervised path to a CUDA context that bypasses ResourceSupervisor timeouts and backoff entirely
    Note over FCA: InitializeGPU :3394 deliberately does NOT set gpu_state.num_nodes - that happens only after a successful upload, preventing ComputeForces on uninitialised buffers and CUDA mutex poisoning :3406-3408
```

## VC-10.7 Message surface — coordinator and supervisors

```mermaid
classDiagram
    class GPUManagerActor {
        +GetGPUSystemHealth() L181
        +InitializeGPU() L253
        +UpdateGPUGraphData() L294
        +ComputeForces() L329
        +RunKMeans() L346
        +RunCommunityDetection() L369
        +RunDBSCAN() L392
        +RunAnomalyDetection() L415
        +PerformGPUClustering() L438
        +TriggerStressMajorization() L469
        +UpdateConstraints() L483
        +GetGPUStatus() L497
        +GetForceComputeActor() L511
        +UploadConstraintsToGPU() L534
        +GetNodeData() L548
        +UpdateSimulationParams() L571
        +UpdateAdvancedParams() L585
        +SetSharedGPUContext() L599
        +ApplyOntologyConstraints() L628
        +ApplyMaterializedAxioms() L645
    }
    class ResourceSupervisor {
        +GetSubsystemHealth() L289
        +InitializeGPU() L298
        +SetSharedGPUContext() L380
        +SetSubsystemSupervisors() L413
        +ActorFailure() L436
        +RestartActor() L446
        +GetContextBus() L462
        +UpdateGPUGraphData() L476
    }
    class PhysicsSupervisor {
        +GetSubsystemHealth() L557
        +InitializeSubsystem() L586
        +ActorFailure() L603
        +RestartActor() L611
        +ComputeForces() L628
        +TriggerStressMajorization() L658
        +UpdateConstraints() L683
        +ApplyOntologyConstraints() L707
        +ApplyMaterializedAxioms() L732
        +GetForceComputeActor() L758
        +SetSharedGPUContext() L768
        +UpdateSimulationParams() L784
        +GetPhysicsStats() L808
        +UpdateGPUGraphData() L832
        +UpdateAdvancedParams() L846
        +UploadConstraintsToGPU() L870
        +GetNodeData() L894
        +GetOntologyConstraintStats() L921
        +GetSemanticConfig() L954
        +GetHierarchyLevels() L981
        +RecalculateHierarchy() L1008
        +ConfigureDAG() L1032
        +ConfigureTypeClustering() L1056
        +ConfigureCollision() L1080
        +AdjustConstraintWeights() L1104
    }
    class AnalyticsSupervisor {
        +GetSubsystemHealth() L366
        +InitializeSubsystem() L393
        +SetSharedGPUContext() L409
        +SetNodeAnalytics() L424
        +WriteClusterAnalytics() L434
        +ActorFailure() L459
        +RestartActor() L467
        +RunKMeans() L484
        +RunCommunityDetection() L514
        +RunDBSCAN() L538
        +RunAnomalyDetection() L568
        +ComputePageRank() L593
        +UpdateGPUGraphData() L617
        +PerformGPUClustering() L642
    }
    class GraphAnalyticsSupervisor {
        +GetSubsystemHealth() L294
        +InitializeSubsystem() L320
        +SetSharedGPUContext() L336
        +ActorFailure() L351
        +RestartActor() L359
        +ComputeShortestPaths() L376
        +ComputeConnectedComponents() L435
    }
    GPUManagerActor --> ResourceSupervisor : spawns L101
    GPUManagerActor --> PhysicsSupervisor : spawns L91
    GPUManagerActor --> AnalyticsSupervisor : spawns L94
    GPUManagerActor --> GraphAnalyticsSupervisor : spawns L97
    ResourceSupervisor --> PhysicsSupervisor : SetSharedGPUContext L138
    ResourceSupervisor --> AnalyticsSupervisor : SetSharedGPUContext L163
    ResourceSupervisor --> GraphAnalyticsSupervisor : SetSharedGPUContext L173
```

## VC-10.8 Message surface — physics leaf actors

```mermaid
classDiagram
    class ForceComputeActor {
        +ComputeForces() L1867
        +SetLayoutMode() L2641
        +SetRadialLayout() L2730
        +UpdateSimulationParams() L2858
        +UpdateClusteringParams() L3081
        +ForceFullBroadcast() L3108
        +SetComputeMode() L3238
        +GetPhysicsStats() L3255
        +UpdateAdvancedParams() L3263
        +UploadPositions() L3303
        +PinNodePositions() L3367
        +InitializeGPU() L3394
        +UpdateGPUGraphData() L3478
        +GetNodeData() L3507
        +GetGPUStatus() L3515
        +GetCurrentPositions() L3528
        +GetSettlementState() L3607
        +SetPhysicsSettled() L3621
        +GetGPUMetrics() L3633
        +RunCommunityDetection() L3651
        +UpdateVisualAnalyticsParams() L3659
        +GetConstraints() L3672
        +UpdateConstraints() L3680
        +UploadConstraintsToGPU() L3700
        +TriggerStressMajorization() L3724
        +GetStressMajorizationStats() L3736
        +ResetStressMajorizationSafety() L3752
        +UpdateStressMajorizationParams() L3767
        +PerformGPUClustering() L3780
        +GetClusteringResults() L3790
        +UpdateOntologyConstraintBuffer() L3805
        +SetSharedGPUContext() L3825
        +SetPhysicsOrchestratorAddr() L3879
        +ResetPositions() L3894
        +ConfigureStressMajorization() L3985
        +GetStressMajorizationConfig() L4026
        +ConfigureBroadcastOptimization() L4059
        +UpdateCameraFrustum() L4107
        +GetBroadcastStats() L4128
        +RunAnomalyDetection() L4159
        +PositionBroadcastAck() L4371
    }
    class StressMajorizationActor {
        +TriggerStressMajorization() L315
        +ResetStressMajorizationSafety() L334
        +UpdateStressMajorizationParams() L347
        +CheckStressMajorization() L369
        +SetSharedGPUContext() L397
        +ConfigureStressMajorization() L410
        +GetStressMajorizationConfig() L475
    }
    class ConstraintActor {
        +UpdateConstraints() L193
        +GetConstraints() L215
        +UploadConstraintsToGPU() L224
        +ClearConstraints() L256
        +GetConstraintStatistics() L264
        +SetSharedGPUContext() L281
    }
    class OntologyConstraintActor {
        +ApplyOntologyConstraints() L451
        +ApplyMaterializedAxioms() L522
        +UpdateOntologyConstraints() L545
        +GetOntologyStats() L557
        +GetOntologyConstraintStats() L565
        +SetForceComputeAddr() L589
        +SetSharedGPUContext() L603
        +GetConstraintStats() L625
        +GetConstraintBuffer() L649
        +UpdateConstraints() L668
        +InitializeGPU() L705
        +AdjustConstraintWeights() L723
    }
    class SemanticForcesActor {
        +ConfigureDAG() L751
        +ConfigureTypeClustering() L778
        +ConfigureCollision() L802
        +GetSemanticConfig() L826
        +GetHierarchyLevels() L834
        +RecalculateHierarchy() L844
        +SetSharedGPUContext() L859
        +ReloadRelationshipBuffer() L869
    }
    PhysicsSupervisor --> ForceComputeActor
    PhysicsSupervisor --> StressMajorizationActor
    PhysicsSupervisor --> ConstraintActor
    PhysicsSupervisor --> OntologyConstraintActor
    PhysicsSupervisor --> SemanticForcesActor
```

## VC-10.9 Message surface — analytics and graph-analytics leaf actors

```mermaid
classDiagram
    class ClusteringActor {
        +SetSharedGPUContext() L1159
        +RunKMeans() L1172
        +RunCommunityDetection() L1192
        +RunDBSCAN() L1209
        +SetNodeAnalytics() L1229
        +WriteClusterAnalytics() L1238
        +UpdateGPUGraphData() L1286
        +PerformGPUClustering() L1304
    }
    class AnomalyDetectionActor {
        +RunAnomalyDetection() L112
        +SetSharedGPUContext() L423
        +SetNodeAnalytics() L435
    }
    class PageRankActor {
        +ComputePageRank() L433
        +GetPageRankResult() L561
        +ClearPageRankCache() L570
        +SetSharedGPUContext() L580
        +SetNodeAnalytics() L592
        +InitializeActor() L602
    }
    class ShortestPathActor {
        +InitializeActor() L185
        +SetSharedGPUContext() L194
        +SetNodeSSSP() L205
        +ComputeSSP() L214
        +ComputeAPSP() L349
        +GetShortestPathStats() L373
    }
    class ConnectedComponentsActor {
        +InitializeActor() L213
        +SetSharedGPUContext() L222
        +ComputeConnectedComponents() L233
        +GetConnectedComponentsStats() L327
        +UpdateComponentEdges() L339
    }
    class GPUResourceActor {
        +InitializeGPU() L451
        +UpdateGPUGraphData() L567
        +GetNodeData() L580
    }
    AnalyticsSupervisor --> ClusteringActor
    AnalyticsSupervisor --> AnomalyDetectionActor
    AnalyticsSupervisor --> PageRankActor
    GraphAnalyticsSupervisor --> ShortestPathActor
    GraphAnalyticsSupervisor --> ConnectedComponentsActor
    ResourceSupervisor --> GPUResourceActor
```

## VC-10.10 GPU-absent and CPU-fallback behaviour per actor

```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:379
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3122
    participant SMA as StressMajorizationActor<br/>stress_majorization_actor.rs:95
    participant CCA as ConnectedComponentsActor<br/>connected_components_actor.rs:82
    participant OCA as OntologyConstraintActor<br/>ontology_constraint_actor.rs:291
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:142

    Note over PO,GRA: There is NO single system-wide CPU fallback - each actor degrades differently, so "GPU absent" is not one branch
    PO->>PO: physics step with no GPU
    alt cpu_fallback_warned is false :602
        Note over PO: warn once then cpu_fallback_warned = true :604 (field declared :123-124, initialised false :241)
    end
    Note over PO: on the CPU path no PhysicsStepCompleted message comes back :379 so the orchestrator must not await one
    FCA->>FCA: ForceFullBroadcast with no context
    Note over FCA: warn "ForceFullBroadcast - no GPU context, skipping" :3122 - the frame is DROPPED, not computed on CPU
    FCA->>FCA: recover_from_divergence with no context
    Note over FCA: warn "recover_from_divergence called with no GPU context" :1695
    SMA->>SMA: stress majorization requested
    Note over SMA: returns Err "GPU not available for stress majorization" :95 - hard failure, no CPU path
    CCA->>CCA: GPU kernel failed
    Note over CCA: DOC-DRIFT — the CPU fallback described here and in the struct's own doc comment<br/>(connected_components_actor.rs:52-55) was REMOVED under ADR-2054 (:213-220): it ran<br/>compute_components_cpu against cached_edges, a field only ever populated by the<br/>UpdateComponentEdges message, which had zero senders tree-wide — a fabricated singleton-<br/>per-node result, not a real fallback. GPU failure now returns Err directly (:221-224),<br/>same hard-fail posture as StressMajorization, not a degrade
    OCA->>OCA: constraints arrive before GPU
    Note over OCA: info "GPU not available, constraints cached for next physics step" :291 and cpu_fallback_count += 1 :286,:293 surfaced via GetConstraintStats :581
    GRA->>GRA: APSP PTX load fails
    Note over GRA: warn "Failed to load APSP PTX (will use CPU fallback)" :142
    Note over CCA: SemanticForcesActor carries CPU fallback implementations :221 in semantic_forces_actor.rs
    Note over CCA: POLICY: ComputeSSP returns Err without a GPU context or on GPU failure<br/>shortest_path_actor.rs:224-249, there is no SSSP CPU fallback.<br/>ComputeAPSP separately refuses all dense all-pairs requests under NFR-7<br/>shortest_path_actor.rs:349-366, the removed fallback at :356 was APSP.
    Note over PO,GRA: DIVERGENCE: coverage is uneven - only SemanticForces degrades to a real CPU implementation,<br/>ConnectedComponents and StressMajorization hard-fail (ADR-2054 removed CC's fabricated CPU path),<br/>ForceFullBroadcast drops the frame, SSSP refuses without GPU, dense APSP is deliberately disabled
```
