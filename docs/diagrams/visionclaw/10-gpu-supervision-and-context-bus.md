---
id: VC-10
title: GPU supervision and context bus
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2007, ADR-2053, ADR-2054, ADR-2140]
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
  - ../project/src/actors/gpu/stress_majorization_actor.rs
  - ../project/src/actors/gpu/clustering_actor.rs
  - ../project/src/actors/gpu/anomaly_detection_actor.rs
  - ../project/src/actors/gpu/pagerank_actor.rs
  - ../project/src/actors/gpu/shortest_path_actor.rs
  - ../project/src/actors/gpu/connected_components_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/app_state.rs
verified_commit: 19417bdff5fc22053a8a0cc185797b0311521d10
---

## VC-10.1 GPU supervision tree

```mermaid
flowchart TD
    APP["AppState::new<br/>src/app_state.rs:943 starts GPUManagerActor when GPU enabled"]
    GM["GPUManagerActor<br/>gpu_manager_actor.rs:57 struct, :146 impl Actor"]
    RS["ResourceSupervisor<br/>resource_supervisor.rs:40 struct, :353 impl Actor"]
    PS["PhysicsSupervisor<br/>physics_supervisor.rs"]
    AS["AnalyticsSupervisor<br/>analytics_supervisor.rs"]
    GAS["GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs"]
    GRA["GPUResourceActor<br/>gpu_resource_actor.rs:451 InitializeGPU"]
    FCA["ForceComputeActor<br/>force_compute_actor.rs:1746 ComputeForces"]
    SMA["StressMajorizationActor<br/>stress_majorization_actor.rs:311"]
    CA["ConstraintActor<br/>constraint_actor.rs:199"]
    OCA["OntologyConstraintActor<br/>ontology_constraint_actor.rs:374"]
    CLA["ClusteringActor<br/>clustering_actor.rs:1156 RunKMeans"]
    ADA["AnomalyDetectionActor<br/>anomaly_detection_actor.rs:118"]
    PRA["PageRankActor<br/>pagerank_actor.rs:361"]
    SPA["ShortestPathActor<br/>shortest_path_actor.rs:213 ComputeSSP"]
    CCA["ConnectedComponentsActor<br/>connected_components_actor.rs:82"]

    APP -->|"InitializeGPU"| GM
    GM -->|"ResourceSupervisor::new().start() (gpu_manager_actor.rs:101)"| RS
    GM -->|"PhysicsSupervisor::new().start() (gpu_manager_actor.rs:91)"| PS
    GM -->|"AnalyticsSupervisor::new().start() (gpu_manager_actor.rs:94)"| AS
    GM -->|"GraphAnalyticsSupervisor::new().start() (gpu_manager_actor.rs:97)"| GAS
    GM -->|"SetSubsystemSupervisors try_send (gpu_manager_actor.rs:105)"| RS
    RS -->|"spawn_resource_actor (resource_supervisor.rs:358) in Actor::started (:356)"| GRA
    RS -->|"SetSharedGPUContext try_send (resource_supervisor.rs:151) then UpdateGPUGraphData (:173)"| PS
    RS -->|"SetSharedGPUContext try_send :196"| AS
    RS -->|"SetSharedGPUContext try_send :215"| GAS
    PS --> FCA
    PS --> SMA
    PS --> CA
    PS --> OCA
    AS --> CLA
    AS --> ADA
    AS --> PRA
    GAS --> SPA
    GAS --> CCA

    N1["LAZY: Actor::started (gpu_manager_actor.rs:149) spawns nothing - spawn_supervisors (:82) runs on the first message via get_supervisors (:126), guarded by supervisors_spawned (:68)"]
    N3["ORDER: ResourceSupervisor is spawned LAST (gpu_manager_actor.rs:101) so the other three addresses already exist for SetSubsystemSupervisors (:105)"]
    GM -.- N3
    N2["ADR-2007 GPUManagerActor is a coordinator not a God Actor - it owns no CUDA handle, only supervisor addresses (gpu_manager_actor.rs:113-118)"]
    GM -.- N1
    GM -.- N2
    N4["SemanticForcesActor was REMOVED (ADR-2140, commit 929fb32c1) - PhysicsSupervisor now spawns exactly four children, ForceComputeActor, StressMajorizationActor, ConstraintActor and OntologyConstraintActor (physics_supervisor.rs:154-185)"]
    PS -.- N4
    N5["The semantic forces are the physics.semantic settings group applied by ForceComputeActor, not a supervised actor, see VC-11.7"]
    FCA -.- N5
```

## VC-10.2 Boot — GPU initialisation with total timeout

```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:943
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:259
    participant RS as ResourceSupervisor<br/>resource_supervisor.rs:40
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:451
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:805

    Note over GM: Actor::started (gpu_manager_actor.rs:149) spawns NO supervisors - they are created on first message via get_supervisors (:126)
    APP->>GM: InitializeGPU with graph + graph_service_addr
    GM->>GM: get_supervisors (gpu_manager_actor.rs:126) then spawn_supervisors (:82)
    GM->>RS: InitializeGPU forwarded
    RS->>RS: pending_graph_data = Some(graph) :401
    alt resource_actor is None
        RS->>RS: spawn_resource_actor :408
    end
    RS->>RS: init_state = InProgress :421
    Note over RS: timeout = self.timeouts.total = 60s<br/>supervisor_messages.rs:222 (device_init 10s :218, ptx_load 5s :219, graph_upload 30s :220, context_distribution 5s :221)
    RS->>GRA: tokio::time::timeout(60s, resource_addr.send(InitializeGPU)) (resource_supervisor.rs:430)
    Note over RS: the inner Ok(Err(mailbox))/Err(elapsed) split collapses to one Result before the outer<br/>match (resource_supervisor.rs:435-439) - the code below distinguishes them by string-matching<br/>"timed out" in the error text (:456), not by nested Result variants
    alt Ok(_) device + PTX ready
        GRA-->>RS: Ok
        RS->>RS: init_state = Completed :446, failure_count = 0 :447, current_delay = 1s :448
        opt shared_context.is_some() :451
            RS->>RS: distribute_context_to_supervisors :452
        end
    else Err(e), e contains "timed out"
        RS->>RS: init_state = TimedOut :457
        Note over RS: warn "GPU initialization timed out, system will continue in degraded mode" :458
        RS->>RS: handle_init_failure :460
    else Err(e), any other error (e.g. mailbox)
        RS->>RS: handle_init_failure :460 with the raw error string (e.g. "Mailbox error: ...", :437)
    end
    GRA->>RS: SetSharedGPUContext (resource_supervisor.rs:470, impl Handler) (context flows back up, not down)
    RS->>RS: init_state = Completed :478 then distribute_context_to_supervisors :481
    RS->>PS: SetSharedGPUContext try_send :151
```

## VC-10.3 SharedGPUContext distribution — direct messages, bus is additive

```mermaid
sequenceDiagram
    autonumber
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:451
    participant RS as ResourceSupervisor<br/>resource_supervisor.rs:129 distribute_context_to_supervisors
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:805
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:415
    participant GAS as GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs:342
    participant BUS as GPUContextBus<br/>context_bus.rs:66 tokio broadcast::Sender

    GRA->>RS: SetSharedGPUContext (gpu_resource_actor.rs:528, try_send to gpu_manager_addr - GM forwards, gpu_manager_actor.rs:591-600)
    RS->>RS: shared_context = Some(msg.context) (resource_supervisor.rs:476, impl Handler at :470)
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
            BUS-->>RS: receiver_count (context_bus.rs:102, a send Err when nobody subscribed maps to 0 via unwrap_or_default on the same line)
        end
        RS->>RS: pending_graph_data = None (resource_supervisor.rs:252)
    end
    Note over RS,BUS: RESOLVED ADR-2053: direct point-to-point delivery is now the DECLARED authoritative mechanism - the bus is a supplementary broadcast for non-supervisor observers and a zero receiver count is normal
    Note over RS,BUS: RESOLVED ADR-2053 + ADR-2060: BASELINE said "not a central handle" - corrected to describe the code, which delivers direct-first and publishes to the bus as a supplement
    Note over RS: RESOLVED ADR-2053: try_send results are now inspected - a failure logs error!, is recorded in context_delivery_failures, and forces get_health to report Degraded instead of Healthy
    Note over RS,BUS: RECORDED in the ADR-2053 addendum of 2026-10-12 (commit 31c87da5e): the supervisor chain is the<br/>authoritative path but NOT the only one - ForceComputeActor::initialize_own_gpu_context (force_compute_actor.rs:649) builds a SharedGPUContext<br/>itself when none has arrived, a deliberate bounded fallback for boot-time message races, see VC-10.6
```

## VC-10.4 GPU readiness lifecycle

```mermaid
stateDiagram-v2
    [*] --> NotStarted
    NotStarted: InitializationState NotStarted - resource_supervisor.rs:30 (enum at :29)
    InProgress: InitializationState InProgress - set at resource_supervisor.rs:421
    Completed: InitializationState Completed - set at resource_supervisor.rs:446 and :478
    Failed: InitializationState Failed(String) - set at resource_supervisor.rs:259
    TimedOut: InitializationState TimedOut - set at resource_supervisor.rs:457

    NotStarted --> InProgress: InitializeGPU handler (resource_supervisor.rs:388-391)
    InProgress --> Completed: Ok(_) within timeouts.total 60s (resource_supervisor.rs:444)
    InProgress --> TimedOut: Err(e), e contains "timed out" (resource_supervisor.rs:455-457)
    InProgress --> Failed: Err(e), any other error e.g. mailbox (resource_supervisor.rs:455-456,460)
    TimedOut --> Failed: handle_init_failure (resource_supervisor.rs:256-259)
    Failed --> NotStarted: ctx.run_later(delay) re-spawns GPUResourceActor (resource_supervisor.rs:286-290)
    Failed --> [*]: failure_count > policy.max_restarts - give up (resource_supervisor.rs:265-271)
    Completed --> Completed: SetSharedGPUContext re-distributes (resource_supervisor.rs:481)

    note right of Failed
        Backoff current_delay starts 1s (resource_supervisor.rs:101)
        multiplied by policy.backoff_multiplier (:276)
        clamped to policy.max_delay (:278)
        reset to 1s on success (:448)
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

    CH->>RS: ActorFailure{actor_name, error} (resource_supervisor.rs:529, impl Handler at :526)
    alt actor_name == "GPUResourceActor" :530
        RS->>RS: handle_init_failure :531
    else other name
        Note over RS: ignored - the if has no else, ResourceSupervisor effectively supervises exactly one child (:530-532)
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
    Note over RS,GRA: MANUAL override - RestartActor (resource_supervisor.rs:536-548) re-spawns and resets state to<br/>NotStarted (:542-543) WITHOUT consuming the backoff budget, and returns Err for any other actor_name (:546)
    Note over RS: GetSubsystemHealth (resource_supervisor.rs:379) reports restart_count = failure_count (:342) so backoff pressure is observable
```

## VC-10.6 ForceComputeActor self-initialisation and supersession

```mermaid
sequenceDiagram
    autonumber
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:649 initialize_own_gpu_context
    participant RS as ResourceSupervisor<br/>resource_supervisor.rs:129
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:805

    Note over FCA: Fields gpu_self_init_attempts :162, gpu_self_init_max_retries :164 (initialised to 3 at :259), gpu_self_init_last_attempt :166
    Note over FCA: Self-init is attempted from three sites - Actor::started (force_compute_actor.rs:1719), the first ComputeForces step while shared_context is None (:1807), and the UpdateGPUGraphData path (:3133)
    alt shared_context already present (force_compute_actor.rs:650)
        Note over FCA: trace "GPU context already present, skipping self-init" - supervisor-supplied context wins (:651)
    else attempts >= max_retries (force_compute_actor.rs:656)
        Note over FCA: trace "GPU self-init exhausted all 3 retries, skipping" (:657-660) - actor stays without a context permanently
    else backoff not elapsed (force_compute_actor.rs:666-675)
        Note over FCA: backoff_secs = 1u64 << (attempts - 1) (:668) giving 1s then 2s then 4s
    else proceed
        FCA->>FCA: gpu_self_init_attempts += 1 :679, gpu_self_init_last_attempt = now :680
        FCA->>FCA: create its own CUDA context
    end
    RS->>PS: SetSharedGPUContext resource_supervisor.rs:151
    PS->>FCA: SetSharedGPUContext (force_compute_actor.rs:3496, impl Handler)
    alt had_context true (force_compute_actor.rs:3500)
        Note over FCA: info "Received SharedGPUContext from supervisor chain (replacing self-initialized context)" (force_compute_actor.rs:3502)
    else first context
        Note over FCA: info "Received SharedGPUContext from supervisor chain" (force_compute_actor.rs:3504)
    end
    FCA->>FCA: shared_context = Some(msg.context) (force_compute_actor.rs:3511) then gpu_state.is_initialized = true (force_compute_actor.rs:3520)
    opt pending_graph_data is Some (force_compute_actor.rs:3525)
        FCA->>FCA: try_upload_pending_graph_data (force_compute_actor.rs:3527)
    end
    Note over FCA,PS: INVARIANT: the externally supplied context always replaces a self-created one so every GPU actor shares one CUDA device and stream force_compute_actor.rs:3507-3510
    Note over FCA: DIVERGENCE, RECORDED as deliberate by the ADR-2053 addendum of 2026-10-12 (commit 31c87da5e): self-init is a second,<br/>unsupervised path to a CUDA context that bypasses ResourceSupervisor timeouts and backoff. It runs only while shared_context is None and a delivered<br/>context always wins (force_compute_actor.rs:650-651), it is bounded by 3 retries with 1s 2s 4s backoff (force_compute_actor.rs:656-675), and the ADR<br/>states that a context built this way is invisible to ResourceSupervisor accounting and GetSubsystemHealth, so the boot line "GPU context<br/>self-initialized successfully" (force_compute_actor.rs:829) means the supervisor path failed. The ADR sets a review trigger: remove the fallback once<br/>delivery has been boot-reliable across a release
    Note over FCA: InitializeGPU (force_compute_actor.rs:3036) deliberately does NOT set gpu_state.num_nodes - that happens only after a successful upload, preventing ComputeForces on uninitialised buffers and CUDA mutex poisoning (force_compute_actor.rs:3048-3050)
```

## VC-10.7 PhysicsSupervisor AllForOne restart and re-attachment of the ForceComputeActor

```mermaid
sequenceDiagram
    autonumber
    participant APP as AppState::new<br/>src/app_state.rs:1024
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:714
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:380 restart_actor
    participant FCA as ForceComputeActor (new)<br/>force_compute_actor.rs:1713
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:1214

    APP->>GM: SetPhysicsSettingsSource (app_state.rs:1024)
    GM->>PS: do_send(SetPhysicsSettingsSource) (gpu_manager_actor.rs:717-719)
    PS->>PS: settings_source = Some(repo), restore_saved_physics (physics_supervisor.rs:787-790)
    Note over GM,PS: SubscribeForceComputeReplaced is forwarded the same way (gpu_manager_actor.rs:696-707) and appended to replacement_subscribers (physics_supervisor.rs:776-777)
    Note over PS: a child mailbox disconnects, the 5 s health check calls handle_actor_failure (physics_supervisor.rs:510-526, :299)
    PS->>PS: restart_actor drops every child address and marks them stopped (physics_supervisor.rs:380-409), AllForOne because the children share one SharedGPUContext mutex (:374-379)
    PS->>FCA: Actor::create via force_compute_factory, mailbox capacity 2048 (physics_supervisor.rs:418-423)
    PS->>PS: re-spawn StressMajorization, Constraint and OntologyConstraint actors, re-wire SetForceComputeAddr (physics_supervisor.rs:428-450)
    FCA->>FCA: started calls initialize_own_gpu_context (force_compute_actor.rs:1719), the new actor starts on compiled-in defaults
    PS->>FCA: restore_saved_physics loads the saved physics row and pushes it (physics_supervisor.rs:456, :203-213)
    PS->>PO: ForceComputeActorReplaced{addr} to each live subscriber (physics_supervisor.rs:461-465)
    PO->>PO: gpu_initialized = false, gpu_compute_addr = new addr, initialize_gpu_if_needed re-sends the graph (physics_orchestrator_actor.rs:1222-1227)
    Note over PS: INVARIANT: every ForceComputeActor replacement is announced to all subscribers so no holder keeps the dead address (physics_supervisor.rs:459-468)
    Note over PS: DEBT: the restart log line says "all 5 physics actors re-spawned" (physics_supervisor.rs:453) but the supervisor now manages four, since SemanticForcesActor was removed (physics_supervisor.rs:154-185)
```

## VC-10.10 GPU-absent and CPU-fallback behaviour per actor

```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs:368
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:1573
    participant SMA as StressMajorizationActor<br/>stress_majorization_actor.rs:95
    participant CCA as ConnectedComponentsActor<br/>connected_components_actor.rs:82
    participant OCA as OntologyConstraintActor<br/>ontology_constraint_actor.rs:256
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:137

    Note over PO,GRA: There is NO single system-wide CPU fallback - each actor degrades differently, so "GPU absent" is not one branch
    PO->>PO: physics step with no GPU
    alt cpu_fallback_warned is false src/actors/physics_orchestrator_actor.rs:585
        Note over PO: warn "CPU physics fallback not implemented - GPU compute is mandatory" once<br/>then cpu_fallback_warned = true :587 (field declared :115-116, initialised false :230)
    end
    Note over PO: on the CPU path no PhysicsStepCompleted message comes back<br/>(physics_orchestrator_actor.rs:368) so the orchestrator re-schedules the next step itself
    FCA->>FCA: ComputeForces with no context
    Note over FCA: initialize_own_gpu_context is retried, and if there is still no context<br/>the handler tells the orchestrator the step was skipped and returns<br/>Err "GPU context not initialized" (force_compute_actor.rs:1807-1822)<br/>the frame is DROPPED, not computed on CPU
    FCA->>FCA: recover_from_divergence with no context
    Note over FCA: warn "recover_from_divergence called with no GPU context" (force_compute_actor.rs:1573)
    SMA->>SMA: stress majorization requested
    Note over SMA: returns Err "GPU not available for stress majorization" (stress_majorization_actor.rs:95) - hard failure, no CPU path
    CCA->>CCA: GPU kernel failed
    Note over CCA: DOC-DRIFT — the CPU fallback described here and in the struct's own doc comment<br/>(connected_components_actor.rs:52-55) was REMOVED under ADR-2054 (:213-220): it ran<br/>compute_components_cpu against cached_edges, a field only ever populated by the<br/>UpdateComponentEdges message, which had zero senders tree-wide — a fabricated singleton-<br/>per-node result, not a real fallback. GPU failure now returns Err directly (:221-224),<br/>same hard-fail posture as StressMajorization, not a degrade
    OCA->>OCA: constraints arrive before GPU
    Note over OCA: debug "GPU not available, constraints cached for next physics step" (ontology_constraint_actor.rs:255-257) and cpu_fallback_count += 1 (:251, :258) surfaced via GetOntologyConstraintStats (:488, field :94)
    GRA->>GRA: APSP PTX load fails
    Note over GRA: warn "Failed to load APSP/landmark PTX" (gpu_resource_actor.rs:137), apsp_ptx = None<br/>the comment at :120-125 says there is no CPU fallback to report<br/>(ComputeAPSP refuses under NFR-7)
    Note over CCA: SemanticForcesActor no longer exists (ADR-2140, commit 929fb32c1)<br/>its kernel_bridge stubs and CPU twins were removed, and the semantic forces<br/>run in ForceComputeActor with no CPU fallback by decision (VC-11.7)
    Note over CCA: POLICY: ComputeSSP returns Err without a GPU context or on GPU failure<br/>shortest_path_actor.rs:223-248, there is no SSSP CPU fallback.<br/>ComputeAPSP separately refuses all dense all-pairs requests under NFR-7<br/>shortest_path_actor.rs:348-365, the removed fallback named at :355 was APSP.
    Note over PO,GRA: DIVERGENCE: coverage is uneven but no actor now degrades to a CPU implementation<br/>ConnectedComponents and StressMajorization hard-fail (ADR-2054 removed CC's fabricated CPU path)<br/>ComputeForces drops the frame, the orchestrator logs once and skips<br/>SSSP refuses without GPU, dense APSP is deliberately disabled
```
