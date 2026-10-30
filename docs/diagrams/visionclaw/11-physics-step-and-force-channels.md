---
id: VC-11
title: Physics step and force channels
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2007, ADR-2028, ADR-2029, ADR-2055, ADR-2060, ADR-2135, ADR-2140, ADR-2141]
sources:
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/physics_supervisor.rs
  - ../project/src/actors/gpu/gpu_manager_actor.rs
  - ../project/src/actors/gpu/constraint_actor.rs
  - ../project/src/actors/gpu/ontology_constraint_actor.rs
  - ../project/src/actors/gpu/stress_majorization_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/actors/messages/physics_messages.rs
  - ../project/src/models/force_channels.rs
  - ../project/docs/GPU-wire-abi.md
  - ../project/src/models/simulation_params.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu
  - ../project/tests/gpu/integrate_bounds.cu
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/utils/unified_gpu_compute/semantic.rs
  - ../project/crates/visionclaw-domain/src/types/semantic_config.rs
  - ../project/src/actors/gpu/domain_charge.rs
  - ../project/crates/visionclaw-domain/src/models/simulation_params.rs
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/actors/gpu/gpu_resource_actor.rs
  - ../project/crates/vault-core/src/domains.rs
verified_commit: e4ba5eb35ee6715cfd7f2daf1cd9af05e8a18cce
---

## VC-11.1 Physics tick — phase 1, params and flag word

```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:640
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:1746
    participant EX as UnifiedGPUCompute::execute_physics_step_with_bypass<br/>src/utils/unified_gpu_compute/execution.rs:1056
    participant FC as derive_dispatch_feature_flags<br/>src/models/force_channels.rs:488

    PO->>FCA: ComputeForces do_send (physics_orchestrator_actor.rs:572), the step loop holds the ForceComputeActor address directly
    Note over PS: the supervisor forwarder is a second route to the same actor<br/>GPUManagerActor (src/actors/gpu/gpu_manager_actor.rs:335) to PhysicsSupervisor (physics_supervisor.rs:640) to ForceComputeActor (:655)<br/>the orchestrator does not use it
    alt shared_context is None
        Note over FCA: no GPU context - initialize_own_gpu_context is retried (force_compute_actor.rs:1807), if still none the frame is skipped,<br/>an error is logged every 300th skip (:1813), the orchestrator is told it was a benign skip and Err is returned (:1822)
    else GPU ready
        FCA->>FCA: stability_warmup_remaining -= 1 :1887 (stability_bypass while it is above 0, :1885)
        Note over FCA: warmup reset on graph upload - 1200 frames when edge_count == 0 else 600 :1303
        FCA->>FCA: reheat_factor read :1889
        FCA->>EX: execute_physics_step_with_bypass(sim_params, stability_bypass) :1944 from spawn_blocking, after inject_velocity_perturbation when reheating :1937
        EX->>FC: derive_dispatch_feature_flags(ForceDispatchInputs) :1069
        FC-->>EX: flags word
        EX->>EX: sim_params.feature_flags = feature_flags :1083
        Note over EX: OVERWRITE - the converter's own flag word is discarded before every execute, so only this word reaches the device
    end
    Note over FC: repel_k > 0.0 sets ENABLE_REPULSION :490-492
    Note over FC: spring_k > 0.0 sets ENABLE_SPRINGS :493-495
    Note over FC: center_gravity_k > 0.0 sets ENABLE_CENTERING :496-498
    Note over FC: use_sssp_distances OR sssp_spring_adjust_enabled sets ENABLE_SSSP_SPRING_ADJUST :499-501
    Note over FC: num_constraints > 0 sets ENABLE_CONSTRAINTS (force_channels.rs:504-506) - KEYSTONE, never from user settings
    Note over EX,FC: INVARIANT ADR-2029 - ENABLE_CONSTRAINTS is residency-derived at step time, never a settings toggle
```

## VC-11.2 Physics tick — phase 2, kernel launch and integrate

```mermaid
sequenceDiagram
    autonumber
    participant EX as UnifiedGPUCompute::execute<br/>execution.rs:171
    participant K1 as force_pass_kernel<br/>crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:385
    participant SG as SemanticGpu::apply<br/>src/utils/unified_gpu_compute/semantic.rs:393
    participant K2 as integrate_pass_kernel<br/>crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:902
    participant EV as cust Event<br/>execution.rs completion poll

    opt num_constraints > 0 (execution.rs:258)
        EX->>EX: bind constraint_force_ptr and constraint_force_epsilon :258
    end
    EX->>K1: launch with pos/vel/force buffers, mass, num_nodes
    EX->>K1: class_id, class_charge, class_mass device pointers (ontology metadata)
    EX->>K1: prev_force_x/y/z for FA2 swing and traction adaptive speed
    EX->>K1: pinned_mask device pointer
    Note over K1: pinned nodes SKIP integration but still exert forces on neighbours
    Note over EX: DEBT: the UploadPositions handler (force_compute_actor.rs:2945) still uploads positions via spawn_blocking and update_positions_only,<br/>but nothing in src, crates or tests sends UploadPositions (src/actors/messages/physics_messages.rs:152 declares it) - drags arrive as PinNodePositions (force_compute_actor.rs:3009)
    K1->>SG: forces written (cluster cohesion and degree-weighted gravity kernels run between, execution.rs:843-898)
    Note over SG: when semantic_active and the metadata matches the graph node count, the semantic kernels ADD their forces to the live force buffers (execution.rs:918-931),<br/>otherwise they are skipped with a warning (execution.rs:932-938). Applied exactly as stored, no scale or cap (ADR-2140)
    SG->>K2: forces including semantic terms
    Note over K2: integrate_pass adds boundary push (viewport_bounds + boundary_damping) and annealing jitter (temperature + cooling_rate) per force_channels.rs:38-39
    EX->>EV: completion_event.record(stream) (execution.rs:985-986)
    loop poll until EventStatus::Ready
        EX->>EV: completion_event.query()
        alt poll_start.elapsed() > 10s
            Note over EX: returns Err "GPU kernel execution timed out after 10s" (execution.rs:995)
        else not ready
            EX->>EX: std::thread::yield_now()
        end
    end
    EX->>EX: swap_buffers() (execution.rs:1000) then iteration += 1
    opt iteration % 100 == 0
        Note over EX: logs memory MB, utilisation pct, grid occupancy pct, resize count
    end
    Note over K1,K2: Gravity and cluster cohesion run as SEPARATE kernels - degree_weighted_gravity_kernel and cluster_cohesion_kernel (force_channels.rs:42-43)
```

## VC-11.3 Feature-flag derivation across the conversion paths

```mermaid
flowchart TD
    A["Path A converter - From SimulationParams<br/>project/src/models/simulation_params.rs:779-792"]
    B["Path B converter - From PhysicsSettings<br/>project/src/models/simulation_params.rs:862-875"]
    C["Registry mutator ForceChannel::apply<br/>src/models/force_channels.rs:185-200"]
    D["derive_dispatch_feature_flags<br/>src/models/force_channels.rs:488"]
    E["execution.rs:1069 calls the helper"]
    F["execution.rs:1083 sim_params.feature_flags = flags"]
    G["Device SimParams word actually read by the kernels"]

    A -->|"builds its own flag word"| X["OVERWRITTEN before execute"]
    B -->|"builds its own flag word"| X
    C -->|"sets bit per channel force_channels.rs:200, skips read-only :193"| X
    X --> F
    D --> E --> F --> G

    N1["RESOLVED ADR-2060: the GPU-wire-abi divergence bullet claiming no single shared helper was STALE and is now marked resolved. derive_dispatch_feature_flags force_channels.rs:488 is the single helper, called from execution.rs:1069."]
    N2["RESOLVED ADR-2060: Invariant 2 formerly cited a kernel-launch line in execution.rs for the ENABLE_CONSTRAINTS rule - repointed to derive_dispatch_feature_flags at force_channels.rs:488, the ENABLE_CONSTRAINTS bit itself at force_channels.rs:504-506, which is where the rule actually lives."]
    N3["ADR-2029 test module adr_2029_dispatch_authority force_channels.rs:511 states it observes 'the word that is actually uploaded - not the converter's word, which is overwritten before every execute'"]
    N4["Converter words are therefore DEAD for dispatch - a divergence between A and B cannot reach the GPU"]
    N5["DRIFT: docs/GPU-wire-abi.md:134 and :136 give the two converters as lines 236-317 and 319-407 of the models file, the From impls are at project/src/models/simulation_params.rs:779 and :862 at this stamp"]
    A -.- N5
    D -.- N1
    D -.- N2
    D -.- N3
    X -.- N4
```

## VC-11.4 Force-channel registry

```mermaid
classDiagram
    class ForceChannel {
        <<enum>>
        +Repulsion
        +Separation
        +Spring
        +Centering
        +Constraints
        +DagRadialBias
        +Boundary
        +Annealing
        +Gravity
        +ClusterCohesion
        +key() L112
        +feature_flag() L129
        +strength_of() L159
        +set_strength() L216
        +bits_declared_in_visionclaw_domain_simulation_params() L98_L104
        +bit_names_in_src_models_simulation_params() L601_L618
        +state(SimParams) ForceChannelState L149
        +apply(SimParams, ForceChannelState) L185
        +is_read_only() L212
    }
    class FeatureFlags {
        +ENABLE_REPULSION bit0 L98
        +ENABLE_SPRINGS bit1 L99
        +ENABLE_CENTERING bit2 L100
        +ENABLE_TEMPORAL_COHERENCE bit3 L101
        +ENABLE_CONSTRAINTS bit4 L102
        +ENABLE_STRESS_MAJORIZATION bit5 L103
        +ENABLE_SSSP_SPRING_ADJUST bit6 L104
    }
    class ForceDispatchInputs {
        +f32 repel_k
        +f32 spring_k
        +f32 center_gravity_k
        +bool use_sssp_distances
        +bool sssp_spring_adjust_enabled
        +usize num_constraints
    }
    ForceChannel --> FeatureFlags : feature_flag() L129
    ForceDispatchInputs --> FeatureFlags : derive_dispatch_feature_flags() L488

    note for FeatureFlags "ADR-2060: bit3 and bit5 are RESERVED, not a divergence - never set"
    note for ForceChannel "ADR-2029 by design: Constraints read-only L212 - apply() early-returns L193"
    note for ForceChannel "RESOLVED ADR-2060: 180-byte header comment corrected to 212"
    note for FeatureFlags "the layout_mode field declared in device SimParams (visionclaw_unified.cu:93) is likewise dead - never read by any kernel; only dag_bias_k and layer_bias_k are, and each is primed only for its matching layout (Radial / Hierarchical) - see VC-16.5 for the REST-side mode switch"
```

## VC-11.5 Constraint residency — upload and flag consequence

```mermaid
sequenceDiagram
    autonumber
    participant API as constraints_handler<br/>src/handlers/constraints_handler.rs:13
    participant CA as ConstraintActor<br/>constraint_actor.rs:199
    participant OCA as OntologyConstraintActor<br/>ontology_constraint_actor.rs:374
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3354
    participant EX as execution.rs:1069

    API->>CA: UpdateConstraints from POST /constraints/define or /apply (constraints_handler.rs:14-15), handled at constraint_actor.rs:199
    CA->>CA: GetConstraints :221, ClearConstraints :262, GetConstraintStatistics :270
    CA->>FCA: UploadConstraintsToGPU :230
    API->>OCA: ApplyOntologyConstraints (ontology_constraint_actor.rs:374)
    alt GPU context absent
        Note over OCA: debug "GPU not available, constraints cached for next physics step" ontology_constraint_actor.rs:255-257 and cpu_fallback_count += 1 :251,:258
    else GPU ready
        OCA->>FCA: UpdateOntologyConstraintBuffer (force_compute_actor.rs:3476, sent from notify_force_compute_actor ontology_constraint_actor.rs:277)
    end
    OCA->>OCA: ApplyMaterializedAxioms ontology_constraint_actor.rs:445, AdjustConstraintWeights :646
    FCA->>EX: execute with num_constraints = resident count
    alt num_constraints > 0
        EX->>EX: ENABLE_CONSTRAINTS set (force_channels.rs:504-506) and constraint_force_ptr bound execution.rs:258
    else num_constraints == 0
        Note over EX: bit CLEARS - required, or force_pass_kernel keeps walking a buffer that no longer describes anything (force_channels.rs:563-564)
    end
    Note over CA,EX: INVARIANT ADR-2029 - enablement is owned by residency, never by settings
    Note over OCA: class_id, class_charge and class_mass are uploaded as separate per-node device buffers and passed to force_pass_kernel, not carried in SimParams
    Note over FCA: class_id = domain_class_id(source_domain) src/actors/gpu/force_compute_actor.rs:1069<br/>from the shared registry crates/vault-core/src/domains.rs:35 since 805219679 - 0 unknown,<br/>1-9 for the nine DOMAIN_ROOTS (domains.rs:4-15, supply chains and logistics added), legacy short names ai, bc, mv, rb, ngm, tc keep ids 1-6<br/>charge from domain_class_charge: 0.6 known, 1.2 unknown (src/actors/gpu/domain_charge.rs:16-22, :25-27), called at force_compute_actor.rs:1070
    Note over FCA: RESOLVED (ER-25b, commit ad3c3ac58): the former second charge table (0.3 and 2.5 in GPUResourceActor) is gone,<br/>both uploads call the one helper - src/actors/gpu/gpu_resource_actor.rs:225 and force_compute_actor.rs:1070,<br/>so whichever upload_class_metadata call runs last writes the same charges (force_compute_actor.rs:1076, gpu_resource_actor.rs:230)
```

## VC-11.7 Semantic forces on the GPU and stress majorization

```mermaid
sequenceDiagram
    autonumber
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:821
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:2891
    participant EX as UnifiedGPUCompute<br/>execution.rs:1016 enable_semantic
    participant SG as SemanticGpu<br/>src/utils/unified_gpu_compute/semantic.rs:393
    participant SMA as StressMajorizationActor<br/>stress_majorization_actor.rs:311

    Note over PS,SG: SemanticForcesActor was REMOVED (ADR-2140, commit 929fb32c1) - its Configure messages, kernel_bridge stubs and CPU twins are gone, the semantic forces are the physics.semantic settings group
    PS->>FCA: UpdateSimulationParams forwarded (physics_supervisor.rs:821)
    FCA->>FCA: idempotency gate compares the semantic group directly (force_compute_actor.rs:2687-2688)
    FCA->>EX: set_semantic_config(stored group) when it changed (force_compute_actor.rs:2725-2733, execution.rs:1036)
    EX->>SG: set_config, semantic_active = config.any_enabled() (execution.rs:1045)
    PS->>FCA: GetHierarchyLevels (physics_supervisor.rs:967) and RecalculateHierarchy (:991) now go to ForceComputeActor (force_compute_actor.rs:2867, :2891)
    FCA->>FCA: refresh_semantic on every graph upload (force_compute_actor.rs:1257, fn at :4045)
    FCA->>EX: derive node type, DAG level, physicality, role, maturity, cross-domain and edge-type metadata, load PTXModule::SemanticForces, enable_semantic (force_compute_actor.rs:4054-4060)
    PS->>FCA: SetRelationshipTable (physics_supervisor.rs:1040) replaces the ontology relationship table on the device (force_compute_actor.rs:2922)
    alt metadata or PTX or upload fails
        Note over FCA: error "semantic forces UNAVAILABLE for this graph", semantic = None, semantic_active = false (force_compute_actor.rs:4078-4080) - forces are OFF, there is no CPU fallback
    else ready
        EX->>SG: apply(positions, forces) each step after force_pass, before integrate (execution.rs:918-931)
        Note over SG: skipped with a warning if the metadata node count differs from the graph (execution.rs:932-938)
    end
    Note over FCA,SG: DAG layering is the semantic dag force (crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:204, group default OFF at semantic_config.rs:55), distinct from the SimParams radial bias dag_bias_k that force_pass_kernel gates on dag_bias_k > 0 with no feature-flag bit (force_channels.rs:34-35)
    PS->>SMA: TriggerStressMajorization (physics_supervisor.rs:670, forwarded at :686)
    alt no GPU context
        Note over SMA: handler returns Err "GPU not initialized" (stress_majorization_actor.rs:321-323), perform_stress_majorization (:71) returns Err "GPU not available for stress majorization" (:95) - hard failure, no CPU degradation
    else GPU ready
        SMA->>SMA: perform_stress_majorization runs entirely on the GPU, CheckStressMajorization :365, UpdateStressMajorizationParams :343
    end
    Note over SMA,FCA: DEBT: ForceComputeActor keeps TriggerStressMajorization, GetStressMajorizationStats and ResetStressMajorizationSafety handlers that only return Err pointing at StressMajorizationActor (force_compute_actor.rs:3395-3402, :3407-3416, :3423-3432)
    Note over SMA,FCA: DIVERGENCE bit5 ENABLE_STRESS_MAJORIZATION is declared but never set by derive_dispatch_feature_flags - stress majorization is not a GPU force channel
    Note over SMA,FCA: Stress majorization params live on CPU in SemanticProcessorActor and are absent from GPU SimParams - project/src/models/simulation_params.rs:77
    Note over SMA,FCA: DIVERGENCE bit3 ENABLE_TEMPORAL_COHERENCE is likewise declared but never set - both are reserved bits, not wired force terms
```

## VC-11.9 Warmup, settle and reheat cycle

```mermaid
stateDiagram-v2
    [*] --> Warmup
    Warmup: Stability warmup - stability_warmup_remaining decremented each tick at line 1887
    Active: Integrating - mean per-node KE at or above SETTLE_KE_EPSILON
    Settling: Consecutive sub-epsilon ticks accumulating in settle_stable_frames
    Settled: is_settled true - reported via GetSettlementState line 3261
    Reheated: reheat_factor greater than 0 - velocity perturbation injected

    Warmup --> Active: warmup frames exhausted
    Active --> Settling: next_stable_frames increments when mean_ke finite and below epsilon line 286-287
    Settling --> Active: KE at or above epsilon resets the run to 0 line 288-289
    Settling --> Settled: settle_stable_frames reaches SETTLE_FRAME_THRESHOLD
    Active --> Settled: paused latch - orchestrator declared an energy plateau line 305-307
    Settled --> Reheated: settings change, pin, or reset raises reheat_factor
    Reheated --> Active: decay to zero
    Reheated --> Reheated: reheat_factor multiplied by 0.997 each step line 1992

    note right of Warmup
        warmup = 1200 frames when edge_count == 0
        else 600 frames, reset after graph upload
        force_compute_actor.rs:1303
    end note
    note right of Settled
        SETTLE_KE_EPSILON = 1e-4 line 274
        SETTLE_FRAME_THRESHOLD = 180 line 279
        180 frames is about 3 s at 60 fps and mirrors
        the auto-balance stabilityFrameCount config
        so REST and auto-balance agree on settled
    end note
    note right of Reheated
        inject_velocity_perturbation(reheat_factor) line 1937 when factor > 0 line 1932
        decay 0.997 per step line 1992, about 1300 steps from 1.0 to the floor, roughly 21 s at 60 fps
        snapped to 0.0 once below 0.02 line 1993-1994
        seeds - SetLayoutMode max 1.5 line 2581 with warmup raised to 900 line 2582
        UpdateSimulationParams log-ratio scaled 1.0 to 5.0 line 2806 with warmup 1800 line 2791
        PinNodePositions max 0.3 line 3022
        ResetPositions 1.0 line 3640 with warmup 600 line 3639
    end note
    note right of Active
        Every good frame is display-projected and published
        to the graph service as UpdateNodePositions line 2265-2276
        pacing is the client coordinator's job (ADR-2141)
        the 300 gate now only thins log lines line 2094, 2252, 2320
        skipped frames logged every 300th line 1771, 1813
    end note
```


## VC-11.10 Hard bounds and connected extent

```mermaid
flowchart TD
    All["All-node AABB reduction<br/>compute_aabb_reduction_kernel<br/>visionclaw_unified.cu:2539"] --> Grid["Spatial index retains isolated nodes"]
    Connected["Connected-only reduction excludes zero degree<br/>compute_connected_aabb_reduction_kernel<br/>visionclaw_unified.cu:2600"] --> Extent["connected_extent<br/>execution.rs:132"]
    Gate["degree_weighted_gravity_kernel launch gate<br/>execution.rs:872"] --> Branch{"viewport_bounds greater than 0"}
    Branch -->|yes| Bounded["extent = full all-node Aabb<br/>execution.rs:881"]
    Branch -->|no, bounds disabled| Extent
    Extent --> Call["called at execution.rs:883 - a zero-extent Aabb when None"]
    Bounded --> Shell["peripheral_shell_radius<br/>execution.rs:59"]
    Call --> Shell
    Shell --> ShellNote["Bounded - fixed fraction of viewport_bounds<br/>Unbounded - connected-extent heuristic, capped"]
    Force["Integration advances free nodes"] --> Clamp["Clamp each axis to viewport bounds<br/>Zero only outward velocity"]
    Pin["Pinned nodes retain host position"] -.-> Clamp
    Test["Actual CUDA fixture tests all axes and both AABBs<br/>integrate_bounds.cu:7"] -.-> Clamp
    Test -.-> Connected
```
