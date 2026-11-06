---
id: VC-11
title: Physics step and force channels
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2007, ADR-2028, ADR-2029, ADR-2055, ADR-2060, ADR-2135, ADR-2140, ADR-2141, ADR-2181]
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
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/actors/gpu/gpu_resource_actor.rs
  - ../project/crates/vault-core/src/domains.rs
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/crates/visionclaw-domain/src/models/simulation_params.rs
  - ../project/client/src/api/settings/defaults.ts
  - ../project/client/src/features/control-center/registry/groups/motion.ts
  - ../project/client/src/features/control-center/registry/settings-manifest.json
  - ../project/docs/adr/ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md
  - ../project/src/actors/gpu/physics_bounds.rs
  - ../project/src/settings/api/settings_routes.rs
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
---

## For developers

One physics step runs ComputeForces through ForceComputeActor: SimParams, the feature-flag word, the kernel launch, integration, and the warm-up, settle and reheat cycle. The force channels are the registry of named forces that settings and constraints switch on. Read this before changing a force, a flag or the settle behaviour.

## For the business

This is the engine that lays the knowledge graph out on screen. Its settle and reheat cycle decides how quickly a graph comes to rest after a change, and how much it moves when someone drags a node.

## VC-11.1 Physics tick — phase 1, params and flag word

```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:639
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:1779
    participant EX as UnifiedGPUCompute::execute_physics_step_with_bypass<br/>src/utils/unified_gpu_compute/execution.rs:1128
    participant FC as derive_dispatch_feature_flags<br/>src/models/force_channels.rs:488

    PO->>FCA: ComputeForces do_send (physics_orchestrator_actor.rs:587), the step loop holds the ForceComputeActor address directly
    Note over PS: the supervisor forwarder is a second route to the same actor<br/>GPUManagerActor (src/actors/gpu/gpu_manager_actor.rs:335) to PhysicsSupervisor (physics_supervisor.rs:639) to ForceComputeActor (:654)<br/>the orchestrator does not use it
    alt shared_context is None
        Note over FCA: no GPU context - initialize_own_gpu_context is retried (force_compute_actor.rs:1840), if still none the frame is skipped,<br/>an error is logged every 300th skip (:1846), the orchestrator is told it was a benign skip and Err is returned (:1855)
    else GPU ready
        FCA->>FCA: holds.expire(HOLD_EXPIRY 5 s) releases a hold nobody refreshed :1920-1933 (const at :4540)
        FCA->>FCA: step_overrides(warmup, any_held, settle window, damping, global_speed, reheat) :1934-1942
        Note over FCA: step_overrides (force_compute_actor.rs:4588) - bypass_stability = warmup > 0 OR any node held OR inside the<br/>2 s release settle window (:4599), global_speed floored at HOLD_GLOBAL_SPEED 2.0 while held or settling (:4600-4604),<br/>reheat withheld while held or settling (:4606), elastic overlay on for hold plus settle, hold springs for the hold only (:4607-4608)
        FCA->>FCA: stability_warmup_remaining -= 1 when above 0 :1949-1951, reheat = overrides.reheat :1952
        Note over FCA: warmup reset on graph upload - 1200 frames when edge_count == 0 else 600 :1322
        FCA->>EX: set_hold_overlay and set_hold_springs :2015-2016 then execute_physics_step_with_bypass(sim_params, stability_bypass) :2017<br/>from spawn_blocking, after inject_velocity_perturbation for the reheat (:2003) and for a one-shot release impulse (:2008)
        EX->>FC: derive_dispatch_feature_flags(ForceDispatchInputs) :1141
        FC-->>EX: flags word
        EX->>EX: sim_params.feature_flags = feature_flags :1155
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
    participant K1 as force_pass_kernel<br/>crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:460
    participant SG as SemanticGpu::apply<br/>src/utils/unified_gpu_compute/semantic.rs:393
    participant K2 as integrate_pass_kernel<br/>crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:978
    participant EV as cust Event<br/>execution.rs completion poll

    opt num_constraints > 0 (execution.rs:259)
        EX->>EX: bind constraint_force_ptr and constraint_force_epsilon :259
    end
    opt elastic hold overlay springs active (execution.rs:718)
        EX->>EX: snapshot_edge_rest_kernel captures every edge length once at grab (:719-735), bind_edge_rest (:741)
    end
    Note over EX: force_pass_with_stability_kernel is chosen when stability_threshold > 0, else force_pass_kernel (execution.rs:661-665)
    EX->>K1: pos, force, spatial grid, CSR edges, SSSP distances, constraint_data and num_constraints,<br/>node degrees, spring_scale, compute mask, node_rank and node_plane (execution.rs:746-822)
    EX->>K1: class_id, class_charge, class_mass on the non-stability variant only (execution.rs:808-810)
    Note over EX: DEBT: the UploadPositions handler (force_compute_actor.rs:2929) still uploads positions via spawn_blocking and update_positions_only,<br/>but nothing in src, crates or tests sends UploadPositions (src/actors/messages/physics_messages.rs:152 declares it) - a live drag arrives as<br/>HoldNodes (force_compute_actor.rs:3027), an explicit pin as PinNodePositions (force_compute_actor.rs:2993)
    K1->>SG: forces written (cluster cohesion and degree-weighted gravity kernels run between, execution.rs:883-953)
    Note over SG: when semantic_active and the metadata matches the graph node count, the semantic kernels ADD their forces to the live force buffers (execution.rs:960-974),<br/>otherwise they are skipped with a warning (execution.rs:975-983). Applied exactly as stored, no scale or cap (ADR-2140)
    SG->>K2: forces including semantic terms
    EX->>K2: pos, vel, force, mass, class metadata, prev_force_x/y/z for FA2 swing and traction, pinned_mask (execution.rs:999-1027)
    Note over K2: pinned and held nodes SKIP integration but still exert forces on neighbours (execution.rs:1025-1027)
    Note over K2: integrate_pass adds boundary push (viewport_bounds + boundary_damping) and annealing jitter (temperature + cooling_rate) per force_channels.rs:38-39
    EX->>EV: completion_event.record(stream) (execution.rs:1033-1034)
    loop poll until EventStatus::Ready
        EX->>EV: completion_event.query()
        alt poll_start.elapsed() > 10s
            Note over EX: returns Err "GPU kernel execution timed out after 10s" (execution.rs:1043)
        else not ready
            EX->>EX: std::thread::yield_now()
        end
    end
    EX->>EX: swap_buffers() (execution.rs:1048) then iteration += 1
    opt iteration % 100 == 0
        Note over EX: logs memory MB, utilisation pct, grid occupancy pct, resize count
    end
    Note over K1,K2: Gravity and cluster cohesion run as SEPARATE kernels - degree_weighted_gravity_kernel and cluster_cohesion_kernel (force_channels.rs:42-43)
```

## VC-11.3 Feature-flag derivation across the conversion paths

```mermaid
flowchart TD
    A["Path A converter - From SimulationParams<br/>project/src/models/simulation_params.rs:779-792"]
    B["Path B converter - From PhysicsSettings<br/>project/src/models/simulation_params.rs:865-867<br/>now delegates to Path A via SimulationParams::from"]
    C["Registry mutator ForceChannel::apply<br/>src/models/force_channels.rs:185-200"]
    D["derive_dispatch_feature_flags<br/>src/models/force_channels.rs:488"]
    E["execution.rs:1141 calls the helper"]
    F["execution.rs:1155 sim_params.feature_flags = flags"]
    G["Device SimParams word actually read by the kernels"]

    A -->|"builds its own flag word"| X["OVERWRITTEN before execute"]
    B -->|"same word, built by Path A"| A
    C -->|"sets bit per channel force_channels.rs:200, skips read-only :193"| X
    X --> F
    D --> E --> F --> G

    N1["RESOLVED ADR-2060: the GPU-wire-abi divergence bullet claiming no single shared helper was STALE and is now marked resolved. derive_dispatch_feature_flags force_channels.rs:488 is the single helper, called from execution.rs:1141."]
    N2["RESOLVED ADR-2060: Invariant 2 formerly cited a kernel-launch line in execution.rs for the ENABLE_CONSTRAINTS rule - repointed to derive_dispatch_feature_flags at force_channels.rs:488, the ENABLE_CONSTRAINTS bit itself at force_channels.rs:504-506, which is where the rule actually lives."]
    N3["ADR-2029 test module adr_2029_dispatch_authority force_channels.rs:511 states it observes 'the word that is actually uploaded - not the converter's word, which is overwritten before every execute'"]
    N4["Converter words are therefore DEAD for dispatch - a divergence between A and B cannot reach the GPU"]
    N5["DRIFT: docs/GPU-wire-abi.md:157 and :159 give the two converters as lines 236-317 and 319-407 of the models file and describe a separate flag derivation for Path B, the From impls are at project/src/models/simulation_params.rs:779 and :865 at this stamp and Path B no longer derives flags of its own"]
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
        +bits_declared_in_visionclaw_domain_simulation_params() L119_L125
        +bit_names_in_src_models_simulation_params() L601_L618
        +state(SimParams) ForceChannelState L149
        +apply(SimParams, ForceChannelState) L185
        +is_read_only() L212
    }
    class FeatureFlags {
        +ENABLE_REPULSION bit0 L119
        +ENABLE_SPRINGS bit1 L120
        +ENABLE_CENTERING bit2 L121
        +ENABLE_TEMPORAL_COHERENCE bit3 L122
        +ENABLE_CONSTRAINTS bit4 L123
        +ENABLE_STRESS_MAJORIZATION bit5 L124
        +ENABLE_SSSP_SPRING_ADJUST bit6 L125
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
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3416
    participant EX as execution.rs:1141

    API->>CA: UpdateConstraints from POST /constraints/define or /apply (constraints_handler.rs:14-15), handled at constraint_actor.rs:199
    CA->>CA: GetConstraints :221, ClearConstraints :262, GetConstraintStatistics :270
    CA->>FCA: UploadConstraintsToGPU :230
    API->>OCA: ApplyOntologyConstraints (ontology_constraint_actor.rs:374)
    alt GPU context absent
        Note over OCA: debug "GPU not available, constraints cached for next physics step" ontology_constraint_actor.rs:255-257 and cpu_fallback_count += 1 :251,:258
    else GPU ready
        OCA->>FCA: UpdateOntologyConstraintBuffer (force_compute_actor.rs:3538, sent from notify_force_compute_actor ontology_constraint_actor.rs:277)
    end
    OCA->>OCA: ApplyMaterializedAxioms ontology_constraint_actor.rs:445, AdjustConstraintWeights :646
    Note over FCA: both writers only replace the cache and mark it dirty, replace_constraint_cache (force_compute_actor.rs:1657-1662).<br/>sync_constraints uploads before a step only when dirty or the resident count differs from the cache (force_compute_actor.rs:1675-1689),<br/>so an unchanged set costs no upload, and a new GPU context re-marks it dirty (force_compute_actor.rs:3574)
    FCA->>EX: execute with num_constraints = resident count
    alt num_constraints > 0
        EX->>EX: ENABLE_CONSTRAINTS set (force_channels.rs:504-506) and constraint_force_ptr bound execution.rs:259
    else num_constraints == 0
        Note over EX: bit CLEARS - required, or force_pass_kernel keeps walking a buffer that no longer describes anything (force_channels.rs:563-564)
    end
    Note over CA,EX: INVARIANT ADR-2029 - enablement is owned by residency, never by settings
    Note over OCA: class_id, class_charge and class_mass are uploaded as separate per-node device buffers and passed to the non-stability force_pass_kernel variant and integrate_pass_kernel (execution.rs:808-810, :1018-1020), not carried in SimParams
    Note over FCA: class_id = domain_class_id(source_domain) src/actors/gpu/force_compute_actor.rs:1088<br/>from the shared registry crates/vault-core/src/domains.rs:35 since 805219679 - 0 unknown,<br/>1-9 for the nine DOMAIN_ROOTS (domains.rs:4-15, supply chains and logistics added), legacy short names ai, bc, mv, rb, ngm, tc keep ids 1-6<br/>charge from domain_class_charge: 0.6 known, 1.2 unknown (src/actors/gpu/domain_charge.rs:16-22, :25-27), called at force_compute_actor.rs:1089
    Note over FCA: RESOLVED (ER-25b, commit ad3c3ac58): the former second charge table (0.3 and 2.5 in GPUResourceActor) is gone,<br/>both uploads call the one helper - src/actors/gpu/gpu_resource_actor.rs:225 and force_compute_actor.rs:1089,<br/>so whichever upload_class_metadata call runs last writes the same charges (force_compute_actor.rs:1095, gpu_resource_actor.rs:230)
```

## VC-11.7 Semantic forces on the GPU and stress majorization

```mermaid
sequenceDiagram
    autonumber
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:820
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:2875
    participant EX as UnifiedGPUCompute<br/>execution.rs:1064 enable_semantic
    participant SG as SemanticGpu<br/>src/utils/unified_gpu_compute/semantic.rs:393
    participant SMA as StressMajorizationActor<br/>stress_majorization_actor.rs:311

    Note over PS,SG: SemanticForcesActor was REMOVED (ADR-2140, commit 929fb32c1) - its Configure messages, kernel_bridge stubs and CPU twins are gone, the semantic forces are the physics.semantic settings group
    PS->>FCA: UpdateSimulationParams forwarded (physics_supervisor.rs:820)
    FCA->>FCA: idempotency gate compares the semantic group directly (force_compute_actor.rs:4701-4702)
    FCA->>EX: set_semantic_config(stored group) when it changed (force_compute_actor.rs:2709-2717, execution.rs:1084)
    EX->>SG: set_config, semantic_active = config.any_enabled() (execution.rs:1093)
    PS->>FCA: GetHierarchyLevels (physics_supervisor.rs:966) and RecalculateHierarchy (:990) now go to ForceComputeActor (force_compute_actor.rs:2851, :2875)
    FCA->>FCA: refresh_semantic on every graph upload (force_compute_actor.rs:1276, fn at :4773)
    FCA->>EX: derive node type, DAG level, physicality, role, maturity, cross-domain and edge-type metadata, load PTXModule::SemanticForces, enable_semantic (force_compute_actor.rs:4782-4788)
    PS->>FCA: SetRelationshipTable (physics_supervisor.rs:1039) replaces the ontology relationship table on the device (force_compute_actor.rs:2906)
    alt metadata or PTX or upload fails
        Note over FCA: error "semantic forces UNAVAILABLE for this graph", semantic = None, semantic_active = false (force_compute_actor.rs:4806-4808) - forces are OFF, there is no CPU fallback
    else ready
        EX->>SG: apply(positions, forces) each step after force_pass, before integrate (execution.rs:960-974)
        Note over SG: skipped with a warning if the metadata node count differs from the graph (execution.rs:975-983)
    end
    Note over FCA,SG: DAG layering is the semantic dag force (crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:204, group default OFF at semantic_config.rs:55), distinct from the SimParams radial bias dag_bias_k that force_pass_kernel gates on dag_bias_k > 0 with no feature-flag bit (force_channels.rs:34-35)
    PS->>SMA: TriggerStressMajorization (physics_supervisor.rs:669, forwarded at :685)
    alt no GPU context
        Note over SMA: handler returns Err "GPU not initialized" (stress_majorization_actor.rs:321-323), perform_stress_majorization (:71) returns Err "GPU not available for stress majorization" (:95) - hard failure, no CPU degradation
    else GPU ready
        SMA->>SMA: perform_stress_majorization runs entirely on the GPU, CheckStressMajorization :365, UpdateStressMajorizationParams :343
    end
    Note over SMA,FCA: DEBT: ForceComputeActor keeps TriggerStressMajorization, GetStressMajorizationStats and ResetStressMajorizationSafety handlers that only return Err pointing at StressMajorizationActor (force_compute_actor.rs:3457-3464, :3469-3478, :3485-3494)
    Note over SMA,FCA: DIVERGENCE bit5 ENABLE_STRESS_MAJORIZATION is declared but never set by derive_dispatch_feature_flags - stress majorization is not a GPU force channel
    Note over SMA,FCA: Stress majorization params live on CPU in SemanticProcessorActor and are absent from GPU SimParams - project/src/models/simulation_params.rs:77
    Note over SMA,FCA: DIVERGENCE bit3 ENABLE_TEMPORAL_COHERENCE is likewise declared but never set - both are reserved bits, not wired force terms
```

## VC-11.9 Warmup, settle and reheat cycle

```mermaid
stateDiagram-v2
    [*] --> Warmup
    Warmup: Stability warmup - decremented each tick, force_compute_actor.rs:1949-1951
    Active: Integrating - mean per-node KE at or above SETTLE_KE_EPSILON
    Settling: Consecutive sub-epsilon ticks accumulating in settle_stable_frames
    Settled: is_settled true - reported via GetSettlementState force_compute_actor.rs:3331
    Reheated: reheat_factor greater than 0 - velocity perturbation injected
    Held: a node is held - stability bypassed, speed floor, no reheat, force_compute_actor.rs:4599-4606

    Warmup --> Active: warmup frames exhausted
    Active --> Settling: next_stable_frames increments when mean_ke finite and below epsilon force_compute_actor.rs:303-304
    Settling --> Active: KE at or above epsilon resets the run to 0 force_compute_actor.rs:305-306
    Settling --> Settled: settle_stable_frames reaches SETTLE_FRAME_THRESHOLD
    Active --> Settled: paused latch - orchestrator declared an energy plateau force_compute_actor.rs:323-324
    Settled --> Reheated: settings change, radial re-key or position reset raises reheat_factor
    Reheated --> Active: decay to zero
    Reheated --> Reheated: reheat_factor multiplied by 0.997 each step force_compute_actor.rs:2065
    Active --> Held: HoldNodes clears reheat_factor and any pending impulse force_compute_actor.rs:3041-3042
    Held --> Active: last release opens the 2 s settle window and queues a one-shot impulse force_compute_actor.rs:381, :390

    note right of Warmup
        warmup = 1200 frames when edge_count == 0
        else 600 frames, reset after graph upload
        force_compute_actor.rs:1322
    end note
    note right of Settled
        SETTLE_KE_EPSILON = 1e-4 force_compute_actor.rs:291
        SETTLE_FRAME_THRESHOLD = 180 force_compute_actor.rs:296
        180 frames is about 3 s at 60 fps and mirrors
        the auto-balance stabilityFrameCount config
        so REST and auto-balance agree on settled
    end note
    note right of Reheated
        inject_velocity_perturbation(reheat_factor) force_compute_actor.rs:2003 when factor > 0 :1996
        decay 0.997 per step :2065, about 1300 steps from 1.0 to the floor, roughly 21 s at 60 fps
        snapped to 0.0 once below 0.02 :2066-2067
        seeds - SetRadialLayout max 1.5 :2602 with warmup raised to 900 :2603
        UpdateSimulationParams log-ratio scaled 1.0 to 5.0 :2790 with warmup 1800 :2775
        a damping-only change applies without reheat or warmup :2664-2670
        PinNodePositions no longer reheats - msg.reheat queues RELEASE_IMPULSE 0.05 once :3013-3014
        ResetPositions 1.0 :3703 with warmup 600 :3702
    end note
    note right of Held
        step_overrides force_compute_actor.rs:4588 decides each tick
        HOLD_GLOBAL_SPEED 2.0 floor :4552, RELEASE_SETTLE_WINDOW 2 s :4536
        a hold not refreshed for HOLD_EXPIRY 5 s is released by the actor :4540, :1920-1933
    end note
    note right of Active
        Every good frame is display-projected and published
        to the graph service as UpdateNodePositions force_compute_actor.rs:2338-2350
        pacing is the client coordinator's job (ADR-2141)
        the 300 gate now only thins log lines :2167, :2325, :2379
        skipped frames logged every 300th :1804, :1846
    end note
```


## VC-11.10 Hard bounds and connected extent

```mermaid
flowchart TD
    All["All-node AABB reduction<br/>compute_aabb_reduction_kernel<br/>visionclaw_unified.cu:2563"] --> Grid["Spatial index retains isolated nodes"]
    Connected["Connected-only reduction excludes zero degree<br/>compute_connected_aabb_reduction_kernel<br/>visionclaw_unified.cu:2624"] --> Extent["connected_extent<br/>execution.rs:132"]
    Gate["degree_weighted_gravity_kernel launch gate<br/>execution.rs:914"] --> Branch{"viewport_bounds greater than 0"}
    Branch -->|yes| Bounded["extent = full all-node Aabb<br/>execution.rs:923"]
    Branch -->|no, bounds disabled| Extent
    Extent --> Call["called at execution.rs:925 - a zero-extent Aabb when None"]
    Bounded --> Shell["peripheral_shell_radius<br/>execution.rs:59"]
    Call --> Shell
    Shell --> ShellNote["Bounded - fixed fraction of viewport_bounds<br/>Unbounded - connected-extent heuristic, capped"]
    Force["Integration advances free nodes"] --> Clamp["Clamp each axis to viewport bounds<br/>Zero only outward velocity"]
    Pin["Pinned nodes retain host position"] -.-> Clamp
    Test["Actual CUDA fixture tests all axes and both AABBs<br/>integrate_bounds.cu:7"] -.-> Clamp
    Test -.-> Connected
```

## VC-11.11 boundsSize: one default in three copies, and the overview slab it caused

```mermaid
flowchart TB
    subgraph RUST["Server default, crates/visionclaw-domain"]
        R1["default_bounds_size returns 1600.0<br/>crates/visionclaw-domain/src/types/physics_config.rs:113-115<br/>serde default for a missing key, physics_config.rs:313-314"]
        R2["PhysicsSettings default bounds_size 1600.0<br/>physics_config.rs:549"]
        R3["test asserts 1600.0<br/>physics_config.rs:832"]
        R1 --> R2 --> R3
    end
    subgraph TS["Client copies"]
        C1["client defaults boundsSize 1600.0<br/>client/src/api/settings/defaults.ts:57"]
        C2["Motion slider, min 100, max 4000, description names default 1600<br/>client/src/features/control-center/registry/groups/motion.ts:41"]
        C3["settings manifest, max 4000<br/>client/src/features/control-center/registry/settings-manifest.json:2144, :2156"]
        C2 --> C3
    end
    RUST --> VB["viewport_bounds = bounds_size, the soft-cube half-extent the integrator clamps to, see VC-11.10<br/>../project/crates/visionclaw-domain/src/models/simulation_params.rs:517"]
    TS --> VB
    VB --> SLAB["RESOLVED 2026-11-06 b11d8cba1: at 400 a 10k-node graph was crushed into the cube,<br/>p5 and p95 on y pinned at plus or minus 400, and read as an overview edge slab<br/>docs/adr/ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md:65"]
    SLAB --> INV["INVARIANT a slab at overview zoom means a boundary-pinned layout, check the position quantiles before touching edge alpha<br/>ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md:65"]
    C2 --> VAL["server range check bounds_size 100 to 2000 in apply_physics_patch<br/>src/settings/api/settings_routes.rs:617, :262-268, :664-669<br/>src/actors/gpu/physics_bounds.rs:73-74"]
    VAL --> T1["TENSION the slider offers 2050 to 4000, which the server refuses with Validation failed"]
    R2 --> D1["DRIFT the comment beside the new default still says the graph sits well inside the 400u cube<br/>physics_config.rs:544-548, and BOUNDS_SIZE says canonical default 400.0, physics_bounds.rs:73"]
```

**What it shows:** the 2026-11-06 change of the soft-cube default from 400 to 1600, made in the Rust default, the client defaults and the Motion slider, and where the server's range check disagrees with the slider.

**Debt (parallel):** the default lives in the Rust crate and twice in the client (`crates/visionclaw-domain/src/types/physics_config.rs:113-115`, `client/src/api/settings/defaults.ts:57`, `client/src/features/control-center/registry/groups/motion.ts:41`), and only the Rust copy is pinned by a test (`crates/visionclaw-domain/src/types/physics_config.rs:832`).

**Open (risk):** the new default only fills a missing key (`crates/visionclaw-domain/src/types/physics_config.rs:313`), so a store whose `physics` row already holds 400 presumably keeps the crushed layout; not checked against the live store.

**Tension (risk):** the slider's maximum of 4000 (`client/src/features/control-center/registry/groups/motion.ts:41`) exceeds the server's `BOUNDS_SIZE` ceiling of 2000 (`src/actors/gpu/physics_bounds.rs:74`), which `validate_physics_settings` enforces on the physics patch path (`src/settings/api/settings_routes.rs:262-268`, `:664-669`), so the top half of the slider is refused there.
