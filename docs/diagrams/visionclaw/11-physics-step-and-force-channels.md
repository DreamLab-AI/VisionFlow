---
id: VC-11
title: Physics step and force channels
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2007, ADR-2028, ADR-2029, ADR-2055, ADR-2060]
sources:
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/physics_supervisor.rs
  - ../project/src/actors/gpu/constraint_actor.rs
  - ../project/src/actors/gpu/ontology_constraint_actor.rs
  - ../project/src/actors/gpu/semantic_forces_actor.rs
  - ../project/src/actors/gpu/stress_majorization_actor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/models/force_channels.rs
  - ../project/src/models/simulation_params.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/tests/gpu/integrate_bounds.cu
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/crates/visionclaw-domain/src/models/simulation_params.rs
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/utils/visionflow_unified.ptx
  - ../project/src/actors/gpu/gpu_resource_actor.rs
  - ../project/crates/vault-core/src/domains.rs
verified_commit: 7d3ea2edb067432a57e6fe1fd951fd8254380bb8
---

## VC-11.1 Physics tick — phase 1, params and flag word

```mermaid
sequenceDiagram
    autonumber
    participant PO as PhysicsOrchestratorActor<br/>src/actors/physics_orchestrator_actor.rs
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:628
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:1860
    participant EX as UnifiedGPUCompute::execute<br/>src/utils/unified_gpu_compute/execution.rs:172
    participant FC as derive_dispatch_feature_flags<br/>src/models/force_channels.rs:486

    PO->>PS: ComputeForces
    PS->>FCA: ComputeForces forwarded :628
    alt shared_context is None
        Note over FCA: no GPU context - frame skipped, logged every 300th skip :1885,:1927
    else GPU ready
        FCA->>FCA: stability_warmup_remaining -= 1 :2000
        Note over FCA: warmup reset on graph upload - 1200 frames when edge_count == 0 else 600 :1417
        FCA->>FCA: reheat_factor read :2002
        FCA->>EX: execute(sim_params) with num_constraints
        EX->>FC: derive_dispatch_feature_flags(ForceDispatchInputs) :1003
        FC-->>EX: flags word
        EX->>EX: sim_params.feature_flags = feature_flags :1017
        Note over EX: OVERWRITE - the converter's own flag word is discarded before every execute, so only this word reaches the device
    end
    Note over FC: repel_k > 0.0 sets ENABLE_REPULSION :488-490
    Note over FC: spring_k > 0.0 sets ENABLE_SPRINGS :491-493
    Note over FC: center_gravity_k > 0.0 sets ENABLE_CENTERING :494-496
    Note over FC: use_sssp_distances OR sssp_spring_adjust_enabled sets ENABLE_SSSP_SPRING_ADJUST :497-499
    Note over FC: num_constraints > 0 sets ENABLE_CONSTRAINTS (force_channels.rs:502-504) - KEYSTONE, never from user settings
    Note over EX,FC: INVARIANT ADR-2029 - ENABLE_CONSTRAINTS is residency-derived at step time, never a settings toggle
```

## VC-11.2 Physics tick — phase 2, kernel launch and integrate

```mermaid
sequenceDiagram
    autonumber
    participant EX as UnifiedGPUCompute::execute<br/>execution.rs:172
    participant K1 as force_pass_kernel<br/>src/utils/visionflow_unified.ptx
    participant K2 as integrate_pass_kernel<br/>visionflow_unified.ptx
    participant EV as cust Event<br/>execution.rs completion poll

    opt num_constraints > 0 (execution.rs:262)
        EX->>EX: bind constraint_force_ptr and constraint_force_epsilon :262
    end
    EX->>K1: launch with pos/vel/force buffers, mass, num_nodes
    EX->>K1: class_id, class_charge, class_mass device pointers (ontology metadata)
    EX->>K1: prev_force_x/y/z for FA2 swing and traction adaptive speed
    EX->>K1: pinned_mask device pointer
    Note over K1: pinned nodes SKIP integration but still exert forces on neighbours
    Note over EX: UploadPositions handler (force_compute_actor.rs:3296) re-uploads a dragged node's new position directly via spawn_blocking, without blocking the Tokio runtime - distinct from PinNodePositions
    K1->>K2: forces written
    Note over K2: integrate_pass adds boundary push (viewport_bounds + boundary_damping) and annealing jitter (temperature + cooling_rate) per force_channels.rs:37-38
    EX->>EV: completion_event.record(stream)
    loop poll until EventStatus::Ready
        EX->>EV: completion_event.query()
        alt poll_start.elapsed() > 10s
            Note over EX: returns Err "GPU kernel execution timed out after 10s"
        else not ready
            EX->>EX: std::thread::yield_now()
        end
    end
    EX->>EX: swap_buffers() then iteration += 1
    opt iteration % 100 == 0
        Note over EX: logs memory MB, utilisation pct, grid occupancy pct, resize count
    end
    Note over K1,K2: Gravity and cluster cohesion run as SEPARATE kernels - degree_weighted_gravity_kernel and cluster_cohesion_kernel (force_channels.rs:40-41)
```

## VC-11.3 Feature-flag derivation across the conversion paths

```mermaid
flowchart TD
    A["Path A converter - From SimulationParams<br/>project/src/models/simulation_params.rs:776-789"]
    B["Path B converter - From PhysicsSettings<br/>project/src/models/simulation_params.rs:859-872"]
    C["Registry mutator ForceChannel::apply<br/>src/models/force_channels.rs:183-198"]
    D["derive_dispatch_feature_flags<br/>src/models/force_channels.rs:486"]
    E["execution.rs:1003 calls the helper"]
    F["execution.rs:1017 sim_params.feature_flags = flags"]
    G["Device SimParams word actually read by the kernels"]

    A -->|"builds its own flag word"| X["OVERWRITTEN before execute"]
    B -->|"builds its own flag word"| X
    C -->|"sets bit per channel :198, skips read-only :191"| X
    X --> F
    D --> E --> F --> G

    N1["RESOLVED ADR-2060: the GPU-wire-abi divergence bullet claiming no single shared helper was STALE and is now marked resolved. derive_dispatch_feature_flags force_channels.rs:486 is the single helper, called from execution.rs:1003."]
    N2["RESOLVED ADR-2060: Invariant 2 formerly cited a kernel-launch line in execution.rs for the ENABLE_CONSTRAINTS rule - repointed to derive_dispatch_feature_flags at force_channels.rs:486, the ENABLE_CONSTRAINTS bit itself at force_channels.rs:502-504, which is where the rule actually lives."]
    N3["ADR-2029 test module adr_2029_dispatch_authority force_channels.rs:509 states it observes 'the word that is actually uploaded - not the converter's word, which is overwritten before every execute'"]
    N4["Converter words are therefore DEAD for dispatch - a divergence between A and B cannot reach the GPU"]
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
        +key() L110
        +feature_flag() L127
        +strength_of() L165
        +set_strength() L213
        +bits_declared_in_visionclaw_domain_simulation_params() L113_L119
        +bit_names_in_src_models_simulation_params() L599_L613
        +state(SimParams) ForceChannelState L147
        +apply(SimParams, ForceChannelState) L183
        +is_read_only() L210
    }
    class FeatureFlags {
        +ENABLE_REPULSION bit0 L113
        +ENABLE_SPRINGS bit1 L114
        +ENABLE_CENTERING bit2 L115
        +ENABLE_TEMPORAL_COHERENCE bit3 L116
        +ENABLE_CONSTRAINTS bit4 L117
        +ENABLE_STRESS_MAJORIZATION bit5 L118
        +ENABLE_SSSP_SPRING_ADJUST bit6 L119
    }
    class ForceDispatchInputs {
        +f32 repel_k
        +f32 spring_k
        +f32 center_gravity_k
        +bool use_sssp_distances
        +bool sssp_spring_adjust_enabled
        +usize num_constraints
    }
    ForceChannel --> FeatureFlags : feature_flag() L127
    ForceDispatchInputs --> FeatureFlags : derive_dispatch_feature_flags() L486

    note for FeatureFlags "ADR-2060: bit3 and bit5 are RESERVED, not a divergence - never set"
    note for ForceChannel "ADR-2029 by design: Constraints read-only L210 - apply() early-returns L191"
    note for ForceChannel "RESOLVED ADR-2060: 180-byte header comment corrected to 212"
    note for FeatureFlags "the layout_mode field declared in device SimParams (visionclaw_unified.cu:92) is likewise dead - never read by any kernel; only dag_bias_k and layer_bias_k are, and each is primed only for its matching layout (Radial / Hierarchical) - see VC-16.5 for the REST-side mode switch"
```

## VC-11.5 Constraint residency — upload and flag consequence

```mermaid
sequenceDiagram
    autonumber
    participant API as constraints_handler<br/>src/handlers/constraints_handler.rs:13
    participant CA as ConstraintActor<br/>constraint_actor.rs:193
    participant OCA as OntologyConstraintActor<br/>ontology_constraint_actor.rs:451
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3693
    participant EX as execution.rs:1003

    API->>CA: UpdateConstraints from POST /constraints/define :91 or /apply, handled at constraint_actor.rs:193
    CA->>CA: GetConstraints :215, ClearConstraints :256, GetConstraintStatistics :264
    CA->>FCA: UploadConstraintsToGPU :224
    API->>OCA: ApplyOntologyConstraints (ontology_constraint_actor.rs:451)
    alt GPU context absent
        Note over OCA: info "GPU not available, constraints cached for next physics step" :291 and cpu_fallback_count += 1 :286,:293
    else GPU ready
        OCA->>FCA: UpdateOntologyConstraintBuffer (force_compute_actor.rs:3798)
    end
    OCA->>OCA: ApplyMaterializedAxioms :522, AdjustConstraintWeights :723
    FCA->>EX: execute with num_constraints = resident count
    alt num_constraints > 0
        EX->>EX: ENABLE_CONSTRAINTS set (force_channels.rs:502-504) and constraint_force_ptr bound execution.rs:262
    else num_constraints == 0
        Note over EX: bit CLEARS - required, or force_pass_kernel keeps walking a buffer that no longer describes anything (force_channels.rs:560-562)
    end
    Note over CA,EX: INVARIANT ADR-2029 - enablement is owned by residency, never by settings
    Note over OCA: class_id, class_charge and class_mass are uploaded as separate per-node device buffers and passed to force_pass_kernel, not carried in SimParams
    Note over FCA: class_id = domain_class_id(source_domain) src/actors/gpu/force_compute_actor.rs:1175<br/>from the shared registry crates/vault-core/src/domains.rs:33 since 805219679 - 0 unknown,<br/>1-8 for the eight DOMAIN_ROOTS, legacy short names ai, bc, mv, rb, ngm, tc keep ids 1-6<br/>charge 0.6 known, 1.2 unknown src/actors/gpu/force_compute_actor.rs:1176
    Note over FCA: DEBT: GPUResourceActor uploads the same ids with charges 0.3 and 2.5<br/>src/actors/gpu/gpu_resource_actor.rs:228-229, upload :234 - two charge tables,<br/>the later upload_class_metadata call wins, src/actors/gpu/force_compute_actor.rs:1182
```

## VC-11.7 Semantic forces and stress majorization

```mermaid
sequenceDiagram
    autonumber
    participant PS as PhysicsSupervisor<br/>physics_supervisor.rs:1032
    participant SFA as SemanticForcesActor<br/>semantic_forces_actor.rs:751
    participant SMA as StressMajorizationActor<br/>stress_majorization_actor.rs:315
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3717

    PS->>SFA: ConfigureDAG (semantic_forces_actor.rs:751)
    PS->>SFA: ConfigureTypeClustering (semantic_forces_actor.rs:778)
    PS->>SFA: ConfigureCollision (semantic_forces_actor.rs:802)
    PS->>SFA: RecalculateHierarchy (semantic_forces_actor.rs:844)
    SFA->>SFA: ReloadRelationshipBuffer :869
    Note over SFA: DAG radial bias is SELF-GATED on dag_bias_k > 0 with dag_level_distance - it has no feature-flag bit (force_channels.rs:34-36)
    alt GPU absent
        Note over SFA: CPU fallback implementations exist :221
    end
    PS->>SMA: TriggerStressMajorization :658, forwarded at :674
    alt no GPU context
        Note over SMA: returns Err "GPU not available for stress majorization" :95 - hard failure, no CPU degradation
    else GPU ready
        SMA->>SMA: CheckStressMajorization :369, UpdateStressMajorizationParams :347
        SMA->>FCA: TriggerStressMajorization (force_compute_actor.rs:3717)
        FCA->>FCA: GetStressMajorizationStats :3729, ResetStressMajorizationSafety :3745
    end
    Note over SMA,FCA: DIVERGENCE bit5 ENABLE_STRESS_MAJORIZATION is declared but never set by derive_dispatch_feature_flags - stress majorization is not a GPU force channel
    Note over SMA,FCA: Stress majorization params live on CPU in SemanticProcessorActor and are absent from GPU SimParams - project/src/models/simulation_params.rs:77
    Note over SMA,FCA: DIVERGENCE bit3 ENABLE_TEMPORAL_COHERENCE is likewise declared but never set - both are reserved bits, not wired force terms
```

## VC-11.9 Warmup, settle and reheat cycle

```mermaid
stateDiagram-v2
    [*] --> Warmup
    Warmup: Stability warmup - stability_warmup_remaining decremented each tick at line 2007
    Active: Integrating - mean per-node KE at or above SETTLE_KE_EPSILON
    Settling: Consecutive sub-epsilon ticks accumulating in settle_stable_frames
    Settled: is_settled true - reported via GetSettlementState line 3607
    Reheated: reheat_factor greater than 0 - velocity perturbation injected

    Warmup --> Active: warmup frames exhausted
    Active --> Settling: next_stable_frames increments when mean_ke finite and below epsilon line 450-456
    Settling --> Active: KE at or above epsilon resets the run to 0 line 495
    Settling --> Settled: settle_stable_frames reaches SETTLE_FRAME_THRESHOLD
    Active --> Settled: paused latch - orchestrator declared an energy plateau line 470
    Settled --> Reheated: settings change, pin, or reset raises reheat_factor
    Reheated --> Active: decay to zero
    Reheated --> Reheated: reheat_factor multiplied by 0.997 each step line 2103

    note right of Warmup
        warmup = 1200 frames when edge_count == 0
        else 600 frames, reset after graph upload
        force_compute_actor.rs:1417
    end note
    note right of Settled
        SETTLE_KE_EPSILON = 1e-4 line 439
        SETTLE_FRAME_THRESHOLD = 180 line 444
        180 frames is about 3 s at 60 fps and mirrors
        the auto-balance stabilityFrameCount config
        so REST and auto-balance agree on settled
    end note
    note right of Reheated
        inject_velocity_perturbation(reheat_factor) line 2053 when factor > 0 line 2048
        decay 0.997 per step over about 30 steps line 2090,2103
        snapped to 0.0 once below 0.02 line 2104-2105
        seeds - UpdateSimulationParams max 1.5 line 2849
        UpdateSimulationParams repel_k-scaled assign line 3055
        PinNodePositions max 0.3 line 3380
        ResetPositions 1.0 line 3969
    end note
    note right of Active
        Periodic full broadcast and projection
        gate on iteration_count % 300 line 2204
        skipped frames logged every 300th line 1892,1934
    end note
```


## VC-11.10 Hard bounds and connected extent

```mermaid
flowchart TD
    All["All-node AABB reduction<br/>compute_aabb_reduction_kernel<br/>visionclaw_unified.cu:2533"] --> Grid["Spatial index retains isolated nodes"]
    Connected["Connected-only reduction excludes zero degree<br/>compute_connected_aabb_reduction_kernel<br/>visionclaw_unified.cu:2594"] --> Extent["connected_extent<br/>execution.rs:133"]
    Gate["degree_weighted_gravity_kernel launch gate<br/>execution.rs:868"] --> Branch{"viewport_bounds greater than 0"}
    Branch -->|yes| Bounded["extent = full all-node AABB<br/>execution.rs:877"]
    Branch -->|no, bounds disabled| Extent
    Extent --> Call["called at execution.rs:879 - AABB.default when None"]
    Bounded --> Shell["peripheral_shell_radius<br/>execution.rs:60"]
    Call --> Shell
    Shell --> ShellNote["Bounded - fixed fraction of viewport_bounds<br/>Unbounded - connected-extent heuristic, capped"]
    Force["Integration advances free nodes"] --> Clamp["Clamp each axis to viewport bounds<br/>Zero only outward velocity"]
    Pin["Pinned nodes retain host position"] -.-> Clamp
    Test["Actual CUDA fixture tests all axes and both AABBs<br/>integrate_bounds.cu:7"] -.-> Clamp
    Test -.-> Connected
```
