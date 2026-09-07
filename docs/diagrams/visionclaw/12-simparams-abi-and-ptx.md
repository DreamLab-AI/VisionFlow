---
id: VC-12
title: SimParams ABI, GPU buffers and PTX
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2028, ADR-2030, ADR-2054, ADR-2055, ADR-2056]
sources:
  - ../project/src/models/simulation_params.rs
  - ../project/crates/visionclaw-domain/src/models/simulation_params.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/src/models/force_channels.rs
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/utils/unified_gpu_compute/construction.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/gpu_resource_actor.rs
  - ../project/crates/visionclaw-gpu/build.rs
  - ../project/crates/visionclaw-gpu/src/ptx_policy.rs
  - ../project/crates/visionclaw-gpu/src/ptx_loader.rs
  - ../project/src/gpu/dynamic_buffer_manager.rs
  - ../project/src/gpu/memory_manager.rs
  - ../project/src/utils/gpu_diagnostics.rs
  - ../project/src/gpu/mod.rs
  - ../project/src/physics/mod.rs
verified_commit: 36bb64e1e
---
## VC-12.1 SimParams full 212-byte repr(C) layout
```mermaid
classDiagram
    class SimParams {
        +f32 dt_off0
        +f32 damping_off4
        +u32 warmup_iterations_off8
        +f32 cooling_rate_off12
        +f32 spring_k_off16
        +f32 rest_length_off20
        +f32 repel_k_off24
        +f32 repulsion_cutoff_off28
        +f32 repulsion_softening_epsilon_off32
        +f32 center_gravity_k_off36
        +f32 max_force_off40
        +f32 max_velocity_off44
        +f32 grid_cell_size_off48
        +u32 feature_flags_off52
        +u32 seed_off56
        +i32 iteration_off60
        +f32 separation_radius_off64
        +f32 cluster_strength_off68
        +f32 alignment_strength_off72
        +f32 temperature_off76
        +f32 viewport_bounds_off80
        +f32 sssp_alpha_off84
        +f32 boundary_damping_off88
        +u32 constraint_ramp_frames_off92
        +f32 constraint_max_force_per_node_off96
        +f32 stability_threshold_off100
        +f32 min_velocity_threshold_off104
        +f32 world_bounds_min_off108
        +f32 world_bounds_max_off112
        +f32 cell_size_lod_off116
        +u32 k_neighbors_max_off120
        +f32 anomaly_detection_radius_off124
        +f32 learning_rate_default_off128
        +f32 norm_delta_cap_off132
        +f32 position_constraint_attraction_off136
        +f32 lof_score_min_off140
        +f32 lof_score_max_off144
        +f32 weight_precision_multiplier_off148
        +f32 gravity_off152
        +u32 lin_log_mode_off156
        +f32 scaling_ratio_off160
        +u32 adaptive_speed_off164
        +f32 global_speed_off168
        +f32 dag_bias_k_off172
        +f32 dag_level_distance_off176
        +u32 layout_mode_off180
        +f32 plane_bias_k_off184
        +f32 plane_spacing_off188
        +f32 radial_center_x_off192
        +f32 radial_center_y_off196
        +f32 radial_center_z_off200
        +f32 layer_bias_k_off204
        +f32 layer_spacing_off208
    }
    class ABIEvidence {
        +rust_assert_simulation_params_rs_228
        +cuda_assert_visionclaw_unified_cu_117
        +manifest_simulation_params_rs_327_381
        +verify_fn_simulation_params_rs_472_515
        +tail_append_rule_simulation_params_rs_49_59
    }
    class ForceChannelBacking {
        +Repulsion_repel_k_off24_bit0
        +Spring_spring_k_off16_bit1
        +Centering_center_gravity_k_off36_bit2
        +Separation_separation_radius_off64
        +Gravity_gravity_off152
        +ClusterCohesion_cluster_strength_off68
        +DagRadialBias_dag_bias_k_off172
        +Annealing_temperature_off76
        +Boundary_viewport_bounds_off80
        +Constraints_off96_bit4_READONLY
    }
    ABIEvidence --> SimParams
    ForceChannelBacking --> SimParams
    note for SimParams "INVARIANT: size_of equals 212<br/>align 4, zero pad, both sides"
    note for SimParams "RESOLVED ADR-2060: 180-byte comment corrected to 212"
    note for ABIEvidence "citations for the 212-byte lock and the tail-append rule"
    note for ForceChannelBacking "force_channels.rs:157-212<br/>strength_of/feature_flag/is_read_only"
```

## VC-12.2 feature_flags bitfield (u32 at offset 52)
```mermaid
classDiagram
    class FeatureFlags_Rust {
        +u32 ENABLE_REPULSION = 1_lsh_0
        +u32 ENABLE_SPRINGS = 1_lsh_1
        +u32 ENABLE_CENTERING = 1_lsh_2
        +u32 ENABLE_TEMPORAL_COHERENCE = 1_lsh_3
        +u32 ENABLE_CONSTRAINTS = 1_lsh_4
        +u32 ENABLE_STRESS_MAJORIZATION = 1_lsh_5
        +u32 ENABLE_SSSP_SPRING_ADJUST = 1_lsh_6
    }
    class FeatureFlags_CUDA {
        +uint ENABLE_REPULSION = 1_lsh_0
        +uint ENABLE_SPRINGS = 1_lsh_1
        +uint ENABLE_CENTERING = 1_lsh_2
        +uint ENABLE_CONSTRAINTS = 1_lsh_4
        +uint ENABLE_SSSP_SPRING_ADJUST = 1_lsh_6
    }
    class KernelGateSites {
        +bit0_force_pass_L442
        +bit1_spring_L542
        +bit1_bit6_linlog_sssp_blend_L548
        +bit2_centering_L600
        +bit4_constraint_loop_L613
        +stability_path_mirror_L2271_2396
    }
    FeatureFlags_Rust ..> FeatureFlags_CUDA
    FeatureFlags_CUDA --> KernelGateSites
    note for FeatureFlags_Rust "crates/visionclaw-domain/src/models/simulation_params.rs:111-119"
    note for FeatureFlags_CUDA "crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:123-129"
    note for FeatureFlags_Rust "ADR-2060: bit3/bit5 are RESERVED Rust-side<br/>zero hits in CUDA, not a defect"
    note for FeatureFlags_Rust "KEYSTONE: ENABLE_CONSTRAINTS is<br/>residency-owned (execution.rs:954)"
```

## VC-12.3 GPU core physics buffers: per-node, per-edge, per-grid-cell
```mermaid
classDiagram
    class PerNodeBuffers {
        +f32 pos_in_x
        +f32 pos_in_y
        +f32 pos_in_z
        +f32 pos_out_x
        +f32 pos_out_y
        +f32 pos_out_z
        +f32 vel_in_x
        +f32 vel_in_y
        +f32 vel_in_z
        +f32 vel_out_x
        +f32 vel_out_y
        +f32 vel_out_z
        +f32 mass
        +i32 node_graph_id
        +i32 class_id
        +f32 class_charge
        +f32 class_mass
        +f32 spring_scale
        +i32 pinned_mask
        +f32 node_rank
        +f32 node_plane
        +f32 force_x
        +f32 force_y
        +f32 force_z
        +f32 prev_force_x
        +f32 prev_force_y
        +f32 prev_force_z
        +f32 degree_weight
        +f32 node_constraint_force
        +i32 compute_mask
        +i32 cell_keys
        +i32 sorted_node_indices
        +i32 sort_keys_out
        +i32 sort_values_out
        +f32 dist
        +i32 current_frontier
        +i32 next_frontier_flags
        +i32 parents_optional
        +i32 cluster_assignments
        +f32 distances_to_centroid
        +f32 min_distances
        +f32 lof_scores
        +f32 local_densities
        +f32 zscore_values
        +f32 feature_values
        +i32 labels_current
        +i32 labels_next
        +i32 label_counts
        +f32 node_degrees
        +i32 community_sizes
        +i32 label_mapping
        +curandState rand_states
    }
    class PerEdgeBuffers {
        +i32 edge_row_offsets
        +i32 edge_col_indices
        +f32 edge_weights
    }
    class PerGridCellBuffers {
        +i32 cell_start
        +i32 cell_end
    }
    class UnifiedGPUCompute {
        production struct construction.rs:31-241
    }
    class PhysicsV2Gated {
        +REMOVED_ADR_2055_PhysicsGpuBuffers
        +never_shipped_in_a_build
    }
    class DeprecatedManager {
        +dynamic_buffer_manager_rs_deprecated
        +since_0_1_0
    }
    class MemoryBudgetTracker {
        +GpuMemoryManager_live_separate
        +own_GpuBufferT_wrapper_type
    }
    UnifiedGPUCompute *-- PerNodeBuffers
    UnifiedGPUCompute *-- PerEdgeBuffers
    UnifiedGPUCompute *-- PerGridCellBuffers
    PhysicsV2Gated ..> DeprecatedManager
    DeprecatedManager ..> MemoryBudgetTracker

    note for PerNodeBuffers "stride 4B all fields except rand_states 48B<br/>count=allocated_nodes (L46-238)"
    note for PerEdgeBuffers "row_offsets=num_nodes+1<br/>col/weights=num_edges (L373-375)"
    note for PerGridCellBuffers "max_grid_cells = 32768 fixed<br/>(construction.rs:369-371)"
    note for UnifiedGPUCompute "production per-node buffer owner<br/>(construction.rs:31-241)"
    note for PhysicsV2Gated "DOC-DRIFT — no buffers.rs exists. REMOVED under ADR-2055 (src/gpu/mod.rs:10-15):<br/>PhysicsGpuBuffers was gated behind the physics-v2 feature, itself retired<br/>(src/physics/mod.rs:69-73). Never shipped in a build; UnifiedGPUCompute was<br/>always the sole per-node GPU buffer owner"
    note for DeprecatedManager "dynamic_buffer_manager.rs:1-16<br/>use memory_manager.rs"
    note for MemoryBudgetTracker "memory_manager.rs:127-152<br/>force_compute_actor.rs:880-910"
```

## VC-12.4 GPU analytics-adjacent buffers: cluster, community, block, scalar
```mermaid
classDiagram
    class PerClusterBuffers {
        +f32 centroids_x
        +f32 centroids_y
        +f32 centroids_z
        +i32 cluster_sizes
        +i32 selected_nodes
    }
    class PerCommunityBuffers {
        +f32 community_centroids_x
        +f32 community_centroids_y
        +f32 community_centroids_z
    }
    class PerBlockBuffers {
        +f32 partial_inertia
        +f32 partial_sums
        +f32 partial_sq_sums
        +f32 partial_kinetic_energy
        +AABB aabb_block_results
    }
    class ScalarBuffers {
        +i32 convergence_flag
        +i32 active_node_count
        +i32 should_skip_physics
        +f32 system_kinetic_energy
    }
    class UnifiedGPUCompute {
        production struct construction.rs:31-241, see VC-12.3
    }
    UnifiedGPUCompute *-- PerClusterBuffers
    UnifiedGPUCompute *-- PerCommunityBuffers
    UnifiedGPUCompute *-- PerBlockBuffers
    UnifiedGPUCompute *-- ScalarBuffers

    note for PerClusterBuffers "stride 4B, max_clusters = 50 fixed (construction.rs:401)"
    note for PerCommunityBuffers "stride 4B, count=num_nodes.max(1), Louvain exceeds 50 (L408)"
    note for PerBlockBuffers "stride 4B, count = ceil(num_nodes/256) per reduction block"
    note for ScalarBuffers "stride 4B, count = 1 (single-element buffers)"
    note for UnifiedGPUCompute "same struct as VC-12.3, split here only for diagram width"
```

## VC-12.5 build.rs: nvcc to PTX to .version 9.0 downgrade
```mermaid
sequenceDiagram
    autonumber
    participant Cargo as cargo build
    participant BR as build.rs<br/>crates/visionclaw-gpu/build.rs:20
    participant Policy as ptx_policy<br/>crates/visionclaw-gpu/src/ptx_policy.rs (include!'d at build.rs:18)
    participant Nvcc as nvcc process
    participant FS as OUT_DIR / fallback PTX files

    Cargo->>BR: cargo build --feature gpu (CARGO_FEATURE_GPU set, build.rs:20-27)
    BR->>BR: resolve CUDA_ARCH: env override, else nvidia-smi compute_cap, else sm_75 (build.rs:57-80)
    rect rgb(240,240,255)
    loop for each of 9 cuda_files (build.rs:30-40, loop at :118)
        BR->>Nvcc: nvcc -ptx -arch sm_ARCH -o OUT/NAME.ptx NAME.cu --use_fast_math -O3 (build.rs:123-146)
        alt nvcc launches and exits 0 (NvccOutcome::Succeeded, ptx_policy.rs:69,78)
            Nvcc-->>BR: ptx_output written, provenance=Compiled (build.rs:157)
        else nvcc not on PATH (NvccOutcome::LaunchFailed, ptx_policy.rs:62,77) or nvcc exits nonzero (NvccOutcome::CompilerFailed, ptx_policy.rs:66,79)
            BR->>Policy: NvccOutcome::classify(spawn_error, success, code) (ptx_policy.rs:75-81, build.rs:147-150)
            Policy-->>BR: outcome.needs_fallback() true for both failure modes (ptx_policy.rs:85-87)
            BR->>FS: search fallback_paths: src/ptx/NAME.ptx, /app/src/utils/ptx/NAME.ptx, /app/crates/visionclaw-gpu/src/ptx/NAME.ptx (build.rs:168-174)
            opt a fallback file exists
                FS-->>BR: fallback path found
                BR->>FS: fs::copy(fallback, ptx_output) (build.rs:181)
                BR->>BR: provenance = FallbackAfterLaunchFailure or FallbackAfterCompilerFailure (ptx_policy.rs:120-125, build.rs:182)
            end
            break no fallback file exists at any candidate path
                BR->>BR: panic! PTX unavailable for NAME, no fallback found (build.rs:185-193)
            end
        end
        BR->>FS: read_to_string(ptx_output) (build.rs:199-200)
        BR->>Policy: rewrite_ptx_version(original, TARGET_PTX_ISA=9.0) (build.rs:207, ptx_policy.rs:233-254)
        alt found version <= 9.0 (VersionRewrite::Unchanged, ptx_policy.rs:242-243)
            Policy-->>BR: Unchanged version - content untouched, no downgrade warning emitted (build.rs:208-211)
        else found version > 9.0, e.g. CUDA 13.x emits 9.2 (VersionRewrite::Rewritten, ptx_policy.rs:245-253)
            Policy-->>BR: Rewritten from,to,text - splice by parsed token span, not fixed width (ptx_policy.rs:245-253)
            BR->>FS: fs::write(ptx_output, downgraded text) (build.rs:214)
            BR->>BR: cargo:warning declared ISA rewritten to 9.0 (build.rs:215-219)
        else no .version token or unparseable token (VersionRewrite::Defective, ptx_policy.rs:234-241)
            BR->>BR: panic! PTX unusable after provenance phase (build.rs:222-227)
        end
        BR->>Policy: validate_ptx(final_text, required_symbols) (build.rs:238, ptx_policy.rs:262-288)
        Note over BR,Policy: required_symbols = force_pass_kernel, integrate_pass_kernel<br/>visionclaw_unified module only (build.rs:233-237)<br/>REQUIRED_UNIFIED_SYMBOLS at ptx_policy.rs:349
        alt validate_ptx returns Err (empty, missing .version/.target/.entry, or missing required symbol)
            BR->>BR: panic! PTX validation failed for NAME (build.rs:239-245)
        else Ok
            BR->>BR: push PtxArtefact module,source,provenance,isa,original_tag,rewritten_tag (build.rs:247-254)
            BR->>Cargo: cargo:rustc-env=NAME_PTX_PATH=OUT/NAME.ptx (build.rs:256-257)
        end
    end
    end
    BR->>FS: write ptx-build-manifest.txt, one manifest_line per module (build.rs:262-267)
    BR->>Cargo: cargo:rustc-env=VISIONCLAW_PTX_MANIFEST=OUT/ptx-build-manifest.txt (build.rs:268-271)
    Note right of BR: INVARIANT - PTX downgraded to .version 9.0 before load<br/>fallback-PTX path exists for nvcc-less builds<br/>GPU-wire-abi.md Invariant 5, ADR-2030
    Note right of Policy: DIVERGENCE - the rewrite is a declared-ISA text splice only<br/>it does not prove every instruction is supported by that ISA<br/>only a real driver load settles that (ADR-2030 Consequences)
    Note over BR,FS: README's "82 CUDA kernels" (root README.md) is the __global__ function count across<br/>these 9 .cu files (visionclaw_unified.cu 25, gpu_clustering_kernels.cu 24, semantic_forces.cu 15,<br/>pagerank.cu 7, gpu_connected_components.cu 3, gpu_landmark_apsp.cu 2, sssp_compact.cu 2,<br/>gpu_aabb_reduction.cu 1, dynamic_grid.cu 0 — verified by count, matches)
```

## VC-12.6 runtime PTX load, module init and kernel handle lookup
```mermaid
sequenceDiagram
    autonumber
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:51
    participant Loader as ptx_loader<br/>ptx_loader.rs:366
    participant Downgrade as downgrade_ptx_isa_if_needed<br/>ptx_loader.rs:305-330
    participant UGC as UnifiedGPUCompute<br/>construction.rs:247
    participant Cust as cust::module::Module<br/>CUDA driver JIT

    GRA->>Loader: load_ptx_module_sync(PTXModule::VisionflowUnified) (gpu_resource_actor.rs:99-101)
    Loader->>Loader: load_ptx_module_sync_raw - source selection, see VC-12.7<br/>(ptx_loader.rs:373)
    Loader-->>Loader: raw PTX string
    Loader->>Downgrade: downgrade_ptx_isa_if_needed(raw) (ptx_loader.rs:370)
    Downgrade->>Downgrade: detect_max_ptx_isa() via nvidia-smi driver_version<br/>(ptx_loader.rs:91-121)
    alt driver query succeeds, CUDA header probe or approximate driver mapping resolves
        Downgrade-->>Downgrade: RUNTIME_MAX_PTX_ISA OnceLock get_or_init caches the result (ptx_loader.rs:23 static, :92 get_or_init)
    else driver query fails or mapping is unavailable
        Downgrade-->>Downgrade: fallback (9,0) (ptx_loader.rs:103-118)
    end
    Downgrade->>Downgrade: build target PtxVersion, call ptx_policy::rewrite_ptx_version(&ptx, target)<br/>(ptx_loader.rs:332-336)
    alt VersionRewrite::Rewritten { from, to, text }
        Downgrade->>Downgrade: info! ISA downgrade log, return the rewritten text (ptx_loader.rs:337-343)
        Note right of Downgrade: RESOLVED ADR-2056 - no longer a second impl<br/>downgrade_ptx_isa_if_needed now delegates the actual span-parsed splice<br/>to ptx_policy::rewrite_ptx_version (ptx_policy.rs:233-243, :245-253) -<br/>it retains only the runtime driver-ISA probe
    else VersionRewrite::Unchanged | VersionRewrite::Defective
        Downgrade-->>Loader: ptx text unchanged (ptx_loader.rs:347-348)
    end
    Downgrade-->>GRA: Ok(ptx_content) (gpu_resource_actor.rs:99-105)
    opt clustering PTX (PTXModule::GpuClusteringKernels)
        GRA->>Loader: load_ptx_module_sync(GpuClusteringKernels) (gpu_resource_actor.rs:111-113)
        alt load fails
            Loader-->>GRA: Err, warn! will use fallback, clustering_ptx=None (gpu_resource_actor.rs:121-124)
        else load succeeds
            Loader-->>GRA: Some(content) (gpu_resource_actor.rs:114-119)
        end
    end
    opt APSP PTX (PTXModule::GpuLandmarkApsp)
        GRA->>Loader: load_ptx_module_sync(GpuLandmarkApsp) (gpu_resource_actor.rs:136-138)
        alt load fails
            Loader-->>GRA: Err, warn! apsp_ptx=None — NO CPU fallback exists for this path<br/>(gpu_resource_actor.rs:146-149, comment :131-135)
        else load succeeds
            Loader-->>GRA: Some(content) (gpu_resource_actor.rs:139-145)
        end
    end
    GRA->>UGC: new_with_modules(num_nodes, num_edges, ptx, clustering_ptx, apsp_ptx)<br/>(gpu_resource_actor.rs:153) delegates to new_with_all_modules (construction.rs:247-254, :261-267)
    UGC->>UGC: validate_ptx_content(ptx_content) (construction.rs:268, gpu_diagnostics.rs:185)
    alt primary PTX fails structural validation
        UGC-->>GRA: Err PTX validation failed, plus diagnose_ptx_error()<br/>(construction.rs:269-271)
    else Ok
        UGC->>Cust: Module::from_ptx(ptx_content, empty_options) (construction.rs:276)
        alt module load fails
            UGC-->>GRA: Err Module::from_ptx() failed, plus diagnosis (construction.rs:277-280)
        else module loads
            opt clustering_ptx is Some
                UGC->>UGC: validate_ptx_content(clustering_ptx_content) (construction.rs:286-289)
                alt validation fails
                    UGC->>UGC: error! GPU DEGRADED, clustering_module=None<br/>(construction.rs:290-295)
                else validation ok
                    UGC->>Cust: Module::from_ptx(clustering_ptx, opts)<br/>(construction.rs:297)
                    alt secondary load fails
                        UGC->>UGC: error! GPU DEGRADED, clustering_module=None<br/>(construction.rs:302-309)
                    else ok
                        UGC->>UGC: clustering_module=Some(module) (construction.rs:298-301)
                    end
                end
            end
        end
    end
    Note over UGC: DOC-DRIFT (predates this window, commit da2f5cac7) - apsp_ptx is NO LONGER loaded<br/>into a module. REMOVED under ADR-2054: apsp_module had zero read sites (nothing ever<br/>resolved a kernel from it), so the load cost a startup PTX parse and logged a false<br/>"GPU APSP is DISABLED" about a capability NFR-7 forbids permanently. The parameter is<br/>retained and ignored (let _ = apsp_ptx, construction.rs:316-322) so callers need not<br/>change shape. See VC-15 for ShortestPathActor::ComputeAPSP, which never used this module.
    Note over UGC,Cust: get_function(name) resolves a kernel by string<br/>e.g. force_pass/integrate_pass_kernel (execution.rs:612,863)<br/>a Rust ? propagates the error if absent
    Note right of UGC: INVARIANT - fallback-PTX for nvcc-less builds<br/>secondary-module failures DEGRADE, not crash<br/>GPU-wire-abi.md Invariant 5, L283-286
    Note right of UGC: DIVERGENCE - get_function by name, ONLY symbol check<br/>unguarded if build-time symbol gate bypassed<br/>e.g. a cached OUT_DIR artefact
```

## VC-12.7 PTX source selection: compiled vs pre-shipped vs runtime-compiled
```mermaid
flowchart TD
    Start["load_ptx_module_sync_raw(module)<br/>ptx_loader.rs:373"] --> DockerCheck{"DOCKER_ENV set?<br/>ptx_loader.rs:374"}
    DockerCheck -->|yes| DockerPre["load_precompiled_ptx(module)<br/>ptx_loader.rs:377"]
    DockerPre --> DockerPreOk{"found and valid?"}
    DockerPreOk -->|yes| ReturnA["Ok(content)<br/>ptx_loader.rs:377-378"]
    DockerPreOk -->|no| DockerCompile["compile_ptx_fallback_sync_module(module)<br/>ptx_loader.rs:382"]
    DockerCompile --> ReturnB["Ok/Err from nvcc runtime compile"]

    DockerCheck -->|no| BakedCheck{"get_compiled_ptx_path(module) resolves?<br/>ptx_loader.rs:385, fn at 286-293"}
    BakedCheck -->|"option_env! baked path (build.rs cargo:rustc-env)"| ReadBaked["fs::read_to_string(path)<br/>ptx_loader.rs:386"]
    BakedCheck -->|"or std::env::var(module.env_var()) runtime override<br/>ptx_loader.rs:292"| ReadBaked
    ReadBaked --> ValidBaked{"validate_ptx ok?<br/>ptx_loader.rs:388"}
    ValidBaked -->|yes| ReturnC["Ok(content) - fastest path, fresh OUT_DIR PTX<br/>ptx_loader.rs:395-396"]
    ValidBaked -->|no| PreCompiled
    BakedCheck -->|"neither set"| PreCompiled["load_precompiled_ptx(module)<br/>ptx_loader.rs:409"]

    PreCompiled --> C1["1. baked compiled_ptx_path option_env!<br/>ptx_loader.rs:434-436"]
    C1 --> C2["1b. runtime env var override module.env_var()<br/>ptx_loader.rs:439-441"]
    C2 --> C3["2. target/release-or-debug/build/star/out/NAME.ptx<br/>newest mtime first<br/>ptx_loader.rs:451-470"]
    C3 --> C4["3a. crates/visionclaw-gpu/src/ptx/NAME.ptx<br/>ptx_loader.rs:476"]
    C4 --> C5["3b. ../visionclaw-gpu/src/ptx/NAME.ptx<br/>ptx_loader.rs:477-479"]
    C5 --> C6["3c. /app/crates/visionclaw-gpu/src/ptx/NAME.ptx<br/>ptx_loader.rs:480"]
    C6 --> C7["3d. legacy /app/src/utils/ptx/NAME.ptx<br/>ptx_loader.rs:482"]
    C7 --> C8["3e. legacy ./src/utils/ptx/NAME.ptx<br/>ptx_loader.rs:483"]
    C8 --> AnyValid{"first candidate that reads and validate_ptx ok?<br/>ptx_loader.rs:486-493"}
    AnyValid -->|yes| ReturnD["Ok(content)<br/>ptx_loader.rs:488-491"]
    AnyValid -->|no candidate found| RuntimeCompile["compile_ptx_fallback_sync_module(module)<br/>ptx_loader.rs:417"]

    RuntimeCompile --> Arch["effective_cuda_arch(): delegates to detect_runtime_cuda_arch()<br/>ptx_loader.rs:301-303"]
    Arch --> CuSrc{"src/utils/NAME.cu exists under CARGO_MANIFEST_DIR?<br/>ptx_loader.rs:554-560"}
    CuSrc -->|no| ErrNoSrc["Err CUDA source not found<br/>ptx_loader.rs:561-565"]
    CuSrc -->|yes| SpawnNvcc["nvcc -ptx -std=c++17 -arch=sm_ARCH NAME.cu -o TMP.ptx<br/>ptx_loader.rs:582-602"]
    SpawnNvcc --> NvccOk{"spawn ok and exit 0?"}
    NvccOk -->|no| ErrNvcc["Err nvcc failed, includes stdout/stderr<br/>ptx_loader.rs:604-617"]
    NvccOk -->|yes| ReadTmp["fs::read_to_string(TMP.ptx)<br/>ptx_loader.rs:619-625"]
    ReadTmp --> ValidateTmp["validate_ptx(ptx_content)<br/>ptx_loader.rs:627"]
    ValidateTmp --> ReturnE["Ok(content)<br/>ptx_loader.rs:628-633"]

    ReturnA --> Downgrade["downgrade_ptx_isa_if_needed(raw) - see VC-12.5<br/>ptx_loader.rs:370"]
    ReturnB --> Downgrade
    ReturnC --> Downgrade
    ReturnD --> Downgrade
    ReturnE --> Downgrade
```
