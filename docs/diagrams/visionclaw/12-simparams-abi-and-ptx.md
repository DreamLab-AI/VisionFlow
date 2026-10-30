---
id: VC-12
title: SimParams ABI, GPU buffers and PTX
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2028, ADR-2030, ADR-2054, ADR-2055, ADR-2056, ADR-2140]
sources:
  - ../project/src/models/simulation_params.rs
  - ../project/crates/visionclaw-domain/src/models/simulation_params.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu
  - ../project/src/models/force_channels.rs
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/utils/unified_gpu_compute/construction.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/gpu_resource_actor.rs
  - ../project/crates/visionclaw-gpu/build.rs
  - ../project/crates/visionclaw-gpu/src/ptx_policy.rs
  - ../project/crates/visionclaw-gpu/src/ptx_loader.rs
  - ../project/crates/visionclaw-gpu/src/nvcc_toolchain.rs
  - ../project/docs/README.md
  - ../project/docs/GPU-wire-abi.md
  - ../project/src/gpu/memory_manager.rs
  - ../project/src/utils/gpu_diagnostics.rs
  - ../project/src/gpu/mod.rs
  - ../project/src/physics/mod.rs
  - ../project/docs/adr/ADR-2054-remove-dead-gpu-and-analytics-code.md
verified_commit: 8692b12ba987ecb87165566beb7c320a2446e4b4
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
        +rust_assert_simulation_params_rs_231
        +cuda_assert_visionclaw_unified_cu_118
        +manifest_simulation_params_rs_330_688
        +verify_fn_simulation_params_rs_696_739
        +tail_append_test_simulation_params_rs_1087_1106
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
    note for ForceChannelBacking "force_channels.rs:129-214<br/>strength_of/feature_flag/is_read_only"
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
        +bit0_force_pass_L443
        +bit1_spring_L543
        +bit1_bit6_linlog_sssp_blend_L549
        +bit2_centering_L601
        +bit4_constraint_loop_L614
        +stability_path_mirror_L2277_2402
    }
    FeatureFlags_Rust ..> FeatureFlags_CUDA
    FeatureFlags_CUDA --> KernelGateSites
    note for FeatureFlags_Rust "crates/visionclaw-domain/src/models/simulation_params.rs:96-104"
    note for FeatureFlags_CUDA "crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:124-130"
    note for FeatureFlags_Rust "ADR-2060: bit3/bit5 are RESERVED Rust-side<br/>zero hits in CUDA, not a defect"
    note for FeatureFlags_Rust "DRIFT: docs/GPU-wire-abi.md:70 places the bit constants at<br/>lines 113-119 of the domain file, they are at crates/visionclaw-domain/src/models/simulation_params.rs:96-104"
    note for FeatureFlags_Rust "KEYSTONE: ENABLE_CONSTRAINTS is<br/>residency-owned (execution.rs:1069-1072)"
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
        production struct construction.rs:30-245
    }
    class PhysicsV2Gated {
        +REMOVED_ADR_2055_PhysicsGpuBuffers
        +never_shipped_in_a_build
    }
    class DeletedManager {
        +dynamic_buffer_manager_rs_DELETED
        +clippy_sweep_ADR_2054_L172
    }
    class MemoryBudgetTracker {
        +GpuMemoryManager_live_separate
        +own_GpuBufferT_wrapper_type
    }
    UnifiedGPUCompute *-- PerNodeBuffers
    UnifiedGPUCompute *-- PerEdgeBuffers
    UnifiedGPUCompute *-- PerGridCellBuffers
    PhysicsV2Gated ..> DeletedManager
    DeletedManager ..> MemoryBudgetTracker

    note for PerNodeBuffers "stride 4B all fields except rand_states 48B<br/>count=allocated_nodes (construction.rs:45-243)"
    note for PerEdgeBuffers "row_offsets=num_nodes+1<br/>col/weights=num_edges (construction.rs:357-359)"
    note for PerGridCellBuffers "max_grid_cells = 32768 fixed<br/>(construction.rs:374-376)"
    note for UnifiedGPUCompute "production per-node buffer owner<br/>(construction.rs:30-245)<br/>also owns semantic Option SemanticGpu and semantic_active, ADR-2140 (construction.rs:221-224)"
    note for PhysicsV2Gated "DOC-DRIFT — no buffers.rs exists. REMOVED under ADR-2055 (src/gpu/mod.rs:10-15):<br/>PhysicsGpuBuffers was gated behind the physics-v2 feature, itself retired<br/>(src/physics/mod.rs:69-73). Never shipped in a build; UnifiedGPUCompute was<br/>always the sole per-node GPU buffer owner"
    note for DeletedManager "DRIFT: module deleted (src/gpu/mod.rs:24-27 lists only memory_manager,<br/>ADR-2054:172) yet memory_manager.rs:5 still names it as a consolidated source"
    note for MemoryBudgetTracker "memory_manager.rs:127-152 GpuBuffer, :382 GpuMemoryManager<br/>force_compute_actor.rs:776-812"
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
        production struct construction.rs:30-245, see VC-12.3
    }
    UnifiedGPUCompute *-- PerClusterBuffers
    UnifiedGPUCompute *-- PerCommunityBuffers
    UnifiedGPUCompute *-- PerBlockBuffers
    UnifiedGPUCompute *-- ScalarBuffers

    note for PerClusterBuffers "stride 4B, max_clusters = 50 fixed (construction.rs:383)"
    note for PerCommunityBuffers "stride 4B, count=num_nodes.max(1), Louvain exceeds 50 (construction.rs:393-395)"
    note for PerBlockBuffers "stride 4B, count = ceil(num_nodes/256) per reduction block"
    note for ScalarBuffers "stride 4B, count = 1 (single-element buffers)"
    note for UnifiedGPUCompute "same struct as VC-12.3, split here only for diagram width"
```

## VC-12.5 build.rs: nvcc to PTX to .version 9.0 downgrade
```mermaid
sequenceDiagram
    autonumber
    participant Cargo as cargo build
    participant BR as build.rs<br/>crates/visionclaw-gpu/build.rs:22
    participant Policy as ptx_policy<br/>crates/visionclaw-gpu/src/ptx_policy.rs (include!'d at build.rs:18)
    participant Nvcc as nvcc process
    participant FS as OUT_DIR / fallback PTX files

    Cargo->>BR: cargo build --feature gpu (CARGO_FEATURE_GPU set, build.rs:22-29)
    BR->>BR: resolve CUDA_ARCH: env override, else nvidia-smi compute_cap, else sm_75 (build.rs:62-85)
    BR->>BR: select the nvcc host toolchain: NVCC_CCBIN, then CUDAHOSTCXX, then four fixed candidate paths for --compiler-bindir, plus -I of CUDA_PATH/include when cuda_runtime.h exists (build.rs:91-117, select_host_compiler nvcc_toolchain.rs:43-61, cuda_include_dir :66-69)
    rect rgb(240,240,255)
    loop for each of 8 cuda_files (build.rs:32-43, dynamic_grid.cu is excluded as host-only, :35-36, loop at :132)
        BR->>Nvcc: nvcc -ptx -arch sm_ARCH -o OUT/NAME.ptx NAME.cu --use_fast_math -O3 plus host_toolchain_args (build.rs:137-154)
        alt nvcc launches and exits 0 (NvccOutcome::Succeeded, ptx_policy.rs:69,78)
            Nvcc-->>BR: ptx_output written, provenance=Compiled (build.rs:171)
        else nvcc not on PATH (NvccOutcome::LaunchFailed, ptx_policy.rs:62,77) or nvcc exits nonzero (NvccOutcome::CompilerFailed, ptx_policy.rs:66,79)
            BR->>Policy: NvccOutcome::classify(spawn_error, success, code) (ptx_policy.rs:75-81, build.rs:161-164)
            Policy-->>BR: outcome.needs_fallback() true for both failure modes (ptx_policy.rs:85-87)
            BR->>FS: search fallback_paths: src/ptx/NAME.ptx, /app/src/utils/ptx/NAME.ptx, /app/crates/visionclaw-gpu/src/ptx/NAME.ptx (build.rs:182-188)
            opt a fallback file exists
                FS-->>BR: fallback path found
                BR->>FS: fs::copy(fallback, ptx_output) (build.rs:195)
                BR->>BR: provenance = FallbackAfterLaunchFailure or FallbackAfterCompilerFailure (ptx_policy.rs:120-125, build.rs:196)
            end
            break no fallback file exists at any candidate path
                BR->>BR: panic! PTX unavailable for NAME, no fallback found (build.rs:199-207)
            end
        end
        BR->>FS: read_to_string(ptx_output) (build.rs:213-214)
        BR->>Policy: rewrite_ptx_version(original, TARGET_PTX_ISA=9.0) (build.rs:221, ptx_policy.rs:233-254)
        alt found version <= 9.0 (VersionRewrite::Unchanged, ptx_policy.rs:242-243)
            Policy-->>BR: Unchanged version - content untouched, no downgrade warning emitted (build.rs:222-225)
        else found version > 9.0, e.g. CUDA 13.x emits 9.2 (VersionRewrite::Rewritten, ptx_policy.rs:245-253)
            Policy-->>BR: Rewritten from,to,text - splice by parsed token span, not fixed width (ptx_policy.rs:245-253)
            BR->>FS: fs::write(ptx_output, downgraded text) (build.rs:228)
            BR->>BR: cargo:warning declared ISA rewritten to 9.0 (build.rs:229-233)
        else no .version token or unparseable token (VersionRewrite::Defective, ptx_policy.rs:234-241)
            BR->>BR: panic! PTX unusable after provenance phase (build.rs:236-241)
        end
        BR->>Policy: validate_ptx(final_text, required_symbols) (build.rs:252, ptx_policy.rs:262-288)
        Note over BR,Policy: required_symbols = force_pass_kernel, integrate_pass_kernel<br/>visionclaw_unified module only (build.rs:247-251)<br/>REQUIRED_UNIFIED_SYMBOLS at ptx_policy.rs:349
        alt validate_ptx returns Err (empty, missing .version/.target/.entry, or missing required symbol)
            BR->>BR: panic! PTX validation failed for NAME (build.rs:253-259)
        else Ok
            BR->>BR: push PtxArtefact module,source,provenance,isa,original_tag,rewritten_tag (build.rs:261-268)
            BR->>Cargo: cargo:rustc-env=NAME_PTX_PATH=OUT/NAME.ptx (build.rs:270-271)
        end
    end
    end
    BR->>FS: write ptx-build-manifest.txt, one manifest_line per module (build.rs:276-281)
    BR->>Cargo: cargo:rustc-env=VISIONCLAW_PTX_MANIFEST=OUT/ptx-build-manifest.txt (build.rs:282-285)
    Note right of BR: INVARIANT - PTX downgraded to .version 9.0 before load<br/>fallback-PTX path exists for nvcc-less builds<br/>GPU-wire-abi.md Invariant 5, ADR-2030
    Note right of Policy: DIVERGENCE - the rewrite is a declared-ISA text splice only<br/>it does not prove every instruction is supported by that ISA<br/>only a real driver load settles that (ADR-2030 Consequences)
    Note over BR,FS: DRIFT: docs/README.md:38 states 82 CUDA __global__ kernels across 9 .cu files<br/>a count of __global__ definitions at this stamp gives 88: visionclaw_unified.cu 29 (26 plain plus 3 extern C),<br/>gpu_clustering_kernels.cu 24, semantic_forces.cu 20, pagerank.cu 7, gpu_connected_components.cu 3,<br/>gpu_landmark_apsp.cu 2, sssp_compact.cu 2, gpu_aabb_reduction.cu 1, dynamic_grid.cu 0<br/>semantic_forces.cu grew from 15 to 20 when the sf_ launch kernels landed (crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:934-973), and only 8 of the 9 files are built to PTX (build.rs:32-43)
```

## VC-12.6 runtime PTX load, module init and kernel handle lookup
```mermaid
sequenceDiagram
    autonumber
    participant GRA as GPUResourceActor<br/>gpu_resource_actor.rs:41
    participant Loader as ptx_loader<br/>ptx_loader.rs:372
    participant Downgrade as downgrade_ptx_isa_if_needed<br/>ptx_loader.rs:311-336
    participant UGC as UnifiedGPUCompute<br/>construction.rs:252
    participant Cust as cust::module::Module<br/>CUDA driver JIT

    GRA->>Loader: load_ptx_module_sync(PTXModule::VisionflowUnified) (gpu_resource_actor.rs:89-91)
    Loader->>Loader: load_ptx_module_sync_raw - source selection, see VC-12.7<br/>(ptx_loader.rs:379)
    Loader-->>Loader: raw PTX string
    Loader->>Downgrade: downgrade_ptx_isa_if_needed(raw) (ptx_loader.rs:376)
    Downgrade->>Downgrade: detect_max_ptx_isa() via nvidia-smi driver_version<br/>(ptx_loader.rs:91-121)
    alt driver query succeeds, CUDA header probe or approximate driver mapping resolves
        Downgrade-->>Downgrade: RUNTIME_MAX_PTX_ISA OnceLock get_or_init caches the result (ptx_loader.rs:23 static, :92 get_or_init)
    else driver query fails or mapping is unavailable
        Downgrade-->>Downgrade: fallback (9,0) (ptx_loader.rs:103-118)
    end
    Downgrade->>Downgrade: build target PtxVersion, call ptx_policy::rewrite_ptx_version(&ptx, target)<br/>(ptx_loader.rs:338-342)
    alt VersionRewrite::Rewritten { from, to, text }
        Downgrade->>Downgrade: info! ISA downgrade log, return the rewritten text (ptx_loader.rs:343-349)
        Note right of Downgrade: RESOLVED ADR-2056 - no longer a second impl<br/>downgrade_ptx_isa_if_needed now delegates the actual span-parsed splice<br/>to ptx_policy::rewrite_ptx_version (ptx_policy.rs:233-243, :245-253) -<br/>it retains only the runtime driver-ISA probe
    else VersionRewrite::Unchanged | VersionRewrite::Defective
        Downgrade-->>Loader: ptx text unchanged (ptx_loader.rs:353-354)
    end
    Downgrade-->>GRA: Ok(ptx_content) (gpu_resource_actor.rs:89-95)
    opt clustering PTX (PTXModule::GpuClusteringKernels)
        GRA->>Loader: load_ptx_module_sync(GpuClusteringKernels) (gpu_resource_actor.rs:101-103)
        alt load fails
            Loader-->>GRA: Err, warn! will use fallback, clustering_ptx=None (gpu_resource_actor.rs:111-114)
        else load succeeds
            Loader-->>GRA: Some(content) (gpu_resource_actor.rs:104-109)
        end
    end
    opt APSP PTX (PTXModule::GpuLandmarkApsp)
        GRA->>Loader: load_ptx_module_sync(GpuLandmarkApsp) (gpu_resource_actor.rs:126-128)
        alt load fails
            Loader-->>GRA: Err, warn! apsp_ptx=None — NO CPU fallback exists for this path<br/>(gpu_resource_actor.rs:136-139, comment :121-125)
        else load succeeds
            Loader-->>GRA: Some(content) (gpu_resource_actor.rs:129-135)
        end
    end
    Note over Loader,GRA: PTXModule::SemanticForces is the ninth loadable module (ptx_loader.rs:221, all_modules at :274-285)<br/>but GPUResourceActor does not load it, ForceComputeActor::refresh_semantic does on each graph upload<br/>(force_compute_actor.rs:4056-4059) and hands the PTX to UnifiedGPUCompute::enable_semantic (execution.rs:1016)
    GRA->>UGC: new_with_modules(num_nodes, num_edges, ptx, clustering_ptx, apsp_ptx)<br/>(gpu_resource_actor.rs:143) delegates to new_with_all_modules (construction.rs:252-259, :266-272)
    UGC->>UGC: validate_ptx_content(ptx_content) (construction.rs:273, gpu_diagnostics.rs:185)
    alt primary PTX fails structural validation
        UGC-->>GRA: Err PTX validation failed, plus diagnose_ptx_error()<br/>(construction.rs:274-276)
    else Ok
        UGC->>Cust: Module::from_ptx(ptx_content, empty_options) (construction.rs:281)
        alt module load fails
            UGC-->>GRA: Err Module::from_ptx() failed, plus diagnosis (construction.rs:282-285)
        else module loads
            opt clustering_ptx is Some
                UGC->>UGC: validate_ptx_content(clustering_ptx_content) (construction.rs:291-294)
                alt validation fails
                    UGC->>UGC: error! GPU DEGRADED, clustering_module=None<br/>(construction.rs:295-300)
                else validation ok
                    UGC->>Cust: Module::from_ptx(clustering_ptx, opts)<br/>(construction.rs:302)
                    alt secondary load fails
                        UGC->>UGC: error! GPU DEGRADED, clustering_module=None<br/>(construction.rs:307-314)
                    else ok
                        UGC->>UGC: clustering_module=Some(module) (construction.rs:303-306)
                    end
                end
            end
        end
    end
    Note over UGC: DOC-DRIFT (predates this window, commit da2f5cac7) - apsp_ptx is NO LONGER loaded<br/>into a module. REMOVED under ADR-2054: apsp_module had zero read sites (nothing ever<br/>resolved a kernel from it), so the load cost a startup PTX parse and logged a false<br/>"GPU APSP is DISABLED" about a capability NFR-7 forbids permanently. The parameter is<br/>retained and ignored (let _ = apsp_ptx, construction.rs:321-327) so callers need not<br/>change shape. See VC-15 for ShortestPathActor::ComputeAPSP, which never used this module.
    Note over UGC,Cust: get_function(name) resolves a kernel by string<br/>e.g. force_pass/integrate_pass_kernel (execution.rs:657, :943)<br/>a Rust ? propagates the error if absent
    Note right of UGC: INVARIANT - fallback-PTX for nvcc-less builds<br/>secondary-module failures DEGRADE, not crash<br/>docs/GPU-wire-abi.md Invariant 5 at :258-259
    Note right of UGC: DIVERGENCE - get_function by name, ONLY symbol check<br/>unguarded if build-time symbol gate bypassed<br/>e.g. a cached OUT_DIR artefact
```

## VC-12.7 PTX source selection: compiled vs pre-shipped vs runtime-compiled
```mermaid
flowchart TD
    Start["load_ptx_module_sync_raw(module)<br/>ptx_loader.rs:379"] --> DockerCheck{"DOCKER_ENV set?<br/>ptx_loader.rs:380"}
    DockerCheck -->|yes| DockerPre["load_precompiled_ptx(module)<br/>ptx_loader.rs:383"]
    DockerPre --> DockerPreOk{"found and valid?"}
    DockerPreOk -->|yes| ReturnA["Ok(content)<br/>ptx_loader.rs:383-384"]
    DockerPreOk -->|no| DockerCompile["compile_ptx_fallback_sync_module(module)<br/>ptx_loader.rs:388"]
    DockerCompile --> ReturnB["Ok/Err from nvcc runtime compile"]

    DockerCheck -->|no| BakedCheck{"get_compiled_ptx_path(module) resolves?<br/>ptx_loader.rs:391, fn at 292-299"}
    BakedCheck -->|"option_env! baked path (build.rs cargo:rustc-env)"| ReadBaked["fs::read_to_string(path)<br/>ptx_loader.rs:392"]
    BakedCheck -->|"or std::env::var(module.env_var()) runtime override<br/>ptx_loader.rs:298"| ReadBaked
    ReadBaked --> ValidBaked{"validate_ptx ok?<br/>ptx_loader.rs:394"}
    ValidBaked -->|yes| ReturnC["Ok(content) - fastest path, fresh OUT_DIR PTX<br/>ptx_loader.rs:401-402"]
    ValidBaked -->|no| PreCompiled
    BakedCheck -->|"neither set"| PreCompiled["load_precompiled_ptx(module)<br/>ptx_loader.rs:415"]

    PreCompiled --> C1["1. baked compiled_ptx_path option_env!<br/>ptx_loader.rs:440-442"]
    C1 --> C2["1b. runtime env var override module.env_var()<br/>ptx_loader.rs:445-447"]
    C2 --> C3["2. target/release-or-debug/build/star/out/NAME.ptx<br/>newest mtime first<br/>ptx_loader.rs:457-476"]
    C3 --> C4["3a. crates/visionclaw-gpu/src/ptx/NAME.ptx<br/>ptx_loader.rs:482"]
    C4 --> C5["3b. ../visionclaw-gpu/src/ptx/NAME.ptx<br/>ptx_loader.rs:483-485"]
    C5 --> C6["3c. /app/crates/visionclaw-gpu/src/ptx/NAME.ptx<br/>ptx_loader.rs:486"]
    C6 --> C7["3d. legacy /app/src/utils/ptx/NAME.ptx<br/>ptx_loader.rs:488"]
    C7 --> C8["3e. legacy ./src/utils/ptx/NAME.ptx<br/>ptx_loader.rs:489"]
    C8 --> AnyValid{"first candidate that reads and validate_ptx ok?<br/>ptx_loader.rs:492-499"}
    AnyValid -->|yes| ReturnD["Ok(content)<br/>ptx_loader.rs:494-497"]
    AnyValid -->|no candidate found| RuntimeCompile["compile_ptx_fallback_sync_module(module)<br/>ptx_loader.rs:423"]

    RuntimeCompile --> Arch["effective_cuda_arch(): delegates to detect_runtime_cuda_arch()<br/>ptx_loader.rs:307-309"]
    Arch --> CuSrc{"src/utils/NAME.cu exists under CARGO_MANIFEST_DIR?<br/>ptx_loader.rs:560-566"}
    CuSrc -->|no| ErrNoSrc["Err CUDA source not found<br/>ptx_loader.rs:567-571"]
    CuSrc -->|yes| SpawnNvcc["nvcc -ptx -std=c++17 -arch=sm_ARCH NAME.cu -o TMP.ptx<br/>ptx_loader.rs:588-608"]
    SpawnNvcc --> NvccOk{"spawn ok and exit 0?"}
    NvccOk -->|no| ErrNvcc["Err nvcc failed, includes stdout/stderr<br/>ptx_loader.rs:610-623"]
    NvccOk -->|yes| ReadTmp["fs::read_to_string(TMP.ptx)<br/>ptx_loader.rs:625-631"]
    ReadTmp --> ValidateTmp["validate_ptx(ptx_content)<br/>ptx_loader.rs:633"]
    ValidateTmp --> ReturnE["Ok(content)<br/>ptx_loader.rs:634-639"]

    ReturnA --> Downgrade["downgrade_ptx_isa_if_needed(raw) - see VC-12.5<br/>ptx_loader.rs:376"]
    ReturnB --> Downgrade
    ReturnC --> Downgrade
    ReturnD --> Downgrade
    ReturnE --> Downgrade
```
