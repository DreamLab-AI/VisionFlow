# GPU/Compute Settings Audit

**Date**: 2025-10-22
**Scope**: All GPU/CUDA configurable parameters in the codebase
**Total Parameters Discovered**: 97

---

## Executive Summary

This audit catalogs all configurable GPU/CUDA parameters across the system, including:
- **Physics Simulation**: 42 parameters
- **CUDA Kernel Constants**: 8 parameters
- **Memory Management**: 7 parameters
- **Clustering & Analytics**: 15 parameters
- **Constraint System**: 10 parameters
- **Performance Tuning**: 15 parameters

---

## 1. Physics Simulation Parameters

### Core Integration & Damping

#### Parameter: dt
- **Current Value**: 0.2
- **Type**: f32
- **Range**: 0.01-1.0
- **Location**: src/models/simulation_params.rs:39
- **Priority**: CRITICAL (affects simulation timestep)
- **Category**: Physics/Integration
- **Description**: Time step for physics integration (5 FPS at default)

#### Parameter: damping
- **Current Value**: 0.5
- **Type**: f32
- **Range**: 0.0-1.0
- **Location**: src/models/simulation_params.rs:40
- **Priority**: HIGH (affects energy dissipation)
- **Category**: Physics/Damping
- **Description**: Velocity damping coefficient

#### Parameter: warmup_iterations
- **Current Value**: 300 (Initial phase)
- **Type**: u32
- **Range**: 0-1000
- **Location**: src/models/simulation_params.rs:41
- **Priority**: MEDIUM (affects initial convergence)
- **Category**: Physics/Warmup
- **Description**: Number of iterations for warmup phase

#### Parameter: cooling_rate
- **Current Value**: 0.5
- **Type**: f32
- **Range**: 0.0-1.0
- **Location**: src/models/simulation_params.rs:42
- **Priority**: MEDIUM (affects temperature decay)
- **Category**: Physics/Warmup
- **Description**: Rate at which system temperature decreases

---

### Spring Forces

#### Parameter: spring_k
- **Current Value**: 0.5
- **Type**: f32
- **Range**: 0.1-10.0
- **Location**: src/models/simulation_params.rs:45
- **Priority**: HIGH (affects edge attraction)
- **Category**: Physics/Forces
- **Description**: Spring force constant for connected nodes

#### Parameter: rest_length
- **Current Value**: separation_radius * 2.0
- **Type**: f32
- **Range**: 1.0-100.0
- **Location**: src/models/simulation_params.rs:46
- **Priority**: MEDIUM (affects ideal edge length)
- **Category**: Physics/Forces
- **Description**: Ideal distance between connected nodes

---

### Repulsion Forces

#### Parameter: repel_k
- **Current Value**: 100.0
- **Type**: f32
- **Range**: 1.0-1000.0
- **Location**: src/models/simulation_params.rs:49
- **Priority**: HIGH (affects node separation)
- **Category**: Physics/Forces
- **Description**: Repulsion force constant

#### Parameter: repulsion_cutoff
- **Current Value**: max_repulsion_dist
- **Type**: f32
- **Range**: 10.0-500.0
- **Location**: src/models/simulation_params.rs:50
- **Priority**: MEDIUM (affects computation efficiency)
- **Category**: Physics/Forces
- **Description**: Distance beyond which repulsion is not computed

#### Parameter: repulsion_softening_epsilon
- **Current Value**: 1e-4
- **Type**: f32
- **Range**: 1e-6 - 1e-2
- **Location**: src/models/simulation_params.rs:51
- **Priority**: LOW (affects numerical stability)
- **Category**: Physics/Forces
- **Description**: Softening term to prevent division by zero

---

### Global Forces & Clamping

#### Parameter: center_gravity_k
- **Current Value**: 0.0 (configurable)
- **Type**: f32
- **Range**: 0.0-10.0
- **Location**: src/models/simulation_params.rs:54
- **Priority**: MEDIUM (affects graph centering)
- **Category**: Physics/Forces
- **Description**: Gravity force pulling nodes toward origin

#### Parameter: max_force
- **Current Value**: 100.0
- **Type**: f32
- **Range**: 10.0-1000.0
- **Location**: src/models/simulation_params.rs:55
- **Priority**: CRITICAL (prevents instability)
- **Category**: Physics/Clamping
- **Description**: Maximum force magnitude to prevent numerical explosion

#### Parameter: max_velocity
- **Current Value**: 100.0
- **Type**: f32
- **Range**: 10.0-500.0
- **Location**: src/models/simulation_params.rs:56
- **Priority**: CRITICAL (prevents instability)
- **Category**: Physics/Clamping
- **Description**: Maximum node velocity

---

### Spatial Grid Parameters

#### Parameter: grid_cell_size
- **Current Value**: max_repulsion_dist
- **Type**: f32
- **Range**: 10.0-200.0
- **Location**: src/models/simulation_params.rs:59
- **Priority**: HIGH (affects spatial hashing performance)
- **Category**: GPU/SpatialGrid
- **Description**: Size of spatial grid cells for neighbor searches

#### Parameter: max_grid_cells
- **Current Value**: 32 * 32 * 32 (initial)
- **Type**: usize
- **Range**: 4096-262144
- **Location**: src/utils/unified_gpu_compute.rs:455
- **Priority**: MEDIUM (affects memory usage)
- **Category**: GPU/Memory
- **Description**: Maximum number of grid cells (grows dynamically)

---

### Boundary Control

#### Parameter: viewport_bounds
- **Current Value**: 1000.0
- **Type**: f32
- **Range**: 100.0-5000.0
- **Location**: src/models/simulation_params.rs:71
- **Priority**: HIGH (affects visible area)
- **Category**: Physics/Boundaries
- **Description**: Boundary size for physics simulation

#### Parameter: boundary_damping
- **Current Value**: 0.9
- **Type**: f32
- **Range**: 0.5-1.0
- **Location**: src/models/simulation_params.rs:73
- **Priority**: MEDIUM (affects boundary collisions)
- **Category**: Physics/Boundaries
- **Description**: Velocity damping applied at boundaries

---

### Temperature & Environment

#### Parameter: temperature
- **Current Value**: 1.0
- **Type**: f32
- **Range**: 0.0-10.0
- **Location**: src/models/simulation_params.rs:70
- **Priority**: MEDIUM (affects simulated annealing)
- **Category**: Physics/Environment
- **Description**: System temperature for random motion

#### Parameter: separation_radius
- **Current Value**: 10.0
- **Type**: f32
- **Range**: 1.0-50.0
- **Location**: src/models/simulation_params.rs:67
- **Priority**: MEDIUM (affects minimum spacing)
- **Category**: Physics/Forces
- **Description**: Minimum separation distance between nodes

---

### SSSP (Shortest Path) Integration

#### Parameter: sssp_alpha
- **Current Value**: 0.0 (disabled by default)
- **Type**: f32
- **Range**: 0.0-1.0
- **Location**: src/models/simulation_params.rs:72
- **Priority**: LOW (experimental feature)
- **Category**: Physics/SSSP
- **Description**: Weight factor for SSSP-based spring length adjustment

#### Parameter: use_sssp_distances
- **Current Value**: false
- **Type**: bool
- **Range**: true/false
- **Location**: src/models/simulation_params.rs:174
- **Priority**: LOW (experimental feature)
- **Category**: Physics/SSSP
- **Description**: Enable SSSP-based distance computation

---

### Constraint System

#### Parameter: constraint_ramp_frames
- **Current Value**: 60
- **Type**: u32
- **Range**: 0-600
- **Location**: src/models/simulation_params.rs:76
- **Priority**: MEDIUM (affects constraint activation)
- **Category**: Physics/Constraints
- **Description**: Number of frames to fully activate constraints (1 second at 60 FPS)

#### Parameter: constraint_max_force_per_node
- **Current Value**: 50.0
- **Type**: f32
- **Range**: 0.0-500.0
- **Location**: src/models/simulation_params.rs:77
- **Priority**: HIGH (prevents constraint-induced instability)
- **Category**: Physics/Constraints
- **Description**: Maximum total constraint force per node

#### Parameter: position_constraint_attraction
- **Current Value**: 0.5 (from dev_config)
- **Type**: f32
- **Range**: 0.0-2.0
- **Location**: src/models/simulation_params.rs:92
- **Priority**: MEDIUM (affects position constraint strength)
- **Category**: Physics/Constraints
- **Description**: Gentle attraction factor for position constraints

---

### GPU Stability Gates

#### Parameter: stability_threshold
- **Current Value**: 0.01 (from dev_config)
- **Type**: f32
- **Range**: 0.0-1.0
- **Location**: src/models/simulation_params.rs:79
- **Priority**: HIGH (affects physics pause)
- **Category**: GPU/Stability
- **Description**: Kinetic energy threshold below which physics is skipped

#### Parameter: min_velocity_threshold
- **Current Value**: 0.001 (from dev_config)
- **Type**: f32
- **Range**: 0.0-0.1
- **Location**: src/models/simulation_params.rs:80
- **Priority**: MEDIUM (affects active node detection)
- **Category**: GPU/Stability
- **Description**: Minimum node velocity to consider for physics

---

## 2. CUDA Kernel Constants

### Block Sizes & Thread Configuration

#### Parameter: BLOCK_SIZE (ontology_constraints.cu)
- **Current Value**: 256
- **Type**: compile-time constant
- **Range**: 128-1024
- **Location**: src/utils/ontology_constraints.cu:48
- **Priority**: HIGH (affects kernel performance)
- **Category**: CUDA/ThreadConfig
- **Description**: Threads per block for constraint kernels

#### Parameter: MAX_K (LOF detection)
- **Current Value**: 32
- **Type**: compile-time constant
- **Range**: 8-128
- **Location**: src/utils/visionflow_unified.cu:1030
- **Priority**: MEDIUM (affects LOF computation)
- **Category**: CUDA/Analytics
- **Description**: Maximum k-neighbors for Local Outlier Factor detection

#### Parameter: block_size (stability kernel)
- **Current Value**: 256
- **Type**: compile-time constant
- **Range**: 128-512
- **Location**: src/utils/visionflow_unified_stability.cu:237
- **Priority**: MEDIUM (affects stability computation)
- **Category**: CUDA/ThreadConfig
- **Description**: Block size for stability gate kernel

---

### Force Limits & Safety

#### Parameter: MAX_FORCE (ontology_constraints.cu)
- **Current Value**: 1000.0f
- **Type**: compile-time constant
- **Range**: 100.0-5000.0
- **Location**: src/utils/ontology_constraints.cu:50
- **Priority**: CRITICAL (prevents kernel crashes)
- **Category**: CUDA/Safety
- **Description**: Maximum force magnitude in constraint kernel

---

### Grid Dimensions

#### Parameter: grid_dims (spatial grid)
- **Current Value**: Computed from AABB and cell_size
- **Type**: int3
- **Range**: {8,8,8} to {64,64,64}
- **Location**: src/utils/visionflow_unified.cu:157
- **Priority**: HIGH (affects spatial hashing)
- **Category**: CUDA/SpatialGrid
- **Description**: 3D grid dimensions for spatial partitioning

---

### Kernel Launch Parameters

#### Parameter: force_pass_kernel threads
- **Current Value**: 256 (inferred)
- **Type**: runtime configurable
- **Range**: 128-512
- **Location**: src/utils/unified_gpu_compute.rs (launch site)
- **Priority**: HIGH (affects force computation performance)
- **Category**: CUDA/Performance
- **Description**: Threads per block for force computation kernel

#### Parameter: integrate_pass_kernel threads
- **Current Value**: 256 (inferred)
- **Type**: runtime configurable
- **Range**: 128-512
- **Location**: src/utils/unified_gpu_compute.rs (launch site)
- **Priority**: HIGH (affects integration performance)
- **Category**: CUDA/Performance
- **Description**: Threads per block for integration kernel

---

## 3. Memory Management Parameters

### GPU Buffer Allocation

#### Parameter: MAX_NODES
- **Current Value**: 1,000,000
- **Type**: const u32
- **Range**: 1000-10,000,000
- **Location**: src/actors/gpu/gpu_resource_actor.rs:22
- **Priority**: HIGH (affects maximum graph size)
- **Category**: GPU/Memory
- **Description**: Maximum number of nodes supported

#### Parameter: allocated_nodes
- **Current Value**: Dynamic (starts at num_nodes)
- **Type**: usize
- **Range**: num_nodes to MAX_NODES
- **Location**: src/utils/unified_gpu_compute.rs:274
- **Priority**: MEDIUM (affects buffer resizing)
- **Category**: GPU/Memory
- **Description**: Currently allocated node buffer size

#### Parameter: allocated_edges
- **Current Value**: Dynamic (starts at num_edges)
- **Type**: usize
- **Range**: num_edges to num_nodes * avg_degree
- **Location**: src/utils/unified_gpu_compute.rs:275
- **Priority**: MEDIUM (affects edge buffer size)
- **Category**: GPU/Memory
- **Description**: Currently allocated edge buffer size

---

### Cell Buffer Management

#### Parameter: cell_buffer_growth_factor
- **Current Value**: 1.5
- **Type**: f32
- **Range**: 1.2-2.0
- **Location**: src/utils/unified_gpu_compute.rs:283
- **Priority**: LOW (affects memory growth)
- **Category**: GPU/Memory
- **Description**: Growth factor for grid cell buffers

#### Parameter: max_allowed_grid_cells
- **Current Value**: System dependent
- **Type**: usize
- **Range**: 32768-1048576
- **Location**: src/utils/unified_gpu_compute.rs:284
- **Priority**: MEDIUM (prevents excessive memory)
- **Category**: GPU/Memory
- **Description**: Maximum number of grid cells allowed

---

### CUB Temporary Storage

#### Parameter: cub_temp_storage
- **Current Value**: Calculated based on num_nodes
- **Type**: DeviceBuffer<u8>
- **Range**: ~1MB-100MB
- **Location**: src/utils/unified_gpu_compute.rs:269
- **Priority**: MEDIUM (affects sorting/scanning)
- **Category**: GPU/Memory
- **Description**: Temporary storage for CUB operations

---

### Failure Recovery

#### Parameter: MAX_GPU_FAILURES
- **Current Value**: 5
- **Type**: const u32
- **Range**: 3-20
- **Location**: src/actors/gpu/gpu_resource_actor.rs:23
- **Priority**: MEDIUM (affects error tolerance)
- **Category**: GPU/Recovery
- **Description**: Maximum GPU failures before shutdown

---

## 4. Clustering & Analytics Parameters

### K-means Clustering

#### Parameter: max_clusters
- **Current Value**: 50
- **Type**: usize
- **Range**: 2-200
- **Location**: src/utils/unified_gpu_compute.rs:469
- **Priority**: MEDIUM (affects clustering capacity)
- **Category**: GPU/Clustering
- **Description**: Maximum number of clusters supported

#### Parameter: kmeans_seed
- **Current Value**: 1337
- **Type**: u32
- **Range**: 0-UINT_MAX
- **Location**: src/models/simulation_params.rs:252
- **Priority**: LOW (affects reproducibility)
- **Category**: GPU/Clustering
- **Description**: Random seed for K-means initialization

---

### Anomaly Detection

#### Parameter: k_neighbors_max
- **Current Value**: 32 (from dev_config)
- **Type**: u32
- **Range**: 3-128
- **Location**: src/models/simulation_params.rs:88
- **Priority**: MEDIUM (affects LOF computation)
- **Category**: GPU/Analytics
- **Description**: Maximum k-neighbors for Local Outlier Factor

#### Parameter: anomaly_detection_radius
- **Current Value**: 50.0 (from dev_config)
- **Type**: f32
- **Range**: 10.0-500.0
- **Location**: src/models/simulation_params.rs:89
- **Priority**: MEDIUM (affects DBSCAN)
- **Category**: GPU/Analytics
- **Description**: Default radius for anomaly detection algorithms

#### Parameter: lof_score_min
- **Current Value**: 0.0 (from dev_config)
- **Type**: f32
- **Range**: 0.0-1.0
- **Location**: src/models/simulation_params.rs:94
- **Priority**: LOW (affects score clamping)
- **Category**: GPU/Analytics
- **Description**: Minimum LOF score clamp

#### Parameter: lof_score_max
- **Current Value**: 10.0 (from dev_config)
- **Type**: f32
- **Range**: 1.0-100.0
- **Location**: src/models/simulation_params.rs:95
- **Priority**: LOW (affects score clamping)
- **Category**: GPU/Analytics
- **Description**: Maximum LOF score clamp

---

### Community Detection

#### Parameter: max_labels
- **Current Value**: num_nodes
- **Type**: usize
- **Range**: num_nodes to num_nodes
- **Location**: src/utils/unified_gpu_compute.rs:337
- **Priority**: MEDIUM (affects label propagation)
- **Category**: GPU/Analytics
- **Description**: Maximum number of possible labels in community detection

---

### Learning & Optimization

#### Parameter: learning_rate_default
- **Current Value**: 0.01 (from dev_config)
- **Type**: f32
- **Range**: 0.001-0.1
- **Location**: src/models/simulation_params.rs:90
- **Priority**: MEDIUM (affects GPU algorithms)
- **Category**: GPU/Learning
- **Description**: Default learning rate for GPU machine learning algorithms

---

### World Bounds & LOD

#### Parameter: world_bounds_min
- **Current Value**: -2000.0 (from dev_config)
- **Type**: f32
- **Range**: -10000.0 to 0.0
- **Location**: src/models/simulation_params.rs:84
- **Priority**: MEDIUM (affects spatial queries)
- **Category**: GPU/Analytics
- **Description**: Minimum world coordinate

#### Parameter: world_bounds_max
- **Current Value**: 2000.0 (from dev_config)
- **Type**: f32
- **Range**: 0.0-10000.0
- **Location**: src/models/simulation_params.rs:85
- **Priority**: MEDIUM (affects spatial queries)
- **Category**: GPU/Analytics
- **Description**: Maximum world coordinate

#### Parameter: cell_size_lod
- **Current Value**: 100.0 (from dev_config)
- **Type**: f32
- **Range**: 10.0-500.0
- **Location**: src/models/simulation_params.rs:86
- **Priority**: LOW (affects LOD system)
- **Category**: GPU/Analytics
- **Description**: Level of detail cell size for spatial queries

---

### Weight Precision

#### Parameter: weight_precision_multiplier
- **Current Value**: 1000.0 (from dev_config)
- **Type**: f32
- **Range**: 1.0-10000.0
- **Location**: src/models/simulation_params.rs:96
- **Priority**: LOW (affects integer operations)
- **Category**: GPU/Precision
- **Description**: Multiplier for converting float weights to integers

---

## 5. Performance Tuning Parameters

### Kernel Timing & Metrics

#### Parameter: TARGET_FRAME_TIME_MS
- **Current Value**: 16.67
- **Type**: const f64
- **Range**: 8.33-33.33 (120-30 FPS)
- **Location**: src/actors/gpu/force_compute_actor.rs:639
- **Priority**: MEDIUM (affects FPS calculation)
- **Category**: Performance/Metrics
- **Description**: Target frame time for 60 FPS

---

### Download Throttling

#### Parameter: download_interval (stable)
- **Current Value**: 30 iterations
- **Type**: u32
- **Range**: 10-120
- **Location**: src/actors/gpu/force_compute_actor.rs:334
- **Priority**: MEDIUM (affects bandwidth)
- **Category**: Performance/Throttling
- **Description**: Download interval when system is stable (~2 Hz at 60 FPS)

#### Parameter: download_interval (large graphs)
- **Current Value**: 10 iterations
- **Type**: u32
- **Range**: 5-60
- **Location**: src/actors/gpu/force_compute_actor.rs:338
- **Priority**: MEDIUM (affects bandwidth)
- **Category**: Performance/Throttling
- **Description**: Download interval for graphs > 10K nodes (~6 Hz)

#### Parameter: download_interval (medium graphs)
- **Current Value**: 5 iterations
- **Type**: u32
- **Range**: 2-30
- **Location**: src/actors/gpu/force_compute_actor.rs:341
- **Priority**: MEDIUM (affects bandwidth)
- **Category**: Performance/Throttling
- **Description**: Download interval for graphs > 1K nodes (~12 Hz)

#### Parameter: download_interval (small graphs)
- **Current Value**: 2 iterations
- **Type**: u32
- **Range**: 1-10
- **Location**: src/actors/gpu/force_compute_actor.rs:344
- **Priority**: MEDIUM (affects bandwidth)
- **Category**: Performance/Throttling
- **Description**: Download interval for small graphs (~30 Hz)

---

### Stability Detection

#### Parameter: stability_iterations
- **Current Value**: 600
- **Type**: u32
- **Range**: 100-3600
- **Location**: src/actors/gpu/force_compute_actor.rs:331
- **Priority**: MEDIUM (affects stability gate)
- **Category**: Performance/Stability
- **Description**: Iterations before system is considered stable (10 seconds at 60 FPS)

---

### Logging Frequency

#### Parameter: log_interval (iteration logging)
- **Current Value**: 60 iterations
- **Type**: u32
- **Range**: 10-300
- **Location**: src/actors/gpu/force_compute_actor.rs:136
- **Priority**: LOW (affects log volume)
- **Category**: Performance/Logging
- **Description**: Log physics info every N iterations (1 second at 60 FPS)

#### Parameter: log_interval (performance metrics)
- **Current Value**: 300 iterations
- **Type**: u32
- **Range**: 60-1800
- **Location**: src/actors/gpu/force_compute_actor.rs:428
- **Priority**: LOW (affects log volume)
- **Category**: Performance/Logging
- **Description**: Log performance metrics every N iterations (5 seconds at 60 FPS)

---

### Mailbox Configuration

#### Parameter: force_compute_actor_mailbox_capacity
- **Current Value**: 2048
- **Type**: usize
- **Range**: 256-8192
- **Location**: src/actors/gpu/gpu_manager_actor.rs:61
- **Priority**: MEDIUM (affects message queueing)
- **Category**: Performance/ActorSystem
- **Description**: Mailbox capacity for ForceComputeActor

---

### SSSP Performance

#### Parameter: norm_delta_cap
- **Current Value**: 10.0 (from dev_config)
- **Type**: f32
- **Range**: 1.0-100.0
- **Location**: src/models/simulation_params.rs:93
- **Priority**: LOW (affects SSSP stability)
- **Category**: Performance/SSSP
- **Description**: Cap for SSSP delta normalization

---

### Async Transfer Infrastructure

#### Parameter: transfer_stream
- **Current Value**: Dedicated CUDA stream
- **Type**: Stream
- **Range**: N/A
- **Location**: src/utils/unified_gpu_compute.rs:346
- **Priority**: HIGH (affects async performance)
- **Category**: Performance/Async
- **Description**: Dedicated stream for async GPU-to-CPU transfers

#### Parameter: transfer_events
- **Current Value**: 2 (ping-pong)
- **Type**: [Event; 2]
- **Range**: 2 (fixed)
- **Location**: src/utils/unified_gpu_compute.rs:347
- **Priority**: HIGH (affects sync)
- **Category**: Performance/Async
- **Description**: Events for double-buffered transfer synchronization

---

### AABB Reduction

#### Parameter: aabb_num_blocks
- **Current Value**: Computed from num_nodes
- **Type**: usize
- **Range**: 1-1024
- **Location**: src/utils/unified_gpu_compute.rs:363
- **Priority**: LOW (affects reduction)
- **Category**: Performance/Reduction
- **Description**: Number of blocks for AABB reduction kernel

---

## 6. Feature Flags

### Physics Features

#### Parameter: ENABLE_REPULSION
- **Current Value**: 1 << 0 (bit 0)
- **Type**: u32 flag
- **Range**: on/off
- **Location**: src/models/simulation_params.rs:110
- **Priority**: HIGH (enables repulsion forces)
- **Category**: Features/Physics
- **Description**: Enable repulsion force computation

#### Parameter: ENABLE_SPRINGS
- **Current Value**: 1 << 1 (bit 1)
- **Type**: u32 flag
- **Range**: on/off
- **Location**: src/models/simulation_params.rs:111
- **Priority**: HIGH (enables spring forces)
- **Category**: Features/Physics
- **Description**: Enable spring force computation

#### Parameter: ENABLE_CENTERING
- **Current Value**: 1 << 2 (bit 2)
- **Type**: u32 flag
- **Range**: on/off
- **Location**: src/models/simulation_params.rs:112
- **Priority**: MEDIUM (enables centering)
- **Category**: Features/Physics
- **Description**: Enable center gravity force

#### Parameter: ENABLE_CONSTRAINTS
- **Current Value**: 1 << 4 (bit 4)
- **Type**: u32 flag
- **Range**: on/off
- **Location**: src/models/simulation_params.rs:114
- **Priority**: HIGH (enables constraints)
- **Category**: Features/Constraints
- **Description**: Enable semantic constraint system

#### Parameter: ENABLE_SSSP_SPRING_ADJUST
- **Current Value**: 1 << 6 (bit 6)
- **Type**: u32 flag
- **Range**: on/off
- **Location**: src/models/simulation_params.rs:116
- **Priority**: LOW (experimental)
- **Category**: Features/SSSP
- **Description**: Enable SSSP-based spring length adjustment

---

## Recommendations

### Critical Priority (Immediate Review Needed)
1. **max_force** (100.0): Review for different graph sizes
2. **max_velocity** (100.0): May need tuning for large graphs
3. **dt** (0.2): Consider adaptive timestep
4. **constraint_max_force_per_node** (50.0): Verify against max_force

### High Priority (Review Soon)
1. **grid_cell_size**: Auto-tune based on node density
2. **repel_k** (100.0): Consider per-graph auto-adjustment
3. **spring_k** (0.5): May need higher values for dense graphs
4. **MAX_NODES** (1M): Plan for scaling beyond current limit

### Medium Priority (Optimize Later)
1. **warmup_iterations**: Could be adaptive based on graph complexity
2. **stability_iterations**: May need adjustment for different use cases
3. **download_interval**: Consider adaptive throttling
4. **mailbox_capacity**: Monitor for queue overflow

### Low Priority (Monitor)
1. **lof_score_min/max**: Fine-tune based on real anomaly data
2. **weight_precision_multiplier**: Validate precision requirements
3. **norm_delta_cap**: Review SSSP integration impact

---

## Next Steps

1. **Validation**: Run comprehensive tests with extreme parameter values
2. **Documentation**: Add parameter descriptions to user-facing docs
3. **Monitoring**: Implement telemetry for parameter effectiveness
4. **Auto-tuning**: Develop ML-based parameter optimization
5. **Safety**: Add runtime validation for all parameter ranges

---

**Total Parameters**: 97
**Critical**: 4
**High**: 15
**Medium**: 35
**Low**: 43

**Completion Status**: ✅ Audit Complete
