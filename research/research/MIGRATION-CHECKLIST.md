# Legacy Knowledge Graph - Migration Checklist

**Purpose**: Systematic checklist for preserving critical functionality during modernization

---

## ✅ Phase 1: Core Physics Engine (Week 1-2)

### 1.1 Extract Physics Actor

- [ ] Copy `ForceComputeActor` (1048 LOC) to new codebase
- [ ] Extract `PhysicsStats` struct
- [ ] Extract `ComputeMode` enum (Basic/Advanced/DualGraph/Constraints)
- [ ] Preserve all helper methods:
  - [ ] `perform_force_computation()`
  - [ ] `handle_physics_tick()`
  - [ ] `update_simulation_params()`
  - [ ] `detect_spatial_hashing_issues()`

**Success Criteria**: Actor compiles without errors

### 1.2 Port CUDA Kernels

- [ ] Copy `visionflow_unified.cu` (1887 LOC) verbatim
- [ ] Verify kernel signatures:
  - [ ] `build_grid_kernel`
  - [ ] `force_pass_kernel`
  - [ ] `integrate_pass_kernel`
  - [ ] `calculate_kinetic_energy_kernel`
  - [ ] `calculate_active_nodes_kernel`
- [ ] Copy `SimulationParams` struct (C++ side)
- [ ] Copy `ConstraintData` struct (C++ side)
- [ ] Verify `#include` statements resolve

**Success Criteria**: Kernels compile with nvcc

### 1.3 Implement GPU Resource Sharing

- [ ] Create `SharedGPUContext` struct
- [ ] Implement `RwLock<()>` for concurrency
- [ ] Add `acquire_gpu_access()` method
- [ ] Add utilization tracking:
  - [ ] `VecDeque<(Instant, f32)>` for history
  - [ ] `get_utilization_percentage()` method
- [ ] Wrap `UnifiedGPUCompute` in `Arc<Mutex<>>`

**Success Criteria**: Multiple threads can acquire read locks concurrently

### 1.4 Stability Gates

- [ ] Copy stability detection logic:
  - [ ] `calculate_kinetic_energy_kernel` integration
  - [ ] `calculate_active_nodes_kernel` integration
  - [ ] Thresholds: `stability_threshold`, `min_velocity_threshold`
- [ ] Implement early exit:
  - [ ] Skip physics if `energy_stable || motion_stable`
  - [ ] Track `stability_iterations` counter
- [ ] Add logging for stability state changes

**Success Criteria**: Physics pauses automatically when graph stabilizes

### 1.5 Adaptive Throttling

- [ ] Implement download interval calculation:
  ```rust
  let download_interval = if stable {
      30  // ~2 Hz when stable
  } else if num_nodes > 10000 {
      10  // ~6 Hz for large
  } else if num_nodes > 1000 {
      5   // ~12 Hz for medium
  } else {
      2   // ~30 Hz for small
  };
  ```
- [ ] Add `iteration % download_interval == 0` check
- [ ] Preserve `get_node_positions()` call structure
- [ ] Test with graphs of varying sizes

**Success Criteria**: Download frequency adapts to graph size

### 1.6 Performance Testing

- [ ] Benchmark 100-node graph:
  - [ ] Target: 200-500 FPS
  - [ ] GPU utilization: 60-70%
- [ ] Benchmark 1K-node graph:
  - [ ] Target: 50-100 FPS
  - [ ] GPU utilization: 70-85%
- [ ] Benchmark 10K-node graph:
  - [ ] Target: 20-60 FPS
  - [ ] GPU utilization: 80-90%
- [ ] Verify stability pause works (0% GPU when stable)

**Success Criteria**: Performance matches legacy system ±10%

---

## ✅ Phase 2: Clustering & Analytics (Week 3-4)

### 2.1 K-means Clustering

- [ ] Copy `gpu_clustering_kernels.cu` (1554 LOC)
- [ ] Extract K-means kernels:
  - [ ] `kmeans_plus_plus_init_kernel`
  - [ ] `assign_clusters_kernel`
  - [ ] `update_centroids_kernel`
  - [ ] `calculate_inertia_kernel`
- [ ] Implement K-means actor:
  - [ ] `KMeansRequest` message
  - [ ] `KMeansResult` response
  - [ ] Convergence checking (max iterations + tolerance)
- [ ] Preserve shared memory optimization in centroid update

**Success Criteria**: K-means completes < 200ms @ 10K nodes

### 2.2 LOF Anomaly Detection

- [ ] Copy LOF kernels from `gpu_clustering_kernels.cu`:
  - [ ] `lof_find_knn_kernel` (spatial grid version)
  - [ ] `lof_compute_reachability_kernel`
  - [ ] `lof_compute_lrd_kernel`
  - [ ] `lof_compute_lof_kernel`
- [ ] Implement LOF actor:
  - [ ] `LOFRequest` message (with k parameter)
  - [ ] `LOFResult` response (anomaly scores)
- [ ] Preserve `MAX_K = 32` neighbor limit
- [ ] Test with known anomalous graphs

**Success Criteria**: LOF detects outliers in < 400ms @ 10K nodes

### 2.3 Label Propagation

- [ ] Copy label propagation kernels:
  - [ ] `label_propagation_sync_kernel`
  - [ ] `label_propagation_async_kernel`
  - [ ] `compute_modularity_kernel`
  - [ ] `count_communities_kernel`
  - [ ] `compact_labels_kernel`
- [ ] Implement `LabelPropagationActor`
- [ ] Add tie-breaking with `cuRAND`
- [ ] Preserve weighted neighbor voting

**Success Criteria**: Community detection works with modularity > 0.3

### 2.4 Clustering Actor

- [ ] Create unified `ClusteringActor`
- [ ] Implement message handlers:
  - [ ] `RunKMeans`
  - [ ] `RunLOF`
  - [ ] `RunLabelPropagation`
  - [ ] `GetClusteringStats`
- [ ] Add GPU context sharing via `SharedGPUContext`
- [ ] Implement concurrent query support (read locks)

**Success Criteria**: All clustering methods work concurrently with physics

---

## ✅ Phase 3: SSSP & Constraints (Week 5-6)

### 3.1 SSSP Frontier Compaction

- [ ] Copy `sssp_compact.cu` (106 LOC):
  - [ ] `compact_frontier_atomic_kernel`
  - [ ] `compact_frontier_scan_kernel`
- [ ] Integrate with existing SSSP implementation
- [ ] Test with large frontiers (> 10K nodes)

**Success Criteria**: Compaction is 10-20x faster than CPU

### 3.2 Hybrid SSSP

- [ ] Copy `gpu/hybrid_sssp/gpu_kernels.rs` (376 LOC)
- [ ] Extract hybrid kernels:
  - [ ] `k_step_relaxation_kernel`
  - [ ] `bounded_dijkstra_kernel`
  - [ ] `detect_pivots_kernel`
  - [ ] `partition_frontier_kernel`
- [ ] Implement adaptive switching logic:
  - [ ] Small graphs (< 1K): CPU/WASM Dijkstra
  - [ ] Large graphs (> 1K): GPU k-step relaxation
- [ ] Add pivot selection heuristic

**Success Criteria**: SSSP completes < 300ms @ 10K nodes

### 3.3 Landmark APSP

- [ ] Copy `gpu_landmark_apsp.cu` (152 LOC):
  - [ ] `approximate_apsp_kernel`
  - [ ] `select_landmarks_kernel`
  - [ ] `stress_majorization_barneshut_kernel`
- [ ] Implement landmark selection (stratified sampling)
- [ ] Add triangle inequality approximation
- [ ] Test with k=16 landmarks

**Success Criteria**: APSP approximation is < 10% error vs exact

### 3.4 Constraint System

- [ ] Copy `models/constraints.rs` (200+ LOC):
  - [ ] `ConstraintKind` enum (10 types)
  - [ ] `Constraint` struct
  - [ ] `ConstraintData` GPU format
  - [ ] `AdvancedParams` struct
- [ ] Implement constraint application in `force_pass_kernel`
- [ ] Add progressive activation logic:
  ```cuda
  float progressive_multiplier = (float)frames_since_activation / (float)ramp_frames;
  float effective_weight = constraint.weight * progressive_multiplier;
  ```
- [ ] Preserve constraint telemetry buffers (optional)

**Success Criteria**: Constraints fade in smoothly over 60 frames

### 3.5 Constraint Actor

- [ ] Create `ConstraintActor`
- [ ] Implement message handlers:
  - [ ] `UpdateConstraints`
  - [ ] `ApplyConstraintsToNodes`
  - [ ] `RemoveConstraints`
  - [ ] `GetActiveConstraints`
  - [ ] `GetConstraintStats`
- [ ] Add GPU upload logic (`UploadConstraintsToGPU`)
- [ ] Implement constraint validation

**Success Criteria**: Constraints can be updated in real-time

---

## ✅ Phase 4: Integration & Testing (Week 7-8)

### 4.1 Actor System Integration

- [ ] Wire up message passing:
  - [ ] `GraphServiceActor` → `PhysicsOrchestratorActor`
  - [ ] `PhysicsOrchestratorActor` → `ForceComputeActor`
  - [ ] `ForceComputeActor` ↔ `GPUResourceActor`
- [ ] Implement 60 Hz physics tick
- [ ] Add client broadcast (binary protocol)
- [ ] Test concurrent analytics queries

**Success Criteria**: Full actor pipeline works end-to-end

### 4.2 Database Integration

- [ ] Migrate SQLite schema from `knowledge_graph.db`
- [ ] Preserve physics state columns:
  - [ ] `x, y, z` (position)
  - [ ] `vx, vy, vz` (velocity)
  - [ ] `ax, ay, az` (acceleration)
  - [ ] `mass, charge` (properties)
  - [ ] `is_pinned, pin_x, pin_y, pin_z` (constraints)
- [ ] Implement state persistence on shutdown
- [ ] Implement state restoration on startup

**Success Criteria**: Physics state survives restarts

### 4.3 Unit Tests

- [ ] GPU kernel tests:
  - [ ] Force computation matches CPU reference
  - [ ] Integration stability (energy conservation)
  - [ ] Boundary handling
  - [ ] Constraint application
- [ ] Clustering tests:
  - [ ] K-means convergence
  - [ ] LOF outlier detection
  - [ ] Label propagation modularity
- [ ] SSSP tests:
  - [ ] Correctness vs Dijkstra
  - [ ] Frontier compaction

**Success Criteria**: All unit tests pass

### 4.4 Integration Tests

- [ ] Full physics pipeline:
  - [ ] Load graph → compute forces → integrate → download
  - [ ] Repeat for 1000 iterations
  - [ ] Verify no memory leaks
- [ ] Concurrent operations:
  - [ ] Physics running + clustering query
  - [ ] Physics running + SSSP query
  - [ ] Multiple analytics queries
- [ ] Parameter updates:
  - [ ] Change simulation params mid-run
  - [ ] Switch compute modes
  - [ ] Add/remove constraints

**Success Criteria**: All integration tests pass

### 4.5 Performance Regression Tests

- [ ] Compare FPS vs legacy:
  - [ ] 100 nodes: ±5%
  - [ ] 1K nodes: ±5%
  - [ ] 10K nodes: ±10%
  - [ ] 100K nodes: ±15%
- [ ] Compare GPU utilization vs legacy:
  - [ ] Active state: ±10%
  - [ ] Stable state: 0% (must match exactly)
- [ ] Compare memory usage vs legacy:
  - [ ] GPU memory: ±10%
  - [ ] CPU memory: ±20%

**Success Criteria**: No performance regressions > 15%

### 4.6 Visual Regression Tests

- [ ] Screenshot comparison:
  - [ ] Load reference graph
  - [ ] Run 1000 iterations
  - [ ] Take screenshot
  - [ ] Compare to legacy screenshot (pixel diff)
- [ ] Layout quality metrics:
  - [ ] Edge crossing count
  - [ ] Node overlap percentage
  - [ ] Average edge length variance
  - [ ] Boundary violations

**Success Criteria**: Visual output indistinguishable from legacy

---

## ✅ Phase 5: Documentation & Handoff (Week 9)

### 5.1 User Documentation

- [ ] Create "GPU Physics Tuning Guide":
  - [ ] Recommended parameter ranges
  - [ ] Performance/quality tradeoffs
  - [ ] Troubleshooting common issues
- [ ] Create "Clustering Guide":
  - [ ] When to use K-means vs LOF vs Label Propagation
  - [ ] Parameter selection guidelines
  - [ ] Performance characteristics

**Deliverable**: User-facing documentation

### 5.2 Developer Documentation

- [ ] Create "CUDA Kernel Performance":
  - [ ] Kernel launch parameters
  - [ ] Shared memory usage
  - [ ] Occupancy analysis
  - [ ] Profiling with nvprof
- [ ] Create "Architecture Overview":
  - [ ] Actor model diagram
  - [ ] GPU concurrency model
  - [ ] Message flow diagrams

**Deliverable**: Developer-facing documentation

### 5.3 Migration Report

- [ ] Document preserved features (Tier 1, 2, 3)
- [ ] Document deprecated features (if any)
- [ ] Document performance comparison (before/after)
- [ ] Document known limitations
- [ ] Document future improvement opportunities

**Deliverable**: Executive summary + detailed report

### 5.4 Knowledge Transfer

- [ ] Conduct walkthrough session:
  - [ ] Physics engine internals
  - [ ] GPU kernel optimization techniques
  - [ ] Actor concurrency patterns
  - [ ] Performance tuning methodology
- [ ] Record video tutorials (optional)
- [ ] Create runbook for common operations

**Deliverable**: Team training completed

---

## 🎯 Success Metrics

### Must-Have (Phase 1-3)

- ✅ Physics runs at 60 FPS @ 10K nodes
- ✅ Stability gates work (0% GPU when stable)
- ✅ Adaptive throttling prevents CPU bottleneck
- ✅ K-means clustering < 200ms @ 10K nodes
- ✅ SSSP < 300ms @ 10K nodes
- ✅ Constraints fade in smoothly

### Nice-to-Have (Phase 4-5)

- ✅ Visual output identical to legacy
- ✅ No memory leaks after 24hr run
- ✅ All unit + integration tests pass
- ✅ Documentation complete
- ✅ Team trained

### Critical Failures (Blockers)

- ❌ Physics < 30 FPS @ 10K nodes
- ❌ GPU memory leaks
- ❌ Stability gates don't trigger
- ❌ Adaptive throttling broken (CPU bottleneck)
- ❌ Visual output significantly different

---

## 📋 Daily Checklist Template

### Day N of Migration

**Date**: ___________

**Phase**: ___________ (1-5)

**Tasks Completed**:
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

**Blockers**:
- Issue 1
- Issue 2

**Performance Measurements**:
- FPS @ 10K nodes: _____ (target: 60)
- GPU utilization (active): _____% (target: 80-90%)
- GPU utilization (stable): _____% (target: 0%)
- Memory usage: _____ MB (target: < 1.5 MB)

**Tests Passing**:
- Unit tests: _____/_____
- Integration tests: _____/_____
- Performance tests: _____/_____

**Notes**:
- Observation 1
- Observation 2

---

## 🔧 Rollback Plan

If migration fails, preserve these artifacts for rollback:

1. **Full Legacy Codebase**:
   - [ ] `src/actors/gpu/force_compute_actor.rs`
   - [ ] `src/utils/visionflow_unified.cu`
   - [ ] `src/utils/gpu_clustering_kernels.cu`
   - [ ] `src/utils/sssp_compact.cu`
   - [ ] `src/gpu/hybrid_sssp/gpu_kernels.rs`
   - [ ] `src/utils/gpu_landmark_apsp.cu`

2. **Build System**:
   - [ ] `Cargo.toml` (with cudarc dependencies)
   - [ ] `build.rs` (CUDA compilation logic)

3. **Database Schema**:
   - [ ] `tests/db_analysis/knowledge_graph.db`

4. **Performance Baselines**:
   - [ ] Benchmark results (FPS, GPU %, memory)
   - [ ] Screenshot comparisons

**Rollback Trigger**: If any critical failure occurs in Phase 4

---

## 📊 Progress Tracking

| Phase | Tasks | Completed | % Done | Status |
|-------|-------|-----------|--------|--------|
| Phase 1 | 6 sections | __ | __% | ⏳ Pending |
| Phase 2 | 4 sections | __ | __% | ⏳ Pending |
| Phase 3 | 5 sections | __ | __% | ⏳ Pending |
| Phase 4 | 6 sections | __ | __% | ⏳ Pending |
| Phase 5 | 4 sections | __ | __% | ⏳ Pending |
| **TOTAL** | **25 sections** | **__** | **__%** | ⏳ Pending |

---

## See Also

- **Full Analysis**: `Legacy-Knowledge-Graph-System-Analysis.md`
- **Executive Summary**: `EXECUTIVE-SUMMARY.md`
- **Architecture Diagrams**: `ARCHITECTURE-DIAGRAMS.md`

**Last Updated**: 2025-10-31
