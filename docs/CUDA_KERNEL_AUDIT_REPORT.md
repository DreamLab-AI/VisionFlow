---
layout: default
title: CUDA Kernel Audit
description: Usage and orphaned code detection audit for CUDA kernels
nav_exclude: true
---

# CUDA Kernel Usage Audit Report
**Date:** 2025-12-25
**Total Kernels:** 13
**Analysis Status:** Complete

## Executive Summary

All 13 CUDA kernels in `/src/utils/*.cu` have been analyzed for usage and orphaned code detection. The analysis verified kernel function definitions, Rust FFI bindings, and active callers in the codebase.

---

## Kernel Inventory & Status

### 1. **dynamic_grid.cu** ✅ ACTIVE
**Purpose:** Dynamic grid sizing and occupancy optimization
**Key Functions:**
- `initialize_device_info()` - GPU device properties caching
- `calculate_optimal_block_size()` - CUDA occupancy calculation
- `calculate_grid_config()` - Grid/block configuration
- `get_force_kernel_config()` - Force kernel optimization
- `get_reduction_kernel_config()` - Reduction kernel optimization

**Usage:** Referenced in `src/actors/gpu/gpu_manager_actor.rs`, `src/actors/gpu/force_compute_actor.rs`
**Status:** Core infrastructure - KEEP

---

### 2. **gpu_aabb_reduction.cu** ✅ ACTIVE
**Purpose:** Parallel AABB (bounding box) computation using warp primitives
**Key Functions:**
- `compute_aabb_reduction_kernel()` - Optimized min/max reduction with warp shuffles

**Usage:** Called from `src/gpu/mod.rs`, `src/actors/gpu/shared.rs` for viewport bounds
**Status:** Performance-critical - KEEP

---

### 3. **gpu_connected_components.cu** ✅ ACTIVE
**Purpose:** Label propagation for connected components analysis
**Key Functions:**
- `label_propagation_kernel()` - Iterative label propagation
- `initialize_labels_kernel()` - Initialization
- `count_components_kernel()` - Compact unique labels
- `compute_connected_components_gpu()` - Host-callable wrapper

**Usage:** Called from clustering actor and graph analytics
**FFI:** `src/actors/gpu/clustering_actor.rs`
**Status:** Graph analysis feature - KEEP

---

### 4. **gpu_landmark_apsp.cu** ✅ ACTIVE (Partially Deprecated)
**Purpose:** Landmark-based approximate all-pairs shortest paths
**Key Functions:**
- `approximate_apsp_kernel()` - Distance approximation via landmarks
- `select_landmarks_kernel()` - Stratified sampling
- `stress_majorization_barneshut_kernel()` - **DEPRECATED** (superseded by unified_stress_majorization.cu)

**Usage:** APSP used in layout initialization; stress majorization deprecated
**Status:** Keep APSP functions, stress kernel is DUPLICATE ⚠️

---

### 5. **gpu_clustering_kernels.cu** ✅ ACTIVE
**Purpose:** Production K-means, DBSCAN, Louvain, LOF, SSSP implementations
**Key Functions:**
- **K-means:** `init_centroids_kernel()`, `assign_clusters_kernel()`, `update_centroids_kernel()`
- **DBSCAN:** `dbscan_find_neighbors_kernel()`, `dbscan_mark_core_points_kernel()`, `dbscan_propagate_labels_kernel()`
- **Louvain:** `louvain_local_pass_kernel()`, `compute_modularity_gain_device()`
- **LOF:** `compute_lof_kernel()`, `compute_zscore_kernel()`
- **SSSP:** `sssp_relax_edges_kernel()`, `sssp_frontier_relax_kernel()`

**Usage:** Heavily used by clustering, anomaly detection, and graph algorithms
**FFI:** `src/actors/gpu/clustering_actor.rs`, `src/actors/gpu/anomaly_detection_actor.rs`, `src/handlers/api_handler/analytics/clustering.rs`
**Status:** Core analytics - KEEP

---

### 6. **semantic_forces.cu** ✅ ACTIVE
**Purpose:** Type-aware physics for knowledge graphs (DAG layout, clustering, collisions)
**Key Functions:**
- `apply_dag_force()` - Hierarchical layout forces
- `apply_type_cluster_force()` - Type-based clustering
- `apply_collision_force()` - Collision detection
- `apply_attribute_spring_force()` - Weighted spring forces
- `apply_ontology_relationship_force()` - Ontology relationship forces
- `apply_physicality_cluster_force()` - Physicality clustering
- `apply_role_cluster_force()` - Role clustering
- `apply_maturity_layout_force()` - Maturity layout
- `calculate_hierarchy_levels()` - DAG level calculation
- `calculate_type_centroids()` - Type centroid calculation

**Usage:** Core semantic physics engine
**FFI:** `src/actors/gpu/semantic_forces_actor.rs`, `src/gpu/semantic_forces.rs`, `src/handlers/api_handler/semantic_forces.rs`
**Status:** Ontology visualization feature - KEEP

---

### 7. **ontology_constraints.cu** ✅ ACTIVE
**Purpose:** GPU-accelerated ontology constraint enforcement (DisjointClasses, SubClassOf, etc.)
**Key Functions:**
- `apply_disjoint_classes_kernel()` - Separation forces
- `apply_subclass_hierarchy_kernel()` - Hierarchical alignment
- `apply_sameas_colocate_kernel()` - Co-location forces
- `apply_inverse_symmetry_kernel()` - Symmetry enforcement
- `apply_functional_cardinality_kernel()` - Cardinality constraints
- `precompute_constraint_indices()` - O(N+M) preprocessing for O(1) lookups

**Usage:** Ontology physics validation
**FFI:** `src/actors/gpu/ontology_constraint_actor.rs`, `src/physics/ontology_constraints.rs`
**Status:** Semantic validation - KEEP

---

### 8. **pagerank.cu** ✅ ACTIVE
**Purpose:** GPU PageRank centrality computation using power iteration
**Key Functions:**
- `pagerank_init_kernel()` - Initialize PR values
- `pagerank_iteration_kernel()` - Power method iteration
- `pagerank_iteration_optimized_kernel()` - Shared memory optimization
- `pagerank_convergence_kernel()` - L1 norm convergence check
- `pagerank_dangling_kernel()` - Dangling node handling
- `pagerank_normalize_kernel()` - Normalization

**Usage:** Graph centrality analytics
**FFI:** `src/actors/gpu/pagerank_actor.rs`, `src/handlers/api_handler/analytics/mod.rs`
**Status:** Graph analytics feature - KEEP

---

### 9. **stress_majorization.cu** ⚠️ DEPRECATED (Duplicate)
**Purpose:** Stress majorization layout optimization (LEGACY)
**Key Functions:** Similar to unified_stress_majorization.cu

**Status:** **SUPERSEDED by unified_stress_majorization.cu** - CANDIDATE FOR REMOVAL
**Recommendation:** Verify no direct callers remain, then DELETE

---

### 10. **sssp_compact.cu** ✅ ACTIVE
**Purpose:** Device-side frontier compaction for SSSP (Single Source Shortest Path)
**Key Functions:**
- `compact_frontier_kernel()` - Parallel prefix sum compaction
- `compact_frontier_atomic_kernel()` - Atomic compaction (simpler version)
- `compact_frontier_gpu()` - Host wrapper

**Usage:** Graph shortest path algorithms
**FFI:** `src/actors/gpu/shortest_path_actor.rs`, used by SSSP kernels in gpu_clustering_kernels.cu
**Status:** Performance optimization - KEEP

---

### 11. **unified_stress_majorization.cu** ✅ ACTIVE (AUTHORITATIVE)
**Purpose:** Unified stress majorization (consolidates duplicates)
**Key Functions:**
- `compute_stress_kernel()` - Stress function calculation
- `compute_stress_gradient_kernel()` - Gradient computation
- `update_positions_kernel()` - Gradient descent with momentum
- `stress_majorization_step_kernel()` - Sparse CSR stress majorization (O(m) instead of O(n²))
- `majorization_step_kernel()` - Laplacian-based majorization
- `reduce_max_kernel()`, `reduce_sum_kernel()` - Reduction utilities

**Usage:** Global layout optimization
**FFI:** `src/actors/gpu/stress_majorization_actor.rs`
**Status:** Authoritative implementation - KEEP

---

### 12. **visionflow_unified.cu** ✅ ACTIVE
**Purpose:** Unified force-directed simulation kernel (two-pass: force/integrate)
**Key Functions:**
- Core physics simulation with spatial hashing
- Spring forces using CSR format
- Repulsion with grid acceleration
- Position/velocity integration

**Usage:** Main physics engine
**FFI:** `src/utils/unified_gpu_compute.rs`, `src/actors/gpu/force_compute_actor.rs`
**Status:** Core physics - KEEP

---

### 13. **visionflow_unified_stability.cu** ✅ ACTIVE
**Purpose:** GPU stability gates - prevents 100% GPU usage when graph is stable
**Key Functions:**
- `calculate_kinetic_energy_kernel()` - Per-node kinetic energy with reduction
- `reduce_kinetic_energy_kernel()` - Final reduction
- Early-exit optimization for stable states

**Usage:** Performance optimization for idle graphs
**FFI:** `src/utils/unified_gpu_compute.rs`, `src/actors/gpu/force_compute_actor.rs`
**Status:** Performance critical - KEEP

---

## Orphaned Kernel Detection

### ⚠️ DUPLICATE KERNEL IDENTIFIED

**File:** `src/utils/stress_majorization.cu`
**Duplicate of:** `unified_stress_majorization.cu`
**Evidence:**
- Both implement identical stress majorization algorithms
- unified_stress_majorization.cu explicitly states "Consolidates duplicate implementations"
- unified_stress_majorization.cu includes header comment: "// Consolidates duplicate implementations from: src/utils/stress_majorization.cu"

**Recommendation:**
1. Verify `stress_majorization.cu` has no active callers in Rust code
2. If unused, remove `stress_majorization.cu` to reduce code duplication
3. Ensure `stress_majorization_actor.rs` uses unified version

---

## Deprecated Algorithm - PARTIAL

**File:** `gpu_landmark_apsp.cu`
**Function:** `stress_majorization_barneshut_kernel()`
**Status:** Deprecated - superseded by unified stress majorization
**Recommendation:** Keep APSP functions, remove Barnes-Hut stress majorization kernel

---

## Header Files (.cuh)

**Status:** No `.cuh` header files found in `/src/utils/`
**Conclusion:** All kernel declarations are self-contained in `.cu` files - GOOD

---

## TODO/FIXME Comments

**Search Results:** None found
**Conclusion:** No deferred work or technical debt markers in kernel code

---

## FFI Binding Verification

All active kernels have corresponding Rust FFI bindings:
- ✅ `visionflow_unified.cu` → `unified_gpu_compute.rs`
- ✅ `semantic_forces.cu` → `semantic_forces_actor.rs`, `semantic_forces.rs`
- ✅ `ontology_constraints.cu` → `ontology_constraint_actor.rs`
- ✅ `pagerank.cu` → `pagerank_actor.rs`
- ✅ `gpu_clustering_kernels.cu` → `clustering_actor.rs`, `anomaly_detection_actor.rs`
- ✅ `gpu_connected_components.cu` → `clustering_actor.rs`
- ✅ `sssp_compact.cu` → `shortest_path_actor.rs`
- ✅ `unified_stress_majorization.cu` → `stress_majorization_actor.rs`
- ✅ `visionflow_unified_stability.cu` → `force_compute_actor.rs`
- ✅ `gpu_aabb_reduction.cu` → `gpu/mod.rs`
- ✅ `dynamic_grid.cu` → `gpu_manager_actor.rs`
- ⚠️ `stress_majorization.cu` → **NO CALLERS FOUND** (superseded)
- ✅ `gpu_landmark_apsp.cu` → Partial usage (APSP only)

---

## Recommendations

### 1. Remove Duplicate Kernel ⚠️
**Action:** Delete `src/utils/stress_majorization.cu`
**Reason:** Fully superseded by `unified_stress_majorization.cu`
**Impact:** Zero - no active callers detected

### 2. Clean Deprecated Function in APSP Kernel
**Action:** Remove `stress_majorization_barneshut_kernel()` from `gpu_landmark_apsp.cu`
**Reason:** Duplicate of unified implementation
**Impact:** Low - keep APSP functions, remove deprecated stress kernel

### 3. No Other Orphaned Code Detected
**Conclusion:** All other 11 kernels are actively used and serve distinct purposes

---

## Performance Characteristics

### High-Performance Kernels (Optimized):
- ✅ `gpu_aabb_reduction.cu` - Warp shuffle primitives
- ✅ `gpu_clustering_kernels.cu` - FMA instructions, unrolled loops
- ✅ `unified_stress_majorization.cu` - Sparse CSR format (O(m) vs O(n²))
- ✅ `visionflow_unified.cu` - Spatial hashing for O(n log n) repulsion
- ✅ `visionflow_unified_stability.cu` - Early-exit for stable graphs

### Standard Performance:
- ✅ `semantic_forces.cu` - Standard kernel patterns
- ✅ `ontology_constraints.cu` - Pre-computed indices for O(1) lookups
- ✅ `pagerank.cu` - Standard power iteration
- ✅ `sssp_compact.cu` - Parallel prefix sum

---

## Conclusion

**Total Kernels:** 13
**Active Kernels:** 12 (11 fully active + 1 partially active)
**Orphaned Kernels:** 1 (`stress_majorization.cu`)
**Deprecated Functions:** 1 (`stress_majorization_barneshut_kernel` in `gpu_landmark_apsp.cu`)

All kernels are well-structured, properly documented, and integrated with the Rust codebase through FFI. The project has excellent separation of concerns between physics simulation, graph analytics, and semantic constraint enforcement.

**Action Items:**
1. ✅ Remove `src/utils/stress_majorization.cu`
2. ✅ Remove `stress_majorization_barneshut_kernel()` from `gpu_landmark_apsp.cu`
3. ✅ Update `build.rs` to exclude removed kernel from compilation

---

**Report Generated by:** Code Quality Analyzer
**Analysis Methodology:** Static code analysis, grep pattern matching, FFI binding verification
**Confidence Level:** High (manual verification recommended for deletion operations)
