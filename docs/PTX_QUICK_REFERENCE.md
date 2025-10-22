# PTX Compilation Quick Reference

## Status: ✅ ALL 8 KERNELS COMPILED

### Kernel Overview

| # | Kernel | Lines | PTX Size | Status |
|---|--------|-------|----------|--------|
| 1 | visionflow_unified | 1,886 | 1.5 MB | ✅ Compiled |
| 2 | gpu_clustering_kernels | 687 | 1.1 MB | ✅ Compiled |
| 3 | ontology_constraints | 487 | 33 KB | ✅ Compiled (NEW) |
| 4 | visionflow_unified_stability | 330 | 19 KB | ✅ Compiled |
| 5 | gpu_landmark_apsp | 151 | 17 KB | ✅ Compiled |
| 6 | gpu_aabb_reduction | 107 | 13 KB | ✅ Compiled |
| 7 | dynamic_grid | 322 | 5.1 KB | ✅ Compiled |
| 8 | sssp_compact | 105 | 4.8 KB | ✅ Compiled |
| **TOTAL** | **8 kernels** | **4,075** | **2.71 MB** | **100% Complete** |

---

## Quick Commands

### Compile Single Kernel
```bash
nvcc --ptx -arch sm_75 -o output.ptx source.cu --use_fast_math -O3
```

### Compile All Kernels
```bash
cd /home/devuser/workspace/project
cargo build --features gpu
```

### Verify PTX Files
```bash
ls -lh src/utils/ptx/*.ptx
```

### Check CUDA Version
```bash
nvcc --version
```

---

## PTX Locations

### Source Directory
```
src/utils/ptx/
├── dynamic_grid.ptx
├── gpu_aabb_reduction.ptx
├── gpu_clustering_kernels.ptx
├── gpu_landmark_apsp.ptx
├── ontology_constraints.ptx ⭐
├── sssp_compact.ptx
├── visionflow_unified.ptx
└── visionflow_unified_stability.ptx
```

### Build Output
```
target/{debug,release}/build/webxr-*/out/*.ptx
```

---

## Key Kernels by Function

### Physics & Simulation
- `visionflow_unified.cu` - Core physics engine
- `visionflow_unified_stability.cu` - Stability gates
- `dynamic_grid.cu` - Spatial partitioning

### Graph Algorithms
- `sssp_compact.cu` - Shortest path frontier
- `gpu_landmark_apsp.cu` - All-pairs shortest path

### Machine Learning
- `gpu_clustering_kernels.cu` - K-means, DBSCAN, LOF, Louvain

### Semantic Reasoning
- `ontology_constraints.cu` - OWL constraint enforcement ⭐

### Utilities
- `gpu_aabb_reduction.cu` - Bounding box computation

---

## Recent Changes

### 2025-10-22
- ✅ Fixed `ontology_constraints.cu` compilation (added `#include <cstdint>`)
- ✅ Compiled `ontology_constraints.cu` to PTX (33 KB)
- ✅ Added `ontology_constraints.cu` to build.rs
- ✅ Created comprehensive PTX compilation report

---

## CUDA Environment

- **Toolkit**: 13.0.88
- **Path**: `/opt/cuda/bin/nvcc`
- **Architecture**: sm_75 (Turing)
- **Flags**: `--use_fast_math -O3 -arch sm_75`

---

## Kernel Function Catalog

### visionflow_unified.cu
- `build_grid_kernel`
- `force_pass_kernel`
- `integrate_pass_kernel`
- `compute_cell_bounds_kernel`
- `relaxation_step_kernel`

### gpu_clustering_kernels.cu
- `init_centroids_kernel`
- `assign_clusters_kernel`
- `update_centroids_kernel`
- `compute_lof_kernel`
- `compute_zscore_kernel`
- `louvain_local_pass_kernel`
- `stress_majorization_step_kernel`

### ontology_constraints.cu ⭐
- `apply_disjoint_classes_kernel`
- `apply_subclass_hierarchy_kernel`
- `apply_sameas_colocate_kernel`
- `apply_inverse_symmetry_kernel`
- `apply_functional_cardinality_kernel`

### Other Kernels
- `approximate_apsp_kernel` (gpu_landmark_apsp)
- `compute_aabb_reduction_kernel` (gpu_aabb_reduction)
- `compact_frontier_kernel` (sssp_compact)
- `compute_feature_stats_kernel` (visionflow_unified_stability)

---

## Performance Targets

- **visionflow_unified**: ~2ms per frame (10K nodes)
- **ontology_constraints**: ~2ms per frame (10K nodes)
- **gpu_clustering_kernels**: Variable (algorithm-dependent)
- **Block Size**: 256 threads (typical)

---

## Next Steps

- [ ] Run performance benchmarks
- [ ] Verify runtime PTX loading
- [ ] Document GPU memory requirements
- [ ] Optimize kernel launch parameters

---

**Last Updated**: 2025-10-22
**Full Report**: See `PTX_COMPILATION_REPORT.md`
