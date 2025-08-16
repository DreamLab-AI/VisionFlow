# GPU Features to Expose - Implementation Plan

## Overview
Based on the GPU CUDA kernel audit, we're currently using less than 30% of the available functionality. This document outlines the features we should expose through the interface.

## Priority 1: Critical Missing Features

### 1. Compute Modes (Currently hardcoded to 0)
The GPU supports 4 compute modes but we only use basic:
```cuda
// Line 410-447 in visionflow_unified.cu
switch (p.params.compute_mode) {
    case 0: // Basic force-directed (current default)
    case 1: // Dual graph mode (UNEXPOSED)
    case 2: // With constraints (UNEXPOSED)
    case 3: // Visual analytics (UNEXPOSED)
}
```

**Required Changes:**
- Add compute_mode to settings UI (dropdown with 4 options)
- Add node properties for modes 1 and 3 (graph_id, importance, temporal, cluster)
- Add constraint system for mode 2

### 2. Hardcoded Constants That Should Be Parameters
```cuda
// Critical constants affecting physics behavior
const float MIN_DISTANCE = 0.15f;          // Line 103, 179, 310 - Should match separation_radius
const float MAX_REPULSION_DIST = 50.0f;    // Line 104 - Should use repulsion_distance parameter
float boundary_margin = 0.85f;             // Line 490 - Percentage of viewport_bounds
float boundary_force_strength = 2.0f;      // Line 491 - Strength of boundary repulsion
int warmup_iterations = 200;               // Line 462 - Warmup period duration
int zero_velocity_iterations = 5;          // Line 482 - Initial freeze period
float cooling_rate = 0.0001f;              // Line 472 - Temperature decay rate
```

**Required Changes:**
- Add these as configurable parameters in SimParams
- Expose through settings interface
- Add to validation logic

### 3. Stress Majorization (Lines 564-613)
Complete stress optimisation system for better layouts:
```cuda
__global__ void stress_majorization_kernel(
    float* ideal_distances,  // N x N matrix
    float* weight_matrix,     // N x N matrix
    SimParams params
)
```

**Required Features:**
- Enable/disable stress optimisation
- Set stress_weight and stress_alpha (already in SimParams)
- Provide ideal distance matrix interface
- Weight matrix for importance-based layout

### 4. Constraint System (Lines 232-294)
Four constraint types available but completely unexposed:
```cuda
struct ConstraintData {
    int type;         // 0=none, 1=separation, 2=boundary, 3=alignment, 4=cluster
    float strength;
    float param1;
    float param2;
    int node_mask;    // Bit mask for selective application
}
```

**Required Features:**
- Constraint definition interface
- Node selection for constraint application
- Constraint strength controls
- Visual feedback for active constraints

## Priority 2: Advanced Analytics Features

### 1. Clustering Algorithms (Lines 618-829)
Three clustering methods implemented but unexposed:
```cuda
extern "C" {
    void run_kmeans_clustering(...);
    void run_spectral_clustering(...);
    void run_louvain_clustering(...);
}
```

**Required Features:**
- Algorithm selection dropdown
- Cluster count parameter
- Cluster visualization options
- Community detection resolution parameter
- Export cluster assignments

### 2. Visual Analytics Mode (Lines 297-366)
Node importance and temporal coherence:
```cuda
float* node_importance;   // Importance weights
float* node_temporal;     // Temporal coherence
int* node_cluster;        // Cluster assignments
```

**Required Features:**
- Import importance values
- Temporal weight controls
- Cluster-aware force scaling
- Importance visualization

### 3. Dual Graph Support (Lines 167-227)
Knowledge vs Agent graph differentiation:
```cuda
int* node_graph_id;  // 0=knowledge, 1=agent
int* edge_graph_id;  // Edge graph membership
```

**Required Features:**
- Graph type assignment
- Inter/intra-graph force scaling
- Graph-specific physics parameters
- Visual differentiation

## Priority 3: Performance & Optimization

### 1. Progressive Warmup System
```cuda
if (p.params.iteration < 200) {
    float warmup = p.params.iteration / 200.0f;
    force = vec3_scale(force, warmup * warmup); // Quadratic warmup
}
```

**Required Features:**
- Warmup duration control
- Warmup curve selection (linear, quadratic, cubic)
- Extra damping during warmup
- Visual warmup indicator

### 2. Boundary System Enhancement
```cuda
// Progressive boundary damping (Lines 489-551)
float distance_ratio = (fabsf(position.x) - boundary_margin) / (viewport_bounds - boundary_margin);
float boundary_force = -distance_ratio * distance_ratio * boundary_force_strength;
```

**Required Features:**
- Boundary margin percentage
- Boundary force strength
- Progressive damping controls
- Soft vs hard boundary toggle

### 3. Natural Length Calculation
```cuda
// Adaptive natural length (Lines 154, 215, 351)
float natural_length = fminf(params.separation_radius * 5.0f, 10.0f);
```

**Required Features:**
- Natural length multiplier
- Max natural length
- Edge-specific natural lengths
- Dynamic natural length based on node density

## Implementation Approach

### Phase 1: Core Parameters (Week 1)
1. Add missing parameters to SimParams
2. Update validation logic
3. Expose through REST API
4. Add to settings UI

### Phase 2: Compute Modes (Week 2)
1. Implement mode switching
2. Add required node/edge properties
3. Create mode-specific UI panels
4. Test each mode thoroughly

### Phase 3: Advanced Features (Week 3-4)
1. Implement constraint system
2. Add clustering interface
3. Enable stress optimisation
4. Implement dual graph support

### Phase 4: Performance Tuning (Week 5)
1. Add warmup controls
2. Enhance boundary system
3. Implement adaptive parameters
4. Performance benchmarking

## Testing Strategy

### Unit Tests:
- Parameter validation ranges
- Conversion functions
- Mode switching logic

### Integration Tests:
- Physics parameter propagation
- GPU kernel communication
- WebSocket updates

### Visual Tests:
- Each compute mode behavior
- Constraint application
- Clustering visualization
- Boundary behavior

## UI/UX Considerations

### Developer Panel Organization:
```
GPU Physics Settings
├── Core Parameters
│   ├── Basic Forces
│   ├── Boundaries
│   └── Stability
├── Compute Mode
│   ├── Mode Selection
│   ├── Mode-Specific Options
│   └── Performance
├── Advanced Features
│   ├── Constraints
│   ├── Clustering
│   └── Stress Optimization
└── Debug & Monitoring
    ├── Force Visualization
    ├── Performance Metrics
    └── Convergence Tracking
```

### Progressive Disclosure:
- Basic users: Simple presets (Stable, Dynamic, Experimental)
- Advanced users: Full parameter control
- Developer mode: All GPU features exposed

## Benefits of Full GPU Exposure

1. **Performance**: 2-4x faster convergence with proper warmup
2. **Quality**: Better layouts with stress optimisation
3. **Flexibility**: Support for multiple visualization paradigms
4. **Analytics**: Built-in clustering and community detection
5. **Stability**: Fine-grained control over physics behavior

## Next Steps

1. Review and prioritize features with team
2. Create detailed technical specifications
3. Update client-side TypeScript interfaces
4. Implement phase 1 core parameters
5. Create migration guide for existing deployments