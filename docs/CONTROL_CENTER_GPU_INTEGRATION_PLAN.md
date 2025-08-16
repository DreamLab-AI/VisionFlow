# Control Center GPU Integration Plan

## Current 9 Sections Audit

### 1. **Dashboard** - System overview and status
Current: Basic system info
GPU Features to Add:
- Compute mode selector (0: Basic, 1: Dual Graph, 2: Constraints, 3: Visual Analytics)
- Iteration counter display
- Convergence indicator
- Active constraints count
- Clustering status

### 2. **Visualization** - Rendering and display settings
Current: Node/edge appearance, quality settings
GPU Features to Add:
- Node importance controls (for Visual Analytics mode)
- Temporal coherence weight
- Graph differentiation (Knowledge vs Agent)
- Cluster visualisation options
- Stress optimisation toggle

### 3. **Physics** - Force and movement settings
Current: Basic physics parameters (OLD NAMES - need updating!)
GPU Features to Add:
- Replace old names with GPU-aligned (spring_k, repel_k, dt, etc.)
- Add stress_weight, stress_alpha controls
- Add warmup controls (iterations, curve type)
- Add natural length multiplier
- Add cooling rate control
- Add MIN_DISTANCE, MAX_REPULSION_DIST controls

### 4. **Analytics** - Graph analysis and metrics
Current: Basic metrics toggles
GPU Features to Add:
- Clustering algorithm selector (K-means, Spectral, Louvain)
- Cluster count control
- Community detection resolution
- Modularity optimisation settings
- Export cluster assignments
- Ideal distance matrix import

### 5. **Performance** - Optimization and quality
Current: FPS, GPU memory, quality presets
GPU Features to Add:
- Warmup duration control
- Convergence threshold
- Adaptive cooling toggle
- GPU block size optimisation
- Memory coalescing options
- Iteration limit controls

### 6. **Integrations** (Currently Visual Effects)
Current: Bloom, hologram, flow effects
Keep as is - this section is well-organised for visual effects

### 7. **Developer** - Debug and development tools
Current: Basic logging and debug toggles
GPU Features to Add:
- Move ALL debug settings from XR/AR section here
- Add force visualization toggles
- Add constraint debug visualization
- Add convergence metrics display
- Add GPU kernel timing stats
- Add boundary force visualization

### 8. **Authentication** - Security and access
Current: Nostr login, auth settings
Keep as is - not related to GPU features

### 9. **XR/AR** - Extended reality settings
Current: Mixed XR settings and DEBUG settings (needs cleanup!)
Changes:
- REMOVE all debug settings (move to Developer)
- Keep only XR-specific settings
- Add XR-optimized compute mode toggle
- Add XR performance presets

## Detailed Integration Plan

### Phase 1: Update Physics Section (CRITICAL)

```typescript
// Replace old parameter names with GPU-aligned
physics: {
  // Core Forces
  spring_k: { min: 0.001, max: 2.0, default: 0.005 },
  repel_k: { min: 1, max: 100, default: 50 },
  attraction_k: { min: 0, max: 1, default: 0.001 },
  
  // Dynamics
  dt: { min: 0.001, max: 0.1, default: 0.016 },
  damping: { min: 0, max: 1, default: 0.95 },
  
  // Boundaries
  separation_radius: { min: 0.1, max: 10, default: 0.15 },
  boundary_limit: { min: 0.8, max: 1.0, default: 0.98 }, // % of viewport
  boundary_margin: { min: 0.7, max: 0.95, default: 0.85 },
  boundary_force_strength: { min: 0.5, max: 5, default: 2.0 },
  
  // Optimization
  stress_weight: { min: 0, max: 1, default: 0.1 },
  stress_alpha: { min: 0, max: 1, default: 0.1 },
  
  // Constants (previously hardcoded)
  min_distance: { min: 0.05, max: 1, default: 0.15 },
  max_repulsion_dist: { min: 10, max: 200, default: 50 },
  
  // Warmup
  warmup_iterations: { min: 0, max: 500, default: 200 },
  warmup_curve: ['linear', 'quadratic', 'cubic'],
  zero_velocity_iterations: { min: 0, max: 20, default: 5 },
  
  // Cooling
  temperature: { min: 0, max: 2, default: 0.5 },
  cooling_rate: { min: 0.00001, max: 0.01, default: 0.0001 }
}
```

### Phase 2: Add Compute Mode to Dashboard

```typescript
dashboard: {
  compute_mode: {
    type: 'select',
    options: [
      { value: 0, label: 'Basic Force-Directed' },
      { value: 1, label: 'Dual Graph (Knowledge + Agent)' },
      { value: 2, label: 'Constraint-Enhanced' },
      { value: 3, label: 'Visual Analytics' }
    ]
  },
  active_constraints: { type: 'display' },
  iteration_count: { type: 'display' },
  convergence_status: { type: 'indicator' },
  clustering_active: { type: 'toggle' }
}
```

### Phase 3: Add Clustering to Analytics

```typescript
analytics: {
  clustering_algorithm: {
    type: 'select',
    options: ['none', 'kmeans', 'spectral', 'louvain']
  },
  cluster_count: { min: 2, max: 20, default: 5 },
  resolution: { min: 0.1, max: 2, default: 1.0 },
  iterations: { min: 10, max: 100, default: 30 },
  export_clusters: { type: 'button' },
  import_distances: { type: 'button' }
}
```

### Phase 4: Add Constraints System (New Subsection)

```typescript
constraints: {
  separation: {
    enabled: { type: 'toggle' },
    min_distance: { min: 0.1, max: 10 },
    strength: { min: 0, max: 10 }
  },
  boundary: {
    enabled: { type: 'toggle' },
    x_limit: { min: 10, max: 1000 },
    y_limit: { min: 10, max: 1000 },
    z_limit: { min: 10, max: 1000 },
    strength: { min: 0, max: 10 }
  },
  alignment: {
    enabled: { type: 'toggle' },
    axis: ['horizontal', 'vertical', 'both'],
    strength: { min: 0, max: 10 }
  },
  cluster: {
    enabled: { type: 'toggle' },
    center_x: { min: -100, max: 100 },
    center_y: { min: -100, max: 100 },
    strength: { min: 0, max: 10 }
  }
}
```

### Phase 5: Move Debug Settings

Move FROM XR/AR TO Developer:
- enableDebug
- showFPS
- showMemory
- perfDebug
- logLevel
- telemetry
- dataDebug
- wsDebug
- physicsDebug
- nodeDebug
- shaderDebug
- matrixDebug

Add NEW to Developer:
- force_vectors: { type: 'toggle' }
- constraint_visualization: { type: 'toggle' }
- boundary_force_display: { type: 'toggle' }
- convergence_graph: { type: 'toggle' }
- gpu_timing_stats: { type: 'toggle' }

## REST API Updates Required

### New Endpoints:
```
POST /api/physics/compute-mode
POST /api/clustering/algorithm
POST /api/constraints/update
GET /api/analytics/clusters
POST /api/stress/optimization
```

### Updated Validation:
- Accept new parameter names
- Validate compute_mode (0-3)
- Validate clustering parameters
- Validate constraint definitions

## Settings Store Updates

```typescript
interface GPUPhysicsSettings {
  // Core parameters (GPU-aligned)
  spring_k: number;
  repel_k: number;
  dt: number;
  damping: number;
  
  // Advanced GPU features
  compute_mode: number;
  stress_weight: number;
  stress_alpha: number;
  
  // Constraints
  constraints: ConstraintData[];
  
  // Clustering
  clustering: {
    algorithm: string;
    params: ClusteringParams;
  };
  
  // Warmup
  warmup: {
    iterations: number;
    curve: string;
    zero_velocity_iterations: number;
  };
}
```

## UI/UX Improvements

### Progressive Disclosure:
1. **Basic Mode**: Show only essential physics controls
2. **Advanced Mode**: Show all GPU parameters
3. **Expert Mode**: Show constraint editor, clustering, stress optimization

### Visual Feedback:
- Color-code sections by GPU compute mode
- Show real-time convergence graph
- Display active constraints visually
- Highlight parameters that differ from defaults

### Presets:
```typescript
presets: {
  'stable': { damping: 0.95, repel_k: 50, spring_k: 0.005 },
  'dynamic': { damping: 0.85, repel_k: 100, spring_k: 0.01 },
  'experimental': { damping: 0.7, repel_k: 200, spring_k: 0.02 },
  'clustering': { compute_mode: 3, clustering_algorithm: 'louvain' }
}
```

## Implementation Order

1. **Update Physics section with new parameter names** (CRITICAL)
2. **Move debug settings from XR to Developer**
3. **Add compute_mode selector to Dashboard**
4. **Update REST API validation for new parameters**
5. **Add clustering controls to Analytics**
6. **Implement constraint system UI**
7. **Add warmup and cooling controls**
8. **Create preset system**
9. **Add visual feedback indicators**
10. **Test all integrations**

## Success Metrics

- All GPU features accessible through UI
- No old parameter names remain
- Debug settings properly organized
- Compute modes fully functional
- Clustering algorithms working
- Constraints can be defined and applied
- Performance improved with warmup system
- User can access 100% of GPU capabilities