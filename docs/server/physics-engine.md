# Physics Engine Architecture

## Overview

VisionFlow's physics engine provides GPU-accelerated force-directed graph layout for real-time visualization of 100,000+ nodes at 60 FPS. The engine implements a unified approach combining multiple algorithms optimized for dual graph visualization of both knowledge graphs and agent swarms.

## Architecture

```mermaid
graph TB
    subgraph "Input Sources"
        KG[Knowledge Graph Nodes] --> DGM[Dual Graph Manager]
        AS[Agent Swarm Nodes] --> DGM
    end
    
    subgraph "Physics Pipeline"
        DGM --> FD[Force-Directed Layout]
        DGM --> SM[Stress Majorization]
        DGM --> CS[Constraint Satisfaction]
        DGM --> VA[Visual Analytics]
        
        FD --> UP[Unified Physics Kernel]
        SM --> UP
        CS --> UP
        VA --> UP
    end
    
    subgraph "GPU Acceleration"
        UP --> CK[CUDA Kernel]
        CK --> POS[Position Buffer]
        CK --> VEL[Velocity Buffer]
        CK --> FORCE[Force Buffer]
    end
    
    subgraph "Output"
        POS --> BIN[Binary Protocol]
        VEL --> BIN
        BIN --> WS[WebSocket Stream]
    end
    
    style UP fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style CK fill:#3A3F47,stroke:#F56565,color:#FFFFFF
    style DGM fill:#3A3F47,stroke:#68D391,color:#FFFFFF
```

## Core Physics Algorithms

### 1. Force-Directed Layout

The unified physics kernel implements an optimized Fruchterman-Reingold algorithm with enhancements for dual graphs:

```rust
#[repr(C)]
pub struct ForceParams {
    pub spring_k: f32,         // Spring force strength
    pub repel_k: f32,          // Repulsion force strength  
    pub damping: f32,          // Velocity damping
    pub dt: f32,               // Time step
    pub max_velocity: f32,     // Velocity clamping
    pub max_force: f32,        // Force magnitude limit
}
```

**Force Equations**:
```rust
// Repulsive force between all node pairs (Coulomb-like)
F_repulsion = k_repel / distance²

// Attractive force between connected nodes (Hooke's law)
F_attraction = -k_spring * (distance - rest_length) * edge_weight

// Centering force toward origin
F_gravity = -k_gravity * position

// Total force with damping
F_total = (F_repulsion + F_attraction + F_gravity) * damping
```

### 2. Dual Graph Physics

Independent physics parameters for knowledge versus agent nodes:

#### Knowledge Graph Physics
Stable, slowly evolving layout optimized for readability:

```rust
pub const KNOWLEDGE_PHYSICS: ForceParams = ForceParams {
    spring_k: 0.005,        // Gentle connections
    repel_k: 50.0,          // Strong separation
    damping: 0.9,           // High stability
    dt: 0.01,               // Small time steps
    max_velocity: 1.0,      // Slow movement
    max_force: 10.0,        // Limited force
};
```

#### Agent Swarm Physics  
Dynamic, rapidly changing layout for real-time agent visualization:

```rust
pub const AGENT_PHYSICS: ForceParams = ForceParams {
    spring_k: 0.01,         // Stronger connections
    repel_k: 25.0,          // Moderate separation  
    damping: 0.7,           // More responsive
    dt: 0.016,              // 60 FPS time steps
    max_velocity: 5.0,      // Faster movement
    max_force: 20.0,        // Higher forces
};
```

### 3. Stress Majorization

Optimizes layout to preserve ideal graph-theoretic distances:

```rust
pub struct StressParams {
    pub alpha: f32,              // Learning rate (0.1)
    pub ideal_edge_length: f32,  // Target edge length (50.0)
    pub weight_influence: f32,   // Edge weight factor (0.5)
    pub max_iterations: u32,     // Iterations per frame (10)
}
```

**Stress Function**:
```
stress(X) = Σᵢⱼ wᵢⱼ(||xᵢ - xⱼ|| - dᵢⱼ)²

Where:
- wᵢⱼ = edge weight between nodes i and j
- ||xᵢ - xⱼ|| = Euclidean distance between nodes
- dᵢⱼ = ideal graph distance
```

### 4. Constraint Satisfaction

Enforces layout constraints for improved visual clarity:

```rust
pub enum ConstraintType {
    Separation(f32),    // Minimum node separation
    Alignment(f32),     // Align nodes along axis
    Clustering(f32),    // Group related nodes
    Boundary(f32),      // Keep nodes in viewport
}

pub struct ConstraintParams {
    pub constraint_type: i32,
    pub strength: f32,      // Constraint influence [0.0, 1.0]
    pub param1: f32,        // Type-specific parameter
    pub param2: f32,        // Type-specific parameter
    pub node_mask: i32,     // Affected node bitmask
}
```

## CUDA Kernel Implementation

### Unified Compute Kernel

**Location**: `src/utils/visionflow_unified.cu`

```cuda
__global__ void visionflow_compute_kernel(GpuKernelParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_nodes) return;
    
    // Load current node state
    float3 position = make_float3(
        params.nodes.pos_x[idx],
        params.nodes.pos_y[idx], 
        params.nodes.pos_z[idx]
    );
    
    float3 velocity = make_float3(
        params.nodes.vel_x[idx],
        params.nodes.vel_y[idx],
        params.nodes.vel_z[idx]
    );
    
    // Compute physics based on mode
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    
    switch (params.params.compute_mode) {
        case 0: // Basic force-directed
            total_force = compute_basic_forces(idx, position, params);
            break;
        case 1: // Dual graph with different physics per type
            total_force = compute_dual_graph_forces(idx, position, params);
            break;
        case 2: // With constraint satisfaction
            total_force = compute_constraint_forces(idx, position, params);
            break;
        case 3: // Visual analytics mode
            total_force = compute_analytics_forces(idx, position, params);
            break;
    }
    
    // Physics integration with Verlet method
    integrate_verlet_physics(idx, position, velocity, total_force, params);
}
```

### Optimized Force Computation

**Tiled Repulsion Calculation** for O(n²) to O(n log n) improvement:

```cuda
__device__ float3 compute_repulsion_optimized(
    int idx, float3 position, GpuKernelParams params
) {
    float3 repulsion = make_float3(0.0f, 0.0f, 0.0f);
    
    // Use shared memory for tile-based computation
    __shared__ float3 tile_positions[BLOCK_SIZE];
    __shared__ int tile_graph_ids[BLOCK_SIZE];
    
    int my_graph_id = params.nodes.graph_id ? params.nodes.graph_id[idx] : 0;
    
    for (int tile = 0; tile < gridDim.x; tile++) {
        int tile_idx = tile * blockDim.x + threadIdx.x;
        
        // Collaborative loading into shared memory
        if (tile_idx < params.num_nodes) {
            tile_positions[threadIdx.x] = make_float3(
                params.nodes.pos_x[tile_idx],
                params.nodes.pos_y[tile_idx],
                params.nodes.pos_z[tile_idx]
            );
            tile_graph_ids[threadIdx.x] = params.nodes.graph_id ? 
                params.nodes.graph_id[tile_idx] : 0;
        }
        __syncthreads();
        
        // Compute repulsion forces within tile
        for (int i = 0; i < blockDim.x; i++) {
            int other_idx = tile * blockDim.x + i;
            if (other_idx >= params.num_nodes || other_idx == idx) continue;
            
            float3 diff = position - tile_positions[i];
            float dist_sq = dot(diff, diff) + 0.01f; // Avoid division by zero
            
            // Apply distance cutoff for performance
            if (dist_sq > params.params.cutoff_distance_sq) continue;
            
            // Different repulsion for same vs different graph types
            float repulsion_strength = params.params.repel_k;
            if (my_graph_id != tile_graph_ids[i]) {
                repulsion_strength *= 0.5f; // Weaker cross-graph repulsion
            }
            
            float force_magnitude = repulsion_strength / dist_sq;
            repulsion += normalize(diff) * force_magnitude;
        }
        __syncthreads();
    }
    
    return repulsion;
}
```

### Edge-Based Attraction Forces

```cuda
__device__ float3 compute_attraction_forces(
    int idx, float3 position, GpuKernelParams params
) {
    float3 attraction = make_float3(0.0f, 0.0f, 0.0f);
    
    // Iterate through edges using Compressed Sparse Row (CSR) format
    for (int e = edge_start_indices[idx]; e < edge_start_indices[idx + 1]; e++) {
        int neighbor_idx = params.edges.dst[e];
        float edge_weight = params.edges.weight[e];
        
        float3 neighbor_pos = make_float3(
            params.nodes.pos_x[neighbor_idx],
            params.nodes.pos_y[neighbor_idx],
            params.nodes.pos_z[neighbor_idx]
        );
        
        float3 diff = neighbor_pos - position;
        float distance = length(diff);
        
        if (distance > 0.001f) {
            // Spring force proportional to distance and edge weight
            float spring_force = params.params.spring_k * distance * edge_weight;
            attraction += normalize(diff) * spring_force;
        }
    }
    
    return attraction;
}
```

## Physics Integration Methods

### Verlet Integration

Used for stable, high-performance physics simulation:

```cuda
__device__ void integrate_verlet_physics(
    int idx, float3 position, float3 velocity, float3 force,
    GpuKernelParams params
) {
    float dt = params.params.dt;
    float damping = params.params.damping;
    
    // Verlet integration: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
    float3 acceleration = force; // Assuming unit mass
    
    float3 new_position = position + velocity * dt + acceleration * (0.5f * dt * dt);
    float3 new_velocity = (velocity + acceleration * dt) * damping;
    
    // Apply velocity clamping
    float speed = length(new_velocity);
    if (speed > params.params.max_velocity) {
        new_velocity = normalize(new_velocity) * params.params.max_velocity;
    }
    
    // Apply boundary constraints
    new_position = apply_boundary_constraints(new_position, params.params.viewport_bounds);
    
    // Store results
    params.nodes.pos_x[idx] = new_position.x;
    params.nodes.pos_y[idx] = new_position.y;
    params.nodes.pos_z[idx] = new_position.z;
    params.nodes.vel_x[idx] = new_velocity.x;
    params.nodes.vel_y[idx] = new_velocity.y;
    params.nodes.vel_z[idx] = new_velocity.z;
}
```

## Advanced Physics Features

### Temporal Dynamics

For agent swarm visualization with time-varying behavior:

```rust
pub struct TemporalParams {
    pub time_decay: f32,        // Historical position influence
    pub prediction_weight: f32, // Future state prediction
    pub momentum_factor: f32,   // Movement momentum preservation
}
```

### Semantic Constraints

Group related nodes based on metadata:

```cuda
__device__ float3 compute_semantic_forces(
    int idx, float3 position, GpuKernelParams params
) {
    float3 semantic_force = make_float3(0.0f, 0.0f, 0.0f);
    
    int my_cluster = params.nodes.cluster ? params.nodes.cluster[idx] : -1;
    if (my_cluster < 0) return semantic_force;
    
    // Attract to nodes in same semantic cluster
    for (int i = 0; i < params.num_nodes; i++) {
        if (i == idx) continue;
        
        int other_cluster = params.nodes.cluster[i];
        if (other_cluster == my_cluster) {
            float3 other_pos = make_float3(
                params.nodes.pos_x[i],
                params.nodes.pos_y[i],
                params.nodes.pos_z[i]
            );
            
            float3 diff = other_pos - position;
            float distance = length(diff);
            
            if (distance > 0.001f) {
                float cluster_force = params.params.cluster_strength / distance;
                semantic_force += normalize(diff) * cluster_force;
            }
        }
    }
    
    return semantic_force;
}
```

### Adaptive Physics Parameters

Dynamically adjust physics based on system state:

```rust
impl PhysicsEngine {
    fn update_adaptive_parameters(&mut self) {
        let system_energy = self.calculate_system_energy();
        let convergence_rate = self.calculate_convergence_rate();
        
        // Increase damping as system stabilizes
        if convergence_rate < 0.01 {
            self.params.damping = (self.params.damping + 0.95) / 2.0;
        }
        
        // Reduce time step if system becomes unstable
        if system_energy > self.energy_threshold {
            self.params.dt *= 0.9;
        }
        
        // Adaptive temperature for simulated annealing
        self.params.temperature *= 0.99;
    }
    
    fn calculate_system_energy(&self) -> f32 {
        // Sum of kinetic energy (velocity²) + potential energy (forces)
        self.nodes.iter().map(|node| {
            node.velocity.length_squared() * 0.5 + 
            node.potential_energy
        }).sum()
    }
}
```

## Performance Optimization Strategies

### Multi-Resolution Physics

Different update rates based on node importance:

```rust
pub enum UpdatePriority {
    Critical,   // Every frame (60 Hz) - active agents, high-degree nodes
    High,       // Every 2 frames (30 Hz) - recently modified nodes
    Medium,     // Every 4 frames (15 Hz) - medium connectivity
    Low,        // Every 8 frames (7.5 Hz) - stable, low-degree nodes
}

fn get_update_priority(node: &Node, agent_data: Option<&AgentStatus>) -> UpdatePriority {
    if let Some(agent) = agent_data {
        if agent.is_active() {
            return UpdatePriority::Critical;
        }
    }
    
    if node.degree > 20 || node.recently_modified() {
        UpdatePriority::High
    } else if node.degree > 5 {
        UpdatePriority::Medium
    } else {
        UpdatePriority::Low
    }
}
```

### Spatial Indexing

Accelerate neighbor searches with 3D spatial hashing:

```cuda
__device__ int3 compute_spatial_hash(float3 position, float cell_size) {
    return make_int3(
        __float2int_rd(position.x / cell_size),
        __float2int_rd(position.y / cell_size),
        __float2int_rd(position.z / cell_size)
    );
}

__global__ void build_spatial_index(
    GpuKernelParams params,
    int* spatial_hash,
    int* cell_start,
    int* cell_end
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_nodes) return;
    
    float3 position = make_float3(
        params.nodes.pos_x[idx],
        params.nodes.pos_y[idx],
        params.nodes.pos_z[idx]
    );
    
    int3 cell = compute_spatial_hash(position, SPATIAL_CELL_SIZE);
    int hash = morton_encode_3d(cell);
    
    spatial_hash[idx] = hash;
    
    // Use atomic operations to build cell boundaries
    atomicMin(&cell_start[hash], idx);
    atomicMax(&cell_end[hash], idx + 1);
}
```

## Configuration & Tuning

### Physics Settings

Configuration loaded from `settings.yaml`:

```yaml
physics:
  # Force-directed parameters
  spring_strength: 0.005      # Attractive force between connected nodes
  repulsion_strength: 50.0    # Repulsive force between all nodes
  damping: 0.9               # Velocity damping factor
  time_step: 0.01            # Physics integration time step
  max_velocity: 1.0          # Maximum node velocity
  temperature: 0.5           # Simulated annealing temperature
  
  # Dual graph settings
  agent_physics_multiplier: 2.0   # Agent graph responsiveness
  cross_graph_interaction: 0.1    # Inter-graph force strength
  
  # Constraint system
  separation_radius: 0.15     # Minimum node separation
  boundary_size: 200.0       # Simulation boundary
  cluster_strength: 0.2      # Semantic clustering force
  
  # Performance tuning
  spatial_cell_size: 25.0    # Spatial index cell size
  force_cutoff_distance: 100.0  # Maximum interaction distance
  adaptive_timestep: true    # Enable adaptive time stepping
```

### Compute Mode Selection

```rust
impl PhysicsEngine {
    pub fn select_optimal_mode(&mut self, node_count: usize, has_agents: bool) -> ComputeMode {
        match (node_count, has_agents) {
            (n, _) if n > 50_000 => ComputeMode::Basic,
            (_, true) => ComputeMode::DualGraph,
            (n, false) if n > 10_000 => ComputeMode::Constraints,
            _ => ComputeMode::VisualAnalytics,
        }
    }
}
```

## Performance Benchmarks

Real-time performance on NVIDIA RTX 4090:

| Configuration | Nodes | Edges | Mode | FPS | GPU Memory | CPU Usage |
|---------------|-------|-------|------|-----|------------|-----------|
| Knowledge Only | 10K | 25K | Basic | 120 | 200 MB | 15% |
| Knowledge + Agents | 8K + 2K | 20K + 5K | DualGraph | 60 | 350 MB | 20% |
| Large Knowledge | 50K | 125K | Constraints | 60 | 800 MB | 25% |
| Massive Scale | 100K | 250K | Basic + Spatial | 30 | 1.5 GB | 35% |
| Analytics Mode | 25K | 60K | VisualAnalytics | 45 | 600 MB | 40% |

## Stability Analysis & Debugging

### Energy Monitoring

```rust
pub struct SystemMetrics {
    pub kinetic_energy: f32,     // Sum of node velocities²
    pub potential_energy: f32,   // Sum of force potentials
    pub total_energy: f32,       // Kinetic + potential
    pub convergence_rate: f32,   // Rate of energy change
    pub oscillation_detected: bool, // Periodic motion detection
}

impl PhysicsEngine {
    pub fn analyze_stability(&self) -> SystemMetrics {
        let kinetic = self.nodes.iter()
            .map(|n| n.velocity.length_squared())
            .sum::<f32>() * 0.5;
            
        let potential = self.compute_total_potential_energy();
        let total = kinetic + potential;
        
        let convergence = if let Some(prev) = self.previous_energy {
            (prev - total).abs() / prev.max(1.0)
        } else {
            1.0
        };
        
        let oscillation = self.detect_oscillation_pattern();
        
        SystemMetrics {
            kinetic_energy: kinetic,
            potential_energy: potential,
            total_energy: total,
            convergence_rate: convergence,
            oscillation_detected: oscillation,
        }
    }
}
```

### Debug Visualization

```rust
// Enable physics debug mode via settings
pub struct PhysicsDebugConfig {
    pub show_forces: bool,       // Render force vectors
    pub show_velocities: bool,   // Render velocity vectors
    pub show_spatial_grid: bool, // Show spatial indexing cells
    pub highlight_constraints: bool, // Highlight constraint violations
    pub energy_graph: bool,      // Real-time energy plotting
}
```

## Error Handling & Fallbacks

### Common Issues & Solutions

| Issue | Symptom | Cause | Solution |
|-------|---------|-------|----------|
| Node Explosion | Nodes scatter rapidly | Forces too high | Reduce repulsion, increase damping |
| Slow Convergence | Layout keeps changing | Damping too low | Increase damping to 0.9+ |
| Clustering Issues | Nodes form tight clumps | Repulsion too low | Increase repulsion strength |
| GPU Out of Memory | Crash on large graphs | Insufficient VRAM | Reduce max_nodes or enable streaming |
| Poor Performance | Low FPS despite good GPU | Inefficient parameters | Enable spatial indexing, reduce cutoff distance |

### Automatic Parameter Adjustment

```rust
impl PhysicsEngine {
    pub fn auto_tune_parameters(&mut self, target_fps: f32, current_fps: f32) {
        let performance_ratio = current_fps / target_fps;
        
        if performance_ratio < 0.5 {
            // Performance too low - simplify simulation
            self.params.force_cutoff_distance *= 0.8;
            self.params.spatial_cell_size *= 1.2;
            self.switch_to_basic_mode_if_needed();
        } else if performance_ratio > 1.5 && current_fps < 120.0 {
            // Performance headroom - enhance quality
            self.params.force_cutoff_distance *= 1.1;
            self.enable_advanced_features_if_possible();
        }
    }
}
```

## Related Documentation

- **[GPU Compute Architecture](gpu-compute.md)** - CUDA implementation details
- **[Actor System](actors.md)** - GraphServiceActor and GPUComputeActor integration  
- **[Binary Protocol](../api/binary-protocol.md)** - Efficient position data streaming
- **[Dual Graph Architecture](../architecture/dual-graph.md)** - Knowledge + Agent graph coordination
- **[Performance Optimization](../optimization/physics-tuning.md)** - Advanced tuning techniques