# GpuPhysicsAdapter Port

## Purpose

The **GpuPhysicsAdapter** port provides GPU-accelerated physics simulation for knowledge graph layout using force-directed algorithms. It abstracts CUDA implementations for physics computations.

## Location

- **Trait Definition**: `src/ports/gpu_physics_adapter.rs`
- **Adapter Implementation**: `src/adapters/cuda_physics_adapter.rs`

## Interface

```rust
#[async_trait]
pub trait GpuPhysicsAdapter: Send + Sync {
    // Initialization
    async fn initialize(&mut self, graph: Arc<GraphData>, params: PhysicsParameters) -> Result<()>;
    async fn cleanup(&mut self) -> Result<()>;

    // Force computation
    async fn compute_forces(&mut self) -> Result<Vec<NodeForce>>;
    async fn update_positions(&mut self, forces: &[NodeForce]) -> Result<Vec<(u32, f32, f32, f32)>>;
    async fn step(&mut self) -> Result<PhysicsStepResult>;

    // Simulation control
    async fn simulate_until_convergence(&mut self) -> Result<PhysicsStepResult>;
    async fn reset(&mut self) -> Result<()>;

    // External forces and constraints
    async fn apply_external_forces(&mut self, forces: Vec<(u32, f32, f32, f32)>) -> Result<()>;
    async fn pin_nodes(&mut self, nodes: Vec<(u32, f32, f32, f32)>) -> Result<()>;
    async fn unpin_nodes(&mut self, node_ids: Vec<u32>) -> Result<()>;

    // Configuration
    async fn update_parameters(&mut self, params: PhysicsParameters) -> Result<()>;
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<()>;

    // Status and statistics
    async fn get_gpu_status(&self) -> Result<GpuDeviceInfo>;
    async fn get_statistics(&self) -> Result<PhysicsStatistics>;
}
```

## Types

### PhysicsParameters

Physics simulation configuration:

```rust
pub struct PhysicsParameters {
    pub time_step: f32,
    pub damping: f32,
    pub spring_constant: f32,
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub max_velocity: f32,
    pub convergence_threshold: f32,
    pub max_iterations: u32,
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            time_step: 0.016,          // 60 FPS
            damping: 0.8,
            spring_constant: 0.01,
            repulsion_strength: 100.0,
            attraction_strength: 0.1,
            max_velocity: 10.0,
            convergence_threshold: 0.01,
            max_iterations: 1000,
        }
    }
}
```

### NodeForce

Force calculation for a single node:

```rust
pub struct NodeForce {
    pub node_id: u32,
    pub force_x: f32,
    pub force_y: f32,
    pub force_z: f32,
}
```

### PhysicsStepResult

Result of a simulation step:

```rust
pub struct PhysicsStepResult {
    pub nodes_updated: usize,
    pub total_energy: f32,
    pub max_displacement: f32,
    pub converged: bool,
    pub computation_time_ms: f32,
}
```

### GpuDeviceInfo

GPU hardware information:

```rust
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub total_memory_mb: usize,
    pub free_memory_mb: usize,
    pub multiprocessor_count: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
}
```

### PhysicsStatistics

Runtime statistics:

```rust
pub struct PhysicsStatistics {
    pub total_steps: u64,
    pub average_step_time_ms: f32,
    pub average_energy: f32,
    pub gpu_memory_used_mb: f32,
    pub cache_hit_rate: f32,
    pub last_convergence_iterations: u32,
}
```

## Usage Examples

### Basic Simulation

```rust
let mut adapter: Box<dyn GpuPhysicsAdapter> = Box::new(CudaPhysicsAdapter::new()?);

// Load graph and parameters
let graph = Arc::new(load_graph_from_db().await?);
let params = PhysicsParameters::default();

adapter.initialize(graph, params).await?;

// Run a single physics step
let result = adapter.step().await?;
println!("Updated {} nodes, energy: {}", result.nodes_updated, result.total_energy);

// Or run until convergence
let result = adapter.simulate_until_convergence().await?;
println!("Converged in {} iterations", result.nodes_updated);
```

### Force Computation and Position Updates

```rust
// Compute forces manually
let forces = adapter.compute_forces().await?;
for force in &forces {
    println!("Node {}: force = ({}, {}, {})",
        force.node_id, force.force_x, force.force_y, force.force_z
    );
}

// Update positions based on forces
let new_positions = adapter.update_positions(&forces).await?;
for (node_id, x, y, z) in new_positions {
    println!("Node {} moved to ({}, {}, {})", node_id, x, y, z);
}
```

### Interactive Manipulation

```rust
// Pin nodes (user dragging)
adapter.pin_nodes(vec![
    (1, 0.0, 0.0, 0.0),   // Pin node 1 at origin
    (2, 10.0, 5.0, 0.0),  // Pin node 2 at (10, 5, 0)
]).await?;

// Apply external forces (e.g., gravity, wind)
adapter.apply_external_forces(vec![
    (3, 0.0, -9.8, 0.0),  // Gravity on node 3
    (4, 5.0, 0.0, 0.0),   // Wind on node 4
]).await?;

// Run simulation with constraints
let result = adapter.step().await?;

// Unpin nodes when done
adapter.unpin_nodes(vec![1, 2]).await?;
```

### Parameter Tuning

```rust
// Adjust physics parameters dynamically
let mut params = PhysicsParameters::default();
params.repulsion_strength = 150.0;  // Increase node spacing
params.damping = 0.9;               // Slower convergence
params.max_velocity = 5.0;          // Limit speed

adapter.update_parameters(params).await?;

// Run with new parameters
let result = adapter.step().await?;
```

### GPU Status Monitoring

```rust
// Get GPU information
let gpu_info = adapter.get_gpu_status().await?;
println!("GPU: {}", gpu_info.device_name);
println!("Memory: {}/{} MB", gpu_info.free_memory_mb, gpu_info.total_memory_mb);
println!("Compute: {}.{}", gpu_info.compute_capability.0, gpu_info.compute_capability.1);

// Get runtime statistics
let stats = adapter.get_statistics().await?;
println!("Average step time: {}ms", stats.average_step_time_ms);
println!("GPU memory used: {}MB", stats.gpu_memory_used_mb);
println!("Convergence iterations: {}", stats.last_convergence_iterations);
```

## Implementation Notes

### CUDA Kernel Integration

The adapter wraps CUDA kernels for force computation:

```cuda
// src/cuda/force_directed.cu
__global__ void compute_repulsion_forces(
    const float3* positions,
    float3* forces,
    int num_nodes,
    float repulsion_strength
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = positions[i];

    // Compute pairwise repulsion
    for (int j = 0; j < num_nodes; j++) {
        if (i == j) continue;

        float3 pos_j = positions[j];
        float3 delta = make_float3(
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z
        );

        float dist_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        float dist = sqrtf(dist_sq + 1e-6f);

        float magnitude = repulsion_strength / (dist_sq + 1e-6f);

        force.x += delta.x / dist * magnitude;
        force.y += delta.y / dist * magnitude;
        force.z += delta.z / dist * magnitude;
    }

    forces[i] = force;
}
```

### Rust-CUDA Bridge

```rust
use cudarc::driver::*;

pub struct CudaPhysicsAdapter {
    device: CudaDevice,
    positions_gpu: CudaSlice<f32>,
    velocities_gpu: CudaSlice<f32>,
    forces_gpu: CudaSlice<f32>,
    params: PhysicsParameters,
    stats: PhysicsStatistics,
}

impl CudaPhysicsAdapter {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;

        Ok(Self {
            device,
            positions_gpu: CudaSlice::new(),
            velocities_gpu: CudaSlice::new(),
            forces_gpu: CudaSlice::new(),
            params: PhysicsParameters::default(),
            stats: PhysicsStatistics::default(),
        })
    }

    async fn compute_forces_cuda(&mut self) -> Result<Vec<NodeForce>> {
        let num_nodes = self.positions_gpu.len() / 3;

        // Launch CUDA kernel
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((num_nodes + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            compute_repulsion_forces<<<cfg>>>(
                self.positions_gpu.as_device_ptr(),
                self.forces_gpu.as_device_ptr(),
                num_nodes as i32,
                self.params.repulsion_strength,
            );
        }

        // Copy forces back to CPU
        let forces_cpu = self.forces_gpu.to_host()?;

        // Convert to NodeForce
        let mut result = Vec::new();
        for i in 0..num_nodes {
            result.push(NodeForce {
                node_id: i as u32,
                force_x: forces_cpu[i * 3],
                force_y: forces_cpu[i * 3 + 1],
                force_z: forces_cpu[i * 3 + 2],
            });
        }

        Ok(result)
    }
}
```

### Fallback to CPU

Provide CPU fallback when GPU is unavailable:

```rust
pub enum PhysicsBackend {
    Cuda(CudaPhysicsAdapter),
    Cpu(CpuPhysicsAdapter),
}

impl PhysicsBackend {
    pub fn new() -> Result<Self> {
        match CudaPhysicsAdapter::new() {
            Ok(cuda) => Ok(Self::Cuda(cuda)),
            Err(_) => Ok(Self::Cpu(CpuPhysicsAdapter::new())),
        }
    }
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum GpuPhysicsAdapterError {
    #[error("GPU not available")]
    GpuNotAvailable,

    #[error("Physics computation error: {0}")]
    ComputationError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Graph not loaded")]
    GraphNotLoaded,
}
```

## Performance Benchmarks

Target performance (CUDA adapter, RTX 3090):
- Initialize (1000 nodes): < 50ms
- Single step (1000 nodes): < 5ms
- Single step (10,000 nodes): < 20ms
- Convergence (1000 nodes): < 500ms

**GPU vs CPU Speedup**:
- 1,000 nodes: 10-20x
- 10,000 nodes: 50-100x
- 100,000 nodes: 200-500x

## References

- **Force-Directed Graph Drawing**: https://en.wikipedia.org/wiki/Force-directed_graph_drawing
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **cudarc**: https://github.com/coreylowman/cudarc

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
