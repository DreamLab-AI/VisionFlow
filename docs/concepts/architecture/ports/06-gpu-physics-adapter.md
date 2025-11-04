# GpuPhysicsAdapter Port

## Purpose

The **GpuPhysicsAdapter** port provides GPU-accelerated physics simulation for knowledge graph layout using force-directed algorithms. It abstracts CUDA implementations for physics computations.

## Location

- **Trait Definition**: `src/ports/gpu-physics-adapter.rs`
- **Adapter Implementation**: `src/adapters/cuda-physics-adapter.rs`

## Interface

```rust
#[async-trait]
pub trait GpuPhysicsAdapter: Send + Sync {
    // Initialization
    async fn initialize(&mut self, graph: Arc<GraphData>, params: PhysicsParameters) -> Result<()>;
    async fn cleanup(&mut self) -> Result<()>;

    // Force computation
    async fn compute-forces(&mut self) -> Result<Vec<NodeForce>>;
    async fn update-positions(&mut self, forces: &[NodeForce]) -> Result<Vec<(u32, f32, f32, f32)>>;
    async fn step(&mut self) -> Result<PhysicsStepResult>;

    // Simulation control
    async fn simulate-until-convergence(&mut self) -> Result<PhysicsStepResult>;
    async fn reset(&mut self) -> Result<()>;

    // External forces and constraints
    async fn apply-external-forces(&mut self, forces: Vec<(u32, f32, f32, f32)>) -> Result<()>;
    async fn pin-nodes(&mut self, nodes: Vec<(u32, f32, f32, f32)>) -> Result<()>;
    async fn unpin-nodes(&mut self, node-ids: Vec<u32>) -> Result<()>;

    // Configuration
    async fn update-parameters(&mut self, params: PhysicsParameters) -> Result<()>;
    async fn update-graph-data(&mut self, graph: Arc<GraphData>) -> Result<()>;

    // Status and statistics
    async fn get-gpu-status(&self) -> Result<GpuDeviceInfo>;
    async fn get-statistics(&self) -> Result<PhysicsStatistics>;
}
```

## Types

### PhysicsParameters

Physics simulation configuration:

```rust
pub struct PhysicsParameters {
    pub time-step: f32,
    pub damping: f32,
    pub spring-constant: f32,
    pub repulsion-strength: f32,
    pub attraction-strength: f32,
    pub max-velocity: f32,
    pub convergence-threshold: f32,
    pub max-iterations: u32,
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            time-step: 0.016,          // 60 FPS
            damping: 0.8,
            spring-constant: 0.01,
            repulsion-strength: 100.0,
            attraction-strength: 0.1,
            max-velocity: 10.0,
            convergence-threshold: 0.01,
            max-iterations: 1000,
        }
    }
}
```

### NodeForce

Force calculation for a single node:

```rust
pub struct NodeForce {
    pub node-id: u32,
    pub force-x: f32,
    pub force-y: f32,
    pub force-z: f32,
}
```

### PhysicsStepResult

Result of a simulation step:

```rust
pub struct PhysicsStepResult {
    pub nodes-updated: usize,
    pub total-energy: f32,
    pub max-displacement: f32,
    pub converged: bool,
    pub computation-time-ms: f32,
}
```

### GpuDeviceInfo

GPU hardware information:

```rust
pub struct GpuDeviceInfo {
    pub device-id: u32,
    pub device-name: String,
    pub compute-capability: (u32, u32),
    pub total-memory-mb: usize,
    pub free-memory-mb: usize,
    pub multiprocessor-count: u32,
    pub warp-size: u32,
    pub max-threads-per-block: u32,
}
```

### PhysicsStatistics

Runtime statistics:

```rust
pub struct PhysicsStatistics {
    pub total-steps: u64,
    pub average-step-time-ms: f32,
    pub average-energy: f32,
    pub gpu-memory-used-mb: f32,
    pub cache-hit-rate: f32,
    pub last-convergence-iterations: u32,
}
```

## Usage Examples

### Basic Simulation

```rust
let mut adapter: Box<dyn GpuPhysicsAdapter> = Box::new(CudaPhysicsAdapter::new()?);

// Load graph and parameters
let graph = Arc::new(load-graph-from-db().await?);
let params = PhysicsParameters::default();

adapter.initialize(graph, params).await?;

// Run a single physics step
let result = adapter.step().await?;
println!("Updated {} nodes, energy: {}", result.nodes-updated, result.total-energy);

// Or run until convergence
let result = adapter.simulate-until-convergence().await?;
println!("Converged in {} iterations", result.nodes-updated);
```

### Force Computation and Position Updates

```rust
// Compute forces manually
let forces = adapter.compute-forces().await?;
for force in &forces {
    println!("Node {}: force = ({}, {}, {})",
        force.node-id, force.force-x, force.force-y, force.force-z
    );
}

// Update positions based on forces
let new-positions = adapter.update-positions(&forces).await?;
for (node-id, x, y, z) in new-positions {
    println!("Node {} moved to ({}, {}, {})", node-id, x, y, z);
}
```

### Interactive Manipulation

```rust
// Pin nodes (user dragging)
adapter.pin-nodes(vec![
    (1, 0.0, 0.0, 0.0),   // Pin node 1 at origin
    (2, 10.0, 5.0, 0.0),  // Pin node 2 at (10, 5, 0)
]).await?;

// Apply external forces (e.g., gravity, wind)
adapter.apply-external-forces(vec![
    (3, 0.0, -9.8, 0.0),  // Gravity on node 3
    (4, 5.0, 0.0, 0.0),   // Wind on node 4
]).await?;

// Run simulation with constraints
let result = adapter.step().await?;

// Unpin nodes when done
adapter.unpin-nodes(vec![1, 2]).await?;
```

### Parameter Tuning

```rust
// Adjust physics parameters dynamically
let mut params = PhysicsParameters::default();
params.repulsion-strength = 150.0;  // Increase node spacing
params.damping = 0.9;               // Slower convergence
params.max-velocity = 5.0;          // Limit speed

adapter.update-parameters(params).await?;

// Run with new parameters
let result = adapter.step().await?;
```

### GPU Status Monitoring

```rust
// Get GPU information
let gpu-info = adapter.get-gpu-status().await?;
println!("GPU: {}", gpu-info.device-name);
println!("Memory: {}/{} MB", gpu-info.free-memory-mb, gpu-info.total-memory-mb);
println!("Compute: {}.{}", gpu-info.compute-capability.0, gpu-info.compute-capability.1);

// Get runtime statistics
let stats = adapter.get-statistics().await?;
println!("Average step time: {}ms", stats.average-step-time-ms);
println!("GPU memory used: {}MB", stats.gpu-memory-used-mb);
println!("Convergence iterations: {}", stats.last-convergence-iterations);
```

## Implementation Notes

### CUDA Kernel Integration

The adapter wraps CUDA kernels for force computation:

```cuda
// src/cuda/force-directed.cu
--global-- void compute-repulsion-forces(
    const float3* positions,
    float3* forces,
    int num-nodes,
    float repulsion-strength
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num-nodes) return;

    float3 force = make-float3(0.0f, 0.0f, 0.0f);
    float3 pos-i = positions[i];

    // Compute pairwise repulsion
    for (int j = 0; j < num-nodes; j++) {
        if (i == j) continue;

        float3 pos-j = positions[j];
        float3 delta = make-float3(
            pos-i.x - pos-j.x,
            pos-i.y - pos-j.y,
            pos-i.z - pos-j.z
        );

        float dist-sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        float dist = sqrtf(dist-sq + 1e-6f);

        float magnitude = repulsion-strength / (dist-sq + 1e-6f);

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
    positions-gpu: CudaSlice<f32>,
    velocities-gpu: CudaSlice<f32>,
    forces-gpu: CudaSlice<f32>,
    params: PhysicsParameters,
    stats: PhysicsStatistics,
}

impl CudaPhysicsAdapter {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;

        Ok(Self {
            device,
            positions-gpu: CudaSlice::new(),
            velocities-gpu: CudaSlice::new(),
            forces-gpu: CudaSlice::new(),
            params: PhysicsParameters::default(),
            stats: PhysicsStatistics::default(),
        })
    }

    async fn compute-forces-cuda(&mut self) -> Result<Vec<NodeForce>> {
        let num-nodes = self.positions-gpu.len() / 3;

        // Launch CUDA kernel
        let cfg = LaunchConfig {
            block-dim: (256, 1, 1),
            grid-dim: ((num-nodes + 255) / 256, 1, 1),
            shared-mem-bytes: 0,
        };

        unsafe {
            compute-repulsion-forces<<<cfg>>>(
                self.positions-gpu.as-device-ptr(),
                self.forces-gpu.as-device-ptr(),
                num-nodes as i32,
                self.params.repulsion-strength,
            );
        }

        // Copy forces back to CPU
        let forces-cpu = self.forces-gpu.to-host()?;

        // Convert to NodeForce
        let mut result = Vec::new();
        for i in 0..num-nodes {
            result.push(NodeForce {
                node-id: i as u32,
                force-x: forces-cpu[i * 3],
                force-y: forces-cpu[i * 3 + 1],
                force-z: forces-cpu[i * 3 + 2],
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
            Err(-) => Ok(Self::Cpu(CpuPhysicsAdapter::new())),
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

- **Force-Directed Graph Drawing**: https://en.wikipedia.org/wiki/Force-directed-graph-drawing
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **cudarc**: https://github.com/coreylowman/cudarc

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
