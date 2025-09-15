# Hybrid CPU-WASM/GPU Architecture for "Breaking the Sorting Barrier" SSSP Algorithm

## Executive Summary

This document presents a hybrid computing architecture that combines CPU-WASM for sophisticated algorithmic control with GPU for massively parallel computation to implement the O(m log^(2/3) n) "Breaking the Sorting Barrier" SSSP algorithm. The hybrid approach leverages the strengths of each computing platform while mitigating their individual limitations.

## 1. Algorithm Overview

The paper's breakthrough SSSP algorithm achieves O(m log^(2/3) n) complexity through:
- **Recursive BMSSP (Bounded Multi-Source Shortest Path)** structure with log n/t levels
- **FindPivots algorithm** for frontier reduction to |U|/log^Ω(1)(n)
- **Adaptive heap operations** (Pull, Insert, BatchPrepend) with sophisticated data structures
- **Divide-and-conquer** partitioning avoiding the Θ(n log n) sorting barrier

### Key Parameters
- `k = ⌊log^(1/3)(n)⌋` - pivot detection parameter
- `t = ⌊log^(2/3)(n)⌋` - recursion branching factor
- `l ∈ [0, ⌈log(n)/t⌉]` - recursion depth level

## 2. Hybrid Architecture Design

### 2.1 CPU-WASM Components (Orchestration & Control)

#### 2.1.1 Recursive BMSSP Controller
**Purpose**: Manages the recursive divide-and-conquer structure
**Implementation**: WASM module with Rust/C++ backend
**Responsibilities**:
- Maintains recursion tree with up to ⌈log(n)/t⌉ levels
- Coordinates frontier partitioning into 2^t pieces
- Manages boundary conditions (B, B') across recursion levels
- Handles successful vs. partial execution branching logic

```rust
// WASM Interface
pub struct BMSSPController {
    recursion_level: u32,
    max_depth: u32,
    frontier_bounds: Vec<f32>,
    pivot_sets: Vec<Vec<u32>>,
    gpu_bridge: GPUBridge,
}

impl BMSSPController {
    pub fn execute_bmssp(&mut self, level: u32, bound: f32,
                         frontier: &[u32]) -> BMSSPResult {
        if level == 0 {
            return self.base_case_dijkstra(bound, frontier);
        }

        // FindPivots orchestration
        let (pivots, working_set) = self.find_pivots(bound, frontier);

        // Recursive calls coordination
        let mut results = Vec::new();
        for chunk in self.partition_frontier(&pivots) {
            let sub_result = self.execute_bmssp(level - 1,
                                               chunk.bound, &chunk.vertices);
            results.push(sub_result);
        }

        self.merge_results(results)
    }
}
```

#### 2.1.2 FindPivots Algorithm Implementation
**Purpose**: Implements the sophisticated pivot selection logic
**Key Features**:
- k-step Bellman-Ford relaxation from frontier vertices
- Shortest path tree construction and analysis
- Pivot identification (vertices with SPT size ≥ k)
- Frontier size reduction to |U|/k

```rust
pub struct FindPivots {
    k_parameter: u32,
    working_set: Vec<u32>,
    spt_sizes: HashMap<u32, u32>,
    gpu_relaxation: RelaxationEngine,
}

impl FindPivots {
    pub fn execute(&mut self, bound: f32, frontier: &[u32]) -> (Vec<u32>, Vec<u32>) {
        // Step 1: k-step relaxation via GPU
        let mut current_wave = frontier.to_vec();
        for step in 0..self.k_parameter {
            let next_wave = self.gpu_relaxation.parallel_relax(&current_wave, bound);
            self.working_set.extend(&next_wave);
            current_wave = next_wave;

            if self.working_set.len() > self.k_parameter * frontier.len() {
                return (frontier.to_vec(), self.working_set.clone());
            }
        }

        // Step 2: Construct shortest path forest
        let forest = self.build_spt_forest(&self.working_set);

        // Step 3: Identify pivots (tree roots with ≥ k vertices)
        let pivots = self.extract_pivots(&forest, self.k_parameter);

        (pivots, self.working_set.clone())
    }
}
```

#### 2.1.3 Adaptive Heap Data Structure
**Purpose**: Implements the block-based linked list from Lemma 3.3
**Operations**:
- **Insert**: O(max{1, log(N/M)}) amortized time
- **BatchPrepend**: O(L·max{1, log(L/M)}) for L elements
- **Pull**: O(|S'|) time to extract M smallest elements

```rust
pub struct AdaptiveHeap {
    d0_sequence: LinkedList<Block>,  // Batch prepend blocks
    d1_sequence: LinkedList<Block>,  // Individual insert blocks
    d1_bounds: BTreeMap<f32, usize>, // Upper bounds for d1 blocks
    block_size_m: usize,
    max_elements_n: usize,
}

pub struct Block {
    elements: Vec<(u32, f32)>,  // (vertex_id, distance) pairs
    upper_bound: f32,
    next: Option<Box<Block>>,
}

impl AdaptiveHeap {
    pub fn insert(&mut self, vertex: u32, distance: f32) -> Result<(), HeapError> {
        // Find appropriate block in D1 via binary search
        let target_block = self.find_insertion_block(distance)?;
        target_block.elements.push((vertex, distance));

        // Split if block exceeds M elements
        if target_block.elements.len() > self.block_size_m {
            self.split_block(target_block)?;
        }
        Ok(())
    }

    pub fn batch_prepend(&mut self, elements: Vec<(u32, f32)>) -> Result<(), HeapError> {
        if elements.len() <= self.block_size_m {
            let new_block = Block::new(elements);
            self.d0_sequence.push_front(new_block);
        } else {
            // Create multiple blocks with median partitioning
            let blocks = self.partition_by_medians(elements, self.block_size_m)?;
            for block in blocks.into_iter().rev() {
                self.d0_sequence.push_front(block);
            }
        }
        Ok(())
    }

    pub fn pull(&mut self, count: usize) -> Result<(Vec<(u32, f32)>, f32), HeapError> {
        let mut result = Vec::with_capacity(count);
        let mut collected = 0;

        // Collect from D0 and D1 prefixes
        let d0_prefix = self.collect_d0_prefix(count - collected);
        result.extend(d0_prefix);
        collected += d0_prefix.len();

        if collected < count {
            let d1_prefix = self.collect_d1_prefix(count - collected);
            result.extend(d1_prefix);
            collected += d1_prefix.len();
        }

        // Extract smallest 'count' elements and determine separator
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result.truncate(count);

        let separator = self.find_separator(&result)?;

        // Remove extracted elements from data structure
        self.remove_elements(&result)?;

        Ok((result, separator))
    }
}
```

### 2.2 GPU Components (Massively Parallel Operations)

#### 2.2.1 Parallel Edge Relaxation Kernels
**Purpose**: Handle the computationally intensive edge relaxation operations
**Key Features**:
- Frontier-based parallel relaxation
- Atomic distance updates for thread safety
- Efficient memory coalescing patterns

```cuda
// CUDA Kernel for k-step relaxation
__global__ void parallel_k_step_relaxation(
    float* __restrict__ distances,
    const int* __restrict__ current_frontier,
    int frontier_size,
    const int* __restrict__ csr_row_offsets,
    const int* __restrict__ csr_col_indices,
    const float* __restrict__ edge_weights,
    int* __restrict__ next_frontier_flags,
    float bound_B,
    int step_count,
    int total_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= frontier_size) return;

    int vertex = current_frontier[tid];
    float vertex_dist = distances[vertex];

    if (vertex_dist >= bound_B) return;

    // Process all outgoing edges
    int edge_start = csr_row_offsets[vertex];
    int edge_end = csr_row_offsets[vertex + 1];

    for (int edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
        int neighbor = csr_col_indices[edge_idx];
        float edge_weight = edge_weights[edge_idx];
        float new_distance = vertex_dist + edge_weight;

        if (new_distance < bound_B) {
            float old_distance = atomicMinFloat(&distances[neighbor], new_distance);

            // Mark for next frontier if distance improved
            if (new_distance < old_distance) {
                next_frontier_flags[neighbor] = 1;
            }
        }
    }
}

// Atomic minimum operation for float
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}
```

#### 2.2.2 Frontier Compaction and Management
**Purpose**: Efficiently maintain active vertex sets between iterations
**Techniques**:
- Stream compaction for sparse frontiers
- Prefix sum-based index calculation
- Double buffering for ping-pong execution

```cuda
__global__ void compact_frontier_kernel(
    const int* __restrict__ frontier_flags,
    int* __restrict__ compacted_frontier,
    int* __restrict__ frontier_size_output,
    const int total_vertices
) {
    extern __shared__ int shared_temp[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Load flags into shared memory
    int flag = (tid < total_vertices) ? frontier_flags[tid] : 0;
    shared_temp[local_tid] = flag;
    __syncthreads();

    // Perform block-level prefix sum
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (local_tid >= stride) {
            temp = shared_temp[local_tid - stride];
        }
        __syncthreads();
        if (local_tid >= stride) {
            shared_temp[local_tid] += temp;
        }
        __syncthreads();
    }

    // Write compacted result
    if (flag && tid < total_vertices) {
        int output_index = shared_temp[local_tid] - 1;
        if (blockIdx.x > 0) {
            // Add global prefix (computed separately)
            output_index += block_prefix_sums[blockIdx.x];
        }
        compacted_frontier[output_index] = tid;
    }
}
```

#### 2.2.3 Shortest Path Tree Construction
**Purpose**: Build SPT forest for pivot identification in parallel
**Approach**:
- Parallel tree construction using parent pointers
- Subtree size calculation via bottom-up traversal
- Root identification for pivot detection

```cuda
__global__ void build_spt_kernel(
    const float* __restrict__ distances,
    const int* __restrict__ predecessors,
    const int* __restrict__ vertices,
    int vertex_count,
    int* __restrict__ subtree_sizes,
    int* __restrict__ tree_roots
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= vertex_count) return;

    int vertex = vertices[tid];
    int current = vertex;
    int path_length = 0;

    // Follow parent pointers to find root
    while (predecessors[current] != -1 && path_length < vertex_count) {
        current = predecessors[current];
        path_length++;
    }

    tree_roots[tid] = current;

    // Atomic increment of subtree size at root
    atomicAdd(&subtree_sizes[current], 1);
}
```

### 2.3 Communication Bridge (CPU-GPU Interface)

#### 2.3.1 Memory Management
**Design**: Zero-copy transfers using pinned memory
**Features**:
- Pre-allocated pinned host memory pools
- GPU memory pools with dynamic allocation
- Memory mapping for efficient data transfer

```rust
pub struct GPUMemoryManager {
    pinned_distances: PinnedBuffer<f32>,
    pinned_frontiers: PinnedBuffer<u32>,
    pinned_csr_data: CSRBuffers,
    gpu_distances: DeviceBuffer<f32>,
    gpu_frontiers: DeviceBuffer<u32>,
    gpu_csr_data: DeviceCSRBuffers,
    stream_pool: Vec<CudaStream>,
}

impl GPUMemoryManager {
    pub fn new(max_vertices: usize, max_edges: usize) -> Result<Self, CudaError> {
        let pinned_distances = PinnedBuffer::new(max_vertices)?;
        let pinned_frontiers = PinnedBuffer::new(max_vertices)?;
        let gpu_distances = DeviceBuffer::new(max_vertices)?;
        let gpu_frontiers = DeviceBuffer::new(max_vertices)?;

        Ok(GPUMemoryManager {
            pinned_distances,
            pinned_frontiers,
            pinned_csr_data: CSRBuffers::new(max_vertices, max_edges)?,
            gpu_distances,
            gpu_frontiers,
            gpu_csr_data: DeviceCSRBuffers::new(max_vertices, max_edges)?,
            stream_pool: (0..4).map(|_| CudaStream::new()).collect::<Result<Vec<_>, _>>()?,
        })
    }

    pub async fn transfer_to_gpu(&mut self,
                                distances: &[f32],
                                frontier: &[u32]) -> Result<(), CudaError> {
        // Copy to pinned memory
        self.pinned_distances.copy_from_slice(distances)?;
        self.pinned_frontiers.copy_from_slice(frontier)?;

        // Async transfer to GPU using streams
        let stream = &self.stream_pool[0];

        futures::join!(
            self.gpu_distances.copy_from_host_async(&self.pinned_distances, stream),
            self.gpu_frontiers.copy_from_host_async(&self.pinned_frontiers, stream)
        );

        stream.synchronize()?;
        Ok(())
    }
}
```

#### 2.3.2 Event-Based Synchronization
**Purpose**: Coordinate CPU-WASM orchestration with GPU computation
**Implementation**: CUDA events and async Rust futures

```rust
pub struct SynchronizationManager {
    computation_events: Vec<CudaEvent>,
    transfer_events: Vec<CudaEvent>,
    streams: Vec<CudaStream>,
    task_queue: VecDeque<ComputeTask>,
}

#[derive(Debug)]
pub enum ComputeTask {
    RelaxationStep { level: u32, frontier: Vec<u32>, bound: f32 },
    FrontierCompaction { input_flags: Vec<i32> },
    SPTConstruction { vertices: Vec<u32> },
    PivotExtraction { spt_sizes: Vec<u32>, threshold: u32 },
}

impl SynchronizationManager {
    pub async fn execute_task(&mut self, task: ComputeTask) -> Result<TaskResult, ComputeError> {
        match task {
            ComputeTask::RelaxationStep { level, frontier, bound } => {
                let event = CudaEvent::new()?;
                let stream = &self.streams[level as usize % self.streams.len()];

                // Launch kernel
                self.launch_relaxation_kernel(&frontier, bound, stream)?;
                event.record(stream)?;

                // Return future that resolves when GPU work completes
                Ok(TaskResult::Relaxation {
                    completion_future: Box::pin(async move {
                        event.synchronize().await?;
                        // Read back results
                        Ok(())
                    })
                })
            },
            _ => todo!("Implement other task types")
        }
    }
}
```

### 2.4 Integration Architecture

#### 2.4.1 Main Algorithm Flow
```rust
pub struct HybridSSSpSolver {
    bmssp_controller: BMSSPController,
    gpu_manager: GPUMemoryManager,
    sync_manager: SynchronizationManager,
    adaptive_heap: AdaptiveHeap,
    graph_data: CSRGraph,
}

impl HybridSSSpSolver {
    pub async fn solve(&mut self, source: u32) -> Result<Vec<f32>, SolverError> {
        // Initialize with source vertex
        self.initialize_distances(source)?;

        // Main BMSSP call
        let result = self.bmssp_controller.execute_bmssp(
            self.calculate_max_depth(),
            f32::INFINITY,
            &[source]
        ).await?;

        // Extract final distances
        Ok(result.distances)
    }

    fn calculate_max_depth(&self) -> u32 {
        let n = self.graph_data.vertex_count() as f32;
        let t = n.log2().powf(2.0/3.0).floor() as u32;
        (n.log2() / t as f32).ceil() as u32
    }
}
```

## 3. Performance Analysis and Tradeoffs

### 3.1 Why Hybrid Beats Pure GPU

#### 3.1.1 Algorithmic Complexity Benefits
**Recursive Control Flow**: The algorithm's recursive BMSSP structure with sophisticated branching logic is naturally suited to CPU execution. GPU's SIMT model struggles with:
- Dynamic recursion depth (up to log n/t levels)
- Irregular control flow in pivot selection
- Variable workload distribution across subproblems

**Adaptive Data Structures**: The block-based linked list with Pull/Insert/BatchPrepend operations requires:
- Complex pointer manipulation unsuitable for GPU
- Dynamic memory allocation patterns
- Sequential dependency chains in heap operations

#### 3.1.2 Memory Access Patterns
**CPU-WASM Advantages**:
- Irregular memory access in tree traversal
- Random access to predecessor arrays
- Cache-friendly sequential processing in heap operations

**GPU Advantages**:
- Regular, coalesced access in edge relaxation
- Parallel reduction operations in frontier compaction
- SIMD-style operations on distance arrays

### 3.2 Why WASM Provides Value

#### 3.2.1 Portability and Safety
- **Cross-platform Execution**: Runs identically on x86, ARM, and other architectures
- **Memory Safety**: Bounds checking prevents buffer overflows in complex pointer operations
- **Sandboxed Execution**: Isolated from system-level GPU driver interactions

#### 3.2.2 Performance Benefits
- **Near-native Performance**: WASM compilation typically achieves 80-90% of native C++ performance
- **Predictable Execution**: No garbage collection pauses, deterministic memory management
- **JIT Optimization**: Modern WASM runtimes provide sophisticated optimization

### 3.3 Performance Implications

#### 3.3.1 Communication Overhead Analysis
**Data Transfer Costs**:
- Graph Structure: O(m) CSR data (transferred once)
- Distance Array: O(n) floats (transferred per recursion level)
- Frontier Sets: O(|frontier|) vertices (variable size)

**Mitigation Strategies**:
- Pinned memory allocation eliminates copy overhead
- Asynchronous transfers overlap computation
- Stream-based execution hides latency

#### 3.3.2 Workload Distribution
**CPU-WASM Workload** (Sequential/Complex):
- FindPivots algorithm: O(k|U|) where k = log^(1/3) n
- Heap operations: O(log(N/M)) per operation
- Recursion management: O(log n) overhead per level

**GPU Workload** (Parallel/Regular):
- Edge relaxation: O(m) with high parallelism
- Frontier compaction: O(n) with efficient reduction
- Distance updates: O(n) with atomic operations

### 3.4 When Hybrid Beats Pure GPU

#### 3.4.1 Graph Characteristics Favoring Hybrid
- **High-diameter graphs**: More recursion levels benefit from CPU orchestration
- **Irregular degree distribution**: Load balancing challenges on GPU
- **Large frontier variations**: Adaptive heap operations essential

#### 3.4.2 Scale Considerations
- **Small graphs (n < 10^4)**: CPU overhead dominates, pure CPU may be faster
- **Medium graphs (10^4 < n < 10^6)**: Hybrid approach optimal
- **Large graphs (n > 10^6)**: GPU parallelism becomes dominant factor

## 4. Implementation Roadmap

### Phase 1: Core Infrastructure
1. **WASM Runtime Setup**: Establish Rust-to-WASM compilation pipeline
2. **GPU Memory Management**: Implement pinned memory pools and device buffers
3. **Basic Synchronization**: Event-based CPU-GPU coordination

### Phase 2: Algorithm Components
1. **FindPivots Implementation**: CPU-WASM with GPU relaxation calls
2. **Adaptive Heap**: Block-based data structure with efficient operations
3. **Parallel Kernels**: Edge relaxation and frontier compaction on GPU

### Phase 3: Integration and Optimization
1. **BMSSP Controller**: Full recursive algorithm orchestration
2. **Performance Tuning**: Memory layout optimization, kernel tuning
3. **Benchmark Validation**: Verify O(m log^(2/3) n) complexity

### Phase 4: Production Hardening
1. **Error Handling**: Robust failure recovery across CPU-GPU boundary
2. **Memory Management**: Dynamic allocation and cleanup
3. **API Integration**: Integration with existing graph processing systems

## 5. Conclusion

The hybrid CPU-WASM/GPU architecture provides an optimal balance for implementing the "Breaking the Sorting Barrier" SSSP algorithm. By leveraging CPU-WASM for sophisticated algorithmic control and GPU for massively parallel operations, we achieve:

- **Algorithmic Fidelity**: Full implementation of the paper's recursive BMSSP structure
- **Performance Optimization**: Near-optimal utilization of both CPU and GPU resources
- **Portability**: WASM ensures cross-platform compatibility
- **Scalability**: Architecture scales from medium to very large graph instances

The design respects the fundamental insight that different algorithmic phases have different computational characteristics, and the optimal approach is to match each phase to its most suitable computing platform.

---

**Architecture Document Version**: 1.0
**Author**: System Architecture Designer
**Date**: 2025-09-15