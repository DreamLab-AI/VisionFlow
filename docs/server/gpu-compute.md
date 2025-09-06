# GPU Compute Architecture

VisionFlow leverages a high-performance GPU compute system to accelerate graph physics simulations and complex analytical workloads. The entire system is built around a single, unified CUDA kernel, `visionflow_unified.cu`, which has replaced all legacy kernels for improved performance, maintainability, and efficiency.

## The Unified Kernel: `visionflow_unified.cu`

All GPU-side computations are handled by a single, highly optimized CUDA C++ kernel. This unified approach provides several key advantages:
-   **Reduced Complexity**: A single codebase is easier to maintain, debug, and optimize.
-   **Improved Performance**: Code and data can be more effectively optimized for a single execution pipeline.
-   **Flexibility**: The kernel is designed to be highly configurable and can switch between different computational modes at runtime.

The `GPUComputeActor` is the sole interface to this kernel, managing data transfer and kernel launches.

## Compute Modes

The unified kernel can operate in one of four modes, which are selected by the `GPUComputeActor` based on the specific task being performed.

1.  **`Basic` Mode**:
    *   **Use Case**: Standard force-directed graph layout.
    *   **Description**: This mode applies a simple and fast physics simulation, ideal for initial graph layouts or when advanced features are not required.

2.  **`DualGraph` Mode**:
    *   **Use Case**: Visualizing the knowledge graph and the agent graph simultaneously.
    *   **Description**: This mode is optimized to handle two separate graph structures, each with its own set of physics parameters, within a single simulation.

3.  **`Constraints` Mode**:
    *   **Use Case**: Applying semantic or structural rules to the graph layout.
    *   **Description**: In this mode, the kernel applies additional forces to satisfy layout constraints, such as forcing certain nodes to separate or align.

4.  **`VisualAnalytics` Mode**:
    *   **Use Case**: Running specialized, compute-intensive graph analysis.
    *   **Description**: This mode is used for advanced analytics that are best performed on the GPU, such as community detection or centrality calculations.

## Memory Layout: Structure of Arrays (SoA)

To maximize memory bandwidth and computational efficiency on the GPU, all graph data is organized using a Structure of Arrays (SoA) layout.

Instead of storing an array of `Node` objects (Array of Structures, AoS), we store separate arrays for each attribute (e.g., `position_x`, `position_y`, `velocity_x`).

**Benefits of SoA:**
-   **Coalesced Memory Access**: When a kernel needs to read the positions of all nodes, it can read them from a contiguous block of memory. This is the most efficient way for a GPU to access memory and dramatically improves performance.
-   **Higher Cache Utilization**: Since only relevant data is loaded into the cache, memory is used more effectively.
-   **Better SIMD Vectorization**: The GPU can perform operations on multiple data points (e.g., multiple x-positions) in a single instruction.

## GPU-Accelerated Analytics

The unified kernel is not limited to physics simulations. It also provides entry points for offloading heavy analytical tasks to the GPU. The `GPUComputeActor` can handle messages like `PerformGPUClustering` and `ComputeShortestPaths`.

When one of these messages is received:
1.  The `GPUComputeActor` switches the kernel to the `VisualAnalytics` mode.
2.  It passes the relevant graph data and algorithm parameters to the GPU.
3.  The kernel executes the computation in parallel across thousands of threads.
4.  The results are returned to the `GPUComputeActor`, which then passes them back to the requester.

This offloading mechanism allows the server to perform complex, large-scale graph analysis without blocking the main application threads, ensuring the system remains responsive.

For more details on clustering, see the [Graph Clustering documentation](features/clustering.md).