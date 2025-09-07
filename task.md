Project Brief: GPU Analytics Engine Maturation
Objective:
To transition the application's analytics capabilities from a mix of simulated, CPU-based, and disabled features into a fully-functional, GPU-accelerated engine. The primary goal is to leverage CUDA for all heavy computational tasks, including clustering, anomaly detection, and advanced layout optimization, eliminating all CPU fallbacks and mock implementations.
Current State of Play:
The system currently operates in a hybrid state:
Functional GPU Physics: The core force-directed layout simulation is successfully implemented and running on the GPU (visionflow_unified.cu).
Functional CPU Pre-processing: Semantic analysis and advanced edge generation are working correctly on the CPU before the graph is sent to the GPU.
Implemented but Disabled Features: Stress Majorization and Semantic Constraints have been implemented but are turned off due to instability issues (e.g., "bouncing," "position explosions").
Simulated/Mocked Analytics: Clustering and Anomaly Detection are fully connected to the API but execute mock logic that returns procedurally generated data.
Disconnected/Orphaned Module: An advanced visual_analytics.rs module exists but is completely unused by the active GPU compute pipeline.
Phase 0: Core Architectural Decision
Before development begins, the team must make a critical decision regarding the GPU compute architecture.
Objective: Decide whether to extend the current UnifiedGPUCompute engine or migrate to the more advanced (but currently unused) VisualAnalyticsGPU data structures.
Current State of Play:
src/utils/unified_gpu_compute.rs is the active engine. It's simpler and directly powers the existing physics simulation. Its data structures are arrays of primitive types (e.g., pos_in_x: DeviceBuffer<f32>).
src/gpu/visual_analytics.rs is an orphaned module. It defines much richer, GPU-compatible data structures like TSNode (Temporal-Spatial Node) and TSEdge, which include fields for trajectories, semantic vectors, and topology metrics.
Key Actions:
Evaluate TSNode/TSEdge: Analyze the data structures in src/gpu/visual_analytics.rs. Determine if their complexity and features are necessary for the project's long-term goals.
Choose a Path:
Path A (Extend Current): Continue using unified_gpu_compute.rs. Add new buffers and kernels as needed for clustering and other analytics. (Recommended for faster initial progress).
Path B (Migrate): Refactor the entire GPU pipeline to use the TSNode and TSEdge structs. This is a major undertaking but may provide a more robust foundation for future features.
The rest of this plan assumes Path A, but the tasks can be adapted for Path B.
Phase 1: Stabilize and Re-enable Existing Features
This phase focuses on getting the implemented-but-disabled features working correctly before adding new ones.
Task 1.1: Fix and Re-enable Stress Majorization
Objective: Debug the instability issues ("position explosions") and re-enable the stress majorization algorithm for periodic global layout optimization.
Current State of Play: The algorithm is implemented in Rust (stress_majorization.rs) but is disabled by default in gpu_compute_actor.rs and models/constraints.rs via a configuration that sets its run interval to u32::MAX.
Affected Files:
src/physics/stress_majorization.rs (Core algorithm)
src/actors/gpu_compute_actor.rs (Where the feature is disabled)
src/actors/graph_actor.rs (Orchestrates the simulation loop)
src/models/constraints.rs (Default AdvancedParams configuration)
Key Actions:
Temporarily enable the feature by setting stress_step_interval_frames to a reasonable value (e.g., 600).
Reproduce and debug the "position explosions." This likely involves issues with the distance or weight matrices, or incorrect gradient updates.
Implement sanity checks within the solver to clamp extreme position values and prevent divergence.
(Optional GPU Porting): For performance, profile the solver and consider porting the most intensive parts (matrix operations) to CUDA kernels. The current implementation uses the nalgebra crate, which is CPU-only.
Task 1.2: Fix and Re-enable Semantic Constraints
Objective: Debug the layout instability ("bouncing") caused by semantic constraints and re-enable their generation and application in the physics simulation.
Current State of Play: The constraint generation logic in semantic_constraints.rs is functional. However, the GraphServiceActor is hardcoded to skip calling it during graph builds and dynamic updates.
Affected Files:
src/actors/graph_actor.rs (Where the feature is disabled)
src/physics/semantic_constraints.rs (Constraint generation logic)
src/utils/visionflow_unified.cu (The kernel that would need to apply these constraints)
Key Actions:
Re-enable the calls to generate_initial_semantic_constraints and update_dynamic_constraints within GraphServiceActor.
Reproduce and debug the "bouncing" behavior. This is likely caused by constraint forces being too strong or poorly balanced with the main physics forces (repulsion/attraction).
Implement logic in visionflow_unified.cu to accept and apply constraints. The UnifiedGPUCompute struct has a placeholder set_constraints method that needs to be implemented to upload constraint data to the GPU.
Modify the force_pass_kernel to read from the constraint buffer and add constraint-based forces to total_force.
Phase 2: Port Mocked Analytics to GPU
This is the core development phase to replace all simulated analytics with real GPU implementations.
Task 2.1: Implement GPU-Accelerated Clustering
Objective: Replace the mock clustering logic with one or more real parallel clustering algorithms implemented in CUDA.
Current State of Play: The API endpoint POST /api/analytics/clustering/run triggers a PerformGPUClustering message. The handler in GPUComputeActor executes a CPU-based simulation that returns random, procedural data. The CPU fallback in clustering.rs is also a simulation.
Affected Files:
src/actors/gpu_compute_actor.rs: Replace the mock handle implementation for PerformGPUClustering.
src/utils/visionflow_unified.cu: Add new CUDA kernels for clustering.
src/utils/unified_gpu_compute.rs: Add new methods to manage clustering buffers and launch the new kernels.
src/handlers/api_handler/analytics/clustering.rs: Remove the CPU fallback logic.
Key Actions:
Research: Choose appropriate parallel clustering algorithms. Good candidates are:
K-means: Highly parallelizable.
Spectral Clustering: Requires GPU-accelerated linear algebra (eigen-solvers), which can be complex.
DBSCAN: Can be parallelized but may be tricky with graph data.
Develop CUDA Kernels: Write the necessary CUDA kernels for the chosen algorithm(s) in visionflow_unified.cu. This will involve calculating distances/similarities between nodes and assigning cluster IDs.
Update UnifiedGPUCompute: Add GPU buffers for cluster assignments and any intermediate data. Create a new public method (e.g., run_clustering) to orchestrate the kernel launches.
Update GPUComputeActor: Modify the PerformGPUClustering handler to call the new run_clustering method on its unified_compute instance and return the real results.
Refactor API Handler: Remove the CPU fallback path in clustering.rs to ensure all clustering happens on the GPU.
Task 2.2: Implement GPU-Accelerated Anomaly Detection
Objective: Replace the random anomaly generator with a real GPU-based implementation that analyzes the graph structure and node properties.
Current State of Play: The API endpoints /api/analytics/anomaly/* interact with a global state that is populated by a tokio::spawn loop generating random anomalies in anomaly.rs.
Affected Files:
src/handlers/api_handler/analytics/anomaly.rs: Replace the timed simulation loop with calls to the GPU actor.
src/actors/gpu_compute_actor.rs: Add a new message handler for triggering anomaly detection.
src/utils/visionflow_unified.cu: Add new kernels for calculating anomaly scores.
Key Actions:
Research: Choose a suitable parallel anomaly detection algorithm. Good candidates for graph data include:
Local Outlier Factor (LOF): Can be parallelized by calculating local densities on the GPU.
Statistical Methods: Kernels can be written to compute statistical properties (e.g., degree distribution, centrality scores) in parallel, identifying outliers.
Develop CUDA Kernels: Write kernels to compute anomaly scores for each node and write them to a new GPU buffer.
Update Actors:
Add a new message (e.g., PerformGPUAnomalyDetection) to messages.rs.
Implement the handler for this message in GPUComputeActor. It should call a new method in UnifiedGPUCompute to run the anomaly detection kernels.
Refactor API Handler: Remove the simulation loop in anomaly.rs. The /api/analytics/anomaly/toggle endpoint should now trigger a periodic task that sends the PerformGPUAnomalyDetection message, and /api/analytics/anomaly/current should retrieve the latest results.
Phase 3: Advanced Feature Integration
With the core analytics running on the GPU, integrate them more deeply.
Task 3.1: Implement Real "AI Insights"
Objective: Replace the mock text generator with a system that synthesizes meaningful insights from the real clustering and anomaly detection results.
Current State of Play: The get_ai_insights handler in api_handler/analytics/mod.rs returns hardcoded and slightly randomized strings.
Affected Files:
src/handlers/api_handler/analytics/mod.rs
Key Actions:
Modify the get_ai_insights handler to first request the latest clustering and anomaly results from the GPUComputeActor.
Write Rust logic to interpret these results. For example:
"Detected 5 clusters. The largest cluster, 'Core Concepts', is highly coherent (0.85) and central to the graph."
"Node 'kernel.cu' is a critical anomaly with a score of 0.95, indicating it is structurally isolated despite its importance."
"A pattern of high-density clusters suggests a modular graph structure, which is a positive architectural sign."
Phase 4: Deprecation and Final Cleanup
The final phase is to remove all legacy code, mocks, and CPU fallbacks.
Objective: Ensure the entire analytics pipeline is GPU-only and remove all dead or simulated code.
Affected Files:
src/handlers/api_handler/analytics/clustering.rs
src/handlers/api_handler/analytics/anomaly.rs
src/gpu/visual_analytics.rs and the entire src/gpu directory if Path B was not chosen.
Any CPU fallback logic in other modules.
Key Actions:
Delete all simulated CPU functions for clustering and anomaly detection.
Remove all logic paths that constitute a "CPU fallback." The system should now return an error if the GPU is unavailable.
If Path A was chosen, delete the entire src/gpu/visual_analytics.rs module and its related exports to avoid confusion.
Review all FIXME and TODO comments related to GPU integration and address them.
Update documentation to reflect the new, fully GPU-accelerated architecture.