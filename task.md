Categorized files, roles, and potential instability risks

GPU execution layer (CUDA and wrapper)
src/utils/visionflow_unified.cu

Core kernel: visionflow_compute_kernel()
Forces: compute_basic_forces(), compute_dual_graph_forces(), compute_visual_analytics(), apply_constraints()
Role: Applies repulsion (1/r^2 with floor), springs toward natural_length, damping, velocity clamp, position integration with dt, boundary damping/soft clamp, plus warmup and temperature scaling.
Instability risks:
Parameter scale: Large SimParams.repel_k + small MIN_DISTANCE + short natural_length can push forces hard even with max_force clamp.
Temperature scaling: Early high temperature increases movement; if >1.0, can amplify forces before damping applies (temp_scale).
Boundary dynamics: Boundary kicks (boundary_force_strength=10) plus boundary_damping near 1.0 can cause bouncing at viewport edges.
Warmup and iteration handling: Warmup scales force quadratically initially (lines 458-466). If iteration counts advance faster than intended (see “double-execute” below), forces heat up quicker.
Dual-graph scaling: Mixed repulsion/spring scaling across graphs (lines 191, 209) adds complexity and can destabilize if graph ids/weights are wrong.
UnifiedGPUCompute::execute()

Role: Increments iteration, prepares GpuKernelParams, launches visionflow_compute_kernel, synchronizes, and copies positions back to host.
Instability risks:
Double stepping: If called more than once per frame, dt integration and iteration warmup advance twice (see server path “GetNodeData double-exec”).
Buffer sizing invariants: Uses fixed-length device buffers set at construction.
UnifiedGPUCompute::upload_positions()

Role: Initializes positions (golden-angle sphere) if all zero, else copies; affects initial overlap and repulsion.
Instability risks: If client/server positions reset to zero frequently, repeated re-seeding creates large displacements.
UnifiedGPUCompute::upload_edges()

Role: Copies edges; expects len == self.num_edges.
Instability risks:
Edge count mismatch after graph changes (unified context constructed with initial counts) can reject updates, leaving stale springs while nodes/positions continue to update.
SimParams mapping, Default

Role: Converts high-level SimulationParams to kernel SimParams; default picks stable-ish values (dt=0.016, damping=0.95, max_force=10).
Instability risks:
If server uses settings with high repulsion or dt, mapping directly increases forces.
temperature, boundary_damping, max_velocity must be consistent with kernel logic/clamps.
Backend orchestration layer (Rust actors, params, config)
GPUComputeActor::compute_forces_internal(), compute_forces_unified()

Role: Sets unified mode and params, runs GPU step, tracks iterations/failures.
Instability risks: Bad mode/param updates or failure-retry flips could stall damping logic if iteration resets inconsistently.
CRITICAL: Get-node-data double-execute

GPUComputeActor::get_node_data_internal() calls unified_compute.execute() again to fetch positions; i.e., one ComputeForces tick plus one extra execute to read positions = two kernel steps per “frame” in legacy mode.
Where it’s triggered: GraphServiceActor::run_legacy_gpu_step() sends ComputeForces and then queries GetNodeData after 5ms, causing a second step. This can easily present as exploding/bouncing nodes due to doubled integration and accelerated warmup.
GPUComputeActor::UpdateSimulationParams handler

Role: Accepts new SimulationParams, converts to SimParams, updates unified compute.
GPUComputeActor::update_graph_data_internal()

Role: Re-uploads positions/edges and maintains node index map.
Instability risks: Size change of edges without reconstructing UnifiedGPUCompute buffers causes upload failure and stale connectivity.
GraphServiceActor::run_simulation_step()

Role: 60fps tick; uses advanced GPU context if available, else legacy GPUComputeActor path.
Instability risks:
Legacy path’s forced double-exec per frame (see above).
Timing: 16ms interval not synchronized with GPU; dt mismatches if server dt != 0.016 but tick at 16ms.
GraphServiceActor::run_advanced_gpu_step()

Role: Direct unified context execute; no double-exec; then broadcasts encoded binary.
Instability risks: Edge/constraint/param staging logic correctness.
SimulationParams, From<&PhysicsSettings> for SimulationParams

Role: High-level physics values and conversion from YAML to runtime.
Instability risks:
Dynamic phase example values (e.g., time_step=0.12, repulsion=600) are much larger than the unified defaults; if used, they can swamp damping/force caps unless clamped.
max_force fixed to 10.0 in From-impl (line 195) independent of repulsion/spring scales; if too low, springs may underperform; if set higher by UI, nodes can overshoot before damping catches them.
AppState::new() initial param push

Role: Converts YAML physics to SimulationParams and sends UpdateSimulationParams to both actors on startup.
Instability risks: Unstable YAML defaults (repulsion, dt, temperature, boundary_damping).
socket_flow_handler “update_physics_params” branch

Role: Parses client WS JSON and sends UpdatePhysicsParams to GPUComputeActor.
Instability risks: Partial param updates missing crucial fields (e.g., dt/temperature), leaving mixed old/new values.
binary_protocol encode/decode, decode_node_data()

Role: Packs/unpacks positions for client; manages knowledge/agent flags.
Instability risks: Not a direct physics source but impacts client interpolation expectations.
gpu_diagnostics PTX checks

Role: Verifies PTX availability.
Instability risks: Missing PTX prevents GPU path and may fallback to non-synchronized modes.
Client control and rendering
PhysicsEngineControls component

Initial settings read: physicsSettings usage
Update handler: handleForceParamChange()
Role: Lets user change repulsion, spring, damping, dt, temperature, maxVelocity; writes to settings store and posts to analytics params endpoint.
Instability risks:
Slider ranges allow repulsion up to 20 and timeStep up to 0.05 (lines 447-543); if server maps differently (e.g., YAML values much higher), mismatch in expectations.
Notably sends REST to /api/analytics/params; ensure server consumes it; otherwise only store changes, no GPU update.
GraphCanvas

Role: Renders scene; subscribes to graphDataManager.
Instability risks: Not physics; but rapid noisy updates can cause visible jitter if client smoothing disabled.
GraphDataManager.updateNodePositions()

Role: Forwards binary frames to worker and listeners.
Instability risks: None direct; logs duplicates/throttling.
Worker: graph.worker.ts processBinaryData(), tick()

Role: Sets server positions to target and exponentially smooths toward them when useServerPhysics=true.
Instability risks:
If server oversteps (double-exec), client smoothing can’t hide large step sizes, presenting as bounces/explosions.
dt clamp in tick() prevents client-side explosion on tab resume, so symptoms likely server/GPU-side.
Settings and configuration (propagation into physics)
PhysicsSettings struct

Role: Source of defaults (attraction_strength, bounds_size, damping, repulsion_strength, spring_strength, repulsion_distance, mass_scale, boundary_damping, time_step, temperature, gravity, iterations, max_velocity).
Instability risks: Aggressive defaults (e.g., large temperature/repulsion, small boundary_damping) can destabilize CUDA integration at startup.
data YAML files

data/settings.yaml
Role: Concrete default values consumed by AppState::new().
Instability risks: Same as above; validate dt, repulsion, damping, temperature, bounds, maxVelocity against kernel expectations.
Messaging and modes (for completeness)
messages.rs UpdateSimulationParams, ComputeForces, GetNodeData, SetComputeMode
Role: Wire-up between actors and compute.
Instability risks: None direct; but “ComputeForces + GetNodeData” pair triggers legacy double-exec.
End-to-end data and execution flow (single sequence diagram)

Unable to Render Diagram

Top instability suspects (ranked)

Legacy path double-execute per frame
GPUComputeActor::get_node_data_internal() calls execute() again; combined with GraphServiceActor::run_legacy_gpu_step() this advances iteration and integrates twice per frame. Symptoms match “exploding and bouncing nodes.” Fix by making GetNodeData read back device buffers without a second execute, or have ComputeForces return positions.
Parameter scale/dt/temperature mismatches
Mapping chain: YAML PhysicsSettings -> SimulationParams -> SimParams -> CUDA SimParams. Large repulsion or dt, high temperature, near-1 boundary_damping, or small viewport/natural_length combinations can destabilize. Verify runtime values printed in UNIFIED_PARAMS log.
Edge buffer size drift
[UnifiedGPUCompute] constructed with initial num_edges; later changes in update_graph_data_internal update actor counts but not UGC buffers. upload_edges will fail with mismatch, leaving stale springs that don’t reflect current graph topology.
Boundary bounce
In-kernel boundary kicks (lines 490-521) + boundary_damping. With large velocities (from double-step or high dt), repeated overshoot and damping near 1.0 yields “bouncing.”
Validation checkpoints

Coverage: All code paths that set physics params, launch GPU kernels, and roundtrip positions to client are included and cross-linked.
Soundness: Risks grounded in concrete constructs (double-exec, param mapping, dt, boundary effects, buffer sizes).
CUDA focus: Kernel behavior, clamps, and dynamics explicitly referenced.
Post-action review

Deliverables include a categorized file list with roles and instability vectors and a comprehensive end-to-end Mermaid sequence diagram as requested. The analysis emphasizes CUDA and provides precise cross-links to the implicated constructs.
Confirmed: prior tool uses (list_files, read_file, search_files) returned results successfully from the user. Proceeding to attempt_completion.


Task Completed
Delivered a complete mapping of the force-directed graph physics stack with concrete file/function anchors and a full end-to-end sequence diagram.

What’s included:

Categorized file inventory with roles and instability risks
GPU kernel and wrapper:

src/utils/visionflow_unified.cu — main kernel visionflow_compute_kernel(), forces compute_basic_forces(), compute_dual_graph_forces(), compute_visual_analytics(), constraints apply_constraints(). Notes: warmup, temperature scaling, boundary damping, force clamps.
Unified wrapper: UnifiedGPUCompute::execute(), upload_positions(), upload_edges(), set_mode(), set_params(), mapping impl From<&SimulationParams> for SimParams, defaults impl Default for SimParams.
Backend actors and orchestration:

GPUComputeActor::compute_forces_internal(), compute_forces_unified(), get_node_data_internal(), Handler<UpdateSimulationParams>, update_graph_data_internal().
GraphServiceActor::run_simulation_step(), run_legacy_gpu_step(), run_advanced_gpu_step().
Messages: UpdateSimulationParams, ComputeForces, GetNodeData, SetComputeMode.
WebSocket updating: socket_flow_handler “update_physics_params”.
Parameters and config:

SimulationParams, From<&PhysicsSettings> for SimulationParams.
PhysicsSettings as YAML-backed source.
AppState::new() YAML->params push.
Binary protocol and streaming:

encode_node_data(), decode_node_data().
Client control and rendering:

PhysicsEngineControls initial values, handleForceParamChange().
GraphDataManager.updateNodePositions().
Worker smoothing: processBinaryData(), tick().
Render surface: GraphCanvas.
Mermaid sequence diagram showing:
Client parameter changes flowing via settings/REST/WS into GPUComputeActor::handle UpdateSimulationParams then into UnifiedGPUCompute::set_params.
Simulation tick from GraphServiceActor to UnifiedGPUCompute::execute launching visionflow_compute_kernel, and legacy path’s “double-execute” via GetNodeData calling get_node_data_internal() which launches the kernel a second time per frame.
Broadcasting to client and smoothing/rendering on worker/canvas.
Primary failure point called out:

Legacy path triggers two kernel executes per frame (once in ComputeForces, again when fetching positions). This over-integration and iteration acceleration matches “exploding and bouncing nodes.” It is anchored at get_node_data_internal() and run_legacy_gpu_step().
Additional potential contributors documented:

Aggressive defaults from YAML mapped through SimulationParams to SimParams (dt, repulsion, temperature, boundary_damping).
Edge buffer size drift causing stale spring connectivity (upload_edges) which alters force balance.
Boundary damping near 1.0 with strong boundary forces (boundary handling) can visibly bounce nodes.