High-Certainty Diagnosis
The root cause of the problem is an architectural disconnect in the actor system. The real-time physics parameter updates from the control center are being sent to a dormant GPUComputeActor, while the main simulation loop, which is running inside the GraphServiceActor, is never notified of these changes.
Consequently, the GraphServiceActor continues to use the initial physics parameters it was loaded with at startup, completely ignoring any subsequent adjustments made by the user. The "exploding" behavior is a classic symptom of an unstable physics simulation running with excessively high repulsion forces and/or an overly large time step, which are likely the default values the active simulation is stuck with.
Detailed Analysis
Let's trace the flow of information and execution to understand how this disconnect occurs:
The Active Simulation Engine: The core physics simulation loop runs inside the GraphServiceActor (src/actors/graph_actor.rs).
Its run_simulation_step method is called on a 16ms interval.
This method calls run_advanced_gpu_step, which directly uses an embedded UnifiedGPUCompute instance (self.advanced_gpu_context) to execute the CUDA kernel.
Crucially, it configures the physics for this kernel using its own internal state: self.simulation_params.
The Incorrect Update Path: When you change a physics parameter in the control center, the frontend sends a request to the /api/analytics/params endpoint.
This request is handled by update_analytics_params in src/handlers/api_handler/analytics/mod.rs.
This handler correctly parses the incoming physics values (e.g., repulsion, damping).
It then creates an UpdateSimulationParams message and sends it... but only to the GPUComputeActor.
The Dormant Actor: The GPUComputeActor (src/actors/gpu_compute_actor.rs) is a separate actor designed to handle GPU computations.
It correctly receives the UpdateSimulationParams message and updates its internal physics parameters. You would even see log messages from this actor confirming the update.
However, the GraphServiceActor never calls the GPUComputeActor to perform the physics calculations. The GraphServiceActor performs them directly using its own UnifiedGPUCompute instance.
Therefore, the GPUComputeActor receives all the real-time updates but sits idle, while the GraphServiceActor runs the simulation completely unaware of these new settings.
Explaining the Symptoms:
No Impact on Graph: Since the active simulation in GraphServiceActor never gets the new parameters, your changes have no effect.
Exploding and Bouncing: The simulation is running with the initial parameters loaded from settings.yaml at startup. If these initial values are unstable or are being overridden by hardcoded defaults that are unstable, the physics simulation will break down. The massive forces push nodes outwards until they hit the boundary defined in the CUDA kernel (visionflow_unified.cu). The "bouncing" is caused by the boundary collision logic, which reverses a node's velocity when it hits the edge of the viewport_bounds, causing it to repeatedly slam against the walls of the simulation cube. The grouping in 8 corners is a typical artifact of this behavior in a 3D bounding box.
The Critical Code Files
src/handlers/api_handler/analytics/mod.rs: The update_analytics_params function sends the update message to the wrong destination.
code
Rust
// src/handlers/api_handler/analytics/mod.rs

pub async fn update_analytics_params(
    app_state: web::Data<AppState>,
    params: web::Json<serde_json::Value>,
) -> Result<HttpResponse> {
    // ... logic to parse params ...

    if let Some(gpu_addr) = app_state.gpu_compute_addr.as_ref() {
        // ...
        // PROBLEM: This message is sent ONLY to the GPUComputeActor.
        // The GraphServiceActor, which is running the simulation, is not notified.
        match gpu_addr.send(UpdateSimulationParams { params: sim_params }).await {
            // ...
        }
    }
    // ...
}
src/actors/graph_actor.rs: This actor runs the simulation loop using its own internal state, which is never updated by the API handler.
code
Rust
// src/actors/graph_actor.rs

fn run_advanced_gpu_step(&mut self, _ctx: &mut Context<Self>) {
    // ...
    if let Some(ref mut gpu_context) = self.advanced_gpu_context {
        // ...
        // It uses its OWN simulation_params, which are never updated after startup.
        let sim_params = crate::utils::unified_gpu_compute::SimParams::from(&self.simulation_params);
        gpu_context.set_params(sim_params);

        // It executes the physics directly, bypassing the GPUComputeActor.
        match gpu_context.execute() {
            // ...
        }
    }
    // ...
}
Solution
To fix this, you must ensure the UpdateSimulationParams message is sent to the correct actor, which is the GraphServiceActor.
Primary Fix: Retarget the Message
Modify the update_analytics_params handler in src/handlers/api_handler/analytics/mod.rs to send the message to app_state.graph_service_addr instead of app_state.gpu_compute_addr.
code
Rust
// In src/handlers/api_handler/analytics/mod.rs

// ... inside update_analytics_params function
if params.get("repulsion").is_some() || params.get("damping").is_some() {
    // ... (logic to build sim_params) ...

    // Get the correct actor address
    let graph_actor_addr = &app_state.graph_service_addr;

    // Send as UpdateSimulationParams to the GraphServiceActor
    use crate::actors::messages::UpdateSimulationParams;
    match graph_actor_addr.send(UpdateSimulationParams { params: sim_params }).await {
        Ok(Ok(())) => {
            info!("Physics parameters forwarded successfully to GraphServiceActor");
        }
        Ok(Err(e)) => {
            warn!("GraphServiceActor failed to update physics params: {}", e);
        }
        Err(e) => {
            warn!("GraphServiceActor mailbox error: {}", e);
        }
    }
}
// ...
Architectural Recommendation: Consolidate Physics Logic
The current architecture is confusing and error-prone because it contains two separate (and competing) physics engines. The GPUComputeActor is effectively redundant.
You should refactor the system to have a single source of truth for physics execution. The most straightforward approach is to remove the GPUComputeActor entirely and have all physics-related messages and logic handled directly by the GraphServiceActor and its embedded UnifiedGPUCompute instance. This will simplify the codebase and prevent this kind of bug from recurring.