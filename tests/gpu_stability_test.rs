#[cfg(test)]
mod gpu_stability_tests {
    use std::fs;
    use unified_compute::models::simulation_params::SimParams;
    use unified_compute::utils::unified_gpu_compute::UnifiedGPUCompute;

    #[test]
    #[ignore] // Only run when GPU is available
    fn test_stability_gate_activation() {
        // Load PTX content
        let ptx_path = concat!(env!("OUT_DIR"), "/visionflow_unified.ptx");
        let ptx_content = fs::read_to_string(ptx_path).expect("Failed to read PTX file");

        // Create GPU compute with simple graph
        let num_nodes = 100;
        let num_edges = 200;
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes, num_edges, &ptx_content)
            .expect("Failed to create GPU compute");

        // Initialize with random positions
        let positions: Vec<(f32, f32, f32)> = (0..num_nodes)
            .map(|i| {
                let angle = (i as f32) * 2.0 * std::f32::consts::PI / (num_nodes as f32);
                (angle.cos() * 100.0, angle.sin() * 100.0, 0.0)
            })
            .collect();

        gpu_compute
            .upload_node_data(&positions, None)
            .expect("Failed to upload node data");

        // Create simple ring topology
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for i in 0..num_nodes {
            let next = (i + 1) % num_nodes;
            edges.push((i, next));
            weights.push(1.0);
        }

        gpu_compute
            .upload_edge_data(&edges, &weights)
            .expect("Failed to upload edge data");

        // Run simulation with stability gates enabled
        let mut params = SimParams::new();
        params.stability_threshold = 1e-5; // More aggressive threshold for testing
        params.min_velocity_threshold = 1e-3;
        params.damping = 0.95; // High damping to reach stability faster

        // Run for multiple iterations and track when stability is reached
        let mut stability_reached = false;
        let max_iterations = 1000;

        for i in 0..max_iterations {
            params.iteration = i;
            gpu_compute
                .execute(params)
                .expect("Failed to execute physics step");

            // Check system kinetic energy
            let mut system_ke = vec![0.0f32; 1];
            gpu_compute
                .system_kinetic_energy
                .copy_to(&mut system_ke)
                .expect("Failed to copy kinetic energy");

            if system_ke[0] < params.stability_threshold * (num_nodes as f32) {
                println!(
                    "Stability reached at iteration {}: KE = {:.8}",
                    i, system_ke[0]
                );
                stability_reached = true;
                break;
            }

            if i % 100 == 0 {
                println!("Iteration {}: KE = {:.8}", i, system_ke[0]);
            }
        }

        assert!(
            stability_reached,
            "System should reach stability within {} iterations",
            max_iterations
        );
    }

    #[test]
    #[ignore] // Only run when GPU is available
    fn test_per_node_stability_optimization() {
        // Load PTX content
        let ptx_path = concat!(env!("OUT_DIR"), "/visionflow_unified.ptx");
        let ptx_content = fs::read_to_string(ptx_path).expect("Failed to read PTX file");

        // Create GPU compute with nodes that have different velocities
        let num_nodes = 1000;
        let num_edges = 0; // No edges for this test
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes, num_edges, &ptx_content)
            .expect("Failed to create GPU compute");

        // Initialize positions
        let positions: Vec<(f32, f32, f32)> = (0..num_nodes)
            .map(|i| ((i as f32) * 0.1, 0.0, 0.0))
            .collect();

        // Initialize velocities - half nodes moving, half stationary
        let velocities: Vec<(f32, f32, f32)> = (0..num_nodes)
            .map(|i| {
                if i < num_nodes / 2 {
                    (0.0, 0.0, 0.0) // Stationary
                } else {
                    (1.0, 0.0, 0.0) // Moving
                }
            })
            .collect();

        gpu_compute
            .upload_node_data(&positions, Some(&velocities))
            .expect("Failed to upload node data");

        // Run physics with per-node stability checking
        let mut params = SimParams::new();
        params.min_velocity_threshold = 0.1; // Threshold between stationary and moving
        params.stability_threshold = 1e-6;

        gpu_compute
            .execute(params)
            .expect("Failed to execute physics step");

        // Check active node count
        let mut active_count = vec![0i32; 1];
        gpu_compute
            .active_node_count
            .copy_to(&mut active_count)
            .expect("Failed to copy active node count");

        println!("Active nodes: {} out of {}", active_count[0], num_nodes);

        // Should detect approximately half the nodes as active
        assert!(
            active_count[0] > (num_nodes as i32 * 4 / 10),
            "Too few active nodes detected"
        );
        assert!(
            active_count[0] < (num_nodes as i32 * 6 / 10),
            "Too many active nodes detected"
        );
    }
}
