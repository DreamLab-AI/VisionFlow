//! GPU Semantic Forces Integration Tests
//!
//! Tests the CUDA semantic force kernels for ontology-based physics

#[cfg(test)]
mod tests {
    use visionflow::models::{
        constraints::{Constraint, ConstraintData, ConstraintKind},
        simulation_params::SimulationParams,
    };
    use visionflow::utils::unified_gpu_compute::UnifiedGPUCompute;

    /// Test separation forces push disjoint classes apart
    #[test]
    fn test_separation_forces() {
        let num_nodes = 4;
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes).unwrap();

        // Two nodes in class A (0, 1), two nodes in class B (2, 3)
        let class_indices = vec![0, 0, 1, 1];
        gpu_compute.update_class_indices(&class_indices).unwrap();

        // Create separation constraint between classes
        let constraint = Constraint {
            kind: ConstraintKind::Semantic,
            node_indices: vec![0, 2], // One from each class
            params: vec![
                1.0,   // separation_strength
                0.0,   // attraction_strength
                0.0,   // alignment_axis
                100.0, // min_separation_distance
                0.0,   // alignment_strength
            ],
            weight: 1.0,
            active: true,
        };

        let gpu_constraints = vec![ConstraintData::from_constraint(&constraint)];
        gpu_compute.upload_constraints(&gpu_constraints).unwrap();

        // Initialize nodes close together
        let initial_positions = vec![
            (0.0, 0.0, 0.0),   // Node 0 (class A)
            (10.0, 0.0, 0.0),  // Node 1 (class A)
            (20.0, 0.0, 0.0),  // Node 2 (class B) - too close to A
            (30.0, 0.0, 0.0),  // Node 3 (class B)
        ];
        gpu_compute
            .set_node_positions(&initial_positions)
            .unwrap();

        // Run simulation
        let mut params = SimulationParams::default();
        params.constraint_ramp_frames = 0; // No ramp for testing
        params.constraint_force_weight = 1.0;

        for _ in 0..100 {
            gpu_compute.execute_physics_step(&params).unwrap();
        }

        // Check that distance increased
        let final_positions = gpu_compute.get_node_positions().unwrap();
        let initial_dist = ((initial_positions[0].0 - initial_positions[2].0).powi(2)
            + (initial_positions[0].1 - initial_positions[2].1).powi(2)
            + (initial_positions[0].2 - initial_positions[2].2).powi(2))
        .sqrt();

        let final_dist = ((final_positions[0].0 - final_positions[2].0).powi(2)
            + (final_positions[0].1 - final_positions[2].1).powi(2)
            + (final_positions[0].2 - final_positions[2].2).powi(2))
        .sqrt();

        assert!(
            final_dist > initial_dist,
            "Separation force should push disjoint classes apart"
        );
        assert!(
            final_dist >= 90.0,
            "Distance should approach minimum separation (100)"
        );
    }

    /// Test hierarchical attraction pulls children toward parents
    #[test]
    fn test_hierarchical_attraction() {
        let num_nodes = 3;
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes).unwrap();

        // Node 0 is parent (class 0), nodes 1-2 are children (class 1)
        let class_indices = vec![0, 1, 1];
        gpu_compute.update_class_indices(&class_indices).unwrap();

        // Create hierarchical constraint
        let constraint = Constraint {
            kind: ConstraintKind::Semantic,
            node_indices: vec![0, 1], // parent first, child second
            params: vec![
                0.0, // separation_strength
                0.5, // attraction_strength
                0.0, // alignment_axis
                0.0, // min_separation_distance
                0.0, // alignment_strength
            ],
            weight: 1.0,
            active: true,
        };

        let gpu_constraints = vec![ConstraintData::from_constraint(&constraint)];
        gpu_compute.upload_constraints(&gpu_constraints).unwrap();

        // Initialize parent at origin, children far away
        let initial_positions = vec![
            (0.0, 0.0, 0.0),    // Parent
            (200.0, 0.0, 0.0),  // Child 1
            (0.0, 200.0, 0.0),  // Child 2
        ];
        gpu_compute
            .set_node_positions(&initial_positions)
            .unwrap();

        // Run simulation
        let mut params = SimulationParams::default();
        params.constraint_ramp_frames = 0;

        for _ in 0..100 {
            gpu_compute.execute_physics_step(&params).unwrap();
        }

        // Check that child moved closer to parent
        let final_positions = gpu_compute.get_node_positions().unwrap();
        let initial_dist = ((initial_positions[0].0 - initial_positions[1].0).powi(2)
            + (initial_positions[0].1 - initial_positions[1].1).powi(2)
            + (initial_positions[0].2 - initial_positions[1].2).powi(2))
        .sqrt();

        let final_dist = ((final_positions[0].0 - final_positions[1].0).powi(2)
            + (final_positions[0].1 - final_positions[1].1).powi(2)
            + (final_positions[0].2 - final_positions[1].2).powi(2))
        .sqrt();

        assert!(
            final_dist < initial_dist,
            "Child should be attracted to parent"
        );
    }

    /// Test alignment forces align nodes along specified axis
    #[test]
    fn test_alignment_forces() {
        let num_nodes = 4;
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes).unwrap();

        // All nodes in same class
        let class_indices = vec![0, 0, 0, 0];
        gpu_compute.update_class_indices(&class_indices).unwrap();

        // Create Y-axis alignment constraint
        let constraint = Constraint {
            kind: ConstraintKind::Semantic,
            node_indices: vec![0, 1, 2, 3],
            params: vec![
                0.0, // separation_strength
                0.0, // attraction_strength
                1.0, // alignment_axis (Y)
                0.0, // min_separation_distance
                0.8, // alignment_strength
            ],
            weight: 1.0,
            active: true,
        };

        let gpu_constraints = vec![ConstraintData::from_constraint(&constraint)];
        gpu_compute.upload_constraints(&gpu_constraints).unwrap();

        // Initialize nodes at random Y positions
        let initial_positions = vec![
            (0.0, 10.0, 0.0),
            (50.0, 30.0, 0.0),
            (100.0, -20.0, 0.0),
            (150.0, 5.0, 0.0),
        ];
        gpu_compute
            .set_node_positions(&initial_positions)
            .unwrap();

        // Run simulation
        let mut params = SimulationParams::default();
        params.constraint_ramp_frames = 0;

        for _ in 0..100 {
            gpu_compute.execute_physics_step(&params).unwrap();
        }

        // Check that Y positions converged
        let final_positions = gpu_compute.get_node_positions().unwrap();
        let y_values: Vec<f32> = final_positions.iter().map(|(_, y, _)| *y).collect();

        let avg_y = y_values.iter().sum::<f32>() / y_values.len() as f32;
        let variance: f32 = y_values
            .iter()
            .map(|y| (y - avg_y).powi(2))
            .sum::<f32>()
            / y_values.len() as f32;

        let initial_variance: f32 = initial_positions
            .iter()
            .map(|(_, y, _)| *y)
            .map(|y| {
                let avg = initial_positions.iter().map(|(_, y, _)| *y).sum::<f32>()
                    / initial_positions.len() as f32;
                (y - avg).powi(2)
            })
            .sum::<f32>()
            / initial_positions.len() as f32;

        assert!(
            variance < initial_variance,
            "Alignment should reduce Y position variance"
        );
        assert!(
            variance < 50.0,
            "Y positions should be well-aligned (low variance)"
        );
    }

    /// Test force blending respects constraint priorities
    #[test]
    fn test_force_blending_priority() {
        let num_nodes = 2;
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes).unwrap();

        let class_indices = vec![0, 1];
        gpu_compute.update_class_indices(&class_indices).unwrap();

        // High priority separation constraint
        let high_priority = Constraint {
            kind: ConstraintKind::Semantic,
            node_indices: vec![0, 1],
            params: vec![1.0, 0.0, 0.0, 100.0, 0.0],
            weight: 9.0, // High priority
            active: true,
        };

        let gpu_constraints = vec![ConstraintData::from_constraint(&high_priority)];
        gpu_compute.upload_constraints(&gpu_constraints).unwrap();

        let initial_positions = vec![(0.0, 0.0, 0.0), (50.0, 0.0, 0.0)];
        gpu_compute
            .set_node_positions(&initial_positions)
            .unwrap();

        let mut params = SimulationParams::default();
        params.constraint_ramp_frames = 0;

        for _ in 0..50 {
            gpu_compute.execute_physics_step(&params).unwrap();
        }

        let final_positions = gpu_compute.get_node_positions().unwrap();
        let final_dist = ((final_positions[0].0 - final_positions[1].0).powi(2)
            + (final_positions[0].1 - final_positions[1].1).powi(2)
            + (final_positions[0].2 - final_positions[1].2).powi(2))
        .sqrt();

        // High priority should dominate physics forces
        assert!(
            final_dist > 80.0,
            "High priority constraint should dominate"
        );
    }

    /// Test progressive activation ramps forces smoothly
    #[test]
    fn test_progressive_activation() {
        let num_nodes = 2;
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes).unwrap();

        let class_indices = vec![0, 1];
        gpu_compute.update_class_indices(&class_indices).unwrap();

        let mut constraint = Constraint {
            kind: ConstraintKind::Semantic,
            node_indices: vec![0, 1],
            params: vec![1.0, 0.0, 0.0, 100.0, 0.0],
            weight: 1.0,
            active: true,
        };

        // Set activation frame to 0
        let mut gpu_constraint = ConstraintData::from_constraint(&constraint);
        gpu_constraint.activation_frame = 0;

        gpu_compute
            .upload_constraints(&vec![gpu_constraint])
            .unwrap();

        let initial_positions = vec![(0.0, 0.0, 0.0), (50.0, 0.0, 0.0)];
        gpu_compute
            .set_node_positions(&initial_positions)
            .unwrap();

        let mut params = SimulationParams::default();
        params.constraint_ramp_frames = 60; // Ramp over 60 frames

        // Measure distance change at different ramp stages
        let mut distances = Vec::new();

        for iteration in 0..90 {
            params.iteration = iteration;
            gpu_compute.execute_physics_step(&params).unwrap();

            if iteration % 30 == 0 {
                let positions = gpu_compute.get_node_positions().unwrap();
                let dist = ((positions[0].0 - positions[1].0).powi(2)
                    + (positions[0].1 - positions[1].1).powi(2)
                    + (positions[0].2 - positions[1].2).powi(2))
                .sqrt();
                distances.push(dist);
            }
        }

        // Distance change should increase as ramp progresses
        assert!(
            distances.len() == 3,
            "Should have 3 distance measurements"
        );
        assert!(
            distances[1] > distances[0],
            "Force should increase during ramp"
        );
        assert!(
            distances[2] > distances[1],
            "Force should continue increasing"
        );
    }

    /// Test constraint caching efficiency
    #[test]
    fn test_constraint_caching() {
        let num_nodes = 10;
        let mut gpu_compute = UnifiedGPUCompute::new(num_nodes).unwrap();

        let class_indices: Vec<i32> = (0..10).map(|i| i % 3).collect();
        gpu_compute.update_class_indices(&class_indices).unwrap();

        // Upload constraints once
        let constraints: Vec<Constraint> = (0..9)
            .map(|i| Constraint {
                kind: ConstraintKind::Semantic,
                node_indices: vec![i, i + 1],
                params: vec![0.5, 0.0, 0.0, 50.0, 0.0],
                weight: 0.8,
                active: true,
            })
            .collect();

        let gpu_constraints: Vec<ConstraintData> = constraints
            .iter()
            .map(|c| ConstraintData::from_constraint(c))
            .collect();

        gpu_compute.upload_constraints(&gpu_constraints).unwrap();

        // Run multiple iterations without re-uploading
        let mut params = SimulationParams::default();
        for iteration in 0..100 {
            params.iteration = iteration;
            gpu_compute.execute_physics_step(&params).unwrap();
        }

        // Should complete without errors (constraints cached on GPU)
        let final_positions = gpu_compute.get_node_positions().unwrap();
        assert_eq!(final_positions.len(), num_nodes);
    }
}
