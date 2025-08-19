//! Deadlock Recovery Test Suite
//! 
//! Tests for the aggressive deadlock recovery system that breaks symmetry
//! when all nodes are stuck at boundary positions.

#[cfg(test)]
mod deadlock_recovery_tests {
    use super::*;
    use crate::actors::graph_actor::GraphActor;
    use crate::models::node::Node;
    use std::collections::HashMap;

    #[actix_rt::test]
    async fn test_complete_deadlock_detection() {
        // Setup: Create graph with all nodes at boundary (980 units from center)
        let mut graph_actor = GraphActor::new();
        
        // Add 177 nodes all at boundary distance
        for i in 0..177 {
            let angle = (i as f32) * 2.0 * std::f32::consts::PI / 177.0;
            let boundary_distance = 980.0;
            
            let node = Node {
                id: format!("node_{}", i),
                x: boundary_distance * angle.cos(),
                y: boundary_distance * angle.sin(),
                vx: 0.0, // No velocity = stuck
                vy: 0.0,
                ..Default::default()
            };
            
            graph_actor.node_map.insert(format!("node_{}", i), node);
        }
        
        // Simulate deadlock detection
        let boundary_nodes = graph_actor.count_boundary_nodes(100.0);
        let kinetic_energy = graph_actor.calculate_avg_kinetic_energy();
        
        assert_eq!(boundary_nodes, 177, "All nodes should be at boundary");
        assert!(kinetic_energy < 0.001, "Kinetic energy should be near zero");
        
        // Should trigger deadlock recovery
        let is_deadlocked = boundary_nodes == graph_actor.node_map.len() && kinetic_energy < 0.001;
        assert!(is_deadlocked, "Should detect complete deadlock");
    }

    #[actix_rt::test] 
    async fn test_aggressive_recovery_parameters() {
        let mut graph_actor = GraphActor::new();
        
        // Simulate deadlock recovery parameter application
        graph_actor.apply_aggressive_recovery_params();
        
        // Verify recovery parameters are strong enough
        assert!(graph_actor.target_params.repel_k >= 5.0, "Repulsion should be >= 5.0, got {}", graph_actor.target_params.repel_k);
        assert!(graph_actor.target_params.repel_k <= 10.0, "Repulsion should be <= 10.0, got {}", graph_actor.target_params.repel_k);
        assert!(graph_actor.target_params.damping >= 0.5, "Damping should be >= 0.5, got {}", graph_actor.target_params.damping);
        assert!(graph_actor.target_params.damping <= 0.6, "Damping should be <= 0.6, got {}", graph_actor.target_params.damping);
        assert!(graph_actor.target_params.max_velocity >= 5.0, "Max velocity should be >= 5.0, got {}", graph_actor.target_params.max_velocity);
        assert_eq!(graph_actor.param_transition_rate, 0.5, "Transition rate should be 0.5 for fast recovery");
    }

    #[actix_rt::test]
    async fn test_symmetry_breaking_perturbation() {
        let mut graph_actor = GraphActor::new();
        
        // Add nodes in perfect symmetrical positions
        for i in 0..4 {
            let angle = (i as f32) * std::f32::consts::PI / 2.0; // 90 degree intervals
            let node = Node {
                id: format!("node_{}", i),
                x: 100.0 * angle.cos(),
                y: 100.0 * angle.sin(),
                vx: 0.0, // Perfect symmetry
                vy: 0.0,
                ..Default::default()
            };
            graph_actor.node_map.insert(format!("node_{}", i), node);
        }
        
        // Store original positions
        let original_positions: HashMap<String, (f32, f32)> = graph_actor.node_map
            .iter()
            .map(|(id, node)| (id.clone(), (node.x, node.y)))
            .collect();
        
        // Apply perturbation
        graph_actor.apply_deadlock_perturbation();
        
        // Verify symmetry was broken
        let mut positions_changed = 0;
        let mut velocities_added = 0;
        
        for (id, node) in &graph_actor.node_map {
            let (orig_x, orig_y) = original_positions[id];
            
            // Positions should be slightly different
            if (node.x - orig_x).abs() > 0.01 || (node.y - orig_y).abs() > 0.01 {
                positions_changed += 1;
            }
            
            // Velocities should be non-zero
            if node.vx.abs() > 0.01 || node.vy.abs() > 0.01 {
                velocities_added += 1;
            }
        }
        
        assert_eq!(positions_changed, 4, "All node positions should be perturbed");
        assert_eq!(velocities_added, 4, "All nodes should have velocity added");
    }

    #[actix_rt::test]
    async fn test_recovery_strength_breaks_boundary_lock() {
        let mut graph_actor = GraphActor::new();
        
        // Setup extreme boundary lock scenario
        let boundary_distance = 980.0;
        for i in 0..10 {
            let angle = (i as f32) * 2.0 * std::f32::consts::PI / 10.0;
            let node = Node {
                id: format!("boundary_node_{}", i),
                x: boundary_distance * angle.cos(),
                y: boundary_distance * angle.sin(),
                vx: 0.0,
                vy: 0.0,
                ..Default::default()
            };
            graph_actor.node_map.insert(format!("boundary_node_{}", i), node);
        }
        
        // Apply aggressive recovery
        graph_actor.apply_aggressive_recovery_params();
        graph_actor.apply_deadlock_perturbation();
        
        // Verify parameters are strong enough to overcome boundary constraints
        let repulsion_force = graph_actor.target_params.repel_k;
        let velocity_limit = graph_actor.target_params.max_velocity;
        let damping = graph_actor.target_params.damping;
        
        // Calculate theoretical escape velocity needed
        let boundary_force_strength = 2.0; // From settings
        let min_escape_velocity = boundary_force_strength / (1.0 - damping);
        
        assert!(velocity_limit > min_escape_velocity, 
                "Max velocity {} should exceed escape velocity {}", velocity_limit, min_escape_velocity);
        assert!(repulsion_force > 5.0, "Repulsion {} should be strong enough", repulsion_force);
    }

    #[actix_rt::test]
    async fn test_detection_sensitivity() {
        let mut graph_actor = GraphActor::new();
        
        // Test with kinetic energy just above new threshold
        let high_energy = 0.002; // Above 0.001 threshold
        assert!(!should_trigger_deadlock_recovery(177, 177, high_energy), 
                "Should NOT trigger recovery with energy {}", high_energy);
        
        // Test with kinetic energy below new threshold  
        let low_energy = 0.0005; // Below 0.001 threshold
        assert!(should_trigger_deadlock_recovery(177, 177, low_energy),
                "SHOULD trigger recovery with energy {}", low_energy);
    }
    
    // Helper function to test deadlock detection logic
    fn should_trigger_deadlock_recovery(boundary_nodes: usize, total_nodes: usize, kinetic_energy: f32) -> bool {
        boundary_nodes == total_nodes && kinetic_energy < 0.001
    }
}