// Example: GPU Semantic Forces for Ontology-Based Layout
//
// This example demonstrates how to use the GPU semantic force kernels
// to create ontology-aware knowledge graph layouts.

use anyhow::Result;
use std::collections::HashMap;

// Import the unified GPU compute module
use visionflow::utils::unified_gpu_compute::UnifiedGPUCompute;
use visionflow::models::{
    constraints::{Constraint, ConstraintData, ConstraintKind},
    simulation_params::SimulationParams,
};

/// Example ontology structure
#[derive(Debug, Clone)]
struct OntologyClass {
    id: u32,
    name: String,
    parent_id: Option<u32>,
    children: Vec<u32>,
}

/// Example: Generate semantic constraints from ontology
fn generate_semantic_constraints(
    ontology: &HashMap<u32, OntologyClass>,
    node_to_class: &HashMap<u32, u32>,
) -> Vec<Constraint> {
    let mut constraints = Vec::new();

    // 1. SEPARATION CONSTRAINTS: Disjoint classes should be far apart
    for (class_a_id, class_a) in ontology.iter() {
        for (class_b_id, class_b) in ontology.iter() {
            if class_a_id >= class_b_id {
                continue;
            }

            // Check if classes are disjoint (no common ancestor)
            if are_classes_disjoint(class_a, class_b, ontology) {
                // Find all nodes in each class
                let nodes_a: Vec<u32> = node_to_class
                    .iter()
                    .filter(|(_, cid)| **cid == *class_a_id)
                    .map(|(nid, _)| *nid)
                    .collect();

                let nodes_b: Vec<u32> = node_to_class
                    .iter()
                    .filter(|(_, cid)| **cid == *class_b_id)
                    .map(|(nid, _)| *nid)
                    .collect();

                // Create pairwise separation constraints
                for &node_a in &nodes_a {
                    for &node_b in &nodes_b {
                        let constraint = Constraint {
                            kind: ConstraintKind::SEMANTIC,
                            node_indices: vec![node_a, node_b],
                            params: vec![
                                0.5,   // separation_strength (params[0])
                                0.0,   // attraction_strength (params[1]) - not used for separation
                                0.0,   // alignment_axis (params[2]) - not used
                                200.0, // min_separation_distance (params[3])
                                0.0,   // alignment_strength (params[4]) - not used
                            ],
                            weight: 0.8, // High priority
                            active: true,
                        };
                        constraints.push(constraint);
                    }
                }
            }
        }
    }

    // 2. HIERARCHICAL ATTRACTION: Child classes attracted to parent
    for (class_id, class) in ontology.iter() {
        if let Some(parent_id) = class.parent_id {
            // Find parent nodes
            let parent_nodes: Vec<u32> = node_to_class
                .iter()
                .filter(|(_, cid)| **cid == parent_id)
                .map(|(nid, _)| *nid)
                .collect();

            // Find child nodes
            let child_nodes: Vec<u32> = node_to_class
                .iter()
                .filter(|(_, cid)| **cid == *class_id)
                .map(|(nid, _)| *nid)
                .collect();

            // Create hierarchical constraints
            for &parent_node in &parent_nodes {
                for &child_node in &child_nodes {
                    let constraint = Constraint {
                        kind: ConstraintKind::SEMANTIC,
                        node_indices: vec![parent_node, child_node], // parent first
                        params: vec![
                            0.0,  // separation_strength (params[0]) - not used
                            0.3,  // attraction_strength (params[1])
                            0.0,  // alignment_axis (params[2]) - not used
                            0.0,  // min_separation_distance (params[3]) - not used
                            0.0,  // alignment_strength (params[4]) - not used
                        ],
                        weight: 0.6, // Medium priority
                        active: true,
                    };
                    constraints.push(constraint);
                }
            }
        }
    }

    // 3. ALIGNMENT CONSTRAINTS: Sibling classes align horizontally
    for (class_id, class) in ontology.iter() {
        if let Some(parent_id) = class.parent_id {
            // Find all sibling classes
            let sibling_classes: Vec<u32> = ontology
                .values()
                .filter(|c| c.parent_id == Some(parent_id) && c.id != *class_id)
                .map(|c| c.id)
                .collect();

            if !sibling_classes.is_empty() {
                // Get nodes in this class
                let my_nodes: Vec<u32> = node_to_class
                    .iter()
                    .filter(|(_, cid)| **cid == *class_id)
                    .map(|(nid, _)| *nid)
                    .collect();

                // Get nodes in sibling classes
                let sibling_nodes: Vec<u32> = node_to_class
                    .iter()
                    .filter(|(_, cid)| sibling_classes.contains(cid))
                    .map(|(nid, _)| *nid)
                    .collect();

                // Create alignment constraint
                let mut all_nodes = my_nodes.clone();
                all_nodes.extend(sibling_nodes);

                if all_nodes.len() >= 2 {
                    let constraint = Constraint {
                        kind: ConstraintKind::SEMANTIC,
                        node_indices: all_nodes,
                        params: vec![
                            0.0, // separation_strength (params[0]) - not used
                            0.0, // attraction_strength (params[1]) - not used
                            1.0, // alignment_axis (params[2]) - Y axis
                            0.0, // min_separation_distance (params[3]) - not used
                            0.4, // alignment_strength (params[4])
                        ],
                        weight: 0.5, // Medium-low priority
                        active: true,
                    };
                    constraints.push(constraint);
                }
            }
        }
    }

    constraints
}

/// Check if two ontology classes are disjoint (no common ancestor)
fn are_classes_disjoint(
    class_a: &OntologyClass,
    class_b: &OntologyClass,
    ontology: &HashMap<u32, OntologyClass>,
) -> bool {
    // Get all ancestors of class A
    let mut ancestors_a = std::collections::HashSet::new();
    let mut current_id = Some(class_a.id);
    while let Some(id) = current_id {
        ancestors_a.insert(id);
        current_id = ontology.get(&id).and_then(|c| c.parent_id);
    }

    // Check if any ancestor of B is in ancestors of A
    let mut current_id = Some(class_b.id);
    while let Some(id) = current_id {
        if ancestors_a.contains(&id) {
            return false; // Common ancestor found
        }
        current_id = ontology.get(&id).and_then(|c| c.parent_id);
    }

    true // No common ancestor
}

/// Main example
fn main() -> Result<()> {
    println!("GPU Semantic Forces Example\n");

    // 1. Create example ontology
    let mut ontology = HashMap::new();

    // Root class
    ontology.insert(
        0,
        OntologyClass {
            id: 0,
            name: "Thing".to_string(),
            parent_id: None,
            children: vec![1, 2],
        },
    );

    // Science branch
    ontology.insert(
        1,
        OntologyClass {
            id: 1,
            name: "Science".to_string(),
            parent_id: Some(0),
            children: vec![3, 4],
        },
    );

    // Arts branch
    ontology.insert(
        2,
        OntologyClass {
            id: 2,
            name: "Arts".to_string(),
            parent_id: Some(0),
            children: vec![5, 6],
        },
    );

    // Science children
    ontology.insert(
        3,
        OntologyClass {
            id: 3,
            name: "Physics".to_string(),
            parent_id: Some(1),
            children: vec![],
        },
    );

    ontology.insert(
        4,
        OntologyClass {
            id: 4,
            name: "Biology".to_string(),
            parent_id: Some(1),
            children: vec![],
        },
    );

    // Arts children
    ontology.insert(
        5,
        OntologyClass {
            id: 5,
            name: "Music".to_string(),
            parent_id: Some(2),
            children: vec![],
        },
    );

    ontology.insert(
        6,
        OntologyClass {
            id: 6,
            name: "Painting".to_string(),
            parent_id: Some(2),
            children: vec![],
        },
    );

    // 2. Create node-to-class mapping (example nodes)
    let mut node_to_class = HashMap::new();
    node_to_class.insert(0, 0); // Thing
    node_to_class.insert(1, 1); // Science
    node_to_class.insert(2, 2); // Arts
    node_to_class.insert(3, 3); // Physics (node 3 -> class 3)
    node_to_class.insert(4, 3); // Physics (node 4 -> class 3)
    node_to_class.insert(5, 4); // Biology (node 5 -> class 4)
    node_to_class.insert(6, 4); // Biology (node 6 -> class 4)
    node_to_class.insert(7, 5); // Music (node 7 -> class 5)
    node_to_class.insert(8, 6); // Painting (node 8 -> class 6)

    let num_nodes = 9;

    println!("Ontology structure:");
    println!("  Thing");
    println!("  ├─ Science");
    println!("  │  ├─ Physics (nodes 3, 4)");
    println!("  │  └─ Biology (nodes 5, 6)");
    println!("  └─ Arts");
    println!("     ├─ Music (node 7)");
    println!("     └─ Painting (node 8)");
    println!();

    // 3. Generate semantic constraints
    let constraints = generate_semantic_constraints(&ontology, &node_to_class);
    println!("Generated {} semantic constraints", constraints.len());

    // 4. Convert to GPU format
    let gpu_constraints: Vec<ConstraintData> = constraints
        .iter()
        .map(|c| ConstraintData::from_constraint(c))
        .collect();

    println!("  Separation constraints (disjoint classes): {}",
        constraints.iter().filter(|c| c.params[0] > 0.0).count());
    println!("  Hierarchical constraints (parent-child): {}",
        constraints.iter().filter(|c| c.params[1] > 0.0).count());
    println!("  Alignment constraints (siblings): {}",
        constraints.iter().filter(|c| c.params[4] > 0.0).count());
    println!();

    // 5. Initialize GPU compute
    println!("Initializing GPU compute...");
    let mut gpu_compute = UnifiedGPUCompute::new(num_nodes)?;

    // 6. Upload constraints to GPU
    println!("Uploading {} constraints to GPU", gpu_constraints.len());
    gpu_compute.upload_constraints(&gpu_constraints)?;

    // 7. Upload class indices
    let class_indices: Vec<i32> = (0..num_nodes)
        .map(|i| *node_to_class.get(&i).unwrap_or(&0) as i32)
        .collect();

    println!("Uploading class indices");
    gpu_compute.update_class_indices(&class_indices)?;

    // 8. Configure simulation parameters
    let mut params = SimulationParams::default();
    params.dt = 0.016; // 60 FPS
    params.spring_k = 0.01;
    params.repel_k = 1000.0;
    params.damping = 0.9;
    params.constraint_force_weight = 0.8;
    params.constraint_ramp_frames = 60; // 1 second ramp at 60 FPS

    println!("Running simulation...");
    println!("  Constraint ramp: {} frames", params.constraint_ramp_frames);
    println!();

    // 9. Run physics simulation
    for iteration in 0..600 {
        params.iteration = iteration;

        gpu_compute.execute_physics_step(&params)?;

        if iteration % 100 == 0 {
            // Get current positions
            let positions = gpu_compute.get_node_positions()?;

            // Calculate kinetic energy
            let velocities = gpu_compute.get_node_velocities()?;
            let ke: f32 = velocities.iter()
                .map(|(vx, vy, vz)| vx * vx + vy * vy + vz * vz)
                .sum::<f32>() * 0.5;

            println!("Iteration {}: KE = {:.2}", iteration, ke);
        }
    }

    println!("\nSimulation complete!");
    println!("Expected behavior:");
    println!("  ✓ Science nodes (3-6) clustered together");
    println!("  ✓ Arts nodes (7-8) clustered together");
    println!("  ✓ Science and Arts clusters separated");
    println!("  ✓ Physics nodes (3, 4) aligned horizontally");
    println!("  ✓ Biology nodes (5, 6) aligned horizontally");

    Ok(())
}
