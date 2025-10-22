//! Demonstration of the physics engine modules
//!
//! This example shows how to use the stress majorization solver and semantic
//! constraint generator together to optimize a knowledge graph layout.

use webxr::physics::{StressMajorizationSolver, SemanticConstraintGenerator};
use webxr::models::{
    constraints::{ConstraintSet, AdvancedParams},
    graph::GraphData,
    node::Node,
    edge::Edge,
    metadata::{Metadata, MetadataStore},
};
use webxr::utils::socket_flow_messages::BinaryNodeData;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    simplelog::TermLogger::init(
        log::LevelFilter::Info,
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    ).unwrap();

    println!("Physics Engine Demo - Knowledge Graph Layout Optimization");
    println!("=========================================================");

    // Create a sample knowledge graph
    let mut graph_data = create_sample_graph();
    let metadata_store = create_sample_metadata();

    println!("Created sample graph with {} nodes and {} edges",
             graph_data.nodes.len(), graph_data.edges.len());

    // Initialize advanced physics parameters
    let physics_params = AdvancedParams::semantic_optimized();
    println!("Using semantic-optimized physics parameters");

    // Create semantic constraint generator
    let mut constraint_generator = SemanticConstraintGenerator::from_advanced_params(&physics_params);

    // Generate semantic constraints
    println!("Generating semantic constraints...");
    let constraint_result = constraint_generator.generate_constraints(&graph_data, Some(&metadata_store))?;

    println!("Generated {} semantic constraints:",
             constraint_result.clustering_constraints.len() +
             constraint_result.separation_constraints.len() +
             constraint_result.alignment_constraints.len() +
             constraint_result.boundary_constraints.len());
    println!("  - {} clustering constraints", constraint_result.clustering_constraints.len());
    println!("  - {} separation constraints", constraint_result.separation_constraints.len());
    println!("  - {} alignment constraints", constraint_result.alignment_constraints.len());
    println!("  - {} boundary constraints", constraint_result.boundary_constraints.len());
    println!("  - {} semantic clusters identified", constraint_result.clusters.len());

    // Create constraint set and apply semantic constraints
    let mut constraint_set = ConstraintSet::default();
    constraint_generator.apply_to_constraint_set(&mut constraint_set, &constraint_result);

    // Create stress majorization solver
    let mut solver = StressMajorizationSolver::from_advanced_params(&physics_params);

    // Optimize layout
    println!("Optimizing graph layout using stress majorization...");
    let optimization_result = solver.optimize(&mut graph_data, &constraint_set)?;

    println!("Optimization completed:");
    println!("  - {} iterations performed", optimization_result.iterations);
    println!("  - Final stress: {:.6}", optimization_result.final_stress);
    println!("  - Converged: {}", optimization_result.converged);
    println!("  - Computation time: {}ms", optimization_result.computation_time);

    // Display constraint satisfaction scores
    println!("Constraint satisfaction scores:");
    for (constraint_type, score) in &optimization_result.constraint_scores {
        println!("  - {:?}: {:.3}", constraint_type, score);
    }

    // Display final node positions
    println!("\nFinal node positions:");
    for node in &graph_data.nodes {
        println!("  - {}: ({:.1}, {:.1}, {:.1})",
                 node.label, node.data.x, node.data.y, node.data.z);
    }

    // Display cluster information
    println!("\nSemantic clusters:");
    for (i, cluster) in constraint_result.clusters.iter().enumerate() {
        println!("  - Cluster {}: {} nodes, coherence: {:.3}",
                 i + 1, cluster.node_ids.len(), cluster.coherence);
        println!("    Topics: {:?}", cluster.primary_topics);
        println!("    Nodes: {:?}", cluster.node_ids.iter().collect::<Vec<_>>());
    }

    println!("\nDemo completed successfully!");
    Ok(())
}

fn create_sample_graph() -> GraphData {
    let nodes = vec![
        create_node(1, "Artificial Intelligence Overview", 0.0, 0.0, 0.0),
        create_node(2, "Machine Learning", 100.0, 0.0, 0.0),
        create_node(3, "Deep Learning", 200.0, 0.0, 0.0),
        create_node(4, "Neural Networks", 150.0, 100.0, 0.0),
        create_node(5, "Computer Vision", 50.0, 150.0, 0.0),
        create_node(6, "Natural Language Processing", 250.0, 150.0, 0.0),
        create_node(7, "Cooking Recipes", -200.0, -200.0, 0.0),
        create_node(8, "Travel Guide", -100.0, -200.0, 0.0),
    ];

    let edges = vec![
        Edge::new(1, 2, 1.0),  // AI -> ML
        Edge::new(2, 3, 1.0),  // ML -> DL
        Edge::new(3, 4, 1.0),  // DL -> NN
        Edge::new(2, 5, 1.0),  // ML -> CV
        Edge::new(2, 6, 1.0),  // ML -> NLP
        Edge::new(4, 5, 1.0),  // NN -> CV
        Edge::new(4, 6, 1.0),  // NN -> NLP
    ];

    GraphData { nodes, edges }
}

fn create_node(id: u32, label: &str, x: f32, y: f32, z: f32) -> Node {
    let mut node = Node::new_with_id(format!("{}.md", label.to_lowercase().replace(' ', "_")), Some(id));
    node.label = label.to_string();
    node.data.x = x;
    node.data.y = y;
    node.data.z = z;
    node
}

fn create_sample_metadata() -> MetadataStore {
    let mut store = MetadataStore::new();

    // AI-related topics
    let ai_topics = create_topics(&[
        ("artificial_intelligence", 20),
        ("technology", 10),
        ("computer_science", 15),
    ]);

    let ml_topics = create_topics(&[
        ("machine_learning", 25),
        ("artificial_intelligence", 15),
        ("algorithms", 12),
        ("statistics", 8),
    ]);

    let dl_topics = create_topics(&[
        ("deep_learning", 30),
        ("machine_learning", 20),
        ("neural_networks", 25),
        ("artificial_intelligence", 10),
    ]);

    let nn_topics = create_topics(&[
        ("neural_networks", 35),
        ("deep_learning", 25),
        ("machine_learning", 15),
        ("backpropagation", 10),
    ]);

    let cv_topics = create_topics(&[
        ("computer_vision", 30),
        ("machine_learning", 15),
        ("image_processing", 20),
        ("deep_learning", 18),
    ]);

    let nlp_topics = create_topics(&[
        ("natural_language_processing", 35),
        ("machine_learning", 15),
        ("linguistics", 12),
        ("deep_learning", 18),
    ]);

    // Unrelated topics
    let cooking_topics = create_topics(&[
        ("cooking", 40),
        ("recipes", 30),
        ("food", 25),
        ("kitchen", 15),
    ]);

    let travel_topics = create_topics(&[
        ("travel", 35),
        ("tourism", 25),
        ("geography", 20),
        ("culture", 15),
    ]);

    // Add metadata entries
    store.insert("artificial_intelligence_overview.md".to_string(), Metadata {
        file_name: "artificial_intelligence_overview.md".to_string(),
        file_size: 8000,
        topic_counts: ai_topics,
        ..Default::default()
    });

    store.insert("machine_learning.md".to_string(), Metadata {
        file_name: "machine_learning.md".to_string(),
        file_size: 12000,
        topic_counts: ml_topics,
        ..Default::default()
    });

    store.insert("deep_learning.md".to_string(), Metadata {
        file_name: "deep_learning.md".to_string(),
        file_size: 15000,
        topic_counts: dl_topics,
        ..Default::default()
    });

    store.insert("neural_networks.md".to_string(), Metadata {
        file_name: "neural_networks.md".to_string(),
        file_size: 10000,
        topic_counts: nn_topics,
        ..Default::default()
    });

    store.insert("computer_vision.md".to_string(), Metadata {
        file_name: "computer_vision.md".to_string(),
        file_size: 9000,
        topic_counts: cv_topics,
        ..Default::default()
    });

    store.insert("natural_language_processing.md".to_string(), Metadata {
        file_name: "natural_language_processing.md".to_string(),
        file_size: 11000,
        topic_counts: nlp_topics,
        ..Default::default()
    });

    store.insert("cooking_recipes.md".to_string(), Metadata {
        file_name: "cooking_recipes.md".to_string(),
        file_size: 5000,
        topic_counts: cooking_topics,
        ..Default::default()
    });

    store.insert("travel_guide.md".to_string(), Metadata {
        file_name: "travel_guide.md".to_string(),
        file_size: 6000,
        topic_counts: travel_topics,
        ..Default::default()
    });

    store
}

fn create_topics(topics: &[(&str, usize)]) -> HashMap<String, usize> {
    topics.iter()
        .map(|(name, count)| (name.to_string(), *count))
        .collect()
}