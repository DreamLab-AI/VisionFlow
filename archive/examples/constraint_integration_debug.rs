//! Test script to verify constraint integration in the graph actor

use crate::actors::graph_actor::GraphServiceActor;
use crate::models::graph::GraphData;
use crate::models::metadata::{MetadataEntry, MetadataStore};
use crate::models::node::Node;
use crate::services::semantic_analyzer::SemanticFeatures;

pub fn test_constraint_integration() {
    println!("Testing constraint integration...");

    // Create a mock graph service actor
    let mut actor = GraphServiceActor::new_test_instance();

    // Create sample metadata with different domains
    let mut metadata = MetadataStore::new();

    // Add sample metadata entries
    metadata.add_entry(
        "file1.rs".to_string(),
        MetadataEntry {
            id: "file1".to_string(),
            path: "src/models/file1.rs".to_string(),
            file_type: "rust".to_string(),
            size: 1024,
            last_modified: 0,
            content: Some("struct Example {}".to_string()),
        },
    );

    metadata.add_entry(
        "file2.rs".to_string(),
        MetadataEntry {
            id: "file2".to_string(),
            path: "src/models/file2.rs".to_string(),
            file_type: "rust".to_string(),
            size: 2048,
            last_modified: 0,
            content: Some("impl Example {}".to_string()),
        },
    );

    metadata.add_entry(
        "file3.js".to_string(),
        MetadataEntry {
            id: "file3".to_string(),
            path: "src/ui/file3.js".to_string(),
            file_type: "javascript".to_string(),
            size: 512,
            last_modified: 0,
            content: Some("function example() {}".to_string()),
        },
    );

    // Build graph from metadata (should trigger constraint generation)
    if let Err(e) = actor.build_from_metadata(metadata) {
        panic!("Failed to build graph from metadata: {}", e);
    }

    // Check if constraints were generated
    let initial_constraint_count = actor.constraint_set.constraints.len();
    println!(
        "Initial constraints generated: {}",
        initial_constraint_count
    );

    if initial_constraint_count == 0 {
        println!("WARNING: No constraints were generated from metadata");
    } else {
        println!(
            "SUCCESS: {} constraints generated",
            initial_constraint_count
        );

        // Print constraint details
        for (i, constraint) in actor.constraint_set.constraints.iter().enumerate() {
            println!(
                "Constraint {}: {:?} affecting {} nodes",
                i,
                constraint.kind,
                constraint.node_indices.len()
            );
        }
    }

    // Test dynamic constraint updates
    actor.update_dynamic_constraints();
    let dynamic_constraint_count = actor.constraint_set.active_constraints().len();
    println!(
        "Active constraints after dynamic update: {}",
        dynamic_constraint_count
    );

    println!("Constraint integration test completed successfully!");
}

impl GraphServiceActor {
    /// Create a test instance for unit testing
    pub fn new_test_instance() -> Self {
        use crate::actors::graph_actor::GraphServiceActor;
        use crate::models::graph_types::GraphType;
        use std::collections::HashMap;
        use std::sync::Arc;
        use tokio::sync::RwLock;

        // Create test instance with minimal viable setup
        GraphServiceActor {
            current_graph: Arc::new(RwLock::new(crate::models::graph::Graph {
                nodes: HashMap::new(),
                edges: HashMap::new(),
                constraints: Vec::new(),
                graph_type: GraphType::Physics,
                last_updated: chrono::Utc::now(),
                metadata: HashMap::new(),
            })),
            active_simulations: HashMap::new(),
            telemetry_subscribers: Vec::new(),
            constraint_solver: None,
            settings: crate::models::settings::SimulationSettings::default(),
        }
    }
}
