//! Integration tests for physics engine modules
//!
//! This module contains integration tests that verify the physics engine
//! components work together correctly for realistic graph scenarios.

#[cfg(test)]
mod tests {
    use crate::models::{
        constraints::{AdvancedParams, ConstraintSet},
        edge::Edge,
        graph::GraphData,
        metadata::{Metadata, MetadataStore},
        node::Node,
    };
    use crate::physics::{SemanticConstraintGenerator, StressMajorizationSolver};
    use crate::utils::socket_flow_messages::BinaryNodeData;
    use std::collections::HashMap;

    #[test]
    fn test_full_physics_pipeline() {
        let mut graph = create_ai_knowledge_graph();
        let metadata = create_ai_metadata();

        
        let mut constraint_generator = SemanticConstraintGenerator::new();
        let constraint_result = constraint_generator
            .generate_constraints(&graph, Some(&metadata))
            .expect("Failed to generate constraints");

        
        assert!(constraint_result.clusters.len() >= 1);
        assert!(constraint_result.clustering_constraints.len() > 0);

        
        let mut constraint_set = ConstraintSet::default();
        constraint_generator.apply_to_constraint_set(&mut constraint_set, &constraint_result);

        
        let mut solver = StressMajorizationSolver::new();
        let optimization_result = solver
            .optimize(&mut graph, &constraint_set)
            .expect("Failed to optimize layout");

        
        assert!(optimization_result.iterations > 0);
        assert!(optimization_result.final_stress.is_finite());
        assert!(optimization_result.computation_time > 0);

        
        for node in &graph.nodes {
            assert!(node.data.x.is_finite());
            assert!(node.data.y.is_finite());
            assert!(node.data.z.is_finite());
        }
    }

    #[test]
    fn test_semantic_clustering_accuracy() {
        let graph = create_mixed_domain_graph();
        let metadata = create_mixed_metadata();

        let mut generator = SemanticConstraintGenerator::new();
        let result = generator
            .generate_constraints(&graph, Some(&metadata))
            .expect("Failed to generate constraints");

        
        assert!(result.clusters.len() >= 2);

        
        for cluster in &result.clusters {
            assert!(cluster.coherence > 0.0);
            assert!(cluster.node_ids.len() >= 2);
        }

        
        assert!(result.separation_constraints.len() > 0);
    }

    #[test]
    fn test_hierarchical_constraint_generation() {
        let graph = create_hierarchical_graph();
        let metadata = create_hierarchical_metadata();

        let mut generator = SemanticConstraintGenerator::with_config(
            crate::physics::semantic_constraints::SemanticConstraintConfig {
                enable_hierarchy: true,
                ..Default::default()
            },
        );

        let result = generator
            .generate_constraints(&graph, Some(&metadata))
            .expect("Failed to generate constraints");

        
        assert!(result.hierarchical_relations.len() > 0);

        
        assert!(result.alignment_constraints.len() > 0);
    }

    #[test]
    fn test_constraint_satisfaction_scoring() {
        let mut graph = create_simple_graph();
        let constraint_set = create_test_constraints();

        let mut solver = StressMajorizationSolver::new();
        let result = solver
            .optimize(&mut graph, &constraint_set)
            .expect("Failed to optimize");

        
        assert!(!result.constraint_scores.is_empty());

        
        for (_, score) in &result.constraint_scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_performance_with_large_graph() {
        let graph = create_large_graph(1000); 
        let mut solver = StressMajorizationSolver::with_config(
            crate::physics::stress_majorization::StressMajorizationConfig {
                max_iterations: 100, 
                ..Default::default()
            },
        );

        let constraint_set = ConstraintSet::default();

        let start_time = std::time::Instant::now();
        let result = solver.optimize(&mut graph.clone(), &constraint_set);
        let elapsed = start_time.elapsed();

        assert!(result.is_ok());
        assert!(elapsed.as_secs() < 10); 

        let result = result.unwrap();
        assert!(result.final_stress.is_finite());
    }

    

    fn create_ai_knowledge_graph() -> GraphData {
        let nodes = vec![
            create_test_node(1, "AI Overview"),
            create_test_node(2, "Machine Learning"),
            create_test_node(3, "Deep Learning"),
            create_test_node(4, "Neural Networks"),
        ];

        let edges = vec![
            Edge::new(1, 2, 1.0),
            Edge::new(2, 3, 1.0),
            Edge::new(3, 4, 1.0),
        ];

        GraphData {
            nodes,
            edges,
            metadata: crate::models::metadata::MetadataStore::new(),
            id_to_metadata: std::collections::HashMap::new(),
        }
    }

    fn create_mixed_domain_graph() -> GraphData {
        let nodes = vec![
            create_test_node(1, "Machine Learning"),
            create_test_node(2, "Deep Learning"),
            create_test_node(3, "Cooking Recipes"),
            create_test_node(4, "Travel Guide"),
        ];

        let edges = vec![
            Edge::new(1, 2, 1.0), 
            Edge::new(3, 4, 1.0), 
        ];

        GraphData {
            nodes,
            edges,
            metadata: crate::models::metadata::MetadataStore::new(),
            id_to_metadata: std::collections::HashMap::new(),
        }
    }

    fn create_hierarchical_graph() -> GraphData {
        let nodes = vec![
            create_test_node(1, "Overview"),
            create_test_node(2, "Chapter 1"),
            create_test_node(3, "Chapter 2"),
            create_test_node(4, "Section 1.1"),
        ];

        let edges = vec![
            Edge::new(1, 2, 1.0),
            Edge::new(1, 3, 1.0),
            Edge::new(2, 4, 1.0),
        ];

        GraphData {
            nodes,
            edges,
            metadata: crate::models::metadata::MetadataStore::new(),
            id_to_metadata: std::collections::HashMap::new(),
        }
    }

    fn create_simple_graph() -> GraphData {
        let nodes = vec![
            create_test_node(1, "Node 1"),
            create_test_node(2, "Node 2"),
            create_test_node(3, "Node 3"),
        ];

        let edges = vec![Edge::new(1, 2, 1.0), Edge::new(2, 3, 1.0)];

        GraphData {
            nodes,
            edges,
            metadata: crate::models::metadata::MetadataStore::new(),
            id_to_metadata: std::collections::HashMap::new(),
        }
    }

    fn create_large_graph(node_count: u32) -> GraphData {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        
        for i in 1..=node_count {
            nodes.push(create_test_node(i, &format!("Node {}", i)));
        }

        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let edge_count = (node_count as f32 * 2.0) as usize; 

        for _ in 0..edge_count {
            let source = rng.gen_range(1..=node_count);
            let target = rng.gen_range(1..=node_count);
            if source != target {
                edges.push(Edge::new(source, target, 1.0));
            }
        }

        GraphData {
            nodes,
            edges,
            metadata: crate::models::metadata::MetadataStore::new(),
            id_to_metadata: std::collections::HashMap::new(),
        }
    }

    fn create_test_node(id: u32, label: &str) -> Node {
        let mut node = Node::new_with_id(
            format!("{}.md", label.to_lowercase().replace(' ', "_")),
            Some(id),
        );
        node.label = label.to_string();

        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        node.data.x = rng.gen_range(-100.0..100.0);
        node.data.y = rng.gen_range(-100.0..100.0);
        node.data.z = rng.gen_range(-100.0..100.0);

        node
    }

    fn create_ai_metadata() -> MetadataStore {
        let mut store = MetadataStore::new();

        let ai_topics = [("artificial_intelligence", 20), ("technology", 10)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let ml_topics = [("machine_learning", 25), ("artificial_intelligence", 15)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let dl_topics = [("deep_learning", 30), ("machine_learning", 20)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let nn_topics = [("neural_networks", 35), ("deep_learning", 25)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        store.insert(
            "ai_overview.md".to_string(),
            Metadata {
                file_name: "ai_overview.md".to_string(),
                file_size: 5000,
                topic_counts: ai_topics,
                ..Default::default()
            },
        );

        store.insert(
            "machine_learning.md".to_string(),
            Metadata {
                file_name: "machine_learning.md".to_string(),
                file_size: 8000,
                topic_counts: ml_topics,
                ..Default::default()
            },
        );

        store.insert(
            "deep_learning.md".to_string(),
            Metadata {
                file_name: "deep_learning.md".to_string(),
                file_size: 10000,
                topic_counts: dl_topics,
                ..Default::default()
            },
        );

        store.insert(
            "neural_networks.md".to_string(),
            Metadata {
                file_name: "neural_networks.md".to_string(),
                file_size: 7000,
                topic_counts: nn_topics,
                ..Default::default()
            },
        );

        store
    }

    fn create_mixed_metadata() -> MetadataStore {
        let mut store = MetadataStore::new();

        let ml_topics = [("machine_learning", 25), ("artificial_intelligence", 15)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let dl_topics = [("deep_learning", 30), ("machine_learning", 20)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let cooking_topics = [("cooking", 40), ("recipes", 30)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let travel_topics = [("travel", 35), ("tourism", 25)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        store.insert(
            "machine_learning.md".to_string(),
            Metadata {
                topic_counts: ml_topics,
                ..Default::default()
            },
        );

        store.insert(
            "deep_learning.md".to_string(),
            Metadata {
                topic_counts: dl_topics,
                ..Default::default()
            },
        );

        store.insert(
            "cooking_recipes.md".to_string(),
            Metadata {
                topic_counts: cooking_topics,
                ..Default::default()
            },
        );

        store.insert(
            "travel_guide.md".to_string(),
            Metadata {
                topic_counts: travel_topics,
                ..Default::default()
            },
        );

        store
    }

    fn create_hierarchical_metadata() -> MetadataStore {
        let mut store = MetadataStore::new();

        let overview_topics = [("overview", 30), ("index", 20)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        store.insert(
            "overview.md".to_string(),
            Metadata {
                file_name: "overview.md".to_string(),
                file_size: 15000,
                topic_counts: overview_topics,
                hyperlink_count: 10,
                ..Default::default()
            },
        );

        store.insert(
            "chapter_1.md".to_string(),
            Metadata {
                file_size: 8000,
                hyperlink_count: 3,
                ..Default::default()
            },
        );

        store.insert(
            "chapter_2.md".to_string(),
            Metadata {
                file_size: 7000,
                hyperlink_count: 2,
                ..Default::default()
            },
        );

        store.insert(
            "section_1.1.md".to_string(),
            Metadata {
                file_size: 3000,
                hyperlink_count: 1,
                ..Default::default()
            },
        );

        store
    }

    fn create_test_constraints() -> ConstraintSet {
        let mut constraint_set = ConstraintSet::default();

        
        constraint_set.add(crate::models::constraints::Constraint::fixed_position(
            1, 0.0, 0.0, 0.0,
        ));
        constraint_set.add(crate::models::constraints::Constraint::separation(
            2, 3, 100.0,
        ));

        constraint_set
    }
}
