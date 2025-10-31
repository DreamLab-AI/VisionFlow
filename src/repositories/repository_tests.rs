// src/repositories/repository_tests.rs
//! Dual-Adapter Comparison Tests
//!
//! These tests verify 99.9% parity between legacy SQLite adapters
//! and new unified adapters. Critical for validating the Adapter Pattern
//! migration strategy preserves all CUDA kernel compatibility.

#[cfg(test)]
mod tests {
    use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
    use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
    use crate::models::edge::Edge;
    use crate::models::node::Node;
    use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
    use crate::ports::ontology_repository::{OwlAxiom, OwlClass, OwlProperty, OntologyRepository, AxiomType, PropertyType};
    use crate::repositories::{UnifiedGraphRepository, UnifiedOntologyRepository};
    use std::collections::HashMap;

    /// Test load_graph() result parity
    #[tokio::test]
    async fn test_load_graph_parity() {
        let old_repo =
            SqliteKnowledgeGraphRepository::new(":memory:").expect("Failed to create old repo");
        let new_repo =
            UnifiedGraphRepository::new(":memory:").expect("Failed to create new repo");

        // Add same test data to both repositories
        let mut node1 = Node::new("test-node-1".to_string());
        node1.label = "Node 1".to_string();
        node1.data.x = 10.0;
        node1.data.y = 20.0;
        node1.data.z = 30.0;

        let mut node2 = Node::new("test-node-2".to_string());
        node2.label = "Node 2".to_string();
        node2.data.x = 40.0;
        node2.data.y = 50.0;
        node2.data.z = 60.0;

        let old_id1 = old_repo.add_node(&node1).await.expect("Failed to add node to old repo");
        let old_id2 = old_repo.add_node(&node2).await.expect("Failed to add node to old repo");

        let new_id1 = new_repo.add_node(&node1).await.expect("Failed to add node to new repo");
        let new_id2 = new_repo.add_node(&node2).await.expect("Failed to add node to new repo");

        // Add edges
        let edge = Edge::new(old_id1, old_id2, 1.5);
        old_repo.add_edge(&edge).await.expect("Failed to add edge to old repo");

        let edge = Edge::new(new_id1, new_id2, 1.5);
        new_repo.add_edge(&edge).await.expect("Failed to add edge to new repo");

        // Load graphs
        let old_graph = old_repo.load_graph().await.expect("Failed to load old graph");
        let new_graph = new_repo.load_graph().await.expect("Failed to load new graph");

        // Assert parity (99.9% match)
        assert_eq!(
            old_graph.nodes.len(),
            new_graph.nodes.len(),
            "Node count mismatch"
        );
        assert_eq!(
            old_graph.edges.len(),
            new_graph.edges.len(),
            "Edge count mismatch"
        );

        // Compare node positions (CRITICAL for CUDA compatibility)
        for (old_node, new_node) in old_graph.nodes.iter().zip(new_graph.nodes.iter()) {
            assert!(
                (old_node.data.x - new_node.data.x).abs() < 0.001,
                "X position mismatch: {} vs {}",
                old_node.data.x,
                new_node.data.x
            );
            assert!(
                (old_node.data.y - new_node.data.y).abs() < 0.001,
                "Y position mismatch: {} vs {}",
                old_node.data.y,
                new_node.data.y
            );
            assert!(
                (old_node.data.z - new_node.data.z).abs() < 0.001,
                "Z position mismatch: {} vs {}",
                old_node.data.z,
                new_node.data.z
            );
            assert_eq!(
                old_node.label, new_node.label,
                "Label mismatch: {} vs {}",
                old_node.label, new_node.label
            );
        }

        println!("✅ test_load_graph_parity: 100% match");
    }

    /// Test batch_update_positions() - CRITICAL for CUDA kernels
    #[tokio::test]
    async fn test_batch_update_positions_parity() {
        let old_repo =
            SqliteKnowledgeGraphRepository::new(":memory:").expect("Failed to create old repo");
        let new_repo =
            UnifiedGraphRepository::new(":memory:").expect("Failed to create new repo");

        // Add test nodes
        let mut node1 = Node::new("node1".to_string());
        node1.label = "Node 1".to_string();
        let old_id1 = old_repo.add_node(&node1).await.expect("Failed to add node");

        let mut node2 = Node::new("node2".to_string());
        node2.label = "Node 2".to_string();
        let old_id2 = old_repo.add_node(&node2).await.expect("Failed to add node");

        let new_id1 = new_repo.add_node(&node1).await.expect("Failed to add node");
        let new_id2 = new_repo.add_node(&node2).await.expect("Failed to add node");

        // Update positions (simulating CUDA kernel output)
        let old_positions = vec![(old_id1, 100.0, 200.0, 300.0), (old_id2, 400.0, 500.0, 600.0)];
        let new_positions = vec![(new_id1, 100.0, 200.0, 300.0), (new_id2, 400.0, 500.0, 600.0)];

        old_repo
            .batch_update_positions(old_positions)
            .await
            .expect("Failed to update positions in old repo");
        new_repo
            .batch_update_positions(new_positions)
            .await
            .expect("Failed to update positions in new repo");

        // Verify positions match (floating point tolerance)
        let old_node1 = old_repo.get_node(old_id1).await.expect("Failed to get node").expect("Node not found");
        let new_node1 = new_repo.get_node(new_id1).await.expect("Failed to get node").expect("Node not found");

        assert!(
            (old_node1.data.x - new_node1.data.x).abs() < 0.001,
            "X position mismatch after update"
        );
        assert!(
            (old_node1.data.y - new_node1.data.y).abs() < 0.001,
            "Y position mismatch after update"
        );
        assert!(
            (old_node1.data.z - new_node1.data.z).abs() < 0.001,
            "Z position mismatch after update"
        );

        println!("✅ test_batch_update_positions_parity: 100% match");
    }

    /// Test get_statistics() parity
    #[tokio::test]
    async fn test_statistics_parity() {
        let old_repo =
            SqliteKnowledgeGraphRepository::new(":memory:").expect("Failed to create old repo");
        let new_repo =
            UnifiedGraphRepository::new(":memory:").expect("Failed to create new repo");

        // Add same data to both
        let mut node1 = Node::new("node1".to_string());
        node1.label = "Node 1".to_string();
        let old_id1 = old_repo.add_node(&node1).await.expect("Failed to add node");
        let new_id1 = new_repo.add_node(&node1).await.expect("Failed to add node");

        let mut node2 = Node::new("node2".to_string());
        node2.label = "Node 2".to_string();
        let old_id2 = old_repo.add_node(&node2).await.expect("Failed to add node");
        let new_id2 = new_repo.add_node(&node2).await.expect("Failed to add node");

        let edge = Edge::new(old_id1, old_id2, 1.0);
        old_repo.add_edge(&edge).await.expect("Failed to add edge");

        let edge = Edge::new(new_id1, new_id2, 1.0);
        new_repo.add_edge(&edge).await.expect("Failed to add edge");

        // Get statistics
        let old_stats = old_repo.get_statistics().await.expect("Failed to get old stats");
        let new_stats = new_repo.get_statistics().await.expect("Failed to get new stats");

        // Assert parity
        assert_eq!(
            old_stats.node_count, new_stats.node_count,
            "Node count mismatch in statistics"
        );
        assert_eq!(
            old_stats.edge_count, new_stats.edge_count,
            "Edge count mismatch in statistics"
        );
        assert!(
            (old_stats.average_degree - new_stats.average_degree).abs() < 0.01,
            "Average degree mismatch in statistics"
        );

        println!("✅ test_statistics_parity: 100% match");
    }

    /// Test ontology load_ontology_graph() parity
    #[tokio::test]
    async fn test_ontology_load_graph_parity() {
        let old_repo =
            SqliteOntologyRepository::new(":memory:").expect("Failed to create old repo");
        let new_repo =
            UnifiedOntologyRepository::new(":memory:").expect("Failed to create new repo");

        // Add same OWL classes to both
        let class1 = OwlClass {
            iri: "http://example.org/Class1".to_string(),
            label: Some("Class 1".to_string()),
            description: Some("First test class".to_string()),
            parent_classes: Vec::new(),
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: Some("abc123".to_string()),
            last_synced: None,
        };

        let class2 = OwlClass {
            iri: "http://example.org/Class2".to_string(),
            label: Some("Class 2".to_string()),
            description: Some("Second test class".to_string()),
            parent_classes: vec!["http://example.org/Class1".to_string()],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: Some("def456".to_string()),
            last_synced: None,
        };

        old_repo
            .save_ontology(&[class1.clone(), class2.clone()], &[], &[])
            .await
            .expect("Failed to save ontology to old repo");

        new_repo
            .save_ontology(&[class1, class2], &[], &[])
            .await
            .expect("Failed to save ontology to new repo");

        // Load ontology graphs
        let old_graph = old_repo.load_ontology_graph().await.expect("Failed to load old graph");
        let new_graph = new_repo.load_ontology_graph().await.expect("Failed to load new graph");

        // Assert parity
        assert_eq!(
            old_graph.nodes.len(),
            new_graph.nodes.len(),
            "Ontology node count mismatch"
        );
        assert_eq!(
            old_graph.edges.len(),
            new_graph.edges.len(),
            "Ontology edge count mismatch (subclass relationships)"
        );

        // Verify node labels match
        for (old_node, new_node) in old_graph.nodes.iter().zip(new_graph.nodes.iter()) {
            assert_eq!(
                old_node.label, new_node.label,
                "Ontology node label mismatch"
            );
            assert_eq!(
                old_node.metadata.get("type"),
                new_node.metadata.get("type"),
                "Ontology node type mismatch"
            );
        }

        println!("✅ test_ontology_load_graph_parity: 100% match");
    }

    /// Test ontology save_ontology() and list_owl_classes() parity
    #[tokio::test]
    async fn test_ontology_save_and_list_parity() {
        let old_repo =
            SqliteOntologyRepository::new(":memory:").expect("Failed to create old repo");
        let new_repo =
            UnifiedOntologyRepository::new(":memory:").expect("Failed to create new repo");

        // Create test ontology data
        let classes = vec![
            OwlClass {
                iri: "http://example.org/Person".to_string(),
                label: Some("Person".to_string()),
                description: Some("A human being".to_string()),
                parent_classes: Vec::new(),
                properties: HashMap::new(),
                source_file: None,
                markdown_content: None,
                file_sha1: Some("person123".to_string()),
                last_synced: None,
            },
            OwlClass {
                iri: "http://example.org/Student".to_string(),
                label: Some("Student".to_string()),
                description: Some("A person who studies".to_string()),
                parent_classes: vec!["http://example.org/Person".to_string()],
                properties: HashMap::new(),
                source_file: None,
                markdown_content: None,
                file_sha1: Some("student456".to_string()),
                last_synced: None,
            },
        ];

        let properties = vec![OwlProperty {
            iri: "http://example.org/hasName".to_string(),
            label: Some("has name".to_string()),
            property_type: PropertyType::DataProperty,
            domain: vec!["http://example.org/Person".to_string()],
            range: vec!["xsd:string".to_string()],
        }];

        let axioms = vec![OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "http://example.org/Student".to_string(),
            object: "http://example.org/Person".to_string(),
            annotations: HashMap::new(),
        }];

        // Save to both repositories
        old_repo
            .save_ontology(&classes, &properties, &axioms)
            .await
            .expect("Failed to save to old repo");

        new_repo
            .save_ontology(&classes, &properties, &axioms)
            .await
            .expect("Failed to save to new repo");

        // List classes from both
        let old_classes = old_repo.list_owl_classes().await.expect("Failed to list old classes");
        let new_classes = new_repo.list_owl_classes().await.expect("Failed to list new classes");

        // Assert parity
        assert_eq!(
            old_classes.len(),
            new_classes.len(),
            "Class count mismatch"
        );

        // Sort by IRI for consistent comparison
        let mut old_sorted = old_classes;
        let mut new_sorted = new_classes;
        old_sorted.sort_by(|a, b| a.iri.cmp(&b.iri));
        new_sorted.sort_by(|a, b| a.iri.cmp(&b.iri));

        for (old_class, new_class) in old_sorted.iter().zip(new_sorted.iter()) {
            assert_eq!(old_class.iri, new_class.iri, "Class IRI mismatch");
            assert_eq!(old_class.label, new_class.label, "Class label mismatch");
            assert_eq!(
                old_class.description, new_class.description,
                "Class description mismatch"
            );
            assert_eq!(
                old_class.parent_classes.len(),
                new_class.parent_classes.len(),
                "Parent class count mismatch for {}",
                old_class.iri
            );
        }

        println!("✅ test_ontology_save_and_list_parity: 100% match");
    }

    /// Test get_axioms() parity
    #[tokio::test]
    async fn test_get_axioms_parity() {
        let old_repo =
            SqliteOntologyRepository::new(":memory:").expect("Failed to create old repo");
        let new_repo =
            UnifiedOntologyRepository::new(":memory:").expect("Failed to create new repo");

        let classes = vec![OwlClass {
            iri: "http://example.org/TestClass".to_string(),
            label: Some("Test Class".to_string()),
            description: None,
            parent_classes: Vec::new(),
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        }];

        let axioms = vec![
            OwlAxiom {
                id: None,
                axiom_type: AxiomType::SubClassOf,
                subject: "http://example.org/TestClass".to_string(),
                object: "http://example.org/BaseClass".to_string(),
                annotations: HashMap::new(),
            },
            OwlAxiom {
                id: None,
                axiom_type: AxiomType::DisjointWith,
                subject: "http://example.org/TestClass".to_string(),
                object: "http://example.org/OtherClass".to_string(),
                annotations: HashMap::new(),
            },
        ];

        old_repo
            .save_ontology(&classes, &[], &axioms)
            .await
            .expect("Failed to save to old repo");

        new_repo
            .save_ontology(&classes, &[], &axioms)
            .await
            .expect("Failed to save to new repo");

        let old_axioms = old_repo.get_axioms().await.expect("Failed to get old axioms");
        let new_axioms = new_repo.get_axioms().await.expect("Failed to get new axioms");

        assert_eq!(
            old_axioms.len(),
            new_axioms.len(),
            "Axiom count mismatch"
        );

        // Sort by subject+object for consistent comparison
        let mut old_sorted = old_axioms;
        let mut new_sorted = new_axioms;
        old_sorted.sort_by(|a, b| format!("{}{}", a.subject, a.object).cmp(&format!("{}{}", b.subject, b.object)));
        new_sorted.sort_by(|a, b| format!("{}{}", a.subject, a.object).cmp(&format!("{}{}", b.subject, b.object)));

        for (old_axiom, new_axiom) in old_sorted.iter().zip(new_sorted.iter()) {
            assert_eq!(
                old_axiom.axiom_type, new_axiom.axiom_type,
                "Axiom type mismatch"
            );
            assert_eq!(
                old_axiom.subject, new_axiom.subject,
                "Axiom subject mismatch"
            );
            assert_eq!(
                old_axiom.object, new_axiom.object,
                "Axiom object mismatch"
            );
        }

        println!("✅ test_get_axioms_parity: 100% match");
    }

    /// Integration test: Full graph + ontology workflow
    #[tokio::test]
    async fn test_full_workflow_integration() {
        let graph_repo =
            UnifiedGraphRepository::new(":memory:").expect("Failed to create graph repo");
        let ontology_repo =
            UnifiedOntologyRepository::new(":memory:").expect("Failed to create ontology repo");

        // 1. Add OWL classes
        let class1 = OwlClass {
            iri: "http://example.org/Entity".to_string(),
            label: Some("Entity".to_string()),
            description: Some("Base entity class".to_string()),
            parent_classes: Vec::new(),
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: Some("entity789".to_string()),
            last_synced: None,
        };

        ontology_repo
            .save_ontology(&[class1.clone()], &[], &[])
            .await
            .expect("Failed to save ontology");

        // 2. Create graph nodes linked to OWL classes
        let mut node1 = Node::new("entity-1".to_string());
        node1.label = "Entity Instance 1".to_string();
        node1
            .metadata
            .insert("owl_class_iri".to_string(), class1.iri.clone());

        let node_id = graph_repo.add_node(&node1).await.expect("Failed to add node");

        // 3. Verify node has OWL linkage
        let retrieved = graph_repo
            .get_node(node_id)
            .await
            .expect("Failed to get node")
            .expect("Node not found");

        assert_eq!(
            retrieved.metadata.get("owl_class_iri"),
            Some(&class1.iri),
            "OWL class linkage not preserved"
        );

        // 4. Verify ontology classes are accessible
        let classes = ontology_repo
            .list_owl_classes()
            .await
            .expect("Failed to list classes");

        assert_eq!(classes.len(), 1, "Should have 1 OWL class");
        assert_eq!(classes[0].iri, class1.iri, "Class IRI mismatch");

        println!("✅ test_full_workflow_integration: PASSED");
    }

    /// Performance benchmark: batch_update_positions with 10K nodes
    #[tokio::test]
    #[ignore] // Run with --ignored for performance testing
    async fn benchmark_batch_update_positions_10k() {
        use std::time::Instant;

        let repo = UnifiedGraphRepository::new(":memory:").expect("Failed to create repo");

        // Add 10K nodes
        let nodes: Vec<Node> = (0..10_000)
            .map(|i| {
                let mut node = Node::new(format!("node-{}", i));
                node.label = format!("Node {}", i);
                node
            })
            .collect();

        let node_ids = repo
            .batch_add_nodes(nodes)
            .await
            .expect("Failed to add nodes");

        // Prepare position updates (simulating CUDA kernel output)
        let positions: Vec<(u32, f32, f32, f32)> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as f32, i as f32 * 2.0, i as f32 * 3.0))
            .collect();

        // Benchmark batch update
        let start = Instant::now();
        repo.batch_update_positions(positions)
            .await
            .expect("Failed to update positions");
        let elapsed = start.elapsed();

        println!(
            "✅ benchmark_batch_update_positions_10k: {:?} ({:.2} nodes/ms)",
            elapsed,
            10_000.0 / elapsed.as_millis() as f64
        );

        // Assert reasonable performance (<500ms for 10K nodes)
        assert!(
            elapsed.as_millis() < 500,
            "Batch update took too long: {:?}",
            elapsed
        );
    }
}
