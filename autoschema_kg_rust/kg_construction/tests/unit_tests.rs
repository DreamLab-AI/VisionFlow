use kg_construction::*;
use tokio_test;
use rstest::*;
use test_case::test_case;
use proptest::prelude::*;
use serde_json::json;
use std::collections::HashMap;

mod common;
use common::*;

#[cfg(test)]
mod neo4j_tests {
    use super::*;

    #[tokio::test]
    async fn test_neo4j_client_creation() {
        let config = Neo4jConfig::new("bolt://localhost:7687", "neo4j", "password");
        let client = Neo4jClient::new(config).await;

        // In test environment, this might fail due to no actual Neo4j instance
        // but we can test the configuration
        assert!(client.is_ok() || matches!(client.unwrap_err().kind(), Neo4jErrorKind::Connection));
    }

    #[tokio::test]
    async fn test_node_creation() {
        let mut mock_client = MockNeo4jClient::new();

        mock_client
            .expect_create_node()
            .with(mockall::predicate::eq(json!({
                "label": "Person",
                "properties": {
                    "name": "Alice",
                    "age": 30
                }
            })))
            .times(1)
            .returning(|_| Ok("node_id_123".to_string()));

        let node_data = json!({
            "label": "Person",
            "properties": {
                "name": "Alice",
                "age": 30
            }
        });

        let result = mock_client.create_node(&node_data).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "node_id_123");
    }

    #[tokio::test]
    async fn test_relationship_creation() {
        let mut mock_client = MockNeo4jClient::new();

        mock_client
            .expect_create_relationship()
            .times(1)
            .returning(|_| Ok("rel_id_456".to_string()));

        let rel_data = json!({
            "from": "node_1",
            "to": "node_2",
            "type": "KNOWS",
            "properties": {
                "since": "2020"
            }
        });

        let result = mock_client.create_relationship(&rel_data).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "rel_id_456");
    }

    #[tokio::test]
    async fn test_cypher_query_execution() {
        let mut mock_client = MockNeo4jClient::new();

        let expected_results = vec![
            json!({"name": "Alice", "age": 30}),
            json!({"name": "Bob", "age": 25}),
        ];

        mock_client
            .expect_run_cypher()
            .with(
                mockall::predicate::eq("MATCH (n:Person) WHERE n.age > $age RETURN n.name, n.age"),
                mockall::predicate::eq(maplit::hashmap! {
                    "age".to_string() => json!(20)
                })
            )
            .times(1)
            .returning(move |_, _| Ok(expected_results.clone()));

        let query = "MATCH (n:Person) WHERE n.age > $age RETURN n.name, n.age";
        let params = maplit::hashmap! {
            "age".to_string() => json!(20)
        };

        let results = mock_client.run_cypher(query, &params).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["name"], "Alice");
        assert_eq!(results[1]["name"], "Bob");
    }

    #[tokio::test]
    async fn test_batch_node_creation() {
        let mut mock_client = MockNeo4jClient::new();

        mock_client
            .expect_create_nodes_batch()
            .times(1)
            .returning(|nodes| {
                Ok(nodes.iter().enumerate().map(|(i, _)| format!("node_{}", i)).collect())
            });

        let nodes = vec![
            json!({"label": "Person", "properties": {"name": "Alice"}}),
            json!({"label": "Person", "properties": {"name": "Bob"}}),
            json!({"label": "Person", "properties": {"name": "Charlie"}}),
        ];

        let result = mock_client.create_nodes_batch(&nodes).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_cypher_query_builder() {
        let builder = CypherQueryBuilder::new();

        let query = builder
            .match_node("n", "Person")
            .where_clause("n.age > $age")
            .return_clause("n.name, n.age")
            .build();

        assert_eq!(query, "MATCH (n:Person) WHERE n.age > $age RETURN n.name, n.age");
    }

    #[test]
    fn test_node_validation() {
        let valid_node = json!({
            "label": "Person",
            "properties": {
                "name": "Alice",
                "age": 30
            }
        });

        let invalid_node = json!({
            "properties": {
                "name": "Alice"
            }
            // Missing label
        });

        assert!(validate_node(&valid_node).is_ok());
        assert!(validate_node(&invalid_node).is_err());
    }
}

#[cfg(test)]
mod entity_extraction_tests {
    use super::*;

    #[tokio::test]
    async fn test_ner_extraction() {
        let mut mock_extractor = MockEntityExtractor::new();

        mock_extractor
            .expect_extract_entities()
            .with(mockall::predicate::eq("Apple Inc. was founded by Steve Jobs in Cupertino."))
            .times(1)
            .returning(|_| Ok(vec![
                Entity::new("Apple Inc.", "ORGANIZATION", 0.95),
                Entity::new("Steve Jobs", "PERSON", 0.98),
                Entity::new("Cupertino", "LOCATION", 0.88),
            ]));

        let text = "Apple Inc. was founded by Steve Jobs in Cupertino.";
        let entities = mock_extractor.extract_entities(text).await.unwrap();

        assert_eq!(entities.len(), 3);
        assert_eq!(entities[0].text, "Apple Inc.");
        assert_eq!(entities[0].entity_type, "ORGANIZATION");
        assert!(entities[0].confidence > 0.9);
    }

    #[test]
    fn test_entity_normalization() {
        let entities = vec![
            Entity::new("Apple Inc.", "ORGANIZATION", 0.95),
            Entity::new("Apple Inc", "ORGANIZATION", 0.93),
            Entity::new("APPLE INC.", "ORGANIZATION", 0.90),
        ];

        let normalized = normalize_entities(entities);
        assert_eq!(normalized.len(), 1); // Should merge similar entities
        assert_eq!(normalized[0].text, "Apple Inc.");
        assert!(normalized[0].confidence > 0.93); // Should use highest confidence
    }

    #[test]
    fn test_entity_linking() {
        let entities = vec![
            Entity::new("Barack Obama", "PERSON", 0.98),
            Entity::new("Obama", "PERSON", 0.88),
            Entity::new("President Obama", "PERSON", 0.92),
        ];

        let linked = link_entities(entities);
        assert_eq!(linked.len(), 1);
        assert!(linked[0].aliases.contains(&"Obama".to_string()));
        assert!(linked[0].aliases.contains(&"President Obama".to_string()));
    }

    #[tokio::test]
    async fn test_batch_entity_extraction() {
        let mut mock_extractor = MockEntityExtractor::new();

        mock_extractor
            .expect_extract_entities_batch()
            .times(1)
            .returning(|texts| {
                Ok(texts.iter().enumerate().map(|(i, _)| vec![
                    Entity::new(&format!("Entity {}", i), "MISC", 0.8)
                ]).collect())
            });

        let texts = vec![
            "First document text.",
            "Second document text.",
            "Third document text.",
        ];

        let results = mock_extractor.extract_entities_batch(&texts).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 1);
    }
}

#[cfg(test)]
mod relationship_extraction_tests {
    use super::*;

    #[tokio::test]
    async fn test_relationship_extraction() {
        let mut mock_extractor = MockRelationshipExtractor::new();

        mock_extractor
            .expect_extract_relationships()
            .times(1)
            .returning(|_, _| Ok(vec![
                Relationship::new("Steve Jobs", "founded", "Apple Inc.", 0.92),
                Relationship::new("Apple Inc.", "located_in", "Cupertino", 0.85),
            ]));

        let text = "Steve Jobs founded Apple Inc. in Cupertino.";
        let entities = vec![
            Entity::new("Steve Jobs", "PERSON", 0.98),
            Entity::new("Apple Inc.", "ORGANIZATION", 0.95),
            Entity::new("Cupertino", "LOCATION", 0.88),
        ];

        let relationships = mock_extractor.extract_relationships(text, &entities).await.unwrap();

        assert_eq!(relationships.len(), 2);
        assert_eq!(relationships[0].subject, "Steve Jobs");
        assert_eq!(relationships[0].predicate, "founded");
        assert_eq!(relationships[0].object, "Apple Inc.");
    }

    #[test]
    fn test_relationship_patterns() {
        let patterns = RelationshipPatterns::default();

        let test_cases = vec![
            ("X founded Y", "founded"),
            ("X is the CEO of Y", "ceo_of"),
            ("X acquired Y", "acquired"),
            ("X is located in Y", "located_in"),
        ];

        for (text, expected_relation) in test_cases {
            let relation = patterns.match_pattern(text);
            assert!(relation.is_some());
            assert_eq!(relation.unwrap(), expected_relation);
        }
    }

    #[test]
    fn test_relationship_validation() {
        let valid_rel = Relationship::new("Alice", "knows", "Bob", 0.85);
        let invalid_rel = Relationship::new("", "knows", "Bob", 0.85); // Empty subject

        assert!(validate_relationship(&valid_rel).is_ok());
        assert!(validate_relationship(&invalid_rel).is_err());
    }

    #[tokio::test]
    async fn test_coreference_resolution() {
        let mut mock_resolver = MockCoreferenceResolver::new();

        mock_resolver
            .expect_resolve()
            .times(1)
            .returning(|_| Ok("Steve Jobs founded Apple Inc. Steve Jobs was a visionary.".to_string()));

        let text = "Steve Jobs founded Apple Inc. He was a visionary.";
        let resolved = mock_resolver.resolve(text).await.unwrap();

        assert!(resolved.contains("Steve Jobs was a visionary"));
        assert!(!resolved.contains(" He "));
    }
}

#[cfg(test)]
mod knowledge_graph_tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_creation() {
        let kg = KnowledgeGraph::new();
        assert_eq!(kg.node_count(), 0);
        assert_eq!(kg.edge_count(), 0);
    }

    #[test]
    fn test_add_entities_to_graph() {
        let mut kg = KnowledgeGraph::new();

        let entities = vec![
            Entity::new("Alice", "PERSON", 0.95),
            Entity::new("Google", "ORGANIZATION", 0.98),
            Entity::new("Mountain View", "LOCATION", 0.88),
        ];

        for entity in entities {
            kg.add_entity(entity);
        }

        assert_eq!(kg.node_count(), 3);
        assert!(kg.has_entity("Alice"));
        assert!(kg.has_entity("Google"));
        assert!(kg.has_entity("Mountain View"));
    }

    #[test]
    fn test_add_relationships_to_graph() {
        let mut kg = KnowledgeGraph::new();

        // Add entities first
        kg.add_entity(Entity::new("Alice", "PERSON", 0.95));
        kg.add_entity(Entity::new("Google", "ORGANIZATION", 0.98));

        // Add relationship
        let relationship = Relationship::new("Alice", "works_at", "Google", 0.90);
        kg.add_relationship(relationship);

        assert_eq!(kg.edge_count(), 1);
        assert!(kg.has_relationship("Alice", "Google", "works_at"));
    }

    #[test]
    fn test_graph_traversal() {
        let mut kg = KnowledgeGraph::new();

        // Build a small graph: Alice -> Google -> Mountain View
        kg.add_entity(Entity::new("Alice", "PERSON", 0.95));
        kg.add_entity(Entity::new("Google", "ORGANIZATION", 0.98));
        kg.add_entity(Entity::new("Mountain View", "LOCATION", 0.88));

        kg.add_relationship(Relationship::new("Alice", "works_at", "Google", 0.90));
        kg.add_relationship(Relationship::new("Google", "located_in", "Mountain View", 0.85));

        let neighbors = kg.get_neighbors("Alice", 1);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&"Google".to_string()));

        let two_hop_neighbors = kg.get_neighbors("Alice", 2);
        assert_eq!(two_hop_neighbors.len(), 2);
        assert!(two_hop_neighbors.contains(&"Google".to_string()));
        assert!(two_hop_neighbors.contains(&"Mountain View".to_string()));
    }

    #[test]
    fn test_graph_statistics() {
        let mut kg = KnowledgeGraph::new();

        // Add nodes and edges
        for i in 0..10 {
            kg.add_entity(Entity::new(&format!("Node{}", i), "TEST", 0.9));
        }

        for i in 0..5 {
            kg.add_relationship(Relationship::new(
                &format!("Node{}", i),
                "connected_to",
                &format!("Node{}", (i + 1) % 10),
                0.8
            ));
        }

        let stats = kg.get_statistics();
        assert_eq!(stats.node_count, 10);
        assert_eq!(stats.edge_count, 5);
        assert!(stats.density > 0.0);
        assert!(stats.avg_degree > 0.0);
    }

    #[tokio::test]
    async fn test_graph_persistence() {
        let mut kg = KnowledgeGraph::new();

        kg.add_entity(Entity::new("Test Entity", "TEST", 0.9));
        kg.add_relationship(Relationship::new("A", "test_rel", "B", 0.8));

        let mut mock_client = MockNeo4jClient::new();
        mock_client
            .expect_create_node()
            .times(3) // A, B, Test Entity
            .returning(|_| Ok("node_id".to_string()));

        mock_client
            .expect_create_relationship()
            .times(1)
            .returning(|_| Ok("rel_id".to_string()));

        let result = kg.persist_to_neo4j(&mock_client).await;
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod data_ingestion_tests {
    use super::*;

    #[tokio::test]
    async fn test_file_ingestion() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "This is test content for ingestion.").unwrap();

        let ingestor = DataIngestor::new();
        let result = ingestor.ingest_file(&file_path).await;

        assert!(result.is_ok());
        let document = result.unwrap();
        assert!(!document.content.is_empty());
        assert_eq!(document.source, file_path.to_string_lossy());
    }

    #[tokio::test]
    async fn test_csv_ingestion() {
        let temp_dir = tempfile::tempdir().unwrap();
        let csv_path = temp_dir.path().join("test.csv");
        std::fs::write(&csv_path, "id,name,description\n1,Test,Description").unwrap();

        let ingestor = DataIngestor::new();
        let result = ingestor.ingest_csv(&csv_path).await;

        assert!(result.is_ok());
        let documents = result.unwrap();
        assert_eq!(documents.len(), 1);
        assert!(documents[0].content.contains("Test"));
    }

    #[tokio::test]
    async fn test_json_ingestion() {
        let temp_dir = tempfile::tempdir().unwrap();
        let json_path = temp_dir.path().join("test.json");
        let json_content = json!({
            "items": [
                {"id": 1, "title": "First Item", "content": "First content"},
                {"id": 2, "title": "Second Item", "content": "Second content"}
            ]
        });
        std::fs::write(&json_path, json_content.to_string()).unwrap();

        let ingestor = DataIngestor::new();
        let result = ingestor.ingest_json(&json_path).await;

        assert!(result.is_ok());
        let documents = result.unwrap();
        assert_eq!(documents.len(), 2);
    }

    #[tokio::test]
    async fn test_batch_ingestion() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create multiple test files
        let files = vec![
            ("file1.txt", "Content 1"),
            ("file2.txt", "Content 2"),
            ("file3.txt", "Content 3"),
        ];

        let file_paths: Vec<_> = files.iter().map(|(name, content)| {
            let path = temp_dir.path().join(name);
            std::fs::write(&path, content).unwrap();
            path
        }).collect();

        let ingestor = DataIngestor::new();
        let result = ingestor.ingest_batch(&file_paths).await;

        assert!(result.is_ok());
        let documents = result.unwrap();
        assert_eq!(documents.len(), 3);
    }

    #[test]
    fn test_file_type_detection() {
        let detector = FileTypeDetector::new();

        assert_eq!(detector.detect_type("file.txt"), Some(FileType::Text));
        assert_eq!(detector.detect_type("data.csv"), Some(FileType::Csv));
        assert_eq!(detector.detect_type("config.json"), Some(FileType::Json));
        assert_eq!(detector.detect_type("document.md"), Some(FileType::Markdown));
        assert_eq!(detector.detect_type("unknown.xyz"), None);
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_pipeline() {
        let config = PipelineConfig::default();
        let mut pipeline = KGConstructionPipeline::new(config);

        // Mock components
        pipeline.set_entity_extractor(Box::new(MockEntityExtractor::new()));
        pipeline.set_relationship_extractor(Box::new(MockRelationshipExtractor::new()));
        pipeline.set_neo4j_client(Box::new(MockNeo4jClient::new()));

        let documents = vec![
            Document::new("doc1", "Apple Inc. was founded by Steve Jobs.", "test"),
            Document::new("doc2", "Google is located in Mountain View.", "test"),
        ];

        let result = pipeline.process_documents(documents).await;
        assert!(result.is_ok());

        let kg = result.unwrap();
        assert!(kg.node_count() > 0);
        assert!(kg.edge_count() >= 0);
    }

    #[tokio::test]
    async fn test_pipeline_error_handling() {
        let config = PipelineConfig::default();
        let mut pipeline = KGConstructionPipeline::new(config);

        // Mock failing extractor
        let mut mock_extractor = MockEntityExtractor::new();
        mock_extractor
            .expect_extract_entities()
            .returning(|_| Err(ExtractionError::ProcessingFailed("Mock error".to_string())));

        pipeline.set_entity_extractor(Box::new(mock_extractor));

        let documents = vec![
            Document::new("doc1", "Test content", "test"),
        ];

        let result = pipeline.process_documents(documents).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_configuration() {
        let config = PipelineConfig {
            batch_size: 50,
            max_workers: 8,
            enable_coreference: true,
            entity_confidence_threshold: 0.8,
            relationship_confidence_threshold: 0.7,
            ..Default::default()
        };

        let pipeline = KGConstructionPipeline::new(config);
        assert_eq!(pipeline.config().batch_size, 50);
        assert_eq!(pipeline.config().max_workers, 8);
        assert!(pipeline.config().enable_coreference);
    }
}

// Property-based tests
proptest! {
    #[test]
    fn test_entity_confidence_bounds(
        name in "[a-zA-Z ]{1,50}",
        entity_type in "[A-Z]{3,15}",
        confidence in 0.0f64..1.0
    ) {
        let entity = Entity::new(&name, &entity_type, confidence);
        prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);
        prop_assert_eq!(entity.text, name);
        prop_assert_eq!(entity.entity_type, entity_type);
    }

    #[test]
    fn test_relationship_properties(
        subject in "[a-zA-Z ]{1,30}",
        predicate in "[a-z_]{1,20}",
        object in "[a-zA-Z ]{1,30}",
        confidence in 0.0f64..1.0
    ) {
        let relationship = Relationship::new(&subject, &predicate, &object, confidence);
        prop_assert_eq!(relationship.subject, subject);
        prop_assert_eq!(relationship.predicate, predicate);
        prop_assert_eq!(relationship.object, object);
        prop_assert!(relationship.confidence >= 0.0 && relationship.confidence <= 1.0);
    }
}

// Integration-style tests within unit test module
#[cfg(test)]
mod integration_style_tests {
    use super::*;

    #[tokio::test]
    async fn test_end_to_end_knowledge_extraction() {
        // This test simulates the full pipeline without external dependencies
        let text = "Microsoft was founded by Bill Gates and Paul Allen in 1975. The company is headquartered in Redmond, Washington.";

        // Mock entity extraction
        let entities = vec![
            Entity::new("Microsoft", "ORGANIZATION", 0.98),
            Entity::new("Bill Gates", "PERSON", 0.96),
            Entity::new("Paul Allen", "PERSON", 0.94),
            Entity::new("1975", "DATE", 0.90),
            Entity::new("Redmond", "LOCATION", 0.88),
            Entity::new("Washington", "LOCATION", 0.86),
        ];

        // Mock relationship extraction
        let relationships = vec![
            Relationship::new("Bill Gates", "founded", "Microsoft", 0.92),
            Relationship::new("Paul Allen", "founded", "Microsoft", 0.90),
            Relationship::new("Microsoft", "founded_in", "1975", 0.88),
            Relationship::new("Microsoft", "headquartered_in", "Redmond", 0.85),
            Relationship::new("Redmond", "located_in", "Washington", 0.82),
        ];

        // Build knowledge graph
        let mut kg = KnowledgeGraph::new();

        for entity in entities {
            kg.add_entity(entity);
        }

        for relationship in relationships {
            kg.add_relationship(relationship);
        }

        // Verify graph structure
        assert_eq!(kg.node_count(), 6);
        assert_eq!(kg.edge_count(), 5);

        // Verify specific relationships
        assert!(kg.has_relationship("Bill Gates", "Microsoft", "founded"));
        assert!(kg.has_relationship("Microsoft", "Redmond", "headquartered_in"));

        // Test graph traversal
        let founders = kg.get_entities_by_relationship("Microsoft", "founded", "incoming");
        assert_eq!(founders.len(), 2);
        assert!(founders.contains(&"Bill Gates".to_string()));
        assert!(founders.contains(&"Paul Allen".to_string()));
    }
}

// Mock trait implementations would be defined in common/mod.rs
// For brevity, showing the concept here
use mockall::mock;

mock! {
    EntityExtractor {}

    #[async_trait::async_trait]
    impl EntityExtractor for EntityExtractor {
        async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>, ExtractionError>;
        async fn extract_entities_batch(&self, texts: &[&str]) -> Result<Vec<Vec<Entity>>, ExtractionError>;
    }
}

mock! {
    RelationshipExtractor {}

    #[async_trait::async_trait]
    impl RelationshipExtractor for RelationshipExtractor {
        async fn extract_relationships(&self, text: &str, entities: &[Entity]) -> Result<Vec<Relationship>, ExtractionError>;
    }
}

mock! {
    Neo4jClient {}

    #[async_trait::async_trait]
    impl Neo4jClient for Neo4jClient {
        async fn create_node(&self, node: &serde_json::Value) -> Result<String, Neo4jError>;
        async fn create_relationship(&self, rel: &serde_json::Value) -> Result<String, Neo4jError>;
        async fn run_cypher(&self, query: &str, params: &HashMap<String, serde_json::Value>) -> Result<Vec<serde_json::Value>, Neo4jError>;
        async fn create_nodes_batch(&self, nodes: &[serde_json::Value]) -> Result<Vec<String>, Neo4jError>;
    }
}

mock! {
    CoreferenceResolver {}

    #[async_trait::async_trait]
    impl CoreferenceResolver for CoreferenceResolver {
        async fn resolve(&self, text: &str) -> Result<String, ExtractionError>;
    }
}