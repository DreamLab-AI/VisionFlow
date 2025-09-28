//! Integration tests for the concept generation pipeline

use kg_construction::{
    concept_generation::ConceptGenerator,
    concept_to_csv::{ConceptToCsv, ConceptValidator},
    llm_integration::MockLlmGenerator,
    statistics::ProcessingLogger,
    types::{ProcessingConfig, ShardConfig},
    graph_traversal::{Graph, GraphNode, GraphEdge, ContextExtractor},
    batch_processing::{build_comprehensive_batched_data, validate_batch_data},
    data_loading::{load_node_data_with_types, validate_csv_structure},
    csv_utils::{write_concept_nodes_csv, CsvConfig},
    prelude::*,
};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;

/// Test the complete concept generation pipeline
#[tokio::test]
async fn test_complete_concept_generation_pipeline() {
    let temp_dir = TempDir::new().unwrap();

    // Create test data
    let input_file = create_test_input_file(&temp_dir);

    // Set up configuration
    let config = ProcessingConfig {
        batch_size: 2,
        max_workers: 1,
        language: "en".to_string(),
        ..Default::default()
    };

    // Create mock LLM
    let llm = Arc::new(MockLlmGenerator::new(10));

    // Create logger
    let log_file = temp_dir.path().join("test.log");
    let logger = ProcessingLogger::new(&log_file).ok();

    // Create and run generator
    let mut generator = ConceptGenerator::new(config, llm, logger);

    let stats = generator.generate_concept(
        &input_file,
        temp_dir.path().join("output"),
        "concepts.csv",
        None,
        true,
    ).await.unwrap();

    // Verify results
    assert!(stats.total_nodes_processed > 0);
    assert!(stats.total_batches_processed > 0);
    assert_eq!(stats.errors_encountered, 0);

    // Check output file exists and has content
    let output_file = temp_dir.path().join("output").join("concepts.csv");
    assert!(output_file.exists());

    let content = std::fs::read_to_string(&output_file).unwrap();
    assert!(content.contains("node,conceptualized_node,node_type"));
    assert!(content.lines().count() > 1); // Header + data
}

/// Test concept generation with graph context
#[tokio::test]
async fn test_concept_generation_with_graph_context() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = create_test_input_file(&temp_dir);

    let config = ProcessingConfig {
        batch_size: 3,
        max_workers: 1,
        ..Default::default()
    };

    let llm = Arc::new(MockLlmGenerator::new(5));
    let mut generator = ConceptGenerator::new(config, llm, None);

    // Create and load graph
    let graph = create_test_graph();
    generator.load_graph(graph).unwrap();

    let stats = generator.generate_concept(
        &input_file,
        temp_dir.path().join("output_with_graph"),
        "concepts_with_graph.csv",
        None,
        false,
    ).await.unwrap();

    assert!(stats.total_nodes_processed > 0);
    assert_eq!(stats.errors_encountered, 0);
}

/// Test sharded processing
#[tokio::test]
async fn test_sharded_processing() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = create_test_input_file(&temp_dir);

    let config = ProcessingConfig {
        batch_size: 2,
        max_workers: 1,
        ..Default::default()
    };

    let llm = Arc::new(MockLlmGenerator::new(5));

    // Process with 2 shards
    let num_shards = 2;
    let mut total_processed = 0;

    for shard_idx in 0..num_shards {
        let shard_config = ShardConfig {
            shard_idx,
            num_shards,
            shuffle_data: false, // Disable shuffle for deterministic tests
        };

        let mut generator = ConceptGenerator::new(config.clone(), llm.clone(), None);

        let stats = generator.generate_concept(
            &input_file,
            temp_dir.path().join("sharded_output"),
            "concepts.csv",
            Some(shard_config),
            false,
        ).await.unwrap();

        total_processed += stats.total_nodes_processed;
        assert_eq!(stats.errors_encountered, 0);
    }

    // Verify that sharding processed all data
    assert!(total_processed > 0);
}

/// Test concept-to-CSV conversion
#[test]
fn test_concept_to_csv_conversion() {
    let temp_dir = TempDir::new().unwrap();

    // Create test files
    let concepts_file = create_test_concepts_file(&temp_dir);
    let nodes_file = create_test_nodes_file(&temp_dir);
    let edges_file = create_test_edges_file(&temp_dir);

    let converter = ConceptToCsv::new(None);

    let output_nodes = temp_dir.path().join("concept_nodes.csv");
    let output_edges = temp_dir.path().join("concept_edges.csv");
    let output_full_edges = temp_dir.path().join("full_concept_edges.csv");

    converter.all_concept_triples_csv_to_csv(
        &nodes_file,
        &edges_file,
        &concepts_file,
        &output_nodes,
        &output_edges,
        &output_full_edges,
    ).unwrap();

    // Verify output files exist
    assert!(output_nodes.exists());
    assert!(output_edges.exists());
    assert!(output_full_edges.exists());

    // Verify content
    let nodes_content = std::fs::read_to_string(&output_nodes).unwrap();
    assert!(nodes_content.contains("concept_id:ID,name,:LABEL"));

    let edges_content = std::fs::read_to_string(&output_edges).unwrap();
    assert!(edges_content.contains(":START_ID,:END_ID,relation,:TYPE"));
}

/// Test data loading and validation
#[test]
fn test_data_loading_and_validation() {
    let temp_dir = TempDir::new().unwrap();
    let input_file = create_test_input_file(&temp_dir);

    // Test loading without sharding
    let node_data = load_node_data_with_types(&input_file, None).unwrap();
    assert!(!node_data.is_empty());

    // Test validation
    validate_batch_data(&node_data).unwrap();

    // Test CSV structure validation
    validate_csv_structure(&input_file, 2).unwrap();

    // Test with sharding
    let shard_config = ShardConfig {
        shard_idx: 0,
        num_shards: 2,
        shuffle_data: false,
    };

    let shard_data = load_node_data_with_types(&input_file, Some(shard_config)).unwrap();
    assert!(!shard_data.is_empty());
    assert!(shard_data.len() <= node_data.len());
}

/// Test batch processing
#[test]
fn test_batch_processing() {
    let test_data = vec![
        ("entity1".to_string(), "entity".to_string()),
        ("event1".to_string(), "event".to_string()),
        ("relation1".to_string(), "relation".to_string()),
        ("entity2".to_string(), "entity".to_string()),
    ];

    // Validate data
    validate_batch_data(&test_data).unwrap();

    // Build batched data
    let batched_data = build_comprehensive_batched_data(&test_data, 2).unwrap();

    assert!(!batched_data.entities.is_empty());
    assert!(!batched_data.events.is_empty());
    assert!(!batched_data.relations.is_empty());

    // Test with larger batch size
    let batched_data_large = build_comprehensive_batched_data(&test_data, 10).unwrap();
    assert_eq!(batched_data_large.entities.len(), 1); // All entities in one batch
    assert_eq!(batched_data_large.events.len(), 1);   // All events in one batch
    assert_eq!(batched_data_large.relations.len(), 1); // All relations in one batch
}

/// Test graph traversal and context extraction
#[test]
fn test_graph_traversal() {
    let graph = create_test_graph();
    let config = GraphTraversal::default();
    let extractor = ContextExtractor::new(graph, config);

    // Test context extraction for existing node
    let context = extractor.extract_context("person1").unwrap();
    assert!(!context.is_empty());

    // Test context extraction for non-existent node
    let result = extractor.extract_context("non_existent");
    assert!(result.is_err());
}

/// Test concept validation
#[test]
fn test_concept_validation() {
    let mut mapping = ConceptMapping::default();

    // Add valid data
    mapping.node_to_concepts.insert(
        "test_node".to_string(),
        vec!["concept1".to_string(), "concept2".to_string()],
    );
    mapping.all_concepts.insert("concept1".to_string());
    mapping.all_concepts.insert("concept2".to_string());

    // Should pass validation
    ConceptValidator::validate_concept_mapping(&mapping).unwrap();

    // Test with empty concepts (should fail)
    mapping.node_to_concepts.insert("empty_node".to_string(), vec![]);
    let result = ConceptValidator::validate_concept_mapping(&mapping);
    assert!(result.is_err());
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() {
    let temp_dir = TempDir::new().unwrap();

    // Create invalid input file (missing required columns)
    let invalid_file = temp_dir.path().join("invalid.csv");
    let mut file = std::fs::File::create(&invalid_file).unwrap();
    writeln!(file, "invalid_header").unwrap();
    writeln!(file, "invalid_data").unwrap();

    let config = ProcessingConfig::default();
    let llm = Arc::new(MockLlmGenerator::new(0));
    let mut generator = ConceptGenerator::new(config, llm, None);

    // Should handle error gracefully
    let result = generator.generate_concept(
        &invalid_file,
        temp_dir.path().join("error_output"),
        "error_test.csv",
        None,
        false,
    ).await;

    // Should return an error due to invalid data format
    assert!(result.is_err());
}

/// Helper function to create test input file
fn create_test_input_file(temp_dir: &TempDir) -> std::path::PathBuf {
    let input_file = temp_dir.path().join("test_input.csv");
    let mut file = std::fs::File::create(&input_file).unwrap();

    writeln!(file, "node,type").unwrap();
    writeln!(file, "person1,entity").unwrap();
    writeln!(file, "company1,entity").unwrap();
    writeln!(file, "meeting1,event").unwrap();
    writeln!(file, "launch1,event").unwrap();
    writeln!(file, "works_at,relation").unwrap();
    writeln!(file, "participates_in,relation").unwrap();

    input_file
}

/// Helper function to create test concepts file
fn create_test_concepts_file(temp_dir: &TempDir) -> std::path::PathBuf {
    let file_path = temp_dir.path().join("test_concepts.csv");
    let mut file = std::fs::File::create(&file_path).unwrap();

    writeln!(file, "node,conceptualized_node,node_type").unwrap();
    writeln!(file, "person1,\"human, individual\",entity").unwrap();
    writeln!(file, "company1,\"organization, business\",entity").unwrap();
    writeln!(file, "meeting1,\"gathering, event\",event").unwrap();
    writeln!(file, "works_at,\"employment, affiliation\",relation").unwrap();

    file_path
}

/// Helper function to create test nodes file
fn create_test_nodes_file(temp_dir: &TempDir) -> std::path::PathBuf {
    let file_path = temp_dir.path().join("test_nodes.csv");
    let mut file = std::fs::File::create(&file_path).unwrap();

    writeln!(file, "name:ID,type,concepts,synsets,:LABEL").unwrap();
    writeln!(file, "person1,Entity,\"[human]\",\"[]\",Entity").unwrap();
    writeln!(file, "company1,Entity,\"[organization]\",\"[]\",Entity").unwrap();

    file_path
}

/// Helper function to create test edges file
fn create_test_edges_file(temp_dir: &TempDir) -> std::path::PathBuf {
    let file_path = temp_dir.path().join("test_edges.csv");
    let mut file = std::fs::File::create(&file_path).unwrap();

    writeln!(file, ":START_ID,:END_ID,relation,concepts,synsets,:TYPE").unwrap();
    writeln!(file, "person1,company1,works_at,\"[employment]\",\"[]\",Relation").unwrap();

    file_path
}

/// Helper function to create test graph
fn create_test_graph() -> Graph {
    let mut graph = Graph::new();

    // Add nodes
    graph.add_node(GraphNode {
        id: "person1".to_string(),
        label: "Person 1".to_string(),
        properties: HashMap::new(),
    });

    graph.add_node(GraphNode {
        id: "company1".to_string(),
        label: "Company 1".to_string(),
        properties: HashMap::new(),
    });

    graph.add_node(GraphNode {
        id: "meeting1".to_string(),
        label: "Meeting 1".to_string(),
        properties: HashMap::new(),
    });

    // Add edges
    graph.add_edge(GraphEdge {
        source: "person1".to_string(),
        target: "company1".to_string(),
        relation: "works_at".to_string(),
        properties: HashMap::new(),
    }).unwrap();

    graph.add_edge(GraphEdge {
        source: "person1".to_string(),
        target: "meeting1".to_string(),
        relation: "participates_in".to_string(),
        properties: HashMap::new(),
    }).unwrap();

    graph
}