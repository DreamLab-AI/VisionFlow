//! Example demonstrating concept generation functionality

use kg_construction::{
    concept_generation::{ConceptGenerator, StreamingConceptGenerator},
    llm_integration::MockLlmGenerator,
    statistics::ProcessingLogger,
    types::{ProcessingConfig, ShardConfig},
    graph_traversal::{Graph, GraphNode, GraphEdge},
    prelude::*,
};
use std::sync::Arc;
use std::collections::HashMap;
use tempfile::TempDir;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Concept Generation Example");

    // Create temporary directory for this example
    let temp_dir = TempDir::new()?;
    println!("📁 Working directory: {:?}", temp_dir.path());

    // Create sample input data
    create_sample_data(&temp_dir)?;

    // Example 1: Basic concept generation
    println!("\n📊 Example 1: Basic Concept Generation");
    basic_concept_generation_example(&temp_dir).await?;

    // Example 2: Concept generation with graph context
    println!("\n🕸️ Example 2: Concept Generation with Graph Context");
    graph_context_example(&temp_dir).await?;

    // Example 3: Streaming concept generation for large datasets
    println!("\n🌊 Example 3: Streaming Concept Generation");
    streaming_concept_generation_example(&temp_dir).await?;

    // Example 4: Sharded processing
    println!("\n🔀 Example 4: Sharded Processing");
    sharded_processing_example(&temp_dir).await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Create sample input data for the examples
fn create_sample_data(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let input_file = temp_dir.path().join("sample_nodes.csv");
    let mut file = std::fs::File::create(&input_file)?;

    writeln!(file, "node,type")?;
    writeln!(file, "John Smith,entity")?;
    writeln!(file, "Apple Inc,entity")?;
    writeln!(file, "conference meeting,event")?;
    writeln!(file, "product launch,event")?;
    writeln!(file, "works_at,relation")?;
    writeln!(file, "participates_in,relation")?;
    writeln!(file, "artificial intelligence,entity")?;
    writeln!(file, "machine learning,entity")?;
    writeln!(file, "neural network training,event")?;
    writeln!(file, "data processing,event")?;
    writeln!(file, "implements,relation")?;
    writeln!(file, "processes,relation")?;

    println!("📝 Created sample data with {} rows", 12);
    Ok(())
}

/// Example 1: Basic concept generation without graph context
async fn basic_concept_generation_example(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let config = ProcessingConfig {
        batch_size: 4,
        max_workers: 2,
        language: "en".to_string(),
        ..Default::default()
    };

    // Create mock LLM generator with 50ms delay to simulate real processing
    let llm = Arc::new(MockLlmGenerator::new(50));

    // Create logger for tracking progress
    let log_file = temp_dir.path().join("basic_example.log");
    let logger = ProcessingLogger::new(&log_file).ok();

    // Create concept generator
    let mut generator = ConceptGenerator::new(config, llm, logger);

    // Generate concepts
    let input_file = temp_dir.path().join("sample_nodes.csv");
    let output_folder = temp_dir.path().join("basic_output");

    let stats = generator.generate_concept(
        &input_file,
        &output_folder,
        "concepts.csv",
        None,
        true, // Record usage statistics
    ).await?;

    println!("📈 Processing Statistics:");
    println!("   - Nodes processed: {}", stats.total_nodes_processed);
    println!("   - Batches processed: {}", stats.total_batches_processed);
    println!("   - Unique concepts: {}", stats.unique_concepts_generated);
    println!("   - Processing time: {}ms", stats.processing_time_ms);
    println!("   - Errors: {}", stats.errors_encountered);

    // Check output file
    let output_file = output_folder.join("concepts.csv");
    if output_file.exists() {
        let content = std::fs::read_to_string(&output_file)?;
        let line_count = content.lines().count();
        println!("📄 Output file created with {} lines", line_count);
    }

    Ok(())
}

/// Example 2: Concept generation with graph context
async fn graph_context_example(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let config = ProcessingConfig {
        batch_size: 3,
        max_workers: 2,
        ..Default::default()
    };

    let llm = Arc::new(MockLlmGenerator::new(30));
    let logger = ProcessingLogger::new(temp_dir.path().join("graph_example.log")).ok();

    let mut generator = ConceptGenerator::new(config, llm, logger);

    // Create a sample graph
    let graph = create_sample_graph();
    generator.load_graph(graph)?;

    let input_file = temp_dir.path().join("sample_nodes.csv");
    let output_folder = temp_dir.path().join("graph_output");

    let stats = generator.generate_concept(
        &input_file,
        &output_folder,
        "concepts_with_context.csv",
        None,
        false,
    ).await?;

    println!("🕸️ Graph-Enhanced Processing:");
    println!("   - Context extraction enabled");
    println!("   - Nodes processed: {}", stats.total_nodes_processed);
    println!("   - Processing time: {}ms", stats.processing_time_ms);

    Ok(())
}

/// Example 3: Streaming concept generation for large datasets
async fn streaming_concept_generation_example(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger sample file for streaming
    let large_input_file = temp_dir.path().join("large_nodes.csv");
    create_large_sample_data(&large_input_file)?;

    let config = ProcessingConfig {
        batch_size: 5,
        max_workers: 3,
        ..Default::default()
    };

    let llm = Arc::new(MockLlmGenerator::new(20));

    let mut streaming_generator = StreamingConceptGenerator::new(
        config,
        llm,
        &large_input_file,
        None,
    )?;

    let output_folder = temp_dir.path().join("streaming_output");

    let stats = streaming_generator.process_streaming(
        &output_folder,
        "streaming_concepts",
        true,
    ).await?;

    println!("🌊 Streaming Processing:");
    println!("   - Memory-efficient processing");
    println!("   - Nodes processed: {}", stats.total_nodes_processed);
    println!("   - Processing time: {}ms", stats.processing_time_ms);

    Ok(())
}

/// Example 4: Sharded processing for distributed computation
async fn sharded_processing_example(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let config = ProcessingConfig {
        batch_size: 3,
        max_workers: 2,
        ..Default::default()
    };

    let llm = Arc::new(MockLlmGenerator::new(25));

    // Process with 2 shards
    let num_shards = 2;
    let mut all_stats = Vec::new();

    for shard_idx in 0..num_shards {
        let shard_config = ShardConfig {
            shard_idx,
            num_shards,
            shuffle_data: true,
        };

        let logger = ProcessingLogger::new(
            temp_dir.path().join(format!("shard_{}.log", shard_idx))
        ).ok();

        let mut generator = ConceptGenerator::new(config.clone(), llm.clone(), logger);

        let input_file = temp_dir.path().join("sample_nodes.csv");
        let output_folder = temp_dir.path().join("sharded_output");

        let stats = generator.generate_concept(
            &input_file,
            &output_folder,
            "concepts.csv",
            Some(shard_config),
            false,
        ).await?;

        all_stats.push(stats);
        println!("🔀 Shard {} completed: {} nodes", shard_idx, all_stats[shard_idx].total_nodes_processed);
    }

    let total_nodes: usize = all_stats.iter().map(|s| s.total_nodes_processed).sum();
    let total_time: u128 = all_stats.iter().map(|s| s.processing_time_ms).max().unwrap_or(0);

    println!("🏁 Sharded Processing Summary:");
    println!("   - Total shards: {}", num_shards);
    println!("   - Total nodes processed: {}", total_nodes);
    println!("   - Max processing time: {}ms", total_time);

    Ok(())
}

/// Create a sample graph for context extraction
fn create_sample_graph() -> Graph {
    let mut graph = Graph::new();

    // Add nodes
    graph.add_node(GraphNode {
        id: "john_smith".to_string(),
        label: "John Smith".to_string(),
        properties: HashMap::new(),
    });

    graph.add_node(GraphNode {
        id: "apple_inc".to_string(),
        label: "Apple Inc".to_string(),
        properties: HashMap::new(),
    });

    graph.add_node(GraphNode {
        id: "ai_conference".to_string(),
        label: "AI Conference".to_string(),
        properties: HashMap::new(),
    });

    // Add edges
    graph.add_edge(GraphEdge {
        source: "john_smith".to_string(),
        target: "apple_inc".to_string(),
        relation: "works_at".to_string(),
        properties: HashMap::new(),
    }).expect("Failed to add edge");

    graph.add_edge(GraphEdge {
        source: "john_smith".to_string(),
        target: "ai_conference".to_string(),
        relation: "participates_in".to_string(),
        properties: HashMap::new(),
    }).expect("Failed to add edge");

    graph
}

/// Create a larger sample file for streaming example
fn create_large_sample_data(file_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(file_path)?;

    writeln!(file, "node,type")?;

    // Generate more sample data
    let entities = [
        "Microsoft", "Google", "Amazon", "Tesla", "Netflix", "Facebook", "Twitter", "LinkedIn",
        "artificial intelligence", "machine learning", "deep learning", "neural networks",
        "data science", "cloud computing", "quantum computing", "blockchain",
        "software engineering", "data engineering", "DevOps", "cybersecurity"
    ];

    let events = [
        "product launch", "conference presentation", "software release", "merger announcement",
        "training session", "data processing", "model training", "system deployment",
        "security audit", "performance optimization", "code review", "testing phase"
    ];

    let relations = [
        "implements", "uses", "creates", "manages", "processes", "optimizes",
        "secures", "deploys", "maintains", "develops", "analyzes", "integrates"
    ];

    for entity in &entities {
        writeln!(file, "{},entity", entity)?;
    }

    for event in &events {
        writeln!(file, "{},event", event)?;
    }

    for relation in &relations {
        writeln!(file, "{},relation", relation)?;
    }

    println!("📝 Created large sample data with {} rows", entities.len() + events.len() + relations.len() + 1);
    Ok(())
}