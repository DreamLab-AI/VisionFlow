//! Main binary for testing concept generation functionality

use kg_construction::{
    concept_generation::ConceptGenerator,
    llm_integration::MockLlmGenerator,
    statistics::ProcessingLogger,
    types::ProcessingConfig,
    prelude::*,
};
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 AutoSchema KG Rust - Concept Generation Demo");

    // Create temporary directory for demo
    let temp_dir = TempDir::new().map_err(|e| KgConstructionError::Io(e))?;
    println!("📁 Working directory: {:?}", temp_dir.path());

    // Create sample input data
    let input_file = create_sample_input(&temp_dir)?;
    println!("📝 Created sample input with test data");

    // Configure processing
    let config = ProcessingConfig {
        batch_size: 3,
        max_workers: 2,
        language: "en".to_string(),
        ..Default::default()
    };

    // Create mock LLM with realistic delay
    let llm = Arc::new(MockLlmGenerator::new(100));
    println!("🤖 Initialized mock LLM generator");

    // Set up logging
    let log_file = temp_dir.path().join("concept_generation.log");
    let logger = ProcessingLogger::new(&log_file).ok();
    println!("📊 Initialized logging to: {:?}", log_file);

    // Create concept generator
    let mut generator = ConceptGenerator::new(config, llm, logger);
    println!("⚙️ Created concept generator");

    // Generate concepts
    println!("\n🔄 Starting concept generation...");
    let start_time = std::time::Instant::now();

    let stats = generator.generate_concept(
        &input_file,
        temp_dir.path().join("output"),
        "generated_concepts.csv",
        None,
        true, // Record token usage
    ).await?;

    let elapsed = start_time.elapsed();
    println!("✅ Concept generation completed in {:?}", elapsed);

    // Display results
    println!("\n📈 Final Statistics:");
    println!("   Total nodes processed: {}", stats.total_nodes_processed);
    println!("   Total batches processed: {}", stats.total_batches_processed);
    println!("   Entities processed: {}", stats.entities_processed);
    println!("   Events processed: {}", stats.events_processed);
    println!("   Relations processed: {}", stats.relations_processed);
    println!("   Unique concepts generated: {}", stats.unique_concepts_generated);
    println!("   Errors encountered: {}", stats.errors_encountered);
    println!("   Processing time: {}ms", stats.processing_time_ms);

    // Show concepts by type
    if !stats.concepts_by_type.is_empty() {
        println!("\n🏷️ Concepts by Type:");
        for (concept_type, count) in &stats.concepts_by_type {
            println!("   {}: {}", concept_type, count);
        }
    }

    // Check output file
    let output_file = temp_dir.path().join("output").join("generated_concepts.csv");
    if output_file.exists() {
        let content = std::fs::read_to_string(&output_file)
            .map_err(|e| KgConstructionError::Io(e))?;
        let line_count = content.lines().count();
        println!("\n📄 Output file created:");
        println!("   Path: {:?}", output_file);
        println!("   Lines: {}", line_count);

        // Show first few lines as preview
        let preview_lines: Vec<&str> = content.lines().take(5).collect();
        println!("   Preview:");
        for line in preview_lines {
            println!("     {}", line);
        }
        if content.lines().count() > 5 {
            println!("     ... ({} more lines)", line_count - 5);
        }
    }

    // Display performance metrics
    let throughput = if stats.processing_time_ms > 0 {
        (stats.total_nodes_processed as f64) / (stats.processing_time_ms as f64 / 1000.0)
    } else {
        0.0
    };

    println!("\n⚡ Performance Metrics:");
    println!("   Throughput: {:.2} nodes/second", throughput);
    println!("   Average batch time: {:.2}ms",
        if stats.total_batches_processed > 0 {
            stats.processing_time_ms as f64 / stats.total_batches_processed as f64
        } else { 0.0 });

    if stats.errors_encountered > 0 {
        println!("   Error rate: {:.2}%",
            (stats.errors_encountered as f64 / stats.total_nodes_processed as f64) * 100.0);
    } else {
        println!("   Error rate: 0.00%");
    }

    println!("\n🎉 Demo completed successfully!");
    println!("   Log file: {:?}", log_file);
    println!("   Output directory: {:?}", temp_dir.path().join("output"));

    Ok(())
}

/// Create sample input data for demonstration
fn create_sample_input(temp_dir: &TempDir) -> Result<std::path::PathBuf> {
    let input_file = temp_dir.path().join("sample_nodes.csv");
    let mut file = std::fs::File::create(&input_file)
        .map_err(|e| KgConstructionError::Io(e))?;

    writeln!(file, "node,type").map_err(|e| KgConstructionError::Io(e))?;

    // Sample entities
    writeln!(file, "artificial intelligence,entity").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "machine learning,entity").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "neural networks,entity").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "data science,entity").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "Python programming,entity").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "software engineering,entity").map_err(|e| KgConstructionError::Io(e))?;

    // Sample events
    writeln!(file, "model training,event").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "data preprocessing,event").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "feature extraction,event").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "hyperparameter tuning,event").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "model evaluation,event").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "deployment process,event").map_err(|e| KgConstructionError::Io(e))?;

    // Sample relations
    writeln!(file, "implements,relation").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "requires,relation").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "produces,relation").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "optimizes,relation").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "depends_on,relation").map_err(|e| KgConstructionError::Io(e))?;
    writeln!(file, "enhances,relation").map_err(|e| KgConstructionError::Io(e))?;

    Ok(input_file)
}