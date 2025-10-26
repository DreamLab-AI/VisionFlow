// sync_local.rs - Standalone program to sync from local markdown files
use std::sync::Arc;

mod models;
mod ports;
mod adapters;
mod services;
mod utils;

use services::local_markdown_sync::LocalMarkdownSync;
use adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use ports::knowledge_graph_repository::KnowledgeGraphRepository;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== Local Markdown Sync ===");
    println!("Reading from: /app/data/markdown");
    println!("Writing to: data/knowledge_graph.db");

    // Initialize repository
    let repo = SqliteKnowledgeGraphRepository::new("data/knowledge_graph.db")
        .await
        .expect("Failed to initialize repository");
    let repo = Arc::new(repo);

    // Run local sync
    let syncer = LocalMarkdownSync::new();
    let result = syncer.sync_from_directory("/app/data/markdown")?;

    println!("\n=== Sync Results ===");
    println!("Total files: {}", result.total_files);
    println!("Processed files: {}", result.processed_files);
    println!("Skipped files: {}", result.skipped_files);
    println!("Nodes: {}", result.nodes.len());
    println!("Edges: {}", result.edges.len());

    // Save to database
    println!("\nSaving to database...");
    repo.save_graph(result.nodes, result.edges).await?;
    println!("✅ Database updated successfully");

    Ok(())
}
