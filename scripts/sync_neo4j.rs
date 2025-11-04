// scripts/sync_neo4j.rs
//! Neo4j Sync Script
//!
//! Synchronizes data from unified.db (SQLite) to Neo4j.
//! Can be run as:
//! - Full sync: Clears Neo4j and loads all data from SQLite
//! - Incremental sync: Only syncs nodes/edges modified since last sync
//!
//! Usage:
//! ```bash
//! cargo run --bin sync_neo4j -- [--full] [--dry-run]
//! ```

use anyhow::{Context, Result};
use log::{info, warn};
use std::sync::Arc;

// Import from main crate
use webxr::adapters::neo4j_adapter::{Neo4jAdapter, Neo4jConfig};
use webxr::ports::knowledge_graph_repository::KnowledgeGraphRepository;
// Note: This script is obsolete - Neo4j is now the primary database.
// unified_graph_repository was removed during SQLite → Neo4j migration.

#[derive(Debug)]
struct SyncOptions {
    full_sync: bool,
    dry_run: bool,
    db_path: String,
}

impl Default for SyncOptions {
    fn default() -> Self {
        Self {
            full_sync: false,
            dry_run: false,
            db_path: std::env::var("DATABASE_PATH")
                .unwrap_or_else(|_| "/app/data/unified.db".to_string()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut options = SyncOptions::default();

    for arg in &args[1..] {
        match arg.as_str() {
            "--full" => options.full_sync = true,
            "--dry-run" => options.dry_run = true,
            path if path.starts_with("--db=") => {
                options.db_path = path.trim_start_matches("--db=").to_string();
            }
            _ => {
                eprintln!("Unknown argument: {}", arg);
                print_usage();
                std::process::exit(1);
            }
        }
    }

    info!("🚀 Starting Neo4j sync");
    info!("   Mode: {}", if options.full_sync { "Full sync" } else { "Incremental sync" });
    info!("   Dry run: {}", options.dry_run);
    info!("   Database: {}", options.db_path);

    // MIGRATION SCRIPT OBSOLETE: UnifiedGraphRepository was removed during Neo4j migration.
    // This script is no longer functional as Neo4j is now the primary database.
    anyhow::bail!("This migration script is obsolete. Neo4j is now the primary database. Use direct Neo4j operations instead.");

    // Initialize SQLite repository (DELETED - kept for reference)
    // let sqlite_repo = UnifiedGraphRepository::new(&options.db_path)
    //     .context("Failed to initialize SQLite repository")?;
    // let sqlite_repo = Arc::new(sqlite_repo);

    // ALL CODE BELOW COMMENTED OUT - SCRIPT IS OBSOLETE
    // Initialize Neo4j adapter (DELETED)
    // let neo4j_config = Neo4jConfig::default();
    // let neo4j = Neo4jAdapter::new(neo4j_config)
    //     .await
    //     .context("Failed to initialize Neo4j adapter")?;
    // let neo4j = Arc::new(neo4j);

    // Check health (DELETED)
    // info!("🔍 Checking database health...");
    // let sqlite_ok = sqlite_repo.health_check().await?;
    // let neo4j_ok = neo4j.health_check().await?;

    // if !sqlite_ok {
    //     anyhow::bail!("SQLite health check failed");
    // }

    // if !neo4j_ok {
    //     anyhow::bail!("Neo4j health check failed");
    // }

    // info!("✅ Both databases healthy");

    // Load data from SQLite (DELETED)
    // info!("📦 Loading data from SQLite...");
    // let graph = sqlite_repo.load_graph().await?;

    // info!("   Nodes: {}", graph.nodes.len());
    // info!("   Edges: {}", graph.edges.len());

    // if options.dry_run {
    //     info!("🔄 Dry run mode - skipping actual sync");
    //     info!("   Would sync {} nodes and {} edges to Neo4j", graph.nodes.len(), graph.edges.len());
    //     return Ok(());
    // }

    // Clear Neo4j if full sync (DELETED)
    // if options.full_sync {
    //     info!("🗑️  Full sync mode - clearing Neo4j...");
    //     neo4j.clear_graph().await?;
    //     info!("✅ Neo4j cleared");
    // }

    // Sync nodes (DELETED)
    // info!("🔄 Syncing {} nodes to Neo4j...", graph.nodes.len());
    // let start_time = std::time::Instant::now();

    // let mut synced_nodes = 0;
    // let mut failed_nodes = 0;

    // for (idx, node) in graph.nodes.iter().enumerate() {
    //     if idx % 100 == 0 && idx > 0 {
    //         info!("   Progress: {}/{} nodes ({:.1}%)", idx, graph.nodes.len(), (idx as f32 / graph.nodes.len() as f32) * 100.0);
    //     }
    //
    //     match neo4j.add_node(node).await {
    //         Ok(_) => synced_nodes += 1,
    //         Err(e) => {
    //             warn!("⚠️  Failed to sync node {}: {}", node.id, e);
    //             failed_nodes += 1;
    //         }
    //     }
    // }
    //
    // let node_sync_time = start_time.elapsed();
    // info!("✅ Nodes synced: {} succeeded, {} failed in {:?}", synced_nodes, failed_nodes, node_sync_time);
    //
    // // Sync edges
    // info!("🔄 Syncing {} edges to Neo4j...", graph.edges.len());
    // let start_time = std::time::Instant::now();
    //
    // let mut synced_edges = 0;
    // let mut failed_edges = 0;
    //
    // for (idx, edge) in graph.edges.iter().enumerate() {
    //     if idx % 100 == 0 && idx > 0 {
    //         info!("   Progress: {}/{} edges ({:.1}%)", idx, graph.edges.len(), (idx as f32 / graph.edges.len() as f32) * 100.0);
    //     }
    //
    //     match neo4j.add_edge(edge).await {
    //         Ok(_) => synced_edges += 1,
    //         Err(e) => {
    //             warn!("⚠️  Failed to sync edge {}: {}", edge.id, e);
    //             failed_edges += 1;
    //         }
    //     }
    // }
    //
    // let edge_sync_time = start_time.elapsed();
    // info!("✅ Edges synced: {} succeeded, {} failed in {:?}", synced_edges, failed_edges, edge_sync_time);
    //
    // // Final statistics
    // info!("🎉 Sync completed!");
    // info!("   Total nodes: {} ({} synced, {} failed)", graph.nodes.len(), synced_nodes, failed_nodes);
    // info!("   Total edges: {} ({} synced, {} failed)", graph.edges.len(), synced_edges, failed_edges);
    // info!("   Total time: {:?}", node_sync_time + edge_sync_time);
    //
    // // Verify sync
    // info!("🔍 Verifying Neo4j statistics...");
    // let stats = neo4j.get_statistics().await?;
    // info!("   Neo4j node count: {}", stats.node_count);
    // info!("   Neo4j edge count: {}", stats.edge_count);
    // info!("   Average degree: {:.2}", stats.average_degree);
    //
    // if stats.node_count != synced_nodes {
    //     warn!("⚠️  Node count mismatch: expected {}, got {}", synced_nodes, stats.node_count);
    // }
    //
    // Ok(())
}

fn print_usage() {
    eprintln!("Usage: sync_neo4j [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --full          Perform full sync (clear Neo4j first)");
    eprintln!("  --dry-run       Show what would be synced without actually syncing");
    eprintln!("  --db=PATH       Path to unified.db (default: /app/data/unified.db)");
    eprintln!();
    eprintln!("Environment variables:");
    eprintln!("  NEO4J_URI       Neo4j connection URI (default: bolt://localhost:7687)");
    eprintln!("  NEO4J_USER      Neo4j username (default: neo4j)");
    eprintln!("  NEO4J_PASSWORD  Neo4j password (required)");
    eprintln!("  DATABASE_PATH   Path to unified.db (default: /app/data/unified.db)");
}
