// src/bin/sync_github.rs
//! GitHub sync binary - pulls all markdown files from GitHub and syncs to Neo4j
//! Uses GitHubSyncService::sync_graphs() for a full remote pull (no local baseline needed)

use std::sync::Arc;
use tokio::sync::RwLock;
use webxr::adapters::neo4j_adapter::{Neo4jAdapter, Neo4jConfig};
use webxr::adapters::neo4j_ontology_repository::{Neo4jOntologyRepository, Neo4jOntologyConfig};
use webxr::config::AppFullSettings;
use webxr::services::github::api::GitHubClient;
use webxr::services::github::config::GitHubConfig;
use webxr::services::github::content_enhanced::EnhancedContentAPI;
use webxr::services::github_sync_service::GitHubSyncService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting full GitHub sync (no local baseline)");

    dotenvy::dotenv().ok();

    // Neo4j config
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| {
        if std::env::var("ALLOW_INSECURE_DEFAULTS").is_ok() {
            eprintln!("WARNING: Using insecure default password.");
            "password".to_string()
        } else {
            panic!("NEO4J_PASSWORD must be set. Use ALLOW_INSECURE_DEFAULTS=1 for dev.");
        }
    });
    let database = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());

    let neo4j_config = Neo4jConfig {
        uri: uri.clone(),
        user: user.clone(),
        password: password.clone(),
        database: Some(database.clone()),
        max_connections: 100,
        query_timeout_secs: 30,
        connection_timeout_secs: 30,
    };

    let ontology_config = Neo4jOntologyConfig {
        uri,
        user,
        password,
        database: Some(database),
    };

    log::info!("Connecting to Neo4j...");

    let kg_repo = Arc::new(Neo4jAdapter::new(neo4j_config).await?);
    let onto_repo = Arc::new(Neo4jOntologyRepository::new(ontology_config).await?);

    // GitHub client
    let github_config = GitHubConfig::from_env()?;
    let settings = Arc::new(RwLock::new(AppFullSettings::default()));
    let github_client = Arc::new(GitHubClient::new(github_config, settings).await?);
    let content_api = Arc::new(EnhancedContentAPI::new(github_client));

    // Create sync service (constructs its own enrichment/reasoning internally)
    let sync_service = GitHubSyncService::new(content_api, kg_repo, onto_repo);

    log::info!("Starting sync from GitHub...");

    let stats = sync_service.sync_graphs().await?;

    println!("\nSync complete!");
    println!("{}", "=".repeat(50));
    println!("  Total files found:        {}", stats.total_files);
    println!("  KG files processed:       {}", stats.kg_files_processed);
    println!("  Ontology files processed: {}", stats.ontology_files_processed);
    println!("  Skipped (unchanged):      {}", stats.skipped_files);
    println!("  Total nodes:              {}", stats.total_nodes);
    println!("  Total edges:              {}", stats.total_edges);
    println!("  Duration:                 {:?}", stats.duration);
    println!("{}", "=".repeat(50));

    if !stats.errors.is_empty() {
        println!("\nErrors ({}):", stats.errors.len());
        for (i, error) in stats.errors.iter().enumerate().take(10) {
            println!("  {}. {}", i + 1, error);
        }
        if stats.errors.len() > 10 {
            println!("  ... and {} more", stats.errors.len() - 10);
        }
    }

    Ok(())
}
