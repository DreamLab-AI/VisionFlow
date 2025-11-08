// src/bin/sync_local.rs
//! Local file sync binary - syncs local baseline with GitHub delta updates

use std::sync::Arc;
use webxr::adapters::neo4j_knowledge_graph_repository::Neo4jKnowledgeGraphRepository;
use webxr::adapters::neo4j_ontology_repository::Neo4jOntologyRepository;
use webxr::services::github::api::GitHubClient;
use webxr::services::github::content_enhanced::EnhancedContentAPI;
use webxr::services::local_file_sync_service::LocalFileSyncService;
use webxr::services::ontology_enrichment_service::OntologyEnrichmentService;
use webxr::utils::database::neo4j_connection::establish_neo4j_connection;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("🚀 Starting local file sync with GitHub delta updates");

    // Load environment variables
    dotenvy::dotenv().ok();

    // Establish Neo4j connection
    let neo4j_pool = establish_neo4j_connection().await?;
    log::info!("✅ Connected to Neo4j");

    // Initialize repositories
    let kg_repo = Arc::new(Neo4jKnowledgeGraphRepository::new(neo4j_pool.clone()));
    let onto_repo = Arc::new(Neo4jOntologyRepository::new(neo4j_pool.clone()));

    // Initialize GitHub client
    let github_owner = std::env::var("GITHUB_OWNER").unwrap_or_else(|_| "jjohare".to_string());
    let github_repo = std::env::var("GITHUB_REPO").unwrap_or_else(|_| "logseq".to_string());
    let github_branch =
        std::env::var("GITHUB_BRANCH").unwrap_or_else(|_| "main".to_string());
    let github_token = std::env::var("GITHUB_TOKEN")
        .expect("GITHUB_TOKEN environment variable is required");
    let github_base_path = std::env::var("GITHUB_BASE_PATH")
        .unwrap_or_else(|_| "mainKnowledgeGraph/pages".to_string());

    let github_client = Arc::new(GitHubClient::new(
        github_owner,
        github_repo,
        github_branch,
        github_token,
        github_base_path,
    ));

    let content_api = Arc::new(EnhancedContentAPI::new(github_client));

    // Initialize ontology enrichment service
    let enrichment_service = Arc::new(OntologyEnrichmentService::new(
        onto_repo.clone(),
        neo4j_pool.clone(),
    ));

    // Create sync service
    let sync_service = LocalFileSyncService::new(
        content_api,
        kg_repo,
        onto_repo,
        enrichment_service,
    );

    log::info!("🔄 Starting sync operation...");

    // Run sync
    let stats = sync_service.sync_with_github_delta().await?;

    // Display results
    println!("\n✅ Sync complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Statistics:");
    println!("  • Total files scanned:      {}", stats.total_files);
    println!(
        "  • Files synced from local:  {}",
        stats.files_synced_from_local
    );
    println!(
        "  • Files updated from GitHub: {}",
        stats.files_updated_from_github
    );
    println!(
        "  • Knowledge graph files:    {}",
        stats.kg_files_processed
    );
    println!(
        "  • Ontology files processed: {}",
        stats.ontology_files_processed
    );
    println!("  • Skipped files:            {}", stats.skipped_files);
    println!("  • Duration:                 {:?}", stats.duration);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if !stats.errors.is_empty() {
        println!("\n⚠️  Errors encountered ({}):", stats.errors.len());
        for (i, error) in stats.errors.iter().enumerate().take(10) {
            println!("  {}. {}", i + 1, error);
        }
        if stats.errors.len() > 10 {
            println!("  ... and {} more errors", stats.errors.len() - 10);
        }
    }

    log::info!("✅ Sync binary completed successfully");

    Ok(())
}
