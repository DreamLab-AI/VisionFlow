//! Test to trigger full GitHub sync from multi-ontology branch
//!
//! NOTE: This test is disabled because Neo4jAdapter::new() and Neo4jOntologyRepository::new()
//! take a single config struct, not separate string arguments for uri, user, password.
//!
//! To re-enable:
//! 1. Update constructor calls to use the proper config structs
//! 2. Uncomment the code below

/*
use std::sync::Arc;
use tokio;

#[tokio::test]
async fn trigger_full_sync_test() {
    // Set environment variables
    std::env::set_var("FORCE_FULL_SYNC", "1");
    std::env::set_var("GITHUB_BRANCH", "multi-ontology");

    println!("🔄 Starting full sync from multi-ontology branch...");
    println!("Environment: FORCE_FULL_SYNC={}", std::env::var("FORCE_FULL_SYNC").unwrap_or_default());
    println!("Environment: GITHUB_BRANCH={}", std::env::var("GITHUB_BRANCH").unwrap_or_default());

    // Initialize services
    let config = webxr::services::github::GitHubConfig::from_env()
        .expect("Failed to load GitHub config");

    println!("✓ GitHub config loaded:");
    println!("  Owner: {}", config.owner);
    println!("  Repo: {}", config.repo);
    println!("  Branch: {}", config.branch);
    println!("  Base Path: {}", config.base_path);

    // Initialize Neo4j connection
    let neo4j_uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string());
    let neo4j_user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
    let neo4j_password = std::env::var("NEO4J_PASSWORD").expect("NEO4J_PASSWORD required");

    let neo4j_adapter = Arc::new(
        webxr::adapters::neo4j_adapter::Neo4jAdapter::new(
            &neo4j_uri,
            &neo4j_user,
            &neo4j_password,
        ).await.expect("Failed to create Neo4j adapter")
    );

    let ontology_repo = Arc::new(
        webxr::adapters::neo4j_ontology_repository::Neo4jOntologyRepository::new(
            &neo4j_uri,
            &neo4j_user,
            &neo4j_password,
        ).await.expect("Failed to create ontology repository")
    );

    println!("✓ Neo4j connections established");

    // Initialize GitHub client
    let settings = Arc::new(tokio::sync::RwLock::new(
        webxr::config::AppFullSettings::default()
    ));

    let github_client = Arc::new(
        webxr::services::github::GitHubClient::new(config, settings.clone())
            .await
            .expect("Failed to create GitHub client")
    );

    let enhanced_api = Arc::new(
        webxr::services::github::content_enhanced::EnhancedContentAPI::new(github_client)
    );

    println!("✓ GitHub API client initialized");

    // Create sync service
    let sync_service = webxr::services::github_sync_service::GitHubSyncService::new(
        enhanced_api,
        neo4j_adapter,
        ontology_repo,
    );

    println!("✓ GitHub Sync Service created");
    println!("\n🚀 Starting synchronization...\n");

    // Run sync
    match sync_service.sync_graphs().await {
        Ok(stats) => {
            println!("\n✅ Sync completed successfully!");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Total files: {}", stats.total_files);
            println!("  KG files processed: {}", stats.kg_files_processed);
            println!("  Ontology files processed: {}", stats.ontology_files_processed);
            println!("  Skipped files: {}", stats.skipped_files);
            println!("  Total nodes: {}", stats.total_nodes);
            println!("  Total edges: {}", stats.total_edges);
            println!("  Duration: {:?}", stats.duration);
            if !stats.errors.is_empty() {
                println!("  Errors: {}", stats.errors.len());
                for error in &stats.errors {
                    println!("    - {}", error);
                }
            }
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }
        Err(e) => {
            eprintln!("\n❌ Sync failed: {}", e);
            panic!("Sync failed");
        }
    }
}
*/
