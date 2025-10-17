//! Ontology System Initialization
//!
//! Handles complete initialization of the ontology system including:
//! - SQLite database setup
//! - GitHub repository scanning
//! - Ontology data download and parsing
//! - Database population
//! - Integration with graph visualization

use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::Duration;
use log::{info, error, warn, debug};

#[cfg(feature = "ontology")]
use crate::services::database_service::DatabaseService;
#[cfg(feature = "ontology")]
use crate::services::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig};
#[cfg(feature = "ontology")]
use crate::services::ontology_storage::OntologyStorage;
#[cfg(feature = "ontology")]
use crate::services::ontology_sync::OntologySync;

/// Initialize the complete ontology system
#[cfg(feature = "ontology")]
pub async fn initialize_ontology_system() -> Result<Arc<DatabaseService>, String> {
    info!("🔮 Initializing Ontology System");

    // 1. Setup database path
    let db_path = std::env::var("DATA_ROOT")
        .unwrap_or_else(|_| "/app/data".to_string());
    let db_file = PathBuf::from(db_path).join("ontology_db.sqlite3");

    info!("📊 Database path: {}", db_file.display());

    // 2. Initialize database with schema
    let db_service = match DatabaseService::new(&db_file) {
        Ok(service) => {
            info!("✅ Database initialized successfully");
            Arc::new(service)
        }
        Err(e) => {
            error!("❌ Failed to initialize database: {}", e);
            return Err(format!("Database initialization failed: {}", e));
        }
    };

    // 3. Initialize database schema
    match db_service.initialize_schema() {
        Ok(_) => info!("✅ Database schema created/verified"),
        Err(e) => {
            error!("❌ Schema initialization failed: {}", e);
            return Err(format!("Schema initialization failed: {}", e));
        }
    }

    // 4. Run settings migration from YAML to SQLite (if not already done)
    let migration_service = crate::services::settings_migration::SettingsMigration::new(Arc::clone(&db_service));
    if !migration_service.is_migrated() {
        info!("⚙️  Running settings migration from YAML to SQLite");
        match migration_service.migrate_from_yaml_files() {
            Ok(result) => {
                info!("✅ Settings migration completed successfully");
                info!("   📝 Settings migrated: {}", result.settings_migrated);
                info!("   ⚡ Physics profiles: {}", result.physics_profiles_migrated);
                info!("   🔧 Dev config params: {}", result.dev_config_params_migrated);
                info!("   ⏱️  Duration: {:?}", result.duration);
                if !result.errors.is_empty() {
                    warn!("   ⚠️  Errors: {} (migration continues)", result.errors.len());
                }
            }
            Err(e) => {
                warn!("⚠️  Settings migration failed (continuing with defaults): {}", e);
            }
        }
    } else {
        info!("✅ Settings already migrated, skipping");
    }

    // 5. Initialize DevConfig from database
    info!("🔧 Initializing DevConfig from database");
    match crate::config::dev_config::DevConfig::initialize(Arc::clone(&db_service)) {
        Ok(_) => {
            info!("✅ DevConfig initialized successfully from database");
        }
        Err(e) => {
            warn!("⚠️  DevConfig initialization failed (using defaults): {}", e);
        }
    }

    // 6. Spawn background task for ontology download
    let db_service_clone = Arc::clone(&db_service);
    tokio::spawn(async move {
        // Wait for server to fully start
        info!("⏳ Waiting 10 seconds for server initialization before starting ontology sync");
        tokio::time::sleep(Duration::from_secs(10)).await;

        if let Err(e) = download_and_process_ontology(db_service_clone).await {
            error!("❌ Ontology download/processing failed: {}", e);
        }
    });

    Ok(db_service)
}

/// Download and process ontology data from GitHub
#[cfg(feature = "ontology")]
async fn download_and_process_ontology(db_service: Arc<DatabaseService>) -> Result<(), String> {
    info!("🚀 Starting ontology download and processing");

    // 1. Create downloader with configuration from environment
    let github_token = std::env::var("GITHUB_TOKEN")
        .map_err(|_| "GITHUB_TOKEN not set in environment".to_string())?;

    let config = OntologyDownloaderConfig {
        repo_owner: std::env::var("GITHUB_OWNER")
            .unwrap_or_else(|_| "jjohare".to_string()),
        repo_name: std::env::var("GITHUB_REPO")
            .unwrap_or_else(|_| "logseq".to_string()),
        base_path: std::env::var("GITHUB_BASE_PATH")
            .unwrap_or_else(|_| "mainKnowledgeGraph/pages".to_string()),
        github_token,
        max_retries: 3,
        initial_retry_delay_ms: 1000,
        max_retry_delay_ms: 30000,
        request_timeout_secs: 30,
        respect_rate_limits: true,
    };

    // 2. Create storage (separate database for ontology blocks)
    let ontology_db_path = std::env::var("DATA_ROOT")
        .unwrap_or_else(|_| "/app/data".to_string());
    let ontology_db_file = PathBuf::from(ontology_db_path).join("ontology_blocks.sqlite3");

    let storage = OntologyStorage::new(&ontology_db_file)
        .map_err(|e| format!("Failed to create storage: {}", e))?;

    // 3. Create sync orchestrator
    let sync = OntologySync::new(config, storage)
        .map_err(|e| format!("Failed to create sync: {}", e))?;

    // 4. Execute sync with progress tracking
    info!("📥 Beginning ontology sync from GitHub");

    match sync.sync().await {
        Ok(result) => {
            info!("✅ Ontology sync completed successfully!");
            info!("   📥 Blocks downloaded: {}", result.blocks_downloaded);
            info!("   📝 Blocks saved: {}", result.blocks_saved);
            info!("   ⏱️  Duration: {} seconds", result.duration_seconds);

            if !result.errors.is_empty() {
                warn!("⚠️  {} errors occurred during sync:", result.errors.len());
                for (idx, error) in result.errors.iter().enumerate().take(5) {
                    warn!("   {}. {}", idx + 1, error);
                }
                if result.errors.len() > 5 {
                    warn!("   ... and {} more errors", result.errors.len() - 5);
                }
            }

            Ok(())
        }
        Err(e) => {
            error!("❌ Ontology sync failed: {}", e);
            Err(format!("Ontology sync failed: {}", e))
        }
    }
}

/// No-op initialization when ontology feature is disabled
#[cfg(not(feature = "ontology"))]
pub async fn initialize_ontology_system() -> Result<(), String> {
    debug!("Ontology feature not enabled, skipping initialization");
    Ok(())
}
