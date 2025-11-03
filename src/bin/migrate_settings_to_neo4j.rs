// src/bin/migrate_settings_to_neo4j.rs
//! Settings Migration Script: SQLite → Neo4j
//!
//! Migrates all settings from SQLite database to Neo4j graph database.
//!
//! Usage:
//!   cargo run --bin migrate_settings_to_neo4j -- [OPTIONS]
//!
//! Options:
//!   --sqlite-path <PATH>    Path to SQLite database (default: data/unified.db)
//!   --neo4j-uri <URI>       Neo4j URI (default: bolt://localhost:7687)
//!   --neo4j-user <USER>     Neo4j username (default: neo4j)
//!   --neo4j-pass <PASS>     Neo4j password (default: from NEO4J_PASSWORD env)
//!   --dry-run               Show what would be migrated without making changes
//!   --verbose               Enable verbose logging

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn, error, debug};
use tracing_subscriber;

use webxr::adapters::sqlite_settings_repository::SqliteSettingsRepository;
use webxr::adapters::neo4j_settings_repository::{Neo4jSettingsRepository, Neo4jSettingsConfig};
use webxr::ports::settings_repository::{SettingsRepository, SettingValue};

/// Migration statistics
#[derive(Debug, Default)]
struct MigrationStats {
    total_settings: usize,
    migrated_settings: usize,
    failed_settings: usize,
    physics_profiles: usize,
    errors: Vec<String>,
}

impl MigrationStats {
    fn print_summary(&self) {
        println!("\n{'='}=50");
        println!("Migration Summary");
        println!("{'='}=50");
        println!("Total settings found:     {}", self.total_settings);
        println!("Successfully migrated:    {}", self.migrated_settings);
        println!("Failed migrations:        {}", self.failed_settings);
        println!("Physics profiles:         {}", self.physics_profiles);
        println!("{'='}=50");

        if !self.errors.is_empty() {
            println!("\nErrors encountered:");
            for (i, error) in self.errors.iter().enumerate().take(10) {
                println!("  {}. {}", i + 1, error);
            }
            if self.errors.len() > 10 {
                println!("  ... and {} more errors", self.errors.len() - 10);
            }
        }
    }
}

/// Migration configuration
struct MigrationConfig {
    sqlite_path: String,
    neo4j_uri: String,
    neo4j_user: String,
    neo4j_password: String,
    dry_run: bool,
    verbose: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            sqlite_path: "data/unified.db".to_string(),
            neo4j_uri: std::env::var("NEO4J_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            neo4j_user: std::env::var("NEO4J_USER")
                .unwrap_or_else(|_| "neo4j".to_string()),
            neo4j_password: std::env::var("NEO4J_PASSWORD")
                .unwrap_or_else(|_| "password".to_string()),
            dry_run: false,
            verbose: false,
        }
    }
}

impl MigrationConfig {
    fn from_args() -> Self {
        let mut config = Self::default();
        let args: Vec<String> = std::env::args().collect();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--sqlite-path" => {
                    if i + 1 < args.len() {
                        config.sqlite_path = args[i + 1].clone();
                        i += 2;
                    } else {
                        eprintln!("Error: --sqlite-path requires a value");
                        std::process::exit(1);
                    }
                }
                "--neo4j-uri" => {
                    if i + 1 < args.len() {
                        config.neo4j_uri = args[i + 1].clone();
                        i += 2;
                    } else {
                        eprintln!("Error: --neo4j-uri requires a value");
                        std::process::exit(1);
                    }
                }
                "--neo4j-user" => {
                    if i + 1 < args.len() {
                        config.neo4j_user = args[i + 1].clone();
                        i += 2;
                    } else {
                        eprintln!("Error: --neo4j-user requires a value");
                        std::process::exit(1);
                    }
                }
                "--neo4j-pass" => {
                    if i + 1 < args.len() {
                        config.neo4j_password = args[i + 1].clone();
                        i += 2;
                    } else {
                        eprintln!("Error: --neo4j-pass requires a value");
                        std::process::exit(1);
                    }
                }
                "--dry-run" => {
                    config.dry_run = true;
                    i += 1;
                }
                "--verbose" => {
                    config.verbose = true;
                    i += 1;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    eprintln!("Unknown argument: {}", args[i]);
                    print_help();
                    std::process::exit(1);
                }
            }
        }

        config
    }
}

fn print_help() {
    println!("Settings Migration Script: SQLite → Neo4j");
    println!();
    println!("USAGE:");
    println!("    migrate_settings_to_neo4j [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --sqlite-path <PATH>    Path to SQLite database (default: data/unified.db)");
    println!("    --neo4j-uri <URI>       Neo4j URI (default: bolt://localhost:7687)");
    println!("    --neo4j-user <USER>     Neo4j username (default: neo4j)");
    println!("    --neo4j-pass <PASS>     Neo4j password (default: from NEO4J_PASSWORD env)");
    println!("    --dry-run               Show what would be migrated without making changes");
    println!("    --verbose               Enable verbose logging");
    println!("    --help, -h              Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    # Dry run to see what would be migrated");
    println!("    migrate_settings_to_neo4j --dry-run --verbose");
    println!();
    println!("    # Migrate with custom paths");
    println!("    migrate_settings_to_neo4j --sqlite-path /path/to/db.db --neo4j-uri bolt://localhost:7687");
}

async fn migrate_settings(config: MigrationConfig) -> Result<MigrationStats> {
    let mut stats = MigrationStats::default();

    info!("Starting settings migration");
    info!("SQLite path: {}", config.sqlite_path);
    info!("Neo4j URI: {}", config.neo4j_uri);
    info!("Dry run: {}", config.dry_run);

    // Initialize SQLite repository
    info!("Connecting to SQLite database...");
    let sqlite_repo = SqliteSettingsRepository::new(&config.sqlite_path)
        .context("Failed to create SQLite repository")?;
    info!("✅ Connected to SQLite");

    // Initialize Neo4j repository
    info!("Connecting to Neo4j database...");
    let neo4j_config = Neo4jSettingsConfig {
        uri: config.neo4j_uri.clone(),
        user: config.neo4j_user.clone(),
        password: config.neo4j_password.clone(),
        database: None,
        fetch_size: 500,
        max_connections: 10,
    };

    let neo4j_repo = Neo4jSettingsRepository::new(neo4j_config)
        .await
        .context("Failed to create Neo4j repository")?;
    info!("✅ Connected to Neo4j");

    // Health checks
    info!("Performing health checks...");
    if !sqlite_repo.health_check().await? {
        error!("SQLite health check failed");
        anyhow::bail!("SQLite database is not healthy");
    }
    if !neo4j_repo.health_check().await? {
        error!("Neo4j health check failed");
        anyhow::bail!("Neo4j database is not healthy");
    }
    info!("✅ Health checks passed");

    // Migrate individual settings
    info!("Migrating individual settings...");
    let setting_keys = sqlite_repo.list_settings(None).await?;
    stats.total_settings = setting_keys.len();

    info!("Found {} settings to migrate", stats.total_settings);

    for key in &setting_keys {
        debug!("Migrating setting: {}", key);

        match sqlite_repo.get_setting(key).await {
            Ok(Some(value)) => {
                if config.dry_run {
                    info!("[DRY RUN] Would migrate: {} = {:?}", key, value);
                    stats.migrated_settings += 1;
                } else {
                    match neo4j_repo.set_setting(key, value, Some(&format!("Migrated from SQLite"))).await {
                        Ok(_) => {
                            debug!("✅ Migrated: {}", key);
                            stats.migrated_settings += 1;
                        }
                        Err(e) => {
                            warn!("❌ Failed to migrate {}: {}", key, e);
                            stats.failed_settings += 1;
                            stats.errors.push(format!("Setting '{}': {}", key, e));
                        }
                    }
                }
            }
            Ok(None) => {
                warn!("Setting {} not found in SQLite", key);
            }
            Err(e) => {
                warn!("Failed to read setting {}: {}", key, e);
                stats.failed_settings += 1;
                stats.errors.push(format!("Read '{}': {}", key, e));
            }
        }
    }

    // Migrate physics profiles
    info!("Migrating physics profiles...");
    let physics_profiles = sqlite_repo.list_physics_profiles().await?;
    stats.physics_profiles = physics_profiles.len();

    info!("Found {} physics profiles to migrate", stats.physics_profiles);

    for profile_name in &physics_profiles {
        debug!("Migrating physics profile: {}", profile_name);

        match sqlite_repo.get_physics_settings(profile_name).await {
            Ok(settings) => {
                if config.dry_run {
                    info!("[DRY RUN] Would migrate physics profile: {}", profile_name);
                } else {
                    match neo4j_repo.save_physics_settings(profile_name, &settings).await {
                        Ok(_) => {
                            debug!("✅ Migrated physics profile: {}", profile_name);
                        }
                        Err(e) => {
                            warn!("❌ Failed to migrate physics profile {}: {}", profile_name, e);
                            stats.errors.push(format!("Physics profile '{}': {}", profile_name, e));
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to read physics profile {}: {}", profile_name, e);
                stats.errors.push(format!("Read physics profile '{}': {}", profile_name, e));
            }
        }
    }

    // Migrate full settings snapshot
    info!("Migrating full settings snapshot...");
    match sqlite_repo.load_all_settings().await {
        Ok(Some(full_settings)) => {
            if config.dry_run {
                info!("[DRY RUN] Would migrate full settings snapshot");
            } else {
                match neo4j_repo.save_all_settings(&full_settings).await {
                    Ok(_) => {
                        info!("✅ Migrated full settings snapshot");
                    }
                    Err(e) => {
                        warn!("❌ Failed to migrate full settings: {}", e);
                        stats.errors.push(format!("Full settings: {}", e));
                    }
                }
            }
        }
        Ok(None) => {
            info!("No full settings snapshot found in SQLite");
        }
        Err(e) => {
            warn!("Failed to load full settings: {}", e);
            stats.errors.push(format!("Load full settings: {}", e));
        }
    }

    if config.dry_run {
        info!("\n{'='}=50");
        info!("DRY RUN COMPLETE - No changes were made");
        info!("{'='}=50");
    } else {
        info!("\n{'='}=50");
        info!("MIGRATION COMPLETE");
        info!("{'='}=50");
    }

    Ok(stats)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse configuration from arguments
    let config = MigrationConfig::from_args();

    // Initialize logging
    let log_level = if config.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::from(log_level.parse::<tracing::Level>().unwrap()))
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    println!("\n{'='}=50");
    println!("Settings Migration: SQLite → Neo4j");
    println!("{'='}=50\n");

    // Run migration
    let result = migrate_settings(config).await;

    match result {
        Ok(stats) => {
            stats.print_summary();

            if stats.failed_settings > 0 {
                println!("\n⚠️  Migration completed with {} errors", stats.failed_settings);
                std::process::exit(1);
            } else {
                println!("\n✅ Migration completed successfully!");
                std::process::exit(0);
            }
        }
        Err(e) => {
            error!("Migration failed: {}", e);
            eprintln!("\n❌ Migration failed: {}", e);
            std::process::exit(1);
        }
    }
}
