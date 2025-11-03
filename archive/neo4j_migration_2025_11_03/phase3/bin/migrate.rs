//! Database migration CLI tool
//!
//! Commands:
//! - up: Apply all pending migrations
//! - down: Rollback the last migration
//! - status: Show current migration status
//! - create <name>: Create a new migration file

use chrono::Utc;
use rusqlite::Connection;
use std::path::PathBuf;
use std::process;

use webxr::migrations::{MigrationRunner, RollbackManager, VersionTracker};
use crate::utils::time;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];
    let result = match command.as_str() {
        "up" => cmd_up(),
        "down" => cmd_down(),
        "status" => cmd_status(),
        "create" => {
            if args.len() < 3 {
                eprintln!("Error: create command requires a migration name");
                eprintln!("Usage: migrate create <name>");
                process::exit(1);
            }
            cmd_create(&args[2])
        }
        "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage();
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn print_usage() {
    println!("Database Migration Tool");
    println!();
    println!("USAGE:");
    println!("    migrate <COMMAND>");
    println!();
    println!("COMMANDS:");
    println!("    up              Apply all pending migrations");
    println!("    down            Rollback the last migration");
    println!("    status          Show current migration status");
    println!("    create <name>   Create a new migration file");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help      Print help information");
}

fn get_database_path() -> PathBuf {
    std::env::var("DATABASE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/settings.db"))
}

fn get_migrations_dir() -> PathBuf {
    std::env::var("MIGRATIONS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("migrations"))
}

fn cmd_up() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_database_path();
    let migrations_dir = get_migrations_dir();

    println!("Database: {}", db_path.display());
    println!("Migrations: {}", migrations_dir.display());
    println!();

    let mut conn = Connection::open(&db_path)?;

    
    let tracker = VersionTracker::new(&conn);
    tracker.initialize()?;

    
    let runner = MigrationRunner::new(&migrations_dir);
    let count = runner.migrate_up(&mut conn)?;

    if count > 0 {
        println!("\n Successfully applied {} migration(s)", count);
    } else {
        println!(" Database is up to date");
    }

    Ok(())
}

fn cmd_down() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_database_path();
    let migrations_dir = get_migrations_dir();

    println!("Database: {}", db_path.display());
    println!("Migrations: {}", migrations_dir.display());
    println!();

    let mut conn = Connection::open(&db_path)?;

    
    let current = {
        let tracker = VersionTracker::new(&conn);
        tracker.current_version()?
    };

    if current.is_none() {
        println!("No migrations to rollback");
        return Ok(());
    }

    println!("Rolling back migration {}...", current.unwrap());

    RollbackManager::rollback_last(&mut conn, &migrations_dir)?;

    let new_version = {
        let tracker = VersionTracker::new(&conn);
        tracker.current_version()?
    };
    println!("\n Rolled back to version {}", new_version.unwrap_or(0));

    Ok(())
}

fn cmd_status() -> Result<(), Box<dyn std::error::Error>> {
    let db_path = get_database_path();
    let migrations_dir = get_migrations_dir();

    println!("Database: {}", db_path.display());
    println!("Migrations: {}", migrations_dir.display());
    println!();

    let conn = Connection::open(&db_path)?;

    
    let table_exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='schema_migrations'",
        [],
        |row| row.get(0),
    ).unwrap_or(false);

    if !table_exists {
        println!("Migration system not initialized");
        println!("Run 'migrate up' to initialize and apply migrations");
        return Ok(());
    }

    let tracker = VersionTracker::new(&conn);
    let current = tracker.current_version()?;

    println!("Current Version: {}", current.unwrap_or(0));
    println!();

    
    let applied = tracker.get_all_applied()?;

    if !applied.is_empty() {
        println!("Applied Migrations:");
        println!(
            "{:<8} {:<40} {:<20} {:<12}",
            "Version", "Name", "Applied At", "Time (ms)"
        );
        println!("{}", "-".repeat(80));

        for migration in &applied {
            let applied_at = migration
                .applied_at
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            let exec_time = migration
                .execution_time_ms
                .map(|ms| ms.to_string())
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{:<8} {:<40} {:<20} {:<12}",
                migration.version, migration.name, applied_at, exec_time
            );
        }

        println!();
    }

    
    let runner = MigrationRunner::new(&migrations_dir);
    let pending = runner.pending_migrations(&conn)?;

    if !pending.is_empty() {
        println!("Pending Migrations:");
        println!("{:<8} {:<40}", "Version", "Name");
        println!("{}", "-".repeat(50));

        for migration in &pending {
            println!("{:<8} {:<40}", migration.version, migration.name);
        }
    } else {
        println!(" No pending migrations");
    }

    Ok(())
}

fn cmd_create(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let migrations_dir = get_migrations_dir();

    
    std::fs::create_dir_all(&migrations_dir)?;

    
    let runner = MigrationRunner::new(&migrations_dir);
    let existing = runner.discover_migrations().unwrap_or_default();

    let next_version = existing.iter().map(|m| m.version).max().unwrap_or(0) + 1;

    
    let filename = format!("{:03}_{}.sql", next_version, name.replace(' ', "_"));
    let filepath = migrations_dir.join(&filename);

    let content = format!(
        r#"-- Migration: {:03}_{}
-- Description: {}
-- Author: Database Migration System
-- Date: {}

-- === UP MIGRATION ===


-- === DOWN MIGRATION ===

"#,
        next_version,
        name.replace(' ', "_"),
        name,
        time::now().format("%Y-%m-%d")
    );

    std::fs::write(&filepath, content)?;

    println!(" Created migration: {}", filepath.display());
    println!();
    println!("Edit the file to add your migration SQL:");
    println!("  - Add UP migration commands after '-- === UP MIGRATION ==='");
    println!("  - Add DOWN migration commands after '-- === DOWN MIGRATION ==='");

    Ok(())
}
