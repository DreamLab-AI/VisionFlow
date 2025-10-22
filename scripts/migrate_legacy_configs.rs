#!/usr/bin/env rust-script
//! One-time migration script to move legacy YAML/TOML configs to SQLite database
//!
//! This script:
//! 1. Reads all legacy configuration files
//! 2. Inserts data into appropriate database tables
//! 3. Generates a migration report
//! 4. DOES NOT delete files (that's a manual step after verification)

use std::fs;
use std::path::Path;
use serde_json::Value;

/// Migration report structure
#[derive(Debug)]
struct MigrationReport {
    files_processed: Vec<String>,
    records_migrated: usize,
    errors: Vec<String>,
    timestamp: String,
}

impl MigrationReport {
    fn new() -> Self {
        Self {
            files_processed: Vec::new(),
            records_migrated: 0,
            errors: Vec::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn add_file(&mut self, path: &str) {
        self.files_processed.push(path.to_string());
    }

    fn add_records(&mut self, count: usize) {
        self.records_migrated += count;
    }

    fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    fn print_summary(&self) {
        println!("\n=== MIGRATION REPORT ===");
        println!("Timestamp: {}", self.timestamp);
        println!("\nFiles Processed:");
        for file in &self.files_processed {
            println!("  - {}", file);
        }
        println!("\nTotal Records Migrated: {}", self.records_migrated);

        if !self.errors.is_empty() {
            println!("\nErrors Encountered:");
            for error in &self.errors {
                println!("  ! {}", error);
            }
        } else {
            println!("\n✓ Migration completed successfully with no errors");
        }
        println!("\n======================");
    }

    fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&serde_json::json!({
            "timestamp": self.timestamp,
            "files_processed": self.files_processed,
            "records_migrated": self.records_migrated,
            "errors": self.errors,
        }))?;
        fs::write(path, json)?;
        Ok(())
    }
}

/// Read YAML config file
fn read_yaml_config(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let yaml_value: serde_yaml::Value = serde_yaml::from_str(&content)?;
    let json_value = serde_json::to_value(yaml_value)?;
    Ok(json_value)
}

/// Read TOML config file
fn read_toml_config(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let toml_value: toml::Value = toml::from_str(&content)?;
    let json_value = serde_json::to_value(toml_value)?;
    Ok(json_value)
}

/// Insert settings into database
fn migrate_to_database(
    db_path: &str,
    settings_data: Value,
    source_file: &str,
    report: &mut MigrationReport,
) -> Result<(), Box<dyn std::error::Error>> {
    use rusqlite::Connection;

    let conn = Connection::open(db_path)?;

    // Flatten JSON and insert into settings table
    let flattened = flatten_json(&settings_data, "");

    for (key, value) in flattened {
        let value_str = match value {
            Value::String(s) => s,
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            other => serde_json::to_string(&other)?,
        };

        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, tier, source, updated_at) VALUES (?1, ?2, ?3, ?4, datetime('now'))",
            rusqlite::params![
                key,
                value_str,
                "system", // Default tier
                source_file,
            ],
        )?;

        report.add_records(1);
    }

    Ok(())
}

/// Flatten nested JSON into dot-notation keys
fn flatten_json(value: &Value, prefix: &str) -> Vec<(String, Value)> {
    let mut result = Vec::new();

    match value {
        Value::Object(map) => {
            for (key, val) in map {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };
                result.extend(flatten_json(val, &new_prefix));
            }
        }
        Value::Array(arr) => {
            for (idx, val) in arr.iter().enumerate() {
                let new_prefix = format!("{}[{}]", prefix, idx);
                result.extend(flatten_json(val, &new_prefix));
            }
        }
        other => {
            result.push((prefix.to_string(), other.clone()));
        }
    }

    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Starting legacy config migration...\n");

    let mut report = MigrationReport::new();
    let db_path = std::env::var("DATABASE_PATH")
        .unwrap_or_else(|_| "data/visionflow.db".to_string());

    // Legacy config files to migrate
    let legacy_files = vec![
        ("data/settings.yaml", "yaml"),
        ("data/settings_ontology_extension.yaml", "yaml"),
        ("data/dev_config.toml", "toml"),
    ];

    for (file_path, file_type) in legacy_files {
        let path = Path::new(file_path);

        if !path.exists() {
            println!("⚠️  File not found: {} (skipping)", file_path);
            report.add_error(format!("File not found: {}", file_path));
            continue;
        }

        println!("📄 Processing: {}", file_path);
        report.add_file(file_path);

        let config_data = match file_type {
            "yaml" => match read_yaml_config(path) {
                Ok(data) => data,
                Err(e) => {
                    let error_msg = format!("Failed to read {}: {}", file_path, e);
                    println!("❌ {}", error_msg);
                    report.add_error(error_msg);
                    continue;
                }
            },
            "toml" => match read_toml_config(path) {
                Ok(data) => data,
                Err(e) => {
                    let error_msg = format!("Failed to read {}: {}", file_path, e);
                    println!("❌ {}", error_msg);
                    report.add_error(error_msg);
                    continue;
                }
            },
            _ => {
                report.add_error(format!("Unknown file type: {}", file_type));
                continue;
            }
        };

        if let Err(e) = migrate_to_database(&db_path, config_data, file_path, &mut report) {
            let error_msg = format!("Failed to migrate {}: {}", file_path, e);
            println!("❌ {}", error_msg);
            report.add_error(error_msg);
        } else {
            println!("✓ Successfully migrated: {}", file_path);
        }
    }

    // Print and save report
    report.print_summary();

    let report_path = "data/migration_report.json";
    match report.save_to_file(report_path) {
        Ok(_) => println!("\n📊 Report saved to: {}", report_path),
        Err(e) => eprintln!("\n❌ Failed to save report: {}", e),
    }

    println!("\n⚠️  IMPORTANT: Legacy files have NOT been deleted.");
    println!("   Review the migration report and verify database contents before deleting:");
    for (file_path, _) in legacy_files {
        println!("   - {}", file_path);
    }

    Ok(())
}
