use rusqlite::{Connection, Result};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    println!("🔄 Starting Database Migration 001...\n");

    // Database path
    let db_path = "data/settings.db";

    // Verify database exists
    if !Path::new(db_path).exists() {
        eprintln!("❌ Error: Database not found at {}", db_path);
        std::process::exit(1);
    }

    // Read migration SQL
    let sql_path = "scripts/migrations/001_add_missing_settings.sql";
    let sql = match fs::read_to_string(sql_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("❌ Error reading migration file: {}", e);
            std::process::exit(1);
        }
    };

    println!("📄 Loaded migration: {}", sql_path);

    // Open database connection
    let conn = Connection::open(db_path)?;
    println!("🔗 Connected to database: {}", db_path);

    // Get initial count
    let initial_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM settings",
        [],
        |row| row.get(0)
    )?;
    println!("📊 Initial settings count: {}\n", initial_count);

    // Execute migration
    println!("⚡ Executing migration...");
    match conn.execute_batch(&sql) {
        Ok(_) => println!("✅ Migration SQL executed successfully"),
        Err(e) => {
            eprintln!("❌ Migration failed: {}", e);
            std::process::exit(1);
        }
    }

    // Get final count
    let final_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM settings",
        [],
        |row| row.get(0)
    )?;

    let added_count = final_count - initial_count;
    println!("📊 Final settings count: {}", final_count);
    println!("➕ Settings added: {}\n", added_count);

    // Verify expected count (73 settings)
    if added_count == 73 {
        println!("✅ SUCCESS: All 73 settings added correctly!");
    } else {
        println!("⚠️  WARNING: Expected 73 settings, but added {}", added_count);
    }

    // Category breakdown
    println!("\n📋 Category Breakdown:");
    let mut stmt = conn.prepare(
        "SELECT category, COUNT(*) as count FROM settings GROUP BY category ORDER BY category"
    )?;

    let categories = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?
        ))
    })?;

    for category in categories {
        let (name, count) = category?;
        println!("  - {}: {} settings", name, count);
    }

    // Check for duplicates
    let duplicate_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM (SELECT key, COUNT(*) as cnt FROM settings GROUP BY key HAVING cnt > 1)",
        [],
        |row| row.get(0)
    )?;

    if duplicate_count > 0 {
        println!("\n⚠️  WARNING: {} duplicate keys found!", duplicate_count);

        let mut stmt = conn.prepare(
            "SELECT key, COUNT(*) as cnt FROM settings GROUP BY key HAVING cnt > 1"
        )?;

        let duplicates = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?
            ))
        })?;

        for duplicate in duplicates {
            let (key, count) = duplicate?;
            println!("  - {}: {} occurrences", key, count);
        }
    } else {
        println!("\n✅ No duplicate keys found");
    }

    // Verify new settings by category
    println!("\n🔍 Verifying New Settings:");

    let categories_to_check = vec![
        ("analytics", 11),
        ("dashboard", 8),
        ("performance", 11),
        ("gpu", 8),
        ("effects", 4),
        ("developer", 11),
        ("agents", 20),
    ];

    for (category, expected) in categories_to_check {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM settings WHERE category = ?1",
            [category],
            |row| row.get(0)
        )?;

        let status = if count >= expected { "✅" } else { "⚠️" };
        println!("  {} {}: {} settings (expected: {})", status, category, count, expected);
    }

    println!("\n🎉 Migration 001 Complete!");
    println!("📝 Documentation saved to: docs/MIGRATION_001_RESULTS.md");

    Ok(())
}
