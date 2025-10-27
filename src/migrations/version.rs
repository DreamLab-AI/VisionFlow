//! Migration version tracking with checksum verification

use super::{MigrationError, Result};
use chrono::{DateTime, Utc};
use log::{info, warn};
use rusqlite::{params, Connection};

/// Represents a single migration version
#[derive(Debug, Clone)]
pub struct MigrationVersion {
    pub version: i32,
    pub name: String,
    pub checksum: String,
    pub applied_at: Option<DateTime<Utc>>,
    pub execution_time_ms: Option<i64>,
}

impl MigrationVersion {
    /// Create new migration version from file
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| MigrationError::InvalidFormat("Invalid filename".into()))?;

        // Parse version from filename (e.g., "001_create_table.sql" -> 1)
        let parts: Vec<&str> = filename.splitn(2, '_').collect();
        if parts.len() != 2 {
            return Err(MigrationError::InvalidFormat(format!(
                "Filename must be in format 'NNN_name.sql': {}",
                filename
            )));
        }

        let version = parts[0].parse::<i32>().map_err(|_| {
            MigrationError::InvalidFormat(format!("Invalid version number: {}", parts[0]))
        })?;

        let name = parts[1]
            .trim_end_matches(".sql")
            .replace('_', " ")
            .to_string();

        // Read file and calculate checksum
        let content = std::fs::read_to_string(path)?;
        let checksum = Self::calculate_checksum(&content);

        Ok(Self {
            version,
            name,
            checksum,
            applied_at: None,
            execution_time_ms: None,
        })
    }

    /// Calculate SHA-1 checksum of migration content
    pub fn calculate_checksum(content: &str) -> String {
        use sha1::{Digest, Sha1};
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Parse migration SQL into UP and DOWN sections
    pub fn parse_sql(content: &str) -> Result<(String, String)> {
        let up_marker = "-- === UP MIGRATION ===";
        let down_marker = "-- === DOWN MIGRATION ===";

        let up_start = content
            .find(up_marker)
            .ok_or_else(|| MigrationError::Parse("Missing UP MIGRATION marker".into()))?;

        let down_start = content
            .find(down_marker)
            .ok_or_else(|| MigrationError::Parse("Missing DOWN MIGRATION marker".into()))?;

        if up_start >= down_start {
            return Err(MigrationError::Parse(
                "UP MIGRATION must come before DOWN MIGRATION".into(),
            ));
        }

        let up_sql = content[up_start + up_marker.len()..down_start]
            .trim()
            .to_string();

        let down_sql = content[down_start + down_marker.len()..].trim().to_string();

        if up_sql.is_empty() {
            return Err(MigrationError::Parse("UP migration SQL is empty".into()));
        }

        if down_sql.is_empty() {
            warn!("DOWN migration SQL is empty - rollback may not be complete");
        }

        Ok((up_sql, down_sql))
    }
}

/// Tracks migration versions in the database
pub struct VersionTracker<'a> {
    conn: &'a Connection,
}

impl<'a> VersionTracker<'a> {
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Initialize schema_migrations table if it doesn't exist
    pub fn initialize(&self) -> Result<()> {
        info!("[VersionTracker] Initializing schema_migrations table");

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT NOT NULL,
                execution_time_ms INTEGER
            )",
            [],
        )?;

        info!("[VersionTracker] schema_migrations table ready");
        Ok(())
    }

    /// Get current schema version (highest applied migration)
    pub fn current_version(&self) -> Result<Option<i32>> {
        let version = self
            .conn
            .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
                row.get::<_, Option<i32>>(0)
            })
            .ok()
            .flatten();

        Ok(version)
    }

    /// Check if a migration version is already applied
    pub fn is_applied(&self, version: i32) -> Result<bool> {
        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = ?1",
            params![version],
            |row| row.get(0),
        )?;

        Ok(count > 0)
    }

    /// Get applied migration details
    pub fn get_applied(&self, version: i32) -> Result<Option<MigrationVersion>> {
        let result = self
            .conn
            .query_row(
                "SELECT version, name, checksum, applied_at, execution_time_ms
             FROM schema_migrations WHERE version = ?1",
                params![version],
                |row| {
                    Ok(MigrationVersion {
                        version: row.get(0)?,
                        name: row.get(1)?,
                        checksum: row.get(2)?,
                        applied_at: row
                            .get::<_, String>(3)
                            .ok()
                            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                            .map(|dt| dt.with_timezone(&Utc)),
                        execution_time_ms: row.get(4).ok(),
                    })
                },
            )
            .ok();

        Ok(result)
    }

    /// Get all applied migrations
    pub fn get_all_applied(&self) -> Result<Vec<MigrationVersion>> {
        let mut stmt = self.conn.prepare(
            "SELECT version, name, checksum, applied_at, execution_time_ms
             FROM schema_migrations ORDER BY version ASC",
        )?;

        let migrations = stmt
            .query_map([], |row| {
                Ok(MigrationVersion {
                    version: row.get(0)?,
                    name: row.get(1)?,
                    checksum: row.get(2)?,
                    applied_at: row
                        .get::<_, String>(3)
                        .ok()
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                    execution_time_ms: row.get(4).ok(),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(migrations)
    }

    /// Record a successful migration
    pub fn record_migration(
        &self,
        migration: &MigrationVersion,
        execution_time_ms: i64,
    ) -> Result<()> {
        info!(
            "[VersionTracker] Recording migration {} - {}",
            migration.version, migration.name
        );

        self.conn.execute(
            "INSERT INTO schema_migrations (version, name, checksum, execution_time_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                migration.version,
                migration.name,
                migration.checksum,
                execution_time_ms
            ],
        )?;

        Ok(())
    }

    /// Remove a migration record (for rollback)
    pub fn remove_migration(&self, version: i32) -> Result<()> {
        info!("[VersionTracker] Removing migration record {}", version);

        let rows = self.conn.execute(
            "DELETE FROM schema_migrations WHERE version = ?1",
            params![version],
        )?;

        if rows == 0 {
            return Err(MigrationError::NotApplied(version));
        }

        Ok(())
    }

    /// Verify checksum of applied migration
    pub fn verify_checksum(&self, migration: &MigrationVersion) -> Result<()> {
        if let Some(applied) = self.get_applied(migration.version)? {
            if applied.checksum != migration.checksum {
                return Err(MigrationError::ChecksumMismatch(
                    migration.version,
                    applied.checksum,
                    migration.checksum.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_checksum() {
        let content = "SELECT 1;";
        let checksum = MigrationVersion::calculate_checksum(content);
        assert_eq!(checksum.len(), 40); // SHA-1 produces 40 hex chars
    }

    #[test]
    fn test_parse_sql() {
        let sql = r#"
-- === UP MIGRATION ===
CREATE TABLE test (id INTEGER);

-- === DOWN MIGRATION ===
DROP TABLE test;
        "#;

        let (up, down) = MigrationVersion::parse_sql(sql).unwrap();
        assert!(up.contains("CREATE TABLE test"));
        assert!(down.contains("DROP TABLE test"));
    }

    #[test]
    fn test_parse_sql_missing_marker() {
        let sql = "CREATE TABLE test (id INTEGER);";
        assert!(MigrationVersion::parse_sql(sql).is_err());
    }
}
