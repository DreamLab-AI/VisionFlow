//! Migration runner with transaction safety and progress logging

use super::{MigrationError, MigrationVersion, Result, VersionTracker};
use log::{error, info, warn};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Migration execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Actually execute migrations
    Execute,
    /// Only show what would be executed (dry-run)
    DryRun,
}

/// Migration runner
pub struct MigrationRunner {
    migrations_dir: PathBuf,
    mode: ExecutionMode,
}

impl MigrationRunner {
    /// Create new migration runner
    pub fn new<P: AsRef<Path>>(migrations_dir: P) -> Self {
        Self {
            migrations_dir: migrations_dir.as_ref().to_path_buf(),
            mode: ExecutionMode::Execute,
        }
    }

    /// Set execution mode
    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Discover all migration files in the migrations directory
    pub fn discover_migrations(&self) -> Result<Vec<MigrationVersion>> {
        if !self.migrations_dir.exists() {
            return Err(MigrationError::FileNotFound(
                self.migrations_dir.display().to_string(),
            ));
        }

        let mut migrations = Vec::new();

        for entry in std::fs::read_dir(&self.migrations_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("sql") {
                let migration = MigrationVersion::from_file(&path)?;
                migrations.push(migration);
            }
        }

        // Sort by version number
        migrations.sort_by_key(|m| m.version);

        if migrations.is_empty() {
            warn!(
                "[MigrationRunner] No migration files found in {}",
                self.migrations_dir.display()
            );
        } else {
            info!(
                "[MigrationRunner] Discovered {} migration(s)",
                migrations.len()
            );
        }

        Ok(migrations)
    }

    /// Get pending migrations (not yet applied)
    pub fn pending_migrations(&self, conn: &Connection) -> Result<Vec<MigrationVersion>> {
        let tracker = VersionTracker::new(conn);
        let all_migrations = self.discover_migrations()?;

        let mut pending = Vec::new();
        for migration in all_migrations {
            if !tracker.is_applied(migration.version)? {
                pending.push(migration);
            } else {
                // Verify checksum for applied migrations
                tracker.verify_checksum(&migration)?;
            }
        }

        Ok(pending)
    }

    /// Apply all pending migrations
    pub fn migrate_up(&self, conn: &mut Connection) -> Result<usize> {
        // Initialize version tracking
        let tracker = VersionTracker::new(conn);
        tracker.initialize()?;

        let pending = self.pending_migrations(conn)?;

        if pending.is_empty() {
            info!("[MigrationRunner] No pending migrations");
            return Ok(0);
        }

        info!(
            "[MigrationRunner] Found {} pending migration(s)",
            pending.len()
        );

        let mut applied_count = 0;

        for migration in pending {
            if self.mode == ExecutionMode::DryRun {
                info!(
                    "[DRY RUN] Would apply migration {} - {}",
                    migration.version, migration.name
                );
                continue;
            }

            info!(
                "[MigrationRunner] Applying migration {} - {}",
                migration.version, migration.name
            );

            let start = Instant::now();

            // Execute in transaction
            let tx = conn.transaction()?;

            match self.execute_migration_up(&tx, &migration) {
                Ok(_) => {
                    let execution_time = start.elapsed().as_millis() as i64;

                    // Record in version table
                    let tracker = VersionTracker::new(&tx);
                    tracker.record_migration(&migration, execution_time)?;

                    tx.commit()?;

                    info!(
                        "[MigrationRunner] ✓ Migration {} completed in {}ms",
                        migration.version, execution_time
                    );

                    applied_count += 1;
                }
                Err(e) => {
                    error!(
                        "[MigrationRunner] ✗ Migration {} failed: {}",
                        migration.version, e
                    );
                    // Transaction will auto-rollback on drop
                    return Err(e);
                }
            }
        }

        info!(
            "[MigrationRunner] Successfully applied {} migration(s)",
            applied_count
        );
        Ok(applied_count)
    }

    /// Execute UP migration SQL
    fn execute_migration_up(&self, conn: &Connection, migration: &MigrationVersion) -> Result<()> {
        let migration_path = self.migrations_dir.join(format!(
            "{:03}_{}.sql",
            migration.version,
            migration.name.replace(' ', "_")
        ));

        let content = std::fs::read_to_string(&migration_path)?;
        let (up_sql, _) = MigrationVersion::parse_sql(&content)?;

        // Execute the UP migration
        conn.execute_batch(&up_sql)?;

        Ok(())
    }

    /// Get migration file path for reading rollback SQL
    pub fn get_migration_path(&self, migration: &MigrationVersion) -> PathBuf {
        self.migrations_dir.join(format!(
            "{:03}_{}.sql",
            migration.version,
            migration.name.replace(' ', "_")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_migration(dir: &Path, version: i32, name: &str) -> std::io::Result<()> {
        let path = dir.join(format!("{:03}_{}.sql", version, name));
        std::fs::write(
            path,
            format!(
                "-- Migration: {:03}_{}\n\
                 -- === UP MIGRATION ===\n\
                 CREATE TABLE test_{} (id INTEGER);\n\
                 -- === DOWN MIGRATION ===\n\
                 DROP TABLE test_{};\n",
                version, name, version, version
            ),
        )
    }

    #[test]
    fn test_discover_migrations() {
        let temp_dir = TempDir::new().unwrap();
        let migrations_dir = temp_dir.path();

        create_test_migration(migrations_dir, 1, "first").unwrap();
        create_test_migration(migrations_dir, 2, "second").unwrap();

        let runner = MigrationRunner::new(migrations_dir);
        let migrations = runner.discover_migrations().unwrap();

        assert_eq!(migrations.len(), 2);
        assert_eq!(migrations[0].version, 1);
        assert_eq!(migrations[1].version, 2);
    }

    #[test]
    fn test_dry_run_mode() {
        let temp_dir = TempDir::new().unwrap();
        let runner = MigrationRunner::new(temp_dir.path()).with_mode(ExecutionMode::DryRun);

        assert_eq!(runner.mode, ExecutionMode::DryRun);
    }
}
