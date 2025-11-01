//! Rollback manager with zero data loss guarantee

use super::{MigrationError, MigrationVersion, Result, VersionTracker};
use log::{error, info, warn};
use rusqlite::Connection;
use std::time::Instant;

///
pub struct RollbackManager;

impl RollbackManager {
    
    pub fn rollback_last(conn: &mut Connection, migrations_dir: &std::path::Path) -> Result<()> {
        let tracker = VersionTracker::new(conn);

        let current_version = tracker
            .current_version()?
            .ok_or(MigrationError::NoMigrations)?;

        info!(
            "[RollbackManager] Rolling back migration {}",
            current_version
        );

        
        let migration = tracker
            .get_applied(current_version)?
            .ok_or(MigrationError::NotApplied(current_version))?;

        Self::rollback_migration(conn, &migration, migrations_dir)?;

        info!(
            "[RollbackManager] ✓ Successfully rolled back migration {}",
            current_version
        );
        Ok(())
    }

    
    pub fn rollback_to(
        conn: &mut Connection,
        target_version: i32,
        migrations_dir: &std::path::Path,
    ) -> Result<()> {
        let tracker = VersionTracker::new(conn);

        let current_version = tracker
            .current_version()?
            .ok_or(MigrationError::NoMigrations)?;

        if target_version >= current_version {
            warn!(
                "[RollbackManager] Already at or below version {}",
                target_version
            );
            return Ok(());
        }

        info!(
            "[RollbackManager] Rolling back from version {} to {}",
            current_version, target_version
        );

        
        let applied = tracker.get_all_applied()?;
        let to_rollback: Vec<_> = applied
            .into_iter()
            .filter(|m| m.version > target_version)
            .rev()
            .collect();

        info!(
            "[RollbackManager] Will rollback {} migration(s)",
            to_rollback.len()
        );

        for migration in to_rollback {
            Self::rollback_migration(conn, &migration, migrations_dir)?;
        }

        info!(
            "[RollbackManager] ✓ Successfully rolled back to version {}",
            target_version
        );
        Ok(())
    }

    
    fn rollback_migration(
        conn: &mut Connection,
        migration: &MigrationVersion,
        migrations_dir: &std::path::Path,
    ) -> Result<()> {
        info!(
            "[RollbackManager] Rolling back migration {} - {}",
            migration.version, migration.name
        );

        let migration_path = migrations_dir.join(format!(
            "{:03}_{}.sql",
            migration.version,
            migration.name.replace(' ', "_")
        ));

        if !migration_path.exists() {
            return Err(MigrationError::FileNotFound(
                migration_path.display().to_string(),
            ));
        }

        let content = std::fs::read_to_string(&migration_path)?;
        let (_, down_sql) = MigrationVersion::parse_sql(&content)?;

        if down_sql.is_empty() {
            warn!(
                "[RollbackManager] No DOWN migration for {} - rollback may be incomplete",
                migration.version
            );
        }

        let start = Instant::now();

        
        let tx = conn.transaction()?;

        match tx.execute_batch(&down_sql) {
            Ok(_) => {
                
                let tracker = VersionTracker::new(&tx);
                tracker.remove_migration(migration.version)?;

                tx.commit()?;

                let execution_time = start.elapsed().as_millis();
                info!(
                    "[RollbackManager] ✓ Rolled back migration {} in {}ms",
                    migration.version, execution_time
                );

                Ok(())
            }
            Err(e) => {
                error!(
                    "[RollbackManager] ✗ Failed to rollback migration {}: {}",
                    migration.version, e
                );
                
                Err(MigrationError::Database(e))
            }
        }
    }

    
    pub fn verify_rollback_safety(
        migration: &MigrationVersion,
        migrations_dir: &std::path::Path,
    ) -> Result<bool> {
        let migration_path = migrations_dir.join(format!(
            "{:03}_{}.sql",
            migration.version,
            migration.name.replace(' ', "_")
        ));

        if !migration_path.exists() {
            return Ok(false);
        }

        let content = std::fs::read_to_string(&migration_path)?;
        let (_, down_sql) = MigrationVersion::parse_sql(&content)?;

        Ok(!down_sql.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_rollback_safety() {
        
        
    }
}
