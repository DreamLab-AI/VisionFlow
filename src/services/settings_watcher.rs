// src/services/settings_watcher.rs
//! Settings hot-reload watcher using notify for file system events
//!
//! This service:
//! - Monitors the settings database file for changes
//! - Triggers automatic settings reload on file modification
//! - Supports both OptimizedSettingsActor and future actor implementations
//! - Provides graceful error handling and logging

use actix::Addr;
use log::{debug, error, info, warn};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::actors::messages::ReloadSettings;
use crate::actors::optimized_settings_actor::OptimizedSettingsActor;

/// Debounce duration to avoid rapid consecutive reloads
const DEBOUNCE_DURATION: Duration = Duration::from_millis(500);

pub struct SettingsWatcher {
    db_path: String,
    settings_actor: Addr<OptimizedSettingsActor>,
    last_reload: Arc<RwLock<std::time::Instant>>,
}

impl SettingsWatcher {
    /// Create a new settings watcher for the given database path
    pub fn new(db_path: String, settings_actor: Addr<OptimizedSettingsActor>) -> Self {
        Self {
            db_path,
            settings_actor,
            last_reload: Arc::new(RwLock::new(std::time::Instant::now())),
        }
    }

    /// Start watching the settings database for changes
    pub async fn start(self) -> notify::Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();

        // Create the watcher
        let mut watcher = notify::recommended_watcher(tx)?;

        // Watch the database file
        let db_path = Path::new(&self.db_path);

        // Watch the parent directory since the file might be replaced atomically
        let watch_path = if db_path.exists() {
            if let Some(parent) = db_path.parent() {
                info!("Watching directory for settings changes: {}", parent.display());
                parent
            } else {
                warn!("No parent directory for settings DB, watching file directly");
                db_path
            }
        } else {
            error!("Settings database does not exist: {}", self.db_path);
            return Err(notify::Error::generic("Database file does not exist"));
        };

        watcher.watch(watch_path, RecursiveMode::NonRecursive)?;

        info!("Settings hot-reload watcher started for: {}", self.db_path);

        let last_reload = self.last_reload.clone();
        let settings_actor = self.settings_actor.clone();
        let db_filename = db_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("settings.db")
            .to_string();

        // Process file system events
        tokio::spawn(async move {
            while let Ok(result) = rx.recv() {
                match result {
                    Ok(event) => {
                        if Self::should_reload(&event, &db_filename) {
                            // Debounce: only reload if enough time has passed
                            let mut last = last_reload.write().await;
                            let elapsed = last.elapsed();

                            if elapsed >= DEBOUNCE_DURATION {
                                debug!("Settings file modified, triggering reload...");
                                *last = std::time::Instant::now();
                                drop(last); // Release lock before async operation

                                Self::trigger_reload(&settings_actor).await;
                            } else {
                                debug!("Skipping reload due to debounce ({}ms < {}ms)",
                                    elapsed.as_millis(), DEBOUNCE_DURATION.as_millis());
                            }
                        }
                    }
                    Err(e) => {
                        error!("File watcher error: {}", e);
                    }
                }
            }
            warn!("Settings watcher event loop terminated");
        });

        // Keep the watcher alive
        std::mem::forget(watcher);
        Ok(())
    }

    /// Check if the event should trigger a reload
    fn should_reload(event: &Event, db_filename: &str) -> bool {
        // Check if event is a modification
        matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        ) && event.paths.iter().any(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|name| name == db_filename || name.ends_with(".db"))
                .unwrap_or(false)
        })
    }

    /// Trigger settings reload via actor message
    async fn trigger_reload(settings_actor: &Addr<OptimizedSettingsActor>) {
        match settings_actor.send(ReloadSettings).await {
            Ok(Ok(())) => {
                info!("✓ Settings reloaded successfully via hot-reload");
            }
            Ok(Err(e)) => {
                error!("Failed to reload settings: {}", e);
            }
            Err(e) => {
                error!("Actor communication error during reload: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::{Event, EventKind};
    use std::path::PathBuf;

    #[test]
    fn test_should_reload_on_modify() {
        let event = Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Any,
            )),
            paths: vec![PathBuf::from("/data/settings.db")],
            attrs: Default::default(),
        };

        assert!(SettingsWatcher::should_reload(&event, "settings.db"));
    }

    #[test]
    fn test_should_not_reload_on_different_file() {
        let event = Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Any,
            )),
            paths: vec![PathBuf::from("/data/other.txt")],
            attrs: Default::default(),
        };

        assert!(!SettingsWatcher::should_reload(&event, "settings.db"));
    }
}
