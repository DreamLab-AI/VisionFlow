use anyhow::Result;
use chrono::Utc;
use log::{info, warn};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig, DownloadProgress};
use super::ontology_storage::{OntologyStorage, DatabaseStatistics};

pub struct OntologySync {
    downloader: OntologyDownloader,
    storage: Arc<OntologyStorage>,
}

impl OntologySync {
    pub fn new(
        downloader_config: OntologyDownloaderConfig,
        storage: OntologyStorage,
    ) -> Result<Self> {
        let downloader = OntologyDownloader::new(downloader_config)?;

        Ok(Self {
            downloader,
            storage: Arc::new(storage),
        })
    }

    pub async fn sync(&self) -> Result<SyncResult> {
        info!("Starting ontology synchronization");

        let start_time = Utc::now();

        let blocks = self.downloader.download_all().await?;

        let saved_count = self.storage.save_blocks(&blocks)?;

        self.storage.set_sync_metadata("last_sync_time", &Utc::now().to_rfc3339())?;
        self.storage.set_sync_metadata("last_sync_blocks", &saved_count.to_string())?;

        let progress = self.downloader.get_progress().await;
        let stats = self.storage.get_statistics()?;

        let duration = Utc::now().signed_duration_since(start_time);

        info!(
            "Synchronization complete: {} blocks saved in {:?}",
            saved_count, duration
        );

        Ok(SyncResult {
            blocks_downloaded: blocks.len(),
            blocks_saved: saved_count,
            errors: progress.errors,
            duration_seconds: duration.num_seconds() as u64,
            statistics: stats,
        })
    }

    pub async fn get_progress(&self) -> DownloadProgress {
        self.downloader.get_progress().await
    }

    pub fn get_statistics(&self) -> Result<DatabaseStatistics> {
        self.storage.get_statistics()
    }

    pub fn storage(&self) -> Arc<OntologyStorage> {
        Arc::clone(&self.storage)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SyncResult {
    pub blocks_downloaded: usize,
    pub blocks_saved: usize,
    pub errors: Vec<String>,
    pub duration_seconds: u64,
    pub statistics: DatabaseStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sync_creation() {
        let config = OntologyDownloaderConfig::with_token("test_token".to_string());
        let storage = OntologyStorage::in_memory().unwrap();
        let sync = OntologySync::new(config, storage);

        assert!(sync.is_ok());
    }
}
