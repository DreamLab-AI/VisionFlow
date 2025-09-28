//! Vector storage and persistence functionality

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{Result, VectorError};

/// Configuration for vector storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub format: SerializationFormat,
    pub compression: CompressionType,
    pub base_path: String,
    pub enable_mmap: bool,
    pub cache_size_mb: usize,
    pub backup_enabled: bool,
    pub backup_interval_minutes: u64,
}

/// Serialization formats for vector data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SerializationFormat {
    Bincode,
    MessagePack,
    Parquet,
    Npy,
    HDF5,
    Json, // For debugging and small datasets
}

/// Compression types for storage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressionType {
    None,
    Gzip,
    Lz4,
    Zstd,
    Snappy,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            format: SerializationFormat::Bincode,
            compression: CompressionType::Zstd,
            base_path: "./vectorstore_data".to_string(),
            enable_mmap: true,
            cache_size_mb: 256,
            backup_enabled: false,
            backup_interval_minutes: 60,
        }
    }
}

/// Metadata about stored vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub id: String,
    pub dimension: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub tags: Vec<String>,
    pub custom_metadata: serde_json::Value,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_vectors: usize,
    pub total_size_bytes: u64,
    pub compression_ratio: f64,
    pub last_backup: Option<chrono::DateTime<chrono::Utc>>,
    pub cache_hit_rate: f64,
}

/// Base trait for vector storage backends
#[async_trait]
pub trait VectorStorage: Send + Sync {
    /// Save vectors with their associated text content
    async fn save_vectors(&self, ids: &[String], texts: &[String]) -> Result<()>;

    /// Load vectors by their IDs
    async fn load_vectors(&self, ids: &[String]) -> Result<HashMap<String, String>>;

    /// Remove vectors by their IDs
    async fn remove_vectors(&self, ids: &[String]) -> Result<()>;

    /// Save metadata associated with vectors
    async fn save_metadata(&self, path: &str) -> Result<()>;

    /// Load metadata from storage
    async fn load_metadata(&self, path: &str) -> Result<()>;

    /// List all stored vector IDs
    async fn list_vector_ids(&self) -> Result<Vec<String>>;

    /// Get storage statistics
    async fn get_stats(&self) -> Result<StorageStats>;

    /// Optimize storage (compaction, cleanup, etc.)
    async fn optimize(&self) -> Result<()>;

    /// Create a backup of the storage
    async fn backup(&self, backup_path: &str) -> Result<()>;

    /// Restore from a backup
    async fn restore(&self, backup_path: &str) -> Result<()>;
}

/// File system based vector storage
pub struct FileSystemStorage {
    config: StorageConfig,
    vector_cache: parking_lot::RwLock<lru::LruCache<String, String>>,
    metadata_cache: parking_lot::RwLock<lru::LruCache<String, VectorMetadata>>,
}

impl FileSystemStorage {
    pub async fn new(config: StorageConfig) -> Result<Self> {
        // Create base directory if it doesn't exist
        fs::create_dir_all(&config.base_path).await?;

        let cache_size = (config.cache_size_mb * 1024 * 1024) / 100; // Rough estimate
        let vector_cache = parking_lot::RwLock::new(
            lru::LruCache::new(std::num::NonZeroUsize::new(cache_size).unwrap()),
        );
        let metadata_cache = parking_lot::RwLock::new(
            lru::LruCache::new(std::num::NonZeroUsize::new(1000).unwrap()),
        );

        Ok(Self {
            config,
            vector_cache,
            metadata_cache,
        })
    }

    fn get_vector_path(&self, id: &str) -> String {
        format!("{}/vectors/{}.vec", self.config.base_path, id)
    }

    fn get_metadata_path(&self, id: &str) -> String {
        format!("{}/metadata/{}.meta", self.config.base_path, id)
    }

    fn get_index_path(&self) -> String {
        format!("{}/index.idx", self.config.base_path)
    }

    async fn serialize_data<T: Serialize>(&self, data: &T) -> Result<Vec<u8>> {
        let serialized = match self.config.format {
            SerializationFormat::Bincode => bincode::serialize(data)?,
            SerializationFormat::MessagePack => rmp_serde::to_vec(data)
                .map_err(|e| VectorError::serialization_error(format!("MessagePack error: {}", e)))?,
            SerializationFormat::Json => serde_json::to_vec(data)?,
            _ => {
                return Err(VectorError::serialization_error(
                    "Unsupported serialization format",
                ));
            }
        };

        let compressed = match self.config.compression {
            CompressionType::None => serialized,
            CompressionType::Gzip => {
                use std::io::Write;
                let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(&serialized)?;
                encoder.finish()?
            }
            CompressionType::Zstd => {
                zstd::bulk::compress(&serialized, 3)
                    .map_err(|e| VectorError::serialization_error(format!("Zstd compression error: {}", e)))?
            }
            _ => {
                return Err(VectorError::serialization_error(
                    "Unsupported compression type",
                ));
            }
        };

        Ok(compressed)
    }

    async fn deserialize_data<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<T> {
        let decompressed = match self.config.compression {
            CompressionType::None => data.to_vec(),
            CompressionType::Gzip => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                decompressed
            }
            CompressionType::Zstd => {
                zstd::bulk::decompress(data, 10 * 1024 * 1024) // 10MB limit
                    .map_err(|e| VectorError::serialization_error(format!("Zstd decompression error: {}", e)))?
            }
            _ => {
                return Err(VectorError::serialization_error(
                    "Unsupported compression type",
                ));
            }
        };

        let deserialized = match self.config.format {
            SerializationFormat::Bincode => bincode::deserialize(&decompressed)?,
            SerializationFormat::MessagePack => rmp_serde::from_slice(&decompressed)
                .map_err(|e| VectorError::serialization_error(format!("MessagePack error: {}", e)))?,
            SerializationFormat::Json => serde_json::from_slice(&decompressed)?,
            _ => {
                return Err(VectorError::serialization_error(
                    "Unsupported serialization format",
                ));
            }
        };

        Ok(deserialized)
    }

    async fn ensure_directories(&self) -> Result<()> {
        let vectors_dir = format!("{}/vectors", self.config.base_path);
        let metadata_dir = format!("{}/metadata", self.config.base_path);
        let backups_dir = format!("{}/backups", self.config.base_path);

        fs::create_dir_all(vectors_dir).await?;
        fs::create_dir_all(metadata_dir).await?;
        fs::create_dir_all(backups_dir).await?;

        Ok(())
    }
}

#[async_trait]
impl VectorStorage for FileSystemStorage {
    async fn save_vectors(&self, ids: &[String], texts: &[String]) -> Result<()> {
        if ids.len() != texts.len() {
            return Err(VectorError::storage_error(
                "IDs and texts length mismatch",
            ));
        }

        self.ensure_directories().await?;

        for (id, text) in ids.iter().zip(texts.iter()) {
            let vector_path = self.get_vector_path(id);
            let metadata_path = self.get_metadata_path(id);

            // Create directory for vector file if needed
            if let Some(parent) = Path::new(&vector_path).parent() {
                fs::create_dir_all(parent).await?;
            }
            if let Some(parent) = Path::new(&metadata_path).parent() {
                fs::create_dir_all(parent).await?;
            }

            // Save vector text
            let serialized_text = self.serialize_data(&text).await?;
            fs::write(&vector_path, serialized_text).await?;

            // Update cache
            {
                let mut cache = self.vector_cache.write();
                cache.put(id.clone(), text.clone());
            }

            // Save metadata
            let metadata = VectorMetadata {
                id: id.clone(),
                dimension: 0, // Will be set by the caller if needed
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                tags: Vec::new(),
                custom_metadata: serde_json::Value::Null,
            };

            let serialized_metadata = self.serialize_data(&metadata).await?;
            fs::write(&metadata_path, serialized_metadata).await?;

            // Update metadata cache
            {
                let mut cache = self.metadata_cache.write();
                cache.put(id.clone(), metadata);
            }
        }

        Ok(())
    }

    async fn load_vectors(&self, ids: &[String]) -> Result<HashMap<String, String>> {
        let mut results = HashMap::new();

        for id in ids {
            // Check cache first
            {
                let cache = self.vector_cache.read();
                if let Some(text) = cache.peek(id) {
                    results.insert(id.clone(), text.clone());
                    continue;
                }
            }

            // Load from disk
            let vector_path = self.get_vector_path(id);
            if Path::new(&vector_path).exists() {
                let data = fs::read(&vector_path).await?;
                let text: String = self.deserialize_data(&data).await?;

                // Update cache
                {
                    let mut cache = self.vector_cache.write();
                    cache.put(id.clone(), text.clone());
                }

                results.insert(id.clone(), text);
            }
        }

        Ok(results)
    }

    async fn remove_vectors(&self, ids: &[String]) -> Result<()> {
        for id in ids {
            let vector_path = self.get_vector_path(id);
            let metadata_path = self.get_metadata_path(id);

            // Remove files if they exist
            if Path::new(&vector_path).exists() {
                fs::remove_file(&vector_path).await?;
            }
            if Path::new(&metadata_path).exists() {
                fs::remove_file(&metadata_path).await?;
            }

            // Remove from caches
            {
                let mut cache = self.vector_cache.write();
                cache.pop(id);
            }
            {
                let mut cache = self.metadata_cache.write();
                cache.pop(id);
            }
        }

        Ok(())
    }

    async fn save_metadata(&self, path: &str) -> Result<()> {
        self.ensure_directories().await?;

        let index_data = IndexMetadata {
            created_at: chrono::Utc::now(),
            config: self.config.clone(),
            version: "1.0".to_string(),
        };

        let serialized = self.serialize_data(&index_data).await?;
        fs::write(path, serialized).await?;

        Ok(())
    }

    async fn load_metadata(&self, path: &str) -> Result<()> {
        if !Path::new(path).exists() {
            return Ok(()); // No metadata to load
        }

        let data = fs::read(path).await?;
        let _index_data: IndexMetadata = self.deserialize_data(&data).await?;

        // TODO: Validate compatibility and update config if needed

        Ok(())
    }

    async fn list_vector_ids(&self) -> Result<Vec<String>> {
        let vectors_dir = format!("{}/vectors", self.config.base_path);
        let mut ids = Vec::new();

        if Path::new(&vectors_dir).exists() {
            let mut entries = fs::read_dir(&vectors_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                if let Some(file_name) = entry.file_name().to_str() {
                    if file_name.ends_with(".vec") {
                        let id = file_name.trim_end_matches(".vec").to_string();
                        ids.push(id);
                    }
                }
            }
        }

        Ok(ids)
    }

    async fn get_stats(&self) -> Result<StorageStats> {
        let vector_ids = self.list_vector_ids().await?;
        let total_vectors = vector_ids.len();

        let mut total_size_bytes = 0u64;
        let vectors_dir = format!("{}/vectors", self.config.base_path);

        if Path::new(&vectors_dir).exists() {
            let mut entries = fs::read_dir(&vectors_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let metadata = entry.metadata().await?;
                total_size_bytes += metadata.len();
            }
        }

        // Calculate cache hit rate (simplified)
        let cache_hit_rate = 0.75; // Placeholder

        Ok(StorageStats {
            total_vectors,
            total_size_bytes,
            compression_ratio: 0.6, // Placeholder
            last_backup: None,
            cache_hit_rate,
        })
    }

    async fn optimize(&self) -> Result<()> {
        // Cleanup empty directories
        let vectors_dir = format!("{}/vectors", self.config.base_path);
        let metadata_dir = format!("{}/metadata", self.config.base_path);

        // TODO: Implement defragmentation, compression optimization, etc.

        log::info!("Storage optimization completed");
        Ok(())
    }

    async fn backup(&self, backup_path: &str) -> Result<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let full_backup_path = format!("{}/backup_{}", backup_path, timestamp);

        // Create backup directory
        fs::create_dir_all(&full_backup_path).await?;

        // Copy all vector and metadata files
        let base_path = &self.config.base_path;
        let copy_options = fs_extra::dir::CopyOptions::new();

        // This is a simplified backup - in practice, you'd want more sophisticated handling
        tokio::task::spawn_blocking(move || {
            fs_extra::dir::copy(base_path, &full_backup_path, &copy_options)
        })
        .await??;

        log::info!("Backup created at: {}", full_backup_path);
        Ok(())
    }

    async fn restore(&self, backup_path: &str) -> Result<()> {
        if !Path::new(backup_path).exists() {
            return Err(VectorError::storage_error("Backup path does not exist"));
        }

        // Clear current data
        if Path::new(&self.config.base_path).exists() {
            fs::remove_dir_all(&self.config.base_path).await?;
        }

        // Restore from backup
        let copy_options = fs_extra::dir::CopyOptions::new();
        let base_path = self.config.base_path.clone();

        tokio::task::spawn_blocking(move || {
            fs_extra::dir::copy(backup_path, &base_path, &copy_options)
        })
        .await??;

        // Clear caches
        {
            let mut cache = self.vector_cache.write();
            cache.clear();
        }
        {
            let mut cache = self.metadata_cache.write();
            cache.clear();
        }

        log::info!("Restored from backup: {}", backup_path);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexMetadata {
    created_at: chrono::DateTime<chrono::Utc>,
    config: StorageConfig,
    version: String,
}

/// In-memory storage implementation for testing and small datasets
pub struct MemoryStorage {
    vectors: parking_lot::RwLock<HashMap<String, String>>,
    metadata: parking_lot::RwLock<HashMap<String, VectorMetadata>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            vectors: parking_lot::RwLock::new(HashMap::new()),
            metadata: parking_lot::RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl VectorStorage for MemoryStorage {
    async fn save_vectors(&self, ids: &[String], texts: &[String]) -> Result<()> {
        if ids.len() != texts.len() {
            return Err(VectorError::storage_error(
                "IDs and texts length mismatch",
            ));
        }

        let mut vectors = self.vectors.write();
        let mut metadata = self.metadata.write();

        for (id, text) in ids.iter().zip(texts.iter()) {
            vectors.insert(id.clone(), text.clone());

            let meta = VectorMetadata {
                id: id.clone(),
                dimension: 0,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                tags: Vec::new(),
                custom_metadata: serde_json::Value::Null,
            };
            metadata.insert(id.clone(), meta);
        }

        Ok(())
    }

    async fn load_vectors(&self, ids: &[String]) -> Result<HashMap<String, String>> {
        let vectors = self.vectors.read();
        let mut results = HashMap::new();

        for id in ids {
            if let Some(text) = vectors.get(id) {
                results.insert(id.clone(), text.clone());
            }
        }

        Ok(results)
    }

    async fn remove_vectors(&self, ids: &[String]) -> Result<()> {
        let mut vectors = self.vectors.write();
        let mut metadata = self.metadata.write();

        for id in ids {
            vectors.remove(id);
            metadata.remove(id);
        }

        Ok(())
    }

    async fn save_metadata(&self, _path: &str) -> Result<()> {
        // No-op for memory storage
        Ok(())
    }

    async fn load_metadata(&self, _path: &str) -> Result<()> {
        // No-op for memory storage
        Ok(())
    }

    async fn list_vector_ids(&self) -> Result<Vec<String>> {
        let vectors = self.vectors.read();
        Ok(vectors.keys().cloned().collect())
    }

    async fn get_stats(&self) -> Result<StorageStats> {
        let vectors = self.vectors.read();
        let total_size: u64 = vectors.values().map(|v| v.len() as u64).sum();

        Ok(StorageStats {
            total_vectors: vectors.len(),
            total_size_bytes: total_size,
            compression_ratio: 1.0,
            last_backup: None,
            cache_hit_rate: 1.0,
        })
    }

    async fn optimize(&self) -> Result<()> {
        // No optimization needed for memory storage
        Ok(())
    }

    async fn backup(&self, _backup_path: &str) -> Result<()> {
        // TODO: Implement memory storage backup
        Ok(())
    }

    async fn restore(&self, _backup_path: &str) -> Result<()> {
        // TODO: Implement memory storage restore
        Ok(())
    }
}

/// Factory function to create storage backend
pub async fn create_storage(config: &crate::VectorStoreConfig) -> Result<Box<dyn VectorStorage>> {
    let storage_config = StorageConfig::default();

    // For now, always use FileSystem storage
    // In the future, this could be configurable
    Ok(Box::new(FileSystemStorage::new(storage_config).await?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = MemoryStorage::new();

        let ids = vec!["id1".to_string(), "id2".to_string()];
        let texts = vec!["text1".to_string(), "text2".to_string()];

        storage.save_vectors(&ids, &texts).await.unwrap();

        let loaded = storage.load_vectors(&ids).await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("id1"), Some(&"text1".to_string()));
        assert_eq!(loaded.get("id2"), Some(&"text2".to_string()));

        let all_ids = storage.list_vector_ids().await.unwrap();
        assert_eq!(all_ids.len(), 2);

        storage.remove_vectors(&["id1".to_string()]).await.unwrap();
        let remaining = storage.list_vector_ids().await.unwrap();
        assert_eq!(remaining.len(), 1);
    }

    #[tokio::test]
    async fn test_filesystem_storage() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            base_path: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };

        let storage = FileSystemStorage::new(config).await.unwrap();

        let ids = vec!["id1".to_string(), "id2".to_string()];
        let texts = vec!["text1".to_string(), "text2".to_string()];

        storage.save_vectors(&ids, &texts).await.unwrap();

        let loaded = storage.load_vectors(&ids).await.unwrap();
        assert_eq!(loaded.len(), 2);

        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_vectors, 2);
    }
}