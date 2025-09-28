//! Vectorstore - High-performance vector embeddings and similarity search
//!
//! This crate provides comprehensive vector storage and retrieval functionality
//! with support for multiple indexing backends, embedding models, and GPU acceleration.

pub mod embedding;
pub mod index;
pub mod search;
pub mod storage;
pub mod metrics;
pub mod operations;
pub mod hooks;
pub mod error;

// Re-export main types for convenience
pub use embedding::{BaseEmbeddingModel, EmbeddingConfig, EmbeddingResult, TransformerEmbedding, MockEmbeddingModel};
pub use index::{
    VectorIndex, IndexConfig, IndexType, HnswIndex, FlatIndex, DistanceMetric, IndexParameters
};
#[cfg(feature = "faiss-backend")]
pub use index::FaissIndex;
pub use search::{SearchQuery, SearchResult, SearchEngine, SimilarityMetric, QueryFilter, RerankMethod};
pub use storage::{VectorStorage, StorageConfig, SerializationFormat, CompressionType, FileSystemStorage, MemoryStorage};
pub use operations::{VectorOperationsEngine, GpuConfig, GpuAccelerated};
pub use hooks::{HookManager, Hook, HookContext, HookResult, HookFactory};
pub use metrics::{MetricsCollector, PerformanceMetrics, OperationType, PerformanceMonitor};
pub use error::{VectorError, Result};

// Main vector store types are defined below, not re-exported from crate

use std::sync::Arc;
use tokio::sync::RwLock;

/// Main vectorstore manager that coordinates all components
pub struct VectorStore {
    embedding_model: Arc<dyn BaseEmbeddingModel>,
    index: Arc<RwLock<dyn VectorIndex>>,
    storage: Arc<dyn VectorStorage>,
    search_engine: Arc<SearchEngine>,
    config: VectorStoreConfig,
}

/// Configuration for the vector store
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VectorStoreConfig {
    pub dimension: usize,
    pub index_type: IndexType,
    pub similarity_metric: SimilarityMetric,
    pub enable_gpu: bool,
    pub enable_compression: bool,
    pub batch_size: usize,
    pub num_threads: Option<usize>,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            index_type: IndexType::Flat,
            similarity_metric: SimilarityMetric::Cosine,
            enable_gpu: false,
            enable_compression: false,
            batch_size: 1000,
            num_threads: None,
        }
    }
}

impl VectorStore {
    /// Create a new vector store with the specified configuration
    pub async fn new(
        embedding_model: Arc<dyn BaseEmbeddingModel>,
        config: VectorStoreConfig,
    ) -> Result<Self> {
        let index = index::create_index(&config).await?;
        let storage = storage::create_storage(&config).await?;
        let search_engine = Arc::new(SearchEngine::new(config.similarity_metric.clone()));

        Ok(Self {
            embedding_model,
            index: Arc::new(RwLock::new(index)),
            storage,
            search_engine,
            config,
        })
    }

    /// Create a new vector store with advanced features enabled
    pub async fn new_with_features(
        embedding_model: Arc<dyn BaseEmbeddingModel>,
        config: VectorStoreConfig,
        gpu_config: Option<GpuConfig>,
        enable_hooks: bool,
    ) -> Result<VectorStoreWithFeatures> {
        let base_store = Self::new(embedding_model.clone(), config.clone()).await?;

        let metrics = Arc::new(MetricsCollector::new());

        let operations_engine = if let Some(gpu_cfg) = gpu_config {
            Some(VectorOperationsEngine::new(gpu_cfg, metrics.clone()).await?)
        } else {
            None
        };

        let hook_manager = if enable_hooks {
            Some(HookManager::new())
        } else {
            None
        };

        Ok(VectorStoreWithFeatures {
            base_store,
            operations_engine,
            hook_manager,
            metrics,
        })
    }

    /// Add vectors to the store with associated metadata
    pub async fn add_vectors(
        &self,
        texts: Vec<String>,
        metadata: Vec<serde_json::Value>,
    ) -> Result<Vec<String>> {
        // Generate embeddings in batches
        let embeddings = self.embedding_model.embed_batch(&texts).await?;

        // Store vectors in the index
        let mut index = self.index.write().await;
        let ids = index.add_vectors(embeddings, metadata).await?;

        // Persist to storage
        self.storage.save_vectors(&ids, &texts).await?;

        Ok(ids)
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        query: &str,
        k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let query_embedding = self.embedding_model.embed(query).await?;

        // Search in index
        let index = self.index.read().await;
        let results = index.search(&query_embedding, k, filter).await?;

        Ok(results)
    }

    /// Get vector by ID
    pub async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let index = self.index.read().await;
        index.get_vector(id).await
    }

    /// Remove vectors by IDs
    pub async fn remove_vectors(&self, ids: &[String]) -> Result<()> {
        let mut index = self.index.write().await;
        index.remove_vectors(ids).await?;
        self.storage.remove_vectors(ids).await?;
        Ok(())
    }

    /// Save the current state to disk
    pub async fn save(&self, path: &str) -> Result<()> {
        let index = self.index.read().await;
        index.save(path).await?;
        self.storage.save_metadata(path).await?;
        Ok(())
    }

    /// Load state from disk
    pub async fn load(&self, path: &str) -> Result<()> {
        let mut index = self.index.write().await;
        index.load(path).await?;
        self.storage.load_metadata(path).await?;
        Ok(())
    }

    /// Get statistics about the vector store
    pub async fn stats(&self) -> Result<VectorStoreStats> {
        let index = self.index.read().await;
        Ok(VectorStoreStats {
            total_vectors: index.size().await?,
            dimension: self.config.dimension,
            index_type: self.config.index_type.clone(),
            memory_usage: index.memory_usage().await?,
        })
    }
}

/// Statistics about the vector store
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VectorStoreStats {
    pub total_vectors: usize,
    pub dimension: usize,
    pub index_type: IndexType,
    pub memory_usage: usize,
}

/// Enhanced vector store with GPU acceleration, hooks, and metrics
pub struct VectorStoreWithFeatures {
    base_store: VectorStore,
    operations_engine: Option<VectorOperationsEngine>,
    hook_manager: Option<HookManager>,
    metrics: Arc<MetricsCollector>,
}

impl VectorStoreWithFeatures {
    /// Add vectors with enhanced features (GPU acceleration, hooks, metrics)
    pub async fn add_vectors_enhanced(
        &self,
        texts: Vec<String>,
        metadata: Vec<serde_json::Value>,
    ) -> Result<Vec<String>> {
        let operation_id = uuid::Uuid::new_v4().to_string();

        // Execute with hooks if enabled
        if let Some(hook_manager) = &self.hook_manager {
            let context = HookContext {
                operation_type: "add_vectors".to_string(),
                operation_id: operation_id.clone(),
                metadata: std::collections::HashMap::new(),
                timestamp: chrono::Utc::now(),
            };
            hook_manager.execute_hooks(&context).await?;
        }

        // Use GPU-accelerated embedding if available
        let embeddings = if let Some(engine) = &self.operations_engine {
            engine.batch_embed(
                self.base_store.embedding_model.as_ref(),
                &texts,
                Some(32)
            ).await?
        } else {
            self.base_store.embedding_model.embed_batch(&texts).await?
        };

        // Add to index
        let mut index = self.base_store.index.write().await;
        let ids = index.add_vectors(embeddings.embeddings, metadata).await?;

        // Store vectors
        self.base_store.storage.save_vectors(&ids, &texts).await?;

        Ok(ids)
    }

    /// Enhanced search with GPU acceleration and hooks
    pub async fn search_enhanced(
        &self,
        query: &str,
        k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        let operation_id = uuid::Uuid::new_v4().to_string();

        // Execute hooks if enabled
        if let Some(hook_manager) = &self.hook_manager {
            let context = HookContext {
                operation_type: "search".to_string(),
                operation_id: operation_id.clone(),
                metadata: std::collections::HashMap::new(),
                timestamp: chrono::Utc::now(),
            };
            hook_manager.execute_hooks(&context).await?;
        }

        // Generate query embedding
        let query_embedding = self.base_store.embedding_model.embed(query).await?;

        // Search with enhanced operations if available
        let results = if let Some(_engine) = &self.operations_engine {
            // Could use GPU-accelerated search here
            let index = self.base_store.index.read().await;
            index.search(&query_embedding, k, filter).await?
        } else {
            let index = self.base_store.index.read().await;
            index.search(&query_embedding, k, filter).await?
        };

        Ok(results)
    }

    /// Get comprehensive performance metrics
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        self.metrics.get_metrics().await
    }

    /// Get hook execution metrics
    pub async fn get_hook_metrics(&self) -> Result<std::collections::HashMap<String, Vec<HookResult>>> {
        if let Some(hook_manager) = &self.hook_manager {
            hook_manager.get_hook_metrics().await
        } else {
            Ok(std::collections::HashMap::new())
        }
    }

    /// Get GPU memory usage if available
    pub async fn get_gpu_memory_usage(&self) -> Result<Option<(u64, u64)>> {
        if let Some(engine) = &self.operations_engine {
            engine.get_gpu_memory_usage().await
        } else {
            Ok(None)
        }
    }

    /// Shutdown and cleanup resources
    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(engine) = &mut self.operations_engine {
            engine.cleanup().await?;
        }

        if let Some(hook_manager) = &self.hook_manager {
            hook_manager.clear_metrics().await;
        }

        Ok(())
    }

    /// Access the base vector store
    pub fn base_store(&self) -> &VectorStore {
        &self.base_store
    }

    /// Access the operations engine
    pub fn operations_engine(&self) -> Option<&VectorOperationsEngine> {
        self.operations_engine.as_ref()
    }

    /// Access the hook manager
    pub fn hook_manager(&self) -> Option<&HookManager> {
        self.hook_manager.as_ref()
    }

    /// Access the metrics collector
    pub fn metrics(&self) -> &MetricsCollector {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbeddingModel;

    #[tokio::test]
    async fn test_vector_store_creation() {
        let embedding_model = Arc::new(MockEmbeddingModel::new(768));
        let config = VectorStoreConfig::default();

        let store = VectorStore::new(embedding_model, config).await;
        assert!(store.is_ok());
    }
}