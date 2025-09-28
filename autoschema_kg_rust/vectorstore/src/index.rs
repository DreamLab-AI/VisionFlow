//! Vector indexing implementations with support for multiple backends

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Result, VectorError};
use crate::search::SearchResult;

/// Configuration for vector indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub index_type: IndexType,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub parameters: IndexParameters,
}

/// Types of vector indices supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    Flat,
    Hnsw,
    #[cfg(feature = "faiss-backend")]
    FaissFlat,
    #[cfg(feature = "faiss-backend")]
    FaissHnsw,
    #[cfg(feature = "faiss-backend")]
    FaissIvf,
    #[cfg(feature = "hnswlib-backend")]
    HnswLib,
}

/// Distance metrics for similarity computation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    InnerProduct,
    Manhattan,
}

/// Index-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParameters {
    pub hnsw_m: Option<usize>,           // HNSW connections per node
    pub hnsw_ef_construction: Option<usize>, // HNSW construction parameter
    pub hnsw_ef_search: Option<usize>,   // HNSW search parameter
    pub ivf_nlist: Option<usize>,        // IVF number of clusters
    pub ivf_nprobe: Option<usize>,       // IVF number of probes
    pub pq_m: Option<usize>,             // Product quantization parameter
    pub gpu_device_id: Option<usize>,    // GPU device for computation
}

impl Default for IndexParameters {
    fn default() -> Self {
        Self {
            hnsw_m: Some(16),
            hnsw_ef_construction: Some(200),
            hnsw_ef_search: Some(50),
            ivf_nlist: Some(1024),
            ivf_nprobe: Some(8),
            pq_m: Some(8),
            gpu_device_id: None,
        }
    }
}

/// Base trait for vector indices
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Add vectors to the index
    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<serde_json::Value>,
    ) -> Result<Vec<String>>;

    /// Search for k nearest neighbors
    async fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>>;

    /// Get vector by ID
    async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>>;

    /// Remove vectors by IDs
    async fn remove_vectors(&mut self, ids: &[String]) -> Result<()>;

    /// Get the number of vectors in the index
    async fn size(&self) -> Result<usize>;

    /// Get memory usage in bytes
    async fn memory_usage(&self) -> Result<usize>;

    /// Save index to disk
    async fn save(&self, path: &str) -> Result<()>;

    /// Load index from disk
    async fn load(&mut self, path: &str) -> Result<()>;

    /// Get index statistics
    async fn stats(&self) -> Result<IndexStats>;

    /// Optimize the index (rebuild, compress, etc.)
    async fn optimize(&mut self) -> Result<()>;
}

/// Statistics about an index
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_vectors: usize,
    pub dimension: usize,
    pub index_type: IndexType,
    pub memory_usage_bytes: usize,
    pub build_time_ms: Option<u64>,
    pub last_optimized: Option<chrono::DateTime<chrono::Utc>>,
}

/// Simple flat index implementation (linear search)
pub struct FlatIndex {
    vectors: Vec<Vec<f32>>,
    metadata: Vec<serde_json::Value>,
    ids: Vec<String>,
    config: IndexConfig,
    id_to_index: HashMap<String, usize>,
}

impl FlatIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            vectors: Vec::new(),
            metadata: Vec::new(),
            ids: Vec::new(),
            config,
            id_to_index: HashMap::new(),
        }
    }

    fn compute_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VectorError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let distance = match self.config.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            DistanceMetric::Cosine => {
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                1.0 - (dot_product / (norm_a * norm_b))
            }
            DistanceMetric::InnerProduct => {
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
            DistanceMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
            }
        };

        Ok(distance)
    }
}

#[async_trait]
impl VectorIndex for FlatIndex {
    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<serde_json::Value>,
    ) -> Result<Vec<String>> {
        if vectors.len() != metadata.len() {
            return Err(VectorError::config_error(
                "Vectors and metadata length mismatch",
            ));
        }

        let mut ids = Vec::new();
        for (i, (vector, meta)) in vectors.into_iter().zip(metadata.into_iter()).enumerate() {
            if vector.len() != self.config.dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: self.config.dimension,
                    actual: vector.len(),
                });
            }

            let id = uuid::Uuid::new_v4().to_string();
            let index = self.vectors.len();

            self.vectors.push(vector);
            self.metadata.push(meta);
            self.ids.push(id.clone());
            self.id_to_index.insert(id.clone(), index);

            ids.push(id);
        }

        Ok(ids)
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        _filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let mut results: Vec<(f32, usize)> = Vec::new();

        for (i, vector) in self.vectors.iter().enumerate() {
            let distance = self.compute_distance(query, vector)?;
            results.push((distance, i));
        }

        // Sort by distance and take top k
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(k);

        let search_results = results
            .into_iter()
            .map(|(distance, index)| SearchResult {
                id: self.ids[index].clone(),
                score: 1.0 - distance, // Convert distance to similarity score
                metadata: self.metadata[index].clone(),
                vector: Some(self.vectors[index].clone()),
            })
            .collect();

        Ok(search_results)
    }

    async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>> {
        if let Some(&index) = self.id_to_index.get(id) {
            Ok(Some(self.vectors[index].clone()))
        } else {
            Ok(None)
        }
    }

    async fn remove_vectors(&mut self, ids: &[String]) -> Result<()> {
        let mut indices_to_remove: Vec<usize> = ids
            .iter()
            .filter_map(|id| self.id_to_index.get(id))
            .copied()
            .collect();

        // Sort in reverse order to remove from the end first
        indices_to_remove.sort_by(|a, b| b.cmp(a));

        for index in indices_to_remove {
            let removed_id = self.ids.remove(index);
            self.vectors.remove(index);
            self.metadata.remove(index);
            self.id_to_index.remove(&removed_id);

            // Update indices for elements that moved
            for (id, idx) in self.id_to_index.iter_mut() {
                if *idx > index {
                    *idx -= 1;
                }
            }
        }

        Ok(())
    }

    async fn size(&self) -> Result<usize> {
        Ok(self.vectors.len())
    }

    async fn memory_usage(&self) -> Result<usize> {
        let vectors_size = self.vectors.len() * self.config.dimension * std::mem::size_of::<f32>();
        let metadata_size = self.metadata.capacity() * std::mem::size_of::<serde_json::Value>();
        let ids_size = self.ids.iter().map(|s| s.len()).sum::<usize>();
        Ok(vectors_size + metadata_size + ids_size)
    }

    async fn save(&self, path: &str) -> Result<()> {
        let data = IndexData {
            vectors: self.vectors.clone(),
            metadata: self.metadata.clone(),
            ids: self.ids.clone(),
            config: self.config.clone(),
        };

        let encoded = bincode::serialize(&data)?;
        tokio::fs::write(path, encoded).await?;
        Ok(())
    }

    async fn load(&mut self, path: &str) -> Result<()> {
        let data = tokio::fs::read(path).await?;
        let decoded: IndexData = bincode::deserialize(&data)?;

        self.vectors = decoded.vectors;
        self.metadata = decoded.metadata;
        self.ids = decoded.ids;
        self.config = decoded.config;

        // Rebuild the ID to index mapping
        self.id_to_index.clear();
        for (index, id) in self.ids.iter().enumerate() {
            self.id_to_index.insert(id.clone(), index);
        }

        Ok(())
    }

    async fn stats(&self) -> Result<IndexStats> {
        Ok(IndexStats {
            total_vectors: self.vectors.len(),
            dimension: self.config.dimension,
            index_type: self.config.index_type.clone(),
            memory_usage_bytes: self.memory_usage().await?,
            build_time_ms: None,
            last_optimized: None,
        })
    }

    async fn optimize(&mut self) -> Result<()> {
        // For flat index, optimization could involve compaction
        self.vectors.shrink_to_fit();
        self.metadata.shrink_to_fit();
        self.ids.shrink_to_fit();
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct IndexData {
    vectors: Vec<Vec<f32>>,
    metadata: Vec<serde_json::Value>,
    ids: Vec<String>,
    config: IndexConfig,
}

/// HNSW implementation using a simplified approach
pub struct HnswIndex {
    config: IndexConfig,
    vectors: Vec<Vec<f32>>,
    metadata: Vec<serde_json::Value>,
    ids: Vec<String>,
    id_to_index: HashMap<String, usize>,
    // HNSW graph structure (simplified)
    connections: Vec<Vec<usize>>,
    entry_point: Option<usize>,
}

impl HnswIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            vectors: Vec::new(),
            metadata: Vec::new(),
            ids: Vec::new(),
            id_to_index: HashMap::new(),
            connections: Vec::new(),
            entry_point: None,
        }
    }

    fn build_hnsw_connections(&mut self, new_index: usize) -> Result<()> {
        let m = self.config.parameters.hnsw_m.unwrap_or(16);
        let ef_construction = self.config.parameters.hnsw_ef_construction.unwrap_or(200);

        if self.vectors.is_empty() {
            self.connections.push(Vec::new());
            self.entry_point = Some(0);
            return Ok(());
        }

        // Simplified HNSW construction
        let mut candidates = Vec::new();
        for (i, existing_vector) in self.vectors[..new_index].iter().enumerate() {
            let distance = self.compute_distance(&self.vectors[new_index], existing_vector)?;
            candidates.push((distance, i));
        }

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(ef_construction.min(candidates.len()));

        let mut connections = Vec::new();
        for (_, candidate_idx) in candidates.into_iter().take(m) {
            connections.push(candidate_idx);
            // Add bidirectional connection
            if self.connections[candidate_idx].len() < m {
                self.connections[candidate_idx].push(new_index);
            }
        }

        self.connections.push(connections);
        Ok(())
    }

    fn compute_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(VectorError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let distance = match self.config.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            DistanceMetric::Cosine => {
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                1.0 - (dot_product / (norm_a * norm_b))
            }
            DistanceMetric::InnerProduct => {
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
            DistanceMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
            }
        };

        Ok(distance)
    }

    fn hnsw_search(&self, query: &[f32], k: usize) -> Result<Vec<(f32, usize)>> {
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let ef_search = self.config.parameters.hnsw_ef_search.unwrap_or(50);
        let entry_point = self.entry_point.unwrap_or(0);

        let mut candidates = std::collections::BinaryHeap::new();
        let mut visited = std::collections::HashSet::new();
        let mut dynamic_list = std::collections::BinaryHeap::new();

        // Start from entry point
        let distance = self.compute_distance(query, &self.vectors[entry_point])?;
        candidates.push(std::cmp::Reverse((
            ordered_float::OrderedFloat(distance),
            entry_point,
        )));
        dynamic_list.push((ordered_float::OrderedFloat(distance), entry_point));
        visited.insert(entry_point);

        while let Some(std::cmp::Reverse((curr_dist, curr_idx))) = candidates.pop() {
            if dynamic_list.len() >= ef_search
                && curr_dist > dynamic_list.peek().unwrap().0
            {
                break;
            }

            for &neighbor_idx in &self.connections[curr_idx] {
                if visited.contains(&neighbor_idx) {
                    continue;
                }
                visited.insert(neighbor_idx);

                let distance = self.compute_distance(query, &self.vectors[neighbor_idx])?;
                let ordered_dist = ordered_float::OrderedFloat(distance);

                if dynamic_list.len() < ef_search {
                    candidates.push(std::cmp::Reverse((ordered_dist, neighbor_idx)));
                    dynamic_list.push((ordered_dist, neighbor_idx));
                } else if ordered_dist < dynamic_list.peek().unwrap().0 {
                    candidates.push(std::cmp::Reverse((ordered_dist, neighbor_idx)));
                    dynamic_list.push((ordered_dist, neighbor_idx));
                    dynamic_list.pop();
                }
            }
        }

        let mut results: Vec<_> = dynamic_list
            .into_iter()
            .map(|(dist, idx)| (dist.into_inner(), idx))
            .collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(k);

        Ok(results)
    }
}

#[async_trait]
impl VectorIndex for HnswIndex {
    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<serde_json::Value>,
    ) -> Result<Vec<String>> {
        if vectors.len() != metadata.len() {
            return Err(VectorError::config_error(
                "Vectors and metadata length mismatch",
            ));
        }

        let mut ids = Vec::new();
        for (vector, meta) in vectors.into_iter().zip(metadata.into_iter()) {
            if vector.len() != self.config.dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: self.config.dimension,
                    actual: vector.len(),
                });
            }

            let id = uuid::Uuid::new_v4().to_string();
            let index = self.vectors.len();

            self.vectors.push(vector);
            self.metadata.push(meta);
            self.ids.push(id.clone());
            self.id_to_index.insert(id.clone(), index);

            // Build HNSW connections for the new vector
            self.build_hnsw_connections(index)?;

            ids.push(id);
        }

        Ok(ids)
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        _filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let results = self.hnsw_search(query, k)?;

        let search_results = results
            .into_iter()
            .map(|(distance, index)| SearchResult {
                id: self.ids[index].clone(),
                score: 1.0 - distance, // Convert distance to similarity score
                metadata: self.metadata[index].clone(),
                vector: Some(self.vectors[index].clone()),
            })
            .collect();

        Ok(search_results)
    }

    async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>> {
        if let Some(&index) = self.id_to_index.get(id) {
            Ok(Some(self.vectors[index].clone()))
        } else {
            Ok(None)
        }
    }

    async fn remove_vectors(&mut self, ids: &[String]) -> Result<()> {
        // This is a simplified implementation
        // In practice, HNSW removal is complex and may require rebuilding
        let mut indices_to_remove: Vec<usize> = ids
            .iter()
            .filter_map(|id| self.id_to_index.get(id))
            .copied()
            .collect();

        indices_to_remove.sort_by(|a, b| b.cmp(a));

        for index in indices_to_remove {
            self.vectors.remove(index);
            self.metadata.remove(index);
            self.connections.remove(index);
            let removed_id = self.ids.remove(index);
            self.id_to_index.remove(&removed_id);

            // Update connections and indices
            for connections in &mut self.connections {
                connections.retain(|&conn_idx| conn_idx != index);
                for conn_idx in connections.iter_mut() {
                    if *conn_idx > index {
                        *conn_idx -= 1;
                    }
                }
            }

            // Update ID mappings
            for (id, idx) in self.id_to_index.iter_mut() {
                if *idx > index {
                    *idx -= 1;
                }
            }

            // Update entry point if necessary
            if let Some(ref mut entry) = self.entry_point {
                if *entry == index {
                    *entry = if self.vectors.is_empty() { 0 } else { 0 };
                } else if *entry > index {
                    *entry -= 1;
                }
            }
        }

        Ok(())
    }

    async fn size(&self) -> Result<usize> {
        Ok(self.vectors.len())
    }

    async fn memory_usage(&self) -> Result<usize> {
        let vectors_size = self.vectors.len() * self.config.dimension * std::mem::size_of::<f32>();
        let connections_size = self.connections.iter().map(|c| c.len() * std::mem::size_of::<usize>()).sum::<usize>();
        let metadata_size = self.metadata.capacity() * std::mem::size_of::<serde_json::Value>();
        let ids_size = self.ids.iter().map(|s| s.len()).sum::<usize>();
        Ok(vectors_size + connections_size + metadata_size + ids_size)
    }

    async fn save(&self, path: &str) -> Result<()> {
        let data = HnswIndexData {
            vectors: self.vectors.clone(),
            metadata: self.metadata.clone(),
            ids: self.ids.clone(),
            config: self.config.clone(),
            connections: self.connections.clone(),
            entry_point: self.entry_point,
        };

        let encoded = bincode::serialize(&data)?;
        tokio::fs::write(path, encoded).await?;
        Ok(())
    }

    async fn load(&mut self, path: &str) -> Result<()> {
        let data = tokio::fs::read(path).await?;
        let decoded: HnswIndexData = bincode::deserialize(&data)?;

        self.vectors = decoded.vectors;
        self.metadata = decoded.metadata;
        self.ids = decoded.ids;
        self.config = decoded.config;
        self.connections = decoded.connections;
        self.entry_point = decoded.entry_point;

        // Rebuild the ID to index mapping
        self.id_to_index.clear();
        for (index, id) in self.ids.iter().enumerate() {
            self.id_to_index.insert(id.clone(), index);
        }

        Ok(())
    }

    async fn stats(&self) -> Result<IndexStats> {
        Ok(IndexStats {
            total_vectors: self.vectors.len(),
            dimension: self.config.dimension,
            index_type: self.config.index_type.clone(),
            memory_usage_bytes: self.memory_usage().await?,
            build_time_ms: None,
            last_optimized: None,
        })
    }

    async fn optimize(&mut self) -> Result<()> {
        // For HNSW, optimization could involve rebuilding the graph
        // This is a placeholder for more sophisticated optimization
        self.vectors.shrink_to_fit();
        self.metadata.shrink_to_fit();
        self.ids.shrink_to_fit();
        for connections in &mut self.connections {
            connections.shrink_to_fit();
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct HnswIndexData {
    vectors: Vec<Vec<f32>>,
    metadata: Vec<serde_json::Value>,
    ids: Vec<String>,
    config: IndexConfig,
    connections: Vec<Vec<usize>>,
    entry_point: Option<usize>,
}

/// FAISS backend implementation
#[cfg(feature = "faiss-backend")]
pub struct FaissIndex {
    index: Arc<RwLock<faiss::Index>>,
    config: IndexConfig,
    metadata: Arc<RwLock<Vec<serde_json::Value>>>,
    ids: Arc<RwLock<Vec<String>>>,
    id_to_index: Arc<RwLock<HashMap<String, i64>>>,
}

#[cfg(feature = "faiss-backend")]
impl FaissIndex {
    pub fn new(config: IndexConfig) -> Result<Self> {
        let index = match config.index_type {
            IndexType::FaissFlat => {
                let metric = match config.metric {
                    DistanceMetric::Euclidean => faiss::MetricType::L2,
                    DistanceMetric::Cosine => faiss::MetricType::InnerProduct,
                    DistanceMetric::InnerProduct => faiss::MetricType::InnerProduct,
                    DistanceMetric::Manhattan => faiss::MetricType::L1,
                };
                faiss::index_factory(config.dimension as i32, "Flat", metric)?
            }
            IndexType::FaissHnsw => {
                let m = config.parameters.hnsw_m.unwrap_or(16);
                let factory_string = format!("HNSW{}", m);
                let metric = match config.metric {
                    DistanceMetric::Euclidean => faiss::MetricType::L2,
                    DistanceMetric::Cosine => faiss::MetricType::InnerProduct,
                    DistanceMetric::InnerProduct => faiss::MetricType::InnerProduct,
                    DistanceMetric::Manhattan => faiss::MetricType::L1,
                };
                faiss::index_factory(config.dimension as i32, &factory_string, metric)?
            }
            IndexType::FaissIvf => {
                let nlist = config.parameters.ivf_nlist.unwrap_or(1024);
                let factory_string = format!("IVF{},Flat", nlist);
                let metric = match config.metric {
                    DistanceMetric::Euclidean => faiss::MetricType::L2,
                    DistanceMetric::Cosine => faiss::MetricType::InnerProduct,
                    DistanceMetric::InnerProduct => faiss::MetricType::InnerProduct,
                    DistanceMetric::Manhattan => faiss::MetricType::L1,
                };
                faiss::index_factory(config.dimension as i32, &factory_string, metric)?
            }
            _ => {
                return Err(VectorError::config_error("Invalid FAISS index type"));
            }
        };

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            config,
            metadata: Arc::new(RwLock::new(Vec::new())),
            ids: Arc::new(RwLock::new(Vec::new())),
            id_to_index: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}

#[cfg(feature = "faiss-backend")]
#[async_trait]
impl VectorIndex for FaissIndex {
    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<serde_json::Value>,
    ) -> Result<Vec<String>> {
        if vectors.len() != metadata.len() {
            return Err(VectorError::config_error(
                "Vectors and metadata length mismatch",
            ));
        }

        // Flatten vectors for FAISS
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        let mut index = self.index.write().await;
        let current_size = index.ntotal();

        // Add vectors to FAISS index
        index.add(&flat_vectors)?;

        // Update metadata and IDs
        let mut ids = Vec::new();
        let mut metadata_guard = self.metadata.write().await;
        let mut ids_guard = self.ids.write().await;
        let mut id_to_index_guard = self.id_to_index.write().await;

        for (i, meta) in metadata.into_iter().enumerate() {
            let id = uuid::Uuid::new_v4().to_string();
            let index_pos = current_size + i as i64;

            metadata_guard.push(meta);
            ids_guard.push(id.clone());
            id_to_index_guard.insert(id.clone(), index_pos);
            ids.push(id);
        }

        Ok(ids)
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        _filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dimension,
                actual: query.len(),
            });
        }

        let index = self.index.read().await;
        let metadata_guard = self.metadata.read().await;
        let ids_guard = self.ids.read().await;

        let (distances, indices) = index.search(query, k as i64)?;

        let mut results = Vec::new();
        for (i, (&idx, &distance)) in indices.iter().zip(distances.iter()).enumerate() {
            if idx < 0 {
                break; // FAISS uses -1 for invalid indices
            }

            let idx = idx as usize;
            if idx < ids_guard.len() {
                results.push(SearchResult {
                    id: ids_guard[idx].clone(),
                    score: 1.0 - distance, // Convert distance to similarity
                    metadata: metadata_guard[idx].clone(),
                    vector: None, // FAISS doesn't return vectors by default
                });
            }
        }

        Ok(results)
    }

    async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>> {
        let id_to_index_guard = self.id_to_index.read().await;
        if let Some(&index) = id_to_index_guard.get(id) {
            let index_guard = self.index.read().await;
            let vector = index_guard.reconstruct(index)?;
            Ok(Some(vector))
        } else {
            Ok(None)
        }
    }

    async fn remove_vectors(&mut self, ids: &[String]) -> Result<()> {
        // FAISS doesn't support direct removal, so we'd need to rebuild
        // This is a placeholder implementation
        Err(VectorError::index_error(
            "FAISS index removal not implemented - requires rebuild",
        ))
    }

    async fn size(&self) -> Result<usize> {
        let index = self.index.read().await;
        Ok(index.ntotal() as usize)
    }

    async fn memory_usage(&self) -> Result<usize> {
        // Approximate memory usage calculation
        let index = self.index.read().await;
        let vectors_size = index.ntotal() as usize * self.config.dimension * std::mem::size_of::<f32>();
        Ok(vectors_size)
    }

    async fn save(&self, path: &str) -> Result<()> {
        let index = self.index.read().await;
        index.write(path)?;

        // Save metadata separately
        let metadata_path = format!("{}.metadata", path);
        let metadata_guard = self.metadata.read().await;
        let ids_guard = self.ids.read().await;

        let metadata_data = FaissMetadata {
            metadata: metadata_guard.clone(),
            ids: ids_guard.clone(),
            config: self.config.clone(),
        };

        let encoded = bincode::serialize(&metadata_data)?;
        tokio::fs::write(metadata_path, encoded).await?;
        Ok(())
    }

    async fn load(&mut self, path: &str) -> Result<()> {
        let mut index = self.index.write().await;
        *index = faiss::read_index(path)?;

        // Load metadata
        let metadata_path = format!("{}.metadata", path);
        let data = tokio::fs::read(metadata_path).await?;
        let decoded: FaissMetadata = bincode::deserialize(&data)?;

        let mut metadata_guard = self.metadata.write().await;
        let mut ids_guard = self.ids.write().await;
        let mut id_to_index_guard = self.id_to_index.write().await;

        *metadata_guard = decoded.metadata;
        *ids_guard = decoded.ids;
        self.config = decoded.config;

        // Rebuild ID mapping
        id_to_index_guard.clear();
        for (i, id) in ids_guard.iter().enumerate() {
            id_to_index_guard.insert(id.clone(), i as i64);
        }

        Ok(())
    }

    async fn stats(&self) -> Result<IndexStats> {
        let index = self.index.read().await;
        Ok(IndexStats {
            total_vectors: index.ntotal() as usize,
            dimension: self.config.dimension,
            index_type: self.config.index_type.clone(),
            memory_usage_bytes: self.memory_usage().await?,
            build_time_ms: None,
            last_optimized: None,
        })
    }

    async fn optimize(&mut self) -> Result<()> {
        // FAISS-specific optimization could involve training for IVF indices
        let mut index = self.index.write().await;
        if !index.is_trained() {
            // For IVF indices, we might need training data
            // This is a placeholder
            return Err(VectorError::index_error("Index training not implemented"));
        }
        Ok(())
    }
}

#[cfg(feature = "faiss-backend")]
#[derive(Serialize, Deserialize)]
struct FaissMetadata {
    metadata: Vec<serde_json::Value>,
    ids: Vec<String>,
    config: IndexConfig,
}

/// Factory function to create appropriate index based on configuration
pub async fn create_index(config: &crate::VectorStoreConfig) -> Result<Box<dyn VectorIndex>> {
    let index_config = IndexConfig {
        index_type: config.index_type.clone(),
        dimension: config.dimension,
        metric: match config.similarity_metric {
            crate::search::SimilarityMetric::Cosine => DistanceMetric::Cosine,
            crate::search::SimilarityMetric::Euclidean => DistanceMetric::Euclidean,
            crate::search::SimilarityMetric::DotProduct => DistanceMetric::InnerProduct,
        },
        parameters: IndexParameters::default(),
    };

    match config.index_type {
        IndexType::Flat => Ok(Box::new(FlatIndex::new(index_config))),
        IndexType::Hnsw => Ok(Box::new(HnswIndex::new(index_config))),
        #[cfg(feature = "faiss-backend")]
        IndexType::FaissFlat | IndexType::FaissHnsw | IndexType::FaissIvf => {
            Ok(Box::new(FaissIndex::new(index_config)?))
        }
        #[cfg(not(feature = "faiss-backend"))]
        IndexType::FaissFlat | IndexType::FaissHnsw | IndexType::FaissIvf => {
            Err(VectorError::config_error("FAISS backend not enabled"))
        }
        _ => Err(VectorError::config_error("Unsupported index type")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_flat_index() {
        let config = IndexConfig {
            index_type: IndexType::Flat,
            dimension: 3,
            metric: DistanceMetric::Euclidean,
            parameters: IndexParameters::default(),
        };

        let mut index = FlatIndex::new(config);

        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let metadata = vec![
            serde_json::json!({"label": "x-axis"}),
            serde_json::json!({"label": "y-axis"}),
        ];

        let ids = index.add_vectors(vectors, metadata).await.unwrap();
        assert_eq!(ids.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0], 1, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, ids[0]);
    }

    #[tokio::test]
    async fn test_hnsw_index() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            dimension: 3,
            metric: DistanceMetric::Euclidean,
            parameters: IndexParameters::default(),
        };

        let mut index = HnswIndex::new(config);

        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let metadata = vec![
            serde_json::json!({"label": "x-axis"}),
            serde_json::json!({"label": "y-axis"}),
            serde_json::json!({"label": "z-axis"}),
        ];

        let ids = index.add_vectors(vectors, metadata).await.unwrap();
        assert_eq!(ids.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 2, None).await.unwrap();
        assert!(results.len() <= 2);
    }
}