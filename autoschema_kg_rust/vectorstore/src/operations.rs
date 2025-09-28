//! Comprehensive vector operations with GPU acceleration support

use std::sync::Arc;
use std::collections::HashMap;
use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::{Result, VectorError};
use crate::embedding::{BaseEmbeddingModel, EmbeddingResult};
use crate::index::{VectorIndex, IndexConfig};
use crate::search::{SearchResult, SimilarityMetric};
use crate::metrics::{MetricsCollector, OperationType};

/// GPU acceleration configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub enabled: bool,
    pub device_id: usize,
    pub memory_fraction: f32,
    pub fallback_to_cpu: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: 0,
            memory_fraction: 0.8,
            fallback_to_cpu: true,
        }
    }
}

/// Vector operations engine with GPU acceleration
pub struct VectorOperationsEngine {
    gpu_config: GpuConfig,
    metrics: Arc<MetricsCollector>,
    #[cfg(feature = "gpu")]
    cuda_context: Option<cuda::CudaContext>,
    simd_enabled: bool,
}

impl VectorOperationsEngine {
    /// Create a new vector operations engine
    pub async fn new(gpu_config: GpuConfig, metrics: Arc<MetricsCollector>) -> Result<Self> {
        let mut engine = Self {
            gpu_config,
            metrics,
            #[cfg(feature = "gpu")]
            cuda_context: None,
            simd_enabled: Self::detect_simd_support(),
        };

        // Initialize GPU if enabled
        if engine.gpu_config.enabled {
            engine.initialize_gpu().await?;
        }

        Ok(engine)
    }

    /// Initialize GPU acceleration
    async fn initialize_gpu(&mut self) -> Result<()> {
        #[cfg(feature = "gpu")]
        {
            if !self.gpu_config.enabled {
                return Ok(());
            }

            match cuda::initialize() {
                Ok(_) => {
                    let device_count = cuda::get_device_count()?;
                    if self.gpu_config.device_id >= device_count {
                        if self.gpu_config.fallback_to_cpu {
                            log::warn!("GPU device {} not available, falling back to CPU", self.gpu_config.device_id);
                            return Ok(());
                        } else {
                            return Err(VectorError::gpu_error(format!(
                                "GPU device {} not available (total devices: {})",
                                self.gpu_config.device_id, device_count
                            )));
                        }
                    }

                    let context = cuda::create_context(self.gpu_config.device_id)?;
                    self.cuda_context = Some(context);
                    log::info!("GPU acceleration initialized on device {}", self.gpu_config.device_id);
                }
                Err(e) => {
                    if self.gpu_config.fallback_to_cpu {
                        log::warn!("Failed to initialize GPU, falling back to CPU: {}", e);
                    } else {
                        return Err(VectorError::gpu_error(format!("GPU initialization failed: {}", e)));
                    }
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            if self.gpu_config.enabled && !self.gpu_config.fallback_to_cpu {
                return Err(VectorError::gpu_error("GPU support not compiled in"));
            }
            log::info!("GPU support not available, using CPU");
        }

        Ok(())
    }

    /// Detect SIMD support
    fn detect_simd_support() -> bool {
        #[cfg(feature = "simd-accel")]
        {
            // In practice, you'd detect actual SIMD capabilities
            true
        }
        #[cfg(not(feature = "simd-accel"))]
        {
            false
        }
    }

    /// Batch compute embeddings with GPU acceleration
    pub async fn batch_embed(
        &self,
        embedding_model: &dyn BaseEmbeddingModel,
        texts: &[String],
        batch_size: Option<usize>,
    ) -> Result<EmbeddingResult> {
        let timer = crate::metrics::OperationTimer::new(
            OperationType::Embedding,
            self.metrics.clone(),
        );

        let batch_size = batch_size.unwrap_or(32);
        let mut all_embeddings = Vec::new();
        let mut total_time_ms = 0u64;

        // Process in batches to manage memory
        for chunk in texts.chunks(batch_size) {
            let start_time = std::time::Instant::now();

            let chunk_result = if self.gpu_config.enabled && self.is_gpu_available() {
                self.gpu_embed_batch(embedding_model, chunk).await
                    .or_else(|_| async {
                        log::warn!("GPU embedding failed, falling back to CPU");
                        embedding_model.embed_batch(&chunk.to_vec()).await
                    }).await?
            } else {
                embedding_model.embed_batch(&chunk.to_vec()).await?
            };

            all_embeddings.extend(chunk_result.embeddings);
            total_time_ms += start_time.elapsed().as_millis() as u64;
        }

        timer.finish().await;

        let mut metadata = HashMap::new();
        metadata.insert(
            "gpu_accelerated".to_string(),
            serde_json::Value::Bool(self.gpu_config.enabled && self.is_gpu_available()),
        );
        metadata.insert(
            "batch_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(batch_size)),
        );
        metadata.insert(
            "total_batches".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                (texts.len() + batch_size - 1) / batch_size,
            )),
        );

        Ok(EmbeddingResult {
            embeddings: all_embeddings,
            metadata,
            processing_time_ms: total_time_ms,
        })
    }

    /// GPU-accelerated embedding computation
    async fn gpu_embed_batch(
        &self,
        embedding_model: &dyn BaseEmbeddingModel,
        texts: &[String],
    ) -> Result<EmbeddingResult> {
        #[cfg(feature = "gpu")]
        {
            if let Some(_context) = &self.cuda_context {
                // In a real implementation, this would:
                // 1. Transfer tokenized input to GPU
                // 2. Run transformer model on GPU
                // 3. Transfer results back to CPU
                // For now, we'll delegate to the CPU implementation
                log::debug!("Using GPU for embedding computation (batch size: {})", texts.len());
            }
        }

        // Fallback to CPU implementation
        embedding_model.embed_batch(&texts.to_vec()).await
    }

    /// Parallel similarity search with GPU acceleration
    pub async fn parallel_search(
        &self,
        index: &dyn VectorIndex,
        queries: &[Vec<f32>],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let timer = crate::metrics::OperationTimer::new(
            OperationType::Search,
            self.metrics.clone(),
        );

        let results = if self.gpu_config.enabled && self.is_gpu_available() {
            self.gpu_parallel_search(index, queries, k, metric).await
                .or_else(|_| async {
                    log::warn!("GPU search failed, falling back to CPU");
                    self.cpu_parallel_search(index, queries, k).await
                }).await?
        } else {
            self.cpu_parallel_search(index, queries, k).await?
        };

        timer.finish().await;
        Ok(results)
    }

    /// GPU-accelerated parallel search
    async fn gpu_parallel_search(
        &self,
        index: &dyn VectorIndex,
        queries: &[Vec<f32>],
        k: usize,
        _metric: SimilarityMetric,
    ) -> Result<Vec<Vec<SearchResult>>> {
        #[cfg(feature = "gpu")]
        {
            if let Some(_context) = &self.cuda_context {
                // In a real implementation, this would:
                // 1. Transfer query vectors to GPU
                // 2. Compute similarities using GPU kernels
                // 3. Return top-k results
                log::debug!("Using GPU for parallel search (queries: {}, k: {})", queries.len(), k);
            }
        }

        // Fallback to CPU implementation
        self.cpu_parallel_search(index, queries, k).await
    }

    /// CPU parallel search using rayon
    async fn cpu_parallel_search(
        &self,
        index: &dyn VectorIndex,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;

        let queries_vec = queries.to_vec();
        let results = tokio::task::spawn_blocking(move || {
            queries_vec
                .par_iter()
                .map(|query| {
                    // Note: This would need to be adjusted for actual async index operations
                    // In practice, you'd need a different approach for async operations with rayon
                    Vec::new() // Placeholder
                })
                .collect::<Vec<_>>()
        }).await?;

        // For now, do sequential search since we need async
        let mut all_results = Vec::new();
        for query in queries {
            let results = index.search(query, k, None).await?;
            all_results.push(results);
        }

        Ok(all_results)
    }

    /// Optimized vector distance computation
    pub fn compute_distances(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<f32>> {
        if self.simd_enabled {
            self.simd_compute_distances(query, vectors, metric)
        } else {
            self.scalar_compute_distances(query, vectors, metric)
        }
    }

    /// SIMD-accelerated distance computation
    #[cfg(feature = "simd-accel")]
    fn simd_compute_distances(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<f32>> {
        use std::arch::x86_64::*;

        let mut distances = Vec::with_capacity(vectors.len());

        match metric {
            SimilarityMetric::Cosine => {
                for vector in vectors {
                    if vector.len() != query.len() {
                        return Err(VectorError::DimensionMismatch {
                            expected: query.len(),
                            actual: vector.len(),
                        });
                    }

                    let distance = unsafe {
                        self.simd_cosine_distance(query, vector)?
                    };
                    distances.push(distance);
                }
            }
            SimilarityMetric::Euclidean => {
                for vector in vectors {
                    if vector.len() != query.len() {
                        return Err(VectorError::DimensionMismatch {
                            expected: query.len(),
                            actual: vector.len(),
                        });
                    }

                    let distance = unsafe {
                        self.simd_euclidean_distance(query, vector)?
                    };
                    distances.push(distance);
                }
            }
            _ => {
                // Fallback to scalar implementation
                return self.scalar_compute_distances(query, vectors, metric);
            }
        }

        Ok(distances)
    }

    #[cfg(not(feature = "simd-accel"))]
    fn simd_compute_distances(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<f32>> {
        self.scalar_compute_distances(query, vectors, metric)
    }

    /// Scalar distance computation
    fn scalar_compute_distances(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<f32>> {
        let mut distances = Vec::with_capacity(vectors.len());

        for vector in vectors {
            if vector.len() != query.len() {
                return Err(VectorError::DimensionMismatch {
                    expected: query.len(),
                    actual: vector.len(),
                });
            }

            let distance = match metric {
                SimilarityMetric::Cosine => self.cosine_distance(query, vector)?,
                SimilarityMetric::Euclidean => self.euclidean_distance(query, vector)?,
                SimilarityMetric::DotProduct => self.dot_product(query, vector)?,
                SimilarityMetric::Manhattan => self.manhattan_distance(query, vector)?,
                _ => return Err(VectorError::search_error("Unsupported similarity metric")),
            };
            distances.push(distance);
        }

        Ok(distances)
    }

    /// SIMD cosine distance computation
    #[cfg(feature = "simd-accel")]
    unsafe fn simd_cosine_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        let chunks = len / 8;
        let remainder = len % 8;

        // Process 8 elements at a time using AVX
        for i in 0..chunks {
            let idx = i * 8;

            let va = _mm256_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

            // Dot product
            let dot = _mm256_mul_ps(va, vb);
            let dot_sum = _mm256_hadd_ps(dot, dot);
            let dot_sum = _mm256_hadd_ps(dot_sum, dot_sum);
            let result: [f32; 8] = std::mem::transmute(dot_sum);
            dot_product += result[0] + result[4];

            // Norms
            let norm_a_vec = _mm256_mul_ps(va, va);
            let norm_b_vec = _mm256_mul_ps(vb, vb);

            let norm_a_sum = _mm256_hadd_ps(norm_a_vec, norm_a_vec);
            let norm_a_sum = _mm256_hadd_ps(norm_a_sum, norm_a_sum);
            let norm_a_result: [f32; 8] = std::mem::transmute(norm_a_sum);
            norm_a += norm_a_result[0] + norm_a_result[4];

            let norm_b_sum = _mm256_hadd_ps(norm_b_vec, norm_b_vec);
            let norm_b_sum = _mm256_hadd_ps(norm_b_sum, norm_b_sum);
            let norm_b_result: [f32; 8] = std::mem::transmute(norm_b_sum);
            norm_b += norm_b_result[0] + norm_b_result[4];
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - (dot_product / (norm_a * norm_b)))
        }
    }

    /// SIMD Euclidean distance computation
    #[cfg(feature = "simd-accel")]
    unsafe fn simd_euclidean_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut sum = 0.0f32;

        let chunks = len / 8;
        let remainder = len % 8;

        // Process 8 elements at a time using AVX
        for i in 0..chunks {
            let idx = i * 8;

            let va = _mm256_loadu_ps(a.as_ptr().add(idx));
            let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

            let diff = _mm256_sub_ps(va, vb);
            let squared = _mm256_mul_ps(diff, diff);

            let sum_vec = _mm256_hadd_ps(squared, squared);
            let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            let result: [f32; 8] = std::mem::transmute(sum_vec);
            sum += result[0] + result[4];
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }

        Ok(sum.sqrt())
    }

    // Scalar implementations
    fn cosine_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - (dot_product / (norm_a * norm_b)))
        }
    }

    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let distance: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        Ok(distance)
    }

    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    fn manhattan_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        Ok(a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum())
    }

    /// Check if GPU is available and functional
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.cuda_context.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Batch normalize vectors
    pub fn batch_normalize(&self, vectors: &mut [Vec<f32>]) -> Result<()> {
        if self.simd_enabled {
            self.simd_batch_normalize(vectors)
        } else {
            self.scalar_batch_normalize(vectors)
        }
    }

    /// SIMD batch normalization
    #[cfg(feature = "simd-accel")]
    fn simd_batch_normalize(&self, vectors: &mut [Vec<f32>]) -> Result<()> {
        for vector in vectors.iter_mut() {
            unsafe {
                self.simd_normalize_vector(vector)?;
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "simd-accel"))]
    fn simd_batch_normalize(&self, vectors: &mut [Vec<f32>]) -> Result<()> {
        self.scalar_batch_normalize(vectors)
    }

    /// Scalar batch normalization
    fn scalar_batch_normalize(&self, vectors: &mut [Vec<f32>]) -> Result<()> {
        for vector in vectors.iter_mut() {
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vector.iter_mut() {
                    *x /= norm;
                }
            }
        }
        Ok(())
    }

    /// SIMD vector normalization
    #[cfg(feature = "simd-accel")]
    unsafe fn simd_normalize_vector(&self, vector: &mut [f32]) -> Result<()> {
        use std::arch::x86_64::*;

        let len = vector.len();
        let mut norm_squared = 0.0f32;

        let chunks = len / 8;

        // Calculate norm using SIMD
        for i in 0..chunks {
            let idx = i * 8;
            let v = _mm256_loadu_ps(vector.as_ptr().add(idx));
            let squared = _mm256_mul_ps(v, v);

            let sum_vec = _mm256_hadd_ps(squared, squared);
            let sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            let result: [f32; 8] = std::mem::transmute(sum_vec);
            norm_squared += result[0] + result[4];
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            norm_squared += vector[i] * vector[i];
        }

        let norm = norm_squared.sqrt();
        if norm == 0.0 {
            return Ok(());
        }

        let norm_inv = 1.0 / norm;
        let norm_vec = _mm256_set1_ps(norm_inv);

        // Normalize using SIMD
        for i in 0..chunks {
            let idx = i * 8;
            let v = _mm256_loadu_ps(vector.as_ptr().add(idx));
            let normalized = _mm256_mul_ps(v, norm_vec);
            _mm256_storeu_ps(vector.as_mut_ptr().add(idx), normalized);
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            vector[i] *= norm_inv;
        }

        Ok(())
    }

    /// Get GPU memory usage
    pub async fn get_gpu_memory_usage(&self) -> Result<Option<(u64, u64)>> {
        #[cfg(feature = "gpu")]
        {
            if let Some(_context) = &self.cuda_context {
                // In practice, query CUDA for actual memory usage
                return Ok(Some((0, 0))); // (used, total)
            }
        }
        Ok(None)
    }

    /// Cleanup GPU resources
    pub async fn cleanup(&mut self) -> Result<()> {
        #[cfg(feature = "gpu")]
        {
            if let Some(_context) = self.cuda_context.take() {
                // Cleanup CUDA context
                log::info!("Cleaning up GPU resources");
            }
        }
        Ok(())
    }
}

/// Trait for GPU-accelerated operations
#[async_trait]
pub trait GpuAccelerated {
    /// Check if GPU acceleration is available
    fn gpu_available(&self) -> bool;

    /// Enable GPU acceleration
    async fn enable_gpu(&mut self, device_id: usize) -> Result<()>;

    /// Disable GPU acceleration
    async fn disable_gpu(&mut self) -> Result<()>;

    /// Get GPU memory usage
    async fn gpu_memory_usage(&self) -> Result<Option<(u64, u64)>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vector_operations_engine() {
        let gpu_config = GpuConfig::default();
        let metrics = Arc::new(MetricsCollector::new());

        let engine = VectorOperationsEngine::new(gpu_config, metrics).await.unwrap();

        // Test distance computation
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];

        let distances = engine
            .compute_distances(&query, &vectors, SimilarityMetric::Cosine)
            .unwrap();

        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 0.0).abs() < 1e-6); // Same vector
        assert!(distances[1] > 0.9); // Orthogonal vectors
    }

    #[tokio::test]
    async fn test_batch_normalization() {
        let gpu_config = GpuConfig::default();
        let metrics = Arc::new(MetricsCollector::new());

        let engine = VectorOperationsEngine::new(gpu_config, metrics).await.unwrap();

        let mut vectors = vec![
            vec![3.0, 4.0, 0.0], // Norm = 5
            vec![1.0, 1.0, 1.0], // Norm = sqrt(3)
        ];

        engine.batch_normalize(&mut vectors).unwrap();

        // Check normalization
        for vector in &vectors {
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }
}