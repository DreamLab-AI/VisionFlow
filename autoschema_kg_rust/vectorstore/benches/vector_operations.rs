//! Performance benchmarks for vector operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use tokio::runtime::Runtime;

use vectorstore::{
    VectorStore, VectorStoreConfig, VectorStoreWithFeatures, MockEmbeddingModel,
    IndexType, SimilarityMetric, GpuConfig, OperationType, MetricsCollector,
    VectorOperationsEngine
};

fn create_test_data(size: usize, dimension: usize) -> (Vec<String>, Vec<serde_json::Value>) {
    let texts: Vec<String> = (0..size)
        .map(|i| format!("test document number {}", i))
        .collect();

    let metadata: Vec<serde_json::Value> = (0..size)
        .map(|i| serde_json::json!({
            "id": i,
            "category": format!("category_{}", i % 10),
            "score": (i as f64) / (size as f64)
        }))
        .collect();

    (texts, metadata)
}

fn create_test_vectors(size: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..size)
        .map(|_| {
            (0..dimension)
                .map(|_| rand::random::<f32>())
                .collect()
        })
        .collect()
}

async fn setup_vector_store(dimension: usize) -> VectorStore {
    let embedding_model = Arc::new(MockEmbeddingModel::new(dimension));
    let config = VectorStoreConfig {
        dimension,
        index_type: IndexType::Flat,
        similarity_metric: SimilarityMetric::Cosine,
        enable_gpu: false,
        enable_compression: false,
        batch_size: 1000,
        num_threads: Some(4),
    };

    VectorStore::new(embedding_model, config).await.unwrap()
}

async fn setup_enhanced_vector_store(dimension: usize, enable_gpu: bool) -> VectorStoreWithFeatures {
    let embedding_model = Arc::new(MockEmbeddingModel::new(dimension));
    let config = VectorStoreConfig {
        dimension,
        index_type: IndexType::Hnsw,
        similarity_metric: SimilarityMetric::Cosine,
        enable_gpu,
        enable_compression: false,
        batch_size: 1000,
        num_threads: Some(4),
    };

    let gpu_config = if enable_gpu {
        Some(GpuConfig {
            enabled: true,
            device_id: 0,
            memory_fraction: 0.8,
            fallback_to_cpu: true,
        })
    } else {
        None
    };

    VectorStore::new_with_features(embedding_model, config, gpu_config, true)
        .await
        .unwrap()
}

fn bench_embedding_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("embedding_operations");

    for batch_size in [10, 100, 1000].iter() {
        for dimension in [128, 384, 768].iter() {
            group.bench_with_input(
                BenchmarkId::new("batch_embed", format!("{}x{}", batch_size, dimension)),
                &(*batch_size, *dimension),
                |b, &(batch_size, dimension)| {
                    let embedding_model = Arc::new(MockEmbeddingModel::new(dimension));
                    let (texts, _) = create_test_data(batch_size, dimension);

                    b.to_async(&rt).iter(|| async {
                        let result = embedding_model.embed_batch(&texts).await.unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_index_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("index_operations");

    for vector_count in [100, 1000, 10000].iter() {
        for dimension in [128, 384, 768].iter() {
            group.bench_with_input(
                BenchmarkId::new("add_vectors", format!("{}x{}", vector_count, dimension)),
                &(*vector_count, *dimension),
                |b, &(vector_count, dimension)| {
                    b.to_async(&rt).iter(|| async {
                        let store = setup_vector_store(dimension).await;
                        let (texts, metadata) = create_test_data(vector_count, dimension);

                        let result = store.add_vectors(texts, metadata).await.unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_search_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("search_operations");

    for k in [1, 10, 100].iter() {
        for dimension in [128, 384, 768].iter() {
            group.bench_with_input(
                BenchmarkId::new("search", format!("k{}x{}", k, dimension)),
                &(*k, *dimension),
                |b, &(k, dimension)| {
                    b.to_async(&rt).iter_batched(
                        || {
                            rt.block_on(async {
                                let store = setup_vector_store(dimension).await;
                                let (texts, metadata) = create_test_data(1000, dimension);
                                store.add_vectors(texts, metadata).await.unwrap();
                                store
                            })
                        },
                        |store| async move {
                            let result = store.search("test query", k, None).await.unwrap();
                            black_box(result);
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_gpu_acceleration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("gpu_acceleration");

    for batch_size in [100, 1000].iter() {
        let dimension = 768;

        // CPU benchmark
        group.bench_with_input(
            BenchmarkId::new("cpu_batch_embed", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let store = setup_enhanced_vector_store(dimension, false).await;
                    let (texts, metadata) = create_test_data(batch_size, dimension);

                    let result = store.add_vectors_enhanced(texts, metadata).await.unwrap();
                    black_box(result);
                });
            },
        );

        // GPU benchmark (will fallback to CPU in most environments)
        group.bench_with_input(
            BenchmarkId::new("gpu_batch_embed", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let store = setup_enhanced_vector_store(dimension, true).await;
                    let (texts, metadata) = create_test_data(batch_size, dimension);

                    let result = store.add_vectors_enhanced(texts, metadata).await.unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_similarity_computation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("similarity_computation");

    for vector_count in [100, 1000, 10000].iter() {
        for dimension in [128, 384, 768].iter() {
            group.bench_with_input(
                BenchmarkId::new("cosine_similarity", format!("{}x{}", vector_count, dimension)),
                &(*vector_count, *dimension),
                |b, &(vector_count, dimension)| {
                    let query = (0..dimension).map(|_| rand::random::<f32>()).collect::<Vec<_>>();
                    let vectors = create_test_vectors(vector_count, dimension);

                    b.to_async(&rt).iter(|| async {
                        let metrics = Arc::new(MetricsCollector::new());
                        let gpu_config = GpuConfig::default();
                        let engine = VectorOperationsEngine::new(gpu_config, metrics).await.unwrap();

                        let result = engine.compute_distances(
                            &query,
                            &vectors,
                            SimilarityMetric::Cosine
                        ).unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("batch_operations");

    for batch_size in [32, 128, 512].iter() {
        let dimension = 384;

        group.bench_with_input(
            BenchmarkId::new("batch_normalize", batch_size),
            batch_size,
            |b, &batch_size| {
                let mut vectors = create_test_vectors(batch_size, dimension);

                b.to_async(&rt).iter(|| async {
                    let metrics = Arc::new(MetricsCollector::new());
                    let gpu_config = GpuConfig::default();
                    let engine = VectorOperationsEngine::new(gpu_config, metrics).await.unwrap();

                    engine.batch_normalize(&mut vectors).unwrap();
                    black_box(&vectors);
                });
            },
        );
    }

    group.finish();
}

fn bench_storage_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("storage_operations");

    for vector_count in [100, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("save_load_cycle", vector_count),
            vector_count,
            |b, &vector_count| {
                b.to_async(&rt).iter(|| async {
                    let store = setup_vector_store(384).await;
                    let (texts, metadata) = create_test_data(vector_count, 384);

                    // Add vectors
                    let ids = store.add_vectors(texts, metadata).await.unwrap();

                    // Save
                    let temp_path = format!("/tmp/bench_index_{}.bin", rand::random::<u32>());
                    store.save(&temp_path).await.unwrap();

                    // Load
                    store.load(&temp_path).await.unwrap();

                    // Cleanup
                    let _ = std::fs::remove_file(&temp_path);

                    black_box(ids);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_embedding_operations,
    bench_index_operations,
    bench_search_operations,
    bench_gpu_acceleration,
    bench_similarity_computation,
    bench_batch_operations,
    bench_storage_operations
);

criterion_main!(benches);