//! Comprehensive example demonstrating all vector store capabilities

use std::sync::Arc;
use vectorstore::{
    VectorStore, VectorStoreWithFeatures, VectorStoreConfig, MockEmbeddingModel,
    IndexType, SimilarityMetric, GpuConfig, HookManager, HookFactory,
    MetricsCollector, PerformanceMonitor, VectorOperationsEngine
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("🚀 Starting comprehensive vector store example...");

    // 1. Basic Vector Store Setup
    println!("\n📦 Setting up basic vector store...");
    let basic_example = basic_vector_store_example().await?;
    println!("✅ Basic vector store completed: {} vectors indexed", basic_example);

    // 2. Enhanced Vector Store with GPU and Hooks
    println!("\n🔧 Setting up enhanced vector store with features...");
    let enhanced_example = enhanced_vector_store_example().await?;
    println!("✅ Enhanced vector store completed: {} vectors processed", enhanced_example);

    // 3. Performance Monitoring
    println!("\n📊 Demonstrating performance monitoring...");
    performance_monitoring_example().await?;
    println!("✅ Performance monitoring completed");

    // 4. GPU Acceleration (fallback to CPU if not available)
    println!("\n⚡ Testing GPU acceleration...");
    gpu_acceleration_example().await?;
    println!("✅ GPU acceleration example completed");

    // 5. Hook Integration
    println!("\n🔗 Demonstrating hook integration...");
    hook_integration_example().await?;
    println!("✅ Hook integration completed");

    // 6. Advanced Search Features
    println!("\n🔍 Testing advanced search features...");
    let search_results = advanced_search_example().await?;
    println!("✅ Advanced search completed: {} results found", search_results);

    println!("\n🎉 All examples completed successfully!");
    Ok(())
}

/// Basic vector store operations
async fn basic_vector_store_example() -> Result<usize, Box<dyn std::error::Error>> {
    // Create embedding model
    let embedding_model = Arc::new(MockEmbeddingModel::new(384));

    // Configure vector store
    let config = VectorStoreConfig {
        dimension: 384,
        index_type: IndexType::Flat,
        similarity_metric: SimilarityMetric::Cosine,
        enable_gpu: false,
        enable_compression: false,
        batch_size: 100,
        num_threads: Some(4),
    };

    // Create vector store
    let store = VectorStore::new(embedding_model, config).await?;

    // Prepare test data
    let texts = vec![
        "Machine learning is a subset of artificial intelligence".to_string(),
        "Vector databases enable semantic search capabilities".to_string(),
        "Neural networks can learn complex patterns in data".to_string(),
        "Natural language processing helps computers understand text".to_string(),
        "Knowledge graphs represent structured information".to_string(),
    ];

    let metadata: Vec<serde_json::Value> = texts.iter().enumerate().map(|(i, text)| {
        serde_json::json!({
            "id": i,
            "length": text.len(),
            "category": "ai_ml",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }).collect();

    // Add vectors
    let ids = store.add_vectors(texts.clone(), metadata).await?;
    println!("Added {} vectors with IDs: {:?}", ids.len(), &ids[0..2]);

    // Search for similar vectors
    let results = store.search("artificial intelligence and neural networks", 3, None).await?;
    println!("Search results:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. ID: {}, Score: {:.4}", i + 1, result.id, result.score);
    }

    // Get statistics
    let stats = store.stats().await?;
    println!("Store statistics: {} vectors, {} MB memory",
             stats.total_vectors, stats.memory_usage / (1024 * 1024));

    Ok(texts.len())
}

/// Enhanced vector store with all features
async fn enhanced_vector_store_example() -> Result<usize, Box<dyn std::error::Error>> {
    // Create embedding model
    let embedding_model = Arc::new(MockEmbeddingModel::new(768));

    // Configure enhanced vector store
    let config = VectorStoreConfig {
        dimension: 768,
        index_type: IndexType::Hnsw,
        similarity_metric: SimilarityMetric::Cosine,
        enable_gpu: false, // Will attempt GPU, fallback to CPU
        enable_compression: true,
        batch_size: 200,
        num_threads: Some(8),
    };

    // GPU configuration
    let gpu_config = GpuConfig {
        enabled: true,
        device_id: 0,
        memory_fraction: 0.7,
        fallback_to_cpu: true,
    };

    // Create enhanced vector store
    let mut store = VectorStore::new_with_features(
        embedding_model,
        config,
        Some(gpu_config),
        true, // Enable hooks
    ).await?;

    // Prepare larger dataset
    let texts: Vec<String> = (0..1000).map(|i| {
        format!("Document {} about machine learning, artificial intelligence, and data science topics. \
                This document contains information about neural networks, deep learning, and vector embeddings.", i)
    }).collect();

    let metadata: Vec<serde_json::Value> = texts.iter().enumerate().map(|(i, _)| {
        serde_json::json!({
            "id": i,
            "category": format!("category_{}", i % 10),
            "importance": (i as f64) / 1000.0,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })
    }).collect();

    // Add vectors with enhanced features
    let ids = store.add_vectors_enhanced(texts.clone(), metadata).await?;
    println!("Enhanced store added {} vectors", ids.len());

    // Enhanced search with filters
    let filter = Some(serde_json::json!({
        "must": [
            {
                "field": "category",
                "operator": "In",
                "value": ["category_1", "category_2", "category_3"]
            }
        ],
        "range": {
            "importance": {
                "gte": 0.1,
                "lte": 0.9
            }
        }
    }));

    let results = store.search_enhanced(
        "neural networks and machine learning algorithms",
        10,
        filter,
    ).await?;

    println!("Enhanced search found {} filtered results", results.len());

    // Get performance metrics
    let metrics = store.get_performance_metrics().await?;
    println!("Performance metrics:");
    println!("  Embedding operations: {}", metrics.embedding_operations);
    println!("  Search operations: {}", metrics.search_operations);
    println!("  Average embedding time: {:.2}ms", metrics.avg_embedding_time_ms);
    println!("  Average search time: {:.2}ms", metrics.avg_search_time_ms);

    // Get hook metrics
    let hook_metrics = store.get_hook_metrics().await?;
    println!("Hook executions: {} types", hook_metrics.len());

    // Get GPU memory usage (if available)
    if let Some((used, total)) = store.get_gpu_memory_usage().await? {
        println!("GPU memory: {:.1}MB used / {:.1}MB total",
                 used as f64 / (1024.0 * 1024.0),
                 total as f64 / (1024.0 * 1024.0));
    }

    // Cleanup
    store.shutdown().await?;

    Ok(texts.len())
}

/// Performance monitoring demonstration
async fn performance_monitoring_example() -> Result<(), Box<dyn std::error::Error>> {
    let metrics = Arc::new(MetricsCollector::new());

    // Start performance monitor
    let monitor = PerformanceMonitor::new(
        metrics.clone(),
        std::time::Duration::from_secs(1)
    );
    monitor.start().await?;

    // Simulate some operations
    for i in 0..10 {
        let operation_time = std::time::Duration::from_millis(50 + (i * 10));
        tokio::time::sleep(operation_time).await;

        metrics.record_embedding_operation(operation_time).await;

        if i % 3 == 0 {
            let search_time = std::time::Duration::from_millis(20);
            metrics.record_search_operation(search_time).await;
        }
    }

    // Set custom metrics
    metrics.set_custom_metric("cache_hit_rate".to_string(), 0.85).await;
    metrics.increment_custom_counter("total_requests".to_string(), 10.0).await;

    // Get final metrics
    let final_metrics = metrics.get_metrics().await?;
    println!("Final performance metrics:");
    println!("  Total embedding operations: {}", final_metrics.embedding_operations);
    println!("  Total search operations: {}", final_metrics.search_operations);
    println!("  P95 embedding time: {:.2}ms", final_metrics.p95_embedding_time_ms);
    println!("  Custom metrics: {:?}", final_metrics.custom_metrics);

    monitor.stop();
    Ok(())
}

/// GPU acceleration testing
async fn gpu_acceleration_example() -> Result<(), Box<dyn std::error::Error>> {
    let metrics = Arc::new(MetricsCollector::new());

    // Test GPU configuration
    let gpu_config = GpuConfig {
        enabled: true,
        device_id: 0,
        memory_fraction: 0.5,
        fallback_to_cpu: true,
    };

    let engine = VectorOperationsEngine::new(gpu_config, metrics.clone()).await?;

    // Test vector operations
    let query = vec![0.1, 0.2, 0.3, 0.4];
    let vectors = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.9, 0.1, 0.2, 0.3],
    ];

    let distances = engine.compute_distances(&query, &vectors, SimilarityMetric::Cosine)?;
    println!("Computed distances: {:?}", distances);

    // Test batch normalization
    let mut test_vectors = vec![
        vec![3.0, 4.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 0.0, 0.0],
    ];

    engine.batch_normalize(&mut test_vectors)?;
    println!("Normalized vectors:");
    for (i, vector) in test_vectors.iter().enumerate() {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  Vector {}: norm = {:.6}", i, norm);
    }

    // Check GPU availability
    if engine.is_gpu_available() {
        println!("GPU acceleration is available and active");
        if let Some((used, total)) = engine.get_gpu_memory_usage().await? {
            println!("GPU memory usage: {}MB / {}MB", used / (1024*1024), total / (1024*1024));
        }
    } else {
        println!("GPU acceleration not available, using CPU fallback");
    }

    Ok(())
}

/// Hook integration demonstration
async fn hook_integration_example() -> Result<(), Box<dyn std::error::Error>> {
    let hook_manager = HookManager::new();

    // Create some test hooks
    let pre_hook = HookFactory::create_pre_operation_hook("embedding_validator".to_string());
    let post_hook = HookFactory::create_post_operation_hook("embedding_cleaner".to_string());
    let session_hook = HookFactory::create_session_hook("session_123".to_string());

    // Register hooks (simplified - in real implementation would handle lifetimes properly)
    println!("Hooks created: {}, {}, {}",
             pre_hook.name(), post_hook.name(), session_hook.name());

    // Simulate hook execution
    let context = vectorstore::HookContext {
        operation_type: "embedding".to_string(),
        operation_id: "op_456".to_string(),
        metadata: std::collections::HashMap::new(),
        timestamp: chrono::Utc::now(),
    };

    // Execute built-in coordination hooks
    let hook_results = hook_manager.execute_hooks(&context).await?;
    println!("Executed {} coordination hooks", hook_results.len());

    for (i, result) in hook_results.iter().enumerate() {
        println!("  Hook {}: success={}, time={}ms",
                 i + 1, result.success, result.execution_time_ms);
        if let Some(message) = &result.message {
            println!("    Message: {}", message);
        }
    }

    // Get hook metrics
    let metrics = hook_manager.get_hook_metrics().await?;
    println!("Hook metrics collected for {} operation types", metrics.len());

    Ok(())
}

/// Advanced search features
async fn advanced_search_example() -> Result<usize, Box<dyn std::error::Error>> {
    // Setup vector store
    let embedding_model = Arc::new(MockEmbeddingModel::new(512));
    let config = VectorStoreConfig {
        dimension: 512,
        index_type: IndexType::Hnsw,
        similarity_metric: SimilarityMetric::Cosine,
        enable_gpu: false,
        enable_compression: false,
        batch_size: 100,
        num_threads: Some(4),
    };

    let store = VectorStore::new(embedding_model, config).await?;

    // Add diverse test data
    let texts = vec![
        "Advanced machine learning algorithms for computer vision".to_string(),
        "Natural language processing with transformer models".to_string(),
        "Deep learning optimization techniques".to_string(),
        "Reinforcement learning in robotics applications".to_string(),
        "Graph neural networks for social network analysis".to_string(),
        "Federated learning for privacy-preserving AI".to_string(),
        "Computer vision for autonomous vehicle navigation".to_string(),
        "Time series forecasting with LSTM networks".to_string(),
    ];

    let metadata: Vec<serde_json::Value> = texts.iter().enumerate().map(|(i, text)| {
        serde_json::json!({
            "id": i,
            "domain": match i % 4 {
                0 => "computer_vision",
                1 => "nlp",
                2 => "optimization",
                _ => "applications"
            },
            "complexity": (i as f64 + 1.0) / texts.len() as f64,
            "year": 2020 + (i % 4),
            "keywords": text.split_whitespace().take(3).collect::<Vec<&str>>()
        })
    }).collect();

    let ids = store.add_vectors(texts, metadata).await?;
    println!("Added {} documents for advanced search", ids.len());

    // Advanced search with complex filters
    let complex_filter = Some(serde_json::json!({
        "must": [
            {
                "field": "domain",
                "operator": "In",
                "value": ["computer_vision", "nlp"]
            }
        ],
        "should": [
            {
                "field": "keywords",
                "operator": "Contains",
                "value": "learning"
            }
        ],
        "range": {
            "complexity": {
                "gte": 0.3,
                "lte": 0.8
            },
            "year": {
                "gte": 2021
            }
        }
    }));

    let advanced_results = store.search(
        "machine learning and computer vision research",
        5,
        complex_filter,
    ).await?;

    println!("Advanced search results:");
    for (i, result) in advanced_results.iter().enumerate() {
        println!("  {}. Score: {:.4}, Metadata: {}",
                 i + 1, result.score, result.metadata);
    }

    // Similarity threshold search
    let threshold_results = store.search(
        "deep learning neural networks",
        10,
        None,
    ).await?;

    let high_similarity_results: Vec<_> = threshold_results
        .into_iter()
        .filter(|r| r.score > 0.7)
        .collect();

    println!("High similarity results (>0.7): {}", high_similarity_results.len());

    Ok(advanced_results.len())
}