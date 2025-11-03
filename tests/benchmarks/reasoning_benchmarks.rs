// Performance Benchmarks for Ontology Reasoning Pipeline
// Tests inference speed, cache performance, and constraint generation throughput

use std::time::Instant;
use webxr::reasoning::{
    custom_reasoner::{CustomReasoner, OntologyReasoner},
    inference_cache::InferenceCache,
};
use webxr::constraints::axiom_mapper::{AxiomMapper, OWLAxiom, AxiomType as MapperAxiomType};
use tempfile::TempDir;

mod fixtures;
use fixtures::ontology::test_ontologies::*;

#[cfg(test)]
mod performance_benchmarks {
    use super::*;

    #[tokio::test]
    async fn bench_inference_simple_ontology() {
        let ontology = create_simple_hierarchy();
        let reasoner = CustomReasoner::new();

        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = reasoner.infer_axioms(&ontology).unwrap();
        }

        let duration = start.elapsed();
        let avg_time = duration / iterations;

        println!("Simple ontology inference:");
        println!("  Total: {:?}", duration);
        println!("  Average: {:?}", avg_time);
        println!("  Throughput: {:.2} inferences/sec", iterations as f64 / duration.as_secs_f64());

        assert!(avg_time.as_millis() < 100, "Should average < 100ms per inference");
    }

    #[tokio::test]
    async fn bench_inference_deep_hierarchy() {
        let ontology = create_deep_hierarchy();
        let reasoner = CustomReasoner::new();

        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = reasoner.infer_axioms(&ontology).unwrap();
        }

        let duration = start.elapsed();
        let avg_time = duration / iterations;

        println!("Deep hierarchy inference:");
        println!("  Average: {:?}", avg_time);
        println!("  Throughput: {:.2} inferences/sec", iterations as f64 / duration.as_secs_f64());
    }

    #[tokio::test]
    async fn bench_inference_large_ontology() {
        let sizes = vec![100, 500, 1000];

        for size in sizes {
            let ontology = create_large_ontology(size);
            let reasoner = CustomReasoner::new();

            let start = Instant::now();
            let result = reasoner.infer_axioms(&ontology).unwrap();
            let duration = start.elapsed();

            println!("Large ontology ({} classes):", size);
            println!("  Inference time: {:?}", duration);
            println!("  Inferred axioms: {}", result.len());
            println!("  Classes per second: {:.2}", size as f64 / duration.as_secs_f64());

            // Performance assertions
            if size == 100 {
                assert!(duration.as_secs() < 1, "100 classes should complete < 1s");
            } else if size == 500 {
                assert!(duration.as_secs() < 5, "500 classes should complete < 5s");
            } else if size == 1000 {
                assert!(duration.as_secs() < 10, "1000 classes should complete < 10s");
            }
        }
    }

    #[tokio::test]
    async fn bench_cache_hit_performance() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_simple_hierarchy();

        // Prime cache
        cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        // Benchmark cache hits
        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        }

        let duration = start.elapsed();
        let avg_time = duration / iterations;

        println!("Cache hit performance:");
        println!("  Average: {:?}", avg_time);
        println!("  Throughput: {:.2} lookups/sec", iterations as f64 / duration.as_secs_f64());

        assert!(avg_time.as_micros() < 1000, "Cache hits should be < 1ms");
    }

    #[tokio::test]
    async fn bench_cache_miss_performance() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_simple_hierarchy();

        let iterations = 10;
        let start = Instant::now();

        for i in 0..iterations {
            let _ = cache.get_or_compute(i, &reasoner, &ontology).unwrap();
        }

        let duration = start.elapsed();
        let avg_time = duration / iterations;

        println!("Cache miss performance:");
        println!("  Average: {:?}", avg_time);
        println!("  Throughput: {:.2} computes/sec", iterations as f64 / duration.as_secs_f64());
    }

    #[tokio::test]
    async fn bench_cache_speedup_ratio() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_large_ontology(500);

        // Cache miss
        let start = Instant::now();
        cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        let miss_duration = start.elapsed();

        // Cache hit
        let start = Instant::now();
        cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        let hit_duration = start.elapsed();

        let speedup = miss_duration.as_secs_f64() / hit_duration.as_secs_f64();

        println!("Cache speedup analysis:");
        println!("  Cache miss: {:?}", miss_duration);
        println!("  Cache hit: {:?}", hit_duration);
        println!("  Speedup: {:.2}x", speedup);

        assert!(speedup > 10.0, "Cache should provide >10x speedup for large ontologies");
    }

    #[tokio::test]
    async fn bench_constraint_generation() {
        let mut mapper = AxiomMapper::new();

        let axiom_counts = vec![10, 100, 1000];

        for count in axiom_counts {
            let axioms: Vec<OWLAxiom> = (0..count)
                .map(|i| {
                    OWLAxiom::asserted(MapperAxiomType::SubClassOf {
                        subclass: i as i64,
                        superclass: (i + 1) as i64,
                    })
                })
                .collect();

            let start = Instant::now();
            let constraints = mapper.translate_axioms(&axioms);
            let duration = start.elapsed();

            println!("Constraint generation ({} axioms):", count);
            println!("  Time: {:?}", duration);
            println!("  Constraints: {}", constraints.len());
            println!("  Throughput: {:.2} axioms/sec", count as f64 / duration.as_secs_f64());

            if count == 10 {
                assert!(duration.as_micros() < 1000, "10 axioms should complete < 1ms");
            } else if count == 100 {
                assert!(duration.as_millis() < 10, "100 axioms should complete < 10ms");
            } else if count == 1000 {
                assert!(duration.as_millis() < 100, "1000 axioms should complete < 100ms");
            }
        }
    }

    #[tokio::test]
    async fn bench_disjoint_constraint_generation() {
        let mut mapper = AxiomMapper::new();

        // Test O(n²) disjoint class constraint generation
        let class_counts = vec![5, 10, 20, 50];

        for count in class_counts {
            let classes: Vec<i64> = (0..count).collect();
            let axiom = OWLAxiom::asserted(MapperAxiomType::DisjointClasses { classes });

            let start = Instant::now();
            let constraints = mapper.translate_axiom(&axiom);
            let duration = start.elapsed();

            let expected_constraints = (count * (count - 1)) / 2;

            println!("Disjoint constraints ({} classes):", count);
            println!("  Time: {:?}", duration);
            println!("  Constraints: {} (expected: {})", constraints.len(), expected_constraints);
            println!("  Time per constraint: {:?}", duration / expected_constraints as u32);

            assert_eq!(constraints.len(), expected_constraints as usize);
        }
    }

    #[tokio::test]
    async fn bench_full_pipeline_end_to_end() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let mut mapper = AxiomMapper::new();

        let ontologies = vec![
            ("simple", create_simple_hierarchy()),
            ("deep", create_deep_hierarchy()),
            ("diamond", create_diamond_pattern()),
            ("equivalent", create_equivalent_classes()),
        ];

        for (name, ontology) in ontologies {
            let start = Instant::now();

            // Step 1: Inference
            let inferred = cache.get_or_compute(1, &reasoner, &ontology).unwrap();

            // Step 2: Convert to axioms (simplified)
            let mapper_axioms: Vec<OWLAxiom> = (0..inferred.len())
                .map(|i| OWLAxiom::inferred(MapperAxiomType::SubClassOf {
                    subclass: i as i64,
                    superclass: (i + 1) as i64,
                }))
                .collect();

            // Step 3: Generate constraints
            let constraints = mapper.translate_axioms(&mapper_axioms);

            let duration = start.elapsed();

            println!("Full pipeline ({}):", name);
            println!("  Total time: {:?}", duration);
            println!("  Inferred axioms: {}", inferred.len());
            println!("  Constraints: {}", constraints.len());
            println!("  Pipeline throughput: {:.2} ops/sec", 3.0 / duration.as_secs_f64());

            assert!(duration.as_millis() < 500, "Full pipeline should complete < 500ms");
        }
    }

    #[tokio::test]
    async fn bench_memory_usage() {
        use sysinfo::{System};

        let mut sys = System::new_all();
        sys.refresh_all();

        let initial_memory = sys.used_memory();

        // Create large ontology and process
        let ontology = create_large_ontology(1000);
        let reasoner = CustomReasoner::new();
        let _result = reasoner.infer_axioms(&ontology).unwrap();

        sys.refresh_all();
        let final_memory = sys.used_memory();

        let memory_increase = final_memory.saturating_sub(initial_memory);

        println!("Memory usage:");
        println!("  Initial: {} KB", initial_memory);
        println!("  Final: {} KB", final_memory);
        println!("  Increase: {} KB", memory_increase);
        println!("  Per class: {} bytes", (memory_increase * 1024) / 1000);

        // Memory should be reasonable
        assert!(memory_increase < 100_000, "Memory increase should be < 100 MB");
    }

    #[tokio::test]
    async fn bench_concurrent_inference() {
        use std::sync::Arc;
        use tokio::task;

        let reasoner = Arc::new(CustomReasoner::new());
        let ontology = Arc::new(create_simple_hierarchy());

        let num_tasks = 10;
        let iterations_per_task = 10;

        let start = Instant::now();

        let mut handles = vec![];
        for _ in 0..num_tasks {
            let reasoner_clone = Arc::clone(&reasoner);
            let ontology_clone = Arc::clone(&ontology);

            let handle = task::spawn(async move {
                for _ in 0..iterations_per_task {
                    let _ = reasoner_clone.infer_axioms(ontology_clone.as_ref());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let duration = start.elapsed();
        let total_inferences = num_tasks * iterations_per_task;

        println!("Concurrent inference:");
        println!("  Total time: {:?}", duration);
        println!("  Total inferences: {}", total_inferences);
        println!("  Throughput: {:.2} inferences/sec", total_inferences as f64 / duration.as_secs_f64());
    }
}

#[cfg(test)]
mod scalability_tests {
    use super::*;

    #[tokio::test]
    async fn test_scalability_linear_growth() {
        let reasoner = CustomReasoner::new();
        let sizes = vec![10, 50, 100, 200];
        let mut times = vec![];

        for size in &sizes {
            let ontology = create_large_ontology(*size);

            let start = Instant::now();
            let _ = reasoner.infer_axioms(&ontology).unwrap();
            let duration = start.elapsed();

            times.push(duration.as_secs_f64());
            println!("Size {}: {:?}", size, duration);
        }

        // Check that growth is roughly linear (not exponential)
        let ratio_1 = times[1] / times[0]; // 50/10
        let ratio_2 = times[2] / times[1]; // 100/50
        let ratio_3 = times[3] / times[2]; // 200/100

        println!("Growth ratios: {:.2}, {:.2}, {:.2}", ratio_1, ratio_2, ratio_3);

        // Ratios should be relatively consistent for linear growth
        assert!(ratio_1 < 10.0, "Should not have exponential growth");
        assert!(ratio_2 < 10.0, "Should not have exponential growth");
        assert!(ratio_3 < 10.0, "Should not have exponential growth");
    }

    #[tokio::test]
    async fn test_cache_scalability() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();

        // Test cache with many entries
        for i in 0..100 {
            let ontology = create_simple_hierarchy();
            cache.get_or_compute(i, &reasoner, &ontology).unwrap();
        }

        let stats = cache.get_stats().unwrap();
        println!("Cache with 100 entries:");
        println!("  Total entries: {}", stats.total_entries);
        println!("  Total size: {} KB", stats.total_size_bytes / 1024);
        println!("  Avg size per entry: {} bytes", stats.total_size_bytes / stats.total_entries);

        assert_eq!(stats.total_entries, 100);
    }
}
