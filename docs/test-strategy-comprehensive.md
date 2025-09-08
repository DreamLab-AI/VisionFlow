# Comprehensive Test Strategy for GPU Analytics Engine Upgrade
## Hive Mind Tester Agent Analysis

**Date:** 2025-09-07
**Agent:** Tester (Hive Collective)
**Session:** swarm-1757250489959-63an2p3d0

---

## Executive Summary

Based on analysis of the existing test structure and the GPU Analytics Engine upgrade requirements, this document outlines a comprehensive testing strategy covering all phases of the maturation plan. The strategy addresses current coverage gaps, introduces new validation frameworks, and provides CI/CD improvements for GPU-enabled testing.

## Current Test Coverage Analysis

### Existing Test Infrastructure

**Strong Areas:**
- ✅ PTX smoke test framework (`tests/ptx_smoke_test.rs`) with GPU-gated execution
- ✅ Comprehensive GPU safety validation (`tests/gpu_safety_tests.rs`)
- ✅ Analytics API endpoint structure validation (`tests/analytics_endpoints_test.rs`)
- ✅ Extensive test utilities for settings and configuration validation

**Coverage Gaps:**
- ❌ GPU kernel correctness validation (beyond smoke tests)
- ❌ Buffer resize state preservation testing
- ❌ Constraint stability regression tests
- ❌ SSSP accuracy validation against CPU reference
- ❌ Spatial hashing efficiency benchmarks
- ❌ Live data preservation during scaling operations
- ❌ CI pipeline GPU execution environment

---

## Test Strategy by Phase

### Phase 0: PTX Pipeline Hardening

#### 1. **Enhanced PTX Validation Tests**
```rust
// //tests/ptx_validation_comprehensive.rs

#[cfg(test)]
mod ptx_comprehensive_tests {
    use crate::utils::ptx::*;
    use crate::utils::gpu_diagnostics::*;

    /// Test PTX compilation across CUDA architectures
    #[test]
    fn test_multi_arch_ptx_compilation() {
        let architectures = vec!["61", "70", "75", "80", "86", "89"];

        for arch in architectures {
            std::env::set_var("CUDA_ARCH", arch);
            let result = load_ptx_sync();
            assert!(result.is_ok(), "PTX compilation failed for arch {}", arch);

            let ptx_content = result.unwrap();
            validate_ptx_content(&ptx_content, arch);
        }
    }

    /// Validate kernel symbol completeness
    #[test]
    fn test_kernel_symbol_completeness() {
        let required_kernels = [
            "build_grid_kernel",
            "compute_cell_bounds_kernel",
            "force_pass_kernel",
            "integrate_pass_kernel",
            "relaxation_step_kernel",
            // Phase 2 analytics kernels
            "kmeans_init_kernel",
            "kmeans_assign_kernel",
            "anomaly_score_kernel"
        ];

        let ptx = load_ptx_sync().expect("PTX should load");
        let module = create_test_cuda_module(&ptx);

        for kernel in required_kernels {
            assert!(module.get_function(kernel).is_ok(),
                   "Missing kernel: {}", kernel);
        }
    }

    /// Test fallback compilation under various failure scenarios
    #[test]
    fn test_compilation_fallback_scenarios() {
        // Test missing PTX file
        std::env::set_var("VISIONFLOW_PTX_PATH", "/nonexistent/path.ptx");

        let result = load_ptx_sync();
        assert!(result.is_ok(), "Fallback compilation should succeed");

        // Test Docker environment path resolution
        std::env::set_var("DOCKER_ENV", "1");
        let result = load_ptx_sync();
        assert!(result.is_ok(), "Docker environment fallback should work");
    }
}
```

#### 2. **Cold Start Performance Tests**
```rust
#[test]
fn test_cold_start_performance() {
    let start_time = Instant::now();

    // Full cold start simulation
    let ptx = load_ptx_sync().expect("PTX load");
    let _ctx = create_cuda_context().expect("CUDA context");
    let _module = Module::from_ptx(&ptx, &[]).expect("Module creation");
    let _gpu = UnifiedGPUCompute::new(1000, 2000, &ptx).expect("GPU compute");

    let total_time = start_time.elapsed();
    assert!(total_time.as_secs() < 3,
           "Cold start should complete within 3 seconds, took {:?}", total_time);
}
```

---

### Phase 1: Core Engine Stabilization

#### 3. **Buffer Management Integration Tests**
```rust
// //tests/buffer_resize_integration.rs

#[cfg(test)]
mod buffer_resize_tests {
    use crate::utils::unified_gpu_compute::*;
    use crate::actors::gpu_compute_actor::*;

    #[tokio::test]
    async fn test_live_buffer_resize_preservation() {
        let initial_nodes = vec![
            (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)
        ];

        let ptx = load_test_ptx();
        let mut gpu = UnifiedGPUCompute::new(3, 2, &ptx).unwrap();

        // Upload initial data
        gpu.upload_node_positions(&initial_nodes).unwrap();

        // Resize to larger buffer
        gpu.resize_buffers(5, 4).unwrap();

        // Verify data preservation
        let preserved_positions = gpu.download_node_positions(3).unwrap();
        for (i, ((orig_x, orig_y, orig_z), (pres_x, pres_y, pres_z)))
            in initial_nodes.iter().zip(preserved_positions.iter()).enumerate() {

            let error = ((orig_x - pres_x).powi(2) +
                        (orig_y - pres_y).powi(2) +
                        (orig_z - pres_z).powi(2)).sqrt();

            assert!(error < 1e-6,
                   "Node {} position error {:.2e} exceeds tolerance", i, error);
        }

        // Test shrink operation
        gpu.resize_buffers(2, 1).unwrap();
        let shrunk_positions = gpu.download_node_positions(2).unwrap();

        // First 2 nodes should be preserved
        for i in 0..2 {
            let (orig_x, orig_y, orig_z) = initial_nodes[i];
            let (pres_x, pres_y, pres_z) = shrunk_positions[i];

            let error = ((orig_x - pres_x).powi(2) +
                        (orig_y - pres_y).powi(2) +
                        (orig_z - pres_z).powi(2)).sqrt();

            assert!(error < 1e-6,
                   "Shrink: Node {} position error {:.2e} exceeds tolerance", i, error);
        }
    }

    #[tokio::test]
    async fn test_actor_resize_integration() {
        let actor = create_test_gpu_actor().await;

        // Simulate graph size change through actor
        let initial_graph_data = create_test_graph_data(100, 150);
        actor.update_graph_data_internal(initial_graph_data).await.unwrap();

        // Verify initial state
        let stats = actor.get_buffer_stats().await.unwrap();
        assert_eq!(stats.node_count, 100);
        assert_eq!(stats.edge_count, 150);

        // Trigger resize through graph data update
        let larger_graph_data = create_test_graph_data(200, 350);
        actor.update_graph_data_internal(larger_graph_data).await.unwrap();

        // Verify resize occurred and data preserved
        let updated_stats = actor.get_buffer_stats().await.unwrap();
        assert_eq!(updated_stats.node_count, 200);
        assert_eq!(updated_stats.edge_count, 350);

        // Verify no NaN/panic conditions
        let positions = actor.get_current_positions().await.unwrap();
        assert!(positions.iter().all(|&(x, y, z)|
               x.is_finite() && y.is_finite() && z.is_finite()));
    }
}
```

#### 4. **Constraint Stability Regression Tests**
```rust
// //tests/constraint_stability_regression.rs

#[cfg(test)]
mod constraint_stability_tests {
    use crate::models::constraints::*;
    use crate::utils::unified_gpu_compute::*;

    #[test]
    fn test_constraint_oscillation_prevention() {
        let scenarios = vec![
            ("hierarchical_layout", create_hierarchical_constraints()),
            ("cluster_preservation", create_cluster_constraints()),
            ("path_constraints", create_path_constraints()),
        ];

        for (scenario_name, constraints) in scenarios {
            println!("Testing {} constraint stability", scenario_name);

            let mut gpu = create_test_gpu_compute();
            gpu.set_constraints(constraints).unwrap();

            let mut kinetic_energies = Vec::new();
            let mut constraint_violations = Vec::new();

            // Run simulation for 10 frames
            for frame in 0..10 {
                gpu.execute_physics_step(0.016).unwrap();

                let ke = gpu.calculate_kinetic_energy().unwrap();
                let violations = gpu.count_constraint_violations().unwrap();

                kinetic_energies.push(ke);
                constraint_violations.push(violations);

                // Check for NaN/Inf
                assert!(ke.is_finite(), "Kinetic energy should be finite at frame {}", frame);
            }

            // Verify energy decay (no oscillation)
            let energy_increases = kinetic_energies.windows(2)
                .filter(|window| window[1] > window[0] * 1.1) // 10% increase threshold
                .count();

            assert!(energy_increases <= 1,
                   "Too many energy increases ({}) in {}", energy_increases, scenario_name);

            // Verify violations decrease
            assert!(constraint_violations[9] <= constraint_violations[0],
                   "Constraint violations should decrease in {}", scenario_name);

            // Verify return to baseline within 120 frames (2 seconds at 60fps)
            let baseline_energy = kinetic_energies[9];
            assert!(baseline_energy < kinetic_energies[0] * 0.1,
                   "Should reach low energy state in {}", scenario_name);
        }
    }

    #[test]
    fn test_constraint_force_capping() {
        let extreme_constraints = create_extreme_test_constraints();
        let mut gpu = create_test_gpu_compute();
        gpu.set_constraints(extreme_constraints).unwrap();

        // Apply very high forces
        let sim_params = SimParams {
            constraint_strength: 1000.0, // Extreme value
            ..Default::default()
        };

        gpu.execute_physics_step_with_params(0.016, &sim_params).unwrap();

        let velocities = gpu.download_velocities().unwrap();
        let max_velocity = velocities.iter()
            .map(|&(vx, vy, vz)| (vx*vx + vy*vy + vz*vz).sqrt())
            .fold(0.0f32, f32::max);

        // Verify force capping prevents explosive velocities
        assert!(max_velocity < 10.0,
               "Max velocity {} should be capped below 10.0", max_velocity);
    }
}
```

#### 5. **SSSP Accuracy Validation Framework**
```rust
// //tests/sssp_accuracy_validation.rs

#[cfg(test)]
mod sssp_accuracy_tests {
    use crate::utils::unified_gpu_compute::*;
    use crate::algorithms::sssp_cpu_reference::*;

    #[test]
    fn test_sssp_cpu_parity() {
        let test_graphs = vec![
            ("grid_10x10", create_grid_graph(10, 10)),
            ("random_sparse", create_random_graph(100, 200)),
            ("scale_free", create_scale_free_graph(50, 150)),
            ("complete_small", create_complete_graph(20)),
        ];

        let tolerance = 1e-5f32;

        for (graph_name, (nodes, edges)) in test_graphs {
            println!("Testing SSSP accuracy on {}", graph_name);

            // CPU reference implementation
            let cpu_distances = compute_sssp_cpu(&nodes, &edges, 0);

            // GPU implementation
            let mut gpu = create_test_gpu_compute();
            gpu.upload_graph_data(&nodes, &edges).unwrap();
            let gpu_distances = gpu.run_sssp(0).unwrap();

            // Compare results
            let max_error = cpu_distances.iter().zip(gpu_distances.iter())
                .map(|(cpu_dist, gpu_dist)| (cpu_dist - gpu_dist).abs())
                .fold(0.0f32, f32::max);

            assert!(max_error < tolerance,
                   "SSSP error {:.2e} exceeds tolerance {:.2e} for {}",
                   max_error, tolerance, graph_name);

            // Test spring length improvement
            let edge_variance_before = calculate_edge_length_variance(&nodes, &edges);

            // Apply SSSP-adjusted springs
            gpu.enable_sssp_spring_adjustment(true).unwrap();
            gpu.execute_physics_step(0.016).unwrap();

            let adjusted_positions = gpu.download_node_positions(nodes.len()).unwrap();
            let edge_variance_after = calculate_edge_length_variance(&adjusted_positions, &edges);

            let improvement = (edge_variance_before - edge_variance_after) / edge_variance_before;
            assert!(improvement >= 0.1,
                   "SSSP spring adjustment should improve variance by ≥10%, got {:.1}%",
                   improvement * 100.0);
        }
    }

    #[test]
    fn test_sssp_api_toggle() {
        // Test API control of SSSP feature
        let mut gpu = create_test_gpu_compute();

        // Initially disabled
        assert!(!gpu.is_sssp_enabled().unwrap());

        // Enable via API
        gpu.set_feature_flag(FeatureFlags::ENABLE_SSSP_SPRING_ADJUST, true).unwrap();
        assert!(gpu.is_sssp_enabled().unwrap());

        // Disable via API
        gpu.set_feature_flag(FeatureFlags::ENABLE_SSSP_SPRING_ADJUST, false).unwrap();
        assert!(!gpu.is_sssp_enabled().unwrap());
    }
}
```

---

### Phase 2: GPU Analytics Implementation

#### 6. **K-means Clustering Validation Harness**
```rust
// //tests/gpu_kmeans_validation.rs

#[cfg(test)]
mod gpu_kmeans_tests {
    use crate::utils::unified_gpu_compute::*;
    use crate::algorithms::kmeans_cpu_reference::*;

    #[test]
    fn test_kmeans_accuracy_validation() {
        let test_datasets = vec![
            ("synthetic_2d_gaussian", create_gaussian_clusters_2d(1000, 5)),
            ("graph_node_features", extract_graph_node_features(500, 3)),
            ("benchmark_iris", load_iris_dataset()),
        ];

        for (dataset_name, (data, true_labels, k)) in test_datasets {
            println!("Testing K-means on {} ({} points, k={})",
                   dataset_name, data.len(), k);

            let seed = 42;

            // CPU reference
            let cpu_labels = kmeans_cpu(&data, k, seed, 100);
            let cpu_ari = calculate_ari(&true_labels, &cpu_labels);
            let cpu_nmi = calculate_nmi(&true_labels, &cpu_labels);

            // GPU implementation
            let mut gpu = create_test_gpu_compute();
            gpu.upload_clustering_data(&data).unwrap();
            let gpu_labels = gpu.run_kmeans(k, seed, 100).unwrap();
            let gpu_ari = calculate_ari(&true_labels, &gpu_labels);
            let gpu_nmi = calculate_nmi(&true_labels, &gpu_labels);

            // Accuracy requirements (within 2-5% of CPU)
            let ari_diff = (gpu_ari - cpu_ari).abs();
            let nmi_diff = (gpu_nmi - cpu_nmi).abs();

            assert!(ari_diff <= 0.05,
                   "ARI difference {:.3} exceeds 5% for {}", ari_diff, dataset_name);
            assert!(nmi_diff <= 0.05,
                   "NMI difference {:.3} exceeds 5% for {}", nmi_diff, dataset_name);

            println!("  ✓ {} - CPU ARI: {:.3}, GPU ARI: {:.3}, diff: {:.3}",
                   dataset_name, cpu_ari, gpu_ari, ari_diff);
        }
    }

    #[test]
    fn test_kmeans_deterministic_seeding() {
        let data = create_test_clustering_data(100, 3);
        let k = 5;
        let seed = 42;

        let mut gpu = create_test_gpu_compute();
        gpu.upload_clustering_data(&data).unwrap();

        // Run multiple times with same seed
        let mut results = Vec::new();
        for run in 0..3 {
            let labels = gpu.run_kmeans(k, seed, 50).unwrap();
            results.push(labels);
        }

        // Verify deterministic results
        for i in 1..results.len() {
            assert_eq!(results[0], results[i],
                      "Run {} should match run 0 with same seed {}", i, seed);
        }

        // Test different seed gives different result
        let different_labels = gpu.run_kmeans(k, 123, 50).unwrap();
        assert_ne!(results[0], different_labels,
                  "Different seeds should produce different results");
    }

    #[test]
    fn test_kmeans_performance_scaling() {
        let node_counts = vec![1000, 10000, 100000];
        let k = 10;

        for node_count in node_counts {
            let data = create_large_test_dataset(node_count);

            let start_time = Instant::now();
            let mut gpu = create_test_gpu_compute();
            gpu.upload_clustering_data(&data).unwrap();
            let _labels = gpu.run_kmeans(k, 42, 50).unwrap();
            let gpu_time = start_time.elapsed();

            // Performance requirements
            if node_count >= 100000 {
                let nodes_per_ms = node_count as f32 / gpu_time.as_millis() as f32;
                assert!(nodes_per_ms >= 1000.0,
                       "Should process ≥1000 nodes/ms for 100k+ nodes, got {:.1}",
                       nodes_per_ms);
            }

            println!("K-means {} nodes: {:.1}ms ({:.0} nodes/ms)",
                   node_count, gpu_time.as_millis(),
                   node_count as f32 / gpu_time.as_millis() as f32);
        }
    }
}
```

#### 7. **Anomaly Detection AUC Validation**
```rust
// //tests/gpu_anomaly_validation.rs

#[cfg(test)]
mod gpu_anomaly_tests {
    use crate::utils::unified_gpu_compute::*;
    use crate::algorithms::anomaly_cpu_reference::*;

    #[test]
    fn test_anomaly_detection_auc() {
        let anomaly_scenarios = vec![
            ("positional_outliers", create_positional_anomalies(1000, 50)),
            ("degree_anomalies", create_degree_anomalies(2000, 100)),
            ("velocity_outliers", create_velocity_anomalies(5000, 250)),
            ("structural_anomalies", create_structural_anomalies(1000, 80)),
        ];

        for (scenario_name, (data, true_anomalies)) in anomaly_scenarios {
            println!("Testing anomaly detection on {} ({} total, {} anomalies)",
                   scenario_name, data.len(), true_anomalies.iter().filter(|&&x| x).count());

            let start_time = Instant::now();

            // GPU anomaly detection
            let mut gpu = create_test_gpu_compute();
            gpu.upload_anomaly_data(&data).unwrap();
            let anomaly_scores = gpu.run_anomaly_detection("isolation_forest").unwrap();
            let detection_time = start_time.elapsed();

            // Calculate AUC score
            let auc_score = calculate_auc(&true_anomalies, &anomaly_scores);

            // AUC requirement: ≥0.85
            assert!(auc_score >= 0.85,
                   "AUC {:.3} should be ≥0.85 for {}", auc_score, scenario_name);

            // Latency requirement for large datasets
            if data.len() >= 100000 {
                let nodes_per_ms = data.len() as f32 / detection_time.as_millis() as f32;
                assert!(nodes_per_ms >= 1000.0,
                       "Should process ≥1000 nodes/ms, got {:.1}", nodes_per_ms);
            }

            println!("  ✓ {} - AUC: {:.3}, latency: {:.1}ms",
                   scenario_name, auc_score, detection_time.as_millis());
        }
    }

    #[test]
    fn test_anomaly_detection_methods() {
        let data = create_test_anomaly_dataset(1000, 100);

        let methods = vec![
            "isolation_forest",
            "lof",
            "statistical",
            "autoencoder"
        ];

        for method in methods {
            let mut gpu = create_test_gpu_compute();
            gpu.upload_anomaly_data(&data.0).unwrap();

            let result = gpu.run_anomaly_detection(method);
            assert!(result.is_ok(), "Method {} should be supported", method);

            let scores = result.unwrap();
            assert_eq!(scores.len(), data.0.len());
            assert!(scores.iter().all(|&score| score >= 0.0 && score <= 1.0),
                   "Scores should be normalized [0,1] for {}", method);
        }
    }
}
```

---

### Phase 3: Performance and Observability

#### 8. **Spatial Hashing Efficiency Tests**
```rust
// //tests/spatial_hashing_efficiency.rs

#[cfg(test)]
mod spatial_hashing_tests {
    use crate::utils::unified_gpu_compute::*;

    #[test]
    fn test_spatial_hashing_efficiency() {
        let workloads = vec![
            ("uniform_1000", create_uniform_distribution(1000)),
            ("clustered_1000", create_clustered_distribution(1000, 5)),
            ("sparse_2000", create_sparse_distribution(2000)),
            ("dense_500", create_dense_distribution(500)),
        ];

        for (workload_name, positions) in workloads {
            println!("Testing spatial hashing efficiency: {}", workload_name);

            let mut gpu = create_test_gpu_compute();
            gpu.upload_node_positions(&positions).unwrap();

            let start_time = Instant::now();
            gpu.build_spatial_grid().unwrap();
            let build_time = start_time.elapsed();

            let hashing_stats = gpu.get_spatial_hashing_stats().unwrap();
            let efficiency = hashing_stats.non_empty_cells as f32 / hashing_stats.total_cells as f32;

            // Efficiency should be 0.2-0.6
            assert!(efficiency >= 0.2 && efficiency <= 0.6,
                   "Efficiency {:.3} should be 0.2-0.6 for {}", efficiency, workload_name);

            // Test scaling behavior
            let doubled_positions = double_workload(&positions);
            gpu.upload_node_positions(&doubled_positions).unwrap();

            let doubled_start = Instant::now();
            gpu.build_spatial_grid().unwrap();
            let doubled_time = doubled_start.elapsed();

            let expected_doubled_time = build_time * 2;
            let time_variance = (doubled_time.as_millis() as f32 - expected_doubled_time.as_millis() as f32).abs()
                              / expected_doubled_time.as_millis() as f32;

            assert!(time_variance < 0.5,
                   "Time variance {:.3} should be <50% for node doubling", time_variance);

            println!("  ✓ {} - Efficiency: {:.3}, scaling variance: {:.3}",
                   workload_name, efficiency, time_variance);
        }
    }

    #[test]
    fn test_dynamic_cell_buffer_sizing() {
        let mut gpu = create_test_gpu_compute();

        // Start with small graph
        let small_positions = create_test_positions(100);
        gpu.upload_node_positions(&small_positions).unwrap();
        gpu.build_spatial_grid().unwrap();

        let initial_stats = gpu.get_spatial_hashing_stats().unwrap();

        // Scale up dramatically (should trigger cell buffer resize)
        let large_positions = create_test_positions(10000);
        gpu.upload_node_positions(&large_positions).unwrap();

        let result = gpu.build_spatial_grid();
        assert!(result.is_ok(), "Should handle cell buffer resizing automatically");

        let scaled_stats = gpu.get_spatial_hashing_stats().unwrap();
        assert!(scaled_stats.total_cells > initial_stats.total_cells,
               "Cell buffer should have been resized");

        // Should not produce overflow errors
        assert!(scaled_stats.overflow_count == 0,
               "No overflow errors should occur after dynamic resizing");
    }
}
```

#### 9. **Performance Benchmark Framework**
```rust
// //tests/performance_benchmarks.rs

#[cfg(test)]
mod performance_benchmarks {
    use crate::utils::unified_gpu_compute::*;
    use std::time::Instant;
    use std::collections::HashMap;

    #[test]
    fn test_kernel_performance_benchmarks() {
        let test_configs = vec![
            ("small", 1000, 2000),
            ("medium", 10000, 20000),
            ("large", 100000, 200000),
        ];

        let mut benchmark_results = HashMap::new();

        for (config_name, node_count, edge_count) in test_configs {
            println!("Benchmarking {} configuration ({} nodes, {} edges)",
                   config_name, node_count, edge_count);

            let positions = create_test_positions(node_count);
            let edges = create_test_edges(edge_count, node_count);

            let mut gpu = create_test_gpu_compute();
            gpu.upload_graph_data(&positions, &edges).unwrap();

            // Benchmark individual kernels
            let mut kernel_times = HashMap::new();

            // Force computation kernel
            let start = Instant::now();
            gpu.run_force_kernel().unwrap();
            kernel_times.insert("force_kernel", start.elapsed());

            // Integration kernel
            let start = Instant::now();
            gpu.run_integration_kernel().unwrap();
            kernel_times.insert("integration_kernel", start.elapsed());

            // Spatial hashing
            let start = Instant::now();
            gpu.build_spatial_grid().unwrap();
            kernel_times.insert("spatial_hashing", start.elapsed());

            // SSSP kernel
            let start = Instant::now();
            gpu.run_sssp(0).unwrap();
            kernel_times.insert("sssp_kernel", start.elapsed());

            benchmark_results.insert(config_name.to_string(), kernel_times);

            // Performance expectations
            let total_time: Duration = kernel_times.values().sum();
            let fps = 1.0 / total_time.as_secs_f32();

            match config_name {
                "small" => assert!(fps >= 60.0, "Small config should achieve ≥60 FPS"),
                "medium" => assert!(fps >= 30.0, "Medium config should achieve ≥30 FPS"),
                "large" => assert!(fps >= 10.0, "Large config should achieve ≥10 FPS"),
                _ => {}
            }

            println!("  ✓ {} - Total: {:.1}ms ({:.1} FPS)",
                   config_name, total_time.as_millis(), fps);
        }

        // Output detailed benchmark report
        output_benchmark_report(&benchmark_results);
    }

    #[test]
    fn test_memory_usage_benchmarks() {
        let configs = vec![
            (1000, 2000),
            (10000, 20000),
            (100000, 200000),
        ];

        for (nodes, edges) in configs {
            let mut gpu = create_test_gpu_compute();

            let initial_memory = gpu.get_memory_usage().unwrap();

            // Upload data
            let positions = create_test_positions(nodes);
            let edges_data = create_test_edges(edges, nodes);

            gpu.upload_graph_data(&positions, &edges_data).unwrap();
            let loaded_memory = gpu.get_memory_usage().unwrap();

            // Calculate memory efficiency
            let expected_memory = nodes * 16 + edges * 12; // Rough estimate
            let actual_memory = loaded_memory.device_allocated - initial_memory.device_allocated;
            let memory_efficiency = expected_memory as f32 / actual_memory as f32;

            assert!(memory_efficiency >= 0.7,
                   "Memory efficiency {:.3} should be ≥70% for {} nodes",
                   memory_efficiency, nodes);

            println!("Memory usage {} nodes: {} MB (efficiency: {:.1}%)",
                   nodes, actual_memory / (1024 * 1024), memory_efficiency * 100.0);
        }
    }

    #[test]
    fn test_throughput_scaling() {
        let base_nodes = 1000;
        let scaling_factors = vec![1, 2, 4, 8, 16];

        let mut throughput_results = Vec::new();

        for factor in scaling_factors {
            let nodes = base_nodes * factor;
            let edges = nodes * 2;

            let positions = create_test_positions(nodes);
            let edges_data = create_test_edges(edges, nodes);

            let mut gpu = create_test_gpu_compute();
            gpu.upload_graph_data(&positions, &edges_data).unwrap();

            let start = Instant::now();

            // Run multiple physics steps
            for _ in 0..10 {
                gpu.execute_physics_step(0.016).unwrap();
            }

            let total_time = start.elapsed();
            let throughput = (nodes * 10) as f32 / total_time.as_secs_f32();

            throughput_results.push((nodes, throughput));

            println!("Throughput {} nodes: {:.0} nodes/sec", nodes, throughput);
        }

        // Verify sub-linear scaling (good GPU utilization)
        for i in 1..throughput_results.len() {
            let (prev_nodes, prev_throughput) = throughput_results[i-1];
            let (curr_nodes, curr_throughput) = throughput_results[i];

            let node_ratio = curr_nodes as f32 / prev_nodes as f32;
            let throughput_ratio = curr_throughput / prev_throughput;

            // Throughput should not decrease dramatically with scale
            assert!(throughput_ratio >= 0.7,
                   "Throughput ratio {:.3} should be ≥70% when scaling {}x nodes",
                   throughput_ratio, node_ratio);
        }
    }
}
```

---

## CI/CD Pipeline Improvements

### 10. **GPU-Enabled CI Configuration**

```yaml
# .github/workflows/gpu-tests.yml
name: GPU Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu, nvidia]
    container:
      image: nvidia/cuda:12.0-devel-ubuntu20.04
      options: --gpus all

    env:
      CUDA_ARCH: "86"
      RUN_GPU_SMOKE: "1"
      RUST_LOG: "debug"

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true

    - name: Install CUDA Toolkit
      run: |
        apt-get update
        apt-get install -y nvidia-cuda-toolkit
        nvcc --version

    - name: Verify GPU Access
      run: |
        nvidia-smi
        cd / && echo "GPU available for testing"

    - name: Cache Dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-gpu-${{ hashFiles('**/Cargo.lock') }}

    - name: Build with GPU Support
      run: |
        cd /
        CUDA_ARCH=86 cargo build --release --features gpu-tests

    - name: Run PTX Smoke Tests
      run: |
        cd /
        RUN_GPU_SMOKE=1 cargo test --test ptx_smoke_test -- --nocapture

    - name: Run GPU Safety Tests
      run: |
        cd /
        cargo test gpu_safety_tests -- --nocapture

    - name: Run GPU Integration Tests
      run: |
        cd /
        cargo test --features gpu-integration -- --nocapture

    - name: Run Performance Benchmarks
      run: |
        cd /
        cargo test performance_benchmarks -- --nocapture --test-threads=1

    - name: Generate Test Report
      if: always()
      run: |
        cd /
        cargo test -- --format json > test-results.json

    - name: Upload Test Results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: gpu-test-results
        path: //test-results.json
```

### 11. **Test Automation Scripts**

```bash
#!/bin/bash
# scripts/run-gpu-test-suite.sh

set -e

echo "🚀 GPU Analytics Engine Test Suite"
echo "=================================="

# Environment validation
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC not found. Please install CUDA toolkit."
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "❌ No NVIDIA GPU detected."
    exit 1
fi

echo "✅ GPU environment validated"

# Build with appropriate CUDA architecture
CUDA_ARCH=${CUDA_ARCH:-86}
echo "🔧 Building for CUDA architecture: $CUDA_ARCH"

cd /
CUDA_ARCH=$CUDA_ARCH cargo build --release

# Test Categories
echo ""
echo "📋 Test Categories:"
echo "  1. PTX Pipeline & Cold Start"
echo "  2. Buffer Management & Resize"
echo "  3. Constraint Stability"
echo "  4. SSSP Accuracy Validation"
echo "  5. GPU Analytics (K-means, Anomaly)"
echo "  6. Spatial Hashing Efficiency"
echo "  7. Performance Benchmarks"
echo ""

# Run test categories
run_test_category() {
    local category=$1
    local tests=$2

    echo "🧪 Running $category tests..."

    if RUN_GPU_SMOKE=1 cargo test $tests -- --nocapture; then
        echo "✅ $category: PASSED"
    else
        echo "❌ $category: FAILED"
        exit 1
    fi
    echo ""
}

# Execute test suite
run_test_category "PTX Pipeline" "ptx_smoke_test ptx_validation_comprehensive"
run_test_category "Buffer Management" "buffer_resize_integration"
run_test_category "Constraint Stability" "constraint_stability_regression"
run_test_category "SSSP Accuracy" "sssp_accuracy_validation"
run_test_category "GPU Analytics" "gpu_kmeans_validation gpu_anomaly_validation"
run_test_category "Spatial Hashing" "spatial_hashing_efficiency"
run_test_category "Performance" "performance_benchmarks"

echo "🎉 All GPU test categories PASSED!"
echo ""
echo "📊 Test Summary:"
echo "  - PTX pipeline: Validated across architectures"
echo "  - Buffer resize: State preservation confirmed"
echo "  - Constraints: Stability without oscillation"
echo "  - SSSP: CPU parity within 1e-5 tolerance"
echo "  - Analytics: AUC ≥0.85, deterministic seeding"
echo "  - Spatial hashing: 0.2-0.6 efficiency range"
echo "  - Performance: Meeting FPS targets"
echo ""
echo "✅ GPU Analytics Engine: Ready for production"
```

---

## Validation Gates and Acceptance Criteria

### Phase 0 Gates
- [ ] PTX module loads on all supported CUDA architectures (61, 70, 75, 80, 86, 89)
- [ ] Cold start completes within 3 seconds
- [ ] All required kernels resolvable in PTX module
- [ ] Fallback compilation succeeds under failure scenarios
- [ ] CI smoke test passes on GPU runner

### Phase 1 Gates
- [ ] Buffer resize preserves existing data within 1e-6 tolerance
- [ ] Constraint violations decrease monotonically
- [ ] No energy oscillation (< 10% increases per window)
- [ ] SSSP accuracy within 1e-5 of CPU reference
- [ ] SSSP improves edge length variance by ≥10%
- [ ] Spatial hashing efficiency 0.2-0.6 across workloads
- [ ] Force capping prevents explosive velocities (max < 10.0)

### Phase 2 Gates
- [ ] K-means ARI/NMI within 5% of CPU reference
- [ ] K-means deterministic with same seed
- [ ] K-means 10-50× speedup at 100k+ nodes
- [ ] Anomaly detection AUC ≥ 0.85 on synthetic tests
- [ ] Anomaly detection ≥1000 nodes/ms processing rate
- [ ] All analytics methods support documented API

### Phase 3 Gates
- [ ] Kernel timing metrics exposed via API
- [ ] Memory usage tracking with <2% overhead
- [ ] Performance benchmarks meet FPS targets
- [ ] Throughput scaling maintains ≥70% efficiency
- [ ] CI pipeline validates GPU functionality

---

## Risk Mitigation and Safety

### Testing Safety Protocols
1. **Resource Limits**: All tests respect memory/time boundaries
2. **Graceful Degradation**: Tests validate CPU fallback scenarios
3. **Error Isolation**: Test failures don't cascade or corrupt state
4. **Concurrent Safety**: Multi-threaded test execution validated
5. **Data Validation**: All GPU outputs checked for NaN/Inf

### Continuous Validation
1. **Regression Prevention**: All tests in CI prevent feature regression
2. **Performance Monitoring**: Benchmark results tracked over time
3. **Accuracy Drift Detection**: Statistical validation prevents model drift
4. **Memory Leak Detection**: Long-running tests validate cleanup
5. **Stability Monitoring**: Physics simulation stability tracked

---

## Implementation Timeline

**Week 1-2: Phase 0 + Phase 1 Core Tests**
- PTX validation framework
- Buffer resize integration tests
- Constraint stability regression suite

**Week 3-4: Phase 1 Completion + Phase 2 Start**
- SSSP accuracy validation
- Spatial hashing efficiency tests
- K-means validation harness foundation

**Week 5-6: Phase 2 Analytics Testing**
- Complete K-means/anomaly validation
- Performance benchmark framework
- API endpoint testing

**Week 7-8: Phase 3 + CI Integration**
- Performance/observability tests
- CI pipeline GPU integration
- Documentation and training

---

This comprehensive test strategy ensures the GPU Analytics Engine upgrade maintains reliability, performance, and accuracy throughout all phases of development while providing robust validation frameworks for ongoing development.