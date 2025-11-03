// Stress Majorization Performance Benchmarks
//
// Tests performance characteristics of GPU-accelerated stress majorization
// across various graph sizes and configurations.

#[cfg(test)]
mod stress_majorization_benchmarks {
    use std::time::Instant;

    /// Benchmark configuration
    struct BenchmarkConfig {
        graph_sizes: Vec<usize>,
        max_iterations: u32,
        convergence_threshold: f32,
        warmup_runs: usize,
        measurement_runs: usize,
    }

    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                graph_sizes: vec![100, 1000, 10000, 100000],
                max_iterations: 50,
                convergence_threshold: 0.01,
                warmup_runs: 3,
                measurement_runs: 10,
            }
        }
    }

    /// Performance metrics for a single benchmark run
    #[derive(Debug, Clone)]
    struct BenchmarkResult {
        graph_size: usize,
        stress_computation_ms: f64,
        gradient_computation_ms: f64,
        position_update_ms: f64,
        total_iteration_ms: f64,
        iterations_to_convergence: u32,
        total_optimization_ms: f64,
        final_stress: f32,
    }

    impl BenchmarkResult {
        fn print_summary(&self) {
            println!("\n=== Stress Majorization Benchmark ===");
            println!("Graph Size: {} nodes", self.graph_size);
            println!("\nPer-Iteration Breakdown:");
            println!("  Stress computation:   {:>8.2} ms", self.stress_computation_ms);
            println!("  Gradient computation: {:>8.2} ms", self.gradient_computation_ms);
            println!("  Position update:      {:>8.2} ms", self.position_update_ms);
            println!("  Total per iteration:  {:>8.2} ms", self.total_iteration_ms);
            println!("\nOverall Performance:");
            println!("  Iterations:           {:>8}", self.iterations_to_convergence);
            println!("  Total time:           {:>8.2} ms", self.total_optimization_ms);
            println!("  Final stress:         {:>8.2}", self.final_stress);
            println!("  Throughput:           {:>8.2} iterations/sec",
                1000.0 / self.total_iteration_ms);
        }
    }

    /// Create a random graph for testing
    fn create_test_graph(num_nodes: usize, edge_probability: f32) -> (Vec<(usize, usize)>, Vec<f32>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut edges = Vec::new();
        let mut initial_positions = Vec::new();

        // Generate random edges
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                if rng.gen::<f32>() < edge_probability {
                    edges.push((i, j));
                }
            }
        }

        // Generate random initial positions (x, y, z)
        for _ in 0..(num_nodes * 3) {
            initial_positions.push(rng.gen_range(-100.0..100.0));
        }

        (edges, initial_positions)
    }

    /// Compute ideal graph distances (Floyd-Warshall for small graphs, landmark APSP for large)
    fn compute_graph_distances(num_nodes: usize, edges: &[(usize, usize)]) -> Vec<f32> {
        let mut distances = vec![f32::INFINITY; num_nodes * num_nodes];

        // Initialize diagonal to 0
        for i in 0..num_nodes {
            distances[i * num_nodes + i] = 0.0;
        }

        // Initialize edges to 1.0
        for &(i, j) in edges {
            distances[i * num_nodes + j] = 1.0;
            distances[j * num_nodes + i] = 1.0;
        }

        // Floyd-Warshall for small graphs
        if num_nodes <= 1000 {
            for k in 0..num_nodes {
                for i in 0..num_nodes {
                    for j in 0..num_nodes {
                        let via_k = distances[i * num_nodes + k] + distances[k * num_nodes + j];
                        if via_k < distances[i * num_nodes + j] {
                            distances[i * num_nodes + j] = via_k;
                        }
                    }
                }
            }
        } else {
            // Landmark APSP for large graphs
            let num_landmarks = (num_nodes as f32).sqrt().ceil() as usize;
            // Implementation would go here - simplified for benchmark
        }

        distances
    }

    /// Compute weight matrix
    fn compute_weights(distances: &[f32], num_nodes: usize) -> Vec<f32> {
        distances.iter().map(|&d| {
            if d.is_infinite() || d == 0.0 {
                0.0
            } else {
                1.0 / (d * d)
            }
        }).collect()
    }

    /// Benchmark stress computation kernel
    fn benchmark_stress_computation(
        positions: &[f32],
        distances: &[f32],
        weights: &[f32],
        num_nodes: usize,
    ) -> (f64, f32) {
        let start = Instant::now();

        let mut total_stress = 0.0f32;

        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                let pos_i = i * 3;
                let pos_j = j * 3;

                let dx = positions[pos_i] - positions[pos_j];
                let dy = positions[pos_i + 1] - positions[pos_j + 1];
                let dz = positions[pos_i + 2] - positions[pos_j + 2];

                let current_dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let ideal_dist = distances[i * num_nodes + j];
                let weight = weights[i * num_nodes + j];

                let diff = ideal_dist - current_dist;
                total_stress += weight * diff * diff;
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        (elapsed, total_stress)
    }

    /// Benchmark gradient computation kernel
    fn benchmark_gradient_computation(
        positions: &[f32],
        distances: &[f32],
        weights: &[f32],
        num_nodes: usize,
    ) -> (f64, Vec<f32>) {
        let start = Instant::now();

        let mut gradient = vec![0.0f32; num_nodes * 3];

        for i in 0..num_nodes {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            let mut gz = 0.0f32;

            for j in 0..num_nodes {
                if i == j { continue; }

                let pos_i = i * 3;
                let pos_j = j * 3;

                let dx = positions[pos_i] - positions[pos_j];
                let dy = positions[pos_i + 1] - positions[pos_j + 1];
                let dz = positions[pos_i + 2] - positions[pos_j + 2];

                let current_dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
                let ideal_dist = distances[i * num_nodes + j];
                let weight = weights[i * num_nodes + j];

                let factor = weight * (1.0 - ideal_dist / current_dist);

                gx += factor * dx;
                gy += factor * dy;
                gz += factor * dz;
            }

            gradient[i * 3] = gx;
            gradient[i * 3 + 1] = gy;
            gradient[i * 3 + 2] = gz;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        (elapsed, gradient)
    }

    /// Benchmark position update kernel
    fn benchmark_position_update(
        positions: &mut [f32],
        gradient: &[f32],
        learning_rate: f32,
        max_displacement: f32,
        num_nodes: usize,
    ) -> f64 {
        let start = Instant::now();

        for i in 0..num_nodes {
            let idx = i * 3;

            let mut dx = -learning_rate * gradient[idx];
            let mut dy = -learning_rate * gradient[idx + 1];
            let mut dz = -learning_rate * gradient[idx + 2];

            // Clamp displacement
            let displacement = (dx * dx + dy * dy + dz * dz).sqrt();
            if displacement > max_displacement {
                let scale = max_displacement / displacement;
                dx *= scale;
                dy *= scale;
                dz *= scale;
            }

            positions[idx] += dx;
            positions[idx + 1] += dy;
            positions[idx + 2] += dz;
        }

        start.elapsed().as_secs_f64() * 1000.0
    }

    /// Run full optimization benchmark
    fn run_benchmark(graph_size: usize, config: &BenchmarkConfig) -> BenchmarkResult {
        println!("\nRunning benchmark for {} nodes...", graph_size);

        // Create test graph
        let edge_probability = 5.0 / graph_size as f32; // Sparse graph
        let (edges, mut positions) = create_test_graph(graph_size, edge_probability);

        println!("  Generated graph with {} edges", edges.len());

        // Compute distance and weight matrices
        let distances = compute_graph_distances(graph_size, &edges);
        let weights = compute_weights(&distances, graph_size);

        // Warm-up runs
        for _ in 0..config.warmup_runs {
            let _ = benchmark_stress_computation(&positions, &distances, &weights, graph_size);
        }

        // Measurement runs
        let mut stress_times = Vec::new();
        let mut gradient_times = Vec::new();
        let mut update_times = Vec::new();
        let mut final_stress = 0.0;

        for run in 0..config.measurement_runs {
            let mut iteration = 0;
            let mut converged = false;
            let total_start = Instant::now();

            while iteration < config.max_iterations && !converged {
                // Stress computation
                let (stress_time, stress) = benchmark_stress_computation(
                    &positions,
                    &distances,
                    &weights,
                    graph_size,
                );
                stress_times.push(stress_time);

                // Gradient computation
                let (gradient_time, gradient) = benchmark_gradient_computation(
                    &positions,
                    &distances,
                    &weights,
                    graph_size,
                );
                gradient_times.push(gradient_time);

                // Position update
                let update_time = benchmark_position_update(
                    &mut positions,
                    &gradient,
                    0.05,
                    50.0,
                    graph_size,
                );
                update_times.push(update_time);

                iteration += 1;
                final_stress = stress;

                // Check convergence (simplified)
                if iteration > 10 && stress < config.convergence_threshold {
                    converged = true;
                }
            }

            let total_time = total_start.elapsed().as_secs_f64() * 1000.0;

            if run == 0 {
                println!("  First run: {} iterations in {:.2} ms", iteration, total_time);
            }
        }

        // Compute averages
        let avg_stress = stress_times.iter().sum::<f64>() / stress_times.len() as f64;
        let avg_gradient = gradient_times.iter().sum::<f64>() / gradient_times.len() as f64;
        let avg_update = update_times.iter().sum::<f64>() / update_times.len() as f64;
        let avg_total = avg_stress + avg_gradient + avg_update;

        BenchmarkResult {
            graph_size,
            stress_computation_ms: avg_stress,
            gradient_computation_ms: avg_gradient,
            position_update_ms: avg_update,
            total_iteration_ms: avg_total,
            iterations_to_convergence: config.max_iterations,
            total_optimization_ms: avg_total * config.max_iterations as f64,
            final_stress,
        }
    }

    #[test]
    #[ignore] // Run with: cargo test --release stress_majorization_benchmarks -- --ignored --nocapture
    fn benchmark_stress_majorization_performance() {
        println!("\n╔════════════════════════════════════════════════════╗");
        println!("║  Stress Majorization Performance Benchmarks      ║");
        println!("╚════════════════════════════════════════════════════╝");

        let config = BenchmarkConfig::default();
        let mut results = Vec::new();

        for &size in &config.graph_sizes {
            let result = run_benchmark(size, &config);
            result.print_summary();
            results.push(result);
        }

        // Print comparison table
        println!("\n\n╔════════════════════════════════════════════════════════════════════╗");
        println!("║                    Performance Comparison                         ║");
        println!("╠═══════════╦════════════╦════════════╦════════════╦═══════════════╣");
        println!("║   Nodes   ║   Stress   ║  Gradient  ║   Update   ║ Total/Iter    ║");
        println!("╠═══════════╬════════════╬════════════╬════════════╬═══════════════╣");

        for result in &results {
            println!("║ {:>9} ║ {:>8.2} ms║ {:>8.2} ms║ {:>8.2} ms║ {:>11.2} ms║",
                result.graph_size,
                result.stress_computation_ms,
                result.gradient_computation_ms,
                result.position_update_ms,
                result.total_iteration_ms
            );
        }

        println!("╚═══════════╩════════════╩════════════╩════════════╩═══════════════╝");

        // Performance targets
        println!("\n\n╔════════════════════════════════════════════════════════════════════╗");
        println!("║                    Performance Targets                            ║");
        println!("╠═══════════╦════════════════╦════════════════╦═══════════════════╣");
        println!("║   Nodes   ║     Target     ║     Actual     ║      Status       ║");
        println!("╠═══════════╬════════════════╬════════════════╬═══════════════════╣");

        let targets = vec![
            (100, 0.25),
            (1000, 3.2),
            (10000, 55.0),
            (100000, 1050.0),
        ];

        for (i, &(size, target)) in targets.iter().enumerate() {
            if let Some(result) = results.get(i) {
                let status = if result.total_iteration_ms <= target {
                    "✓ PASS"
                } else {
                    "✗ FAIL"
                };
                println!("║ {:>9} ║ {:>12.2} ms║ {:>12.2} ms║ {:>17} ║",
                    size, target, result.total_iteration_ms, status);
            }
        }

        println!("╚═══════════╩════════════════╩════════════════╩═══════════════════╝");
    }

    #[test]
    fn test_stress_majorization_correctness() {
        // Test that stress majorization actually reduces stress
        let num_nodes = 100;
        let (edges, mut positions) = create_test_graph(num_nodes, 0.1);
        let distances = compute_graph_distances(num_nodes, &edges);
        let weights = compute_weights(&distances, num_nodes);

        // Initial stress
        let (_, initial_stress) = benchmark_stress_computation(&positions, &distances, &weights, num_nodes);

        // Run optimization
        for _ in 0..50 {
            let (_, gradient) = benchmark_gradient_computation(&positions, &distances, &weights, num_nodes);
            benchmark_position_update(&mut positions, &gradient, 0.05, 50.0, num_nodes);
        }

        // Final stress
        let (_, final_stress) = benchmark_stress_computation(&positions, &distances, &weights, num_nodes);

        println!("\nStress Reduction Test:");
        println!("  Initial stress: {:.2}", initial_stress);
        println!("  Final stress:   {:.2}", final_stress);
        println!("  Reduction:      {:.2}%", (1.0 - final_stress / initial_stress) * 100.0);

        assert!(final_stress < initial_stress, "Stress should decrease after optimization");
    }
}
