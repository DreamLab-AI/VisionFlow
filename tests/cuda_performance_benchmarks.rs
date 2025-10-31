// REAL CUDA Performance Benchmarks - NO MOCKS
//
// Benchmarks with ACTUAL performance targets from roadmap:
// - 30 FPS physics simulation (33ms per frame)
// - 10K+ nodes in real-time
// - Sub-millisecond kernel launches

#[cfg(all(feature = "gpu", not(debug_assertions)))]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
    use std::time::Duration;
    use rusqlite::Connection;

    use webxr::repositories::unified_graph_repository::UnifiedGraphRepository;
    use webxr::ports::knowledge_graph_repository::KnowledgeGraphRepository;
    use webxr::utils::unified_gpu_compute::{UnifiedGPUCompute, SimParams};
    use webxr::models::constraints::{Constraint, ConstraintKind};

    /// Load REAL PTX
    fn load_ptx() -> String {
        std::fs::read_to_string("target/visionflow_unified.ptx")
            .or_else(|_| std::fs::read_to_string("../target/visionflow_unified.ptx"))
            .or_else(|_| std::fs::read_to_string("./visionflow_unified.ptx"))
            .expect("PTX not found - compile CUDA kernels first")
    }

    /// Create REAL test database
    fn create_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        let schema = include_str!("../migration/unified_schema.sql");
        conn.execute_batch(schema).unwrap();
        conn
    }

    /// Insert REAL node data with Fibonacci sphere distribution
    fn insert_nodes(conn: &Connection, count: usize) {
        let tx = conn.unchecked_transaction().unwrap();

        for i in 0..count {
            let phi = std::f32::consts::PI * (1.0 + 5.0_f32.sqrt());
            let y = 1.0 - (i as f32 / (count - 1) as f32) * 2.0;
            let radius = (1.0 - y * y).sqrt();
            let theta = phi * i as f32;

            let x = radius * theta.cos() * 50.0;
            let z = radius * theta.sin() * 50.0;
            let y_scaled = y * 50.0;

            conn.execute(
                "INSERT INTO graph_nodes (metadata_id, label, x, y, z, vx, vy, vz, mass)
                 VALUES (?, ?, ?, ?, ?, 0.0, 0.0, 0.0, 1.0)",
                rusqlite::params![
                    format!("node_{}", i),
                    format!("Node {}", i),
                    x, y_scaled, z,
                ],
            ).unwrap();
        }

        tx.commit().unwrap();
    }

    /// Insert REAL edges based on k-nearest neighbors
    fn insert_edges(conn: &Connection, node_count: usize, k: usize) {
        let mut nodes: Vec<(i64, f32, f32, f32)> = Vec::new();

        {
            let mut stmt = conn.prepare("SELECT id, x, y, z FROM graph_nodes").unwrap();
            let rows = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            }).unwrap();

            for row in rows {
                nodes.push(row.unwrap());
            }
        }

        let tx = conn.unchecked_transaction().unwrap();

        for (i, (id_i, x_i, y_i, z_i)) in nodes.iter().enumerate() {
            let mut distances: Vec<(usize, f32)> = Vec::new();

            for (j, (_, x_j, y_j, z_j)) in nodes.iter().enumerate() {
                if i != j {
                    let dx = x_i - x_j;
                    let dy = y_i - y_j;
                    let dz = z_i - z_j;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    distances.push((j, dist));
                }
            }

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (j_idx, dist) in distances.iter().take(k) {
                let id_j = nodes[*j_idx].0;
                let weight = 1.0 / (1.0 + dist);

                conn.execute(
                    "INSERT INTO graph_edges (id, source, target, weight)
                     VALUES (?, ?, ?, ?)",
                    rusqlite::params![
                        format!("edge_{}_to_{}", id_i, id_j),
                        id_i,
                        id_j,
                        weight,
                    ],
                ).unwrap();
            }
        }

        tx.commit().unwrap();
    }

    fn benchmark_spatial_grid(c: &mut Criterion) {
        let mut group = c.benchmark_group("spatial_grid");
        group.measurement_time(Duration::from_secs(10));

        for &size in &[1000, 5000, 10000, 50000] {
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
                let conn = create_test_db();
                insert_nodes(&conn, size);

                let rt = tokio::runtime::Runtime::new().unwrap();
                let repo = UnifiedGraphRepository::new(":memory:").unwrap();
                let graph = rt.block_on(async { repo.load_graph().await.unwrap() });

                let ptx = load_ptx();
                let mut gpu = UnifiedGPUCompute::new(size as u32, 0, &ptx).unwrap();

                let pos_x: Vec<f32> = graph.nodes.iter().map(|n| n.x).collect();
                let pos_y: Vec<f32> = graph.nodes.iter().map(|n| n.y).collect();
                let pos_z: Vec<f32> = graph.nodes.iter().map(|n| n.z).collect();

                gpu.upload_positions(&pos_x, &pos_y, &pos_z).unwrap();

                let params = SimParams::default();

                b.iter(|| {
                    gpu.build_spatial_grid(black_box(&params)).unwrap();
                });
            });
        }

        group.finish();
    }

    fn benchmark_force_computation(c: &mut Criterion) {
        let mut group = c.benchmark_group("force_computation");
        group.measurement_time(Duration::from_secs(15));

        // REAL performance target: 30 FPS = 33ms max
        group.warm_up_time(Duration::from_secs(3));
        group.sample_size(50);

        for &size in &[1000, 5000, 10000] {
            group.bench_with_input(
                BenchmarkId::new("barnes_hut", size),
                &size,
                |b, &size| {
                    let conn = create_test_db();
                    insert_nodes(&conn, size);

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let repo = UnifiedGraphRepository::new(":memory:").unwrap();
                    let graph = rt.block_on(async { repo.load_graph().await.unwrap() });

                    let ptx = load_ptx();
                    let mut gpu = UnifiedGPUCompute::new(size as u32, 0, &ptx).unwrap();

                    let pos_x: Vec<f32> = graph.nodes.iter().map(|n| n.x).collect();
                    let pos_y: Vec<f32> = graph.nodes.iter().map(|n| n.y).collect();
                    let pos_z: Vec<f32> = graph.nodes.iter().map(|n| n.z).collect();

                    gpu.upload_positions(&pos_x, &pos_y, &pos_z).unwrap();

                    let mut params = SimParams::default();
                    params.feature_flags = 1; // Enable repulsion

                    gpu.build_spatial_grid(&params).unwrap();

                    b.iter(|| {
                        gpu.compute_forces(black_box(&params)).unwrap();
                    });
                },
            );
        }

        group.finish();
    }

    fn benchmark_full_physics_step(c: &mut Criterion) {
        let mut group = c.benchmark_group("full_physics_step");
        group.measurement_time(Duration::from_secs(20));

        // CRITICAL: This is the 30 FPS target benchmark
        for &size in &[1000, 5000, 10000] {
            group.bench_with_input(
                BenchmarkId::new("complete_iteration", size),
                &size,
                |b, &size| {
                    let conn = create_test_db();
                    insert_nodes(&conn, size);

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let repo = UnifiedGraphRepository::new(":memory:").unwrap();
                    let graph = rt.block_on(async { repo.load_graph().await.unwrap() });

                    let ptx = load_ptx();
                    let mut gpu = UnifiedGPUCompute::new(size as u32, 0, &ptx).unwrap();

                    let pos_x: Vec<f32> = graph.nodes.iter().map(|n| n.x).collect();
                    let pos_y: Vec<f32> = graph.nodes.iter().map(|n| n.y).collect();
                    let pos_z: Vec<f32> = graph.nodes.iter().map(|n| n.z).collect();

                    gpu.upload_positions(&pos_x, &pos_y, &pos_z).unwrap();

                    let mut params = SimParams::default();
                    params.feature_flags = 0b111; // All features

                    b.iter(|| {
                        gpu.execute_physics_step(black_box(&params)).unwrap();
                    });
                },
            );
        }

        group.finish();
    }

    fn benchmark_constraint_evaluation(c: &mut Criterion) {
        let mut group = c.benchmark_group("constraint_evaluation");
        group.measurement_time(Duration::from_secs(10));

        for &num_constraints in &[10, 100, 1000] {
            group.bench_with_input(
                BenchmarkId::from_parameter(num_constraints),
                &num_constraints,
                |b, &num_constraints| {
                    let size = 1000;
                    let conn = create_test_db();
                    insert_nodes(&conn, size);

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let repo = UnifiedGraphRepository::new(":memory:").unwrap();
                    let graph = rt.block_on(async { repo.load_graph().await.unwrap() });

                    let ptx = load_ptx();
                    let mut gpu = UnifiedGPUCompute::new(size as u32, 0, &ptx).unwrap();

                    let pos_x: Vec<f32> = graph.nodes.iter().map(|n| n.x).collect();
                    let pos_y: Vec<f32> = graph.nodes.iter().map(|n| n.y).collect();
                    let pos_z: Vec<f32> = graph.nodes.iter().map(|n| n.z).collect();

                    gpu.upload_positions(&pos_x, &pos_y, &pos_z).unwrap();

                    // Create REAL constraints
                    let mut constraints = Vec::new();
                    for i in 0..num_constraints {
                        let idx1 = i % size;
                        let idx2 = (i + 1) % size;

                        constraints.push(Constraint {
                            kind: ConstraintKind::Distance,
                            node_indices: vec![idx1, idx2],
                            target_value: Some(10.0),
                            weight: 1.0,
                            priority: 1,
                        });
                    }

                    gpu.upload_constraints(&constraints).unwrap();

                    let mut params = SimParams::default();
                    params.feature_flags = 0b10000; // Enable constraints

                    b.iter(|| {
                        gpu.compute_forces(black_box(&params)).unwrap();
                    });
                },
            );
        }

        group.finish();
    }

    fn benchmark_kmeans_clustering(c: &mut Criterion) {
        let mut group = c.benchmark_group("kmeans_clustering");
        group.measurement_time(Duration::from_secs(15));

        for &(size, k) in &[(1000, 5), (5000, 10), (10000, 20)] {
            group.bench_with_input(
                BenchmarkId::new("k_clusters", format!("{}_nodes_{}_k", size, k)),
                &(size, k),
                |b, &(size, k)| {
                    let conn = create_test_db();
                    insert_nodes(&conn, size);

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let repo = UnifiedGraphRepository::new(":memory:").unwrap();
                    let graph = rt.block_on(async { repo.load_graph().await.unwrap() });

                    let ptx = load_ptx();
                    let mut gpu = UnifiedGPUCompute::new(size as u32, 0, &ptx).unwrap();

                    let pos_x: Vec<f32> = graph.nodes.iter().map(|n| n.x).collect();
                    let pos_y: Vec<f32> = graph.nodes.iter().map(|n| n.y).collect();
                    let pos_z: Vec<f32> = graph.nodes.iter().map(|n| n.z).collect();

                    gpu.upload_positions(&pos_x, &pos_y, &pos_z).unwrap();

                    b.iter(|| {
                        gpu.kmeans_clustering(black_box(k), black_box(100)).unwrap();
                    });
                },
            );
        }

        group.finish();
    }

    fn benchmark_sssp(c: &mut Criterion) {
        let mut group = c.benchmark_group("sssp_pathfinding");
        group.measurement_time(Duration::from_secs(15));

        for &size in &[100, 500, 1000] {
            group.bench_with_input(
                BenchmarkId::from_parameter(size),
                &size,
                |b, &size| {
                    let conn = create_test_db();
                    insert_nodes(&conn, size);
                    insert_edges(&conn, size, 5); // k=5 nearest neighbors

                    let rt = tokio::runtime::Runtime::new().unwrap();
                    let repo = UnifiedGraphRepository::new(":memory:").unwrap();
                    let graph = rt.block_on(async { repo.load_graph().await.unwrap() });

                    let ptx = load_ptx();
                    let mut gpu = UnifiedGPUCompute::new(
                        size as u32,
                        graph.edges.len() as u32,
                        &ptx,
                    ).unwrap();

                    // Build CSR
                    let mut row_offsets = vec![0i32; size + 1];
                    let mut col_indices = Vec::new();
                    let mut weights = Vec::new();

                    let mut edges_by_source: std::collections::BTreeMap<i64, Vec<(i64, f32)>> =
                        std::collections::BTreeMap::new();

                    for edge in &graph.edges {
                        edges_by_source.entry(edge.source as i64)
                            .or_insert_with(Vec::new)
                            .push((edge.target as i64, edge.weight));
                    }

                    let mut current_offset = 0;
                    for i in 0..size {
                        row_offsets[i] = current_offset;

                        if let Some(edges) = edges_by_source.get(&(i as i64)) {
                            for (target, weight) in edges {
                                col_indices.push(*target as i32);
                                weights.push(*weight);
                                current_offset += 1;
                            }
                        }
                    }
                    row_offsets[size] = current_offset;

                    gpu.upload_csr_graph(&row_offsets, &col_indices, &weights).unwrap();

                    b.iter(|| {
                        gpu.compute_sssp(black_box(0)).unwrap();
                    });
                },
            );
        }

        group.finish();
    }

    criterion_group! {
        name = cuda_benchmarks;
        config = Criterion::default()
            .sample_size(30)
            .measurement_time(Duration::from_secs(10));
        targets =
            benchmark_spatial_grid,
            benchmark_force_computation,
            benchmark_full_physics_step,
            benchmark_constraint_evaluation,
            benchmark_kmeans_clustering,
            benchmark_sssp
    }

    criterion_main!(cuda_benchmarks);
}

#[cfg(not(all(feature = "gpu", not(debug_assertions))))]
fn main() {
    println!("Benchmarks require --features gpu and --release mode");
}
