//! Integration tests for stress majorization with safety controls
//!
//! This test suite validates the production-ready stress majorization implementation
//! with comprehensive safety controls, stability testing, and regression validation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use actix::prelude::*;
use cudarc::driver::CudaDevice;
use tokio::time::sleep;

use webxr::actors::gpu_compute_actor::*;
use webxr::actors::messages::*;
use webxr::models::constraints::*;
use webxr::models::graph::*;
use webxr::types::vec3::Vec3Data;
use webxr::utils::socket_flow_messages::BinaryNodeData;

/// Test configuration for stress majorization stability testing
#[derive(Debug, Clone)]
struct StabilityTestConfig {
    /// Number of test runs to perform
    pub runs: u32,
    /// Number of nodes in test graph
    pub num_nodes: u32,
    /// Number of iterations per run
    pub iterations: u32,
    /// Maximum allowed displacement per iteration
    pub max_displacement_threshold: f32,
    /// Maximum allowed stress value
    pub max_stress_threshold: f32,
    /// Required convergence rate (percentage of runs that must converge)
    pub min_convergence_rate: f32,
}

impl Default for StabilityTestConfig {
    fn default() -> Self {
        Self {
            runs: 5,
            num_nodes: 100,
            iterations: 100,
            max_displacement_threshold: 50.0,
            max_stress_threshold: 1e5,
            min_convergence_rate: 0.8, // 80% of runs must converge
        }
    }
}

/// Results from a single stress majorization test run
#[derive(Debug, Clone)]
struct TestRunResult {
    pub run_id: u32,
    pub converged: bool,
    pub final_stress: f32,
    pub max_displacement: f32,
    pub iterations: u32,
    pub computation_time_ms: u64,
    pub energy_conserved: bool,
    pub position_explosion: bool,
    pub numerical_stability: bool,
}

/// Aggregated results from multiple test runs
#[derive(Debug, Clone)]
struct StabilityTestResults {
    pub config: StabilityTestConfig,
    pub individual_runs: Vec<TestRunResult>,
    pub convergence_rate: f32,
    pub avg_stress: f32,
    pub avg_displacement: f32,
    pub avg_computation_time: u64,
    pub energy_conservation_rate: f32,
    pub stability_rate: f32,
    pub passed: bool,
}

/// Create a test graph with specified number of nodes in a predictable layout
fn create_test_graph(num_nodes: u32) -> GraphData {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Create nodes in a grid layout for predictable initial conditions
    let grid_size = (num_nodes as f32).sqrt().ceil() as u32;
    let spacing = 10.0;

    for i in 0..num_nodes {
        let x = (i % grid_size) as f32 * spacing;
        let y = (i / grid_size) as f32 * spacing;
        let z = 0.0;

        let node = Node {
            id: i,
            data: BinaryNodeData {
                position: Vec3Data { x, y, z },
                velocity: Vec3Data::zero(),
                force: Vec3Data::zero(),
                // Add some noise to prevent perfect symmetry
                mass: 1.0 + (i as f32 * 0.01) % 0.1,
            },
            metadata: Default::default(),
        };

        nodes.push(node);
    }

    // Create edges to form a connected graph
    for i in 0..num_nodes {
        // Connect to right neighbor
        if (i + 1) % grid_size != 0 {
            edges.push(Edge {
                id: edges.len() as u32,
                source: i,
                target: i + 1,
                weight: Some(1.0),
                metadata: Default::default(),
            });
        }

        // Connect to bottom neighbor
        if i + grid_size < num_nodes {
            edges.push(Edge {
                id: edges.len() as u32,
                source: i,
                target: i + grid_size,
                weight: Some(1.0),
                metadata: Default::default(),
            });
        }
    }

    GraphData {
        nodes,
        edges,
        metadata: Default::default(),
        id_to_metadata: Default::default(),
    }
}

/// Create test constraints for stability testing
fn create_test_constraints(graph: &GraphData) -> Vec<Constraint> {
    let mut constraints = Vec::new();

    // Add separation constraints between nearby nodes
    for i in 0..graph.nodes.len().min(10) {
        for j in (i + 1)..graph.nodes.len().min(10) {
            constraints.push(Constraint {
                kind: ConstraintKind::Distance,
                nodes: vec![i as u32, j as u32],
                target_value: 15.0, // Target separation
                weight: 0.5,
                metadata: Default::default(),
            });
        }
    }

    // Add position constraints to prevent drift
    constraints.push(Constraint {
        kind: ConstraintKind::Position,
        nodes: vec![0], // Anchor first node
        target_value: 0.0,
        weight: 1.0,
        metadata: Default::default(),
    });

    constraints
}

/// Calculate total energy of the system for conservation validation
fn calculate_system_energy(nodes: &[BinaryNodeData]) -> f32 {
    let mut kinetic_energy = 0.0;
    let mut potential_energy = 0.0;

    // Kinetic energy: 0.5 * m * v^2
    for node in nodes {
        let velocity_mag_sq = node.velocity.x * node.velocity.x
            + node.velocity.y * node.velocity.y
            + node.velocity.z * node.velocity.z;
        kinetic_energy += 0.5 * node.mass * velocity_mag_sq;
    }

    // Potential energy: sum of pairwise distances (simplified)
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let dx = nodes[i].position.x - nodes[j].position.x;
            let dy = nodes[i].position.y - nodes[j].position.y;
            let dz = nodes[i].position.z - nodes[j].position.z;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            potential_energy += 1.0 / (distance + 1.0); // Avoid division by zero
        }
    }

    kinetic_energy + potential_energy
}

/// Check if positions indicate an explosion (nodes moving too far from origin)
fn check_position_explosion(nodes: &[BinaryNodeData], threshold: f32) -> bool {
    for node in nodes {
        let distance_from_origin = (node.position.x * node.position.x
            + node.position.y * node.position.y
            + node.position.z * node.position.z)
            .sqrt();
        if distance_from_origin > threshold {
            return true;
        }
    }
    false
}

/// Check numerical stability (no NaN or infinite values)
fn check_numerical_stability(nodes: &[BinaryNodeData]) -> bool {
    for node in nodes {
        if !node.position.x.is_finite()
            || !node.position.y.is_finite()
            || !node.position.z.is_finite()
            || !node.velocity.x.is_finite()
            || !node.velocity.y.is_finite()
            || !node.velocity.z.is_finite()
        {
            return false;
        }
    }
    true
}

/// Run a single stress majorization test and return results
async fn run_single_test(
    run_id: u32,
    config: &StabilityTestConfig,
    gpu_actor: &Addr<GPUComputeActor>,
) -> Result<TestRunResult, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();

    // Create test graph and constraints
    let test_graph = create_test_graph(config.num_nodes);
    let test_constraints = create_test_constraints(&test_graph);

    // Initialize GPU with test data
    let graph_data_msg = UpdateGPUGraphData {
        graph_data: test_graph.clone(),
    };

    gpu_actor.send(graph_data_msg).await??;

    // Set test constraints
    let constraints_msg = UpdateConstraints {
        constraints: test_constraints,
    };

    gpu_actor.send(constraints_msg).await??;

    // Get initial state
    let initial_nodes: Vec<BinaryNodeData> = gpu_actor.send(GetNodeData).await??;
    let initial_energy = calculate_system_energy(&initial_nodes);

    // Trigger stress majorization
    let stress_msg = TriggerStressMajorization;
    gpu_actor.send(stress_msg).await??;

    // Wait a bit for processing
    sleep(Duration::from_millis(100)).await;

    // Get final state
    let final_nodes: Vec<BinaryNodeData> = gpu_actor.send(GetNodeData).await??;
    let final_energy = calculate_system_energy(&final_nodes);

    // Get stress majorization stats
    let stats = gpu_actor.send(GetStressMajorizationStats).await?;

    let computation_time = start_time.elapsed().as_millis() as u64;

    // Calculate maximum displacement
    let max_displacement = if initial_nodes.len() == final_nodes.len() {
        initial_nodes
            .iter()
            .zip(final_nodes.iter())
            .map(|(initial, final_node)| {
                let dx = final_node.position.x - initial.position.x;
                let dy = final_node.position.y - initial.position.y;
                let dz = final_node.position.z - initial.position.z;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(0.0, f32::max)
    } else {
        0.0
    };

    // Energy conservation check (within 10% tolerance)
    let energy_conserved = if initial_energy > 0.0 {
        let energy_change_ratio = (final_energy - initial_energy).abs() / initial_energy;
        energy_change_ratio < 0.1
    } else {
        true // If no initial energy, consider it conserved
    };

    Ok(TestRunResult {
        run_id,
        converged: stats.is_converging,
        final_stress: stats.avg_stress,
        max_displacement,
        iterations: stats.total_runs as u32,
        computation_time_ms: computation_time,
        energy_conserved,
        position_explosion: check_position_explosion(
            &final_nodes,
            config.max_displacement_threshold * 2.0,
        ),
        numerical_stability: check_numerical_stability(&final_nodes),
    })
}

/// Run comprehensive stability tests
async fn run_stability_tests(
    config: StabilityTestConfig,
) -> Result<StabilityTestResults, Box<dyn std::error::Error>> {
    // Initialize GPU actor
    let system = System::new();
    let gpu_actor = GPUComputeActor::new().start();

    // Initialize GPU
    let init_msg = InitializeGPU {
        max_nodes: config.num_nodes * 2, // Extra capacity for safety
        max_edges: config.num_nodes * 4,
    };

    gpu_actor.send(init_msg).await??;

    // Configure safety parameters
    let advanced_params = AdvancedParams {
        stress_step_interval_frames: 1, // Run immediately for testing
        max_velocity: config.max_displacement_threshold / 10.0,
        target_edge_length: 10.0,
        collision_threshold: 0.1,
        ..Default::default()
    };

    let params_msg = UpdateAdvancedParams {
        params: advanced_params,
    };
    gpu_actor.send(params_msg).await??;

    let mut results = Vec::new();

    // Run multiple tests
    for run_id in 0..config.runs {
        println!("Running stability test {}/{}", run_id + 1, config.runs);

        // Reset safety state before each run
        gpu_actor.send(ResetStressMajorizationSafety).await??;

        match run_single_test(run_id, &config, &gpu_actor).await {
            Ok(result) => {
                println!(
                    "  Run {} - Converged: {}, Stress: {:.2}, Displacement: {:.2}, Stable: {}",
                    result.run_id,
                    result.converged,
                    result.final_stress,
                    result.max_displacement,
                    result.numerical_stability
                );
                results.push(result);
            }
            Err(e) => {
                eprintln!("  Run {} failed: {}", run_id, e);
                // Create a failed result
                results.push(TestRunResult {
                    run_id,
                    converged: false,
                    final_stress: f32::INFINITY,
                    max_displacement: f32::INFINITY,
                    iterations: 0,
                    computation_time_ms: 0,
                    energy_conserved: false,
                    position_explosion: true,
                    numerical_stability: false,
                });
            }
        }
    }

    // Calculate aggregate statistics
    let converged_count = results.iter().filter(|r| r.converged).count() as f32;
    let convergence_rate = converged_count / config.runs as f32;

    let stable_count = results
        .iter()
        .filter(|r| r.numerical_stability && !r.position_explosion)
        .count() as f32;
    let stability_rate = stable_count / config.runs as f32;

    let energy_conserved_count = results.iter().filter(|r| r.energy_conserved).count() as f32;
    let energy_conservation_rate = energy_conserved_count / config.runs as f32;

    let valid_results: Vec<_> = results
        .iter()
        .filter(|r| r.final_stress.is_finite())
        .collect();

    let avg_stress = if !valid_results.is_empty() {
        valid_results.iter().map(|r| r.final_stress).sum::<f32>() / valid_results.len() as f32
    } else {
        0.0
    };

    let avg_displacement = if !valid_results.is_empty() {
        valid_results
            .iter()
            .map(|r| r.max_displacement)
            .sum::<f32>()
            / valid_results.len() as f32
    } else {
        0.0
    };

    let avg_computation_time = if !valid_results.is_empty() {
        valid_results
            .iter()
            .map(|r| r.computation_time_ms)
            .sum::<u64>()
            / valid_results.len() as u64
    } else {
        0
    };

    // Test passes if convergence rate and stability rate meet thresholds
    let passed = convergence_rate >= config.min_convergence_rate &&
                 stability_rate >= 0.9 && // 90% stability required
                 energy_conservation_rate >= 0.8; // 80% energy conservation required

    Ok(StabilityTestResults {
        config,
        individual_runs: results,
        convergence_rate,
        avg_stress,
        avg_displacement,
        avg_computation_time,
        energy_conservation_rate,
        stability_rate,
        passed,
    })
}

#[tokio::test]
#[ignore] // Only run with explicit GPU testing
async fn test_stress_majorization_stability_small_graph() -> Result<(), Box<dyn std::error::Error>>
{
    let config = StabilityTestConfig {
        runs: 5,
        num_nodes: 50,
        iterations: 50,
        max_displacement_threshold: 25.0,
        max_stress_threshold: 1e4,
        min_convergence_rate: 0.6, // Lower threshold for small test
    };

    let results = run_stability_tests(config).await?;

    println!("=== Stress Majorization Stability Test Results ===");
    println!("Convergence Rate: {:.1}%", results.convergence_rate * 100.0);
    println!("Stability Rate: {:.1}%", results.stability_rate * 100.0);
    println!(
        "Energy Conservation Rate: {:.1}%",
        results.energy_conservation_rate * 100.0
    );
    println!("Average Stress: {:.2}", results.avg_stress);
    println!("Average Displacement: {:.2}", results.avg_displacement);
    println!(
        "Average Computation Time: {}ms",
        results.avg_computation_time
    );
    println!(
        "Overall Result: {}",
        if results.passed { "PASS" } else { "FAIL" }
    );

    assert!(results.passed, "Stress majorization stability test failed");

    Ok(())
}

#[tokio::test]
#[ignore] // Only run with explicit GPU testing
async fn test_stress_majorization_stability_medium_graph() -> Result<(), Box<dyn std::error::Error>>
{
    let config = StabilityTestConfig {
        runs: 5,
        num_nodes: 200,
        iterations: 100,
        max_displacement_threshold: 50.0,
        max_stress_threshold: 5e4,
        min_convergence_rate: 0.8,
    };

    let results = run_stability_tests(config).await?;

    println!("=== Medium Graph Stability Test Results ===");
    println!("Convergence Rate: {:.1}%", results.convergence_rate * 100.0);
    println!("Stability Rate: {:.1}%", results.stability_rate * 100.0);
    println!(
        "Energy Conservation Rate: {:.1}%",
        results.energy_conservation_rate * 100.0
    );

    assert!(results.passed, "Medium graph stability test failed");

    Ok(())
}

#[tokio::test]
#[ignore] // Only run with explicit GPU testing
async fn test_stress_majorization_safety_controls() -> Result<(), Box<dyn std::error::Error>> {
    let system = System::new();
    let gpu_actor = GPUComputeActor::new().start();

    // Initialize with small graph
    let init_msg = InitializeGPU {
        max_nodes: 100,
        max_edges: 200,
    };

    gpu_actor.send(init_msg).await??;

    // Configure with very aggressive parameters that should trigger safety controls
    let dangerous_params = AdvancedParams {
        stress_step_interval_frames: 1,
        max_velocity: 1.0,          // Very small threshold
        target_edge_length: 1.0,    // Very small threshold
        collision_threshold: 0.001, // Very tight convergence
        ..Default::default()
    };

    gpu_actor
        .send(UpdateAdvancedParams {
            params: dangerous_params,
        })
        .await??;

    // Create a problematic graph (nodes very close together)
    let mut nodes = Vec::new();
    for i in 0..20 {
        nodes.push(Node {
            id: i,
            data: BinaryNodeData {
                position: Vec3Data {
                    x: (i as f32) * 0.1, // Very close spacing
                    y: 0.0,
                    z: 0.0,
                },
                velocity: Vec3Data::zero(),
                force: Vec3Data::zero(),
                mass: 1.0,
            },
            metadata: Default::default(),
        });
    }

    let problematic_graph = GraphData {
        nodes,
        edges: vec![],
        metadata: Default::default(),
        id_to_metadata: Default::default(),
    };

    // Add conflicting constraints
    let conflicting_constraints = vec![
        Constraint {
            kind: ConstraintKind::Distance,
            nodes: vec![0, 1],
            target_value: 100.0, // Force far apart
            weight: 2.0,
            metadata: Default::default(),
        },
        Constraint {
            kind: ConstraintKind::Distance,
            nodes: vec![0, 1],
            target_value: 0.1, // Force very close
            weight: 2.0,
            metadata: Default::default(),
        },
    ];

    gpu_actor
        .send(UpdateGPUGraphData {
            graph_data: problematic_graph,
        })
        .await??;
    gpu_actor
        .send(UpdateConstraints {
            constraints: conflicting_constraints,
        })
        .await??;

    // Trigger stress majorization - should be handled safely
    gpu_actor.send(TriggerStressMajorization).await??;

    // Check safety stats
    let stats = gpu_actor.send(GetStressMajorizationStats).await?;

    println!("=== Safety Controls Test Results ===");
    println!("Emergency Stopped: {}", stats.is_emergency_stopped);
    println!("Consecutive Failures: {}", stats.consecutive_failures);
    println!("Success Rate: {:.1}%", stats.success_rate);

    // Safety controls should have prevented catastrophic failure
    // Either the system should work or it should safely disable itself
    assert!(
        stats.success_rate > 0.0 || stats.is_emergency_stopped || stats.consecutive_failures >= 3,
        "Safety controls did not activate properly"
    );

    Ok(())
}

#[tokio::test]
#[ignore] // Only run with explicit GPU testing
async fn test_stress_majorization_parameter_updates() -> Result<(), Box<dyn std::error::Error>> {
    let system = System::new();
    let gpu_actor = GPUComputeActor::new().start();

    gpu_actor
        .send(InitializeGPU {
            max_nodes: 50,
            max_edges: 100,
        })
        .await??;

    // Test parameter updates
    let params1 = AdvancedParams {
        stress_step_interval_frames: 100,
        max_velocity: 10.0,
        target_edge_length: 20.0,
        collision_threshold: 0.1,
        ..Default::default()
    };

    gpu_actor
        .send(UpdateStressMajorizationParams {
            params: params1.clone(),
        })
        .await??;

    let stats1 = gpu_actor.send(GetStressMajorizationStats).await?;

    // Parameters should have been updated and safety state reset
    assert_eq!(stats1.consecutive_failures, 0);
    assert!(!stats1.is_emergency_stopped);

    // Test different parameters
    let params2 = AdvancedParams {
        stress_step_interval_frames: 50,
        max_velocity: 5.0,
        target_edge_length: 10.0,
        collision_threshold: 0.01,
        ..Default::default()
    };

    gpu_actor
        .send(UpdateStressMajorizationParams {
            params: params2.clone(),
        })
        .await??;

    let stats2 = gpu_actor.send(GetStressMajorizationStats).await?;

    // State should still be reset
    assert_eq!(stats2.consecutive_failures, 0);
    assert!(!stats2.is_emergency_stopped);

    Ok(())
}
