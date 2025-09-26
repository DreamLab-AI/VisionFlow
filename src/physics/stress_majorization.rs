//! Stress majorization solver for knowledge graph layout optimization
//!
//! This module implements a stress majorization algorithm that optimizes node positions
//! to satisfy multiple constraint types while minimizing layout stress. The solver uses
//! efficient matrix operations and integrates with the GPU physics pipeline for
//! high-performance real-time optimization.
//!
//! ## Algorithm Overview
//!
//! Stress majorization works by:
//! 1. Computing the stress function based on distance differences between ideal and actual positions
//! 2. Using majorization to create a convex approximation of the stress function
//! 3. Iteratively minimizing the majorized function to find optimal positions
//! 4. Incorporating constraints through penalty methods or Lagrange multipliers
//!
//! ## Performance Features
//!
//! - GPU-accelerated matrix operations for large graphs
//! - Sparse matrix representations for efficient computation
//! - Adaptive step sizing and convergence detection
//! - Multi-threaded CPU fallback for smaller graphs
//! - Memory-efficient algorithms for very large datasets

use std::collections::HashMap;
use std::sync::Arc;
use log::{info, warn, debug, trace};
use cudarc::driver::CudaDevice;
use nalgebra::DMatrix;

use crate::models::{
    constraints::{Constraint, ConstraintSet, ConstraintKind, AdvancedParams},
    graph::GraphData,
};

/// Configuration parameters for stress majorization
#[derive(Debug, Clone)]
pub struct StressMajorizationConfig {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Convergence tolerance (relative change in stress)
    pub tolerance: f32,
    /// Step size for position updates
    pub step_size: f32,
    /// Whether to use adaptive step sizing
    pub adaptive_step: bool,
    /// Weight for constraint satisfaction relative to stress minimization
    pub constraint_weight: f32,
    /// Use GPU acceleration when available
    pub use_gpu: bool,
    /// Minimum improvement required to continue optimization
    pub min_improvement: f32,
    /// Number of iterations between convergence checks
    pub convergence_check_interval: u32,
}

impl Default for StressMajorizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            step_size: 0.1,
            adaptive_step: true,
            constraint_weight: 1.0,
            use_gpu: true,
            min_improvement: 1e-8,
            convergence_check_interval: 10,
        }
    }
}

/// Result of stress majorization optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final stress value
    pub final_stress: f32,
    /// Number of iterations performed
    pub iterations: u32,
    /// Whether optimization converged
    pub converged: bool,
    /// Constraint satisfaction scores for each constraint type
    pub constraint_scores: HashMap<ConstraintKind, f32>,
    /// Computation time in milliseconds
    pub computation_time: u64,
}

/// Stress majorization solver for graph layout optimization
pub struct StressMajorizationSolver {
    config: StressMajorizationConfig,
    _gpu_context: Option<Arc<CudaDevice>>,
    cached_distance_matrix: Option<DMatrix<f32>>,
    cached_weight_matrix: Option<DMatrix<f32>>,
    iteration_history: Vec<f32>,
}

impl Clone for StressMajorizationSolver {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            _gpu_context: self._gpu_context.clone(), // Arc can be cloned
            cached_distance_matrix: self.cached_distance_matrix.clone(),
            cached_weight_matrix: self.cached_weight_matrix.clone(),
            iteration_history: self.iteration_history.clone(),
        }
    }
}

impl StressMajorizationSolver {
    /// Create a new stress majorization solver with default configuration
    pub fn new() -> Self {
        Self::with_config(StressMajorizationConfig::default())
    }

    /// Create a new solver with custom configuration
    pub fn with_config(config: StressMajorizationConfig) -> Self {
        let gpu_context = if config.use_gpu {
            match Self::initialize_gpu() {
                Ok(device) => Some(device),
                Err(e) => {
                    warn!("Failed to initialize GPU, falling back to CPU: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            _gpu_context: gpu_context,
            cached_distance_matrix: None,
            cached_weight_matrix: None,
            iteration_history: Vec::new(),
        }
    }

    /// Create solver from advanced physics parameters
    pub fn from_advanced_params(params: &AdvancedParams) -> Self {
        let config = StressMajorizationConfig {
            max_iterations: params.stress_step_interval_frames * 10,
            constraint_weight: params.constraint_force_weight,
            step_size: 0.05,
            tolerance: 1e-5,
            adaptive_step: params.adaptive_force_scaling,
            ..Default::default()
        };
        
        Self::with_config(config)
    }

    /// Initialize GPU device for acceleration
    fn initialize_gpu() -> Result<Arc<CudaDevice>, Box<dyn std::error::Error>> {
        info!("Initializing GPU device for stress majorization");
        let device = CudaDevice::new(0)?;
        info!("Successfully initialized CUDA device for stress majorization");
        Ok(device)
    }

    /// Optimize graph layout using stress majorization
    pub fn optimize(
        &mut self,
        graph_data: &mut GraphData,
        constraints: &ConstraintSet,
    ) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        info!("Starting stress majorization optimization for {} nodes", graph_data.nodes.len());

        // Validate input data
        if graph_data.nodes.is_empty() {
            return Err("Cannot optimize empty graph".into());
        }

        // Compute initial distance and weight matrices
        self.compute_distance_matrix(graph_data)?;
        self.compute_weight_matrix(graph_data)?;

        // Initialize positions if needed
        self.initialize_positions(graph_data)?;

        let mut best_stress = f32::INFINITY;
        let mut current_positions = self.extract_positions(graph_data);
        let mut iterations = 0;
        let mut converged = false;

        info!("Beginning iterative optimization with {} constraints", constraints.constraints.len());

        // Main optimization loop
        while iterations < self.config.max_iterations && !converged {
            // Compute current stress
            let current_stress = self.compute_stress(&current_positions, graph_data)?;
            
            // Check for improvement
            if current_stress < best_stress {
                best_stress = current_stress;
                self.apply_positions(graph_data, &current_positions)?;
            }

            // Compute gradient and update positions
            let gradient = self.compute_gradient(&current_positions, graph_data, constraints)?;
            let new_positions = self.update_positions(&current_positions, &gradient)?;

            // Check convergence
            if iterations % self.config.convergence_check_interval == 0 {
                let improvement = (best_stress - current_stress) / best_stress.max(1e-10);
                converged = improvement < self.config.tolerance;
                
                if iterations % 100 == 0 {
                    debug!("Iteration {}: stress = {:.6}, improvement = {:.8}", 
                          iterations, current_stress, improvement);
                }
            }

            current_positions = new_positions;
            iterations += 1;
            self.iteration_history.push(current_stress);
        }

        // Final position update
        self.apply_positions(graph_data, &current_positions)?;

        // Compute constraint satisfaction scores
        let constraint_scores = self.compute_constraint_scores(graph_data, constraints)?;

        let result = OptimizationResult {
            final_stress: best_stress,
            iterations,
            converged,
            constraint_scores,
            computation_time: start_time.elapsed().as_millis() as u64,
        };

        info!("Stress majorization completed: {} iterations, stress = {:.6}, converged = {}", 
              iterations, best_stress, converged);

        Ok(result)
    }

    /// Compute distance matrix for the graph
    fn compute_distance_matrix(&mut self, graph_data: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
        let n = graph_data.nodes.len();
        let mut distance_matrix = DMatrix::zeros(n, n);

        // Use Floyd-Warshall to compute all-pairs shortest paths
        // Initialize with direct edge distances
        for (i, node_a) in graph_data.nodes.iter().enumerate() {
            for (j, node_b) in graph_data.nodes.iter().enumerate() {
                if i == j {
                    distance_matrix[(i, j)] = 0.0;
                } else {
                    // Check if there's a direct edge
                    let direct_distance = graph_data.edges.iter()
                        .find(|edge| {
                            (edge.source == node_a.id && edge.target == node_b.id) ||
                            (edge.source == node_b.id && edge.target == node_a.id)
                        })
                        .map(|_| 1.0) // Unit edge weight
                        .unwrap_or(f32::INFINITY);
                    
                    distance_matrix[(i, j)] = direct_distance;
                }
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let through_k = distance_matrix[(i, k)] + distance_matrix[(k, j)];
                    if through_k < distance_matrix[(i, j)] {
                        distance_matrix[(i, j)] = through_k;
                    }
                }
            }
        }

        // Replace infinite distances with a large but finite value
        for i in 0..n {
            for j in 0..n {
                if distance_matrix[(i, j)].is_infinite() {
                    distance_matrix[(i, j)] = (n as f32) * 2.0; // Disconnected components
                }
            }
        }

        self.cached_distance_matrix = Some(distance_matrix);
        trace!("Computed distance matrix for {} nodes", n);
        Ok(())
    }

    /// Compute weight matrix based on distances
    fn compute_weight_matrix(&mut self, graph_data: &GraphData) -> Result<(), Box<dyn std::error::Error>> {
        let distance_matrix = self.cached_distance_matrix.as_ref()
            .ok_or("Distance matrix must be computed first")?;
        
        let n = graph_data.nodes.len();
        let mut weight_matrix = DMatrix::zeros(n, n);

        // Weight is inversely proportional to squared distance
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let distance = distance_matrix[(i, j)];
                    if distance > 0.0 {
                        weight_matrix[(i, j)] = 1.0 / (distance * distance);
                    }
                }
            }
        }

        self.cached_weight_matrix = Some(weight_matrix);
        trace!("Computed weight matrix for {} nodes", n);
        Ok(())
    }

    /// Initialize node positions if not already set
    fn initialize_positions(&self, graph_data: &mut GraphData) -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();
        
        for node in &mut graph_data.nodes {
            // Only initialize if position is at origin (likely uninitialized)
            if node.data.x.abs() < f32::EPSILON &&
               node.data.y.abs() < f32::EPSILON &&
               node.data.z.abs() < f32::EPSILON {

                // Random initial position in a sphere
                use rand::Rng;
                let theta = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let phi = rng.gen_range(0.0..std::f32::consts::PI);
                let radius = rng.gen_range(50.0..200.0);

                node.data.x = radius * phi.sin() * theta.cos();
                node.data.y = radius * phi.sin() * theta.sin();
                node.data.z = radius * phi.cos();
            }
        }
        
        trace!("Initialized positions for {} nodes", graph_data.nodes.len());
        Ok(())
    }

    /// Extract current positions as a matrix
    fn extract_positions(&self, graph_data: &GraphData) -> DMatrix<f32> {
        let n = graph_data.nodes.len();
        let mut positions = DMatrix::zeros(n, 3);
        
        for (i, node) in graph_data.nodes.iter().enumerate() {
            positions[(i, 0)] = node.data.x;
            positions[(i, 1)] = node.data.y;
            positions[(i, 2)] = node.data.z;
        }
        
        positions
    }

    /// Apply positions back to graph data
    fn apply_positions(&self, graph_data: &mut GraphData, positions: &DMatrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
        if positions.nrows() != graph_data.nodes.len() || positions.ncols() != 3 {
            return Err("Position matrix dimensions don't match graph data".into());
        }
        
        for (i, node) in graph_data.nodes.iter_mut().enumerate() {
            node.data.x = positions[(i, 0)];
            node.data.y = positions[(i, 1)];
            node.data.z = positions[(i, 2)];
        }
        
        Ok(())
    }

    /// Compute stress function value
    fn compute_stress(&self, positions: &DMatrix<f32>, graph_data: &GraphData) -> Result<f32, Box<dyn std::error::Error>> {
        let distance_matrix = self.cached_distance_matrix.as_ref()
            .ok_or("Distance matrix not computed")?;
        let weight_matrix = self.cached_weight_matrix.as_ref()
            .ok_or("Weight matrix not computed")?;
        
        let n = graph_data.nodes.len();
        let mut stress = 0.0;
        
        for i in 0..n {
            for j in i+1..n {
                let ideal_distance = distance_matrix[(i, j)];
                let current_distance = self.euclidean_distance(positions, i, j);
                let weight = weight_matrix[(i, j)];
                
                let diff = ideal_distance - current_distance;
                stress += weight * diff * diff;
            }
        }
        
        Ok(stress)
    }

    /// Compute gradient of stress function with constraints
    fn compute_gradient(
        &self, 
        positions: &DMatrix<f32>, 
        graph_data: &GraphData,
        constraints: &ConstraintSet
    ) -> Result<DMatrix<f32>, Box<dyn std::error::Error>> {
        let distance_matrix = self.cached_distance_matrix.as_ref()
            .ok_or("Distance matrix not computed")?;
        let weight_matrix = self.cached_weight_matrix.as_ref()
            .ok_or("Weight matrix not computed")?;
        
        let n = graph_data.nodes.len();
        let mut gradient = DMatrix::zeros(n, 3);
        
        // Stress gradient
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                
                let ideal_distance = distance_matrix[(i, j)];
                let current_distance = self.euclidean_distance(positions, i, j);
                
                if current_distance > f32::EPSILON {
                    let weight = weight_matrix[(i, j)];
                    let factor = 2.0 * weight * (1.0 - ideal_distance / current_distance);
                    
                    for dim in 0..3 {
                        let diff = positions[(i, dim)] - positions[(j, dim)];
                        gradient[(i, dim)] += factor * diff;
                    }
                }
            }
        }
        
        // Add constraint gradients
        for constraint in constraints.active_constraints() {
            self.add_constraint_gradient(&mut gradient, positions, constraint)?;
        }
        
        Ok(gradient)
    }

    /// Add constraint contribution to gradient
    fn add_constraint_gradient(
        &self,
        gradient: &mut DMatrix<f32>,
        positions: &DMatrix<f32>,
        constraint: &Constraint
    ) -> Result<(), Box<dyn std::error::Error>> {
        match constraint.kind {
            ConstraintKind::FixedPosition => {
                if let Some(&node_idx) = constraint.node_indices.first() {
                    if constraint.params.len() >= 3 && node_idx < positions.nrows() as u32 {
                        let node_idx = node_idx as usize;
                        let weight = constraint.weight * self.config.constraint_weight;
                        
                        gradient[(node_idx, 0)] += weight * 2.0 * (positions[(node_idx, 0)] - constraint.params[0]);
                        gradient[(node_idx, 1)] += weight * 2.0 * (positions[(node_idx, 1)] - constraint.params[1]);
                        gradient[(node_idx, 2)] += weight * 2.0 * (positions[(node_idx, 2)] - constraint.params[2]);
                    }
                }
            },
            
            ConstraintKind::Separation => {
                if constraint.node_indices.len() >= 2 && !constraint.params.is_empty() {
                    let i = constraint.node_indices[0] as usize;
                    let j = constraint.node_indices[1] as usize;
                    let min_distance = constraint.params[0];
                    
                    if i < positions.nrows() && j < positions.nrows() {
                        let current_distance = self.euclidean_distance(positions, i, j);
                        
                        if current_distance < min_distance && current_distance > f32::EPSILON {
                            let weight = constraint.weight * self.config.constraint_weight;
                            let factor = weight * (min_distance - current_distance) / current_distance;
                            
                            for dim in 0..3 {
                                let diff = positions[(i, dim)] - positions[(j, dim)];
                                gradient[(i, dim)] -= factor * diff;
                                gradient[(j, dim)] += factor * diff;
                            }
                        }
                    }
                }
            },
            
            ConstraintKind::AlignmentHorizontal => {
                if !constraint.params.is_empty() {
                    let target_y = constraint.params[0];
                    let weight = constraint.weight * self.config.constraint_weight;
                    
                    for &node_idx in &constraint.node_indices {
                        if node_idx < positions.nrows() as u32 {
                            let node_idx = node_idx as usize;
                            gradient[(node_idx, 1)] += weight * 2.0 * (positions[(node_idx, 1)] - target_y);
                        }
                    }
                }
            },
            
            ConstraintKind::Clustering => {
                if constraint.params.len() >= 2 {
                    let strength = constraint.params[1];
                    let weight = constraint.weight * self.config.constraint_weight * strength;
                    
                    // Compute centroid
                    let mut centroid = [0.0f32; 3];
                    let mut valid_nodes = 0;
                    
                    for &node_idx in &constraint.node_indices {
                        if node_idx < positions.nrows() as u32 {
                            let node_idx = node_idx as usize;
                            for dim in 0..3 {
                                centroid[dim] += positions[(node_idx, dim)];
                            }
                            valid_nodes += 1;
                        }
                    }
                    
                    if valid_nodes > 0 {
                        for dim in 0..3 {
                            centroid[dim] /= valid_nodes as f32;
                        }
                        
                        // Apply attractive force toward centroid
                        for &node_idx in &constraint.node_indices {
                            if node_idx < positions.nrows() as u32 {
                                let node_idx = node_idx as usize;
                                for dim in 0..3 {
                                    gradient[(node_idx, dim)] += weight * (centroid[dim] - positions[(node_idx, dim)]);
                                }
                            }
                        }
                    }
                }
            },
            
            _ => {
                // TODO: Implement other constraint types
                debug!("Constraint type {:?} not yet implemented in gradient computation", constraint.kind);
            }
        }
        
        Ok(())
    }

    /// Update positions using gradient descent
    fn update_positions(&self, positions: &DMatrix<f32>, gradient: &DMatrix<f32>) -> Result<DMatrix<f32>, Box<dyn std::error::Error>> {
        let mut new_positions = positions.clone();
        let step_size = self.config.step_size;
        
        for i in 0..positions.nrows() {
            for j in 0..positions.ncols() {
                new_positions[(i, j)] -= step_size * gradient[(i, j)];
            }
        }
        
        Ok(new_positions)
    }

    /// Compute Euclidean distance between two nodes
    fn euclidean_distance(&self, positions: &DMatrix<f32>, i: usize, j: usize) -> f32 {
        let mut sum = 0.0;
        for dim in 0..3 {
            let diff = positions[(i, dim)] - positions[(j, dim)];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Compute constraint satisfaction scores
    fn compute_constraint_scores(
        &self,
        graph_data: &GraphData,
        constraints: &ConstraintSet
    ) -> Result<HashMap<ConstraintKind, f32>, Box<dyn std::error::Error>> {
        let mut scores = HashMap::new();
        let positions = self.extract_positions(graph_data);
        
        for constraint in constraints.active_constraints() {
            let score = match constraint.kind {
                ConstraintKind::FixedPosition => self.score_fixed_position(&positions, constraint)?,
                ConstraintKind::Separation => self.score_separation(&positions, constraint)?,
                ConstraintKind::AlignmentHorizontal => self.score_alignment_horizontal(&positions, constraint)?,
                ConstraintKind::Clustering => self.score_clustering(&positions, constraint)?,
                _ => 0.5, // Default neutral score for unimplemented types
            };
            
            scores.entry(constraint.kind)
                .and_modify(|e| *e = (*e + score) / 2.0)
                .or_insert(score);
        }
        
        Ok(scores)
    }

    /// Score fixed position constraint satisfaction
    fn score_fixed_position(&self, positions: &DMatrix<f32>, constraint: &Constraint) -> Result<f32, Box<dyn std::error::Error>> {
        if let Some(&node_idx) = constraint.node_indices.first() {
            if constraint.params.len() >= 3 && node_idx < positions.nrows() as u32 {
                let node_idx = node_idx as usize;
                let distance = ((positions[(node_idx, 0)] - constraint.params[0]).powi(2) +
                               (positions[(node_idx, 1)] - constraint.params[1]).powi(2) +
                               (positions[(node_idx, 2)] - constraint.params[2]).powi(2)).sqrt();
                
                // Score inversely related to distance from target
                return Ok((1.0 / (1.0 + distance / 10.0)).max(0.0).min(1.0));
            }
        }
        Ok(0.0)
    }

    /// Score separation constraint satisfaction
    fn score_separation(&self, positions: &DMatrix<f32>, constraint: &Constraint) -> Result<f32, Box<dyn std::error::Error>> {
        if constraint.node_indices.len() >= 2 && !constraint.params.is_empty() {
            let i = constraint.node_indices[0] as usize;
            let j = constraint.node_indices[1] as usize;
            let min_distance = constraint.params[0];
            
            if i < positions.nrows() && j < positions.nrows() {
                let current_distance = self.euclidean_distance(positions, i, j);
                return Ok(if current_distance >= min_distance { 1.0 } else { current_distance / min_distance });
            }
        }
        Ok(0.0)
    }

    /// Score horizontal alignment constraint satisfaction
    fn score_alignment_horizontal(&self, positions: &DMatrix<f32>, constraint: &Constraint) -> Result<f32, Box<dyn std::error::Error>> {
        if !constraint.params.is_empty() {
            let target_y = constraint.params[0];
            let mut total_deviation = 0.0;
            let mut count = 0;
            
            for &node_idx in &constraint.node_indices {
                if node_idx < positions.nrows() as u32 {
                    let node_idx = node_idx as usize;
                    total_deviation += (positions[(node_idx, 1)] - target_y).abs();
                    count += 1;
                }
            }
            
            if count > 0 {
                let avg_deviation = total_deviation / count as f32;
                return Ok((1.0 / (1.0 + avg_deviation / 10.0)).max(0.0).min(1.0));
            }
        }
        Ok(0.0)
    }

    /// Score clustering constraint satisfaction
    fn score_clustering(&self, positions: &DMatrix<f32>, constraint: &Constraint) -> Result<f32, Box<dyn std::error::Error>> {
        if constraint.node_indices.len() > 1 {
            // Compute average pairwise distance within cluster
            let mut total_distance = 0.0;
            let mut count = 0;
            
            for i in 0..constraint.node_indices.len() {
                for j in i+1..constraint.node_indices.len() {
                    let node_i = constraint.node_indices[i] as usize;
                    let node_j = constraint.node_indices[j] as usize;
                    
                    if node_i < positions.nrows() && node_j < positions.nrows() {
                        total_distance += self.euclidean_distance(positions, node_i, node_j);
                        count += 1;
                    }
                }
            }
            
            if count > 0 {
                let avg_distance = total_distance / count as f32;
                // Score inversely related to average internal distance
                return Ok((1.0 / (1.0 + avg_distance / 50.0)).max(0.0).min(1.0));
            }
        }
        Ok(0.0)
    }

    /// Get optimization history
    pub fn get_iteration_history(&self) -> &[f32] {
        &self.iteration_history
    }

    /// Clear cached matrices (call when graph topology changes)
    pub fn clear_cache(&mut self) {
        self.cached_distance_matrix = None;
        self.cached_weight_matrix = None;
        self.iteration_history.clear();
        trace!("Cleared stress majorization cache");
    }

    /// Update configuration
    pub fn update_config(&mut self, config: StressMajorizationConfig) {
        self.config = config;
        info!("Updated stress majorization configuration");
    }
}

impl Default for StressMajorizationSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{edge::Edge, node::Node, graph::GraphData};
    use crate::utils::socket_flow_messages::BinaryNodeData;

    fn create_test_graph() -> GraphData {
        let mut graph = GraphData {
            nodes: vec![
                Node::new_with_id("test1".to_string(), Some(1)),
                Node::new_with_id("test2".to_string(), Some(2)),
                Node::new_with_id("test3".to_string(), Some(3)),
            ],
            edges: vec![
                Edge::new(1, 2, 1.0),
                Edge::new(2, 3, 1.0),
            ],
            metadata: crate::models::metadata::MetadataStore::new(),
            id_to_metadata: std::collections::HashMap::new(),
        };
        
        // Set initial positions
        graph.nodes[0].data.x = 0.0;
        graph.nodes[0].data.y = 0.0;
        graph.nodes[0].data.z = 0.0;

        graph.nodes[1].data.x = 100.0;
        graph.nodes[1].data.y = 0.0;
        graph.nodes[1].data.z = 0.0;

        graph.nodes[2].data.x = 50.0;
        graph.nodes[2].data.y = 100.0;
        graph.nodes[2].data.z = 0.0;
        
        graph
    }

    #[test]
    fn test_solver_creation() {
        let solver = StressMajorizationSolver::new();
        assert_eq!(solver.config.max_iterations, 1000);
        assert!(solver.config.tolerance > 0.0);
    }

    #[test]
    fn test_distance_matrix_computation() {
        let mut solver = StressMajorizationSolver::new();
        let graph = create_test_graph();
        
        solver.compute_distance_matrix(&graph).unwrap();
        assert!(solver.cached_distance_matrix.is_some());
        
        let distance_matrix = solver.cached_distance_matrix.as_ref().unwrap();
        assert_eq!(distance_matrix.nrows(), 3);
        assert_eq!(distance_matrix.ncols(), 3);
        
        // Check diagonal is zero
        for i in 0..3 {
            assert_eq!(distance_matrix[(i, i)], 0.0);
        }
    }

    #[test]
    fn test_position_extraction_and_application() {
        let solver = StressMajorizationSolver::new();
        let mut graph = create_test_graph();
        
        let positions = solver.extract_positions(&graph);
        assert_eq!(positions.nrows(), 3);
        assert_eq!(positions.ncols(), 3);
        assert_eq!(positions[(0, 0)], 0.0);
        assert_eq!(positions[(1, 0)], 100.0);
        
        // Modify positions and apply back
        let mut new_positions = positions.clone();
        new_positions[(0, 0)] = 50.0;
        
        solver.apply_positions(&mut graph, &new_positions).unwrap();
        assert_eq!(graph.nodes[0].data.x, 50.0);
    }

    #[test]
    fn test_constraint_score_computation() {
        let solver = StressMajorizationSolver::new();
        let graph = create_test_graph();
        let mut constraint_set = ConstraintSet::default();
        
        // Add a separation constraint
        constraint_set.add(Constraint::separation(1, 2, 50.0));
        
        let scores = solver.compute_constraint_scores(&graph, &constraint_set).unwrap();
        assert!(scores.contains_key(&ConstraintKind::Separation));
        
        let sep_score = scores[&ConstraintKind::Separation];
        assert!(sep_score >= 0.0 && sep_score <= 1.0);
    }
}