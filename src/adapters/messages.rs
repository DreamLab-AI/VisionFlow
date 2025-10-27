// src/adapters/messages.rs
//! Message Translation Layer for Actor-Port Adapters
//!
//! This module provides bidirectional conversion between:
//! - Port domain types (from hexagonal architecture)
//! - Actor message types (Actix message passing)

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::constraints::ConstraintSet;
use crate::models::graph::GraphData;
use crate::ports::gpu_physics_adapter::{
    GpuDeviceInfo, NodeForce, PhysicsParameters, PhysicsStatistics, PhysicsStepResult,
};
use crate::ports::gpu_semantic_analyzer::{
    ClusteringAlgorithm, CommunityDetectionResult, ImportanceAlgorithm, OptimizationResult,
    PathfindingResult, SemanticConstraintConfig, SemanticStatistics,
};

// ============================================================================
// Physics Adapter Messages
// ============================================================================

/// Initialize physics simulation with graph and parameters
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitializePhysicsMessage {
    pub graph: Arc<GraphData>,
    pub params: PhysicsParameters,
}

/// Compute forces for all nodes
#[derive(Message)]
#[rtype(result = "Result<Vec<NodeForce>, String>")]
pub struct ComputeForcesMessage;

/// Update node positions based on computed forces
#[derive(Message)]
#[rtype(result = "Result<Vec<(u32, f32, f32, f32)>, String>")]
pub struct UpdatePositionsMessage {
    pub forces: Vec<NodeForce>,
}

/// Perform complete physics simulation step
#[derive(Message)]
#[rtype(result = "Result<PhysicsStepResult, String>")]
pub struct PhysicsStepMessage;

/// Run simulation until convergence
#[derive(Message)]
#[rtype(result = "Result<PhysicsStepResult, String>")]
pub struct SimulateUntilConvergenceMessage;

/// Apply external forces to specific nodes
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ApplyExternalForcesMessage {
    pub forces: Vec<(u32, f32, f32, f32)>,
}

/// Pin nodes at specific positions
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct PinNodesMessage {
    pub nodes: Vec<(u32, f32, f32, f32)>,
}

/// Unpin nodes to allow free movement
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UnpinNodesMessage {
    pub node_ids: Vec<u32>,
}

/// Update physics parameters
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdatePhysicsParametersMessage {
    pub params: PhysicsParameters,
}

/// Update graph data for physics
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdatePhysicsGraphDataMessage {
    pub graph: Arc<GraphData>,
}

/// Get GPU device status
#[derive(Message)]
#[rtype(result = "Result<GpuDeviceInfo, String>")]
pub struct GetGpuStatusMessage;

/// Get physics statistics
#[derive(Message)]
#[rtype(result = "Result<PhysicsStatistics, String>")]
pub struct GetPhysicsStatisticsMessage;

/// Reset physics simulation state
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ResetPhysicsMessage;

/// Cleanup physics GPU resources
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct CleanupPhysicsMessage;

// ============================================================================
// Semantic Analyzer Messages
// ============================================================================

/// Initialize semantic analyzer with graph
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitializeSemanticMessage {
    pub graph: Arc<GraphData>,
}

/// Detect communities using specified algorithm
#[derive(Message)]
#[rtype(result = "Result<CommunityDetectionResult, String>")]
pub struct DetectCommunitiesMessage {
    pub algorithm: ClusteringAlgorithm,
}

/// Compute shortest paths from source node
#[derive(Message)]
#[rtype(result = "Result<PathfindingResult, String>")]
pub struct ComputeShortestPathsMessage {
    pub source_node_id: u32,
}

/// Compute SSSP distances only (no path reconstruction)
#[derive(Message)]
#[rtype(result = "Result<Vec<f32>, String>")]
pub struct ComputeSsspDistancesMessage {
    pub source_node_id: u32,
}

/// Compute all-pairs shortest paths
#[derive(Message)]
#[rtype(result = "Result<HashMap<(u32, u32), Vec<u32>>, String>")]
pub struct ComputeAllPairsShortestPathsMessage;

/// Compute landmark-based approximate APSP
#[derive(Message)]
#[rtype(result = "Result<Vec<Vec<f32>>, String>")]
pub struct ComputeLandmarkApspMessage {
    pub num_landmarks: usize,
}

/// Generate semantic constraints
#[derive(Message)]
#[rtype(result = "Result<ConstraintSet, String>")]
pub struct GenerateSemanticConstraintsMessage {
    pub config: SemanticConstraintConfig,
}

/// Optimize layout using stress majorization
#[derive(Message)]
#[rtype(result = "Result<OptimizationResult, String>")]
pub struct OptimizeLayoutMessage {
    pub constraints: ConstraintSet,
    pub max_iterations: usize,
}

/// Analyze node importance
#[derive(Message)]
#[rtype(result = "Result<HashMap<u32, f32>, String>")]
pub struct AnalyzeNodeImportanceMessage {
    pub algorithm: ImportanceAlgorithm,
}

/// Update semantic graph data
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateSemanticGraphDataMessage {
    pub graph: Arc<GraphData>,
}

/// Get semantic statistics
#[derive(Message)]
#[rtype(result = "Result<SemanticStatistics, String>")]
pub struct GetSemanticStatisticsMessage;

/// Invalidate pathfinding cache
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InvalidatePathfindingCacheMessage;

// ============================================================================
// Message Conversion Helpers
// ============================================================================

impl InitializePhysicsMessage {
    pub fn new(graph: Arc<GraphData>, params: PhysicsParameters) -> Self {
        Self { graph, params }
    }
}

impl UpdatePositionsMessage {
    pub fn new(forces: Vec<NodeForce>) -> Self {
        Self { forces }
    }
}

impl ApplyExternalForcesMessage {
    pub fn new(forces: Vec<(u32, f32, f32, f32)>) -> Self {
        Self { forces }
    }
}

impl PinNodesMessage {
    pub fn new(nodes: Vec<(u32, f32, f32, f32)>) -> Self {
        Self { nodes }
    }
}

impl UnpinNodesMessage {
    pub fn new(node_ids: Vec<u32>) -> Self {
        Self { node_ids }
    }
}

impl UpdatePhysicsParametersMessage {
    pub fn new(params: PhysicsParameters) -> Self {
        Self { params }
    }
}

impl UpdatePhysicsGraphDataMessage {
    pub fn new(graph: Arc<GraphData>) -> Self {
        Self { graph }
    }
}

impl InitializeSemanticMessage {
    pub fn new(graph: Arc<GraphData>) -> Self {
        Self { graph }
    }
}

impl DetectCommunitiesMessage {
    pub fn new(algorithm: ClusteringAlgorithm) -> Self {
        Self { algorithm }
    }
}

impl ComputeShortestPathsMessage {
    pub fn new(source_node_id: u32) -> Self {
        Self { source_node_id }
    }
}

impl ComputeSsspDistancesMessage {
    pub fn new(source_node_id: u32) -> Self {
        Self { source_node_id }
    }
}

impl ComputeLandmarkApspMessage {
    pub fn new(num_landmarks: usize) -> Self {
        Self { num_landmarks }
    }
}

impl GenerateSemanticConstraintsMessage {
    pub fn new(config: SemanticConstraintConfig) -> Self {
        Self { config }
    }
}

impl OptimizeLayoutMessage {
    pub fn new(constraints: ConstraintSet, max_iterations: usize) -> Self {
        Self {
            constraints,
            max_iterations,
        }
    }
}

impl AnalyzeNodeImportanceMessage {
    pub fn new(algorithm: ImportanceAlgorithm) -> Self {
        Self { algorithm }
    }
}

impl UpdateSemanticGraphDataMessage {
    pub fn new(graph: Arc<GraphData>) -> Self {
        Self { graph }
    }
}
