//! Graph Service Actor Messages
//!
//! This module defines all message types used for communication with the GraphServiceActor
//! and its separated child actors. The messages are organized into logical groups:
//!
//! - **Graph State Messages**: Node/edge CRUD operations, graph data management
//! - **Physics Messages**: Simulation control, GPU operations, position updates
//! - **Semantic Messages**: AI features, constraint management, semantic analysis
//! - **Client Messages**: WebSocket communication, position broadcasting

use actix::prelude::*;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::metadata::FileMetadata;
use crate::models::constraints::{ConstraintSet, AdvancedParams};
use crate::models::simulation_params::SimulationParams;
use crate::utils::socket_flow_messages::{BinaryNodeData, BinaryNodeDataClient};
use crate::errors::VisionFlowError;
use std::collections::HashMap;
use std::sync::Arc;

/// Message-based graph operations to replace Arc::make_mut() patterns
#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct AddNode {
    pub node: Node,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct RemoveNode {
    pub node_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct AddEdge {
    pub edge: Edge,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct RemoveEdge {
    pub edge_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct UpdateMetadata {
    pub metadata: FileMetadata,
}

#[derive(Message)]
#[rtype(result = "Result<Node, Box<dyn std::error::Error>>")]
pub struct GetNode {
    pub node_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<GraphData, Box<dyn std::error::Error>>")]
pub struct GetGraphData;

#[derive(Message)]
#[rtype(result = "Result<HashMap<u32, Node>, Box<dyn std::error::Error>>")]
pub struct GetNodeMap;

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct ClearGraph;

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct BatchUpdateNodes {
    pub nodes: Vec<Node>,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct UpdateNodePosition {
    pub node_id: u32,
    pub position: (f32, f32, f32),
}

// ============================================================================
// GRAPH STATE MESSAGES - Node and Edge operations
// ============================================================================

/// Message for updating multiple node positions at once
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodePositions {
    pub positions: Vec<BinaryNodeData>,
}

/// Message for building graph from metadata
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BuildGraphFromMetadata {
    pub metadata: Vec<FileMetadata>,
}

/// Message for adding nodes from metadata
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct AddNodesFromMetadata {
    pub metadata: Vec<FileMetadata>,
}

/// Batch message for adding multiple nodes efficiently
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BatchAddNodes {
    pub nodes: Vec<Node>,
}

/// Batch message for adding multiple edges efficiently
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BatchAddEdges {
    pub edges: Vec<Edge>,
}

/// Batch message for mixed node and edge operations
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BatchGraphUpdate {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub remove_node_ids: Vec<u32>,
    pub remove_edge_ids: Vec<String>,
}

/// Message to flush pending queue operations
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct FlushUpdateQueue;

/// Message to configure queue parameters
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ConfigureUpdateQueue {
    pub max_operations: usize,
    pub flush_interval_ms: u64,
    pub enable_auto_flush: bool,
}

/// Message for updating a node from metadata
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodeFromMetadata {
    pub metadata_id: String,
    pub metadata: FileMetadata,
}

/// Message for removing a node by metadata ID
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RemoveNodeByMetadata {
    pub metadata_id: String,
}

/// Message for updating entire graph data
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateGraphData {
    pub graph_data: GraphData,
}

/// Message for updating bots graph
#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateBotsGraph {
    pub agents: Vec<serde_json::Value>, // Generic JSON for agent data
}

/// Message for getting bots graph data
#[derive(Message)]
#[rtype(result = "Result<Arc<GraphData>, String>")]
pub struct GetBotsGraphData;

/// Message for computing shortest paths
#[derive(Message)]
#[rtype(result = "Result<HashMap<u32, Option<f32>>, String>")]
pub struct ComputeShortestPaths {
    pub source_node_id: u32,
}

// ============================================================================
// PHYSICS MESSAGES - Simulation control and physics operations
// ============================================================================

/// Message for starting physics simulation
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct StartSimulation;

/// Message for stopping physics simulation
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct StopSimulation;

/// Message for performing one simulation step
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SimulationStep;

/// Message for updating simulation parameters
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateSimulationParams {
    pub params: SimulationParams,
}

/// Message for storing GPU compute actor address
#[derive(Message)]
#[rtype(result = "()")]
pub struct StoreGPUComputeAddress {
    pub addr: Option<()>, // Placeholder for actor address
}

/// Message for initializing GPU connection
#[derive(Message)]
#[rtype(result = "()")]
pub struct InitializeGPUConnection {
    pub force_reinit: bool,
}

/// Message indicating GPU has been initialized
#[derive(Message)]
#[rtype(result = "()")]
pub struct GPUInitialized;

/// Message for setting advanced GPU context
#[derive(Message)]
#[rtype(result = "()")]
pub struct SetAdvancedGPUContext;

/// Message for resetting GPU initialization flag
#[derive(Message)]
#[rtype(result = "()")]
pub struct ResetGPUInitFlag;

/// Message for requesting position snapshot
#[derive(Message)]
#[rtype(result = "Result<PositionSnapshot, String>")]
pub struct RequestPositionSnapshot {
    pub include_metadata: bool,
}

/// Position snapshot data structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PositionSnapshot {
    pub knowledge_nodes: Vec<BinaryNodeDataClient>,
    pub agent_nodes: Vec<BinaryNodeDataClient>,
    pub timestamp: u64,
}

/// Message for physics pause control
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct PhysicsPauseMessage {
    pub pause: bool,
    pub source: String,
}

/// Message for node interaction events
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct NodeInteractionMessage {
    pub node_id: u32,
    pub interaction_type: String,
    pub client_id: Option<String>,
}

/// Message for forcing physics resume
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct ForceResumePhysics {
    pub reason: String,
}

/// Message for getting equilibrium status
#[derive(Message)]
#[rtype(result = "Result<bool, VisionFlowError>")]
pub struct GetEquilibriumStatus;

/// Auto-balance notification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutoBalanceNotification {
    pub message: String,
    pub timestamp: i64,
    pub severity: String, // "info", "warning", "success"
}

/// Message for getting auto-balance notifications
#[derive(Message)]
#[rtype(result = "Result<Vec<AutoBalanceNotification>, String>")]
pub struct GetAutoBalanceNotifications {
    pub since_timestamp: Option<i64>,
    pub limit: Option<usize>,
}

// ============================================================================
// SEMANTIC MESSAGES - AI and constraint operations
// ============================================================================

/// Message for updating advanced physics parameters
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateAdvancedParams {
    pub params: AdvancedParams,
}

/// Message for updating constraints
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateConstraints {
    pub constraint_data: serde_json::Value, // Generic constraint data
}

/// Message for getting current constraints
#[derive(Message)]
#[rtype(result = "Result<ConstraintSet, String>")]
pub struct GetConstraints;

/// Message for triggering stress majorization
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct TriggerStressMajorization;

/// Message for regenerating semantic constraints
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RegenerateSemanticConstraints;

// ============================================================================
// CLIENT MESSAGES - WebSocket and client operations
// ============================================================================

/// Message for forcing position broadcast
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ForcePositionBroadcast {
    pub reason: String,
}

/// Message for initial client synchronization
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitialClientSync {
    pub client_identifier: String,
    pub trigger_source: String,
}

// ============================================================================
// MESSAGE ENUMS FOR GROUPING
// ============================================================================

/// Grouped messages for graph state operations
#[derive(Message)]
#[rtype(result = "Result<GraphStateResponse, String>")]
pub enum GraphStateMessages {
    AddNode(AddNode),
    RemoveNode(RemoveNode),
    AddEdge(AddEdge),
    RemoveEdge(RemoveEdge),
    GetGraphData(GetGraphData),
    GetNodeMap(GetNodeMap),
    UpdateGraphData(UpdateGraphData),
    BuildFromMetadata(BuildGraphFromMetadata),
    AddNodesFromMetadata(AddNodesFromMetadata),
    UpdateNodeFromMetadata(UpdateNodeFromMetadata),
    RemoveNodeByMetadata(RemoveNodeByMetadata),
    UpdateBotsGraph(UpdateBotsGraph),
    GetBotsGraphData(GetBotsGraphData),
    ComputeShortestPaths(ComputeShortestPaths),
    BatchAddNodes(BatchAddNodes),
    BatchAddEdges(BatchAddEdges),
    BatchGraphUpdate(BatchGraphUpdate),
    FlushUpdateQueue(FlushUpdateQueue),
    ConfigureUpdateQueue(ConfigureUpdateQueue),
}

/// Response types for graph state operations
#[derive(Debug, Clone)]
pub enum GraphStateResponse {
    Success,
    GraphData(Arc<GraphData>),
    NodeMap(Arc<HashMap<u32, Node>>),
    ShortestPaths(HashMap<u32, Option<f32>>),
    Error(String),
}

/// Grouped messages for physics operations
#[derive(Message)]
#[rtype(result = "Result<PhysicsResponse, String>")]
pub enum PhysicsMessages {
    StartSimulation(StartSimulation),
    StopSimulation(StopSimulation),
    SimulationStep(SimulationStep),
    UpdateSimulationParams(UpdateSimulationParams),
    UpdateNodePositions(UpdateNodePositions),
    UpdateNodePosition(UpdateNodePosition),
    StoreGPUAddress(StoreGPUComputeAddress),
    InitializeGPU(InitializeGPUConnection),
    GPUInitialized(GPUInitialized),
    SetAdvancedGPUContext(SetAdvancedGPUContext),
    ResetGPUFlag(ResetGPUInitFlag),
    RequestSnapshot(RequestPositionSnapshot),
    PhysicsPause(PhysicsPauseMessage),
    NodeInteraction(NodeInteractionMessage),
    ForceResume(ForceResumePhysics),
    GetEquilibrium(GetEquilibriumStatus),
    GetAutoBalance(GetAutoBalanceNotifications),
}

/// Response types for physics operations
#[derive(Debug, Clone)]
pub enum PhysicsResponse {
    Success,
    PositionSnapshot(PositionSnapshot),
    EquilibriumStatus(bool),
    AutoBalanceNotifications(Vec<AutoBalanceNotification>),
    Error(String),
}

/// Grouped messages for semantic operations
#[derive(Message)]
#[rtype(result = "Result<SemanticResponse, String>")]
pub enum SemanticMessages {
    UpdateAdvanced(UpdateAdvancedParams),
    UpdateConstraints(UpdateConstraints),
    GetConstraints(GetConstraints),
    TriggerStress(TriggerStressMajorization),
    RegenerateConstraints(RegenerateSemanticConstraints),
}

/// Response types for semantic operations
#[derive(Debug, Clone)]
pub enum SemanticResponse {
    Success,
    Constraints(ConstraintSet),
    Error(String),
}

/// Grouped messages for client operations
#[derive(Message)]
#[rtype(result = "Result<ClientResponse, String>")]
pub enum ClientMessages {
    ForceBroadcast(ForcePositionBroadcast),
    InitialSync(InitialClientSync),
}

/// Response types for client operations
#[derive(Debug, Clone)]
pub enum ClientResponse {
    Success,
    Error(String),
}

// ============================================================================
// INTER-ACTOR COMMUNICATION PROTOCOLS
// ============================================================================

/// Protocol for communication between GraphStateActor and PhysicsOrchestratorActor
pub trait GraphStateToPhysicsProtocol {
    fn notify_node_added(&self, node: &Node) -> Result<(), String>;
    fn notify_node_removed(&self, node_id: u32) -> Result<(), String>;
    fn notify_edge_added(&self, edge: &Edge) -> Result<(), String>;
    fn notify_edge_removed(&self, edge_id: u32) -> Result<(), String>;
    fn notify_positions_updated(&self, positions: &[BinaryNodeData]) -> Result<(), String>;
}

/// Protocol for communication between PhysicsOrchestratorActor and SemanticProcessorActor
pub trait PhysicsToSemanticProtocol {
    fn notify_simulation_started(&self) -> Result<(), String>;
    fn notify_simulation_stopped(&self) -> Result<(), String>;
    fn notify_equilibrium_reached(&self) -> Result<(), String>;
    fn request_constraint_update(&self) -> Result<(), String>;
}

/// Protocol for communication between SemanticProcessorActor and ClientCoordinatorActor
pub trait SemanticToClientProtocol {
    fn notify_constraints_updated(&self, constraints: &ConstraintSet) -> Result<(), String>;
    fn notify_semantic_analysis_complete(&self) -> Result<(), String>;
}

/// Protocol for communication between ClientCoordinatorActor and GraphStateActor
pub trait ClientToGraphStateProtocol {
    fn request_initial_sync(&self, client_id: &str) -> Result<(), String>;
    fn request_position_broadcast(&self) -> Result<(), String>;
    fn notify_client_connected(&self, client_id: &str) -> Result<(), String>;
    fn notify_client_disconnected(&self, client_id: &str) -> Result<(), String>;
}

/// Unified message routing interface for the GraphServiceSupervisor
pub trait MessageRouter {
    fn route_graph_state_message(&self, msg: GraphStateMessages) -> Result<GraphStateResponse, String>;
    fn route_physics_message(&self, msg: PhysicsMessages) -> Result<PhysicsResponse, String>;
    fn route_semantic_message(&self, msg: SemanticMessages) -> Result<SemanticResponse, String>;
    fn route_client_message(&self, msg: ClientMessages) -> Result<ClientResponse, String>;
}

// ============================================================================
// ACTOR ADDRESSES AND ROUTING
// ============================================================================

/// Message for health check between actors
#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct HealthCheck {
    pub actor_name: String,
}