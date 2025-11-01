//! Graph Service Actor Messages
//!
//! This module defines all message types used for communication with the GraphServiceActor
//! and its separated child actors. The messages are organized into logical groups:
//!
//! - **Graph State Messages**: Node/edge CRUD operations, graph data management
//! - **Physics Messages**: Simulation control, GPU operations, position updates
//! - **Semantic Messages**: AI features, constraint management, semantic analysis
//! - **Client Messages**: WebSocket communication, position broadcasting

use crate::errors::VisionFlowError;
use crate::models::constraints::{AdvancedParams, ConstraintSet};
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::metadata::FileMetadata;
use crate::models::node::Node;
use crate::models::simulation_params::SimulationParams;
use crate::utils::socket_flow_messages::{BinaryNodeData, BinaryNodeDataClient};
use actix::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/
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

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodePositions {
    pub positions: Vec<BinaryNodeData>,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BuildGraphFromMetadata {
    pub metadata: Vec<FileMetadata>,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct AddNodesFromMetadata {
    pub metadata: Vec<FileMetadata>,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BatchAddNodes {
    pub nodes: Vec<Node>,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BatchAddEdges {
    pub edges: Vec<Edge>,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BatchGraphUpdate {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub remove_node_ids: Vec<u32>,
    pub remove_edge_ids: Vec<String>,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct FlushUpdateQueue;

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ConfigureUpdateQueue {
    pub max_operations: usize,
    pub flush_interval_ms: u64,
    pub enable_auto_flush: bool,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodeFromMetadata {
    pub metadata_id: String,
    pub metadata: FileMetadata,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RemoveNodeByMetadata {
    pub metadata_id: String,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateGraphData {
    pub graph_data: GraphData,
}

/
#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateBotsGraph {
    pub agents: Vec<serde_json::Value>, 
}

/
#[derive(Message)]
#[rtype(result = "Result<Arc<GraphData>, String>")]
pub struct GetBotsGraphData;

/
#[derive(Message)]
#[rtype(result = "Result<HashMap<u32, Option<f32>>, String>")]
pub struct ComputeShortestPaths {
    pub source_node_id: u32,
}

// ============================================================================
// PHYSICS MESSAGES - Simulation control and physics operations
// ============================================================================

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct StartSimulation;

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct StopSimulation;

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SimulationStep;

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateSimulationParams {
    pub params: SimulationParams,
}

/
#[derive(Message)]
#[rtype(result = "()")]
pub struct StoreGPUComputeAddress {
    pub addr: Option<()>, 
}

/
#[derive(Message)]
#[rtype(result = "()")]
pub struct InitializeGPUConnection {
    pub force_reinit: bool,
}

/
#[derive(Message)]
#[rtype(result = "()")]
pub struct GPUInitialized;

/
#[derive(Message)]
#[rtype(result = "()")]
pub struct SetAdvancedGPUContext;

/
#[derive(Message)]
#[rtype(result = "()")]
pub struct ResetGPUInitFlag;

/
#[derive(Message)]
#[rtype(result = "Result<PositionSnapshot, String>")]
pub struct RequestPositionSnapshot {
    pub include_metadata: bool,
}

/
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PositionSnapshot {
    pub knowledge_nodes: Vec<BinaryNodeDataClient>,
    pub agent_nodes: Vec<BinaryNodeDataClient>,
    pub timestamp: u64,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct PhysicsPauseMessage {
    pub pause: bool,
    pub source: String,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct NodeInteractionMessage {
    pub node_id: u32,
    pub interaction_type: String,
    pub client_id: Option<String>,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct ForceResumePhysics {
    pub reason: String,
}

/
#[derive(Message)]
#[rtype(result = "Result<bool, VisionFlowError>")]
pub struct GetEquilibriumStatus;

/
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutoBalanceNotification {
    pub message: String,
    pub timestamp: i64,
    pub severity: String, 
}

/
#[derive(Message)]
#[rtype(result = "Result<Vec<AutoBalanceNotification>, String>")]
pub struct GetAutoBalanceNotifications {
    pub since_timestamp: Option<i64>,
    pub limit: Option<usize>,
}

// ============================================================================
// SEMANTIC MESSAGES - AI and constraint operations
// ============================================================================

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateAdvancedParams {
    pub params: AdvancedParams,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateConstraints {
    pub constraint_data: serde_json::Value, 
}

/
#[derive(Message)]
#[rtype(result = "Result<ConstraintSet, String>")]
pub struct GetConstraints;

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct TriggerStressMajorization;

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RegenerateSemanticConstraints;

// ============================================================================
// CLIENT MESSAGES - WebSocket and client operations
// ============================================================================

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ForcePositionBroadcast {
    pub reason: String,
}

/
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitialClientSync {
    pub client_identifier: String,
    pub trigger_source: String,
}

// ============================================================================
// MESSAGE ENUMS FOR GROUPING
// ============================================================================

/
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

/
#[derive(Debug, Clone)]
pub enum GraphStateResponse {
    Success,
    GraphData(Arc<GraphData>),
    NodeMap(Arc<HashMap<u32, Node>>),
    ShortestPaths(HashMap<u32, Option<f32>>),
    Error(String),
}

/
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

/
#[derive(Debug, Clone)]
pub enum PhysicsResponse {
    Success,
    PositionSnapshot(PositionSnapshot),
    EquilibriumStatus(bool),
    AutoBalanceNotifications(Vec<AutoBalanceNotification>),
    Error(String),
}

/
#[derive(Message)]
#[rtype(result = "Result<SemanticResponse, String>")]
pub enum SemanticMessages {
    UpdateAdvanced(UpdateAdvancedParams),
    UpdateConstraints(UpdateConstraints),
    GetConstraints(GetConstraints),
    TriggerStress(TriggerStressMajorization),
    RegenerateConstraints(RegenerateSemanticConstraints),
}

/
#[derive(Debug, Clone)]
pub enum SemanticResponse {
    Success,
    Constraints(ConstraintSet),
    Error(String),
}

/
#[derive(Message)]
#[rtype(result = "Result<ClientResponse, String>")]
pub enum ClientMessages {
    ForceBroadcast(ForcePositionBroadcast),
    InitialSync(InitialClientSync),
}

/
#[derive(Debug, Clone)]
pub enum ClientResponse {
    Success,
    Error(String),
}

// ============================================================================
// INTER-ACTOR COMMUNICATION PROTOCOLS
// ============================================================================

/
pub trait GraphStateToPhysicsProtocol {
    fn notify_node_added(&self, node: &Node) -> Result<(), String>;
    fn notify_node_removed(&self, node_id: u32) -> Result<(), String>;
    fn notify_edge_added(&self, edge: &Edge) -> Result<(), String>;
    fn notify_edge_removed(&self, edge_id: u32) -> Result<(), String>;
    fn notify_positions_updated(&self, positions: &[BinaryNodeData]) -> Result<(), String>;
}

/
pub trait PhysicsToSemanticProtocol {
    fn notify_simulation_started(&self) -> Result<(), String>;
    fn notify_simulation_stopped(&self) -> Result<(), String>;
    fn notify_equilibrium_reached(&self) -> Result<(), String>;
    fn request_constraint_update(&self) -> Result<(), String>;
}

/
pub trait SemanticToClientProtocol {
    fn notify_constraints_updated(&self, constraints: &ConstraintSet) -> Result<(), String>;
    fn notify_semantic_analysis_complete(&self) -> Result<(), String>;
}

/
pub trait ClientToGraphStateProtocol {
    fn request_initial_sync(&self, client_id: &str) -> Result<(), String>;
    fn request_position_broadcast(&self) -> Result<(), String>;
    fn notify_client_connected(&self, client_id: &str) -> Result<(), String>;
    fn notify_client_disconnected(&self, client_id: &str) -> Result<(), String>;
}

/
pub trait MessageRouter {
    fn route_graph_state_message(
        &self,
        msg: GraphStateMessages,
    ) -> Result<GraphStateResponse, String>;
    fn route_physics_message(&self, msg: PhysicsMessages) -> Result<PhysicsResponse, String>;
    fn route_semantic_message(&self, msg: SemanticMessages) -> Result<SemanticResponse, String>;
    fn route_client_message(&self, msg: ClientMessages) -> Result<ClientResponse, String>;
}

// ============================================================================
// ACTOR ADDRESSES AND ROUTING
// ============================================================================

/
#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct HealthCheck {
    pub actor_name: String,
}
