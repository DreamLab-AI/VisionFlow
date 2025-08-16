//! Message definitions for actor system communication

use actix::prelude::*;
use glam::Vec3;
use serde_json::Value;
use std::collections::HashMap;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::MetadataStore;
use crate::config::AppFullSettings;
use crate::models::graph::GraphData as ServiceGraphData;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::models::simulation_params::SimulationParams;
use crate::models::graph::GraphData as ModelsGraphData;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use crate::models::constraints::{AdvancedParams, ConstraintSet};
use crate::actors::gpu_compute_actor::ComputeMode;
use crate::gpu::visual_analytics::{VisualAnalyticsParams, IsolationLayer};

// Graph Service Actor Messages
#[derive(Message)]
#[rtype(result = "Result<ServiceGraphData, String>")]
pub struct GetGraphData;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodePositions {
    pub positions: Vec<(u32, BinaryNodeData)>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct AddNode {
    pub node: Node,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RemoveNode {
    pub node_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct AddEdge {
    pub edge: Edge,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RemoveEdge {
    pub edge_id: String,
}

#[derive(Message)]
#[rtype(result = "Result<HashMap<u32, Node>, String>")]
pub struct GetNodeMap;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BuildGraphFromMetadata {
    pub metadata: MetadataStore,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct StartSimulation;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodePosition {
    pub node_id: u32,
    pub position: Vec3,
    pub velocity: Vec3,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SimulationStep;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct StopSimulation;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateGraphData {
    pub graph_data: ServiceGraphData,
}

// Advanced Physics and Constraint Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateAdvancedParams {
    pub params: AdvancedParams,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateConstraintData {
    pub constraint_data: Value,
}

#[derive(Message)]
#[rtype(result = "Result<ConstraintSet, String>")]
pub struct GetConstraints;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct TriggerStressMajorization;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RegenerateSemanticConstraints;

#[derive(Message)]
#[rtype(result = "()")]
pub struct SetAdvancedGPUContext {
    pub context: crate::utils::unified_gpu_compute::UnifiedGPUCompute,
}

// Visual Analytics Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitializeVisualAnalytics {
    pub max_nodes: usize,
    pub max_edges: usize,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateVisualAnalyticsParams {
    pub params: VisualAnalyticsParams,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct AddIsolationLayer {
    pub layer: IsolationLayer,
}

#[derive(Message)]
#[rtype(result = "Result<bool, String>")]
pub struct RemoveIsolationLayer {
    pub layer_id: i32,
}

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct GetKernelMode;

// Settings Actor Messages
#[derive(Message)]
#[rtype(result = "Result<AppFullSettings, String>")]
pub struct GetSettings;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateSettings {
    pub settings: AppFullSettings,
}

#[derive(Message)]
#[rtype(result = "Result<Value, String>")]
pub struct GetSettingByPath {
    pub path: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SetSettingByPath {
    pub path: String,
    pub value: Value,
}

// Metadata Actor Messages
#[derive(Message)]
#[rtype(result = "Result<MetadataStore, String>")]
pub struct GetMetadata;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateMetadata {
    pub metadata: MetadataStore,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RefreshMetadata;

// Client Manager Actor Messages
#[derive(Message)]
#[rtype(result = "Result<usize, String>")]
pub struct RegisterClient {
    pub addr: actix::Addr<crate::handlers::socket_flow_handler::SocketFlowServer>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UnregisterClient {
    pub client_id: usize,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BroadcastNodePositions {
    pub positions: Vec<u8>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BroadcastMessage {
    pub message: String,
}

#[derive(Message)]
#[rtype(result = "Result<usize, String>")]
pub struct GetClientCount;

// Messages for ClientManagerActor to send to individual SocketFlowServer clients
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendToClientBinary(pub Vec<u8>);

#[derive(Message)]
#[rtype(result = "()")]
pub struct SendToClientText(pub String);

// Claude Flow Actor Messages - Enhanced for Hive Mind Swarm
use crate::types::claude_flow::AgentStatus;
use crate::models::graph::GraphData;

#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateBotsGraph {
    pub agents: Vec<AgentStatus>,
}

#[derive(Message)]
#[rtype(result = "Result<GraphData, String>")]
pub struct GetBotsGraphData;

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct InitializeSwarm {
    pub topology: String,
    pub max_agents: u32,
    pub strategy: String,
    pub enable_neural: bool,
    pub agent_types: Vec<String>,
    pub custom_prompt: Option<String>,
}

// Connection status messages
#[derive(Message)]
#[rtype(result = "()")]
pub struct ConnectionFailed;

#[derive(Message)]
#[rtype(result = "()")]
pub struct PollAgentStatuses;

// Agent update structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUpdate {
    pub agent_id: String,
    pub status: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// Enhanced MCP Tool Messages for Hive Mind Swarm
#[derive(Message)]
#[rtype(result = "Result<SwarmStatus, String>")]
pub struct GetSwarmStatus;

#[derive(Message)]
#[rtype(result = "Result<Vec<AgentMetrics>, String>")]
pub struct GetAgentMetrics;

#[derive(Message)]
#[rtype(result = "Result<AgentStatus, String>")]
pub struct SpawnAgent {
    pub agent_type: String,
    pub name: String,
    pub capabilities: Vec<String>,
    pub swarm_id: Option<String>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct TaskOrchestrate {
    pub task_id: String,
    pub task_type: String,
    pub assigned_agents: Vec<String>,
    pub priority: u8,
}

#[derive(Message)]
#[rtype(result = "Result<SwarmMonitorData, String>")]
pub struct SwarmMonitor;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct TopologyOptimize {
    pub current_topology: String,
    pub performance_metrics: HashMap<String, f32>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct LoadBalance {
    pub agent_workloads: HashMap<String, f32>,
    pub target_efficiency: f32,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct CoordinationSync {
    pub coordination_pattern: String,
    pub participants: Vec<String>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SwarmScale {
    pub target_agent_count: u32,
    pub scaling_strategy: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SwarmDestroy {
    pub swarm_id: String,
    pub graceful_shutdown: bool,
}

// Neural Network MCP Tool Messages
#[derive(Message)]
#[rtype(result = "Result<NeuralStatus, String>")]
pub struct GetNeuralStatus;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct NeuralTrain {
    pub pattern_data: Vec<f32>,
    pub training_config: HashMap<String, Value>,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<f32>, String>")]
pub struct NeuralPredict {
    pub input_data: Vec<f32>,
    pub model_id: String,
}

// Memory & Persistence MCP Tool Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct MemoryPersist {
    pub namespace: String,
    pub key: String,
    pub data: Value,
}

#[derive(Message)]
#[rtype(result = "Result<Value, String>")]
pub struct MemorySearch {
    pub namespace: String,
    pub pattern: String,
    pub limit: Option<u32>,
}

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct StateSnapshot {
    pub snapshot_id: String,
    pub include_agent_states: bool,
}

// Analysis & Monitoring MCP Tool Messages
#[derive(Message)]
#[rtype(result = "Result<PerformanceReport, String>")]
pub struct GetPerformanceReport {
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    pub agent_filter: Option<Vec<String>>,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<Bottleneck>, String>")]
pub struct BottleneckAnalyze;

#[derive(Message)]
#[rtype(result = "Result<SystemMetrics, String>")]
pub struct MetricsCollect;

// Data Structures for MCP Responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    pub swarm_id: String,
    pub active_agents: u32,
    pub total_agents: u32,
    pub topology: String,
    pub health_score: f32,
    pub coordination_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub agent_id: String,
    pub performance_score: f32,
    pub tasks_completed: u32,
    pub success_rate: f32,
    pub resource_utilization: f32,
    pub token_usage: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMonitorData {
    pub timestamp: DateTime<Utc>,
    pub agent_states: HashMap<String, String>,
    pub message_flow: Vec<MessageFlowEvent>,
    pub coordination_patterns: Vec<CoordinationPattern>,
    pub system_metrics: SystemMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageFlowEvent {
    pub id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub message_type: String,
    pub priority: u8,
    pub timestamp: DateTime<Utc>,
    pub latency_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationPattern {
    pub id: String,
    pub pattern_type: String, // hierarchy, mesh, consensus, pipeline
    pub participants: Vec<String>,
    pub status: String, // forming, active, completing, completed
    pub progress: f32, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralStatus {
    pub models_loaded: u32,
    pub training_active: bool,
    pub wasm_optimization: bool,
    pub memory_usage_mb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub swarm_performance: f32,
    pub agent_performances: HashMap<String, f32>,
    pub bottlenecks: Vec<Bottleneck>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub component: String,
    pub severity: f32, // 0.0 to 1.0
    pub description: String,
    pub suggested_fix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemMetrics {
    pub active_agents: u32,
    pub message_rate: f32, // messages per second
    pub average_latency: f32, // milliseconds
    pub error_rate: f32, // 0.0 to 1.0
    pub network_health: f32, // 0.0 to 1.0
    pub cpu_usage: f32, // 0.0 to 1.0
    pub memory_usage: f32, // 0.0 to 1.0
    pub gpu_usage: Option<f32>, // 0.0 to 1.0
}

// GPU Compute Actor Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitializeGPU {
    pub graph: ModelsGraphData,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateGPUGraphData {
    pub graph: ModelsGraphData,
}

#[derive(Message, Clone)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateSimulationParams {
    pub params: SimulationParams,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ComputeForces;

#[derive(Message)]
#[rtype(result = "Result<Vec<BinaryNodeData>, String>")]
pub struct GetNodeData;

#[derive(Message)]
#[rtype(result = "GPUStatus")]
pub struct GetGPUStatus;

#[derive(Debug, Clone)]
pub struct GPUStatus {
    pub is_initialized: bool,
    pub cpu_fallback_active: bool,
    pub failure_count: u32,
    pub iteration_count: u32,
    pub num_nodes: u32,
}

// Position synchronization messages
#[derive(Message, Clone)]
#[rtype(result = "Result<PositionSnapshot, String>")]
pub struct RequestPositionSnapshot {
    pub include_knowledge_graph: bool,
    pub include_agent_graph: bool,
}

// Removed UpdatePhysicsParams - deprecated WebSocket physics path
// Use UpdateSimulationParams via REST API instead

#[derive(Debug, Clone)]
pub struct PositionSnapshot {
    pub knowledge_nodes: Vec<(u32, BinaryNodeData)>,
    pub agent_nodes: Vec<(u32, BinaryNodeData)>,
    pub timestamp: std::time::Instant,
}

// Enhanced Claude Flow Actor Messages (for polling and retry)
#[derive(Message)]
#[rtype(result = "()")]
pub struct PollSwarmData;

#[derive(Message)]
#[rtype(result = "()")]
pub struct PollSystemMetrics;

#[derive(Message)]
#[rtype(result = "()")]
pub struct RetryMCPConnection;

#[derive(Message)]
#[rtype(result = "Result<Vec<AgentStatus>, String>")]
pub struct GetCachedAgentStatuses;

// GPU Compute Mode Control Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SetComputeMode {
    pub mode: ComputeMode,
}

#[derive(Message)]
#[rtype(result = "Result<crate::actors::gpu_compute_actor::PhysicsStats, String>")]
pub struct GetPhysicsStats;

// GPU Force Parameters
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateForceParams {
    pub repulsion: f32,
    pub attraction: f32,
    pub damping: f32,
    pub temperature: f32,
    pub spring: f32,
    pub gravity: f32,
    pub time_step: f32,
    pub max_velocity: f32,
}

// GPU Clustering Messages
#[derive(Message, Clone)]
#[rtype(result = "Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>")]
pub struct PerformGPUClustering {
    pub method: String,
    pub params: crate::handlers::api_handler::analytics::ClusteringParams,
    pub task_id: String,
}

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct StartGPUClustering {
    pub algorithm: String,
    pub cluster_count: u32,
    pub task_id: String,
}

#[derive(Message)]
#[rtype(result = "Result<Value, String>")]
pub struct GetClusteringStatus;

#[derive(Message)]
#[rtype(result = "Result<Value, String>")]
pub struct GetClusteringResults;

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct ExportClusterAssignments {
    pub format: String,
}

// GPU Constraint Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateConstraints {
    pub constraint_data: Value,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ApplyConstraintsToNodes {
    pub constraint_type: String,
    pub node_ids: Vec<u32>,
    pub strength: f32,
}

#[derive(Message)]
#[rtype(result = "Result<u32, String>")]
pub struct RemoveConstraints {
    pub constraint_type: Option<String>,
    pub node_ids: Option<Vec<u32>>,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<Value>, String>")]
pub struct GetActiveConstraints;

