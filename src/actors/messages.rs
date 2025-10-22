//! Message definitions for actor system communication

#[cfg(feature = "gpu")]
use crate::actors::gpu::force_compute_actor::PhysicsStats;
use crate::config::AppFullSettings;
use crate::errors::VisionFlowError;
#[cfg(feature = "gpu")]
use crate::gpu::visual_analytics::{IsolationLayer, VisualAnalyticsParams};
use crate::models::constraints::{AdvancedParams, ConstraintSet};
use crate::models::edge::Edge;
use crate::models::graph::GraphData as ServiceGraphData;
use crate::models::graph::GraphData as ModelsGraphData;
use crate::models::metadata::{FileMetadata, MetadataStore};
use crate::models::node::Node;
use crate::models::simulation_params::SimulationParams;
use crate::models::workspace::{
    CreateWorkspaceRequest, UpdateWorkspaceRequest, Workspace, WorkspaceFilter, WorkspaceQuery,
};
use crate::utils::socket_flow_messages::BinaryNodeData;
#[cfg(feature = "gpu")]
use crate::utils::unified_gpu_compute::ComputeMode;
use actix::prelude::*;
use chrono::{DateTime, Utc};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// K-means clustering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansResult {
    pub cluster_assignments: Vec<i32>,
    pub centroids: Vec<(f32, f32, f32)>,
    pub inertia: f32,
    pub iterations: u32,
    pub clusters: Vec<crate::handlers::api_handler::analytics::Cluster>,
    pub stats: crate::actors::gpu::clustering_actor::ClusteringStats,
    pub converged: bool,
    pub final_iteration: u32,
}

// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub lof_scores: Option<Vec<f32>>,
    pub local_densities: Option<Vec<f32>>,
    pub zscore_values: Option<Vec<f32>>,
    pub anomaly_threshold: f32,
    pub num_anomalies: usize,
    pub anomalies: Vec<crate::actors::gpu::anomaly_detection_actor::AnomalyNode>,
    pub stats: AnomalyDetectionStats,
    pub method: AnomalyDetectionMethod,
    pub threshold: f32,
}

// Anomaly detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionStats {
    pub total_nodes_analyzed: u32,
    pub anomalies_found: usize,
    pub detection_threshold: f32,
    pub computation_time_ms: u64,
    pub method: AnomalyDetectionMethod,
    pub average_anomaly_score: f32,
    pub max_anomaly_score: f32,
    pub min_anomaly_score: f32,
}

// Community detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionResult {
    pub node_labels: Vec<i32>,     // Community assignment for each node
    pub num_communities: usize,    // Total number of communities found
    pub modularity: f32,           // Quality metric (higher is better)
    pub iterations: u32,           // Number of iterations until convergence
    pub community_sizes: Vec<i32>, // Size of each community
    pub converged: bool,           // Whether algorithm converged
    pub communities: Vec<crate::actors::gpu::clustering_actor::Community>,
    pub stats: crate::actors::gpu::clustering_actor::CommunityDetectionStats,
    pub algorithm: CommunityDetectionAlgorithm,
}

// K-means clustering parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansParams {
    pub num_clusters: usize,
    pub max_iterations: Option<u32>,
    pub tolerance: Option<f32>,
    pub seed: Option<u32>,
}

// Anomaly detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyParams {
    pub method: AnomalyMethod,
    pub k_neighbors: i32,
    pub radius: f32,
    pub feature_data: Option<Vec<f32>>,
    pub threshold: f32,
}

// Enhanced anomaly detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionParams {
    pub method: AnomalyDetectionMethod,
    pub threshold: Option<f32>,
    pub k_neighbors: Option<i32>,
    pub window_size: Option<usize>,
    pub feature_data: Option<Vec<f32>>,
}

// Community detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionParams {
    pub algorithm: CommunityDetectionAlgorithm,
    pub max_iterations: Option<u32>,
    pub convergence_tolerance: Option<f32>,
    pub synchronous: Option<bool>, // True for sync, false for async propagation
    pub seed: Option<u32>,         // Random seed for tie-breaking
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyMethod {
    LocalOutlierFactor,
    ZScore,
}

// Enhanced anomaly detection method enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    LOF,
    ZScore,
    IsolationForest,
    DBSCAN,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunityDetectionAlgorithm {
    LabelPropagation,
    Louvain,
    // Future algorithms: Leiden, etc.
}

// Graph Service Actor Messages
#[derive(Message)]
#[rtype(result = "Result<std::sync::Arc<ServiceGraphData>, String>")]
pub struct GetGraphData;

/// NEW: Get settlement/physics state for client optimization
#[derive(Message)]
#[rtype(result = "Result<crate::actors::graph_actor::PhysicsState, String>")]
pub struct GetPhysicsState;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodePositions {
    pub positions: Vec<(u32, BinaryNodeData)>,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<(u32, Vec3)>, String>")]
pub struct GetNodePositions;

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
#[rtype(result = "Result<std::sync::Arc<HashMap<u32, Node>>, String>")]
pub struct GetNodeMap;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BuildGraphFromMetadata {
    pub metadata: MetadataStore,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct AddNodesFromMetadata {
    pub metadata: MetadataStore,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodeFromMetadata {
    pub metadata_id: String,
    pub metadata: FileMetadata,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RemoveNodeByMetadata {
    pub metadata_id: String,
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
    pub graph_data: std::sync::Arc<ServiceGraphData>,
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
pub struct ResetStressMajorizationSafety;

// SSSP (Single-Source Shortest Path) Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ComputeSSSP {
    pub source_node: u32,
}

// Voice Command and Agent Spawning Messages
#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct SpawnAgentCommand {
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub session_id: String,
}

#[derive(Message)]
#[rtype(
    result = "Result<crate::actors::gpu::stress_majorization_actor::StressMajorizationStats, String>"
)]
pub struct GetStressMajorizationStats;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateStressMajorizationParams {
    pub params: AdvancedParams,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RegenerateSemanticConstraints;

#[derive(Message)]
#[rtype(result = "()")]
pub struct SetAdvancedGPUContext {
    // Don't send the whole GPU context, just a signal to initialize it
    pub initialize: bool,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct ResetGPUInitFlag;

#[derive(Message)]
#[rtype(result = "()")]
pub struct StoreAdvancedGPUContext {
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

// GPU K-means and Anomaly Detection Messages
#[derive(Message)]
#[rtype(result = "Result<KMeansResult, String>")]
pub struct RunKMeans {
    pub params: KMeansParams,
}

#[derive(Message)]
#[rtype(result = "Result<AnomalyResult, String>")]
pub struct RunAnomalyDetection {
    pub params: AnomalyParams,
}

#[derive(Message)]
#[rtype(result = "Result<CommunityDetectionResult, String>")]
pub struct RunCommunityDetection {
    pub params: CommunityDetectionParams,
}

// Settings Actor Messages
#[derive(Message)]
#[rtype(result = "Result<AppFullSettings, VisionFlowError>")]
pub struct GetSettings;

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct UpdateSettings {
    pub settings: AppFullSettings,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ReloadSettings;

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct MergeSettingsUpdate {
    pub update: serde_json::Value,
}

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct PartialSettingsUpdate {
    pub partial_settings: serde_json::Value,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdatePhysicsFromAutoBalance {
    pub physics_update: serde_json::Value,
}

#[derive(Message)]
#[rtype(result = "Result<Value, VisionFlowError>")]
pub struct GetSettingByPath {
    pub path: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct SetSettingByPath {
    pub path: String,
    pub value: Value,
}

// Batch path-based settings messages for performance
#[derive(Message)]
#[rtype(result = "Result<HashMap<String, Value>, VisionFlowError>")]
pub struct GetSettingsByPaths {
    pub paths: Vec<String>,
}

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct SetSettingsByPaths {
    pub updates: HashMap<String, Value>,
}

// Priority-based update for concurrent update handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpdatePriority {
    Critical = 1, // Physics parameters that affect GPU simulation
    High = 2,     // Visual settings that impact rendering
    Normal = 3,   // General configuration changes
    Low = 4,      // Non-critical settings like UI preferences
}

impl PartialOrd for UpdatePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UpdatePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

// Enhanced update structure with priority and batching support
#[derive(Debug, Clone, PartialEq)]
pub struct PriorityUpdate {
    pub path: String,
    pub value: Value,
    pub priority: UpdatePriority,
    pub timestamp: std::time::Instant,
    pub client_id: Option<String>,
}

impl Eq for PriorityUpdate {}

impl PartialOrd for PriorityUpdate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityUpdate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by priority (Critical < High < Normal < Low)
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => {
                // If priorities are equal, compare by timestamp (earlier first)
                self.timestamp.cmp(&other.timestamp)
            }
            other => other,
        }
    }
}

impl PriorityUpdate {
    pub fn new(path: String, value: Value) -> Self {
        let priority = Self::determine_priority(&path);
        Self {
            path,
            value,
            priority,
            timestamp: std::time::Instant::now(),
            client_id: None,
        }
    }

    pub fn with_client_id(mut self, client_id: String) -> Self {
        self.client_id = Some(client_id);
        self
    }

    fn determine_priority(path: &str) -> UpdatePriority {
        if path.contains(".physics.") {
            // Physics parameters are critical for GPU simulation
            UpdatePriority::Critical
        } else if path.contains(".bloom.") || path.contains(".glow.") || path.contains(".visual") {
            // Visual settings are high priority for user experience
            UpdatePriority::High
        } else if path.contains(".system.") || path.contains(".security.") {
            // System settings have normal priority
            UpdatePriority::Normal
        } else {
            // UI preferences and other settings are low priority
            UpdatePriority::Low
        }
    }
}

// Batched update message for handling concurrent updates efficiently
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct BatchedUpdate {
    pub updates: Vec<PriorityUpdate>,
    pub max_batch_size: usize,
    pub timeout_ms: u64,
}

impl BatchedUpdate {
    pub fn new(updates: Vec<PriorityUpdate>) -> Self {
        Self {
            updates,
            max_batch_size: 50, // Default batch size to prevent mailbox overflow
            timeout_ms: 100,    // Default 100ms timeout for batching
        }
    }

    pub fn with_batch_config(mut self, max_batch_size: usize, timeout_ms: u64) -> Self {
        self.max_batch_size = max_batch_size;
        self.timeout_ms = timeout_ms;
        self
    }

    /// Sort updates by priority (Critical first, Low last)
    pub fn sort_by_priority(&mut self) {
        self.updates.sort_by(|a, b| a.priority.cmp(&b.priority));
    }

    /// Group updates by priority level
    pub fn group_by_priority(&self) -> HashMap<UpdatePriority, Vec<&PriorityUpdate>> {
        let mut groups = HashMap::new();
        for update in &self.updates {
            groups
                .entry(update.priority.clone())
                .or_insert_with(Vec::new)
                .push(update);
        }
        groups
    }
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

// WEBSOCKET SETTLING FIX: Message to force immediate position broadcast for new clients
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ForcePositionBroadcast {
    pub reason: String, // For debugging: "new_client", "settled_override", etc.
}

// UNIFIED INIT: Message to coordinate REST-triggered broadcasts
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitialClientSync {
    pub client_identifier: String, // Can be IP, session ID, or other identifier
    pub trigger_source: String,    // "rest_api", "websocket", etc.
}

// WEBSOCKET SETTLING FIX: Message to set graph service address in client manager
#[derive(Message)]
#[rtype(result = "()")]
pub struct SetGraphServiceAddress {
    pub addr: actix::Addr<crate::actors::graph_actor::GraphServiceActor>,
}

// Message to get the GraphServiceActor from TransitionalGraphSupervisor
#[derive(Message)]
#[rtype(result = "Option<actix::Addr<crate::actors::graph_actor::GraphServiceActor>>")]
pub struct GetGraphServiceActor;

// Messages for ClientManagerActor to send to individual SocketFlowServer clients
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendToClientBinary(pub Vec<u8>);

#[derive(Message)]
#[rtype(result = "()")]
pub struct SendToClientText(pub String);

// Claude Flow Actor Messages - Enhanced for Hive Mind Swarm
use crate::types::claude_flow::AgentStatus;

/// Message to update the agent cache in ClaudeFlowActorTcp
#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateAgentCache {
    pub agents: Vec<AgentStatus>,
}
use crate::models::graph::GraphData;

#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateBotsGraph {
    pub agents: Vec<crate::services::bots_client::Agent>,
}

#[derive(Message)]
#[rtype(result = "Result<std::sync::Arc<GraphData>, String>")]
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

// Auto-pause related messages for equilibrium detection and interaction handling
#[derive(Message, Debug, Clone, Serialize, Deserialize)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct PhysicsPauseMessage {
    pub pause: bool,    // true to pause, false to resume
    pub reason: String, // reason for pause/resume
}

#[derive(Message, Debug, Clone, Serialize, Deserialize)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct NodeInteractionMessage {
    pub node_id: u32,
    pub interaction_type: NodeInteractionType,
    pub position: Option<[f32; 3]>, // Changed from Vec3 to simple array for serde compatibility
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeInteractionType {
    Dragged,  // Node is being dragged
    Selected, // Node was selected
    Released, // Node drag ended
}

#[derive(Message, Debug, Clone, Serialize, Deserialize)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct ForceResumePhysics {
    pub reason: String,
}

#[derive(Message, Debug, Clone, Serialize, Deserialize)]
#[rtype(result = "Result<bool, VisionFlowError>")]
pub struct GetEquilibriumStatus;

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
    pub progress: f32,  // 0.0 to 1.0
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
    pub message_rate: f32,      // messages per second
    pub average_latency: f32,   // milliseconds
    pub error_rate: f32,        // 0.0 to 1.0
    pub network_health: f32,    // 0.0 to 1.0
    pub cpu_usage: f32,         // 0.0 to 1.0
    pub memory_usage: f32,      // 0.0 to 1.0
    pub gpu_usage: Option<f32>, // 0.0 to 1.0
}

// GPU Compute Actor Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitializeGPU {
    pub graph: std::sync::Arc<ModelsGraphData>,
    pub graph_service_addr: Option<Addr<crate::actors::graph_actor::GraphServiceActor>>,
    pub gpu_manager_addr: Option<Addr<crate::actors::GPUManagerActor>>,
}

// Message to notify GraphServiceActor that GPU is ready
#[derive(Message)]
#[rtype(result = "()")]
pub struct GPUInitialized;

// Message to share GPU context with ForceComputeActor and other GPU actors
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SetSharedGPUContext {
    pub context: std::sync::Arc<crate::actors::gpu::shared::SharedGPUContext>,
    pub graph_service_addr: Option<Addr<crate::actors::graph_actor::GraphServiceActor>>,
}

// Message to store GPU compute actor address in GraphServiceActor
#[derive(Message)]
#[rtype(result = "()")]
pub struct StoreGPUComputeAddress {
    // Changed to GPUManagerActor for proper message routing
    pub addr: Option<Addr<crate::actors::gpu::GPUManagerActor>>,
}

// Message to get the ForceComputeActor address from GPUManagerActor
#[derive(Message)]
#[rtype(result = "Result<Addr<crate::actors::gpu::ForceComputeActor>, String>")]
pub struct GetForceComputeActor;

// Message to initialize GPU connection after system startup
#[derive(Message)]
#[rtype(result = "()")]
pub struct InitializeGPUConnection {
    pub gpu_manager: Option<Addr<crate::actors::GPUManagerActor>>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateGPUGraphData {
    pub graph: std::sync::Arc<ModelsGraphData>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateGPUPositions {
    pub positions: Vec<(f32, f32, f32)>, // (x, y, z) positions
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

#[derive(Debug, Clone, MessageResponse)]
pub struct GPUStatus {
    pub is_initialized: bool,
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
#[rtype(result = "Result<PhysicsStats, String>")]
pub struct GetPhysicsStats;

#[derive(Message)]
#[rtype(result = "Result<serde_json::Value, String>")]
pub struct GetGPUMetrics;

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

// SSSP (Single-Source Shortest Path) Message
/// Compute shortest paths from a source node using GPU SSSP
#[derive(Message)]
#[rtype(result = "Result<PathfindingResult, String>")]
pub struct ComputeShortestPaths {
    pub source_node_id: u32,
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

// GPU Position Upload Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UploadPositions {
    pub positions_x: Vec<f32>,
    pub positions_y: Vec<f32>,
    pub positions_z: Vec<f32>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UploadConstraintsToGPU {
    pub constraint_data: Vec<crate::models::constraints::ConstraintData>,
}

// Auto-balance messages
#[derive(Message)]
#[rtype(result = "Result<Vec<crate::actors::graph_actor::AutoBalanceNotification>, String>")]
pub struct GetAutoBalanceNotifications {
    pub since_timestamp: Option<i64>, // Only get notifications after this timestamp
}

// TCP Connection Actor Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct EstablishTcpConnection;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct CloseTcpConnection;

#[derive(Message)]
#[rtype(result = "()")]
pub struct RecordPollSuccess;

#[derive(Message)]
#[rtype(result = "()")]
pub struct RecordPollFailure;

// JSON-RPC Client Messages
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitializeJsonRpc;

// Graph update messages for supervision system
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RequestGraphUpdate {
    pub graph_type: crate::models::graph_types::GraphType,
    pub force_refresh: bool,
}

// ============================================================================
// Ontology Actor Messages
// ============================================================================

/// Load ontology axioms from various sources (file, URL, or direct content)
#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct LoadOntologyAxioms {
    pub source: String,         // File path, URL, or direct ontology content
    pub format: Option<String>, // "turtle", "rdf-xml", "n-triples" - auto-detect if None
}

/// Update ontology mapping configuration
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateOntologyMapping {
    pub config: crate::services::owl_validator::ValidationConfig,
}

/// Validate ontology against property graph with different modes
#[derive(Message)]
#[rtype(result = "Result<crate::services::owl_validator::ValidationReport, String>")]
pub struct ValidateOntology {
    pub ontology_id: String,
    pub graph_data: crate::services::owl_validator::PropertyGraph,
    pub mode: ValidationMode,
}

/// Validation mode for ontology validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    Quick,       // Basic constraints only
    Full,        // Complete validation with inference
    Incremental, // Only validate changes since last run
}

/// Apply inferred relationships to the graph
#[derive(Message)]
#[rtype(result = "Result<Vec<crate::services::owl_validator::RdfTriple>, String>")]
pub struct ApplyInferences {
    pub rdf_triples: Vec<crate::services::owl_validator::RdfTriple>,
    pub max_depth: Option<usize>,
}

/// Get latest ontology validation report
#[derive(Message)]
#[rtype(result = "Result<Option<crate::services::owl_validator::ValidationReport>, String>")]
pub struct GetOntologyReport {
    pub report_id: Option<String>, // Get specific report or latest if None
}

/// Get ontology system health status
#[derive(Message)]
#[rtype(result = "Result<OntologyHealth, String>")]
pub struct GetOntologyHealth;

/// Ontology system health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyHealth {
    pub loaded_ontologies: u32,
    pub cached_reports: u32,
    pub validation_queue_size: u32,
    pub last_validation: Option<DateTime<Utc>>,
    pub cache_hit_rate: f32,
    pub avg_validation_time_ms: f32,
    pub active_jobs: u32,
    pub memory_usage_mb: f32,
}

/// Clear ontology caches
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ClearOntologyCaches;

/// Get cached ontology list
#[derive(Message)]
#[rtype(result = "Result<Vec<CachedOntologyInfo>, String>")]
pub struct GetCachedOntologies;

/// Information about cached ontology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedOntologyInfo {
    pub id: String,
    pub loaded_at: DateTime<Utc>,
    pub signature: String,
    pub source: String,
    pub size_kb: u32,
    pub access_count: u32,
}

// ============================================================================
// Workspace Actor Messages
// ============================================================================

/// Get all workspaces with optional filtering and pagination
#[derive(Message)]
#[rtype(result = "Result<crate::models::workspace::WorkspaceListResponse, String>")]
pub struct GetWorkspaces {
    pub query: WorkspaceQuery,
}

/// Get a specific workspace by ID
#[derive(Message)]
#[rtype(result = "Result<Workspace, String>")]
pub struct GetWorkspace {
    pub workspace_id: String,
}

/// Create a new workspace
#[derive(Message)]
#[rtype(result = "Result<Workspace, String>")]
pub struct CreateWorkspace {
    pub request: CreateWorkspaceRequest,
}

/// Update an existing workspace
#[derive(Message)]
#[rtype(result = "Result<Workspace, String>")]
pub struct UpdateWorkspace {
    pub workspace_id: String,
    pub request: UpdateWorkspaceRequest,
}

/// Soft delete a workspace (archive it)
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct DeleteWorkspace {
    pub workspace_id: String,
}

/// Toggle favorite status of a workspace
#[derive(Message)]
#[rtype(result = "Result<bool, String>")]
pub struct ToggleFavoriteWorkspace {
    pub workspace_id: String,
}

/// Archive or unarchive a workspace
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ArchiveWorkspace {
    pub workspace_id: String,
    pub archive: bool, // true to archive, false to unarchive
}

/// Get workspace count for statistics
#[derive(Message)]
#[rtype(result = "Result<usize, String>")]
pub struct GetWorkspaceCount {
    pub filter: Option<WorkspaceFilter>,
}

/// Load workspaces from persistent storage
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct LoadWorkspaces;

/// Save workspaces to persistent storage
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SaveWorkspaces;

/// Internal message for workspace state updates via WebSocket
#[derive(Message)]
#[rtype(result = "()")]
pub struct WorkspaceStateChanged {
    pub workspace: Workspace,
    pub change_type: WorkspaceChangeType,
}

/// Type of workspace change for WebSocket notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkspaceChangeType {
    Created,
    Updated,
    Deleted,
    Favorited,
    Unfavorited,
    Archived,
    Unarchived,
}

// ============================================================================
// Ontology Actor Messages
// ============================================================================

use crate::ontology::parser::parser::LogseqPage;

/// Message to send parsed ontology data to the OntologyActor.
/// This is sent by the file handler after parsing the Logseq markdown files.
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ProcessOntologyData {
    pub pages: Vec<LogseqPage>,
}

/// Triggers a validation run of the current graph data against the loaded ontology.
#[derive(Message)]
#[rtype(result = "Result<String, String>")] // Returns a job ID
pub struct ValidateGraph {
    pub mode: ValidationMode,
}

/// Retrieves a validation report.
#[derive(Message)]
#[rtype(result = "Result<Option<String>, String>")] // Returns JSON string of the report
pub struct GetValidationReport {
    pub report_id: Option<String>, // Get specific report or latest if None
}

/// Retrieves the system health of the ontology module.
#[derive(Message)]
#[rtype(result = "Result<String, String>")] // Returns JSON string of the health status
pub struct GetOntologyHealthLegacy;

// ============================================================================
// Ontology-Physics Integration Messages
// ============================================================================

/// Apply ontology-derived constraints to the physics simulation
#[derive(Message, Clone)]
#[rtype(result = "Result<(), String>")]
pub struct ApplyOntologyConstraints {
    pub constraint_set: ConstraintSet,
    pub merge_mode: ConstraintMergeMode,
    pub graph_id: u32,
}

/// Mode for merging ontology constraints with existing constraints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstraintMergeMode {
    /// Replace all existing ontology constraints
    Replace,
    /// Merge with existing constraints, keeping both
    Merge,
    /// Add only if no conflicts with existing constraints
    AddIfNoConflict,
}

/// Enable or disable specific constraint groups
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SetConstraintGroupActive {
    pub group_name: String,
    pub active: bool,
}

/// Get statistics about currently active constraints
#[derive(Message)]
#[rtype(result = "Result<ConstraintStats, String>")]
pub struct GetConstraintStats;

/// Get statistics about ontology constraints
#[derive(Message)]
#[rtype(result = "Result<OntologyConstraintStats, String>")]
pub struct GetOntologyConstraintStats;

/// Ontology constraint statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyConstraintStats {
    pub total_axioms_processed: u32,
    pub active_ontology_constraints: u32,
    pub constraint_evaluation_count: u32,
    pub last_update_time_ms: f32,
    pub gpu_failure_count: u32,
    pub cpu_fallback_count: u32,
}

/// Statistics about constraint application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintStats {
    pub total_constraints: usize,
    pub active_constraints: usize,
    pub constraint_groups: HashMap<String, usize>,
    pub ontology_constraints: usize,
    pub user_constraints: usize,
}

// ============================================================================
// GPU Pathfinding Messages (SemanticProcessorActor)
// ============================================================================
// Note: ComputeShortestPaths already defined above at line ~1107
// Note: PathfindingResult is defined in ports::gpu_semantic_analyzer

/// Compute all-pairs shortest paths using landmark approximation
#[derive(Message)]
#[rtype(result = "Result<HashMap<(u32, u32), Vec<u32>>, String>")]
pub struct ComputeAllPairsShortestPaths {
    pub num_landmarks: Option<usize>,
}

// Re-export PathfindingResult from the port for convenience
pub use crate::ports::gpu_semantic_analyzer::PathfindingResult;
