//! Agent/bot/swarm-domain messages: Claude Flow agent lifecycle, swarm orchestration,
//! neural network, memory persistence, performance monitoring, and MCP tool responses.

use actix::prelude::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::models::graph::GraphData;
use crate::types::claude_flow::AgentStatus;

// ---------------------------------------------------------------------------
// Voice Command and Agent Spawning
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct SpawnAgentCommand {
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub session_id: String,
}

// ---------------------------------------------------------------------------
// Claude Flow Actor Messages
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateAgentCache {
    pub agents: Vec<AgentStatus>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUpdate {
    pub agent_id: String,
    pub status: String,
    pub timestamp: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Enhanced MCP Tool Messages for Hive Mind Swarm
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Neural Network MCP Tool Messages
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Memory & Persistence MCP Tool Messages
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Analysis & Monitoring MCP Tool Messages
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Polling and retry messages
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MCP Response Data Structures
// ---------------------------------------------------------------------------

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
    pub pattern_type: String,
    pub participants: Vec<String>,
    pub status: String,
    pub progress: f32,
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
    pub severity: f32,
    pub description: String,
    pub suggested_fix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemMetrics {
    pub active_agents: u32,
    pub message_rate: f32,
    pub average_latency: f32,
    pub error_rate: f32,
    pub network_health: f32,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub gpu_usage: Option<f32>,
}
