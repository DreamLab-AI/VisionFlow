//! Types for claude flow integration via TCP
//! These replace the local claude_flow module types

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub agent_id: String,
    pub profile: AgentProfile,
    pub status: String,
    pub active_tasks_count: u32,
    pub completed_tasks_count: u32,
    pub failed_tasks_count: u32,
    pub success_rate: f32,
    pub timestamp: DateTime<Utc>,
    pub current_task: Option<TaskReference>,
    
    // Additional fields needed by existing code
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub health: f32,
    pub activity: f32,
    pub tasks_active: u32,
    pub performance_metrics: PerformanceMetrics,
    pub token_usage: TokenUsage,
    pub swarm_id: Option<String>,
    pub agent_mode: Option<String>,
    pub parent_queen_id: Option<String>,
    pub processing_logs: Option<Vec<String>>,
    pub total_execution_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tasks_completed: u32,
    pub success_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub total: u64,
    pub token_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub name: String,
    pub agent_type: AgentType,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    Coordinator,
    Researcher,
    Coder,
    Analyst,
    Architect,
    Tester,
    Reviewer,
    Optimizer,
    Documenter,
}

impl ToString for AgentType {
    fn to_string(&self) -> String {
        match self {
            AgentType::Coordinator => "coordinator".to_string(),
            AgentType::Researcher => "researcher".to_string(),
            AgentType::Coder => "coder".to_string(),
            AgentType::Analyst => "analyst".to_string(),
            AgentType::Architect => "architect".to_string(),
            AgentType::Tester => "tester".to_string(),
            AgentType::Reviewer => "reviewer".to_string(),
            AgentType::Optimizer => "optimizer".to_string(),
            AgentType::Documenter => "documenter".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskReference {
    pub task_id: String,
    pub description: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
}

// TCP Client for communicating with external claude flow service
#[derive(Clone)]
pub struct ClaudeFlowClient {
    host: String,
    port: u16,
    // Remove transport field - we're using direct TCP now
}

impl ClaudeFlowClient {
    pub fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }

    pub async fn connect(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement TCP connection
        log::info!("Connecting to Claude Flow at {}:{}", self.host, self.port);
        Ok(())
    }

    pub async fn get_agent_statuses(&self) -> Result<Vec<AgentStatus>, Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement TCP request to get agent statuses
        log::warn!("TCP client not fully implemented yet");
        Ok(vec![])
    }

    // Method to send MCP request via TCP
    pub async fn send_mcp_request(&self, request: &McpRequest) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement actual TCP communication
        log::warn!("send_mcp_request not fully implemented yet");
        log::debug!("Would send request: {:?}", request);
        Ok(serde_json::Value::Null)
    }
}

// Error types
#[derive(Debug)]
pub enum ConnectorError {
    NotConnected,
    NetworkError(String),
    ParseError(String),
}

impl std::fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectorError::NotConnected => write!(f, "Not connected to Claude Flow service"),
            ConnectorError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ConnectorError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ConnectorError {}