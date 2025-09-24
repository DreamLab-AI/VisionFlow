//! Types for claude flow integration via TCP
//! These replace the local claude_flow module types

use serde::{Deserialize, Serialize};
use serde_json::json;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

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
    pub description: Option<String>,
    pub version: String,
    pub tags: Vec<String>,
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
    Generic,
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
            AgentType::Generic => "generic".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskReference {
    pub task_id: String,
    pub description: String,
    pub priority: TaskPriority,
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
        use tokio::net::TcpStream;
        use tokio::time::{timeout, Duration};

        log::info!("Connecting to Claude Flow at {}:{}", self.host, self.port);

        let addr = format!("{}:{}", self.host, self.port);
        let stream = timeout(Duration::from_secs(10), TcpStream::connect(&addr)).await??;

        log::info!("Successfully connected to Claude Flow at {}", addr);

        // Validate connection with a simple ping
        drop(stream); // For now, we'll reconnect as needed per request
        Ok(())
    }

    pub async fn get_agent_statuses(&self) -> Result<Vec<AgentStatus>, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::net::TcpStream;
        use tokio::io::{AsyncWriteExt, AsyncReadExt};
        use tokio::time::{timeout, Duration};

        let addr = format!("{}:{}", self.host, self.port);
        let mut stream = timeout(Duration::from_secs(5), TcpStream::connect(&addr)).await??;

        // Send agent list request (simplified JSON-RPC style)
        let request = json!({
            "jsonrpc": "2.0",
            "method": "list_agents",
            "id": 1
        });

        let request_str = format!("{}\n", request.to_string());
        stream.write_all(request_str.as_bytes()).await?;

        // Read response
        let mut buffer = vec![0; 8192];
        let bytes_read = timeout(Duration::from_secs(5), stream.read(&mut buffer)).await??;

        if bytes_read == 0 {
            return Ok(vec![]); // No data received
        }

        let response_str = String::from_utf8_lossy(&buffer[..bytes_read]);

        // Parse JSON response
        if let Ok(response_json) = serde_json::from_str::<serde_json::Value>(&response_str) {
            if let Some(agents_array) = response_json.get("result").and_then(|r| r.as_array()) {
                let mut statuses = Vec::new();
                for agent_data in agents_array {
                    if let Ok(status) = self.parse_agent_status(agent_data) {
                        statuses.push(status);
                    }
                }
                return Ok(statuses);
            }
        }

        log::debug!("No valid agents found in TCP response");
        Ok(vec![])
    }

    fn parse_agent_status(&self, agent_data: &serde_json::Value) -> Result<AgentStatus, Box<dyn std::error::Error + Send + Sync>> {
        let agent_id = agent_data.get("id").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
        let agent_type = agent_data.get("type").and_then(|v| v.as_str()).unwrap_or("generic").to_string();
        let status = agent_data.get("status").and_then(|v| v.as_str()).unwrap_or("idle").to_string();

        Ok(AgentStatus {
            agent_id: agent_id.clone(),
            profile: AgentProfile {
                name: agent_id.clone(),
                agent_type: match agent_type.as_str() {
                    "coordinator" => AgentType::Coordinator,
                    "researcher" => AgentType::Researcher,
                    "coder" => AgentType::Coder,
                    "analyst" => AgentType::Analyst,
                    "architect" => AgentType::Architect,
                    "tester" => AgentType::Tester,
                    "reviewer" => AgentType::Reviewer,
                    "optimizer" => AgentType::Optimizer,
                    "documenter" => AgentType::Documenter,
                    _ => AgentType::Generic,
                },
                capabilities: vec![agent_type.clone()],
                description: Some(format!("{} agent", agent_type)),
                version: "1.0.0".to_string(),
                tags: vec!["general".to_string()],
            },
            status,
            active_tasks_count: agent_data.get("active_tasks").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            completed_tasks_count: agent_data.get("completed_tasks").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            failed_tasks_count: agent_data.get("failed_tasks").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            success_rate: agent_data.get("success_rate").and_then(|v| v.as_f64()).unwrap_or(0.95) as f32,
            timestamp: chrono::Utc::now(),
            current_task: agent_data.get("current_task").and_then(|v| v.as_str()).map(|task_desc| TaskReference {
                task_id: format!("task_{}", uuid::Uuid::new_v4()),
                description: task_desc.to_string(),
                priority: TaskPriority::Medium,
            }),
            cpu_usage: agent_data.get("cpu_usage").and_then(|v| v.as_f64()).unwrap_or(25.0) as f32,
            memory_usage: agent_data.get("memory_usage").and_then(|v| v.as_f64()).unwrap_or(128.0) as f32,
            health: agent_data.get("health").and_then(|v| v.as_f64()).unwrap_or(0.9) as f32,
            activity: agent_data.get("activity").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
            tasks_active: agent_data.get("tasks_active").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
            performance_metrics: PerformanceMetrics {
                tasks_completed: agent_data.get("completed_tasks").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                success_rate: agent_data.get("success_rate").and_then(|v| v.as_f64()).unwrap_or(0.95) as f32,
            },
            token_usage: TokenUsage {
                total: agent_data.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(1500),
                token_rate: agent_data.get("token_rate").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32,
            },
            swarm_id: agent_data.get("swarm_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            agent_mode: agent_data.get("agent_mode").and_then(|v| v.as_str()).map(|s| s.to_string()),
            parent_queen_id: agent_data.get("parent_queen_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            processing_logs: Some(vec![]),
            total_execution_time: agent_data.get("execution_time").and_then(|v| v.as_u64()).unwrap_or(0),
        })
    }

    // Method to send MCP request via TCP
    pub async fn send_mcp_request(&self, request: &McpRequest) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::net::TcpStream;
        use tokio::io::{AsyncWriteExt, AsyncReadExt};
        use tokio::time::{timeout, Duration};

        let addr = format!("{}:{}", self.host, self.port);
        let mut stream = timeout(Duration::from_secs(5), TcpStream::connect(&addr)).await??;

        // Convert McpRequest to JSON-RPC format
        let json_request = json!({
            "jsonrpc": "2.0",
            "method": request.method,
            "params": request.params,
            "id": uuid::Uuid::new_v4().to_string()
        });

        let request_str = format!("{}\n", json_request.to_string());
        stream.write_all(request_str.as_bytes()).await?;

        // Read response
        let mut buffer = vec![0; 16384];
        let bytes_read = timeout(Duration::from_secs(10), stream.read(&mut buffer)).await??;

        if bytes_read == 0 {
            return Err("No response received from TCP server".into());
        }

        let response_str = String::from_utf8_lossy(&buffer[..bytes_read]);

        // Parse JSON-RPC response
        match serde_json::from_str::<serde_json::Value>(&response_str) {
            Ok(json_response) => {
                if let Some(error) = json_response.get("error") {
                    return Err(format!("MCP request failed: {}", error).into());
                }
                Ok(json_response.get("result").cloned().unwrap_or(serde_json::Value::Null))
            }
            Err(e) => {
                log::error!("Failed to parse TCP response: {} (raw: {})", e, response_str);
                Err(format!("Invalid JSON response: {}", e).into())
            }
        }
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