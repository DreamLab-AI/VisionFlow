use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tokio_tungstenite::tungstenite::Message;
use futures_util::{StreamExt, SinkExt};
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{info, error, debug, warn};
use actix::Addr;
use crate::actors::graph_actor::GraphServiceActor;
use crate::actors::messages::UpdateBotsGraph;
use crate::types::claude_flow::{AgentStatus, AgentProfile, AgentType, TokenUsage, PerformanceMetrics};
use crate::types::mcp_responses::{McpResponse, McpContentResult, AgentListResponse};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotsUpdate {
    pub agents: Vec<Agent>,
    #[serde(default)]
    pub metrics: BotsMetrics,
    #[serde(default)]
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub agent_type: String,
    pub status: String,
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    #[serde(default)]
    pub z: f32,
    #[serde(default = "default_cpu_usage")]
    pub cpu_usage: f32,
    #[serde(default = "default_health")]
    pub health: f32,
    #[serde(default = "default_workload")]
    pub workload: f32,
    #[serde(default = "default_memory_usage")]
    pub memory_usage: f32,
    #[serde(rename = "createdAt", skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>, // ISO 8601 timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<u64>, // milliseconds
}

fn default_cpu_usage() -> f32 { 50.0 }
fn default_health() -> f32 { 90.0 }
fn default_workload() -> f32 { 0.7 }
fn default_memory_usage() -> f32 { 30.0 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BotsMetrics {
    pub total_tokens: u64,
    pub active_agents: u32,
    pub tasks_completed: u32,
}

#[derive(Clone)]
pub struct BotsClient {
    updates: Arc<RwLock<Option<BotsUpdate>>>,
    graph_service_addr: Option<Addr<GraphServiceActor>>,
}

impl BotsClient {
    pub fn new() -> Self {
        Self {
            updates: Arc::new(RwLock::new(None)),
            graph_service_addr: None,
        }
    }
    
    pub fn with_graph_service(graph_addr: Addr<GraphServiceActor>) -> Self {
        Self {
            updates: Arc::new(RwLock::new(None)),
            graph_service_addr: Some(graph_addr),
        }
    }

    pub async fn connect(&self, bots_url: &str) -> Result<()> {
        info!("Attempting to connect to bots orchestrator at: {}", bots_url);

        let url = url::Url::parse(bots_url)?;
        info!("Parsed URL: {:?}", url);

        match connect_async(url.as_str()).await {
            Ok((ws_stream, response)) => {
                info!("Successfully connected to bots orchestrator at {}", bots_url);
                info!("WebSocket response status: {:?}", response.status());

                // FIXED: Try to get bots data using proper MCP protocol
                let subscribe_msg = serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": {
                            "major": 2024,
                            "minor": 11,
                            "patch": 5
                        },
                        "clientInfo": {
                            "name": "VisionFlow-BotsClient",
                            "version": "1.0.0"
                        },
                        "capabilities": {
                            "tools": {
                                "listChanged": true
                            }
                        }
                    },
                    "id": "init-1"
                });

                self.handle_connection(ws_stream, subscribe_msg).await
            }
            Err(e) => {
                error!("Failed to connect to bots orchestrator at {}: {:?}", bots_url, e);
                Err(e.into())
            }
        }
    }

    async fn handle_connection(
        &self,
        ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
        subscribe_msg: serde_json::Value,
    ) -> Result<()> {
        let (mut write, mut read) = ws_stream.split();

        // FIXED: Send MCP initialization
        let init_text = subscribe_msg.to_string();
        info!("Sending MCP initialization: {}", init_text);
        write.send(Message::Text(init_text)).await?;
        info!("Successfully sent MCP initialization to bots orchestrator");
        
        // Wait a moment for initialization to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        // Send initial agent list request
        let agent_list_request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "agent_list",
            "params": {
                "filter": "all"
            },
            "id": "initial-agent-list"
        });
        
        info!("Sending initial agent list request");
        write.send(Message::Text(agent_list_request.to_string())).await?;

        // Clone for the read task
        let updates = self.updates.clone();
        let graph_service_addr = self.graph_service_addr.clone();

        // Spawn read task
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        info!("Received message from bots orchestrator: {}", text);

                        // Parse and process bots updates
                        match serde_json::from_str::<serde_json::Value>(&text) {
                            Ok(json) => {
                                // Check for different message types
                                if let Some(method) = json.get("method").and_then(|m| m.as_str()) {
                                    info!("Received method notification: {}", method);

                                    // Handle connection established
                                    if method == "connection/established" {
                                        info!("Connection established with claude-flow server");
                                        // Connection is ready, we can now try other methods
                                    }
                                }

                                // Check for error responses
                                if let Some(error) = json.get("error") {
                                    error!("Received error from server: {:?}", error);
                                    // If we get an unknown method error, try to parse any data that might be available
                                    if let Some(error_str) = error.as_str() {
                                        if error_str.contains("Unknown method") {
                                            info!("Server doesn't recognize our method, it might be a simple echo/test server");
                                        }
                                    }
                                }

                                // Try to parse as BotsUpdate directly if it has the right fields
                                if json.get("agents").is_some() || json.get("bots").is_some() {
                                    match serde_json::from_value::<BotsUpdate>(json.clone()) {
                                        Ok(update) => {
                                            info!("Parsed BotsUpdate directly from message with {} agents", update.agents.len());
                                            let mut lock = updates.write().await;
                                            *lock = Some(update);
                                        }
                                        Err(e) => {
                                            debug!("Failed to parse as BotsUpdate: {:?}", e);
                                        }
                                    }
                                }

                                // Try to parse as MCP response first
                                if let Ok(mcp_response) = serde_json::from_value::<McpResponse<McpContentResult>>(json.clone()) {
                                    match mcp_response.into_result() {
                                        Ok(content_result) => {
                                            info!("Successfully parsed MCP response structure");
                                            
                                            // Extract agent data using type-safe parsing
                                            match content_result.extract_data::<AgentListResponse>() {
                                                Ok(agent_list) => {
                                                    if !agent_list.agents.is_empty() {
                                                        let update = BotsUpdate {
                                                            agents: agent_list.agents.clone(),
                                                            metrics: BotsMetrics::default(),
                                                            timestamp: std::time::SystemTime::now()
                                                                .duration_since(std::time::UNIX_EPOCH)
                                                                .unwrap_or_default()
                                                                .as_secs(),
                                                        };
                                                        info!("Successfully parsed {} agents from MCP response", update.agents.len());
                                                        for agent in &update.agents {
                                                            debug!("Agent: {} ({}) - status: {}", agent.name, agent.agent_type, agent.status);
                                                        }

                                                        // CRITICAL FIX: Send agents to graph
                                                        if let Some(ref graph_addr) = graph_service_addr {
                                                            info!("📨 BotsClient sending {} agents to graph", update.agents.len());
                                                            graph_addr.do_send(UpdateBotsGraph {
                                                                agents: update.agents.clone()
                                                                    .into_iter()
                                                                    .map(|a| a.into())
                                                                    .collect()
                                                            });
                                                        }
                                                        
                                                        let mut lock = updates.write().await;
                                                        *lock = Some(update);
                                                        continue;
                                                    }
                                                }
                                                Err(e) => {
                                                    debug!("Failed to extract agent data from MCP response: {}", e);
                                                    // Fall through to legacy parsing
                                                }
                                            }
                                        }
                                        Err(mcp_error) => {
                                            warn!("MCP response returned error: {}", mcp_error.message);
                                            // Fall through to legacy parsing
                                        }
                                    }
                                }

                                // Legacy parsing for non-MCP responses
                                if let Some(result) = json.get("result") {
                                    info!("Processing result field from bots response (legacy format)");
                                    debug!("Raw result JSON: {}", serde_json::to_string_pretty(result).unwrap_or_default());

                                    // MCP responses come directly as result objects, not wrapped
                                    let actual_result = result.clone();

                                    // Handle different possible result structures
                                    if let Some(result_obj) = actual_result.as_object() {
                                        // Check if result contains bots data directly
                                        if result_obj.contains_key("agents") || result_obj.contains_key("bots") {
                                            // Try to parse the agents manually to handle timestamp format issues
                                            let mut agents = Vec::new();
                                            if let Some(agents_array) = result_obj.get("agents").and_then(|a| a.as_array()) {
                                                for agent_val in agents_array {
                                                    if let Ok(agent) = serde_json::from_value::<Agent>(agent_val.clone()) {
                                                        agents.push(agent);
                                                    }
                                                }
                                            }
                                            
                                            if !agents.is_empty() {
                                                let update = BotsUpdate {
                                                    agents: agents.clone(),
                                                    metrics: BotsMetrics::default(),
                                                    timestamp: std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap_or_default()
                                                        .as_secs(),
                                                };
                                                info!("Successfully constructed BotsUpdate with {} agents", update.agents.len());
                                                for agent in &update.agents {
                                                    debug!("Agent: {} ({}) - status: {}", agent.name, agent.agent_type, agent.status);
                                                }
                                                let mut lock = updates.write().await;
                                                *lock = Some(update);
                                            } else {
                                                // Fall back to direct parsing
                                                match serde_json::from_value::<BotsUpdate>(actual_result.clone()) {
                                                    Ok(update) => {
                                                        info!("Successfully parsed BotsUpdate with {} agents", update.agents.len());
                                                        for agent in &update.agents {
                                                            debug!("Agent: {} ({}) - status: {}", agent.name, agent.agent_type, agent.status);
                                                        }
                                                        let mut lock = updates.write().await;
                                                        *lock = Some(update);
                                                    }
                                                    Err(e) => {
                                                        error!("Failed to parse BotsUpdate from result: {:?}", e);
                                                        debug!("Result JSON was: {:?}", actual_result);
                                                    }
                                                }
                                            }
                                        } else if let Some(bots_data) = result_obj.get("bots") {
                                            // Try parsing from nested bots field
                                            match serde_json::from_value::<BotsUpdate>(bots_data.clone()) {
                                                Ok(update) => {
                                                    info!("Successfully parsed BotsUpdate from result.bots with {} agents", update.agents.len());
                                                    let mut lock = updates.write().await;
                                                    *lock = Some(update);
                                                }
                                                Err(e) => {
                                                    error!("Failed to parse BotsUpdate from result.bots: {:?}", e);
                                                }
                                            }
                                        } else {
                                            // Try to construct BotsUpdate from various formats
                                            let mut agents = Vec::new();

                                            // Check for agents array at root level
                                            if let Some(agents_array) = result_obj.get("agents").and_then(|a| a.as_array()) {
                                                for agent_val in agents_array {
                                                    if let Ok(agent) = serde_json::from_value::<Agent>(agent_val.clone()) {
                                                        agents.push(agent);
                                                    }
                                                }
                                            }

                                            // Check for activeAgents field
                                            if let Some(active_agents) = result_obj.get("activeAgents").and_then(|a| a.as_array()) {
                                                for agent_val in active_agents {
                                                    if let Ok(agent) = serde_json::from_value::<Agent>(agent_val.clone()) {
                                                        agents.push(agent);
                                                    }
                                                }
                                            }

                                            if !agents.is_empty() {
                                                let update = BotsUpdate {
                                                    agents,
                                                    metrics: BotsMetrics::default(),
                                                    timestamp: std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap_or_default()
                                                        .as_secs(),
                                                };
                                                info!("Constructed BotsUpdate from various fields with {} agents", update.agents.len());
                                                let mut lock = updates.write().await;
                                                *lock = Some(update);
                                            } else {
                                                warn!("No agents found in result object. Keys: {:?}", result_obj.keys().collect::<Vec<_>>());
                                            }
                                        }
                                    }
                                }

                                // Also check for params field (for notifications)
                                if let Some(params) = json.get("params") {
                                    info!("Attempting to parse params as BotsUpdate");
                                    match serde_json::from_value::<BotsUpdate>(params.clone()) {
                                        Ok(update) => {
                                            info!("Successfully parsed BotsUpdate from params with {} agents", update.agents.len());
                                            let mut lock = updates.write().await;
                                            *lock = Some(update);
                                        }
                                        Err(e) => {
                                            debug!("Params is not a BotsUpdate: {:?}", e);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Failed to parse JSON from bots: {:?}", e);
                                error!("Raw text was: {}", text);
                            }
                        }
                    }
                    Ok(Message::Binary(bin)) => {
                        debug!("Received binary message: {} bytes", bin.len());
                    }
                    Ok(Message::Ping(_)) => {
                        debug!("Received ping from bots");
                    }
                    Err(e) => {
                        error!("WebSocket error from bots: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
            info!("Bots WebSocket connection closed");
        });

        // Clone updates for the polling task
        let updates_for_polling = self.updates.clone();

        // Keep connection alive and poll for bots status
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
            let mut request_id = 1;

            loop {
                interval.tick().await;

                // FIXED: Send a proper MCP request using available methods
                let status_request = if request_id % 2 == 0 {
                    // Use tools/list which is a standard MCP method
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "params": {},
                        "id": format!("tools-list-{}", request_id)
                    })
                } else {
                    // Use correct MCP tools/call format
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "agent_list",
                            "arguments": {
                                "filter": "all"
                            }
                        },
                        "id": format!("agent-list-{}", request_id)
                    })
                };

                if let Err(e) = write.send(Message::Text(status_request.to_string())).await {
                    error!("Failed to send status request to bots: {}", e);
                    break;
                }

                request_id += 1;

                // If we haven't received any real data after 10 seconds, log warning
                if request_id == 3 {
                    let lock = updates_for_polling.read().await;
                    if lock.is_none() {
                        drop(lock);
                        warn!("No bots data received from orchestrator after 15 seconds");
                    }
                }

                // Also send pings to keep connection alive
                if request_id % 6 == 0 {  // Every 30 seconds
                    if let Err(e) = write.send(Message::Ping(vec![])).await {
                        error!("Failed to send ping to bots: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    pub async fn get_latest_update(&self) -> Option<BotsUpdate> {
        self.updates.read().await.clone()
    }
}

// Implement conversion from Agent to AgentStatus
impl From<Agent> for AgentStatus {
    fn from(agent: Agent) -> Self {
        use chrono::Utc;
        
        // Parse agent type
        let agent_type = match agent.agent_type.as_str() {
            "coordinator" => AgentType::Coordinator,
            "researcher" => AgentType::Researcher,
            "coder" => AgentType::Coder,
            "analyst" => AgentType::Analyst,
            "architect" => AgentType::Architect,
            "tester" => AgentType::Tester,
            "reviewer" => AgentType::Reviewer,
            "optimizer" => AgentType::Optimizer,
            "documenter" => AgentType::Documenter,
            _ => AgentType::Coder, // Default
        };
        
        // Create agent profile
        let profile = AgentProfile {
            name: agent.name.clone(),
            agent_type: agent_type.clone(),
            capabilities: vec![], // Could be populated from agent data if available
            description: Some(format!("Bot client agent of type {:?}", agent_type)),
            version: "1.0.0".to_string(),
            tags: vec!["bot-client".to_string()],
        };
        
        // Calculate performance metrics from real agent data
        let tasks_completed = (agent.workload * 100.0) as u32; // Estimate from workload
        let performance_metrics = PerformanceMetrics {
            tasks_completed,
            success_rate: agent.health / 100.0,
        };
        
        // Calculate token usage based on agent activity
        let estimated_tokens = (agent.workload * agent.cpu_usage * 100.0) as u64;
        let token_usage = TokenUsage {
            total: estimated_tokens,
            token_rate: agent.workload * 10.0,
        };
        
        AgentStatus {
            agent_id: agent.id.clone(),
            profile,
            status: agent.status.clone(),
            active_tasks_count: (agent.workload * 10.0) as u32, // Estimate from workload
            completed_tasks_count: tasks_completed,
            failed_tasks_count: ((100.0 - agent.health) / 10.0) as u32,
            success_rate: agent.health / 100.0,
            timestamp: Utc::now(),
            current_task: None,
            cpu_usage: agent.cpu_usage,
            memory_usage: agent.memory_usage,
            health: agent.health,
            activity: agent.workload,
            tasks_active: (agent.workload * 10.0) as u32,
            performance_metrics,
            token_usage,
            swarm_id: None,
            agent_mode: Some("active".to_string()),
            parent_queen_id: None,
            processing_logs: None,
            total_execution_time: 0,
        }
    }
}

