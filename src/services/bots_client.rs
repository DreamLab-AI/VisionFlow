use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{info, error, debug, warn};
use actix::Addr;
use crate::actors::graph_service_supervisor::TransitionalGraphSupervisor;
use crate::actors::messages::UpdateBotsGraph;
use crate::utils::mcp_tcp_client::{McpTcpClient, create_mcp_client};
use crate::services::agent_visualization_protocol::{McpServerType, MultiMcpAgentStatus};
use crate::utils::mcp_connection::call_agent_spawn;

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

impl From<MultiMcpAgentStatus> for Agent {
    fn from(mcp_agent: MultiMcpAgentStatus) -> Self {
        Agent {
            id: mcp_agent.agent_id,
            name: mcp_agent.name,
            agent_type: mcp_agent.agent_type,
            status: mcp_agent.status,
            x: 0.0, // Position will be set by graph layout
            y: 0.0,
            z: 0.0,
            cpu_usage: mcp_agent.performance.cpu_usage,
            health: mcp_agent.performance.health_score,
            workload: mcp_agent.performance.activity_level / 100.0,
            memory_usage: mcp_agent.performance.memory_usage,
            created_at: Some(chrono::DateTime::from_timestamp(mcp_agent.created_at, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default()),
            age: Some((chrono::Utc::now().timestamp() - mcp_agent.created_at) as u64 * 1000),
        }
    }
}

#[derive(Clone)]
pub struct BotsClient {
    mcp_client: McpTcpClient,
    graph_service_addr: Option<Addr<TransitionalGraphSupervisor>>,
    agents: Arc<RwLock<Vec<Agent>>>,
}

impl BotsClient {
    pub fn new() -> Self {
        // Get MCP configuration from environment
        let host = std::env::var("CLAUDE_FLOW_HOST")
            .or_else(|_| std::env::var("MCP_HOST"))
            .unwrap_or_else(|_| "multi-agent-container".to_string());
        let port = std::env::var("MCP_TCP_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(9500);

        let mcp_client = create_mcp_client(&McpServerType::ClaudeFlow, &host, port);
        
        Self {
            mcp_client,
            graph_service_addr: None,
            agents: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub fn with_graph_service(graph_addr: Addr<TransitionalGraphSupervisor>) -> Self {
        let mut client = Self::new();
        client.graph_service_addr = Some(graph_addr);
        client
    }

    pub async fn connect(&self, _bots_url: &str) -> Result<()> {
        info!("Initializing MCP connection to {}:{}", self.mcp_client.host, self.mcp_client.port);

        // Test connection
        match self.mcp_client.test_connection().await {
            Ok(true) => {
                info!("✓ MCP server is reachable");
                
                // Initialize MCP session
                match self.mcp_client.initialize_session().await {
                    Ok(_) => {
                        info!("✓ MCP session initialized successfully");
                    }
                    Err(e) => {
                        warn!("Failed to initialize MCP session: {}", e);
                        // Continue anyway - some operations might still work
                    }
                }
            }
            Ok(false) => {
                warn!("MCP server is not reachable");
                return Err(anyhow::anyhow!("MCP server is not reachable"));
            }
            Err(e) => {
                error!("Failed to test MCP connection: {}", e);
                return Err(anyhow::anyhow!("Failed to test MCP connection: {}", e));
            }
        }

        // Start polling for agents
        self.start_polling().await;

        Ok(())
    }

    async fn start_polling(&self) {
        let mcp_client = self.mcp_client.clone();
        let graph_service_addr = self.graph_service_addr.clone();
        let agents = self.agents.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));

            loop {
                interval.tick().await;

                match mcp_client.query_agent_list().await {
                    Ok(mcp_agents) => {
                        if !mcp_agents.is_empty() {
                            info!("📊 Received {} agents from MCP server", mcp_agents.len());

                            // Convert MCP agents to our Agent format
                            let converted_agents: Vec<Agent> = mcp_agents
                                .into_iter()
                                .map(Agent::from)
                                .collect();

                            // Update stored agents
                            {
                                let mut agents_lock = agents.write().await;
                                *agents_lock = converted_agents.clone();
                            }

                            // Send to graph if connected
                            if let Some(ref graph_addr) = graph_service_addr {
                                info!("📨 BotsClient sending {} agents to graph", converted_agents.len());

                                // Send agents directly without conversion
                                graph_addr.do_send(UpdateBotsGraph { agents: converted_agents.clone() });
                            }
                        } else {
                            // Clear stored agents if MCP returns empty list
                            let mut agents_lock = agents.write().await;
                            if !agents_lock.is_empty() {
                                debug!("Clearing stored agents - MCP returned empty list");
                                agents_lock.clear();
                            }
                        }
                    }
                    Err(e) => {
                        debug!("Failed to query agents from MCP: {}", e);
                    }
                }
            }
        });
    }

    pub async fn get_agents_snapshot(&self) -> Result<Vec<Agent>> {
        let agents = self.agents.read().await;
        Ok(agents.clone())
    }

    pub async fn get_status(&self) -> Result<serde_json::Value> {
        let connected = self.mcp_client.test_connection().await.unwrap_or(false);
        let agents = self.agents.read().await;
        
        Ok(serde_json::json!({
            "connected": connected,
            "host": self.mcp_client.host,
            "port": self.mcp_client.port,
            "agent_count": agents.len(),
            "agents": agents.iter().map(|a| {
                serde_json::json!({
                    "id": a.id,
                    "name": a.name,
                    "type": a.agent_type,
                    "status": a.status
                })
            }).collect::<Vec<_>>()
        }))
    }

    pub async fn test_connection(&self) -> Result<bool> {
        match self.mcp_client.test_connection().await {
            Ok(result) => Ok(result),
            Err(e) => Err(anyhow::anyhow!("Connection test failed: {}", e))
        }
    }

    pub async fn spawn_agent_mcp(&self, agent_type: &str, swarm_id: &str) -> Result<String> {
        info!("Spawning MCP agent: type={}, swarm={}", agent_type, swarm_id);

        // Use the MCP connection utility to spawn an agent
        let port_str = self.mcp_client.port.to_string();
        match call_agent_spawn(&self.mcp_client.host, &port_str, agent_type, swarm_id).await {
            Ok(response) => {
                // Extract agent ID from response
                let agent_id = if let Some(content) = response.get("content") {
                    if let Some(agent_data) = content.get("agent_id") {
                        agent_data.as_str().unwrap_or("unknown").to_string()
                    } else if let Some(result) = content.get("result") {
                        result.as_str().unwrap_or("mcp_agent").to_string()
                    } else {
                        format!("mcp_{}_{}", agent_type, swarm_id)
                    }
                } else {
                    format!("mcp_{}_{}", agent_type, swarm_id)
                };

                info!("Successfully spawned MCP agent {} of type {}", agent_id, agent_type);
                Ok(agent_id)
            }
            Err(e) => {
                error!("Failed to spawn MCP agent: {}", e);
                Err(anyhow::anyhow!("MCP agent spawn failed: {}", e))
            }
        }
    }
}