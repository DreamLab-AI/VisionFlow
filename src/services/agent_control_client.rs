use actix::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use log::{info, error, warn, debug};

/// Agent Control System TCP Client
/// Connects to the agent-control-system running in the agent Docker container
pub struct AgentControlClient {
    addr: String,
    stream: Option<TcpStream>,
    message_id: u64,
}

impl AgentControlClient {
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            stream: None,
            message_id: 0,
        }
    }
    
    pub async fn connect(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Connecting to Agent Control System at {}", self.addr);
        let stream = TcpStream::connect(&self.addr).await?;
        self.stream = Some(stream);
        
        // Initialize protocol
        let response = self.initialize().await?;
        info!("Connected to Agent Control System: {:?}", response);
        
        Ok(())
    }
    
    pub async fn disconnect(&mut self) {
        if let Some(mut stream) = self.stream.take() {
            let _ = stream.shutdown().await;
        }
    }
    
    async fn send_request(&mut self, method: &str, params: Value) -> Result<Value, Box<dyn std::error::Error>> {
        let stream = self.stream.as_mut().ok_or("Not connected")?;
        
        self.message_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.message_id.to_string(),
            "method": method,
            "params": params
        });
        
        // Send request
        let msg = format!("{}\n", request);
        stream.write_all(msg.as_bytes()).await?;
        stream.flush().await?;
        
        // Read response
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        
        let response: Value = serde_json::from_str(&line)?;
        
        // Check for error
        if let Some(error) = response.get("error") {
            return Err(format!("RPC Error: {}", error).into());
        }
        
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }
    
    async fn initialize(&mut self) -> Result<Value, Box<dyn std::error::Error>> {
        self.send_request("initialize", json!({
            "protocolVersion": "0.1.0",
            "clientInfo": {
                "name": "rust-backend",
                "version": "1.0.0"
            }
        })).await
    }
    
    pub async fn initialize_swarm(&mut self, topology: &str, agent_types: Vec<&str>) -> Result<Value, Box<dyn std::error::Error>> {
        self.send_request("tools/call", json!({
            "name": "swarm.initialize",
            "arguments": {
                "topology": topology,
                "maxAgents": 20,
                "agentTypes": agent_types
            }
        })).await
    }
    
    pub async fn get_visualization_snapshot(&mut self) -> Result<VisualizationSnapshot, Box<dyn std::error::Error>> {
        let result = self.send_request("tools/call", json!({
            "name": "visualization.snapshot",
            "arguments": {
                "includePositions": true,
                "includeConnections": true
            }
        })).await?;
        
        Ok(serde_json::from_value(result)?)
    }
    
    pub async fn get_all_agents(&mut self) -> Result<Vec<Agent>, Box<dyn std::error::Error>> {
        let result = self.send_request("agents/list", json!({})).await?;
        
        if let Some(agents) = result.get("agents") {
            Ok(serde_json::from_value(agents.clone())?)
        } else {
            Ok(Vec::new())
        }
    }
    
    pub async fn get_metrics(&mut self) -> Result<SystemMetrics, Box<dyn std::error::Error>> {
        let result = self.send_request("tools/call", json!({
            "name": "metrics.get",
            "arguments": {
                "includeAgents": true,
                "includePerformance": true
            }
        })).await?;
        
        Ok(serde_json::from_value(result)?)
    }
}

// Data structures matching the Agent Control System

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    #[serde(rename = "type")]
    pub agent_type: String,
    pub name: String,
    pub status: String,
    pub health: f64,
    pub capabilities: Vec<String>,
    #[serde(rename = "swarmId")]
    pub swarm_id: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "lastActivity")]
    pub last_activity: String,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    #[serde(rename = "tasksCompleted")]
    pub tasks_completed: u32,
    #[serde(rename = "tasksActive")]
    pub tasks_active: u32,
    #[serde(rename = "successRate")]
    pub success_rate: f64,
    #[serde(rename = "cpuUsage")]
    pub cpu_usage: f64,
    #[serde(rename = "memoryUsage")]
    pub memory_usage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSnapshot {
    pub timestamp: String,
    #[serde(rename = "agentCount")]
    pub agent_count: u32,
    pub positions: HashMap<String, Position>,
    pub connections: Vec<Connection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub id: String,
    pub from: String,
    pub to: String,
    #[serde(rename = "messageCount")]
    pub message_count: u32,
    #[serde(rename = "lastActivity")]
    pub last_activity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: String,
    pub system: SystemInfo,
    pub agents: Option<AgentStats>,
    pub performance: Option<PerformanceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub uptime: f64,
    #[serde(rename = "memoryUsage")]
    pub memory_usage: MemoryUsage,
    #[serde(rename = "cpuUsage")]
    pub cpu_usage: CpuUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub rss: u64,
    #[serde(rename = "heapTotal")]
    pub heap_total: u64,
    #[serde(rename = "heapUsed")]
    pub heap_used: u64,
    pub external: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsage {
    pub user: u64,
    pub system: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub total: u32,
    #[serde(rename = "byType")]
    pub by_type: HashMap<String, u32>,
    #[serde(rename = "byStatus")]
    pub by_status: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub fps: f64,
    #[serde(rename = "updateTime")]
    pub update_time: f64,
    #[serde(rename = "nodeCount")]
    pub node_count: u32,
}

// Actor for managing the client connection
pub struct AgentControlActor {
    client: AgentControlClient,
}

impl AgentControlActor {
    pub fn new(addr: String) -> Self {
        Self {
            client: AgentControlClient::new(addr),
        }
    }
}

impl Actor for AgentControlActor {
    type Context = Context<Self>;
    
    fn started(&mut self, ctx: &mut Self::Context) {
        info!("AgentControlActor started");
        
        // Connect to agent control system
        let fut = async move {
            self.client.connect().await
        }.into_actor(self);
        
        ctx.spawn(fut.map(|result, _actor, ctx| {
            match result {
                Ok(()) => {
                    info!("Successfully connected to Agent Control System");
                }
                Err(e) => {
                    error!("Failed to connect to Agent Control System: {}", e);
                    ctx.stop();
                }
            }
        }));
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("AgentControlActor stopped");
    }
}

// Messages
#[derive(Message)]
#[rtype(result = "Result<Value, String>")]
pub struct InitializeSwarm {
    pub topology: String,
    pub agent_types: Vec<String>,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<Agent>, String>")]
pub struct GetAllAgents;

#[derive(Message)]
#[rtype(result = "Result<VisualizationSnapshot, String>")]
pub struct GetVisualizationSnapshot;

#[derive(Message)]
#[rtype(result = "Result<SystemMetrics, String>")]
pub struct GetSystemMetrics;

// Message handlers
impl Handler<InitializeSwarm> for AgentControlActor {
    type Result = ResponseFuture<Result<Value, String>>;
    
    fn handle(&mut self, msg: InitializeSwarm, _ctx: &mut Self::Context) -> Self::Result {
        Box::pin(async move {
            let agent_types: Vec<&str> = msg.agent_types.iter().map(|s| s.as_str()).collect();
            self.client.initialize_swarm(&msg.topology, agent_types)
                .await
                .map_err(|e| e.to_string())
        }.into_actor(self))
    }
}

impl Handler<GetAllAgents> for AgentControlActor {
    type Result = ResponseFuture<Result<Vec<Agent>, String>>;
    
    fn handle(&mut self, _msg: GetAllAgents, _ctx: &mut Self::Context) -> Self::Result {
        Box::pin(async move {
            self.client.get_all_agents()
                .await
                .map_err(|e| e.to_string())
        }.into_actor(self))
    }
}

impl Handler<GetVisualizationSnapshot> for AgentControlActor {
    type Result = ResponseFuture<Result<VisualizationSnapshot, String>>;
    
    fn handle(&mut self, _msg: GetVisualizationSnapshot, _ctx: &mut Self::Context) -> Self::Result {
        Box::pin(async move {
            self.client.get_visualization_snapshot()
                .await
                .map_err(|e| e.to_string())
        }.into_actor(self))
    }
}

impl Handler<GetSystemMetrics> for AgentControlActor {
    type Result = ResponseFuture<Result<SystemMetrics, String>>;
    
    fn handle(&mut self, _msg: GetSystemMetrics, _ctx: &mut Self::Context) -> Self::Result {
        Box::pin(async move {
            self.client.get_metrics()
                .await
                .map_err(|e| e.to_string())
        }.into_actor(self))
    }
}