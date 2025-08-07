use actix::prelude::*;
use actix::fut;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use log::{info, error, warn, debug};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Agent Control System TCP Client
/// Connects to the agent-control-system running in the agent Docker container
#[derive(Clone)]
pub struct AgentControlClient {
    addr: String,
    message_id: Arc<Mutex<u64>>,
}

impl AgentControlClient {
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            message_id: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn connect(&self) -> Result<TcpStream, Box<dyn std::error::Error + Send + Sync>> {
        info!("Connecting to Agent Control System at {}", self.addr);
        let stream = TcpStream::connect(&self.addr).await?;
        Ok(stream)
    }

    async fn send_request_on_stream(
        &self, 
        stream: TcpStream, 
        method: &str, 
        params: Value
    ) -> Result<(TcpStream, Value), Box<dyn std::error::Error + Send + Sync>> {
        let mut message_id = self.message_id.lock().await;
        *message_id += 1;
        let msg_id = *message_id;
        drop(message_id);

        let request = json!({
            "jsonrpc": "2.0",
            "id": msg_id.to_string(),
            "method": method,
            "params": params
        });

        // Split the stream for reading and writing
        let (read_half, mut write_half) = stream.into_split();

        // Send request
        let msg = format!("{}\n", request);
        debug!("Sending request: {}", msg);
        write_half.write_all(msg.as_bytes()).await?;
        write_half.flush().await?;

        // Read response
        let mut reader = BufReader::new(read_half);
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        debug!("Received response: {}", line);

        let response: Value = serde_json::from_str(&line)?;

        // Reunite the stream
        let stream = reader.into_inner().reunite(write_half)?;

        // Check for error
        if let Some(error) = response.get("error") {
            return Err(format!("RPC Error: {}", error).into());
        }

        Ok((stream, response.get("result").cloned().unwrap_or(Value::Null)))
    }

    pub async fn initialize(&self, stream: TcpStream) -> Result<(TcpStream, Value), Box<dyn std::error::Error + Send + Sync>> {
        self.send_request_on_stream(stream, "initialize", json!({
            "protocolVersion": "0.1.0",
            "clientInfo": {
                "name": "rust-backend",
                "version": "1.0.0"
            }
        })).await
    }

    pub async fn initialize_swarm(
        &self, 
        stream: TcpStream, 
        topology: &str, 
        agent_types: Vec<String>
    ) -> Result<(TcpStream, Value), Box<dyn std::error::Error + Send + Sync>> {
        self.send_request_on_stream(stream, "tools/call", json!({
            "name": "swarm.initialize",
            "arguments": {
                "topology": topology,
                "maxAgents": 20,
                "agentTypes": agent_types
            }
        })).await
    }

    pub async fn get_visualization_snapshot(&self, stream: TcpStream) -> Result<(TcpStream, VisualizationSnapshot), Box<dyn std::error::Error + Send + Sync>> {
        let (stream, result) = self.send_request_on_stream(stream, "tools/call", json!({
            "name": "visualization.snapshot",
            "arguments": {
                "includePositions": true,
                "includeConnections": true
            }
        })).await?;

        Ok((stream, serde_json::from_value(result)?))
    }

    pub async fn get_all_agents(&self, stream: TcpStream) -> Result<(TcpStream, Vec<Agent>), Box<dyn std::error::Error + Send + Sync>> {
        let (stream, result) = self.send_request_on_stream(stream, "agents/list", json!({})).await?;
        
        // Parse the agents from the result
        let agents = if let Some(agents_value) = result.get("agents") {
            serde_json::from_value(agents_value.clone())?
        } else {
            Vec::new()
        };
        
        Ok((stream, agents))
    }

    pub async fn get_system_metrics(&self, stream: TcpStream) -> Result<(TcpStream, SystemMetrics), Box<dyn std::error::Error + Send + Sync>> {
        let (stream, result) = self.send_request_on_stream(stream, "tools/call", json!({
            "name": "metrics.get",
            "arguments": {
                "includeAgents": true,
                "includePerformance": true
            }
        })).await?;

        Ok((stream, serde_json::from_value(result)?))
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
    stream: Option<TcpStream>,
}

impl AgentControlActor {
    pub fn new(addr: String) -> Self {
        Self {
            client: AgentControlClient::new(addr),
            stream: None,
        }
    }
}

impl Actor for AgentControlActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("AgentControlActor started");

        // Connect to agent control system
        let client = self.client.clone();
        let fut = async move {
            match client.connect().await {
                Ok(stream) => {
                    match client.initialize(stream).await {
                        Ok((stream, _)) => Ok(stream),
                        Err(e) => Err(e as Box<dyn std::error::Error + Send + Sync>)
                    }
                }
                Err(e) => Err(e as Box<dyn std::error::Error + Send + Sync>)
            }
        }
        .into_actor(self)
        .then(|result: Result<TcpStream, Box<dyn std::error::Error + Send + Sync>>, actor, _ctx| {
            match result {
                Ok(stream) => {
                    info!("Successfully connected to Agent Control System");
                    actor.stream = Some(stream);
                }
                Err(e) => {
                    error!("Failed to connect to Agent Control System: {}", e);
                    // Don't stop the actor - we can reconnect on demand
                    warn!("AgentControlActor will attempt to reconnect on first request");
                }
            }
            fut::ready(())
        });
        ctx.spawn(fut);
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

// Message handlers - FIXED to actually use the TCP connection
impl Handler<InitializeSwarm> for AgentControlActor {
    type Result = ResponseFuture<Result<Value, String>>;

    fn handle(&mut self, msg: InitializeSwarm, ctx: &mut Self::Context) -> Self::Result {
        let client = self.client.clone();
        let stream = self.stream.take();
        
        let actor_addr = ctx.address();
        
        Box::pin(async move {
            // Try to ensure connection first
            if stream.is_none() {
                match client.connect().await {
                    Ok(new_stream) => {
                        match client.initialize(new_stream).await {
                            Ok((initialized_stream, _)) => {
                                match client.initialize_swarm(initialized_stream, &msg.topology, msg.agent_types).await {
                                    Ok((stream, result)) => {
                                        // Store the stream back
                                        actor_addr.do_send(StoreStream { stream });
                                        Ok(result)
                                    }
                                    Err(e) => Err(e.to_string())
                                }
                            }
                            Err(e) => Err(format!("Failed to initialize connection: {}", e))
                        }
                    }
                    Err(e) => Err(format!("Failed to connect: {}", e))
                }
            } else {
                match client.initialize_swarm(stream.unwrap(), &msg.topology, msg.agent_types).await {
                    Ok((stream, result)) => {
                        // Store the stream back
                        actor_addr.do_send(StoreStream { stream });
                        Ok(result)
                    }
                    Err(e) => Err(e.to_string())
                }
            }
        })
    }
}

impl Handler<GetAllAgents> for AgentControlActor {
    type Result = ResponseFuture<Result<Vec<Agent>, String>>;

    fn handle(&mut self, _msg: GetAllAgents, ctx: &mut Self::Context) -> Self::Result {
        let client = self.client.clone();
        let stream = self.stream.take();
        let actor_addr = ctx.address();
        
        Box::pin(async move {
            if stream.is_none() {
                match client.connect().await {
                    Ok(new_stream) => {
                        match client.initialize(new_stream).await {
                            Ok((initialized_stream, _)) => {
                                match client.get_all_agents(initialized_stream).await {
                                    Ok((stream, agents)) => {
                                        actor_addr.do_send(StoreStream { stream });
                                        Ok(agents)
                                    }
                                    Err(e) => Err(e.to_string())
                                }
                            }
                            Err(e) => Err(format!("Failed to initialize connection: {}", e))
                        }
                    }
                    Err(e) => Err(format!("Failed to connect: {}", e))
                }
            } else {
                match client.get_all_agents(stream.unwrap()).await {
                    Ok((stream, agents)) => {
                        actor_addr.do_send(StoreStream { stream });
                        Ok(agents)
                    }
                    Err(e) => Err(e.to_string())
                }
            }
        })
    }
}

impl Handler<GetVisualizationSnapshot> for AgentControlActor {
    type Result = ResponseFuture<Result<VisualizationSnapshot, String>>;

    fn handle(&mut self, _msg: GetVisualizationSnapshot, ctx: &mut Self::Context) -> Self::Result {
        let client = self.client.clone();
        let stream = self.stream.take();
        let actor_addr = ctx.address();
        
        Box::pin(async move {
            if stream.is_none() {
                match client.connect().await {
                    Ok(new_stream) => {
                        match client.initialize(new_stream).await {
                            Ok((initialized_stream, _)) => {
                                match client.get_visualization_snapshot(initialized_stream).await {
                                    Ok((stream, snapshot)) => {
                                        actor_addr.do_send(StoreStream { stream });
                                        Ok(snapshot)
                                    }
                                    Err(e) => Err(e.to_string())
                                }
                            }
                            Err(e) => Err(format!("Failed to initialize connection: {}", e))
                        }
                    }
                    Err(e) => Err(format!("Failed to connect: {}", e))
                }
            } else {
                match client.get_visualization_snapshot(stream.unwrap()).await {
                    Ok((stream, snapshot)) => {
                        actor_addr.do_send(StoreStream { stream });
                        Ok(snapshot)
                    }
                    Err(e) => Err(e.to_string())
                }
            }
        })
    }
}

impl Handler<GetSystemMetrics> for AgentControlActor {
    type Result = ResponseFuture<Result<SystemMetrics, String>>;

    fn handle(&mut self, _msg: GetSystemMetrics, ctx: &mut Self::Context) -> Self::Result {
        let client = self.client.clone();
        let stream = self.stream.take();
        let actor_addr = ctx.address();
        
        Box::pin(async move {
            if stream.is_none() {
                match client.connect().await {
                    Ok(new_stream) => {
                        match client.initialize(new_stream).await {
                            Ok((initialized_stream, _)) => {
                                match client.get_system_metrics(initialized_stream).await {
                                    Ok((stream, metrics)) => {
                                        actor_addr.do_send(StoreStream { stream });
                                        Ok(metrics)
                                    }
                                    Err(e) => Err(e.to_string())
                                }
                            }
                            Err(e) => Err(format!("Failed to initialize connection: {}", e))
                        }
                    }
                    Err(e) => Err(format!("Failed to connect: {}", e))
                }
            } else {
                match client.get_system_metrics(stream.unwrap()).await {
                    Ok((stream, metrics)) => {
                        actor_addr.do_send(StoreStream { stream });
                        Ok(metrics)
                    }
                    Err(e) => Err(e.to_string())
                }
            }
        })
    }
}

// Internal message to store the stream back
#[derive(Message)]
#[rtype(result = "()")]
struct StoreStream {
    stream: TcpStream,
}

impl Handler<StoreStream> for AgentControlActor {
    type Result = ();

    fn handle(&mut self, msg: StoreStream, _ctx: &mut Self::Context) -> Self::Result {
        self.stream = Some(msg.stream);
    }
}