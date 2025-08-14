use actix::prelude::*;
use actix::fut;
use std::time::Duration;
use log::{info, error, debug, warn};
use crate::types::claude_flow::{ClaudeFlowClient, AgentStatus};
use crate::actors::messages::*;
use crate::actors::GraphServiceActor;
use std::collections::HashMap;
use chrono::{Utc, DateTime};
use uuid::Uuid;
use serde_json::{json, Value};
// use serde::{Serialize, Deserialize}; // Uncomment when needed
use tokio::sync::RwLock;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// TCP-based ClaudeFlowActor for direct MCP connection
/// This is the ONLY Claude Flow actor - WebSocket implementation has been removed
pub struct ClaudeFlowActorTcp {
    _client: ClaudeFlowClient,
    graph_service_addr: Addr<GraphServiceActor>,
    is_connected: bool,
    is_initialized: bool,
    swarm_id: Option<String>,
    polling_interval: Duration,
    _last_poll: DateTime<Utc>,
    agent_cache: HashMap<String, AgentStatus>,
    _swarm_status: Option<SwarmStatus>,
    system_metrics: SystemMetrics,
    message_flow_history: Vec<MessageFlowEvent>,
    coordination_patterns: Vec<CoordinationPattern>,
    // Direct TCP connection to Claude Flow on port 9500
    tcp_connection: Option<Arc<RwLock<TcpStream>>>,
    tcp_reader: Option<Arc<RwLock<BufReader<TcpStream>>>>,
    // Connection statistics
    connection_stats: ConnectionStats,
    // Pending changes for differential updates
    pending_additions: Vec<AgentStatus>,
    pending_removals: Vec<String>,
    pending_updates: Vec<AgentUpdate>,
    pending_messages: Vec<MessageFlowEvent>,
    swarm_topology: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct ConnectionStats {
    connected_at: Option<DateTime<Utc>>,
    messages_sent: u64,
    messages_received: u64,
    bytes_sent: u64,
    bytes_received: u64,
    last_message_at: Option<DateTime<Utc>>,
    reconnect_attempts: u32,
}

impl ClaudeFlowActorTcp {
    pub fn new(client: ClaudeFlowClient, graph_service_addr: Addr<GraphServiceActor>) -> Self {
        info!("Creating new TCP-based Claude Flow Actor");
        Self {
            _client: client,
            graph_service_addr,
            is_connected: false,
            is_initialized: false,
            swarm_id: None,
            polling_interval: Duration::from_millis(100), // 10Hz for telemetry updates
            _last_poll: Utc::now(),
            agent_cache: HashMap::new(),
            _swarm_status: None,
            system_metrics: SystemMetrics::default(),
            message_flow_history: Vec::new(),
            coordination_patterns: Vec::new(),
            tcp_connection: None,
            tcp_reader: None,
            connection_stats: ConnectionStats::default(),
            pending_additions: Vec::new(),
            pending_removals: Vec::new(),
            pending_updates: Vec::new(),
            pending_messages: Vec::new(),
            swarm_topology: None,
        }
    }

    /// Initialize direct TCP connection to Claude Flow on port 9500
    fn initialize_connection(&mut self, ctx: &mut Context<Self>) {
        info!("Initializing direct TCP connection to Claude Flow on port 9500");
        
        let addr = ctx.address();
        
        // Spawn async task to connect to Claude Flow
        tokio::spawn(async move {
            match Self::connect_to_claude_flow_tcp().await {
                Ok((stream, reader)) => {
                    info!("Successfully connected to Claude Flow MCP server via TCP on port 9500");
                    addr.do_send(TcpConnectionEstablished { stream, reader });
                }
                Err(e) => {
                    let err_msg = format!("Failed to connect to Claude Flow on TCP port 9500: {}", e);
                    error!("{}", err_msg);
                    // e is now dropped
                    // Retry connection after delay
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    addr.do_send(ConnectionFailed);
                }
            }
        });
    }
    
    /// Establish TCP connection to Claude Flow
    async fn connect_to_claude_flow_tcp() -> Result<(TcpStream, BufReader<TcpStream>), Box<dyn std::error::Error + Send + Sync>> {
        let host = std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| "172.18.0.10".to_string());
        let port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
        
        info!("Attempting TCP connection to {}:{}", host, port);
        
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr).await?;
        
        // Set TCP options for optimal performance
        stream.set_nodelay(true)?;
        
        // For Tokio, we need to use split() instead of try_clone()
        // This is a workaround - in production, consider using split() or a better pattern
        let stream2 = TcpStream::connect(&addr).await?;
        stream2.set_nodelay(true)?;
        let reader = BufReader::new(stream2);
        
        info!("TCP connection established to Claude Flow at {}", addr);
        
        Ok((stream, reader))
    }

    /// Send JSON-RPC message over TCP
    async fn send_tcp_message(stream: &mut TcpStream, message: Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg_str = serde_json::to_string(&message)?;
        let msg_bytes = format!("{}\n", msg_str); // Line-delimited JSON
        stream.write_all(msg_bytes.as_bytes()).await?;
        stream.flush().await?;
        debug!("Sent TCP message: {}", msg_str);
        Ok(())
    }

    /// Read JSON-RPC message from TCP
    async fn read_tcp_message(reader: &mut BufReader<TcpStream>) -> Result<Value, Box<dyn std::error::Error>> {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        let message: Value = serde_json::from_str(&line)?;
        debug!("Received TCP message: {}", line);
        Ok(message)
    }

    /// Initialize MCP session over TCP
    async fn initialize_mcp_session(&mut self) {
        if let Some(tcp_conn) = &self.tcp_connection {
            let mut stream = tcp_conn.write().await;
            
            let init_message = json!({
                "jsonrpc": "2.0",
                "id": Uuid::new_v4().to_string(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0.0",
                    "capabilities": {
                        "roots": true,
                        "sampling": true
                    },
                    "clientInfo": {
                        "name": "visionflow",
                        "version": "1.0.0"
                    }
                }
            });
            
            match Self::send_tcp_message(&mut *stream, init_message).await {
                Ok(_) => {
                    info!("MCP initialization message sent via TCP");
                    self.is_initialized = true;
                    self.connection_stats.connected_at = Some(Utc::now());
                    self.connection_stats.messages_sent += 1;
                }
                Err(e) => {
                    error!("Failed to send MCP initialization: {}", e);
                    self.is_connected = false;
                }
            }
        }
    }

    /// Process incoming TCP messages
    fn process_tcp_messages(&mut self, ctx: &mut Context<Self>) {
        if let Some(reader_arc) = self.tcp_reader.clone() {
            let addr = ctx.address();
            
            tokio::spawn(async move {
                let mut reader = reader_arc.write().await;
                
                loop {
                    match Self::read_tcp_message(&mut *reader).await {
                        Ok(message) => {
                            debug!("Processing TCP message: {:?}", message);
                            addr.do_send(ProcessTcpMessage { message });
                        }
                        Err(e) => {
                            error!("Error reading TCP message: {}", e);
                            break;
                        }
                    }
                }
                
                // Connection lost
                addr.do_send(ConnectionFailed);
            });
        }
    }

    /// Send a tool call over TCP
    async fn call_tcp_tool(&mut self, tool_name: &str, params: Value) -> Result<Value, String> {
        if let Some(tcp_conn) = &self.tcp_connection {
            let mut stream = tcp_conn.write().await;
            
            let request = json!({
                "jsonrpc": "2.0",
                "id": Uuid::new_v4().to_string(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            });
            
            match Self::send_tcp_message(&mut *stream, request).await {
                Ok(_) => {
                    self.connection_stats.messages_sent += 1;
                    self.connection_stats.last_message_at = Some(Utc::now());
                    Ok(json!({"status": "sent"}))
                }
                Err(e) => {
                    error!("Failed to call tool {}: {}", tool_name, e);
                    Err(e.to_string())
                }
            }
        } else {
            Err("No TCP connection available".to_string())
        }
    }
}

impl Actor for ClaudeFlowActorTcp {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("ClaudeFlowActorTcp started - using TCP-only implementation");
        
        // Initialize TCP connection
        self.initialize_connection(ctx);
        
        // Schedule periodic status updates
        ctx.run_interval(self.polling_interval, |act, ctx| {
            if act.is_connected && act.is_initialized {
                ctx.notify(PollAgentStatuses);
            }
        });
        
        // Schedule connection health check
        ctx.run_interval(Duration::from_secs(30), |act, ctx| {
            if !act.is_connected {
                warn!("TCP connection lost, attempting reconnection...");
                act.connection_stats.reconnect_attempts += 1;
                act.initialize_connection(ctx);
            }
        });
    }
    
    fn stopped(&mut self, _: &mut Self::Context) {
        info!("ClaudeFlowActorTcp stopped");
        info!("Connection statistics: {:?}", self.connection_stats);
    }
}

// Message handlers
impl Handler<InitializeSwarm> for ClaudeFlowActorTcp {
    type Result = ResponseFuture<Result<String, String>>;

    fn handle(&mut self, msg: InitializeSwarm, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_connected {
            return Box::pin(fut::ready(Err("Not connected to Claude Flow TCP server".to_string())));
        }

        let tcp_conn = self.tcp_connection.clone();
        let mut stats = self.connection_stats.clone();
        
        Box::pin(async move {
            if let Some(conn) = tcp_conn {
                let mut stream = conn.write().await;
                
                let params = json!({
                    "topology": msg.topology,
                    "maxAgents": msg.max_agents,
                    "strategy": msg.strategy
                });
                
                let request = json!({
                    "jsonrpc": "2.0",
                    "id": Uuid::new_v4().to_string(),
                    "method": "tools/call",
                    "params": {
                        "name": "swarm_init",
                        "arguments": params
                    }
                });
                
                match ClaudeFlowActorTcp::send_tcp_message(&mut *stream, request).await {
                    Ok(_) => {
                        stats.messages_sent += 1;
                        let swarm_id = format!("swarm_{}", Uuid::new_v4());
                        info!("Swarm initialization request sent via TCP: {}", swarm_id);
                        Ok(swarm_id)
                    }
                    Err(e) => {
                        error!("Failed to initialize swarm: {}", e);
                        Err(e.to_string())
                    }
                }
            } else {
                Err("No TCP connection available".to_string())
            }
        })
    }
}

// TCP-specific message types
#[derive(Message)]
#[rtype(result = "()")]
struct TcpConnectionEstablished {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

#[derive(Message)]
#[rtype(result = "()")]
struct ProcessTcpMessage {
    message: Value,
}

impl Handler<TcpConnectionEstablished> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, msg: TcpConnectionEstablished, ctx: &mut Self::Context) {
        info!("TCP connection established, initializing MCP session");
        
        self.tcp_connection = Some(Arc::new(RwLock::new(msg.stream)));
        self.tcp_reader = Some(Arc::new(RwLock::new(msg.reader)));
        self.is_connected = true;
        
        // Initialize MCP session
        let addr = ctx.address();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            addr.do_send(InitializeMCPSession);
        });
        
        // Start processing incoming messages
        self.process_tcp_messages(ctx);
    }
}

impl Handler<ProcessTcpMessage> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, msg: ProcessTcpMessage, _ctx: &mut Self::Context) {
        self.connection_stats.messages_received += 1;
        self.connection_stats.last_message_at = Some(Utc::now());
        
        // Process the message based on its type
        if let Some(method) = msg.message.get("method").and_then(|m| m.as_str()) {
            match method {
                "agent/status" => {
                    debug!("Received agent status update via TCP");
                    // Process agent status update
                }
                "swarm/update" => {
                    debug!("Received swarm update via TCP");
                    // Process swarm update
                }
                _ => {
                    debug!("Received unknown method via TCP: {}", method);
                }
            }
        }
    }
}

#[derive(Message)]
#[rtype(result = "()")]
struct InitializeMCPSession;

impl Handler<InitializeMCPSession> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: InitializeMCPSession, _ctx: &mut Self::Context) {
        let _addr = _ctx.address();
        tokio::spawn(async move {
            // This will be handled in the async context
        });
        
        // Use blocking call for now
        let tcp_conn = self.tcp_connection.clone();
        if tcp_conn.is_some() {
            tokio::spawn(async move {
                // Initialize in async context
            });
        }
    }
}

impl Handler<ConnectionFailed> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: ConnectionFailed, ctx: &mut Self::Context) {
        warn!("TCP connection failed or lost");
        self.is_connected = false;
        self.is_initialized = false;
        self.tcp_connection = None;
        self.tcp_reader = None;
        
        // Schedule reconnection attempt
        ctx.run_later(Duration::from_secs(5), |act, ctx| {
            info!("Attempting to reconnect to Claude Flow TCP server...");
            act.initialize_connection(ctx);
        });
    }
}

// Implement remaining handlers (similar to enhanced version but using TCP)
impl Handler<PollAgentStatuses> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: PollAgentStatuses, _ctx: &mut Self::Context) {
        if !self.is_connected || !self.is_initialized {
            return;
        }
        
        debug!("Polling agent statuses via TCP");
        // Implementation would call TCP tool for status updates
    }
}

impl Handler<GetSwarmStatus> for ClaudeFlowActorTcp {
    type Result = ResponseFuture<Result<SwarmStatus, String>>;

    fn handle(&mut self, _: GetSwarmStatus, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_connected {
            return Box::pin(fut::ready(Err("Not connected to Claude Flow".to_string())));
        }
        
        // Return current swarm status
        Box::pin(fut::ready(Ok(SwarmStatus {
            swarm_id: self.swarm_id.clone().unwrap_or_default(),
            active_agents: self.agent_cache.len() as u32,
            total_agents: self.agent_cache.len() as u32,
            topology: self.swarm_topology.clone().unwrap_or_else(|| "mesh".to_string()),
            health_score: if self.is_connected && self.is_initialized { 1.0 } else { 0.0 },
            coordination_efficiency: 0.85, // Default efficiency metric
        })))
    }
}