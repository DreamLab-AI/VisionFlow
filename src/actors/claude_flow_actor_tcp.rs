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
use tokio::sync::{RwLock, oneshot};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use crate::utils::network::{
    NetworkResilienceManager, CircuitBreaker,
    retry_tcp_connection, RetryableError,
    TimeoutConfig
};

/// Make Box<dyn Error> retryable for network operations
impl RetryableError for Box<dyn std::error::Error + Send + Sync> {
    fn is_retryable(&self) -> bool {
        // Check if the underlying error is retryable
        if let Some(io_error) = self.downcast_ref::<std::io::Error>() {
            io_error.is_retryable()
        } else {
            // Default to retryable for network operations
            true
        }
    }
}

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
    tcp_writer: Option<Arc<RwLock<BufWriter<tokio::net::tcp::OwnedWriteHalf>>>>,
    tcp_reader: Option<Arc<RwLock<BufReader<tokio::net::tcp::OwnedReadHalf>>>>,
    // Connection statistics
    connection_stats: ConnectionStats,
    // Request/response correlation
    pending_requests: Arc<RwLock<HashMap<String, oneshot::Sender<Value>>>>,
    // Pending changes for differential updates
    pending_additions: Vec<AgentStatus>,
    pending_removals: Vec<String>,
    pending_updates: Vec<AgentUpdate>,
    pending_messages: Vec<MessageFlowEvent>,
    swarm_topology: Option<String>,
    // Network resilience components
    resilience_manager: Arc<NetworkResilienceManager>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    timeout_config: TimeoutConfig,
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
            tcp_writer: None,
            tcp_reader: None,
            connection_stats: ConnectionStats::default(),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            pending_additions: Vec::new(),
            pending_removals: Vec::new(),
            pending_updates: Vec::new(),
            pending_messages: Vec::new(),
            swarm_topology: None,
            resilience_manager: Arc::new(NetworkResilienceManager::new()),
            circuit_breaker: None,
            timeout_config: TimeoutConfig::tcp_connection(),
        }
    }

    /// Initialize direct TCP connection to Claude Flow on port 9500 with resilience patterns
    fn initialize_connection(&mut self, ctx: &mut Context<Self>) {
        info!("Initializing resilient TCP connection to Claude Flow on port 9500");
        
        let addr = ctx.address();
        let resilience_manager = self.resilience_manager.clone();
        let timeout_config = self.timeout_config.clone();
        
        // Spawn async task to connect to Claude Flow with resilience
        tokio::spawn(async move {
            // Use resilient connection with retry logic
            let connection_operation = || async {
                Self::connect_to_claude_flow_tcp().await
                    .map_err(|e| std::sync::Arc::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
            };
            
            match retry_tcp_connection(connection_operation).await {
                Ok((writer, reader)) => {
                    info!("Successfully connected to Claude Flow MCP server via TCP on port 9500 with resilience");
                    addr.do_send(TcpConnectionEstablished { writer, reader });
                }
                Err(e) => {
                    let err_msg = format!("Failed to connect to Claude Flow on TCP port 9500 after retries: {:?}", e);
                    error!("{}", err_msg);
                    addr.do_send(ConnectionFailed);
                }
            }
        });
    }
    
    /// Establish TCP connection to Claude Flow
    async fn connect_to_claude_flow_tcp() -> Result<(BufWriter<tokio::net::tcp::OwnedWriteHalf>, BufReader<tokio::net::tcp::OwnedReadHalf>), Box<dyn std::error::Error + Send + Sync>> {
        // Use service name for Docker networking, fallback to localhost for development
        let host = std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| {
            if std::env::var("DOCKER_ENV").is_ok() {
                "claude-flow-mcp".to_string()  // Docker service name
            } else {
                "localhost".to_string()  // Local development
            }
        });
        let port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
        
        info!("Attempting TCP connection to {}:{}", host, port);
        
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr).await?;
        
        // Set TCP options for optimal performance
        stream.set_nodelay(true)?;
        
        // Properly split the stream into read and write halves
        let (read_half, write_half) = stream.into_split();
        let reader = BufReader::new(read_half);
        let writer = BufWriter::new(write_half);
        
        info!("TCP connection established to Claude Flow at {}", addr);
        
        Ok((writer, reader))
    }

    /// Send JSON-RPC message over TCP
    async fn send_tcp_message(writer: &mut BufWriter<tokio::net::tcp::OwnedWriteHalf>, message: Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg_str = serde_json::to_string(&message)?;
        let msg_bytes = format!("{}\n", msg_str); // Line-delimited JSON
        writer.write_all(msg_bytes.as_bytes()).await?;
        writer.flush().await?;
        debug!("Sent TCP message: {}", msg_str);
        Ok(())
    }

    /// Read JSON-RPC message from TCP
    async fn read_tcp_message(reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        let message: Value = serde_json::from_str(&line)?;
        debug!("Received TCP message: {}", line);
        Ok(message)
    }

    /// Initialize MCP session over TCP with proper correlation
    async fn initialize_mcp_session(
        writer: Arc<RwLock<BufWriter<tokio::net::tcp::OwnedWriteHalf>>>,
        pending_requests: Arc<RwLock<HashMap<String, oneshot::Sender<Value>>>>
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let request_id = Uuid::new_v4().to_string();
        
        let init_message = json!({
            "jsonrpc": "2.0",
            "id": request_id.clone(),
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {
                    "roots": true,
                    "sampling": true,
                    "tools": true
                },
                "clientInfo": {
                    "name": "visionflow",
                    "version": "1.0.0"
                }
            }
        });
        
        // Set up response channel
        let (tx, rx) = oneshot::channel();
        pending_requests.write().await.insert(request_id.clone(), tx);
        
        // Send the message
        let mut writer_guard = writer.write().await;
        Self::send_tcp_message(&mut *writer_guard, init_message).await?;
        drop(writer_guard);
        
        // Wait for response with timeout
        match tokio::time::timeout(Duration::from_secs(5), rx).await {
            Ok(Ok(response)) => {
                info!("MCP initialization successful: {:?}", response);
                Ok(())
            }
            Ok(Err(_)) => Err("Response channel closed".into()),
            Err(_) => Err("MCP initialization timeout".into())
        }
    }

    /// Process incoming TCP messages
    fn process_tcp_messages(&mut self, ctx: &mut Context<Self>) {
        if let Some(reader_arc) = self.tcp_reader.clone() {
            let addr = ctx.address();
            let pending_requests = self.pending_requests.clone();
            
            tokio::spawn(async move {
                let mut reader = reader_arc.write().await;
                
                loop {
                    match Self::read_tcp_message(&mut *reader).await {
                        Ok(message) => {
                            debug!("Processing TCP message: {:?}", message);
                            
                            // Check if this is a response to a pending request
                            if let Some(id) = message.get("id").and_then(|v| v.as_str()) {
                                let mut requests = pending_requests.write().await;
                                if let Some(sender) = requests.remove(id) {
                                    // Send response to waiting caller
                                    let _ = sender.send(message.clone());
                                }
                            }
                            
                            // Also send to actor for processing
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

    /// Send a tool call over TCP with proper response correlation
    async fn call_tcp_tool(
        writer: Arc<RwLock<BufWriter<tokio::net::tcp::OwnedWriteHalf>>>,
        pending_requests: Arc<RwLock<HashMap<String, oneshot::Sender<Value>>>>,
        tool_name: &str,
        params: Value
    ) -> Result<Value, String> {
        let request_id = Uuid::new_v4().to_string();
        
        let request = json!({
            "jsonrpc": "2.0",
            "id": request_id.clone(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        });
        
        // Set up response channel
        let (tx, rx) = oneshot::channel();
        pending_requests.write().await.insert(request_id.clone(), tx);
        
        // Send the request
        let mut writer_guard = writer.write().await;
        match Self::send_tcp_message(&mut *writer_guard, request).await {
            Ok(_) => {
                drop(writer_guard);
                
                // Wait for response with timeout
                match tokio::time::timeout(Duration::from_secs(10), rx).await {
                    Ok(Ok(response)) => Ok(response),
                    Ok(Err(_)) => Err("Response channel closed".to_string()),
                    Err(_) => Err(format!("Tool call {} timeout", tool_name))
                }
            }
            Err(e) => {
                error!("Failed to call tool {}: {}", tool_name, e);
                pending_requests.write().await.remove(&request_id);
                Err(e.to_string())
            }
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

        let tcp_writer = self.tcp_writer.clone();
        let pending_requests = self.pending_requests.clone();
        
        Box::pin(async move {
            if let Some(writer) = tcp_writer {
                let params = json!({
                    "topology": msg.topology,
                    "maxAgents": msg.max_agents,
                    "strategy": msg.strategy
                });
                
                match ClaudeFlowActorTcp::call_tcp_tool(writer, pending_requests, "swarm_init", params).await {
                    Ok(response) => {
                        if let Some(swarm_id) = response.get("swarmId").and_then(|s| s.as_str()) {
                            info!("Swarm initialized successfully: {}", swarm_id);
                            Ok(swarm_id.to_string())
                        } else {
                            Ok(format!("swarm_{}", Uuid::new_v4()))
                        }
                    }
                    Err(e) => {
                        error!("Failed to initialize swarm: {}", e);
                        Err(e)
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
    writer: BufWriter<tokio::net::tcp::OwnedWriteHalf>,
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
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
        
        self.tcp_writer = Some(Arc::new(RwLock::new(msg.writer)));
        self.tcp_reader = Some(Arc::new(RwLock::new(msg.reader)));
        self.is_connected = true;
        
        // Initialize MCP session
        let writer = match self.tcp_writer.clone() {
            Some(writer) => writer,
            None => {
                error!("TCP writer not available for MCP session initialization");
                ctx.address().do_send(ConnectionFailed);
                return;
            }
        };
        let pending_requests = self.pending_requests.clone();
        let addr = ctx.address();
        
        tokio::spawn(async move {
            match Self::initialize_mcp_session(writer, pending_requests).await {
                Ok(_) => {
                    info!("MCP session initialized successfully");
                    addr.do_send(MCPSessionInitialized);
                }
                Err(e) => {
                    error!("Failed to initialize MCP session: {}", e);
                    addr.do_send(ConnectionFailed);
                }
            }
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
struct MCPSessionInitialized;

impl Handler<MCPSessionInitialized> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: MCPSessionInitialized, _ctx: &mut Self::Context) {
        info!("MCP session has been initialized");
        self.is_initialized = true;
        self.connection_stats.connected_at = Some(Utc::now());
        
        // Now we can start polling for agent statuses
        _ctx.notify(PollAgentStatuses);
    }
}

impl Handler<ConnectionFailed> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: ConnectionFailed, ctx: &mut Self::Context) {
        warn!("TCP connection failed or lost");
        self.is_connected = false;
        self.is_initialized = false;
        self.tcp_writer = None;
        self.tcp_reader = None;
        
        // Clear pending requests
        let pending_requests = self.pending_requests.clone();
        tokio::spawn(async move {
            let mut requests = pending_requests.write().await;
            requests.clear();
        });
        
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
        
        // Call the agent_list tool to get current agent statuses
        if let Some(writer) = &self.tcp_writer {
            let writer_clone = writer.clone();
            let pending_requests = self.pending_requests.clone();
            let graph_addr = self.graph_service_addr.clone();
            
            tokio::spawn(async move {
                let params = json!({
                    "filter": "active"
                });
                
                match Self::call_tcp_tool(writer_clone, pending_requests, "agent_list", params).await {
                    Ok(response) => {
                        debug!("Received agent list: {:?}", response);
                        
                        // Parse agents for monitoring
                        if let Some(agents) = response.get("agents").and_then(|a| a.as_array()) {
                            for agent in agents {
                                if let Ok(status) = serde_json::from_value::<AgentStatus>(agent.clone()) {
                                    // Agent status monitoring - could be sent to monitoring service
                                    debug!("Agent {} status: {:?}", status.agent_id, status.status);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to poll agent statuses: {}", e);
                    }
                }
            });
        }
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