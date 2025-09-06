use actix::prelude::*;
use actix::fut;
use std::time::Duration;
use log::{info, error, debug, warn};
use crate::types::claude_flow::{ClaudeFlowClient, AgentStatus, AgentProfile, AgentType, PerformanceMetrics, TokenUsage};
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
    TimeoutConfig, ConnectionPool, ConnectionPoolConfig
};
use crate::utils::resource_monitor::{ResourceMonitor, ResourceLimits};

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
    // Error tracking for circuit breaker
    consecutive_poll_failures: u32,
    last_successful_poll: Option<DateTime<Utc>>,
    // Network resilience components
    resilience_manager: Arc<NetworkResilienceManager>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    timeout_config: TimeoutConfig,
    // Connection pool for resource management
    connection_pool: Option<Arc<tokio::sync::Mutex<ConnectionPool>>>,
    // Track active connections for cleanup
    active_connections: Arc<tokio::sync::RwLock<std::collections::HashMap<String, std::time::Instant>>>,
    // Resource monitoring for preventing file descriptor exhaustion
    resource_monitor: Arc<ResourceMonitor>,
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
    /// Convert MCP agent format to VisionFlow AgentStatus
    fn mcp_agent_to_status(agent_data: &Value) -> Result<AgentStatus, String> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let agent_id = agent_data.get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
            
        let name = agent_data.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&agent_id)
            .to_string();
            
        let agent_type_str = agent_data.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("coordinator");
            
        let status = agent_data.get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("active")
            .to_string();
            
        let swarm_id = agent_data.get("swarmId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
            
        // Parse agent type - map to existing VisionFlow types
        let agent_type = match agent_type_str {
            "coordinator" | "task-orchestrator" => AgentType::Coordinator,
            "researcher" => AgentType::Researcher,
            "coder" | "worker" => AgentType::Coder,  // Workers do implementation
            "analyst" | "analyzer" | "code-analyzer" | "specialist" => AgentType::Analyst,  // Specialists analyze
            "architect" => AgentType::Architect,
            "tester" => AgentType::Tester,
            "reviewer" => AgentType::Reviewer,
            "optimizer" => AgentType::Optimizer,
            "documenter" => AgentType::Documenter,
            _ => AgentType::Coordinator, // Default
        };
        
        // Create AgentStatus with defaults for missing fields
        Ok(AgentStatus {
            agent_id: agent_id.clone(),
            profile: AgentProfile {
                name,
                agent_type,
                capabilities: Vec::new(),
            },
            status,
            active_tasks_count: 0,
            completed_tasks_count: 0,
            failed_tasks_count: 0,
            success_rate: 100.0,
            timestamp: Utc::now(),
            current_task: None,
            cpu_usage: 10.0 + rng.r#gen::<f32>() * 20.0,
            memory_usage: 20.0 + rng.r#gen::<f32>() * 30.0,
            health: 100.0,
            activity: 50.0,
            tasks_active: 0,
            performance_metrics: PerformanceMetrics {
                tasks_completed: 0,
                success_rate: 100.0,
            },
            token_usage: TokenUsage {
                total: 1000,
                token_rate: 0.0,
            },
            swarm_id,
            agent_mode: Some("autonomous".to_string()),
            parent_queen_id: None,
            processing_logs: None,
            total_execution_time: 0,
        })
    }
    
    pub fn new(client: ClaudeFlowClient, graph_service_addr: Addr<GraphServiceActor>) -> Self {
        info!("Creating new TCP-based Claude Flow Actor");
        Self {
            _client: client,
            graph_service_addr,
            is_connected: false,
            is_initialized: false,
            swarm_id: None,
            polling_interval: Duration::from_millis(1000), // 1Hz for telemetry updates
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
            consecutive_poll_failures: 0,
            last_successful_poll: None,
            resilience_manager: Arc::new(NetworkResilienceManager::new()),
            circuit_breaker: None,
            timeout_config: TimeoutConfig::tcp_connection(),
            connection_pool: None,
            active_connections: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            resource_monitor: Arc::new(ResourceMonitor::new(ResourceLimits::default())),
        }
    }

    /// Initialize direct TCP connection to Claude Flow on port 9500 with resilience patterns
    fn initialize_connection(&mut self, ctx: &mut Context<Self>) {
        debug!("Initializing resilient TCP connection to Claude Flow on port 9500");
        
        // Initialize connection pool if not already present
        if self.connection_pool.is_none() {
            let pool_config = ConnectionPoolConfig {
                max_connections_per_endpoint: 2, // Limit connections per endpoint
                max_total_connections: 5,        // Total connection limit
                connection_timeout: std::time::Duration::from_secs(10),
                idle_timeout: std::time::Duration::from_secs(60),
                max_connection_lifetime: std::time::Duration::from_secs(300),
                cleanup_interval: std::time::Duration::from_secs(30),
                validate_on_borrow: true,
                validate_while_idle: false,
            };
            let mut pool = ConnectionPool::new(pool_config);
            pool.start_cleanup_task();
            self.connection_pool = Some(Arc::new(tokio::sync::Mutex::new(pool)));
        }
        
        let addr = ctx.address();
        let resilience_manager = self.resilience_manager.clone();
        let timeout_config = self.timeout_config.clone();
        let active_connections = self.active_connections.clone();
        
        // Spawn async task to connect to Claude Flow with resilience
        tokio::spawn(async move {
            // Track this connection attempt
            let connection_id = uuid::Uuid::new_v4().to_string();
            {
                let mut connections = active_connections.write().await;
                connections.insert(connection_id.clone(), std::time::Instant::now());
            }
            
            // Use resilient connection with retry logic with resource limits
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
            
            // Remove connection tracking on completion
            {
                let mut connections = active_connections.write().await;
                connections.remove(&connection_id);
            }
        });
    }
    
    /// Establish TCP connection to Claude Flow
    async fn connect_to_claude_flow_tcp() -> Result<(BufWriter<tokio::net::tcp::OwnedWriteHalf>, BufReader<tokio::net::tcp::OwnedReadHalf>), Box<dyn std::error::Error + Send + Sync>> {
        // Use service name for Docker networking, fallback to localhost for development
        let host = std::env::var("CLAUDE_FLOW_HOST")
            .or_else(|_| std::env::var("MCP_HOST"))
            .unwrap_or_else(|_| {
                if std::env::var("DOCKER_ENV").is_ok() {
                    "multi-agent-container".to_string()  // Docker service name
                } else {
                    "localhost".to_string()  // Local development
                }
            });
        let port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
        
        debug!("Attempting TCP connection to {}:{}", host, port);
        
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
        
        // Start resource monitoring
        let resource_monitor = self.resource_monitor.clone();
        tokio::spawn(async move {
            if let Err(e) = resource_monitor.start_monitoring(std::time::Duration::from_secs(10)).await {
                error!("Failed to start resource monitoring: {}", e);
            }
        });
        
        // Initialize TCP connection
        self.initialize_connection(ctx);
        
        // Schedule periodic status updates
        ctx.run_interval(self.polling_interval, |act, ctx| {
            if act.is_connected && act.is_initialized {
                ctx.notify(PollAgentStatuses);
            }
        });
        
        // DEPRECATED: The ConnectionFailed handler is now responsible for all reconnection logic.
        // This periodic check is redundant and can cause cascading failures.
        // ctx.run_interval(Duration::from_secs(30), |act, ctx| {
        //     if !act.is_connected {
        //         warn!("TCP connection lost, attempting reconnection...");
        //         act.connection_stats.reconnect_attempts += 1;
        //         act.initialize_connection(ctx);
        //     }
        // });
    }
    
    fn stopped(&mut self, _: &mut Self::Context) {
        info!("ClaudeFlowActorTcp stopping - cleaning up resources");
        info!("Connection statistics: {:?}", self.connection_stats);
        
        // Cleanup connections and pools
        if let Some(pool) = self.connection_pool.take() {
            tokio::spawn(async move {
                let mut pool_guard = pool.lock().await;
                pool_guard.shutdown().await;
            });
        }
        
        // Clear connection tracking
        let active_connections = self.active_connections.clone();
        tokio::spawn(async move {
            let mut connections = active_connections.write().await;
            connections.clear();
        });
        
        info!("ClaudeFlowActorTcp stopped - resources cleaned up");
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

#[derive(Message)]
#[rtype(result = "()")]
struct UpdateAgentCache {
    agents: Vec<AgentStatus>,
}

#[derive(Message)]
#[rtype(result = "()")]
struct RecordPollSuccess;

#[derive(Message)]
#[rtype(result = "()")]
struct RecordPollFailure;

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
        
        // Reset failure counters on successful connection
        self.consecutive_poll_failures = 0;
        self.last_successful_poll = Some(Utc::now());
        
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
        
        // Properly close existing connections
        if let Some(writer_arc) = self.tcp_writer.take() {
            tokio::spawn(async move {
                // Attempt graceful shutdown of writer
                if let Ok(mut writer) = writer_arc.try_write() {
                    let _ = writer.shutdown().await;
                }
            });
        }
        
        if let Some(_reader_arc) = self.tcp_reader.take() {
            // Reader will be dropped automatically, closing the connection
        }
        
        // Clear pending requests with proper error responses
        let pending_requests = self.pending_requests.clone();
        tokio::spawn(async move {
            let mut requests = pending_requests.write().await;
            // Send error responses to pending requests before clearing
            for (id, sender) in requests.drain() {
                let error_response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -1,
                        "message": "Connection lost"
                    }
                });
                let _ = sender.send(error_response);
            }
        });
        
        // Increment reconnect attempts with exponential backoff
        self.connection_stats.reconnect_attempts += 1;
        let backoff_delay = std::cmp::min(
            Duration::from_secs(5 * (1 << self.connection_stats.reconnect_attempts.min(5))),
            Duration::from_secs(300) // Max 5 minutes
        );
        
        info!("Scheduling reconnection attempt {} in {:?}", 
              self.connection_stats.reconnect_attempts, backoff_delay);
        
        // Schedule reconnection attempt with exponential backoff
        ctx.run_later(backoff_delay, |act, ctx| {
            info!("Attempting to reconnect to Claude Flow TCP server (attempt {})", 
                  act.connection_stats.reconnect_attempts);
            act.initialize_connection(ctx);
        });
    }
}

// Implement remaining handlers (similar to enhanced version but using TCP)
impl Handler<PollAgentStatuses> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: PollAgentStatuses, _ctx: &mut Self::Context) {
        if !self.is_connected || !self.is_initialized {
            debug!("Skipping agent status poll - not connected or initialized (connected: {}, initialized: {})", 
                   self.is_connected, self.is_initialized);
            return;
        }
        
        // Circuit breaker: if too many consecutive failures, reduce polling frequency
        if self.consecutive_poll_failures > 10 {
            if let Some(last_success) = self.last_successful_poll {
                let time_since_success = Utc::now().signed_duration_since(last_success);
                if time_since_success.num_seconds() < 30 {
                    debug!("Circuit breaker active - skipping poll due to {} consecutive failures", 
                           self.consecutive_poll_failures);
                    return;
                }
            }
        }
        
        // DISABLED: ClaudeFlowActor TCP polling is broken due to persistent connection issues
        // The MCP server closes connections after each request, but this actor expects persistent connections
        // BotsClient handles agent fetching correctly with fresh connections
        debug!("ClaudeFlowActor polling DISABLED - using BotsClient instead");
        return;
        
        debug!("Polling agent statuses via TCP (100ms cycle) - {} consecutive failures", 
               self.consecutive_poll_failures);
        
        // Call the agent_list tool to get current agent statuses
        if let Some(writer) = &self.tcp_writer {
            let writer_clone = writer.clone();
            let pending_requests = self.pending_requests.clone();
            let graph_addr = self.graph_service_addr.clone();
            
            // Get context address for cache updates
            let ctx_addr = _ctx.address();
            
            tokio::spawn(async move {
                let params = json!({
                    "filter": "all" // Get all agents, not just active ones
                });
                
                match Self::call_tcp_tool(writer_clone, pending_requests, "agent_list", params).await {
                    Ok(response) => {
                        debug!("Received agent list response: {:?}", response);
                        
                        // Extract the actual content from MCP response format
                        // MCP returns: {"jsonrpc":"2.0","id":"...","result":{"content": [{"type": "text", "text": "{JSON}"}]}}
                        // Note: response already contains the full JSON-RPC response
                        let parsed_response = if let Some(result) = response.get("result") {
                            if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
                                if let Some(first_content) = content.first() {
                                    if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                                        // Parse the nested JSON string
                                        match serde_json::from_str::<Value>(text) {
                                            Ok(parsed) => {
                                                info!("Successfully parsed nested MCP response");
                                                debug!("Parsed content: {:?}", parsed);
                                                parsed
                                            }
                                            Err(e) => {
                                                warn!("Failed to parse nested JSON: {}", e);
                                                warn!("Raw text was: {}", text);
                                                // Return empty object with agents array
                                                json!({"agents": []})
                                            }
                                        }
                                    } else {
                                        warn!("No text field in content");
                                        json!({"agents": []})
                                    }
                                } else {
                                    warn!("No content array elements");
                                    json!({"agents": []})
                                }
                            } else {
                                warn!("No content field in result");
                                json!({"agents": []})
                            }
                        } else {
                            warn!("No result field in response - raw response: {:?}", response);
                            json!({"agents": []})
                        };
                        
                        // Parse agents and prepare for graph visualization
                        if let Some(agents) = parsed_response.get("agents").and_then(|a| a.as_array()) {
                            let mut agent_statuses = Vec::new();
                            let mut parsing_errors = 0u32;
                            
                            for (idx, agent_data) in agents.iter().enumerate() {
                                // Convert MCP agent format to VisionFlow AgentStatus
                                let agent_status = Self::mcp_agent_to_status(agent_data);
                                match agent_status {
                                    Ok(status) => {
                                        info!("Agent [{}] {} - Status: {}, Type: {:?}, Tasks: {} active / {} completed", 
                                              idx, 
                                              status.agent_id, 
                                              status.status,
                                              status.profile.agent_type,
                                              status.active_tasks_count,
                                              status.completed_tasks_count);
                                        
                                        agent_statuses.push(status);
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse agent data at index {}: {} - Raw data: {:?}", 
                                              idx, e, agent_data);
                                        parsing_errors += 1;
                                    }
                                }
                            }
                            
                            // Always send graph update, even if no agents (clears visualization)
                            let message = UpdateBotsGraph {
                                agents: agent_statuses.clone()
                            };
                            
                            info!("🔄 Sending graph update to GraphServiceActor: {} agents parsed ({} parsing errors)", 
                                  agent_statuses.len(), parsing_errors);
                            
                            // Send to GraphServiceActor for real-time visualization
                            graph_addr.do_send(message);
                            
                            // Update the actor's agent cache with latest data
                            if !agent_statuses.is_empty() {
                                ctx_addr.do_send(UpdateAgentCache { 
                                    agents: agent_statuses.clone() 
                                });
                            }
                            
                            // Mark poll as successful
                            ctx_addr.do_send(RecordPollSuccess);
                            
                            // Log detailed agent information for debugging
                            if !agent_statuses.is_empty() {
                                info!("📊 Agent Summary:");
                                let mut by_type = std::collections::HashMap::new();
                                let mut by_status = std::collections::HashMap::new();
                                
                                for agent in &agent_statuses {
                                    *by_type.entry(format!("{:?}", agent.profile.agent_type)).or_insert(0u32) += 1;
                                    *by_status.entry(agent.status.clone()).or_insert(0u32) += 1;
                                }
                                
                                info!("  Types: {:?}", by_type);
                                info!("  Statuses: {:?}", by_status);
                                info!("  Total tokens: {}", agent_statuses.iter().map(|a| a.token_usage.total).sum::<u64>());
                                info!("  Avg success rate: {:.2}%", 
                                      agent_statuses.iter().map(|a| a.success_rate).sum::<f32>() / agent_statuses.len().max(1) as f32 * 100.0);
                                
                                // CRITICAL FIX: Send the agents to the graph!
                                info!("📨 Sending {} agents to graph update", agent_statuses.len());
                                graph_addr.do_send(UpdateBotsGraph {
                                    agents: agent_statuses
                                });
                            } else {
                                info!("📭 No agents found - sending empty graph update");
                                // Send empty graph update
                                graph_addr.do_send(UpdateBotsGraph {
                                    agents: Vec::new()
                                });
                            }
                        } else {
                            warn!("Invalid response format - missing 'agents' array in: {:?}", parsed_response);
                            
                            // Send empty graph update to clear visualization
                            graph_addr.do_send(UpdateBotsGraph {
                                agents: Vec::new()
                            });
                        }
                    }
                    Err(e) => {
                        error!("❌ Failed to poll agent statuses via TCP: {}", e);
                        
                        // Send empty graph update on error to avoid stale data
                        graph_addr.do_send(UpdateBotsGraph {
                            agents: Vec::new()
                        });
                        
                        // Mark poll as failed
                        ctx_addr.do_send(RecordPollFailure);
                    }
                }
            });
        } else {
            warn!("No TCP writer available for agent status polling");
        }
    }
}

impl Handler<UpdateAgentCache> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, msg: UpdateAgentCache, _ctx: &mut Self::Context) {
        debug!("Updating agent cache with {} agents", msg.agents.len());
        
        // Clear old cache and update with new agent data
        self.agent_cache.clear();
        
        for agent in msg.agents {
            self.agent_cache.insert(agent.agent_id.clone(), agent);
        }
        
        debug!("Agent cache updated: {} agents cached", self.agent_cache.len());
    }
}

impl Handler<RecordPollSuccess> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: RecordPollSuccess, _ctx: &mut Self::Context) {
        self.consecutive_poll_failures = 0;
        self.last_successful_poll = Some(Utc::now());
        debug!("Poll success recorded - reset failure counter");
    }
}

impl Handler<RecordPollFailure> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, _: RecordPollFailure, _ctx: &mut Self::Context) {
        self.consecutive_poll_failures += 1;
        warn!("Poll failure recorded - {} consecutive failures", 
              self.consecutive_poll_failures);
        
        if self.consecutive_poll_failures > 20 {
            error!("Too many consecutive polling failures ({}), may need reconnection", 
                   self.consecutive_poll_failures);
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

// Implement Drop to ensure proper cleanup of resources
impl Drop for ClaudeFlowActorTcp {
    fn drop(&mut self) {
        info!("Dropping ClaudeFlowActorTcp - performing emergency cleanup");
        
        // Forcibly close any remaining connections
        if let Some(writer_arc) = self.tcp_writer.take() {
            if let Ok(mut writer) = writer_arc.try_write() {
                let _ = futures::executor::block_on(writer.shutdown());
            }
        }
        
        // Clear connection tracking
        if let Ok(mut connections) = self.active_connections.try_write() {
            connections.clear();
        }
        
        info!("ClaudeFlowActorTcp drop completed - connections cleaned up");
    }
}