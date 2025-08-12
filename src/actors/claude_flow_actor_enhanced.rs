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
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use std::sync::Arc;
use futures_util::SinkExt;
use tokio_tungstenite::{connect_async, WebSocketStream, MaybeTlsStream};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Enhanced ClaudeFlowActor with direct MCP WebSocket integration
pub struct EnhancedClaudeFlowActor {
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
    // Direct WebSocket connection to Claude Flow on port 3002
    ws_connection: Option<Arc<RwLock<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    // Pending changes for differential updates
    pending_additions: Vec<AgentStatus>,
    pending_removals: Vec<String>,
    pending_updates: Vec<AgentUpdate>,
    pending_messages: Vec<MessageFlowEvent>,
}

impl EnhancedClaudeFlowActor {
    pub fn new(client: ClaudeFlowClient, graph_service_addr: Addr<GraphServiceActor>) -> Self {
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
            ws_connection: None,
            pending_additions: Vec::new(),
            pending_removals: Vec::new(),
            pending_updates: Vec::new(),
            pending_messages: Vec::new(),
        }
    }

    /// Initialize direct MCP WebSocket connection to Claude Flow on port 3002
    fn initialize_connection(&mut self, ctx: &mut Context<Self>) {
        info!("Initializing direct MCP WebSocket connection to Claude Flow on port 3002");
        
        let addr = ctx.address();
        
        // Spawn async task to connect to Claude Flow
        tokio::spawn(async move {
            match Self::connect_to_claude_flow().await {
                Ok(ws_stream) => {
                    info!("Successfully connected to Claude Flow MCP server on port 3002");
                    addr.do_send(ConnectionEstablished { ws_stream });
                }
                Err(e) => {
                    warn!("Failed to connect to Claude Flow on port 3002: {}. Using fallback mode.", e);
                    // Still mark as initialized to use fallback/mock data
                    addr.do_send(ConnectionFailed);
                }
            }
        });
    }
    
    /// Establish WebSocket connection to Claude Flow (WebSocket only, no TCP/stdio fallback)
    async fn connect_to_claude_flow() -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>, Box<dyn std::error::Error>> {
        // Try multiple possible endpoints for Claude Flow
        let possible_endpoints = vec![
            (std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| "multi-agent-container".to_string()), 
             std::env::var("CLAUDE_FLOW_PORT").unwrap_or_else(|_| "3002".to_string()),
             "/mcp".to_string()),
            ("localhost".to_string(), "3002".to_string(), "/mcp".to_string()),
            ("multi-agent-container".to_string(), "3002".to_string(), "/ws".to_string()),
            ("claude-flow".to_string(), "3002".to_string(), "/mcp".to_string()),
        ];
        
        let mut last_error = None;
        
        for (host, port, path) in possible_endpoints {
            let url = format!("ws://{}:{}{}", host, port, path);
            info!("Attempting to connect to Claude Flow at: {}", url);
            
            match connect_async(&url).await {
                Ok((ws_stream, response)) => {
                    info!("Successfully connected to Claude Flow at: {} (status: {})", url, response.status());
                    return Ok(ws_stream);
                }
                Err(e) => {
                    warn!("Failed to connect to {}: {}", url, e);
                    last_error = Some(e);
                }
            }
        }
        
        Err(Box::new(last_error.unwrap_or_else(|| tokio_tungstenite::tungstenite::Error::Io(std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "No endpoints available")))))
    }

    /// Start real-time telemetry streaming
    fn start_real_time_streaming(&mut self, ctx: &mut Context<Self>) {
        info!("Starting real-time telemetry streaming at 10Hz");

        // Schedule interval for telemetry updates (100ms = 10Hz)
        ctx.run_interval(self.polling_interval, |actor, ctx| {
            if actor.is_connected {
                ctx.address().do_send(ProcessTelemetryStream);
            }
        });

        // Schedule differential update processing
        ctx.run_interval(Duration::from_millis(50), |actor, ctx| {
            if !actor.pending_additions.is_empty() || 
               !actor.pending_removals.is_empty() || 
               !actor.pending_updates.is_empty() ||
               !actor.pending_messages.is_empty() {
                ctx.address().do_send(ProcessDeltaUpdates);
            }
        });
    }
    
    /// Handle incoming telemetry event from Claude Flow
    #[allow(dead_code)]
    fn handle_telemetry_event(&mut self, event: Value) {
        if let Some(event_type) = event.get("type").and_then(|t| t.as_str()) {
            match event_type {
                "agent.spawned" => {
                    if let Ok(agent) = serde_json::from_value::<AgentStatus>(event["data"].clone()) {
                        self.pending_additions.push(agent.clone());
                        self.agent_cache.insert(agent.agent_id.clone(), agent);
                    }
                }
                "agent.terminated" => {
                    if let Some(id) = event.get("data").and_then(|d| d.get("id")).and_then(|i| i.as_str()) {
                        self.pending_removals.push(id.to_string());
                        self.agent_cache.remove(id);
                    }
                }
                "agent.status" => {
                    if let Ok(update) = serde_json::from_value::<AgentUpdate>(event["data"].clone()) {
                        self.pending_updates.push(update.clone());
                        if let Some(agent) = self.agent_cache.get_mut(&update.agent_id) {
                            agent.status = update.status.clone();
                            if let Some(task) = &update.current_task {
                                agent.current_task = Some(task.clone());
                            }
                        }
                    }
                }
                "message.flow" => {
                    if let Ok(msg_flow) = serde_json::from_value::<MessageFlowEvent>(event["data"].clone()) {
                        self.pending_messages.push(msg_flow.clone());
                        self.message_flow_history.push(msg_flow);
                        
                        // Keep only recent messages (last 5 minutes)
                        let cutoff = Utc::now() - chrono::Duration::minutes(5);
                        self.message_flow_history.retain(|msg| msg.timestamp > cutoff);
                    }
                }
                "metrics.update" => {
                    if let Ok(metrics) = serde_json::from_value::<SystemMetrics>(event["data"].clone()) {
                        self.system_metrics = metrics;
                    }
                }
                _ => debug!("Unhandled telemetry event type: {}", event_type),
            }
        }
    }
    
    /// Convert agent data to graph format and push to GraphServiceActor
    fn push_to_graph(&self) {
        let agents: Vec<AgentStatus> = self.agent_cache.values().cloned().collect();
        
        if !agents.is_empty() {
            // Send to GraphServiceActor for GPU processing
            self.graph_service_addr.do_send(UpdateBotsGraph { agents });
        }
    }
    
    /// Compute differential updates to minimize data transfer
    fn compute_delta(&mut self) -> TelemetryDelta {
        TelemetryDelta {
            added_agents: self.pending_additions.drain(..).collect(),
            removed_agents: self.pending_removals.drain(..).collect(),
            updated_agents: self.pending_updates.drain(..).collect(),
            new_messages: self.pending_messages.drain(..).collect(),
        }
    }
    
    /// Send MCP request through WebSocket
    #[allow(dead_code)]
    async fn send_mcp_request(&self, method: &str, params: Value) -> Result<Value, String> {
        if let Some(ws_conn) = &self.ws_connection {
            let request = json!({
                "jsonrpc": "2.0",
                "id": Uuid::new_v4().to_string(),
                "method": method,
                "params": params
            });
            
            let mut ws = ws_conn.write().await;
            ws.send(WsMessage::Text(request.to_string())).await
                .map_err(|e| format!("Failed to send MCP request: {}", e))?;
            
            // For now, return success - actual response handling would be via stream
            Ok(json!({ "success": true }))
        } else {
            Err("No WebSocket connection established".to_string())
        }
    }

    /// Execute MCP tool call with error handling and response parsing
    #[allow(dead_code)]
    async fn execute_mcp_tool(&self, tool_name: &str, arguments: Value) -> Result<Value, String> {
        if !self.is_initialized {
            return Err("MCP client not initialized".to_string());
        }

        debug!("Executing MCP tool: {} with args: {}", tool_name, arguments);

        // Create MCP tool call request
        let params = json!({
            "name": tool_name,
            "arguments": arguments
        });

        let request = crate::types::claude_flow::McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Uuid::new_v4().to_string(),
            method: "tools/call".to_string(),
            params: params,
        };

        // Execute request through transport
        match self._client.send_mcp_request(&request).await {
            Ok(response) => {
                if let Some(result) = response.get("result") {
                    debug!("MCP tool {} executed successfully", tool_name);
                    Ok(result.clone())
                } else if let Some(error) = response.get("error") {
                    error!("MCP tool {} failed: {:?}", tool_name, error);
                    Err(format!("MCP tool error: {:?}", error))
                } else {
                    error!("MCP tool {} returned no result or error", tool_name);
                    Err("No result from MCP tool".to_string())
                }
            }
            Err(e) => {
                error!("Failed to execute MCP tool {}: {}", tool_name, e);
                Err(format!("Transport error: {}", e))
            }
        }
    }

    /// Update system metrics with real-time data
    fn update_system_metrics(&mut self, agents: &[AgentStatus]) {
        let active_agents = agents.iter().filter(|a| a.status == "active").count() as u32;
        let _total_agents = agents.len() as u32;

        // Calculate message rate based on recent message flow
        let recent_messages = self.message_flow_history
            .iter()
            .filter(|msg| (Utc::now() - msg.timestamp).num_seconds() <= 60)
            .count() as f32 / 60.0; // messages per second

        // Calculate average latency
        let avg_latency = if self.message_flow_history.is_empty() {
            0.0
        } else {
            self.message_flow_history.iter().map(|msg| msg.latency_ms).sum::<f32>() 
                / self.message_flow_history.len() as f32
        };

        // Calculate error rate
        let total_messages = self.message_flow_history.len() as f32;
        let error_messages = self.message_flow_history
            .iter()
            .filter(|msg| msg.latency_ms > 1000.0) // Consider >1s as error
            .count() as f32;
        let error_rate = if total_messages > 0.0 { error_messages / total_messages } else { 0.0 };

        // Update system metrics
        self.system_metrics = SystemMetrics {
            active_agents,
            message_rate: recent_messages,
            average_latency: avg_latency,
            error_rate,
            network_health: 1.0 - error_rate, // Simple network health calculation
            cpu_usage: 0.0, // TODO: Implement actual CPU monitoring
            memory_usage: 0.0, // TODO: Implement actual memory monitoring
            gpu_usage: None, // TODO: Implement GPU monitoring
        };
    }

    /// Ensure WebSocket connection to Claude Flow with exponential backoff retry (WebSocket only)
    #[allow(dead_code)]
    async fn ensure_connection(&mut self) -> Result<(), String> {
        if !self.is_connected {
            info!("Attempting WebSocket reconnection to Claude Flow...");
            
            // Exponential backoff retry logic
            let mut retry_delay = 1000; // Start with 1 second
            let max_delay = 30000; // Max 30 seconds
            let max_retries = 3;
            
            for attempt in 1..=max_retries {
                match Self::connect_to_claude_flow().await {
                    Ok(ws_stream) => {
                        info!("WebSocket reconnection successful on attempt {}", attempt);
                        self.ws_connection = Some(Arc::new(RwLock::new(ws_stream)));
                        self.is_connected = true;
                        self.is_initialized = true;
                        return Ok(());
                    }
                    Err(e) => {
                        warn!("WebSocket connection attempt {} failed: {}", attempt, e);
                        
                        if attempt < max_retries {
                            info!("Waiting {}ms before retry...", retry_delay);
                            tokio::time::sleep(tokio::time::Duration::from_millis(retry_delay)).await;
                            retry_delay = std::cmp::min(retry_delay * 2, max_delay);
                        }
                    }
                }
            }
            
            Err(format!("WebSocket connection failed after {} attempts. No TCP/stdio fallback available.", max_retries))
        } else {
            Ok(())
        }
    }
    
    /// Return empty agent list when no connection available (no mock data)
    fn get_empty_agents(&self) -> Vec<AgentStatus> {
        Vec::new()
    }

}


// Internal messages for Claude Flow integration
#[derive(Message)]
#[rtype(result = "()")]
struct ProcessTelemetryStream;

#[derive(Message)]
#[rtype(result = "()")]
struct ProcessDeltaUpdates;

#[derive(Message)]
#[rtype(result = "()")]
struct ConnectionEstablished {
    ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

#[derive(Message)]
#[rtype(result = "()")]
struct ConnectionFailed;

// Differential update structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TelemetryDelta {
    added_agents: Vec<AgentStatus>,
    removed_agents: Vec<String>,
    updated_agents: Vec<AgentUpdate>,
    new_messages: Vec<MessageFlowEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentUpdate {
    agent_id: String,
    status: String,
    current_task: Option<crate::types::claude_flow::TaskReference>,
}

impl Actor for EnhancedClaudeFlowActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Enhanced Claude Flow Actor started - initializing WebSocket-only connection to Claude Flow");
        
        // Initialize direct MCP WebSocket connection (no fallback protocols)
        self.initialize_connection(ctx);
        
        // Schedule periodic health checks for the WebSocket connection
        ctx.run_interval(Duration::from_secs(30), |actor, ctx| {
            if !actor.is_connected {
                info!("WebSocket connection lost, attempting reconnection...");
                ctx.address().do_send(RetryMCPConnection);
            }
        });
    }
}

// Message Handlers

impl Handler<InitializeSwarm> for EnhancedClaudeFlowActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: InitializeSwarm, _ctx: &mut Self::Context) -> Self::Result {
        info!("Initializing swarm with topology: {}", msg.topology);

        // For now, skip the async MCP call and directly use mock data
        // This avoids the lifetime issues with async methods on &self
        Box::pin(fut::ready(()).into_actor(self).map(move |_, actor, _ctx| {
            info!("Swarm initialization (mock mode)");
            
            // Generate swarm ID
            actor.swarm_id = Some(Uuid::new_v4().to_string());
            
            // Start with empty agents until real data arrives
            let agents = actor.get_empty_agents();
            
            // Update graph service with new agents
            actor.graph_service_addr.do_send(UpdateBotsGraph { agents: agents.clone() });
            
            // Update system metrics
            actor.update_system_metrics(&agents);
            
            info!("Swarm initialized successfully with {} agents", agents.len());
            Ok(())
        }))
    }
}

impl Handler<GetSwarmStatus> for EnhancedClaudeFlowActor {
    type Result = ResponseActFuture<Self, Result<SwarmStatus, String>>;

    fn handle(&mut self, _msg: GetSwarmStatus, _ctx: &mut Self::Context) -> Self::Result {
        info!("Getting swarm status");

        // For now, directly return cached status without async MCP call
        Box::pin(fut::ready(()).into_actor(self).map(|_, actor, _ctx| {
            let status = SwarmStatus {
                swarm_id: actor.swarm_id.clone().unwrap_or_else(|| "default".to_string()),
                active_agents: actor.system_metrics.active_agents,
                total_agents: actor.agent_cache.len() as u32,
                topology: "hierarchical".to_string(),
                health_score: actor.system_metrics.network_health,
                coordination_efficiency: 0.95, // TODO: Calculate from real data
            };
            
            info!("Swarm status retrieved: {:?}", status);
            Ok(status)
        }))
    }
}

impl Handler<SwarmMonitor> for EnhancedClaudeFlowActor {
    type Result = Result<SwarmMonitorData, String>;

    fn handle(&mut self, _msg: SwarmMonitor, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Collecting swarm monitoring data");

        let agent_states: HashMap<String, String> = self.agent_cache
            .iter()
            .map(|(id, agent)| (id.clone(), agent.status.clone()))
            .collect();

        let monitor_data = SwarmMonitorData {
            timestamp: Utc::now(),
            agent_states,
            message_flow: self.message_flow_history.clone(),
            coordination_patterns: self.coordination_patterns.clone(),
            system_metrics: self.system_metrics.clone(),
        };

        Ok(monitor_data)
    }
}

impl Handler<ProcessTelemetryStream> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, _msg: ProcessTelemetryStream, _ctx: &mut Self::Context) -> Self::Result {
        // Process any incoming telemetry and push to graph
        if self.is_connected && !self.agent_cache.is_empty() {
            self.push_to_graph();
        }
    }
}

impl Handler<ProcessDeltaUpdates> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, _msg: ProcessDeltaUpdates, _ctx: &mut Self::Context) -> Self::Result {
        // Process differential updates
        let delta = self.compute_delta();
        
        // Only send if there are changes
        if !delta.added_agents.is_empty() || 
           !delta.removed_agents.is_empty() || 
           !delta.updated_agents.is_empty() {
            
            // Update graph with changes
            self.push_to_graph();
            
            debug!("Processed delta: {} additions, {} removals, {} updates", 
                   delta.added_agents.len(), 
                   delta.removed_agents.len(), 
                   delta.updated_agents.len());
        }
    }
}

impl Handler<ConnectionEstablished> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, msg: ConnectionEstablished, ctx: &mut Self::Context) -> Self::Result {
        info!("Claude Flow WebSocket connection established");
        self.ws_connection = Some(Arc::new(RwLock::new(msg.ws_stream)));
        self.is_connected = true;
        self.is_initialized = true;
        
        // Start streaming
        self.start_real_time_streaming(ctx);
        
        // Subscribe to telemetry events
        let ws_conn = self.ws_connection.clone();
        let _addr = ctx.address();
        
        tokio::spawn(async move {
            if let Some(ws) = ws_conn {
                let subscribe_req = json!({
                    "jsonrpc": "2.0",
                    "id": Uuid::new_v4().to_string(),
                    "method": "telemetry.subscribe",
                    "params": {
                        "events": ["agent.*", "message.*", "metrics.*"]
                    }
                });
                
                let mut ws = ws.write().await;
                if let Err(e) = ws.send(WsMessage::Text(subscribe_req.to_string())).await {
                    error!("Failed to subscribe to telemetry: {}", e);
                }
            }
        });
    }
}

impl Handler<ConnectionFailed> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, _msg: ConnectionFailed, ctx: &mut Self::Context) -> Self::Result {
        warn!("Claude Flow connection failed, running in disconnected mode");
        self.is_connected = false;
        self.is_initialized = true; // Still initialized, but in fallback mode
        
        // Start with empty agents in fallback mode
        let agents = self.get_empty_agents();
        self.graph_service_addr.do_send(UpdateBotsGraph { agents });
        
        // Start polling with mock data
        self.start_real_time_streaming(ctx);
        
        // Schedule reconnection attempts
        ctx.run_interval(Duration::from_secs(30), |actor, ctx| {
            if !actor.is_connected {
                ctx.address().do_send(RetryMCPConnection);
            }
        });
    }
}

impl Handler<MetricsCollect> for EnhancedClaudeFlowActor {
    type Result = Result<SystemMetrics, String>;

    fn handle(&mut self, _msg: MetricsCollect, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Collecting system metrics");
        Ok(self.system_metrics.clone())
    }
}

// Additional handlers for other MCP tools...
impl Handler<GetAgentMetrics> for EnhancedClaudeFlowActor {
    type Result = Result<Vec<AgentMetrics>, String>;

    fn handle(&mut self, _msg: GetAgentMetrics, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Getting agent metrics");

        let metrics: Vec<AgentMetrics> = self.agent_cache
            .values()
            .map(|agent| AgentMetrics {
                agent_id: agent.agent_id.clone(),
                performance_score: (agent.success_rate / 100.0),
                tasks_completed: agent.completed_tasks_count,
                success_rate: (agent.success_rate / 100.0),
                resource_utilization: 0.7, // TODO: Calculate from real data
                token_usage: agent.token_usage.total, // Approximate
            })
            .collect();

        Ok(metrics)
    }
}

impl Handler<RetryMCPConnection> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, _msg: RetryMCPConnection, ctx: &mut Self::Context) -> Self::Result {
        info!("Retrying MCP connection");
        
        // Reinitialize connection
        if !self.is_connected {
            self.initialize_connection(ctx);
        }
    }
}

impl Handler<GetCachedAgentStatuses> for EnhancedClaudeFlowActor {
    type Result = Result<Vec<AgentStatus>, String>;

    fn handle(&mut self, _msg: GetCachedAgentStatuses, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Returning cached agent statuses");
        Ok(self.agent_cache.values().cloned().collect())
    }
}