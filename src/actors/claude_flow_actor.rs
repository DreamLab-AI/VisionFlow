//! Refactored Claude Flow Actor TCP - Application Logic Only
//! 
//! This actor now focuses solely on:
//! - Agent data management and caching
//! - Polling coordination and business logic  
//! - System metrics and telemetry
//! - Graph service integration
//! - Application-level error handling and retry logic
//! 
//! Low-level concerns are delegated to:
//! - TcpConnectionActor: TCP stream management
//! - JsonRpcClient: MCP protocol and message correlation

use actix::prelude::*;
use std::time::Duration;
use log::{info, error, debug, warn};
use std::collections::HashMap;
use chrono::{Utc, DateTime};
use uuid::Uuid;
use rand::Rng;
use serde_json::{json, Value};

use crate::types::claude_flow::{ClaudeFlowClient, AgentStatus, AgentProfile, AgentType, PerformanceMetrics, TokenUsage};
use crate::types::mcp_responses::{McpResponse, McpContentResult, AgentListResponse, McpParseError};
use crate::actors::messages::*;
use crate::actors::GraphServiceActor;
use crate::actors::tcp_connection_actor::{TcpConnectionActor, TcpConnectionEvent, TcpConnectionEventType, EstablishConnection, SubscribeToEvents};
use crate::actors::jsonrpc_client::{JsonRpcClient, ConnectToTcpActor, InitializeMcpSession, CallTool, ClientInfo};
use crate::actors::messages::UpdateAgentCache;

/// Refactored ClaudeFlowActorTcp - pure application logic
pub struct ClaudeFlowActorTcp {
    _client: ClaudeFlowClient,
    graph_service_addr: Addr<GraphServiceActor>,
    
    /// Application state
    is_connected: bool,
    is_initialized: bool,
    swarm_id: Option<String>,
    swarm_topology: Option<String>,
    
    /// Sub-actors for specific concerns
    tcp_actor: Option<Addr<TcpConnectionActor>>,
    jsonrpc_client: Option<Addr<JsonRpcClient>>,
    
    /// Polling configuration
    polling_interval: Duration,
    last_poll: DateTime<Utc>,
    
    /// Agent management
    agent_cache: HashMap<String, AgentStatus>,
    pending_additions: Vec<AgentStatus>,
    pending_removals: Vec<String>,
    pending_updates: Vec<AgentUpdate>,
    
    /// System telemetry
    system_metrics: SystemMetrics,
    message_flow_history: Vec<MessageFlowEvent>,
    coordination_patterns: Vec<CoordinationPattern>,
    pending_messages: Vec<MessageFlowEvent>,
    
    /// Error tracking for circuit breaker logic
    consecutive_poll_failures: u32,
    last_successful_poll: Option<DateTime<Utc>>,
}

impl ClaudeFlowActorTcp {
    pub fn new(client: ClaudeFlowClient, graph_service_addr: Addr<GraphServiceActor>) -> Self {
        info!("Creating refactored ClaudeFlowActorTcp with separated concerns");
        Self {
            _client: client,
            graph_service_addr,
            is_connected: false,
            is_initialized: false,
            swarm_id: None,
            swarm_topology: None,
            tcp_actor: None,
            jsonrpc_client: None,
            polling_interval: Duration::from_millis(1000),
            last_poll: Utc::now(),
            agent_cache: HashMap::new(),
            pending_additions: Vec::new(),
            pending_removals: Vec::new(),
            pending_updates: Vec::new(),
            system_metrics: SystemMetrics::default(),
            message_flow_history: Vec::new(),
            coordination_patterns: Vec::new(),
            pending_messages: Vec::new(),
            consecutive_poll_failures: 0,
            last_successful_poll: None,
        }
    }
    
    /// Initialize sub-actors and establish connections
    fn initialize_sub_actors(&mut self, ctx: &mut Context<Self>) {
        info!("Initializing sub-actors for TCP connection and JSON-RPC handling");

        // CRITICAL FIX: We're in visionflow container, MCP is in multi-agent-container
        // Use container hostname for better resilience in the docker_ragflow network
        let host = std::env::var("MCP_HOST")
            .unwrap_or_else(|_| {
                // We are ALWAYS in Docker when WebXR is running
                // The MCP server is ALWAYS in multi-agent-container
                warn!("MCP_HOST not set, using multi-agent-container as default");
                "multi-agent-container".to_string()
            });
        let port = std::env::var("MCP_TCP_PORT")
            .unwrap_or_else(|_| "9500".to_string())
            .parse::<u16>()
            .unwrap_or(9500);

        info!("Connecting to MCP server at {}:{} (from logseq container)", host, port);
        
        // Create TCP connection actor
        let tcp_actor = TcpConnectionActor::new(host, port).start();
        
        // Subscribe to TCP connection events
        let subscriber = ctx.address().recipient::<TcpConnectionEvent>();
        tcp_actor.do_send(SubscribeToEvents { subscriber });
        
        // Create JSON-RPC client
        let jsonrpc_client = JsonRpcClient::new()
            .with_client_info(ClientInfo {
                name: "visionflow".to_string(),
                version: "1.0.0".to_string(),
            })
            .start();
        
        // Connect JSON-RPC client to TCP actor
        jsonrpc_client.do_send(ConnectToTcpActor { 
            tcp_actor: tcp_actor.clone()
        });
        
        // Store actor references
        self.tcp_actor = Some(tcp_actor);
        self.jsonrpc_client = Some(jsonrpc_client);
        
        // Initiate TCP connection
        if let Some(ref tcp_actor) = self.tcp_actor {
            tcp_actor.do_send(EstablishConnection);
        }
    }

    /// Convert Agent struct to AgentStatus (new type-safe method)
    fn agent_to_status(agent: &crate::services::bots_client::Agent) -> AgentStatus {
        
        let agent_type = match agent.agent_type.as_str() {
            "coordinator" | "task-orchestrator" => AgentType::Coordinator,
            "researcher" => AgentType::Researcher,
            "coder" | "worker" => AgentType::Coder,
            "analyst" | "analyzer" | "code-analyzer" | "specialist" => AgentType::Analyst,
            "architect" => AgentType::Architect,
            "tester" => AgentType::Tester,
            "reviewer" => AgentType::Reviewer,
            "optimizer" => AgentType::Optimizer,
            "documenter" => AgentType::Documenter,
            _ => AgentType::Coordinator,
        };
        
        AgentStatus {
            agent_id: agent.id.clone(),
            profile: AgentProfile {
                name: agent.name.clone(),
                agent_type,
                capabilities: Vec::new(),
            },
            status: agent.status.clone(),
            active_tasks_count: 0,
            completed_tasks_count: 0,
            failed_tasks_count: 0,
            success_rate: 100.0,
            timestamp: Utc::now(),
            current_task: None,
            cpu_usage: agent.cpu_usage,
            memory_usage: agent.memory_usage,
            health: agent.health,
            activity: agent.workload,
            tasks_active: 0,
            performance_metrics: PerformanceMetrics {
                tasks_completed: 0,
                success_rate: 100.0,
            },
            token_usage: TokenUsage {
                total: 1000,
                token_rate: 0.0,
            },
            swarm_id: None,
            agent_mode: Some("autonomous".to_string()),
            parent_queen_id: None,
            processing_logs: None,
            total_execution_time: 0,
        }
    }

    /// Parse legacy response format for backward compatibility
    fn parse_legacy_response(&self, response: &Value) -> AgentListResponse {
        if let Some(result) = response.get("result") {
            if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
                if let Some(first_content) = content.first() {
                    if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                        match serde_json::from_str::<Value>(text) {
                            Ok(parsed) => {
                                if let Some(agents_array) = parsed.get("agents").and_then(|a| a.as_array()) {
                                    let mut agents = Vec::new();
                                    for agent_val in agents_array {
                                        if let Ok(agent) = serde_json::from_value::<crate::services::bots_client::Agent>(agent_val.clone()) {
                                            agents.push(agent);
                                        }
                                    }
                                    return AgentListResponse { agents };
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse legacy nested JSON: {}", e);
                            }
                        }
                    }
                }
            }
        }
        // Return empty list instead of mock data
        warn!("No valid agent data found in response, returning empty agent list");
        AgentListResponse { agents: Vec::new() }
    }
    
    /// Convert MCP agent format to VisionFlow AgentStatus (preserved from original)
    fn mcp_agent_to_status(agent_data: &Value) -> Result<AgentStatus, String> {
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
            
        // Parse agent type
        let agent_type = match agent_type_str {
            "coordinator" | "task-orchestrator" => AgentType::Coordinator,
            "researcher" => AgentType::Researcher,
            "coder" | "worker" => AgentType::Coder,
            "analyst" | "analyzer" | "code-analyzer" | "specialist" => AgentType::Analyst,
            "architect" => AgentType::Architect,
            "tester" => AgentType::Tester,
            "reviewer" => AgentType::Reviewer,
            "optimizer" => AgentType::Optimizer,
            "documenter" => AgentType::Documenter,
            _ => AgentType::Coordinator,
        };
        
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
            cpu_usage: 10.0 + rng.gen::<f32>() * 20.0,
            memory_usage: 20.0 + rng.gen::<f32>() * 30.0,
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
    
    /// Handle agent updates (preserved business logic)
    fn handle_agent_update(&mut self, update: AgentUpdate) {
        debug!("Processing agent update for agent: {}", update.agent_id);
        
        self.pending_updates.push(update.clone());
        self.system_metrics.active_agents = self.agent_cache.len() as u32;
        
        let flow_event = MessageFlowEvent {
            id: Uuid::new_v4().to_string(),
            from_agent: "system".to_string(),
            to_agent: update.agent_id.clone(),
            message_type: "status_update".to_string(),
            priority: 5,
            timestamp: update.timestamp,
            latency_ms: 0.0,
        };
        
        self.message_flow_history.push(flow_event.clone());
        self.pending_messages.push(flow_event);
        
        if self.message_flow_history.len() > 1000 {
            self.message_flow_history.drain(..100);
        }
        
        let pattern_id_opt = self.get_or_create_coordination_pattern(&update.agent_id);
        if let Some(pattern_id) = pattern_id_opt {
            let mut participants_to_check = Vec::new();
            let mut pattern_found = false;
            
            for pattern in &self.coordination_patterns {
                if pattern.id == pattern_id {
                    participants_to_check = pattern.participants.clone();
                    pattern_found = true;
                    break;
                }
            }
            
            if pattern_found {
                let progress = self.calculate_coordination_progress(&participants_to_check);
                
                if let Some(pattern) = self.coordination_patterns.iter_mut().find(|p| p.id == pattern_id) {
                    pattern.progress = progress;
                    if progress >= 1.0 && pattern.status == "active" {
                        pattern.status = "completed".to_string();
                    }
                }
            }
        }
        
        info!("Agent update processed: {} pending updates, {} message events", 
              self.pending_updates.len(), self.pending_messages.len());
    }
    
    /// Get or create a coordination pattern for an agent (preserved logic)
    fn get_or_create_coordination_pattern(&mut self, agent_id: &str) -> Option<String> {
        for pattern in &self.coordination_patterns {
            if pattern.participants.contains(&agent_id.to_string()) && pattern.status == "active" {
                return Some(pattern.id.clone());
            }
        }
        
        if self.agent_cache.len() > 1 {
            let pattern = CoordinationPattern {
                id: Uuid::new_v4().to_string(),
                pattern_type: "mesh".to_string(),
                participants: vec![agent_id.to_string()],
                status: "forming".to_string(),
                progress: 0.0,
            };
            let pattern_id = pattern.id.clone();
            self.coordination_patterns.push(pattern);
            Some(pattern_id)
        } else {
            None
        }
    }
    
    /// Calculate coordination progress (preserved logic)
    fn calculate_coordination_progress(&self, participants: &[String]) -> f32 {
        let active_count = participants.iter()
            .filter(|id| self.agent_cache.get(*id).map_or(false, |agent| agent.status == "active"))
            .count();
        
        if participants.is_empty() {
            0.0
        } else {
            active_count as f32 / participants.len() as f32
        }
    }
    
    /// Process pending queues (preserved logic)
    fn process_pending_queues(&mut self) {
        if !self.pending_additions.is_empty() {
            info!("Processing {} pending agent additions", self.pending_additions.len());
            self.pending_additions.clear();
        }
        
        if !self.pending_removals.is_empty() {
            info!("Processing {} pending agent removals", self.pending_removals.len());
            for agent_id in &self.pending_removals {
                self.agent_cache.remove(agent_id);
            }
            self.pending_removals.clear();
        }
        
        if !self.pending_updates.is_empty() {
            info!("Processing {} pending agent updates", self.pending_updates.len());
            self.pending_updates.clear();
        }
        
        if !self.pending_messages.is_empty() {
            info!("Processing {} pending message flow events", self.pending_messages.len());
            self.pending_messages.clear();
        }
    }
    
    /// Calculate average latency from recent message flow events (preserved logic)
    fn calculate_average_latency(&self) -> f32 {
        if self.message_flow_history.is_empty() {
            return 0.0;
        }
        
        let recent_events = self.message_flow_history.iter()
            .rev()
            .take(100)
            .collect::<Vec<_>>();
            
        let total_latency: f32 = recent_events.iter()
            .map(|event| event.latency_ms)
            .sum();
            
        total_latency / recent_events.len() as f32
    }
    
    /// Calculate error rate (preserved logic)
    fn calculate_error_rate(&self) -> f32 {
        let total_attempts = self.consecutive_poll_failures + 1;
        let error_count = self.consecutive_poll_failures.min(10);
        
        if total_attempts == 0 {
            0.0
        } else {
            error_count as f32 / total_attempts as f32
        }
    }
    
    /// Poll agent statuses using JSON-RPC client
    fn poll_agent_statuses(&mut self, ctx: &mut Context<Self>) {
        if !self.is_connected || !self.is_initialized {
            debug!("Skipping agent status poll - not connected or initialized");
            return;
        }
        
        if self.consecutive_poll_failures > 10 {
            if let Some(last_success) = self.last_successful_poll {
                let time_since_success = Utc::now().signed_duration_since(last_success);
                if time_since_success.num_seconds() < 30 {
                    debug!("Circuit breaker active - skipping poll");
                    return;
                }
            }
        }
        
        debug!("Polling agent statuses via JSON-RPC client");
        
        self.system_metrics.active_agents = self.agent_cache.len() as u32;
        self.process_pending_queues();
        
        if let Some(ref jsonrpc_client) = self.jsonrpc_client {
            let params = json!({
                "filter": "all"
            });
            
            let graph_addr = self.graph_service_addr.clone();
            let ctx_addr = ctx.address();
            let client = jsonrpc_client.clone();
            
            tokio::spawn(async move {
                match client.send(CallTool {
                    tool_name: "agent_list".to_string(),
                    params,
                    timeout: Some(Duration::from_secs(10)),
                }).await {
                    Ok(Ok(response)) => {
                        debug!("Received agent list response: {:?}", response);
                        ctx_addr.do_send(ProcessAgentListResponse { response });
                    }
                    Ok(Err(e)) => {
                        error!("Tool call failed: {}", e);
                        ctx_addr.do_send(RecordPollFailure);
                    }
                    Err(e) => {
                        error!("Actor communication error: {}", e);
                        ctx_addr.do_send(RecordPollFailure);
                    }
                }
            });
        } else {
            warn!("No JSON-RPC client available for polling");
        }
    }
}

/// New message types for refactored actor
#[derive(Message)]
#[rtype(result = "()")]
struct ProcessAgentListResponse {
    response: Value,
}

impl Actor for ClaudeFlowActorTcp {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Refactored ClaudeFlowActorTcp started");
        
        // Initialize sub-actors
        self.initialize_sub_actors(ctx);
        
        // Schedule periodic status updates using run_later to ensure proper runtime context
        ctx.run_later(Duration::from_millis(100), |act, ctx| {
            // Now schedule the interval from within the actor's execution context
            ctx.run_interval(act.polling_interval, |act, ctx| {
                if act.is_connected && act.is_initialized {
                    act.poll_agent_statuses(ctx);
                }
            });
        });
    }
    
    fn stopped(&mut self, _: &mut Self::Context) {
        info!("Refactored ClaudeFlowActorTcp stopping");
        // Sub-actors will handle their own cleanup
    }
}

impl Handler<TcpConnectionEvent> for ClaudeFlowActorTcp {
    type Result = ();
    
    fn handle(&mut self, msg: TcpConnectionEvent, ctx: &mut Self::Context) {
        match msg.event_type {
            TcpConnectionEventType::Connected => {
                info!("TCP connection established, initializing MCP session");
                self.is_connected = true;
                
                // Initialize MCP session
                if let Some(ref jsonrpc_client) = self.jsonrpc_client {
                    let client = jsonrpc_client.clone();
                    let addr = ctx.address();
                    
                    tokio::spawn(async move {
                        match client.send(InitializeMcpSession).await {
                            Ok(Ok(())) => {
                                info!("MCP session initialized successfully");
                                addr.do_send(MCPSessionReady);
                            }
                            Ok(Err(e)) => {
                                error!("Failed to initialize MCP session: {}", e);
                            }
                            Err(e) => {
                                error!("Actor communication error: {}", e);
                            }
                        }
                    });
                }
            }
            TcpConnectionEventType::Disconnected => {
                warn!("TCP connection lost");
                self.is_connected = false;
                self.is_initialized = false;
            }
            TcpConnectionEventType::MessageReceived => {
                // Messages are handled by JsonRpcClient
            }
            TcpConnectionEventType::MessageSent => {
                debug!("Message sent successfully");
            }
            TcpConnectionEventType::Error(error) => {
                error!("TCP connection error: {}", error);
            }
        }
    }
}

#[derive(Message)]
#[rtype(result = "()")]
struct MCPSessionReady;

impl Handler<MCPSessionReady> for ClaudeFlowActorTcp {
    type Result = ();
    
    fn handle(&mut self, _: MCPSessionReady, _ctx: &mut Self::Context) {
        info!("MCP session is ready");
        self.is_initialized = true;
        self.consecutive_poll_failures = 0;
        self.last_successful_poll = Some(Utc::now());
    }
}

impl Handler<ProcessAgentListResponse> for ClaudeFlowActorTcp {
    type Result = ();
    
    fn handle(&mut self, msg: ProcessAgentListResponse, _ctx: &mut Self::Context) {
        // Type-safe MCP response parsing
        let response_clone = msg.response.clone();
        let agent_list = match serde_json::from_value::<McpResponse<McpContentResult>>(msg.response) {
            Ok(mcp_response) => {
                match mcp_response.into_result() {
                    Ok(content_result) => {
                        match content_result.extract_data::<AgentListResponse>() {
                            Ok(agents) => {
                                info!("Successfully parsed MCP response with {} agents", agents.agents.len());
                                agents
                            }
                            Err(e) => {
                                warn!("Failed to extract agent data from MCP response: {}", e);
                                // Return empty agent list as fallback
                                AgentListResponse { agents: Vec::new() }
                            }
                        }
                    }
                    Err(mcp_error) => {
                        error!("MCP response returned error: {}", mcp_error.message);
                        AgentListResponse { agents: Vec::new() }
                    }
                }
            }
            Err(e) => {
                warn!("Failed to parse as MCP response format: {}", e);
                // Fallback to legacy parsing for backward compatibility
                self.parse_legacy_response(&response_clone)
            }
        };
        
        // Process agents and send to graph
        if !agent_list.agents.is_empty() {
            let mut agent_statuses = Vec::new();
            let mut parsing_errors = 0u32;
            
            for (idx, agent) in agent_list.agents.iter().enumerate() {
                // Convert Agent to AgentStatus
                let agent_status = Self::agent_to_status(agent);
                info!("Agent [{}] {} - Status: {}, Type: {:?}", 
                      idx, agent_status.agent_id, agent_status.status, agent_status.profile.agent_type);
                agent_statuses.push(agent_status);
            }
            
            // Send graph update
            let message = UpdateBotsGraph {
                agents: agent_statuses.clone()
            };
            
            info!("🔄 Sending graph update: {} agents parsed ({} errors)", 
                  agent_statuses.len(), parsing_errors);
            
            self.graph_service_addr.do_send(message);
            
            // Update cache
            if !agent_statuses.is_empty() {
                self.agent_cache.clear();
                for agent in agent_statuses {
                    self.agent_cache.insert(agent.agent_id.clone(), agent);
                }
            }
            
            // Mark poll as successful
            self.consecutive_poll_failures = 0;
            self.last_successful_poll = Some(Utc::now());
        } else {
            info!("No agent data available - sending empty update to graph service");
            self.graph_service_addr.do_send(UpdateBotsGraph {
                agents: Vec::new()
            });
        }
    }
}

impl Handler<RecordPollFailure> for ClaudeFlowActorTcp {
    type Result = ();
    
    fn handle(&mut self, _: RecordPollFailure, _ctx: &mut Self::Context) {
        self.consecutive_poll_failures += 1;
        warn!("Poll failure recorded - {} consecutive failures", 
              self.consecutive_poll_failures);
    }
}

impl Handler<InitializeSwarm> for ClaudeFlowActorTcp {
    type Result = ResponseFuture<Result<String, String>>;

    fn handle(&mut self, msg: InitializeSwarm, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_connected || !self.is_initialized {
            return Box::pin(async move { 
                Err("Not connected to Claude Flow".to_string()) 
            });
        }

        let jsonrpc_client = match self.jsonrpc_client.clone() {
            Some(client) => client,
            None => return Box::pin(async move { 
                Err("No JSON-RPC client available".to_string()) 
            }),
        };
        
        Box::pin(async move {
            let params = json!({
                "topology": msg.topology,
                "maxAgents": msg.max_agents,
                "strategy": msg.strategy
            });
            
            match jsonrpc_client.send(CallTool {
                tool_name: "swarm_init".to_string(),
                params,
                timeout: Some(Duration::from_secs(10)),
            }).await {
                Ok(Ok(response)) => {
                    if let Some(swarm_id) = response.get("swarmId").and_then(|s| s.as_str()) {
                        info!("Swarm initialized successfully: {}", swarm_id);
                        Ok(swarm_id.to_string())
                    } else {
                        Ok(format!("swarm_{}", Uuid::new_v4()))
                    }
                }
                Ok(Err(e)) => {
                    error!("Failed to initialize swarm: {}", e);
                    Err(e)
                }
                Err(e) => {
                    error!("Actor communication error: {}", e);
                    Err(e.to_string())
                }
            }
        })
    }
}

impl Handler<GetSwarmStatus> for ClaudeFlowActorTcp {
    type Result = ResponseFuture<Result<SwarmStatus, String>>;

    fn handle(&mut self, _: GetSwarmStatus, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_connected {
            return Box::pin(async move { 
                Err("Not connected to Claude Flow".to_string()) 
            });
        }
        
        let swarm_status = SwarmStatus {
            swarm_id: self.swarm_id.clone().unwrap_or_default(),
            active_agents: self.agent_cache.len() as u32,
            total_agents: self.agent_cache.len() as u32,
            topology: self.swarm_topology.clone().unwrap_or_else(|| "mesh".to_string()),
            health_score: if self.is_connected && self.is_initialized { 1.0 } else { 0.0 },
            coordination_efficiency: 0.85,
        };
        
        Box::pin(async move { Ok(swarm_status) })
    }
}

impl Handler<UpdateAgentCache> for ClaudeFlowActorTcp {
    type Result = ();

    fn handle(&mut self, msg: UpdateAgentCache, _ctx: &mut Self::Context) {
        debug!("Updating agent cache with {} agents", msg.agents.len());
        
        self.agent_cache.clear();
        for agent in msg.agents {
            self.agent_cache.insert(agent.agent_id.clone(), agent);
        }
        
        debug!("Agent cache updated: {} agents cached", self.agent_cache.len());
    }
}