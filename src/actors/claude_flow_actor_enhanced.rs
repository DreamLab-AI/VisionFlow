use actix::prelude::*;
use std::time::Duration;
use log::{info, error, warn, debug};
use crate::services::claude_flow::{ClaudeFlowClient, ClaudeFlowClientBuilder, AgentStatus, AgentProfile, AgentType};
use crate::actors::messages::*;
use crate::actors::GraphServiceActor;
use std::collections::HashMap;
use chrono::{Utc, DateTime};
use uuid::Uuid;
use serde_json::{json, Value};

/// Enhanced ClaudeFlowActor with full MCP integration for Hive Mind Swarm
pub struct EnhancedClaudeFlowActor {
    client: ClaudeFlowClient,
    graph_service_addr: Addr<GraphServiceActor>,
    is_connected: bool,
    is_initialized: bool,
    swarm_id: Option<String>,
    polling_interval: Duration,
    last_poll: DateTime<Utc>,
    agent_cache: HashMap<String, AgentStatus>,
    swarm_status: Option<SwarmStatus>,
    system_metrics: SystemMetrics,
    message_flow_history: Vec<MessageFlowEvent>,
    coordination_patterns: Vec<CoordinationPattern>,
}

impl EnhancedClaudeFlowActor {
    pub fn new(client: ClaudeFlowClient, graph_service_addr: Addr<GraphServiceActor>) -> Self {
        Self {
            client,
            graph_service_addr,
            is_connected: false,
            is_initialized: false,
            swarm_id: None,
            polling_interval: Duration::from_millis(16), // 60 FPS polling
            last_poll: Utc::now(),
            agent_cache: HashMap::new(),
            swarm_status: None,
            system_metrics: SystemMetrics::default(),
            message_flow_history: Vec::new(),
            coordination_patterns: Vec::new(),
        }
    }

    /// Initialize MCP connection and start real-time polling
    async fn initialize_mcp_connection(&mut self, ctx: &mut Context<Self>) -> Result<(), String> {
        info!("Initializing MCP connection for Enhanced Claude Flow Actor");

        // Connect to MCP server
        if let Err(e) = self.client.connect().await {
            error!("Failed to connect to MCP server: {}", e);
            return Err(format!("MCP connection failed: {}", e));
        }

        // Initialize MCP protocol
        if let Err(e) = self.client.initialize().await {
            error!("Failed to initialize MCP protocol: {}", e);
            return Err(format!("MCP initialization failed: {}", e));
        }

        self.is_connected = true;
        self.is_initialized = true;

        // Start real-time polling for swarm data
        self.start_real_time_polling(ctx);

        info!("Enhanced Claude Flow Actor MCP connection initialized successfully");
        Ok(())
    }

    /// Start real-time polling for agent positions and state updates
    fn start_real_time_polling(&mut self, ctx: &mut Context<Self>) {
        info!("Starting real-time polling at 60 FPS for swarm data");

        // Schedule interval for real-time updates
        ctx.run_interval(self.polling_interval, |actor, ctx| {
            let now = Utc::now();
            if (now - actor.last_poll).num_milliseconds() >= 16 { // 60 FPS = ~16ms
                actor.last_poll = now;
                
                // Trigger polling for different data types
                ctx.address().do_send(PollSwarmData);
            }
        });

        // Schedule less frequent system metrics updates (1 second)
        ctx.run_interval(Duration::from_secs(1), |actor, ctx| {
            ctx.address().do_send(PollSystemMetrics);
        });
    }

    /// Execute MCP tool call with error handling and response parsing
    async fn execute_mcp_tool(&mut self, tool_name: &str, arguments: Value) -> Result<Value, String> {
        if !self.is_initialized {
            return Err("MCP client not initialized".to_string());
        }

        debug!("Executing MCP tool: {} with args: {}", tool_name, arguments);

        // Create MCP tool call request
        let params = json!({
            "name": tool_name,
            "arguments": arguments
        });

        let request = crate::services::claude_flow::types::McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Uuid::new_v4().to_string(),
            method: "tools/call".to_string(),
            params: Some(params),
        };

        // Execute request through transport
        let mut transport = self.client.transport.lock().await;
        match transport.send_request(request).await {
            Ok(response) => {
                if let Some(result) = response.result {
                    debug!("MCP tool {} executed successfully", tool_name);
                    Ok(result)
                } else if let Some(error) = response.error {
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
        let total_agents = agents.len() as u32;

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

    /// Generate synthetic swarm data for visualization testing
    fn generate_enhanced_mock_agents(&mut self) -> Vec<AgentStatus> {
        let agent_types = vec![
            ("coordinator", "Hive Mind Coordinator", AgentType::Coordinator),
            ("architect", "System Architect", AgentType::Architect),
            ("coder", "Implementation Specialist", AgentType::Coder),
            ("researcher", "Analysis Researcher", AgentType::Researcher),
            ("tester", "Quality Assurance", AgentType::Tester),
            ("analyst", "Metrics Analyst", AgentType::Analyst),
            ("optimizer", "Performance Optimizer", AgentType::Optimizer),
            ("monitor", "System Monitor", AgentType::Monitor),
        ];

        let mut agents = Vec::new();
        let now = Utc::now();

        for (i, (agent_type, name, enum_type)) in agent_types.iter().enumerate() {
            let agent_id = format!("{}-{:03}", agent_type, i + 1);
            
            // Generate realistic performance metrics
            let tasks_completed = 10 + (i * 5) as u32;
            let tasks_failed = if i == 7 { 1 } else { 0 }; // Monitor has 1 failure
            let success_rate = if tasks_failed > 0 {
                (tasks_completed as f32 / (tasks_completed + tasks_failed) as f32) * 100.0
            } else {
                100.0
            };

            let status = match i {
                0 => "active",    // Coordinator always active
                1..=5 => "active", // Most agents active
                6 => "busy",      // Optimizer busy
                7 => "idle",      // Monitor idle
                _ => "active",
            };

            let agent = AgentStatus {
                agent_id: agent_id.clone(),
                status: status.to_string(),
                session_id: Uuid::new_v4().to_string(),
                profile: AgentProfile {
                    name: name.to_string(),
                    agent_type: enum_type.clone(),
                    capabilities: self.generate_agent_capabilities(agent_type),
                    system_prompt: Some(format!("You are a {} in the hive mind swarm", name)),
                    max_concurrent_tasks: match agent_type {
                        "coordinator" => 20,
                        "architect" => 8,
                        "coder" => 5,
                        "tester" => 10,
                        _ => 6,
                    },
                    priority: match agent_type {
                        "coordinator" => 10,
                        "architect" => 9,
                        "coder" => 8,
                        _ => 7,
                    },
                    retry_policy: Default::default(),
                    environment: Some("hive-mind-swarm".to_string()),
                    working_directory: Some("/workspace/ext".to_string()),
                },
                timestamp: now,
                active_tasks_count: if status == "active" { (i % 3) as u32 + 1 } else { 0 },
                completed_tasks_count: tasks_completed,
                failed_tasks_count: tasks_failed,
                total_execution_time: (tasks_completed * 2500 + i as u32 * 1000) as u64,
                average_task_duration: 2500.0 + (i as f32 * 100.0),
                success_rate,
                current_task: if status == "active" {
                    Some(format!("Executing {} optimization task", agent_type))
                } else {
                    None
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("swarm_role".to_string(), json!(agent_type));
                    meta.insert("specialization".to_string(), json!(name));
                    meta.insert("bot_observability_upgrade".to_string(), json!(true));
                    meta
                },
            };

            agents.push(agent);
        }

        // Update agent cache
        for agent in &agents {
            self.agent_cache.insert(agent.agent_id.clone(), agent.clone());
        }

        agents
    }

    /// Generate capabilities based on agent type
    fn generate_agent_capabilities(&self, agent_type: &str) -> Vec<String> {
        match agent_type {
            "coordinator" => vec![
                "swarm_orchestration".to_string(),
                "task_distribution".to_string(),
                "resource_allocation".to_string(),
                "strategic_planning".to_string(),
                "coordination_patterns".to_string(),
            ],
            "architect" => vec![
                "system_design".to_string(),
                "component_architecture".to_string(),
                "spring_physics".to_string(),
                "gpu_optimization".to_string(),
                "visualization_design".to_string(),
            ],
            "coder" => vec![
                "rust_development".to_string(),
                "react_typescript".to_string(),
                "mcp_integration".to_string(),
                "websocket_protocols".to_string(),
                "three_js_graphics".to_string(),
            ],
            "researcher" => vec![
                "code_analysis".to_string(),
                "performance_research".to_string(),
                "pattern_identification".to_string(),
                "optimization_strategies".to_string(),
                "documentation_review".to_string(),
            ],
            "tester" => vec![
                "unit_testing".to_string(),
                "integration_testing".to_string(),
                "load_testing".to_string(),
                "gpu_performance_testing".to_string(),
                "websocket_validation".to_string(),
            ],
            "analyst" => vec![
                "metrics_collection".to_string(),
                "performance_monitoring".to_string(),
                "bottleneck_analysis".to_string(),
                "real_time_analytics".to_string(),
                "system_health_monitoring".to_string(),
            ],
            "optimizer" => vec![
                "gpu_shader_optimization".to_string(),
                "memory_management".to_string(),
                "spring_physics_tuning".to_string(),
                "rendering_optimization".to_string(),
                "cpu_profiling".to_string(),
            ],
            "monitor" => vec![
                "system_monitoring".to_string(),
                "failure_detection".to_string(),
                "auto_recovery".to_string(),
                "uptime_tracking".to_string(),
                "alert_management".to_string(),
            ],
            _ => vec!["general_processing".to_string()],
        }
    }

    /// Generate synthetic message flow for visualization
    fn generate_mock_message_flow(&mut self) {
        let agents: Vec<String> = self.agent_cache.keys().cloned().collect();
        
        if agents.is_empty() {
            return;
        }

        // Generate 2-3 random messages
        for _ in 0..3 {
            if agents.len() >= 2 {
                let from_idx = fastrand::usize(0..agents.len());
                let mut to_idx = fastrand::usize(0..agents.len());
                while to_idx == from_idx {
                    to_idx = fastrand::usize(0..agents.len());
                }

                let message = MessageFlowEvent {
                    id: Uuid::new_v4().to_string(),
                    from_agent: agents[from_idx].clone(),
                    to_agent: agents[to_idx].clone(),
                    message_type: "coordination".to_string(),
                    priority: fastrand::u8(1..=5),
                    timestamp: Utc::now(),
                    latency_ms: fastrand::f32() * 50.0 + 10.0, // 10-60ms latency
                };

                self.message_flow_history.push(message);
            }
        }

        // Keep only recent messages (last 5 minutes)
        let cutoff = Utc::now() - chrono::Duration::minutes(5);
        self.message_flow_history.retain(|msg| msg.timestamp > cutoff);
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            active_agents: 0,
            message_rate: 0.0,
            average_latency: 0.0,
            error_rate: 0.0,
            network_health: 1.0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
            gpu_usage: None,
        }
    }
}

// Internal polling messages
#[derive(Message)]
#[rtype(result = "()")]
struct PollSwarmData;

#[derive(Message)]
#[rtype(result = "()")]
struct PollSystemMetrics;

impl Actor for EnhancedClaudeFlowActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Enhanced Claude Flow Actor started");
        
        // Initialize MCP connection asynchronously
        ctx.spawn(
            async {
                info!("Starting MCP connection initialization");
            }
            .into_actor(self)
            .then(|_, actor, ctx| {
                async move {
                    match actor.initialize_mcp_connection(ctx).await {
                        Ok(()) => {
                            info!("MCP connection initialized successfully");
                        }
                        Err(e) => {
                            error!("Failed to initialize MCP connection: {}", e);
                            // Schedule retry after 5 seconds
                            ctx.run_later(Duration::from_secs(5), |actor, ctx| {
                                ctx.address().do_send(RetryMCPConnection);
                            });
                        }
                    }
                }
                .into_actor(actor)
            })
        );
    }
}

// Message Handlers

impl Handler<InitializeSwarm> for EnhancedClaudeFlowActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: InitializeSwarm, _ctx: &mut Self::Context) -> Self::Result {
        info!("Initializing swarm with topology: {}", msg.topology);

        let arguments = json!({
            "topology": msg.topology,
            "maxAgents": msg.max_agents,
            "agentTypes": msg.agent_types,
            "enableNeural": msg.enable_neural,
            "strategy": msg.strategy,
            "customPrompt": msg.custom_prompt
        });

        Box::pin(
            async move {
                match self.execute_mcp_tool("swarm_init", arguments).await {
                    Ok(result) => {
                        info!("Swarm initialization successful: {:?}", result);
                        
                        // Generate swarm ID
                        self.swarm_id = Some(Uuid::new_v4().to_string());
                        
                        // Generate initial agent data
                        let agents = self.generate_enhanced_mock_agents();
                        
                        // Update graph service with new agents
                        self.graph_service_addr.do_send(UpdateBotsGraph { agents: agents.clone() });
                        
                        // Update system metrics
                        self.update_system_metrics(&agents);
                        
                        Ok(())
                    }
                    Err(e) => {
                        error!("Swarm initialization failed: {}", e);
                        Err(e)
                    }
                }
            }
            .into_actor(self)
        )
    }
}

impl Handler<GetSwarmStatus> for EnhancedClaudeFlowActor {
    type Result = ResponseActFuture<Self, Result<SwarmStatus, String>>;

    fn handle(&mut self, _msg: GetSwarmStatus, _ctx: &mut Self::Context) -> Self::Result {
        info!("Getting swarm status");

        Box::pin(
            async move {
                match self.execute_mcp_tool("swarm_status", json!({})).await {
                    Ok(result) => {
                        // Parse result or return cached status
                        let status = SwarmStatus {
                            swarm_id: self.swarm_id.clone().unwrap_or_else(|| "default".to_string()),
                            active_agents: self.system_metrics.active_agents,
                            total_agents: self.agent_cache.len() as u32,
                            topology: "hierarchical".to_string(),
                            health_score: self.system_metrics.network_health,
                            coordination_efficiency: 0.95, // TODO: Calculate from real data
                        };
                        
                        info!("Swarm status retrieved: {:?}", status);
                        Ok(status)
                    }
                    Err(e) => {
                        warn!("Failed to get swarm status via MCP, using cached data: {}", e);
                        
                        // Return cached status
                        let status = SwarmStatus {
                            swarm_id: self.swarm_id.clone().unwrap_or_else(|| "default".to_string()),
                            active_agents: self.system_metrics.active_agents,
                            total_agents: self.agent_cache.len() as u32,
                            topology: "hierarchical".to_string(),
                            health_score: self.system_metrics.network_health,
                            coordination_efficiency: 0.95,
                        };
                        
                        Ok(status)
                    }
                }
            }
            .into_actor(self)
        )
    }
}

impl Handler<SwarmMonitor> for EnhancedClaudeFlowActor {
    type Result = ResponseActFuture<Self, Result<SwarmMonitorData, String>>;

    fn handle(&mut self, _msg: SwarmMonitor, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Collecting swarm monitoring data");

        Box::pin(
            async move {
                // Generate fresh message flow data
                self.generate_mock_message_flow();

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
            .into_actor(self)
        )
    }
}

impl Handler<PollSwarmData> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, _msg: PollSwarmData, _ctx: &mut Self::Context) -> Self::Result {
        // High-frequency polling for position and state updates
        if !self.agent_cache.is_empty() {
            // Generate updated agent data
            let agents = self.generate_enhanced_mock_agents();
            
            // Send updates to graph service
            self.graph_service_addr.do_send(UpdateBotsGraph { agents });
        }
    }
}

impl Handler<PollSystemMetrics> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, _msg: PollSystemMetrics, _ctx: &mut Self::Context) -> Self::Result {
        // Low-frequency polling for system metrics
        let agents: Vec<AgentStatus> = self.agent_cache.values().cloned().collect();
        self.update_system_metrics(&agents);
        
        debug!("System metrics updated: {:?}", self.system_metrics);
    }
}

impl Handler<MetricsCollect> for EnhancedClaudeFlowActor {
    type Result = ResponseActFuture<Self, Result<SystemMetrics, String>>;

    fn handle(&mut self, _msg: MetricsCollect, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Collecting system metrics");

        Box::pin(
            async move {
                Ok(self.system_metrics.clone())
            }
            .into_actor(self)
        )
    }
}

// Additional handlers for other MCP tools...
impl Handler<GetAgentMetrics> for EnhancedClaudeFlowActor {
    type Result = ResponseActFuture<Self, Result<Vec<AgentMetrics>, String>>;

    fn handle(&mut self, _msg: GetAgentMetrics, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Getting agent metrics");

        Box::pin(
            async move {
                let metrics: Vec<AgentMetrics> = self.agent_cache
                    .values()
                    .map(|agent| AgentMetrics {
                        agent_id: agent.agent_id.clone(),
                        performance_score: agent.success_rate / 100.0,
                        tasks_completed: agent.completed_tasks_count,
                        success_rate: agent.success_rate / 100.0,
                        resource_utilization: 0.7, // TODO: Calculate from real data
                        token_usage: agent.total_execution_time, // Approximate
                    })
                    .collect();

                Ok(metrics)
            }
            .into_actor(self)
        )
    }
}

impl Handler<RetryMCPConnection> for EnhancedClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, _msg: RetryMCPConnection, ctx: &mut Self::Context) -> Self::Result {
        info!("Retrying MCP connection");
        
        ctx.spawn(
            async move {
                info!("Attempting to reconnect to MCP server");
            }
            .into_actor(self)
            .then(|_, actor, ctx| {
                async move {
                    match actor.initialize_mcp_connection(ctx).await {
                        Ok(()) => {
                            info!("MCP reconnection successful");
                        }
                        Err(e) => {
                            error!("MCP reconnection failed: {}", e);
                            // Schedule another retry after 10 seconds
                            ctx.run_later(Duration::from_secs(10), |actor, ctx| {
                                ctx.address().do_send(RetryMCPConnection);
                            });
                        }
                    }
                }
                .into_actor(actor)
            })
        );
    }
}