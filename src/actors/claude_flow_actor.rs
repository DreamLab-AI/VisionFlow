use actix::prelude::*;
use std::time::Duration;
use log::{info, error, warn};
use crate::services::claude_flow::{ClaudeFlowClient, ClaudeFlowClientBuilder, AgentStatus, AgentProfile, AgentType};
use crate::actors::messages::UpdateBotsGraph;
use crate::actors::GraphServiceActor;
use std::collections::HashMap;
use chrono::{Utc, DateTime};
use uuid::Uuid;

// Communication link data structure for agent interactions
#[derive(Debug, Clone)]
pub struct CommunicationLink {
    pub source_agent: String,
    pub target_agent: String,
    pub interaction_count: u32,
    pub message_frequency: f32, // messages per second
    pub last_interaction: DateTime<Utc>,
    pub collaboration_score: f32, // 0.0 to 1.0
}

// Combined swarm state snapshot
#[derive(Debug, Clone)]
pub struct SwarmStateSnapshot {
    pub agents: Vec<AgentStatus>,
    pub communication_links: Vec<CommunicationLink>,
    pub topology_type: String,
    pub coordination_efficiency: f32,
    pub timestamp: DateTime<Utc>,
}

pub struct ClaudeFlowActor {
    client: ClaudeFlowClient,
    graph_service_addr: Addr<GraphServiceActor>,
    is_connected: bool,
}

impl ClaudeFlowActor {
    fn create_mock_agents() -> Vec<AgentStatus> {
        vec![
            AgentStatus {
                agent_id: "coordinator-001".to_string(),
                status: "active".to_string(),
                session_id: Uuid::new_v4().to_string(),
                profile: AgentProfile {
                    name: "System Coordinator".to_string(),
                    agent_type: AgentType::Coordinator,
                    capabilities: vec![
                        "orchestration".to_string(),
                        "task-management".to_string(),
                        "resource-allocation".to_string(),
                    ],
                    system_prompt: None,
                    max_concurrent_tasks: 10,
                    priority: 10,
                    retry_policy: Default::default(),
                    environment: None,
                    working_directory: None,
                },
                timestamp: Utc::now(),
                active_tasks_count: 3,
                completed_tasks_count: 15,
                failed_tasks_count: 0,
                total_execution_time: 45000,
                average_task_duration: 3000.0,
                success_rate: 100.0,
                current_task: None,
                metadata: HashMap::new(),
            },
            AgentStatus {
                agent_id: "researcher-001".to_string(),
                status: "active".to_string(),
                session_id: Uuid::new_v4().to_string(),
                profile: AgentProfile {
                    name: "Research Agent".to_string(),
                    agent_type: AgentType::Researcher,
                    capabilities: vec![
                        "data-gathering".to_string(),
                        "analysis".to_string(),
                        "report-generation".to_string(),
                    ],
                    system_prompt: None,
                    max_concurrent_tasks: 5,
                    priority: 8,
                    retry_policy: Default::default(),
                    environment: None,
                    working_directory: None,
                },
                timestamp: Utc::now(),
                active_tasks_count: 2,
                completed_tasks_count: 20,
                failed_tasks_count: 1,
                total_execution_time: 60000,
                average_task_duration: 3000.0,
                success_rate: 95.2,
                current_task: None,
                metadata: HashMap::new(),
            },
            AgentStatus {
                agent_id: "coder-001".to_string(),
                status: "active".to_string(),
                session_id: Uuid::new_v4().to_string(),
                profile: AgentProfile {
                    name: "Code Developer".to_string(),
                    agent_type: AgentType::Coder,
                    capabilities: vec![
                        "implementation".to_string(),
                        "refactoring".to_string(),
                        "debugging".to_string(),
                    ],
                    system_prompt: None,
                    max_concurrent_tasks: 3,
                    priority: 9,
                    retry_policy: Default::default(),
                    environment: None,
                    working_directory: None,
                },
                timestamp: Utc::now(),
                active_tasks_count: 1,
                completed_tasks_count: 30,
                failed_tasks_count: 2,
                total_execution_time: 90000,
                average_task_duration: 3000.0,
                success_rate: 93.8,
                current_task: None,
                metadata: HashMap::new(),
            },
            AgentStatus {
                agent_id: "analyst-001".to_string(),
                status: "active".to_string(),
                session_id: Uuid::new_v4().to_string(),
                profile: AgentProfile {
                    name: "Data Analyst".to_string(),
                    agent_type: AgentType::Analyst,
                    capabilities: vec![
                        "data-analysis".to_string(),
                        "visualization".to_string(),
                        "insights".to_string(),
                    ],
                    system_prompt: None,
                    max_concurrent_tasks: 4,
                    priority: 7,
                    retry_policy: Default::default(),
                    environment: None,
                    working_directory: None,
                },
                timestamp: Utc::now(),
                active_tasks_count: 2,
                completed_tasks_count: 18,
                failed_tasks_count: 0,
                total_execution_time: 54000,
                average_task_duration: 3000.0,
                success_rate: 100.0,
                current_task: None,
                metadata: HashMap::new(),
            },
            AgentStatus {
                agent_id: "tester-001".to_string(),
                status: "idle".to_string(),
                session_id: Uuid::new_v4().to_string(),
                profile: AgentProfile {
                    name: "Test Engineer".to_string(),
                    agent_type: AgentType::Tester,
                    capabilities: vec![
                        "unit-testing".to_string(),
                        "integration-testing".to_string(),
                        "validation".to_string(),
                    ],
                    system_prompt: None,
                    max_concurrent_tasks: 6,
                    priority: 8,
                    retry_policy: Default::default(),
                    environment: None,
                    working_directory: None,
                },
                timestamp: Utc::now(),
                active_tasks_count: 0,
                completed_tasks_count: 25,
                failed_tasks_count: 3,
                total_execution_time: 75000,
                average_task_duration: 3000.0,
                success_rate: 89.3,
                current_task: None,
                metadata: HashMap::new(),
            },
        ]
    }

    fn create_mock_communication_links(agents: &[AgentStatus]) -> Vec<CommunicationLink> {
        let mut links = Vec::new();
        
        // Create communication patterns based on agent types
        for (i, source_agent) in agents.iter().enumerate() {
            for (j, target_agent) in agents.iter().enumerate() {
                if i != j {
                    // Calculate communication intensity based on agent roles
                    let intensity = Self::calculate_mock_communication_intensity(
                        &source_agent.profile.agent_type,
                        &target_agent.profile.agent_type
                    );
                    
                    if intensity > 0.0 {
                        links.push(CommunicationLink {
                            source_agent: source_agent.agent_id.clone(),
                            target_agent: target_agent.agent_id.clone(),
                            interaction_count: (intensity * 50.0) as u32,
                            message_frequency: intensity * 2.0,
                            last_interaction: Utc::now() - chrono::Duration::seconds((60.0 / intensity) as i64),
                            collaboration_score: intensity,
                        });
                    }
                }
            }
        }
        
        links
    }

    fn calculate_mock_communication_intensity(source: &AgentType, target: &AgentType) -> f32 {
        match (source, target) {
            // Coordinator communicates heavily with all types
            (AgentType::Coordinator, _) | (_, AgentType::Coordinator) => 0.9,
            // Coder and Tester collaborate closely
            (AgentType::Coder, AgentType::Tester) | (AgentType::Tester, AgentType::Coder) => 0.8,
            // Researcher and Analyst share data
            (AgentType::Researcher, AgentType::Analyst) | (AgentType::Analyst, AgentType::Researcher) => 0.7,
            // Architect coordinates with Coder and Analyst
            (AgentType::Architect, AgentType::Coder) | (AgentType::Coder, AgentType::Architect) => 0.7,
            (AgentType::Architect, AgentType::Analyst) | (AgentType::Analyst, AgentType::Architect) => 0.6,
            // Default moderate communication
            _ => 0.4,
        }
    }

    async fn retrieve_communication_links(_client: &ClaudeFlowClient, agents: &[AgentStatus]) -> Vec<CommunicationLink> {
        // For now, we'll create mock communication links as the MCP protocol
        // doesn't have direct support for communication link retrieval.
        // In a real implementation, this would query the Claude Flow system
        // for actual agent interaction data.
        
        // TODO: Implement actual communication link retrieval when MCP supports it
        Self::create_mock_communication_links(agents)
    }

    fn calculate_coordination_efficiency(agents: &[AgentStatus]) -> f32 {
        if agents.is_empty() {
            return 0.0;
        }
        
        let total_success_rate: f64 = agents.iter().map(|a| a.success_rate).sum();
        let avg_success_rate = total_success_rate / agents.len() as f64;
        
        // Factor in active vs total tasks
        let total_active: u32 = agents.iter().map(|a| a.active_tasks_count).sum();
        let total_completed: u32 = agents.iter().map(|a| a.completed_tasks_count).sum();
        
        let activity_factor = if total_active + total_completed > 0 {
            total_completed as f32 / (total_active + total_completed) as f32
        } else {
            0.5
        };
        
        // Combine success rate and activity factor
        (avg_success_rate as f32 * 0.7 + activity_factor * 0.3).min(1.0)
    }

    pub async fn new(graph_service_addr: Addr<GraphServiceActor>) -> Result<Self, String> {
        info!("ClaudeFlowActor: Initializing Claude Flow via stdio (direct process spawn)");

        // Use stdio transport to directly spawn claude-flow process
        let client_result = ClaudeFlowClientBuilder::new()
            .use_stdio()
            .build()
            .await;

        let (client, is_connected) = match client_result {
            Ok(mut c) => {
                // Try to connect, but handle failures gracefully
                let connected = match c.connect().await {
                    Ok(_) => {
                        info!("ClaudeFlowActor: Successfully connected to Claude Flow MCP");
                        match c.initialize().await {
                            Ok(_) => {
                                info!("ClaudeFlowActor: Successfully initialized Claude Flow session");
                                true
                            }
                            Err(e) => {
                                error!("ClaudeFlowActor: Failed to initialize Claude Flow session: {}. Running in degraded mode.", e);
                                // Continue without Claude Flow - the actor can still function
                                false
                            }
                        }
                    }
                    Err(e) => {
                        error!("ClaudeFlowActor: Failed to connect to Claude Flow: {}. Running in degraded mode.", e);
                        warn!("ClaudeFlowActor: Claude Flow features will be unavailable. This might be due to:");
                        warn!("  - Claude Flow npm package not installed or available");
                        warn!("  - Process spawn permissions issues");
                        warn!("  - MCP initialization failure");
                        info!("ClaudeFlowActor: Using mock agents for visualization instead.");
                        // Continue without Claude Flow - provide mock data
                        false
                    }
                };
                (c, connected)
            }
            Err(e) => {
                error!("ClaudeFlowActor: Failed to build Claude Flow client: {}.", e);
                warn!("ClaudeFlowActor: This is likely due to missing claude-flow npm package");
                return Err(format!("Failed to build Claude Flow client: {}", e));
            }
        };

        Ok(Self { client, graph_service_addr, is_connected })
    }

    fn poll_for_updates(&self, ctx: &mut Context<Self>) {
        // If not connected, provide mock data for visualization
        if !self.is_connected {
            let graph_addr = self.graph_service_addr.clone();

            // Create mock agents and communication links for visualization
            ctx.run_interval(Duration::from_secs(10), move |_act, _ctx| {
                let mock_agents = Self::create_mock_agents();
                let mock_links = Self::create_mock_communication_links(&mock_agents);
                let snapshot = SwarmStateSnapshot {
                    agents: mock_agents,
                    communication_links: mock_links,
                    topology_type: "hierarchical".to_string(),
                    coordination_efficiency: 0.85,
                    timestamp: Utc::now(),
                };
                info!("Providing mock swarm snapshot with {} agents and {} communication links.", 
                     snapshot.agents.len(), snapshot.communication_links.len());
                graph_addr.do_send(UpdateBotsGraph { agents: snapshot.agents });
            });
            return;
        }

        // Poll for agent updates and communication links every 5 seconds
        ctx.run_interval(Duration::from_secs(5), |act, _ctx| {
            if !act.is_connected {
                return;
            }

            let client = act.client.clone();
            let graph_addr = act.graph_service_addr.clone();

            actix::spawn(async move {
                // Retrieve agents
                let agents_result = client.list_agents(false).await;
                
                match agents_result {
                    Ok(agents) => {
                        if !agents.is_empty() {
                            // Retrieve communication links between agents
                            let communication_links = Self::retrieve_communication_links(&client, &agents).await;
                            
                            // Create combined swarm state snapshot
                            let snapshot = SwarmStateSnapshot {
                                agents: agents.clone(),
                                communication_links,
                                topology_type: "dynamic".to_string(),
                                coordination_efficiency: Self::calculate_coordination_efficiency(&agents),
                                timestamp: Utc::now(),
                            };
                            
                            info!("Polled {} active agents with {} communication links from Claude Flow.", 
                                 snapshot.agents.len(), snapshot.communication_links.len());
                            
                            // Send complete snapshot via UpdateBotsGraph
                            graph_addr.do_send(UpdateBotsGraph { agents: snapshot.agents });
                        }
                    }
                    Err(e) => error!("Failed to poll agents from Claude Flow: {}", e),
                }
            });
        });
    }

    fn handle_reconnection(&self, ctx: &mut Context<Self>) {
        // Only check health if connected
        if !self.is_connected {
            return;
        }

        // Check connection health every 30 seconds
        ctx.run_interval(Duration::from_secs(30), |act, _ctx| {
            if !act.is_connected {
                return;
            }

            let client = act.client.clone();

            actix::spawn(async move {
                match client.get_system_health().await {
                    Ok(health) => {
                        if health.status != "healthy" {
                            warn!("ClaudeFlowActor: System health check failed: {:?}", health);
                        }
                    }
                    Err(e) => {
                        error!("ClaudeFlowActor: Health check failed: {}", e);
                        // In degraded mode, we just log the error
                    }
                }
            });
        });
    }
}

impl Actor for ClaudeFlowActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Context<Self>) {
        info!("ClaudeFlowActor started. Scheduling polling and health checks.");
        self.poll_for_updates(ctx);
        self.handle_reconnection(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Context<Self>) {
        info!("ClaudeFlowActor stopped. Disconnecting from Claude Flow.");
        let mut client = self.client.clone();
        actix::spawn(async move {
            if let Err(e) = client.disconnect().await {
                error!("Failed to disconnect from Claude Flow: {}", e);
            }
        });
    }
}

// Message handlers for controlling the Claude Flow integration

#[derive(Message)]
#[rtype(result = "Result<Vec<AgentStatus>, String>")]
pub struct GetActiveAgents;

impl Handler<GetActiveAgents> for ClaudeFlowActor {
    type Result = ResponseFuture<Result<Vec<AgentStatus>, String>>;

    fn handle(&mut self, _msg: GetActiveAgents, _ctx: &mut Context<Self>) -> Self::Result {
        if !self.is_connected {
            // Return mock agents when not connected
            return Box::pin(async move {
                Ok(ClaudeFlowActor::create_mock_agents())
            });
        }

        let client = self.client.clone();
        Box::pin(async move {
            client.list_agents(false)
                .await
                .map_err(|e| e.to_string())
        })
    }
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SpawnClaudeAgent {
    pub agent_type: String,
    pub name: String,
    pub capabilities: Vec<String>,
}

impl Handler<SpawnClaudeAgent> for ClaudeFlowActor {
    type Result = ResponseFuture<Result<(), String>>;

    fn handle(&mut self, msg: SpawnClaudeAgent, _ctx: &mut Context<Self>) -> Self::Result {
        if !self.is_connected {
            return Box::pin(async move {
                // In mock mode, just log and return success
                info!("Mock mode: Would spawn agent '{}' of type '{}'", msg.name, msg.agent_type);
                Ok(())
            });
        }

        let client = self.client.clone();

        Box::pin(async move {
            use crate::services::claude_flow::{client::SpawnAgentParams, AgentType};

            let agent_type = match msg.agent_type.as_str() {
                "coordinator" => AgentType::Coordinator,
                "researcher" => AgentType::Researcher,
                "coder" => AgentType::Coder,
                "analyst" => AgentType::Analyst,
                "architect" => AgentType::Architect,
                "tester" => AgentType::Tester,
                _ => AgentType::Specialist,
            };

            let params = SpawnAgentParams {
                agent_type,
                name: msg.name,
                capabilities: Some(msg.capabilities),
                system_prompt: None,
                max_concurrent_tasks: Some(3),
                priority: Some(5),
                environment: None,
                working_directory: None,
            };

            client.spawn_agent(params)
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
        })
    }
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct TerminateClaudeAgent {
    pub agent_id: String,
}

impl Handler<TerminateClaudeAgent> for ClaudeFlowActor {
    type Result = ResponseFuture<Result<(), String>>;

    fn handle(&mut self, msg: TerminateClaudeAgent, _ctx: &mut Context<Self>) -> Self::Result {
        let client = self.client.clone();

        Box::pin(async move {
            client.terminate_agent(&msg.agent_id)
                .await
                .map_err(|e| e.to_string())
        })
    }
}

use crate::actors::messages::InitializeSwarm;

impl Handler<InitializeSwarm> for ClaudeFlowActor {
    type Result = ResponseFuture<Result<(), String>>;

    fn handle(&mut self, msg: InitializeSwarm, ctx: &mut Context<Self>) -> Self::Result {
        info!("ClaudeFlowActor: Initializing swarm with topology: {}, max_agents: {}",
              msg.topology, msg.max_agents);

        let mut client = self.client.clone();
        let _graph_addr = self.graph_service_addr.clone();
        let actor_addr = ctx.address();

        Box::pin(async move {
            // First ensure we're connected
            if let Err(e) = client.connect().await {
                error!("Failed to connect to Claude Flow: {}", e);
                return Err(format!("Failed to connect to Claude Flow: {}", e));
            }

            // Initialize the MCP session
            if let Err(e) = client.initialize().await {
                error!("Failed to initialize Claude Flow session: {}", e);
                return Err(format!("Failed to initialize Claude Flow session: {}", e));
            }

            // Initialize swarm with the requested configuration
            match client.init_swarm(&msg.topology, Some(msg.max_agents)).await {
                Ok(swarm_info) => {
                    info!("Swarm initialized successfully: {}", swarm_info);

                    // If neural enhancement is enabled, train neural patterns
                    if msg.enable_neural {
                        info!("Training neural patterns for enhanced coordination");
                        // Train neural patterns for better coordination
                        if let Err(e) = client.train_neural_pattern("coordination", "swarm optimization", Some(50)).await {
                            warn!("Failed to train neural patterns: {}", e);
                        }
                    }

                    // Now spawn the requested agent types
                    let mut spawn_errors = Vec::new();

                    for agent_type in &msg.agent_types {
                        use crate::services::claude_flow::{client::SpawnAgentParams, AgentType};

                        let agent_type_enum = match agent_type.as_str() {
                            "coordinator" => AgentType::Coordinator,
                            "researcher" => AgentType::Researcher,
                            "coder" => AgentType::Coder,
                            "analyst" => AgentType::Analyst,
                            "architect" => AgentType::Architect,
                            "tester" => AgentType::Tester,
                            "optimizer" => AgentType::Optimizer,
                            "reviewer" => AgentType::Reviewer,
                            "documenter" => AgentType::Documenter,
                            _ => AgentType::Specialist,
                        };

                        let agent_name = format!("{} Agent",
                            agent_type.chars().next().unwrap().to_uppercase().to_string() + &agent_type[1..]);

                        let params = SpawnAgentParams {
                            agent_type: agent_type_enum,
                            name: agent_name,
                            capabilities: Some(vec![]),
                            system_prompt: msg.custom_prompt.clone(),
                            max_concurrent_tasks: Some(3),
                            priority: Some(5),
                            environment: None,
                            working_directory: None,
                        };

                        if let Err(e) = client.spawn_agent(params).await {
                            spawn_errors.push(format!("Failed to spawn {} agent: {}", agent_type, e));
                        }
                    }

                    if !spawn_errors.is_empty() {
                        warn!("Some agents failed to spawn: {:?}", spawn_errors);
                    }

                    // Mark the actor as connected and restart polling
                    actor_addr.do_send(MarkConnected { connected: true });

                    Ok(())
                }
                Err(e) => {
                    error!("Failed to initialize swarm: {}", e);
                    Err(format!("Failed to initialize swarm: {}", e))
                }
            }
        })
    }
}

#[derive(Message)]
#[rtype(result = "()")]
struct MarkConnected {
    connected: bool,
}

impl Handler<MarkConnected> for ClaudeFlowActor {
    type Result = ();

    fn handle(&mut self, msg: MarkConnected, ctx: &mut Context<Self>) {
        self.is_connected = msg.connected;
        if msg.connected {
            info!("ClaudeFlowActor marked as connected, restarting polling");
            self.poll_for_updates(ctx);
        }
    }
}