//! Agent Monitor Actor - Simplified Monitoring via Direct MCP TCP
//!
//! This actor focuses solely on:
//! - Polling the MCP TCP server (port 9500) for agent statuses
//! - Caching agent data
//! - Forwarding updates to GraphServiceSupervisor
//!
//! All task management is handled by the Management API (port 9090) via TaskOrchestratorActor.
//! This actor only monitors agent statuses.

use actix::prelude::*;
use std::time::Duration;
use log::{info, error, debug, warn};
use std::collections::HashMap;
use chrono::{Utc, DateTime};

use crate::types::claude_flow::{ClaudeFlowClient, AgentStatus, AgentProfile, AgentType, PerformanceMetrics, TokenUsage};
use crate::actors::messages::*;
use crate::utils::mcp_tcp_client::create_mcp_client;
use crate::services::agent_visualization_protocol::McpServerType;

/// AgentMonitorActor - Pure monitoring via direct MCP TCP
pub struct AgentMonitorActor {
    _client: ClaudeFlowClient,
    graph_service_addr: Addr<crate::actors::graph_service_supervisor::TransitionalGraphSupervisor>,

    /// Connection state
    is_connected: bool,

    /// Polling configuration
    polling_interval: Duration,
    last_poll: DateTime<Utc>,

    /// Agent cache
    agent_cache: HashMap<String, AgentStatus>,

    /// Error tracking
    consecutive_poll_failures: u32,
    last_successful_poll: Option<DateTime<Utc>>,
}

impl AgentMonitorActor {
    pub fn new(client: ClaudeFlowClient, graph_service_addr: Addr<crate::actors::graph_service_supervisor::TransitionalGraphSupervisor>) -> Self {
        info!("[AgentMonitorActor] Initializing with direct MCP TCP monitoring");
        Self {
            _client: client,
            graph_service_addr,
            is_connected: false,
            polling_interval: Duration::from_secs(2),
            last_poll: Utc::now(),
            agent_cache: HashMap::new(),
            consecutive_poll_failures: 0,
            last_successful_poll: None,
        }
    }

    /// Poll agent statuses directly from MCP TCP server
    fn poll_agent_statuses(&mut self, ctx: &mut Context<Self>) {
        // Circuit breaker logic
        if self.consecutive_poll_failures > 10 {
            if let Some(last_success) = self.last_successful_poll {
                let time_since_success = Utc::now().signed_duration_since(last_success);
                if time_since_success.num_seconds() < 30 {
                    debug!("[AgentMonitorActor] Circuit breaker active - skipping poll");
                    return;
                }
            }
        }

        debug!("[AgentMonitorActor] Polling agent statuses via MCP TCP");

        let host = std::env::var("MCP_HOST").unwrap_or_else(|_| "agentic-workstation".to_string());
        let port = std::env::var("MCP_TCP_PORT")
            .unwrap_or_else(|_| "9500".to_string())
            .parse::<u16>()
            .unwrap_or(9500);

        let ctx_addr = ctx.address();

        tokio::spawn(async move {
            let mcp_client = create_mcp_client(&McpServerType::ClaudeFlow, &host, port);

            match mcp_client.query_agent_list().await {
                Ok(agents) => {
                    info!("[AgentMonitorActor] Retrieved {} agents from MCP TCP server", agents.len());

                    // Convert MultiMcpAgentStatus to AgentStatus
                    let agent_statuses: Vec<AgentStatus> = agents.into_iter().map(|mcp_agent| {
                        AgentStatus {
                            agent_id: mcp_agent.agent_id.clone(),
                            profile: AgentProfile {
                                name: mcp_agent.name.clone(),
                                agent_type: match mcp_agent.agent_type.as_str() {
                                    "coordinator" => AgentType::Coordinator,
                                    "researcher" => AgentType::Researcher,
                                    "coder" => AgentType::Coder,
                                    "analyst" => AgentType::Analyst,
                                    "architect" => AgentType::Architect,
                                    "tester" => AgentType::Tester,
                                    "reviewer" => AgentType::Reviewer,
                                    "optimizer" => AgentType::Optimizer,
                                    "documenter" => AgentType::Documenter,
                                    _ => AgentType::Coordinator,
                                },
                                capabilities: mcp_agent.capabilities.clone(),
                                description: Some("MCP agent retrieved via TCP connection".to_string()),
                                version: "1.0.0".to_string(),
                                tags: vec!["mcp".to_string(), mcp_agent.agent_type.clone()],
                            },
                            status: mcp_agent.status.clone(),
                            active_tasks_count: mcp_agent.performance.tasks_active,
                            completed_tasks_count: mcp_agent.performance.tasks_completed,
                            failed_tasks_count: mcp_agent.performance.tasks_failed,
                            success_rate: mcp_agent.performance.success_rate,
                            timestamp: Utc::now(),
                            current_task: None,

                            // Client compatibility fields
                            agent_type: mcp_agent.agent_type.clone(),
                            current_task_description: None,
                            capabilities: mcp_agent.capabilities.clone(),
                            position: None,
                            cpu_usage: mcp_agent.performance.cpu_usage,
                            memory_usage: mcp_agent.performance.memory_usage,
                            health: mcp_agent.performance.health_score,
                            activity: mcp_agent.performance.activity_level,
                            tasks_active: mcp_agent.performance.tasks_active,
                            tasks_completed: mcp_agent.performance.tasks_completed,
                            success_rate_normalized: mcp_agent.performance.success_rate,
                            tokens: mcp_agent.performance.token_usage,
                            token_rate: mcp_agent.performance.token_rate,
                            created_at: Utc::now().to_rfc3339(),
                            age: 0,
                            workload: Some(mcp_agent.performance.activity_level),

                            performance_metrics: PerformanceMetrics {
                                tasks_completed: mcp_agent.performance.tasks_completed,
                                success_rate: mcp_agent.performance.success_rate,
                            },
                            token_usage: TokenUsage {
                                total: mcp_agent.performance.token_usage,
                                token_rate: mcp_agent.performance.token_rate,
                            },
                            swarm_id: Some(mcp_agent.swarm_id),
                            agent_mode: Some("autonomous".to_string()),
                            parent_queen_id: mcp_agent.metadata.parent_id,
                            processing_logs: None,
                        }
                    }).collect();

                    ctx_addr.do_send(ProcessAgentStatuses { agents: agent_statuses });
                }
                Err(e) => {
                    error!("[AgentMonitorActor] MCP TCP query failed: {}", e);
                    ctx_addr.do_send(RecordPollFailure);
                }
            }
        });
    }
}

/// Message to process agent statuses from MCP
#[derive(Message)]
#[rtype(result = "()")]
struct ProcessAgentStatuses {
    agents: Vec<AgentStatus>,
}

impl Actor for AgentMonitorActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[AgentMonitorActor] Started - beginning MCP TCP polling");

        self.is_connected = true;

        // Start periodic polling
        ctx.run_later(Duration::from_millis(100), |act, ctx| {
            ctx.run_interval(act.polling_interval, |act, ctx| {
                act.poll_agent_statuses(ctx);
            });
        });
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("[AgentMonitorActor] Stopped");
    }
}

impl Handler<ProcessAgentStatuses> for AgentMonitorActor {
    type Result = ();

    fn handle(&mut self, msg: ProcessAgentStatuses, _ctx: &mut Self::Context) {
        info!("[AgentMonitorActor] Processing {} agent statuses from MCP", msg.agents.len());

        // Convert AgentStatus to Agent for UpdateBotsGraph
        let agents: Vec<crate::services::bots_client::Agent> = msg.agents.iter().map(|status| {
            crate::services::bots_client::Agent {
                id: status.agent_id.clone(),
                name: status.profile.name.clone(),
                agent_type: format!("{:?}", status.profile.agent_type).to_lowercase(),
                status: status.status.clone(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
                cpu_usage: status.cpu_usage,
                memory_usage: status.memory_usage,
                health: status.health,
                workload: status.activity,
                created_at: Some(status.timestamp.to_rfc3339()),
                age: Some((chrono::Utc::now().timestamp() - status.timestamp.timestamp()) as u64 * 1000),
            }
        }).collect();

        // Send graph update
        let message = UpdateBotsGraph { agents };
        info!("[AgentMonitorActor] Sending graph update with {} agents", msg.agents.len());
        self.graph_service_addr.do_send(message);

        // Update cache
        if !msg.agents.is_empty() {
            self.agent_cache.clear();
            for agent in msg.agents {
                self.agent_cache.insert(agent.agent_id.clone(), agent);
            }
        }

        // Mark poll as successful
        self.consecutive_poll_failures = 0;
        self.last_successful_poll = Some(Utc::now());
    }
}

impl Handler<RecordPollFailure> for AgentMonitorActor {
    type Result = ();

    fn handle(&mut self, _: RecordPollFailure, _ctx: &mut Self::Context) {
        self.consecutive_poll_failures += 1;
        warn!("[AgentMonitorActor] Poll failure recorded - {} consecutive failures",
              self.consecutive_poll_failures);
    }
}

impl Handler<UpdateAgentCache> for AgentMonitorActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateAgentCache, _ctx: &mut Self::Context) {
        debug!("[AgentMonitorActor] Updating agent cache with {} agents", msg.agents.len());

        self.agent_cache.clear();
        for agent in msg.agents {
            self.agent_cache.insert(agent.agent_id.clone(), agent);
        }

        debug!("[AgentMonitorActor] Agent cache updated: {} agents", self.agent_cache.len());
    }
}
