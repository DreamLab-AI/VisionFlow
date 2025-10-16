//! Agent Monitor Actor - Monitoring via Management API
//!
//! This actor focuses solely on:
//! - Polling the Management API (port 9090) for active task statuses
//! - Converting tasks to agent nodes
//! - Forwarding updates to GraphStateActor
//!
//! All task management is handled by TaskOrchestratorActor.
//! This actor only monitors and displays running agents.

use actix::prelude::*;
use std::time::Duration;
use log::{info, error, debug, warn};
use std::collections::HashMap;
use chrono::{Utc, DateTime};

use crate::types::claude_flow::{ClaudeFlowClient, AgentStatus, AgentProfile, AgentType, PerformanceMetrics, TokenUsage};
use crate::actors::messages::*;
use crate::services::management_api_client::{ManagementApiClient, TaskInfo};
use crate::actors::graph_state_actor::{GraphStateActor, AddNodes};
use crate::models::node::Node;
use crate::utils::socket_flow_messages::BinaryNodeData;

/// Convert Management API TaskInfo to AgentStatus for graph visualization
fn task_to_agent_status(task: TaskInfo) -> AgentStatus {
    use chrono::TimeZone;

    // Map agent type string to AgentType enum
    let agent_type_enum = match task.agent.as_str() {
        "coder" => AgentType::Coder,
        "planner" => AgentType::Coordinator,
        "researcher" => AgentType::Researcher,
        "reviewer" => AgentType::Analyst,
        "tester" => AgentType::Tester,
        _ => AgentType::Coordinator,
    };

    // Create timestamp
    let timestamp = chrono::Utc.timestamp_millis_opt(task.start_time as i64)
        .single()
        .unwrap_or_else(|| chrono::Utc::now());

    // Calculate age in seconds
    let age = (chrono::Utc::now().timestamp_millis() - task.start_time as i64) / 1000;

    AgentStatus {
        // Core identification
        agent_id: task.task_id.clone(),
        profile: AgentProfile {
            name: format!("{} ({})", task.agent, &task.task_id[..8]),
            agent_type: agent_type_enum,
            capabilities: vec![format!("Provider: {}", task.provider)],
            description: Some(task.task.chars().take(100).collect::<String>()),
            version: "1.0.0".to_string(),
            tags: vec![task.agent.clone(), task.provider.clone()],
        },
        status: format!("{:?}", task.status),
        active_tasks_count: 1,
        completed_tasks_count: 0,
        failed_tasks_count: 0,
        success_rate: 100.0,
        timestamp,
        current_task: None,
        agent_type: task.agent.clone(),
        current_task_description: Some(task.task.clone()),
        capabilities: vec![format!("Provider: {}", task.provider)],
        position: None, // Will be positioned by physics engine
        cpu_usage: 0.5,
        memory_usage: 200.0,
        health: 1.0,
        activity: 0.8,
        tasks_active: 1,
        tasks_completed: 0,
        success_rate_normalized: 1.0,
        tokens: 0,
        token_rate: 0.0,
        performance_metrics: PerformanceMetrics {
            tasks_completed: 0,
            success_rate: 100.0,
        },
        token_usage: TokenUsage {
            total: 0,
            token_rate: 0.0,
        },
        swarm_id: None,
        agent_mode: Some("active".to_string()),
        parent_queen_id: None,
        processing_logs: None,
        created_at: timestamp.to_rfc3339(),
        age: age as u64,
        workload: Some(0.5),
    }
}

/// Convert AgentStatus to Node for GraphStateActor
fn agent_status_to_node(status: &AgentStatus) -> Node {
    // Use agent_id hash for consistent node ID
    let node_id = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        status.agent_id.hash(&mut hasher);
        // Use high ID range (starting at 10000) to avoid conflicts with main graph
        (hasher.finish() as u32 % 50000) + 10000
    };

    let mut node = Node::new(status.agent_id.clone());
    node.id = node_id;

    // Set position data
    node.data = BinaryNodeData {
        node_id,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        vx: 0.0,
        vy: 0.0,
        vz: 0.0,
    };

    // Add metadata
    node.label = status.profile.name.clone();
    node.size = Some(5.0);
    node.color = Some(match status.profile.agent_type {
        AgentType::Coder => "#00FF00",
        AgentType::Coordinator => "#0000FF",
        AgentType::Researcher => "#FFFF00",
        AgentType::Analyst => "#FF00FF",
        AgentType::Tester => "#00FFFF",
        _ => "#FFFFFF",
    }.to_string());

    node
}

/// AgentMonitorActor - Monitoring via Management API
pub struct AgentMonitorActor {
    _client: ClaudeFlowClient,
    graph_state_addr: Addr<GraphStateActor>,
    management_api_client: ManagementApiClient,

    /// Connection state
    is_connected: bool,

    /// Polling configuration
    polling_interval: Duration,
    last_poll: DateTime<Utc>,

    /// Agent cache (task_id -> AgentStatus)
    agent_cache: HashMap<String, AgentStatus>,

    /// Node ID mapping (agent_id -> node_id)
    node_id_map: HashMap<String, u32>,

    /// Error tracking
    consecutive_poll_failures: u32,
    last_successful_poll: Option<DateTime<Utc>>,
}

impl AgentMonitorActor {
    pub fn new(client: ClaudeFlowClient, graph_state_addr: Addr<GraphStateActor>) -> Self {
        info!("[AgentMonitorActor] Initializing with Management API monitoring");

        // Create Management API client
        let host = std::env::var("MANAGEMENT_API_HOST")
            .unwrap_or_else(|_| "agentic-workstation".to_string());
        let port = std::env::var("MANAGEMENT_API_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(9090);
        let api_key = std::env::var("MANAGEMENT_API_KEY")
            .unwrap_or_else(|_| "change-this-secret-key".to_string());

        let management_api_client = ManagementApiClient::new(host, port, api_key);

        Self {
            _client: client,
            graph_state_addr,
            management_api_client,
            is_connected: false,
            polling_interval: Duration::from_secs(3), // Poll every 3 seconds
            last_poll: Utc::now(),
            agent_cache: HashMap::new(),
            node_id_map: HashMap::new(),
            consecutive_poll_failures: 0,
            last_successful_poll: None,
        }
    }

    /// Poll agent statuses from Management API
    fn poll_agent_statuses(&mut self, ctx: &mut Context<Self>) {
        debug!("[AgentMonitorActor] Polling active tasks from Management API");

        let api_client = self.management_api_client.clone();
        let ctx_addr = ctx.address();

        tokio::spawn(async move {
            match api_client.list_tasks().await {
                Ok(task_list) => {
                    let active_count = task_list.active_tasks.len();
                    debug!("[AgentMonitorActor] Retrieved {} active tasks from Management API", active_count);

                    // Convert tasks to agent statuses
                    let agents: Vec<AgentStatus> = task_list.active_tasks.into_iter().map(|task| {
                        task_to_agent_status(task)
                    }).collect();

                    ctx_addr.do_send(ProcessAgentStatuses { agents });
                }
                Err(e) => {
                    error!("[AgentMonitorActor] Management API query failed: {}", e);
                    ctx_addr.do_send(RecordPollFailure);
                }
            }
        });
    }
}

/// Message to process agent statuses from Management API
#[derive(Message)]
#[rtype(result = "()")]
struct ProcessAgentStatuses {
    agents: Vec<AgentStatus>,
}

impl Actor for AgentMonitorActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[AgentMonitorActor] Started - beginning Management API polling");

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
        info!("[AgentMonitorActor] Processing {} agent statuses", msg.agents.len());

        // Convert AgentStatus to Node for GraphStateActor
        let nodes: Vec<Node> = msg.agents.iter().map(|status| {
            let node = agent_status_to_node(status);
            // Store node ID mapping
            self.node_id_map.insert(status.agent_id.clone(), node.id);
            node
        }).collect();

        // Send to GraphStateActor
        if !nodes.is_empty() {
            info!("[AgentMonitorActor] Sending {} nodes to GraphStateActor", nodes.len());
            let graph_state_addr = self.graph_state_addr.clone();

            tokio::spawn(async move {
                match graph_state_addr.send(AddNodes { nodes }).await {
                    Ok(Ok(added_ids)) => {
                        debug!("[AgentMonitorActor] Successfully added {} nodes to graph state", added_ids.len());
                    }
                    Ok(Err(e)) => {
                        warn!("[AgentMonitorActor] Failed to add nodes: {}", e);
                    }
                    Err(e) => {
                        error!("[AgentMonitorActor] Mailbox error sending to GraphStateActor: {}", e);
                    }
                }
            });
        }

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
