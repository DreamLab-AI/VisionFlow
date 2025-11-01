//! Agent Monitor Actor - Monitoring via Management API
//!
//! This actor focuses solely on:
//! - Polling the Management API (port 9090) for active task statuses
//! - Converting tasks to agent nodes
//! - Forwarding updates to GraphServiceSupervisor
//!
//! All task management is handled by TaskOrchestratorActor.
//! This actor only monitors and displays running agents.

use actix::prelude::*;
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::time::Duration;

use crate::actors::messages::*;
use crate::services::management_api_client::{ManagementApiClient, TaskInfo};
use crate::types::claude_flow::{
    AgentProfile, AgentStatus, AgentType, ClaudeFlowClient, PerformanceMetrics, TokenUsage,
};

/
fn task_to_agent_status(task: TaskInfo) -> AgentStatus {
    use chrono::TimeZone;

    
    let agent_type_enum = match task.agent.as_str() {
        "coder" => AgentType::Coder,
        "planner" => AgentType::Coordinator,
        "researcher" => AgentType::Researcher,
        "reviewer" => AgentType::Analyst,
        "tester" => AgentType::Tester,
        _ => AgentType::Coordinator,
    };

    
    let timestamp = chrono::Utc
        .timestamp_millis_opt(task.start_time as i64)
        .single()
        .unwrap_or_else(|| chrono::Utc::now());

    
    let age = (chrono::Utc::now().timestamp_millis() - task.start_time as i64) / 1000;

    AgentStatus {
        
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
        position: None, 
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
/
pub struct AgentMonitorActor {
    _client: ClaudeFlowClient,
    graph_service_addr: Addr<crate::actors::graph_service_supervisor::TransitionalGraphSupervisor>,
    management_api_client: ManagementApiClient,

    
    is_connected: bool,

    
    polling_interval: Duration,
    last_poll: DateTime<Utc>,

    
    agent_cache: HashMap<String, AgentStatus>,

    
    consecutive_poll_failures: u32,
    last_successful_poll: Option<DateTime<Utc>>,
}

impl AgentMonitorActor {
    pub fn new(
        client: ClaudeFlowClient,
        graph_service_addr: Addr<
            crate::actors::graph_service_supervisor::TransitionalGraphSupervisor,
        >,
    ) -> Self {
        info!("[AgentMonitorActor] Initializing with Management API monitoring");

        
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
            graph_service_addr,
            management_api_client,
            is_connected: false,
            polling_interval: Duration::from_secs(3), 
            last_poll: Utc::now(),
            agent_cache: HashMap::new(),
            consecutive_poll_failures: 0,
            last_successful_poll: None,
        }
    }

    
    fn poll_agent_statuses(&mut self, ctx: &mut Context<Self>) {
        debug!("[AgentMonitorActor] Polling active tasks from Management API");

        let api_client = self.management_api_client.clone();
        let ctx_addr = ctx.address();

        ctx.spawn(
            async move {
                match api_client.list_tasks().await {
                    Ok(task_list) => {
                        let active_count = task_list.active_tasks.len();
                        debug!(
                            "[AgentMonitorActor] Retrieved {} active tasks from Management API",
                            active_count
                        );

                        
                        let agents: Vec<AgentStatus> = task_list
                            .active_tasks
                            .into_iter()
                            .map(|task| task_to_agent_status(task))
                            .collect();

                        ctx_addr.do_send(ProcessAgentStatuses { agents });
                    }
                    Err(e) => {
                        error!("[AgentMonitorActor] Management API query failed: {}", e);
                        ctx_addr.do_send(RecordPollFailure);
                    }
                }
            }
            .into_actor(self),
        );
    }
}

/
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

        
        ctx.address()
            .do_send(crate::actors::messages::InitializeActor);
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("[AgentMonitorActor] Stopped");
    }
}

impl Handler<crate::actors::messages::InitializeActor> for AgentMonitorActor {
    type Result = ();

    fn handle(
        &mut self,
        _msg: crate::actors::messages::InitializeActor,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("[AgentMonitorActor] Initializing periodic polling (deferred from started)");
        
        ctx.run_later(Duration::from_millis(100), |act, ctx| {
            ctx.run_interval(act.polling_interval, |act, ctx| {
                act.poll_agent_statuses(ctx);
            });
        });
    }
}

impl Handler<ProcessAgentStatuses> for AgentMonitorActor {
    type Result = ();

    fn handle(&mut self, msg: ProcessAgentStatuses, _ctx: &mut Self::Context) {
        info!(
            "[AgentMonitorActor] Processing {} agent statuses from MCP",
            msg.agents.len()
        );

        
        let agents: Vec<crate::services::bots_client::Agent> = msg
            .agents
            .iter()
            .map(|status| crate::services::bots_client::Agent {
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
                age: Some(
                    (chrono::Utc::now().timestamp() - status.timestamp.timestamp()) as u64 * 1000,
                ),
            })
            .collect();

        
        let message = UpdateBotsGraph { agents };
        info!(
            "[AgentMonitorActor] Sending graph update with {} agents",
            msg.agents.len()
        );
        self.graph_service_addr.do_send(message);

        
        if !msg.agents.is_empty() {
            self.agent_cache.clear();
            for agent in msg.agents {
                self.agent_cache.insert(agent.agent_id.clone(), agent);
            }
        }

        
        self.consecutive_poll_failures = 0;
        self.last_successful_poll = Some(Utc::now());
    }
}

impl Handler<RecordPollFailure> for AgentMonitorActor {
    type Result = ();

    fn handle(&mut self, _: RecordPollFailure, _ctx: &mut Self::Context) {
        self.consecutive_poll_failures += 1;
        warn!(
            "[AgentMonitorActor] Poll failure recorded - {} consecutive failures",
            self.consecutive_poll_failures
        );
    }
}

impl Handler<UpdateAgentCache> for AgentMonitorActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateAgentCache, _ctx: &mut Self::Context) {
        debug!(
            "[AgentMonitorActor] Updating agent cache with {} agents",
            msg.agents.len()
        );

        self.agent_cache.clear();
        for agent in msg.agents {
            self.agent_cache.insert(agent.agent_id.clone(), agent);
        }

        debug!(
            "[AgentMonitorActor] Agent cache updated: {} agents",
            self.agent_cache.len()
        );
    }
}
