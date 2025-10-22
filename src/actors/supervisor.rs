//! Actor supervision system to replace panic! calls with graceful recovery
//!
//! This module provides supervision trees that can restart failed actors
//! and implement exponential backoff retry strategies.

#[cfg(test)]
use crate::errors::ActorError;
use crate::errors::VisionFlowError;
use actix::prelude::*;
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Supervision strategy for handling actor failures
#[derive(Debug, Clone)]
pub enum SupervisionStrategy {
    /// Restart the failed actor immediately
    Restart,
    /// Restart after a delay with exponential backoff
    RestartWithBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
    },
    /// Escalate the error to parent supervisor
    Escalate,
    /// Stop the actor permanently
    Stop,
}

/// Information about a supervised actor
#[derive(Debug, Clone)]
pub struct SupervisedActorInfo {
    pub name: String,
    pub strategy: SupervisionStrategy,
    pub max_restart_count: u32,
    pub restart_window: Duration,
    pub actor_type: String,
    pub is_running: bool,
    pub session_id: Option<String>,
    pub last_heartbeat: DateTime<Utc>,
}

/// Internal state tracking for supervised actors
#[derive(Debug)]
struct ActorState {
    actor_info: SupervisedActorInfo,
    restart_count: u32,
    last_restart: Option<Instant>,
    current_delay: Duration,
    is_running: bool,
    session_id: Option<String>,
}

/// Messages for the supervisor
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct RegisterActor {
    pub actor_name: String,
    pub strategy: SupervisionStrategy,
    pub max_restart_count: u32,
    pub restart_window: Duration,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct ActorFailed {
    pub actor_name: String,
    pub error: VisionFlowError,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct ActorStarted {
    pub actor_name: String,
}

#[derive(Message)]
#[rtype(result = "Result<SupervisionStatus, VisionFlowError>")]
pub struct GetSupervisionStatus;

/// Status information about supervised actors
#[derive(Debug, Clone)]
pub struct SupervisionStatus {
    pub total_actors: usize,
    pub running_actors: usize,
    pub failed_actors: usize,
    pub actors: Vec<ActorStatusInfo>,
}

#[derive(Debug, Clone)]
pub struct ActorStatusInfo {
    pub name: String,
    pub is_running: bool,
    pub restart_count: u32,
    pub last_restart: Option<Instant>,
    pub strategy: SupervisionStrategy,
}

/// The supervisor actor that manages other actors
pub struct SupervisorActor {
    supervised_actors: HashMap<String, ActorState>,
    supervisor_name: String,
}

impl SupervisorActor {
    pub fn new(supervisor_name: String) -> Self {
        Self {
            supervised_actors: HashMap::new(),
            supervisor_name,
        }
    }

    fn should_restart(&self, actor_name: &str, state: &ActorState) -> bool {
        // Check if we've exceeded maximum restart count within the window
        if state.restart_count >= state.actor_info.max_restart_count {
            if let Some(last_restart) = state.last_restart {
                if last_restart.elapsed() < state.actor_info.restart_window {
                    warn!(
                        "Actor '{}' has exceeded max restart count ({}) within window ({:?})",
                        actor_name,
                        state.actor_info.max_restart_count,
                        state.actor_info.restart_window
                    );
                    return false;
                }
                // Window has passed, reset restart count
            }
        }
        true
    }

    fn calculate_restart_delay(&self, state: &ActorState) -> Duration {
        match &state.actor_info.strategy {
            SupervisionStrategy::RestartWithBackoff {
                initial_delay: _,
                max_delay,
                multiplier,
            } => {
                let delay = Duration::from_millis(
                    (state.current_delay.as_millis() as f64 * multiplier) as u64,
                );
                std::cmp::min(delay, *max_delay)
            }
            _ => Duration::from_millis(0),
        }
    }

    fn restart_actor(&mut self, actor_name: &str, ctx: &mut Context<Self>) {
        let delay = if let Some(state) = self.supervised_actors.get(actor_name) {
            self.calculate_restart_delay(state)
        } else {
            return;
        };

        if let Some(state) = self.supervised_actors.get_mut(actor_name) {
            state.restart_count += 1;
            state.last_restart = Some(Instant::now());
            state.current_delay = delay;

            info!(
                "Scheduling restart for actor '{}' in {:?} (attempt {})",
                actor_name, delay, state.restart_count
            );

            let actor_name_clone = actor_name.to_string();
            let supervisor_name = self.supervisor_name.clone();

            ctx.run_later(delay, move |_act, ctx| {
                info!("Attempting to restart actor '{}'", actor_name_clone);

                // In a real implementation, this would call a factory method
                // to recreate the specific actor type. For now, we just mark
                // the restart attempt.
                ctx.notify(RestartAttempt {
                    actor_name: actor_name_clone,
                    supervisor_name,
                });
            });
        }
    }
}

impl Actor for SupervisorActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("Supervisor '{}' started", self.supervisor_name);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Supervisor '{}' stopped", self.supervisor_name);
    }
}

impl Handler<RegisterActor> for SupervisorActor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: RegisterActor, _ctx: &mut Self::Context) -> Self::Result {
        let actor_info = SupervisedActorInfo {
            name: msg.actor_name.clone(),
            strategy: msg.strategy.clone(),
            max_restart_count: msg.max_restart_count,
            restart_window: msg.restart_window,
            actor_type: "generic".to_string(),
            is_running: false,
            session_id: None,
            last_heartbeat: Utc::now(),
        };

        let initial_delay = match &msg.strategy {
            SupervisionStrategy::RestartWithBackoff { initial_delay, .. } => *initial_delay,
            _ => Duration::from_millis(100),
        };

        let state = ActorState {
            actor_info,
            restart_count: 0,
            last_restart: None,
            current_delay: initial_delay,
            is_running: true,
            session_id: None,
        };

        self.supervised_actors.insert(msg.actor_name.clone(), state);
        info!("Registered actor '{}' for supervision", msg.actor_name);
        Ok(())
    }
}

impl Handler<ActorFailed> for SupervisorActor {
    type Result = ();

    fn handle(&mut self, msg: ActorFailed, ctx: &mut Self::Context) {
        error!("Actor '{}' failed: {}", msg.actor_name, msg.error);

        if let Some(state) = self.supervised_actors.get_mut(&msg.actor_name) {
            state.is_running = false;
            let strategy = state.actor_info.strategy.clone();

            // FIX: Simplified logic to avoid borrowing issues and incorrect drop
            let should_restart = match &strategy {
                SupervisionStrategy::Restart | SupervisionStrategy::RestartWithBackoff { .. } => {
                    // Check restart limits directly on the state
                    if state.restart_count >= state.actor_info.max_restart_count {
                        if let Some(last_restart) = state.last_restart {
                            if last_restart.elapsed() < state.actor_info.restart_window {
                                warn!("Actor '{}' has exceeded max restart count ({}) within window ({:?})",
                                      &msg.actor_name, state.actor_info.max_restart_count,
                                      state.actor_info.restart_window);
                                false
                            } else {
                                // Window has passed, reset counter and allow restart
                                state.restart_count = 0;
                                true
                            }
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                }
                _ => false,
            };

            // Now handle the strategy
            match strategy {
                SupervisionStrategy::Restart => {
                    if should_restart {
                        self.restart_actor(&msg.actor_name, ctx);
                    } else {
                        error!(
                            "Actor '{}' will not be restarted (too many failures)",
                            msg.actor_name
                        );
                    }
                }
                SupervisionStrategy::RestartWithBackoff { .. } => {
                    if should_restart {
                        self.restart_actor(&msg.actor_name, ctx);
                    } else {
                        error!(
                            "Actor '{}' will not be restarted (too many failures)",
                            msg.actor_name
                        );
                    }
                }
                SupervisionStrategy::Escalate => {
                    warn!(
                        "Escalating failure of actor '{}' to parent supervisor",
                        msg.actor_name
                    );
                    // In a real implementation, this would notify a parent supervisor
                }
                SupervisionStrategy::Stop => {
                    info!(
                        "Actor '{}' stopped permanently due to supervision strategy",
                        msg.actor_name
                    );
                    if let Some(state) = self.supervised_actors.get_mut(&msg.actor_name) {
                        state.is_running = false;
                    }
                }
            }
        } else {
            warn!(
                "Received failure notification for unregistered actor '{}'",
                msg.actor_name
            );
        }
    }
}

impl Handler<ActorStarted> for SupervisorActor {
    type Result = ();

    fn handle(&mut self, msg: ActorStarted, _ctx: &mut Self::Context) {
        if let Some(state) = self.supervised_actors.get_mut(&msg.actor_name) {
            state.is_running = true;
            info!("Actor '{}' started successfully", msg.actor_name);
        }
    }
}

impl Handler<GetSupervisionStatus> for SupervisorActor {
    type Result = Result<SupervisionStatus, VisionFlowError>;

    fn handle(&mut self, _msg: GetSupervisionStatus, _ctx: &mut Self::Context) -> Self::Result {
        let total_actors = self.supervised_actors.len();
        let running_actors = self
            .supervised_actors
            .values()
            .filter(|s| s.is_running)
            .count();
        let failed_actors = total_actors - running_actors;

        let actors = self
            .supervised_actors
            .iter()
            .map(|(name, state)| ActorStatusInfo {
                name: name.clone(),
                is_running: state.is_running,
                restart_count: state.restart_count,
                last_restart: state.last_restart,
                strategy: state.actor_info.strategy.clone(),
            })
            .collect();

        Ok(SupervisionStatus {
            total_actors,
            running_actors,
            failed_actors,
            actors,
        })
    }
}

#[derive(Message)]
#[rtype(result = "()")]
struct RestartAttempt {
    actor_name: String,
    supervisor_name: String,
}

impl Handler<RestartAttempt> for SupervisorActor {
    type Result = ();

    fn handle(&mut self, msg: RestartAttempt, _ctx: &mut Self::Context) {
        debug!("Processing restart attempt for actor '{}'", msg.actor_name);

        // In a real implementation, this would:
        // 1. Call the appropriate factory method to recreate the actor
        // 2. Start the new actor instance
        // 3. Update the supervision state
        // 4. Notify other parts of the system about the restart

        if let Some(state) = self.supervised_actors.get_mut(&msg.actor_name) {
            // For now, just mark as running (in real implementation, this would happen
            // after successful restart)
            state.is_running = true;
            info!("Actor '{}' restart attempt completed", msg.actor_name);
        }
    }
}

// DEPRECATED: Voice command handler removed - uses legacy DockerHiveMind
// Replace with TaskOrchestratorActor integration

/*
/// Handler for voice commands from the speech system
impl Handler<VoiceCommand> for SupervisorActor {
    type Result = Result<SwarmVoiceResponse, String>;

    fn handle(&mut self, msg: VoiceCommand, _ctx: &mut Self::Context) -> Self::Result {
        info!("Supervisor processing voice command: {:?} (tag: {:?})",
              msg.parsed_intent, msg.voice_tag);

        // Process voice command and generate appropriate response
        let response_text = match &msg.parsed_intent {
            crate::actors::voice_commands::SwarmIntent::SpawnAgent { agent_type, .. } => {
                // Use DockerHiveMind to spawn real agents
                let hive_mind = crate::utils::docker_hive_mind::create_docker_hive_mind();
                let task_description = format!("Spawn {} agent for multi-agent coordination", agent_type);

                match tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(
                        hive_mind.spawn_swarm(&task_description, crate::utils::docker_hive_mind::SwarmConfig::default())
                    )
                }) {
                    Ok(session_id) => {
                        // Track the spawned agent
                        let actor_info = SupervisedActorInfo {
                            name: format!("{}_{}", agent_type, session_id),
                            strategy: SupervisionStrategy::Restart,
                            max_restart_count: 3,
                            restart_window: Duration::from_secs(60),
                            actor_type: agent_type.clone(),
                            is_running: true,
                            session_id: Some(session_id.clone()),
                            last_heartbeat: chrono::Utc::now(),
                        };
                        let actor_state = ActorState {
                            actor_info,
                            restart_count: 0,
                            last_restart: None,
                            current_delay: Duration::from_secs(1),
                            is_running: true,
                            session_id: Some(session_id.clone()),
                        };
                        self.supervised_actors.insert(session_id.clone(), actor_state);

                        format!("Successfully spawned {} agent with session ID: {}", agent_type, session_id)
                    }
                    Err(e) => {
                        error!("Failed to spawn {} agent: {}", agent_type, e);
                        format!("Failed to spawn {} agent: {}", agent_type, e)
                    }
                }
            },
            crate::actors::voice_commands::SwarmIntent::QueryStatus { target } => {
                let target_str = target.as_ref().map(|s| s.as_str()).unwrap_or("all");
                let running_count = self.supervised_actors.values()
                    .filter(|actor| actor.is_running)
                    .count();
                format!("Status for {}: {} supervised actors running", target_str, running_count)
            },
            crate::actors::voice_commands::SwarmIntent::ListAgents => {
                let agent_names: Vec<String> = self.supervised_actors.keys().cloned().collect();
                if agent_names.is_empty() {
                    "No supervised actors are currently registered".to_string()
                } else {
                    format!("Supervised actors: {}", agent_names.join(", "))
                }
            },
            crate::actors::voice_commands::SwarmIntent::StopAgent { agent_id } => {
                if let Some(actor_info) = self.supervised_actors.get_mut(agent_id) {
                    if let Some(ref session_id) = actor_info.session_id {
                        // Use DockerHiveMind to terminate the agent
                        let hive_mind = crate::utils::docker_hive_mind::create_docker_hive_mind();

                        match tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(
                                hive_mind.terminate_swarm(session_id)
                            )
                        }) {
                            Ok(_) => {
                                actor_info.is_running = false;
                                format!("Successfully stopped agent: {}", agent_id)
                            }
                            Err(e) => {
                                error!("Failed to stop agent {}: {}", agent_id, e);
                                format!("Failed to stop agent {}: {}", agent_id, e)
                            }
                        }
                    } else {
                        actor_info.is_running = false;
                        format!("Stopped agent: {} (no active session)", agent_id)
                    }
                } else {
                    format!("Agent {} not found", agent_id)
                }
            },
            crate::actors::voice_commands::SwarmIntent::ExecuteTask { description, .. } => {
                // Use DockerHiveMind to execute tasks
                let hive_mind = crate::utils::docker_hive_mind::create_docker_hive_mind();

                match tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(
                        hive_mind.spawn_swarm(description, crate::utils::docker_hive_mind::SwarmConfig {
                            priority: crate::utils::docker_hive_mind::SwarmPriority::High,
                            strategy: crate::utils::docker_hive_mind::SwarmStrategy::Tactical,
                            ..Default::default()
                        })
                    )
                }) {
                    Ok(session_id) => {
                        // Track the task execution session
                        let task_info = SupervisedActorInfo {
                            name: format!("task-executor_{}", session_id),
                            strategy: SupervisionStrategy::Restart,
                            max_restart_count: 3,
                            restart_window: Duration::from_secs(60),
                            actor_type: "task-executor".to_string(),
                            is_running: true,
                            session_id: Some(session_id.clone()),
                            last_heartbeat: chrono::Utc::now(),
                        };
                        let actor_state = ActorState {
                            actor_info: task_info,
                            restart_count: 0,
                            last_restart: None,
                            current_delay: Duration::from_secs(1),
                            is_running: true,
                            session_id: Some(session_id.clone()),
                        };
                        self.supervised_actors.insert(session_id.clone(), actor_state);

                        format!("Task execution started with session ID: {}", session_id)
                    }
                    Err(e) => {
                        error!("Failed to execute task '{}': {}", description, e);
                        format!("Failed to execute task: {}", e)
                    }
                }
            },
            crate::actors::voice_commands::SwarmIntent::UpdateGraph { .. } => {
                // Request graph update from GraphServiceActor
                // Graph service functionality temporarily disabled for compilation
                warn!("Graph service temporarily disabled for compilation");
                "Graph service temporarily disabled - operation skipped".to_string()
            },
            crate::actors::voice_commands::SwarmIntent::Help => {
                "I can help you spawn agents, check status, list agents, or execute tasks. What would you like to do?".to_string()
            }
        };

        // Create response with tag if present
        let mut response = SwarmVoiceResponse {
            text: response_text,
            use_voice: msg.respond_via_voice,
            metadata: None,
            follow_up: None,
            voice_tag: msg.voice_tag.clone(),
            is_final: Some(true),
        };

        // Add contextual follow-up for certain commands
        match &msg.parsed_intent {
            crate::actors::voice_commands::SwarmIntent::SpawnAgent { .. } => {
                response.follow_up = Some("What would you like the agent to do?".to_string());
            },
            crate::actors::voice_commands::SwarmIntent::Help => {
                response.follow_up = Some("Try saying 'spawn researcher agent' or 'show status'.".to_string());
            },
            _ => {}
        }

        Ok(response)
    }
}
*/

/// Helper trait for actors to integrate with supervision
pub trait SupervisedActorTrait: Actor {
    fn actor_name() -> &'static str;

    fn supervision_strategy() -> SupervisionStrategy {
        SupervisionStrategy::RestartWithBackoff {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
        }
    }

    fn max_restart_count() -> u32 {
        5
    }

    fn restart_window() -> Duration {
        Duration::from_secs(300) // 5 minutes
    }

    /// Called when the actor encounters an error that might require supervision
    fn report_error(&self, supervisor: &Addr<SupervisorActor>, error: VisionFlowError) {
        supervisor.do_send(ActorFailed {
            actor_name: Self::actor_name().to_string(),
            error,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[actix::test]
    async fn test_actor_registration() {
        let supervisor = SupervisorActor::new("TestSupervisor".to_string()).start();

        let register_msg = RegisterActor {
            actor_name: "TestActor".to_string(),
            strategy: SupervisionStrategy::Restart,
            max_restart_count: 3,
            restart_window: Duration::from_secs(60),
        };

        let result = supervisor.send(register_msg).await.unwrap();
        assert!(result.is_ok());

        let status = supervisor
            .send(GetSupervisionStatus)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(status.total_actors, 1);
        assert_eq!(status.running_actors, 1);
    }

    #[actix::test]
    async fn test_actor_failure_handling() {
        let supervisor = SupervisorActor::new("TestSupervisor".to_string()).start();

        // Register an actor
        let register_msg = RegisterActor {
            actor_name: "TestActor".to_string(),
            strategy: SupervisionStrategy::Restart,
            max_restart_count: 3,
            restart_window: Duration::from_secs(60),
        };

        supervisor.send(register_msg).await.unwrap().unwrap();

        // Simulate actor failure
        let failure_msg = ActorFailed {
            actor_name: "TestActor".to_string(),
            error: VisionFlowError::Actor(ActorError::RuntimeFailure {
                actor_name: "TestActor".to_string(),
                reason: "Test failure".to_string(),
            }),
        };

        supervisor.send(failure_msg).await.unwrap();

        // Give some time for processing
        sleep(Duration::from_millis(100)).await;

        let status = supervisor
            .send(GetSupervisionStatus)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(status.total_actors, 1);
    }
}
