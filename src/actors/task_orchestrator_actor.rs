//! Task Orchestrator Actor
//!
//! Actix actor wrapper for ManagementApiClient that provides:
//! - Async task creation with retry logic
//! - Task state caching and tracking
//! - Background polling for task completion
//! - Actor message-based API for integration with VisionFlow actor system
//!
//! ## Architecture
//!
//! ```
//! VisionFlow API Handler
//!     ↓ (send CreateTask message)
//! TaskOrchestratorActor
//!     ↓ (HTTP POST)
//! ManagementApiClient
//!     ↓ (HTTP request)
//! agentic-workstation:9090/v1/tasks
//! ```

use actix::prelude::*;
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::time::Duration;

use crate::services::management_api_client::{
    ManagementApiClient, ManagementApiError, TaskInfo, TaskResponse, TaskState as ApiTaskState,
    TaskStatus as ApiTaskStatus,
};

///
#[derive(Debug, Clone)]
pub struct TaskState {
    pub task_id: String,
    pub agent: String,
    pub task_description: String,
    pub provider: String,
    pub status: ApiTaskState,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub retry_count: u32,
}

///
pub struct TaskOrchestratorActor {
    api_client: ManagementApiClient,
    active_tasks: HashMap<String, TaskState>,
    max_retries: u32,
    retry_delay: Duration,
}

impl TaskOrchestratorActor {
    
    pub fn new(api_client: ManagementApiClient) -> Self {
        info!("[TaskOrchestratorActor] Initializing");
        Self {
            api_client,
            active_tasks: HashMap::new(),
            max_retries: 3,
            retry_delay: Duration::from_secs(2),
        }
    }

    
    async fn create_task_with_retry(
        &mut self,
        agent: String,
        task: String,
        provider: String,
    ) -> Result<TaskResponse, ManagementApiError> {
        let mut attempts = 0;

        loop {
            match self.api_client.create_task(&agent, &task, &provider).await {
                Ok(response) => {
                    info!(
                        "[TaskOrchestratorActor] Task created successfully: {}",
                        response.task_id
                    );

                    
                    self.active_tasks.insert(
                        response.task_id.clone(),
                        TaskState {
                            task_id: response.task_id.clone(),
                            agent: agent.clone(),
                            task_description: task.clone(),
                            provider: provider.clone(),
                            status: ApiTaskState::Running,
                            created_at: Utc::now(),
                            last_updated: Utc::now(),
                            retry_count: attempts,
                        },
                    );

                    return Ok(response);
                }
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_retries {
                        error!(
                            "[TaskOrchestratorActor] Task creation failed after {} attempts: {}",
                            attempts, e
                        );
                        return Err(e);
                    }

                    warn!(
                        "[TaskOrchestratorActor] Task creation attempt {} failed: {}, retrying...",
                        attempts, e
                    );
                    tokio::time::sleep(self.retry_delay * attempts).await;
                }
            }
        }
    }

    
    fn update_task_status(&mut self, task_status: ApiTaskStatus) {
        if let Some(task) = self.active_tasks.get_mut(&task_status.task_id) {
            task.status = task_status.status.clone();
            task.last_updated = Utc::now();

            
            if task_status.status == ApiTaskState::Completed
                || task_status.status == ApiTaskState::Failed
            {
                debug!(
                    "[TaskOrchestratorActor] Task {} finished with status: {:?}",
                    task_status.task_id, task_status.status
                );
            }
        }
    }
}

impl Actor for TaskOrchestratorActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("[TaskOrchestratorActor] Actor started");

        
        ctx.address()
            .do_send(crate::actors::messages::InitializeActor);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("[TaskOrchestratorActor] Actor stopped");
    }
}

impl Handler<crate::actors::messages::InitializeActor> for TaskOrchestratorActor {
    type Result = ();

    fn handle(
        &mut self,
        _msg: crate::actors::messages::InitializeActor,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("[TaskOrchestratorActor] Initializing periodic cleanup (deferred from started)");

        
        ctx.run_interval(Duration::from_secs(300), |act, _ctx| {
            let now = Utc::now();
            let mut to_remove = Vec::new();

            for (task_id, task) in &act.active_tasks {
                
                if (task.status == ApiTaskState::Completed || task.status == ApiTaskState::Failed)
                    && (now - task.last_updated).num_minutes() > 5
                {
                    to_remove.push(task_id.clone());
                }
            }

            for task_id in to_remove {
                debug!(
                    "[TaskOrchestratorActor] Removing old task from cache: {}",
                    task_id
                );
                act.active_tasks.remove(&task_id);
            }
        });
    }
}

// ========================================
// Message Definitions
// ========================================

///
#[derive(Message)]
#[rtype(result = "Result<TaskResponse, String>")]
pub struct CreateTask {
    pub agent: String,
    pub task: String,
    pub provider: String,
}

///
#[derive(Message)]
#[rtype(result = "Result<ApiTaskStatus, String>")]
pub struct GetTaskStatus {
    pub task_id: String,
}

///
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct StopTask {
    pub task_id: String,
}

///
#[derive(Message)]
#[rtype(result = "Result<Vec<TaskInfo>, String>")]
pub struct ListActiveTasks;

///
#[derive(Message)]
#[rtype(result = "Result<SystemStatusInfo, String>")]
pub struct GetSystemStatus;

///
#[derive(Debug, Clone)]
pub struct SystemStatusInfo {
    pub active_tasks: usize,
    pub cached_tasks: usize,
    pub api_available: bool,
}

// ========================================
// Message Handlers
// ========================================

impl Handler<CreateTask> for TaskOrchestratorActor {
    type Result = ResponseFuture<Result<TaskResponse, String>>;

    fn handle(&mut self, msg: CreateTask, _ctx: &mut Self::Context) -> Self::Result {
        info!(
            "[TaskOrchestratorActor] Received CreateTask: agent={}, provider={}",
            msg.agent, msg.provider
        );

        let mut actor = self.clone_for_async();

        Box::pin(async move {
            actor
                .create_task_with_retry(msg.agent, msg.task, msg.provider)
                .await
                .map_err(|e| e.to_string())
        })
    }
}

impl Handler<GetTaskStatus> for TaskOrchestratorActor {
    type Result = ResponseFuture<Result<ApiTaskStatus, String>>;

    fn handle(&mut self, msg: GetTaskStatus, _ctx: &mut Self::Context) -> Self::Result {
        debug!(
            "[TaskOrchestratorActor] Received GetTaskStatus: {}",
            msg.task_id
        );

        let client = self.api_client.clone();
        let task_id = msg.task_id.clone();

        Box::pin(async move {
            client
                .get_task_status(&task_id)
                .await
                .map_err(|e| e.to_string())
        })
    }
}

impl Handler<StopTask> for TaskOrchestratorActor {
    type Result = ResponseFuture<Result<(), String>>;

    fn handle(&mut self, msg: StopTask, _ctx: &mut Self::Context) -> Self::Result {
        info!("[TaskOrchestratorActor] Received StopTask: {}", msg.task_id);

        let client = self.api_client.clone();
        let task_id = msg.task_id.clone();

        Box::pin(async move { client.stop_task(&task_id).await.map_err(|e| e.to_string()) })
    }
}

impl Handler<ListActiveTasks> for TaskOrchestratorActor {
    type Result = ResponseFuture<Result<Vec<TaskInfo>, String>>;

    fn handle(&mut self, _msg: ListActiveTasks, _ctx: &mut Self::Context) -> Self::Result {
        debug!("[TaskOrchestratorActor] Received ListActiveTasks");

        let client = self.api_client.clone();

        Box::pin(async move {
            client
                .list_tasks()
                .await
                .map(|response| response.active_tasks)
                .map_err(|e| e.to_string())
        })
    }
}

impl Handler<GetSystemStatus> for TaskOrchestratorActor {
    type Result = ResponseFuture<Result<SystemStatusInfo, String>>;

    fn handle(&mut self, _msg: GetSystemStatus, _ctx: &mut Self::Context) -> Self::Result {
        debug!("[TaskOrchestratorActor] Received GetSystemStatus");

        let client = self.api_client.clone();
        let cached_tasks = self.active_tasks.len();

        Box::pin(async move {
            match client.get_system_status().await {
                Ok(status) => Ok(SystemStatusInfo {
                    active_tasks: status.tasks.active as usize,
                    cached_tasks,
                    api_available: true,
                }),
                Err(e) => {
                    warn!("[TaskOrchestratorActor] Failed to get system status: {}", e);
                    Ok(SystemStatusInfo {
                        active_tasks: 0,
                        cached_tasks,
                        api_available: false,
                    })
                }
            }
        })
    }
}

// Helper methods for TaskOrchestratorActor
impl TaskOrchestratorActor {
    
    fn clone_for_async(&self) -> Self {
        Self {
            api_client: self.api_client.clone(),
            active_tasks: self.active_tasks.clone(),
            max_retries: self.max_retries,
            retry_delay: self.retry_delay,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::management_api_client::ManagementApiClient;

    #[test]
    fn test_actor_creation() {
        let client = ManagementApiClient::new(
            "agentic-workstation".to_string(),
            9090,
            "test-key".to_string(),
        );

        let actor = TaskOrchestratorActor::new(client);
        assert_eq!(actor.max_retries, 3);
        assert_eq!(actor.active_tasks.len(), 0);
    }
}
