//! Management API Client for Multi-Agent Docker Container
//!
//! This client provides HTTP access to the Management API (port 9090) running in the
//! agentic-workstation container. It handles all task creation, monitoring, and control
//! operations, replacing the legacy DockerHiveMind system.
//!
//! ## Architecture
//!
//! VisionFlow Container → ManagementApiClient (HTTP) → agentic-workstation:9090 → Management API
//!
//! ## Features
//!
//! - Task creation via POST /v1/tasks
//! - Task status polling via GET /v1/tasks/:taskId
//! - Task cancellation via DELETE /v1/tasks/:taskId
//! - System status via GET /v1/status
//! - Automatic retry with exponential backoff
//! - Bearer token authentication

use log::{debug, info};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct ManagementApiClient {
    base_url: String,
    api_key: String,
    client: Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskResponse {
    pub task_id: String,
    pub status: String,
    pub message: String,
    pub task_dir: Option<String>,
    pub log_file: Option<String>,
    pub start_time: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatus {
    pub task_id: String,
    pub agent: String,
    pub task: String,
    pub provider: String,
    pub status: TaskState,
    pub start_time: u64,
    pub exit_time: Option<u64>,
    pub exit_code: Option<i32>,
    pub duration: u64,
    pub task_dir: String,
    pub log_file: String,
    pub log_tail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskState {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskInfo {
    pub task_id: String,
    pub agent: String,
    pub task: String,
    pub provider: String,
    pub status: TaskState,
    pub start_time: u64,
    pub duration: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskListResponse {
    pub active_tasks: Vec<TaskInfo>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub timestamp: String,
    pub api: ApiStatus,
    pub tasks: TasksStatus,
    pub gpu: Option<GpuStatus>,
    pub providers: serde_json::Value,
    pub system: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiStatus {
    pub uptime: u64,
    pub version: String,
    pub pid: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasksStatus {
    pub active: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    pub available: bool,
    pub gpus: Option<Vec<serde_json::Value>>,
}

#[derive(Debug)]
pub enum ManagementApiError {
    NetworkError(String),
    ApiError(String, StatusCode),
    DeserializationError(String),
    Timeout,
}

impl std::fmt::Display for ManagementApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManagementApiError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ManagementApiError::ApiError(msg, status) => {
                write!(f, "API error ({}): {}", status, msg)
            }
            ManagementApiError::DeserializationError(msg) => {
                write!(f, "Deserialization error: {}", msg)
            }
            ManagementApiError::Timeout => write!(f, "Request timeout"),
        }
    }
}

impl std::error::Error for ManagementApiError {}

impl ManagementApiClient {
    
    
    
    
    
    
    
    pub fn new(host: String, port: u16, api_key: String) -> Self {
        let base_url = format!("http://{}:{}", host, port);

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        info!(
            "[ManagementApiClient] Initialized with base_url: {}",
            base_url
        );

        Self {
            base_url,
            api_key,
            client,
        }
    }

    
    
    
    
    
    
    
    pub async fn create_task(
        &self,
        agent: &str,
        task: &str,
        provider: &str,
    ) -> Result<TaskResponse, ManagementApiError> {
        let url = format!("{}/v1/tasks", self.base_url);

        let request_body = serde_json::json!({
            "agent": agent,
            "task": task,
            "provider": provider,
        });

        debug!(
            "[ManagementApiClient] Creating task: agent={}, provider={}",
            agent, provider
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ManagementApiError::NetworkError(e.to_string()))?;

        let status = response.status();

        if status == StatusCode::ACCEPTED || status == StatusCode::OK {
            let task_response: TaskResponse = response
                .json()
                .await
                .map_err(|e| ManagementApiError::DeserializationError(e.to_string()))?;

            info!(
                "[ManagementApiClient] Task created: {}",
                task_response.task_id
            );
            Ok(task_response)
        } else {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ManagementApiError::ApiError(error_text, status))
        }
    }

    
    pub async fn get_task_status(&self, task_id: &str) -> Result<TaskStatus, ManagementApiError> {
        let url = format!("{}/v1/tasks/{}", self.base_url, task_id);

        debug!("[ManagementApiClient] Getting task status: {}", task_id);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| ManagementApiError::NetworkError(e.to_string()))?;

        let status = response.status();

        if status == StatusCode::OK {
            let task_status: TaskStatus = response
                .json()
                .await
                .map_err(|e| ManagementApiError::DeserializationError(e.to_string()))?;
            Ok(task_status)
        } else {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ManagementApiError::ApiError(error_text, status))
        }
    }

    
    pub async fn list_tasks(&self) -> Result<TaskListResponse, ManagementApiError> {
        let url = format!("{}/v1/tasks", self.base_url);

        debug!("[ManagementApiClient] Listing tasks");

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| ManagementApiError::NetworkError(e.to_string()))?;

        let status = response.status();

        if status == StatusCode::OK {
            let task_list: TaskListResponse = response
                .json()
                .await
                .map_err(|e| ManagementApiError::DeserializationError(e.to_string()))?;
            Ok(task_list)
        } else {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ManagementApiError::ApiError(error_text, status))
        }
    }

    
    pub async fn stop_task(&self, task_id: &str) -> Result<(), ManagementApiError> {
        let url = format!("{}/v1/tasks/{}", self.base_url, task_id);

        info!("[ManagementApiClient] Stopping task: {}", task_id);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| ManagementApiError::NetworkError(e.to_string()))?;

        let status = response.status();

        if status == StatusCode::OK {
            info!("[ManagementApiClient] Task stopped: {}", task_id);
            Ok(())
        } else {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ManagementApiError::ApiError(error_text, status))
        }
    }

    
    pub async fn get_system_status(&self) -> Result<SystemStatus, ManagementApiError> {
        let url = format!("{}/v1/status", self.base_url);

        debug!("[ManagementApiClient] Getting system status");

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| ManagementApiError::NetworkError(e.to_string()))?;

        let status = response.status();

        if status == StatusCode::OK {
            let system_status: SystemStatus = response
                .json()
                .await
                .map_err(|e| ManagementApiError::DeserializationError(e.to_string()))?;
            Ok(system_status)
        } else {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(ManagementApiError::ApiError(error_text, status))
        }
    }

    
    pub async fn health_check(&self) -> Result<bool, ManagementApiError> {
        let url = format!("{}/health", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ManagementApiError::NetworkError(e.to_string()))?;

        Ok(response.status() == StatusCode::OK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = ManagementApiClient::new(
            "agentic-workstation".to_string(),
            9090,
            "test-key".to_string(),
        );

        assert_eq!(client.base_url, "http://agentic-workstation:9090");
        assert_eq!(client.api_key, "test-key");
    }
}
