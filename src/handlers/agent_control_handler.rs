use actix_web::{web, HttpResponse, Result};
use log::{info, error};
use serde::{Deserialize, Serialize};
use crate::app_state::AppState;
use crate::services::agent_control_client::{InitializeSwarm, GetAllAgents, GetVisualizationSnapshot};

#[derive(Deserialize)]
pub struct InitializeSwarmRequest {
    pub topology: String,
    #[serde(rename = "maxAgents")]
    pub max_agents: Option<u32>,
    pub strategy: Option<String>,
    #[serde(rename = "enableNeural")]
    pub enable_neural: Option<bool>,
    #[serde(rename = "agentTypes")]
    pub agent_types: Vec<String>,
    #[serde(rename = "customPrompt")]
    pub custom_prompt: Option<String>,
}

#[derive(Serialize)]
pub struct InitializeSwarmResponse {
    pub success: bool,
    pub message: Option<String>,
    pub error: Option<String>,
    pub data: Option<serde_json::Value>,
}

/// Initialize a new agent swarm
pub async fn initialize_swarm(
    data: web::Data<AppState>,
    req: web::Json<InitializeSwarmRequest>,
) -> Result<HttpResponse> {
    info!("Initializing swarm with topology: {}", req.topology);
    
    if let Some(agent_control_addr) = &data.agent_control_addr {
        match agent_control_addr.send(InitializeSwarm {
            topology: req.topology.clone(),
            agent_types: req.agent_types.clone(),
        }).await {
            Ok(Ok(result)) => {
                Ok(HttpResponse::Ok().json(InitializeSwarmResponse {
                    success: true,
                    message: Some("Swarm initialized successfully".to_string()),
                    error: None,
                    data: Some(result),
                }))
            }
            Ok(Err(e)) => {
                error!("Failed to initialize swarm: {}", e);
                Ok(HttpResponse::Ok().json(InitializeSwarmResponse {
                    success: false,
                    message: None,
                    error: Some(e),
                    data: None,
                }))
            }
            Err(e) => {
                error!("Actor mailbox error: {}", e);
                Ok(HttpResponse::InternalServerError().json(InitializeSwarmResponse {
                    success: false,
                    message: None,
                    error: Some("Internal server error".to_string()),
                    data: None,
                }))
            }
        }
    } else {
        Ok(HttpResponse::ServiceUnavailable().json(InitializeSwarmResponse {
            success: false,
            message: None,
            error: Some("Agent control system not available".to_string()),
            data: None,
        }))
    }
}

/// Get all agents from the control system
pub async fn get_agents(data: web::Data<AppState>) -> Result<HttpResponse> {
    if let Some(agent_control_addr) = &data.agent_control_addr {
        match agent_control_addr.send(GetAllAgents).await {
            Ok(Ok(agents)) => {
                // Convert to bots format for frontend compatibility
                let bots_agents: Vec<serde_json::Value> = agents.into_iter().map(|agent| {
                    serde_json::json!({
                        "id": agent.id,
                        "name": agent.name,
                        "type": agent.agent_type,
                        "status": agent.status,
                        "health": agent.health,
                        "cpuUsage": agent.metrics.cpu_usage,
                        "memoryUsage": agent.metrics.memory_usage,
                        "tasksActive": agent.metrics.tasks_active,
                        "tasksCompleted": agent.metrics.tasks_completed,
                        "successRate": agent.metrics.success_rate,
                        "capabilities": agent.capabilities,
                        "swarmId": agent.swarm_id,
                        "createdAt": agent.created_at,
                        "age": 0, // Calculate from created_at if needed
                        "position": [0.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "force": [0.0, 0.0, 0.0],
                        "connections": []
                    })
                }).collect();
                
                Ok(HttpResponse::Ok().json(serde_json::json!({
                    "agents": bots_agents,
                    "mcpConnected": true
                })))
            }
            Ok(Err(e)) => {
                error!("Failed to get agents: {}", e);
                Ok(HttpResponse::Ok().json(serde_json::json!({
                    "agents": [],
                    "mcpConnected": false,
                    "error": e
                })))
            }
            Err(e) => {
                error!("Actor mailbox error: {}", e);
                Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                    "agents": [],
                    "mcpConnected": false,
                    "error": "Internal server error"
                })))
            }
        }
    } else {
        Ok(HttpResponse::Ok().json(serde_json::json!({
            "agents": [],
            "mcpConnected": false,
            "error": "Agent control system not available"
        })))
    }
}

/// Get visualization snapshot for physics rendering
pub async fn get_visualization(data: web::Data<AppState>) -> Result<HttpResponse> {
    if let Some(agent_control_addr) = &data.agent_control_addr {
        match agent_control_addr.send(GetVisualizationSnapshot).await {
            Ok(Ok(snapshot)) => {
                Ok(HttpResponse::Ok().json(snapshot))
            }
            Ok(Err(e)) => {
                error!("Failed to get visualization: {}", e);
                Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": e
                })))
            }
            Err(e) => {
                error!("Actor mailbox error: {}", e);
                Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                    "error": "Internal server error"
                })))
            }
        }
    } else {
        Ok(HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "error": "Agent control system not available"
        })))
    }
}