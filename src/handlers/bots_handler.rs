use actix_web::{web, HttpResponse, Responder, HttpRequest, Result};
use crate::AppState;
use crate::handlers::validation_handler::ValidationService;
use crate::utils::validation::rate_limit::{RateLimiter, EndpointRateLimits, extract_client_id};
use crate::utils::validation::errors::DetailedValidationError;
use crate::utils::validation::MAX_REQUEST_SIZE;
use crate::models::graph::GraphData;
use crate::models::metadata::MetadataStore;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::models::node::Node;
use crate::types::vec3::Vec3Data;
use crate::models::edge::Edge;
use crate::models::simulation_params::{SimulationParams};
use crate::actors::messages::{GetSettings, GetBotsGraphData};
use crate::services::bots_client::{BotsClient, Agent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use log::{info, debug, error, warn};
use tokio::sync::RwLock;
use std::sync::Arc;
use tokio::time::timeout;
use std::time::Duration;
use chrono;
use glam::Vec3;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BotsDataRequest {
    pub nodes: Vec<Agent>,
    pub edges: Vec<serde_json::Value>, // Generic edges for now
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BotsResponse {
    pub success: bool,
    pub message: String,
    pub nodes: Option<Vec<Node>>,
    pub edges: Option<Vec<Edge>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeSwarmRequest {
    pub topology: String,
    pub max_agents: u32,
    pub strategy: String,
    pub enable_neural: bool,
    pub agent_types: Vec<String>,
    pub custom_prompt: Option<String>,
}

// Static bots graph data storage
use once_cell::sync::Lazy;
static BOTS_GRAPH: Lazy<Arc<RwLock<GraphData>>> =
    Lazy::new(|| Arc::new(RwLock::new(GraphData::new())));
static CURRENT_SWARM_ID: Lazy<Arc<RwLock<Option<String>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

pub async fn fetch_hive_mind_agents(state: &AppState) -> Result<Vec<Agent>, Box<dyn std::error::Error>> {
    // Use the unified BotsClient which already handles MCP TCP connections
    match state.bots_client.get_agents_snapshot().await {
        Ok(agents) => {
            info!("Retrieved {} agents from BotsClient", agents.len());
            Ok(agents)
        }
        Err(e) => {
            error!("Failed to get agents from BotsClient: {}", e);
            Err(e.into())
        }
    }
}

// Enhanced agent to nodes conversion with hive-mind properties and Queen agent special handling
fn convert_agents_to_nodes(agents: Vec<Agent>) -> Vec<Node> {
    agents.into_iter().enumerate().map(|(idx, agent)| {
        // Map agent ID to numeric ID for physics processing
        let node_id = (idx + 1000) as u32; // Start at 1000 to avoid conflicts

        // Enhanced positioning based on agent type and hierarchy
        let (_radius, vertical_offset) = match agent.agent_type.as_str() {
            "queen" => (0.0, 0.0), // Queen at center
            "coordinator" => (20.0, 2.0),
            "researcher" => (30.0, 0.0),
            "analyst" => (30.0, 0.0),
            "coder" => (40.0, -1.0),
            "optimizer" => (40.0, -1.0),
            "tester" => (50.0, -2.0),
            _ => (60.0, -3.0),
        };

        // Node color and size based on agent type
        let (color, size) = match agent.agent_type.as_str() {
            "queen" => ("#FFD700", 25.0), // Gold for Queen
            "coordinator" => ("#FF6B6B", 20.0), // Red for coordinators
            "researcher" => ("#4ECDC4", 18.0), // Teal for researchers
            "analyst" => ("#45B7D1", 18.0), // Blue for analysts
            "coder" => ("#95E1D3", 16.0), // Mint for coders
            "optimizer" => ("#F38181", 16.0), // Coral for optimizers
            "tester" => ("#F6B93B", 14.0), // Orange for testers
            "worker" => ("#B8E994", 12.0), // Light green for workers
            _ => ("#DFE4EA", 10.0), // Gray for unknown types
        };

        Node {
            id: node_id,
            hash: agent.id.clone(),
            name: agent.name.clone(),
            label: format!("{} ({})", agent.name, agent.agent_type),
            label_url: None,
            color: color.to_string(),
            size,
            size3d: Vec3::new(size, size, size),
            position: Vec3::new(agent.x, agent.y + vertical_offset, agent.z),
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            force: Vec3::ZERO,
            spring_force: Vec3::ZERO,
            charge_force: Vec3::ZERO,
            mass: 1.0,
            edges: vec![],
            metadata: None,
            cluster_id: Some(agent.agent_type.clone()),
            importance_score: None,
            embedding: None,
            block_properties: None,
            created_at: agent.created_at.as_ref()
                .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
                .map(|dt| dt.timestamp_millis() as u64),
            updated_at: Some(chrono::Utc::now().timestamp_millis() as u64),
            ai_generated: true,
            content_preview: Some(format!("Status: {}", agent.status)),
            status: Some(agent.status.clone()),
            properties: Some(json!({
                "agent_type": agent.agent_type,
                "cpu_usage": agent.cpu_usage,
                "memory_usage": agent.memory_usage,
                "health": agent.health,
                "workload": agent.workload,
                "age": agent.age,
            })),
        }
    }).collect()
}

pub async fn update_bots_graph(request: web::Json<BotsDataRequest>, state: web::Data<AppState>) -> Result<impl Responder> {
    info!("Received bots graph update with {} nodes", request.nodes.len());

    let nodes = convert_agents_to_nodes(request.nodes.clone());
    let edges = vec![]; // TODO: Extract edges from request

    let mut graph = BOTS_GRAPH.write().await;
    graph.nodes = nodes;
    graph.edges = edges;
    graph.metadata = MetadataStore::default();

    Ok(HttpResponse::Ok().json(BotsResponse {
        success: true,
        message: "Graph updated successfully".to_string(),
        nodes: Some(graph.nodes.clone()),
        edges: Some(graph.edges.clone()),
    }))
}

pub async fn get_bots_data(state: web::Data<AppState>) -> Result<impl Responder> {
    // First try to get data from graph actor if available
    if let Ok(graph_data) = state.graph_service_addr.send(GetBotsGraphData).await {
        if let Ok((nodes, edges)) = graph_data {
            if !nodes.is_empty() {
                info!("Retrieved bots data from graph actor: {} nodes", nodes.len());
                return Ok(HttpResponse::Ok().json(json!({
                    "success": true,
                    "nodes": nodes,
                    "edges": edges,
                })));
            }
        }
    }

    // Fall back to static storage
    let graph = BOTS_GRAPH.read().await;
    info!("Retrieved bots data from static storage: {} nodes", graph.nodes.len());

    Ok(HttpResponse::Ok().json(json!({
        "success": true,
        "nodes": graph.nodes.clone(),
        "edges": graph.edges.clone(),
        "metadata": graph.metadata,
    })))
}

// Rate limited wrapper for update_bots_graph
pub async fn update_bots_graph_limited(
    req: HttpRequest,
    validation: web::Data<ValidationService>,
    request: web::Json<BotsDataRequest>,
    state: web::Data<AppState>,
) -> Result<impl Responder> {
    info!("update_bots_graph_limited endpoint called");
    
    let client_id = extract_client_id(&req)?;
    
    match validation.check_rate_limit_only(&client_id, &EndpointRateLimits::BotsUpdate).await {
        Ok(_) => update_bots_graph(request, state).await,
        Err(e) => {
            warn!("Rate limit exceeded for client {}: {:?}", client_id, e);
            match e {
                DetailedValidationError::RateLimit { requests_per_minute, current_count } => {
                    Ok(HttpResponse::TooManyRequests()
                        .insert_header(("X-RateLimit-Limit", requests_per_minute.to_string()))
                        .insert_header(("X-RateLimit-Remaining", (requests_per_minute.saturating_sub(current_count)).to_string()))
                        .insert_header(("X-RateLimit-Reset", "60"))
                        .json(json!({
                            "error": "Rate limit exceeded",
                            "limit": requests_per_minute,
                            "current": current_count,
                            "retry_after_seconds": 60
                        })))
                },
                _ => Ok(HttpResponse::BadRequest().json(json!({
                    "error": format!("Validation failed: {}", e)
                }))),
            }
        }
    }
}

// Rate limited wrapper for get_bots_data
pub async fn get_bots_data_limited(
    req: HttpRequest,
    validation: web::Data<ValidationService>,
    state: web::Data<AppState>,
) -> Result<impl Responder> {
    info!("get_bots_data_limited endpoint called");
    
    let client_id = extract_client_id(&req)?;
    
    match validation.check_rate_limit_only(&client_id, &EndpointRateLimits::BotsStatus).await {
        Ok(_) => get_bots_data(state).await,
        Err(e) => {
            warn!("Rate limit exceeded for client {}: {:?}", client_id, e);
            match e {
                DetailedValidationError::RateLimit { requests_per_minute, current_count } => {
                    Ok(HttpResponse::TooManyRequests()
                        .insert_header(("X-RateLimit-Limit", requests_per_minute.to_string()))
                        .insert_header(("X-RateLimit-Remaining", (requests_per_minute.saturating_sub(current_count)).to_string()))
                        .insert_header(("X-RateLimit-Reset", "60"))
                        .json(json!({
                            "error": "Rate limit exceeded",
                            "limit": requests_per_minute,
                            "current": current_count,
                            "retry_after_seconds": 60
                        })))
                },
                _ => Ok(HttpResponse::BadRequest().json(json!({
                    "error": format!("Validation failed: {}", e)
                }))),
            }
        }
    }
}

pub async fn initialize_hive_mind_swarm_limited(
    req: HttpRequest,
    validation: web::Data<ValidationService>,
    request: web::Json<InitializeSwarmRequest>,
    state: web::Data<AppState>,
) -> Result<impl Responder> {
    info!("🤖 Initialize hive mind swarm request received");

    let client_id = extract_client_id(&req)?;
    
    match validation.check_rate_limit_only(&client_id, &EndpointRateLimits::BotsSwarmInit).await {
        Ok(_) => initialize_hive_mind_swarm(request, state).await,
        Err(e) => {
            warn!("Rate limit exceeded for client {}: {:?}", client_id, e);
            match e {
                DetailedValidationError::RateLimit { requests_per_minute, current_count } => {
                    Ok(HttpResponse::TooManyRequests()
                        .insert_header(("X-RateLimit-Limit", requests_per_minute.to_string()))
                        .insert_header(("X-RateLimit-Remaining", (requests_per_minute.saturating_sub(current_count)).to_string()))
                        .insert_header(("X-RateLimit-Reset", "60"))
                        .json(json!({
                            "error": "Rate limit exceeded",
                            "limit": requests_per_minute,
                            "current": current_count,
                            "retry_after_seconds": 60
                        })))
                },
                _ => Ok(HttpResponse::BadRequest().json(json!({
                    "error": format!("Validation failed: {}", e)
                }))),
            }
        }
    }
}

async fn initialize_hive_mind_swarm(
    request: web::Json<InitializeSwarmRequest>,
    state: web::Data<AppState>,
) -> Result<impl Responder> {
    info!("🐝 Initializing hive mind swarm with topology: {}", request.topology);
    
    // Test MCP connection first
    match state.bots_client.test_connection().await {
        Ok(true) => info!("✓ MCP server is connected"),
        Ok(false) => {
            error!("✗ MCP server is not connected");
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "success": false,
                "error": "MCP server is not connected. Please check the multi-agent-container is running."
            })));
        }
        Err(e) => {
            error!("✗ Failed to test MCP connection: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "success": false,
                "error": format!("Failed to test MCP connection: {}", e)
            })));
        }
    }

    // Create swarm initialization parameters
    let swarm_params = json!({
        "topology": request.topology,
        "max_agents": request.max_agents,
        "strategy": request.strategy,
        "enable_neural": request.enable_neural,
        "agent_types": request.agent_types,
        "custom_prompt": request.custom_prompt,
    });

    info!("🔧 Swarm params: {:?}", swarm_params);

    // Initialize swarm via MCP
    let mcp_client = &state.bots_client;
    match mcp_client.mcp_client.send_tool_call("initialize_swarm", swarm_params).await {
        Ok(response) => {
            info!("✓ Swarm initialized successfully: {:?}", response);
            
            // Extract swarm ID if available
            let swarm_id = response.get("swarm_id")
                .and_then(|v| v.as_str())
                .map(String::from);
            
            if let Some(id) = &swarm_id {
                let mut current_id = CURRENT_SWARM_ID.write().await;
                *current_id = Some(id.clone());
                info!("📝 Stored swarm ID: {}", id);
            }

            // Immediately fetch agents to get the initial state
            match fetch_hive_mind_agents(&state).await {
                Ok(agents) => {
                    info!("🎯 Initial swarm has {} agents", agents.len());
                    
                    let nodes = convert_agents_to_nodes(agents);
                    let edges = vec![]; // TODO: Generate edges based on swarm topology

                    // Update bots graph
                    let mut graph = BOTS_GRAPH.write().await;
                    graph.nodes = nodes.clone();
                    graph.edges = edges.clone();
                    
                    Ok(HttpResponse::Ok().json(json!({
                        "success": true,
                        "message": "Hive mind swarm initialized successfully",
                        "swarm_id": swarm_id,
                        "initial_agents": nodes.len(),
                        "nodes": nodes,
                        "edges": edges,
                    })))
                }
                Err(e) => {
                    error!("Failed to fetch initial agents: {}", e);
                    Ok(HttpResponse::Ok().json(json!({
                        "success": true,
                        "message": "Swarm initialized but failed to fetch initial agents",
                        "swarm_id": swarm_id,
                        "error": e.to_string(),
                    })))
                }
            }
        }
        Err(e) => {
            error!("✗ Failed to initialize swarm: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "success": false,
                "error": format!("Failed to initialize swarm: {}", e)
            })))
        }
    }
}

pub async fn get_bots_connection_status(state: web::Data<AppState>) -> Result<impl Responder> {
    match state.bots_client.get_status().await {
        Ok(status) => Ok(HttpResponse::Ok().json(status)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to get bots status: {}", e)
        }))),
    }
}

pub async fn get_bots_agents(state: web::Data<AppState>) -> Result<impl Responder> {
    match fetch_hive_mind_agents(&state).await {
        Ok(agents) => Ok(HttpResponse::Ok().json(json!({
            "success": true,
            "agents": agents,
            "count": agents.len(),
        }))),
        Err(e) => Ok(HttpResponse::InternalServerError().json(json!({
            "success": false,
            "error": format!("Failed to fetch agents: {}", e)
        }))),
    }
}