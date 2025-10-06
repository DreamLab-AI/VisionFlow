use actix_web::{web, HttpResponse, Responder, Result};
use crate::AppState;
use crate::models::graph::GraphData;
use crate::models::metadata::MetadataStore;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::models::node::Node;
use crate::types::vec3::Vec3Data;
use crate::models::edge::Edge;
use crate::models::simulation_params::{SimulationParams};
use crate::actors::messages::{GetSettings, GetBotsGraphData};
use crate::services::bots_client::{BotsClient, Agent};
use crate::handlers::hybrid_health_handler::{HybridHealthManager, SpawnSwarmRequest};
use crate::utils::docker_hive_mind::{DockerHiveMind, SessionInfo, SwarmConfig, SwarmPriority, SwarmStrategy};
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpawnAgentHybridRequest {
    pub agent_type: String,
    pub swarm_id: String,
    pub method: String, // "docker" or "mcp-fallback"
    pub priority: Option<String>,
    pub strategy: Option<String>,
    pub config: Option<SpawnAgentConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpawnAgentConfig {
    pub auto_scale: Option<bool>,
    pub monitor: Option<bool>,
    pub max_workers: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpawnAgentResponse {
    pub success: bool,
    pub swarm_id: Option<String>,
    pub error: Option<String>,
    pub method_used: Option<String>,
    pub message: Option<String>,
}

// Static bots graph data storage
use once_cell::sync::Lazy;
static BOTS_GRAPH: Lazy<Arc<RwLock<GraphData>>> =
    Lazy::new(|| Arc::new(RwLock::new(GraphData::new())));
static CURRENT_SWARM_ID: Lazy<Arc<RwLock<Option<String>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Convert Docker sessions to Agent format for compatibility
fn convert_sessions_to_agents(sessions: Vec<SessionInfo>) -> Vec<Agent> {
    sessions.into_iter().enumerate().map(|(idx, session)| {
        let base_id = idx as f32;
        Agent {
            id: session.session_id.clone(),
            name: format!("Swarm-{}", &session.session_id[..8]),
            agent_type: match session.config.strategy {
                SwarmStrategy::Strategic => "queen".to_string(),
                SwarmStrategy::Tactical => "coordinator".to_string(),
                SwarmStrategy::Adaptive => "optimizer".to_string(),
                SwarmStrategy::HiveMind => "worker".to_string(),
            },
            status: match session.status {
                crate::utils::docker_hive_mind::SwarmStatus::Active => "active".to_string(),
                crate::utils::docker_hive_mind::SwarmStatus::Spawning => "spawning".to_string(),
                crate::utils::docker_hive_mind::SwarmStatus::Completed => "completed".to_string(),
                crate::utils::docker_hive_mind::SwarmStatus::Failed => "failed".to_string(),
                _ => "unknown".to_string(),
            },
            x: (base_id * 50.0) % 300.0 - 150.0, // Spread agents in a grid
            y: (base_id * 30.0) % 200.0 - 100.0,
            z: 0.0,
            cpu_usage: session.metrics.cpu_usage_percent as f32,
            memory_usage: session.metrics.memory_usage_mb as f32,
            health: if session.metrics.failed_tasks == 0 { 100.0 } else { 50.0 },
            workload: session.metrics.active_workers as f32,
            created_at: Some(session.created_at.to_rfc3339()),
            age: Some((chrono::Utc::now() - session.created_at).num_seconds() as u64),
        }
    }).collect()
}

fn convert_monitored_sessions_to_agents(sessions: Vec<crate::services::mcp_session_bridge::MonitoredSessionMetadata>) -> Vec<Agent> {
    sessions.into_iter().enumerate().map(|(idx, session)| {
        let base_id = idx as f32;
        let display_id = session.swarm_id.as_ref().unwrap_or(&session.uuid);

        Agent {
            id: display_id.clone(),
            name: format!("Session-{}", &session.uuid[..8]),
            agent_type: "worker".to_string(), // Default type, can be enhanced with MCP data
            status: session.status.clone(),
            x: (base_id * 50.0) % 300.0 - 150.0,
            y: (base_id * 30.0) % 200.0 - 100.0,
            z: 0.0,
            cpu_usage: 0.0, // TODO: Get from MCP telemetry
            memory_usage: 0.0, // TODO: Get from MCP telemetry
            health: if session.status == "active" { 100.0 } else { 50.0 },
            workload: session.agent_count as f32,
            created_at: Some(session.created_at.to_rfc3339()),
            age: Some((chrono::Utc::now() - session.created_at).num_seconds() as u64),
        }
    }).collect()
}

pub async fn fetch_hive_mind_agents(
    state: &AppState,
    hybrid_manager: Option<&Arc<HybridHealthManager>>,
) -> Result<Vec<Agent>, Box<dyn std::error::Error>> {
    // Try MCP session bridge first to get real session data
    let bridge = state.get_mcp_session_bridge();
    match bridge.list_monitored_sessions().await {
        sessions if !sessions.is_empty() => {
            info!("Retrieved {} sessions from MCP bridge", sessions.len());
            let agents = convert_monitored_sessions_to_agents(sessions);
            return Ok(agents);
        }
        _ => {
            info!("No sessions from MCP bridge, trying other methods");
        }
    }

    // Try hybrid approach if available
    if let Some(manager) = hybrid_manager {
        match manager.get_system_status().await {
            Ok(status) => {
                info!("Retrieved {} active sessions from DockerHiveMind", status.active_sessions.len());
                let agents = convert_sessions_to_agents(status.active_sessions);
                return Ok(agents);
            }
            Err(e) => {
                warn!("Failed to get sessions from DockerHiveMind, falling back to BotsClient: {}", e);
            }
        }
    }

    // Fallback to existing BotsClient
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
            metadata_id: agent.id.clone(),
            label: format!("{} ({})", agent.name, agent.agent_type),
            data: BinaryNodeData {
                node_id,
                x: agent.x,
                y: agent.y + vertical_offset,
                z: agent.z,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("agent_type".to_string(), agent.agent_type.clone());
                meta.insert("name".to_string(), agent.name.clone());
                meta.insert("status".to_string(), agent.status.clone());
                meta.insert("cpu_usage".to_string(), agent.cpu_usage.to_string());
                meta.insert("memory_usage".to_string(), agent.memory_usage.to_string());
                meta.insert("health".to_string(), agent.health.to_string());
                meta.insert("workload".to_string(), agent.workload.to_string());
                if let Some(age) = agent.age {
                    meta.insert("age".to_string(), age.to_string());
                }
                meta
            },
            file_size: 0,
            node_type: Some("agent".to_string()),
            size: Some(size),
            color: Some(color.to_string()),
            group: None,
            user_data: None,
            weight: Some(1.0),
        }
    }).collect()
}

pub async fn update_bots_graph(request: web::Json<BotsDataRequest>, _state: web::Data<AppState>) -> Result<impl Responder> {
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
        if let Ok(graph) = graph_data {
            let nodes = &graph.nodes;
            let edges = &graph.edges;
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



pub async fn initialize_hive_mind_swarm(
    request: web::Json<InitializeSwarmRequest>,
    state: web::Data<AppState>,
    hybrid_manager: web::Data<Arc<HybridHealthManager>>,
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
    let _bots_client = &state.bots_client;
    // For now, we'll return a simple success response since the MCP client is private
    // The actual swarm initialization happens through the periodic polling in BotsClient
    
    // Store the swarm parameters for reference
    let swarm_id = format!("swarm_{}", chrono::Utc::now().timestamp_millis());
    {
        let mut current_id = CURRENT_SWARM_ID.write().await;
        *current_id = Some(swarm_id.clone());
    }
    
    // The swarm will be initialized on the next polling cycle
    info!("Swarm initialization requested: {}", swarm_id);
    
    // Immediately fetch agents to get the initial state
    match fetch_hive_mind_agents(&state, Some(&hybrid_manager)).await {
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
                "message": "Swarm initialization requested, agents will appear shortly",
                "swarm_id": swarm_id,
                "error": e.to_string(),
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

pub async fn get_bots_agents(
    state: web::Data<AppState>,
    hybrid_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<impl Responder> {
    match fetch_hive_mind_agents(&state, Some(&hybrid_manager)).await {
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

// Structure for bot node data used by socket handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotsNodeData {
    pub id: u32,
    pub data: BotData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotData {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

pub async fn spawn_agent_hybrid(
    state: web::Data<AppState>,
    req: web::Json<SpawnAgentHybridRequest>,
) -> Result<impl Responder> {
    info!("Spawning agent hybrid: {:?}", req);

    // Parse priority and strategy from request
    let priority = match req.priority.as_deref() {
        Some("low") => Some(SwarmPriority::Low),
        Some("high") => Some(SwarmPriority::High),
        Some("critical") => Some(SwarmPriority::Critical),
        _ => Some(SwarmPriority::Medium),
    };

    let strategy = match req.strategy.as_deref() {
        Some("strategic") => Some(SwarmStrategy::Strategic),
        Some("tactical") => Some(SwarmStrategy::Tactical),
        Some("adaptive") => Some(SwarmStrategy::Adaptive),
        _ => Some(SwarmStrategy::HiveMind),
    };

    // Try Docker method first if requested
    if req.method == "docker" {
        let task = format!("Spawn {} agent for hive mind coordination", req.agent_type);

        // Use MCP session bridge to spawn and monitor
        match spawn_docker_agent_monitored(state.get_ref(), &task, priority, strategy).await {
            Ok((uuid, swarm_id)) => {
                info!("Successfully spawned {} agent via Docker - UUID: {}, Swarm ID: {:?}",
                      req.agent_type, uuid, swarm_id);
                return Ok(HttpResponse::Ok().json(SpawnAgentResponse {
                    success: true,
                    swarm_id: swarm_id.or(Some(uuid.clone())),
                    error: None,
                    method_used: Some("docker".to_string()),
                    message: Some(format!("Successfully spawned {} agent via Docker (UUID: {})", req.agent_type, uuid)),
                }));
            }
            Err(e) => {
                warn!("Docker spawn failed for {} agent: {}", req.agent_type, e);
                // Fall through to MCP fallback if Docker fails
            }
        }
    }

    // MCP fallback method
    match spawn_mcp_agent(state.get_ref(), &req.agent_type, &req.swarm_id).await {
        Ok(agent_id) => {
            info!("Successfully spawned {} agent via MCP with ID: {}", req.agent_type, agent_id);
            Ok(HttpResponse::Ok().json(SpawnAgentResponse {
                success: true,
                swarm_id: Some(agent_id),
                error: None,
                method_used: Some("mcp".to_string()),
                message: Some(format!("Successfully spawned {} agent via MCP", req.agent_type)),
            }))
        }
        Err(e) => {
            error!("Both Docker and MCP failed for {} agent: {}", req.agent_type, e);
            Ok(HttpResponse::InternalServerError().json(SpawnAgentResponse {
                success: false,
                swarm_id: None,
                error: Some(format!("Both Docker and MCP methods failed: {}", e)),
                method_used: None,
                message: None,
            }))
        }
    }
}

async fn spawn_docker_agent_monitored(
    state: &AppState,
    task: &str,
    priority: Option<SwarmPriority>,
    strategy: Option<SwarmStrategy>,
) -> Result<(String, Option<String>), Box<dyn std::error::Error + Send + Sync>> {
    use crate::utils::docker_hive_mind::SwarmConfig;

    let config = SwarmConfig {
        priority: priority.unwrap_or(SwarmPriority::Medium),
        strategy: strategy.unwrap_or(SwarmStrategy::HiveMind),
        auto_scale: true,
        monitor: true,
        max_workers: Some(8),
        consensus_type: None,
        memory_size_mb: None,
        encryption: false,
        verbose: false,
    };

    // Use MCP session bridge to spawn and monitor for swarm ID
    let bridge = state.get_mcp_session_bridge();
    let monitored = bridge.spawn_and_monitor(task, config).await?;

    Ok((monitored.uuid, monitored.swarm_id))
}

async fn spawn_mcp_agent(
    state: &AppState,
    agent_type: &str,
    swarm_id: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Use the bots_client from AppState for MCP spawning
    match timeout(Duration::from_secs(10), state.bots_client.spawn_agent_mcp(agent_type, swarm_id)).await {
        Ok(result) => result.map_err(|e| format!("MCP spawn failed: {}", e).into()),
        Err(_) => Err("MCP spawn timeout".into()),
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskResponse {
    pub success: bool,
    pub message: String,
    pub task_id: Option<String>,
    pub error: Option<String>,
}

/// Remove/stop a task by ID
pub async fn remove_task(
    path: web::Path<String>,
    hybrid_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<impl Responder> {
    let task_id = path.into_inner();
    info!("Removing task: {}", task_id);

    // Use DockerHiveMind directly from the hybrid manager
    match hybrid_manager.docker_hive_mind.stop_swarm(&task_id).await {
        Ok(()) => {
            info!("Successfully stopped task: {}", task_id);
            Ok(HttpResponse::Ok().json(TaskResponse {
                success: true,
                message: format!("Task {} stopped successfully", task_id),
                task_id: Some(task_id),
                error: None,
            }))
        }
        Err(e) => {
            error!("Failed to stop task {}: {}", task_id, e);
            Ok(HttpResponse::InternalServerError().json(TaskResponse {
                success: false,
                message: format!("Failed to stop task: {}", e),
                task_id: Some(task_id),
                error: Some(e.to_string()),
            }))
        }
    }
}

/// Pause a task by ID
pub async fn pause_task(
    path: web::Path<String>,
    hybrid_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<impl Responder> {
    let task_id = path.into_inner();
    info!("Pausing task: {}", task_id);

    match hybrid_manager.docker_hive_mind.pause_swarm(&task_id).await {
        Ok(()) => {
            info!("Successfully paused task: {}", task_id);
            Ok(HttpResponse::Ok().json(TaskResponse {
                success: true,
                message: format!("Task {} paused successfully", task_id),
                task_id: Some(task_id),
                error: None,
            }))
        }
        Err(e) => {
            error!("Failed to pause task {}: {}", task_id, e);
            Ok(HttpResponse::InternalServerError().json(TaskResponse {
                success: false,
                message: format!("Failed to pause task: {}", e),
                task_id: Some(task_id),
                error: Some(e.to_string()),
            }))
        }
    }
}

/// Resume a paused task by ID
pub async fn resume_task(
    path: web::Path<String>,
    hybrid_manager: web::Data<Arc<HybridHealthManager>>,
) -> Result<impl Responder> {
    let task_id = path.into_inner();
    info!("Resuming task: {}", task_id);

    match hybrid_manager.docker_hive_mind.resume_swarm(&task_id).await {
        Ok(()) => {
            info!("Successfully resumed task: {}", task_id);
            Ok(HttpResponse::Ok().json(TaskResponse {
                success: true,
                message: format!("Task {} resumed successfully", task_id),
                task_id: Some(task_id),
                error: None,
            }))
        }
        Err(e) => {
            error!("Failed to resume task {}: {}", task_id, e);
            Ok(HttpResponse::InternalServerError().json(TaskResponse {
                success: false,
                message: format!("Failed to resume task: {}", e),
                task_id: Some(task_id),
                error: Some(e.to_string()),
            }))
        }
    }
}

// Helper function for socket handler to get bot positions
pub async fn get_bots_positions(bots_client: &Arc<BotsClient>) -> Vec<BotsNodeData> {
    match bots_client.get_agents_snapshot().await {
        Ok(agents) => {
            agents.into_iter().enumerate().map(|(idx, agent)| {
                BotsNodeData {
                    id: (idx as u32) + 1000, // Convert to numeric ID
                    data: BotData {
                        x: agent.x,
                        y: agent.y,
                        z: agent.z,
                        vx: 0.0, // No velocity data from agents yet
                        vy: 0.0,
                        vz: 0.0,
                    },
                }
            }).collect()
        }
        Err(e) => {
            error!("Failed to get bots positions: {}", e);
            vec![]
        }
    }
}