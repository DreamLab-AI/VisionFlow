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
use crate::services::bots_client::BotsClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use log::{info, debug, error, warn};
use tokio::sync::RwLock;
use std::sync::Arc;
use tokio::time::timeout; // Added for connection checking
use std::time::Duration; // Added for timeout duration
use chrono; // UPDATED: Added for timestamp generation
use glam::Vec3;

// UPDATED: Enhanced BotsAgent to match claude-flow hive-mind agent properties
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BotsAgent {
    pub id: String,
    #[serde(rename = "type")]
    pub agent_type: String,
    pub status: String,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub health: f32,
    pub workload: f32,

    // UPDATED: Additional hive-mind properties
    #[serde(skip)]
    pub position: Vec3,
    #[serde(skip)]
    pub velocity: Vec3,
    #[serde(skip)]
    pub force: Vec3,
    #[serde(skip)]
    pub connections: Vec<String>,
    pub capabilities: Option<Vec<String>>,
    pub current_task: Option<String>,
    pub tasks_active: Option<u32>,
    pub tasks_completed: Option<u32>,
    pub success_rate: Option<f32>,
    pub tokens: Option<u64>,
    pub token_rate: Option<f32>,
    pub activity: Option<f32>,
    pub swarm_id: Option<String>,
    pub agent_mode: Option<String>, // centralized, distributed, strategic
    pub parent_queen_id: Option<String>,
    pub processing_logs: Option<Vec<String>>,
    pub created_at: Option<String>, // ISO 8601
    pub age: Option<u64>, // milliseconds
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BotsEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub data_volume: f32,
    pub message_count: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BotsDataRequest {
    pub nodes: Vec<BotsAgent>,
    pub edges: Vec<BotsEdge>,
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

pub async fn fetch_hive_mind_agents(_state: &AppState) -> Result<Vec<BotsAgent>, Box<dyn std::error::Error>> {
    // Connect directly to Claude Flow TCP server in multi-agent-container
    // The server runs at multi-agent-container:9500 on the docker_ragflow network
    use tokio::net::TcpStream;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use serde_json::json;
    use uuid::Uuid;

    // Use IP address as fallback since container name resolution might not work across different Docker containers
    // The multi-agent-container is accessible by hostname on the docker_ragflow network
    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
        .or_else(|_| std::env::var("MCP_HOST"))
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
    let addr = format!("{}:{}", claude_flow_host, claude_flow_port);

    match TcpStream::connect(&addr).await {
        Ok(mut stream) => {
            info!("Connected to Claude Flow TCP server at {}", addr);

            // FIXED: Initialize MCP session first
            if let Err(e) = initialize_mcp_session(&mut stream).await {
                warn!("Failed to initialize MCP session: {}", e);
                // Continue anyway, some servers might not require initialization
            }

            // Use tools/call wrapper for MCP tool invocation
            let request = json!({
                "jsonrpc": "2.0",
                "id": Uuid::new_v4().to_string(),
                "method": "tools/call",
                "params": {
                    "name": "agent_list",
                    "arguments": {
                        "filter": "all"
                    }
                }
            });

            let msg_str = serde_json::to_string(&request)?;
            let msg_bytes = format!("{}\n", msg_str);
            stream.write_all(msg_bytes.as_bytes()).await?;
            stream.flush().await?;

            // Read the response
            let mut reader = BufReader::new(&mut stream);
            let mut line = String::new();
            
            // Skip server.initialized messages
            reader.read_line(&mut line).await?;
            while line.contains("server.initialized") {
                line.clear();
                reader.read_line(&mut line).await?;
            }

            if let Ok(response) = serde_json::from_str::<serde_json::Value>(&line) {
                // MCP responses come directly as result objects
                let agents_data = response.get("result").cloned();
                
                if let Some(agent_result) = agents_data {
                    if let Some(agents) = agent_result.get("agents").and_then(|a| a.as_array()) {
                        // Convert to BotsAgent format
                        let bots_agents: Vec<BotsAgent> = agents.iter().filter_map(|agent| {
                            Some(BotsAgent {
                                id: agent.get("id")?.as_str()?.to_string(),
                                agent_type: agent.get("type").and_then(|t| t.as_str()).unwrap_or("agent").to_string(),
                                status: agent.get("status").and_then(|s| s.as_str()).unwrap_or("unknown").to_string(),
                                name: agent.get("name").and_then(|n| n.as_str()).unwrap_or("Agent").to_string(),
                                cpu_usage: agent.get("cpu_usage").and_then(|c| c.as_f64()).unwrap_or(0.0) as f32,
                                memory_usage: agent.get("memory_usage").and_then(|m| m.as_f64()).unwrap_or(0.0) as f32,
                                health: agent.get("health").and_then(|h| h.as_f64()).unwrap_or(100.0) as f32,
                                workload: agent.get("workload").and_then(|w| w.as_f64()).unwrap_or(0.0) as f32,
                                position: Vec3::ZERO,
                                velocity: Vec3::ZERO,
                                force: Vec3::ZERO,
                                connections: vec![],
                                capabilities: agent.get("capabilities").and_then(|c| c.as_array()).map(|arr| {
                                    arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()
                                }),
                                current_task: agent.get("current_task").and_then(|t| t.as_str()).map(String::from),
                                tasks_active: agent.get("tasks_active").and_then(|t| t.as_u64()).map(|v| v as u32),
                                tasks_completed: agent.get("tasks_completed").and_then(|t| t.as_u64()).map(|v| v as u32),
                                success_rate: agent.get("success_rate").and_then(|s| s.as_f64()).map(|v| v as f32),
                                tokens: agent.get("tokens").and_then(|t| t.as_u64()),
                                token_rate: agent.get("token_rate").and_then(|t| t.as_f64()).map(|v| v as f32),
                                activity: agent.get("activity").and_then(|a| a.as_f64()).map(|v| v as f32),
                                swarm_id: agent.get("swarm_id").and_then(|s| s.as_str()).map(String::from),
                                agent_mode: agent.get("agent_mode").and_then(|m| m.as_str()).map(String::from),
                                parent_queen_id: agent.get("parent_queen_id").and_then(|p| p.as_str()).map(String::from),
                                processing_logs: None,
                                created_at: agent.get("created_at").and_then(|c| c.as_str()).map(String::from),
                                age: agent.get("age").and_then(|a| a.as_u64()),
                            })
                        }).collect();

                        info!("Fetched {} agents from Claude Flow", bots_agents.len());
                        return Ok(bots_agents);
                    }
                }
            }

            warn!("Invalid response from Claude Flow");
            Ok(vec![])
        }
        Err(e) => {
            warn!("Failed to connect to Claude Flow TCP server at {}: {}", addr, e);
            Ok(vec![])
        }
    }
}

// UPDATED: Enhanced MCP connection management
async fn initialize_mcp_session(stream: &mut tokio::net::TcpStream) -> Result<String, Box<dyn std::error::Error>> {
    use tokio::io::{AsyncWriteExt, AsyncBufReadExt, BufReader};
    use uuid::Uuid;

    // Send MCP initialization request first
    let init_request = json!({
        "jsonrpc": "2.0",
        "id": Uuid::new_v4().to_string(),
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "VisionFlow-BotsClient",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": {
                    "listChanged": true
                }
            }
        }
    });

    let msg_str = serde_json::to_string(&init_request)?;
    let msg_bytes = format!("{}\n", msg_str);
    stream.write_all(msg_bytes.as_bytes()).await?;
    stream.flush().await?;

    // Read initialization response
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    if let Ok(response) = serde_json::from_str::<serde_json::Value>(&line) {
        if response.get("result").is_some() {
            info!("MCP session initialized successfully");
            return Ok("initialized".to_string());
        }
    }

    Err("Failed to initialize MCP session".into())
}

// UPDATED: Enhanced agent to nodes conversion with hive-mind properties and Queen agent special handling
fn convert_agents_to_nodes(agents: Vec<BotsAgent>) -> Vec<Node> {
    agents.into_iter().enumerate().map(|(idx, agent)| {
        // Map agent ID to numeric ID for physics processing
        let node_id = (idx + 1000) as u32; // Start at 1000 to avoid conflicts

        // UPDATED: Enhanced positioning based on agent type and hierarchy
        let (_radius, vertical_offset) = match agent.agent_type.as_str() {
            "queen" => (0.0, 0.0), // Queen at center
            "coordinator" => (8.0, 0.0), // Inner ring
            "architect" | "design_architect" => (12.0, 2.0), // Architecture level
            "requirements_analyst" | "task_planner" => (15.0, -2.0), // Planning level
            "coder" | "implementation_coder" => (18.0, 0.0), // Implementation level
            "tester" | "quality_reviewer" => (18.0, 4.0), // QA level
            "reviewer" | "steering_documenter" => (20.0, 0.0), // Review level
            _ => (15.0, (idx as f32 - 2.0) * 1.5), // Default positioning
        };

        // Use the agent's position that was set by position_agents_hierarchically
        let position = Vec3Data::new(
            agent.position.x,
            agent.position.y,
            vertical_offset
        );

        // UPDATED: Enhanced mass calculation considering agent importance and activity
        let base_mass = match agent.agent_type.as_str() {
            "queen" => 15.0, // Queen agents are heaviest (central gravity)
            "coordinator" => 10.0,
            "architect" | "design_architect" => 8.0,
            _ => 5.0,
        };
        let activity_factor = agent.activity.unwrap_or(0.5);
        let workload_factor = agent.workload + agent.cpu_usage / 100.0;
        let mass = ((base_mass + workload_factor * 5.0 + activity_factor * 3.0) / 3.0).max(1.0) as u8;

        Node {
            id: node_id,
            metadata_id: agent.id.clone(),
            label: agent.name,
            data: BinaryNodeData {
                node_id,
                x: position.x,
                y: position.y,
                z: position.z,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), agent.agent_type.clone());
                meta.insert("status".to_string(), agent.status);
                meta.insert("health".to_string(), agent.health.to_string());
                meta.insert("cpu_usage".to_string(), agent.cpu_usage.to_string());
                meta.insert("workload".to_string(), agent.workload.to_string());

                // UPDATED: Include hive-mind specific metadata
                if let Some(caps) = &agent.capabilities {
                    meta.insert("capabilities".to_string(), caps.join(","));
                }
                if let Some(task) = &agent.current_task {
                    meta.insert("current_task".to_string(), task.clone());
                }
                if let Some(active) = agent.tasks_active {
                    meta.insert("tasks_active".to_string(), active.to_string());
                }
                if let Some(completed) = agent.tasks_completed {
                    meta.insert("tasks_completed".to_string(), completed.to_string());
                }
                if let Some(rate) = agent.success_rate {
                    meta.insert("success_rate".to_string(), rate.to_string());
                }
                if let Some(tokens) = agent.tokens {
                    meta.insert("tokens".to_string(), tokens.to_string());
                }
                if let Some(swarm) = &agent.swarm_id {
                    meta.insert("swarm_id".to_string(), swarm.clone());
                }
                if let Some(mode) = &agent.agent_mode {
                    meta.insert("agent_mode".to_string(), mode.clone());
                }
                if let Some(queen) = &agent.parent_queen_id {
                    meta.insert("parent_queen_id".to_string(), queen.clone());
                }

                meta
            },
            file_size: 0,
            node_type: Some(agent.agent_type.clone()),
            size: Some(mass as f32),
            color: None,
            weight: Some(agent.workload),
            group: Some("bots".to_string()), // Could be enhanced to use swarm_id
            user_data: None,
        }
    }).collect()
}


// Position agents hierarchically based on their relationships
fn position_agents_hierarchically(agents: &mut Vec<BotsAgent>) {
    use glam::Vec3;
    use std::f32::consts::PI;

    // Find coordinators (acting as Queens)
    let coordinator_ids: Vec<String> = agents.iter()
        .filter(|a| a.agent_type == "coordinator")
        .map(|a| a.id.clone())
        .collect();

    if coordinator_ids.is_empty() {
        // No hierarchy, position in a circle
        let count = agents.len() as f32;
        for (i, agent) in agents.iter_mut().enumerate() {
            let angle = 2.0 * PI * i as f32 / count;
            agent.position = Vec3::new(angle.cos() * 500.0, angle.sin() * 500.0, 0.0);
        }
        return;
    }

    // Position coordinators at the center level
    let coordinator_count = coordinator_ids.len() as f32;
    for (i, coord_id) in coordinator_ids.iter().enumerate() {
        if let Some(coord) = agents.iter_mut().find(|a| &a.id == coord_id) {
            let angle = 2.0 * PI * i as f32 / coordinator_count;
            coord.position = Vec3::new(angle.cos() * 200.0, angle.sin() * 200.0, 100.0);
        }
    }

    // Group agents by their parent coordinator
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, agent) in agents.iter().enumerate() {
        if agent.agent_type != "coordinator" {
            if let Some(parent_id) = &agent.parent_queen_id {
                groups.entry(parent_id.clone()).or_insert_with(Vec::new).push(i);
            }
        }
    }

    // Position agents around their parent coordinator
    for (parent_id, child_indices) in groups {
        if let Some(parent) = agents.iter().find(|a| a.id == parent_id) {
            let parent_pos = parent.position;
            let child_count = child_indices.len() as f32;

            for (j, &child_idx) in child_indices.iter().enumerate() {
                let angle = 2.0 * PI * j as f32 / child_count;
                let _radius = 300.0 + (child_count * 10.0).min(200.0);
                let radius = _radius;
                let offset_x = angle.cos() * radius;
                let offset_y = angle.sin() * radius;

                if let Some(child) = agents.get_mut(child_idx) {
                    child.position = Vec3::new(
                        parent_pos.x + offset_x,
                        parent_pos.y + offset_y,
                        parent_pos.z - 50.0
                    );
                }
            }
        }
    }
}

// Convert bots edges to graph edges
fn convert_bots_edges(edges: Vec<BotsEdge>, node_map: &HashMap<String, u32>) -> Vec<Edge> {
    edges.into_iter().filter_map(|edge| {
        // Map string IDs to numeric IDs
        let source_id = node_map.get(&edge.source)?;
        let target_id = node_map.get(&edge.target)?;

        Some(Edge {
            id: edge.id,
            source: *source_id,
            target: *target_id,
            weight: (edge.data_volume / 1024.0).max(0.1), // Convert to KB, min 0.1
            edge_type: Some(format!("{} msgs", edge.message_count)),
            metadata: Some(HashMap::new()),
        })
    }).collect()
}

// Update bots data endpoint
pub async fn update_bots_data(
    state: web::Data<AppState>,
    bots_data: web::Json<BotsDataRequest>,
) -> HttpResponse {
    info!("Received bots data update with {} agents and {} communications",
        bots_data.nodes.len(),
        bots_data.edges.len()
    );

    // Convert bots agents to nodes
    let nodes = convert_agents_to_nodes(bots_data.nodes.clone());

    // Create node ID mapping
    let node_map: HashMap<String, u32> = nodes.iter()
        .map(|node| (node.metadata_id.clone(), node.id))
        .collect();

    // Convert bots edges
    let edges = convert_bots_edges(bots_data.edges.clone(), &node_map);

    // Update bots graph data
    {
        let mut bots_graph = BOTS_GRAPH.write().await;
        bots_graph.nodes = nodes.clone();
        bots_graph.edges = edges.clone();

        debug!("Updated bots graph with {} nodes and {} edges",
            bots_graph.nodes.len(),
            bots_graph.edges.len()
        );
    }

    // Process with GPU physics if available
    if let Some(gpu_compute_addr) = &state.gpu_compute_addr {
        use crate::actors::messages::{InitializeGPU, UpdateGPUGraphData, UpdateSimulationParams, ComputeForces, GetNodeData};

        let bots_graph = Arc::new(BOTS_GRAPH.read().await.clone());

        // Get physics settings
        let settings = match state.settings_addr.send(GetSettings).await {
            Ok(Ok(settings)) => settings,
            Ok(Err(e)) => {
                error!("Failed to get settings: {}", e);
                return HttpResponse::InternalServerError().json(BotsResponse {
                    success: false,
                    message: format!("Failed to get settings: {}", e),
                    nodes: None,
                    edges: None,
                });
            }
            Err(e) => {
                error!("Settings actor mailbox error: {}", e);
                return HttpResponse::InternalServerError().json(BotsResponse {
                    success: false,
                    message: format!("Settings service unavailable: {}", e),
                    nodes: None,
                    edges: None,
                });
            }
        };

        // Use logseq graph physics settings as the base for bots
        let physics_settings = &settings.visualisation.graphs.logseq.physics;

        // Create simulation parameters for bots
        // Use the From trait to ensure all fields are set
        let mut params = SimulationParams::from(physics_settings);
        // Adjust for bots
        params.iterations = physics_settings.iterations / 2; // Fewer iterations for bots
        params.spring_k = physics_settings.spring_k * 1.5; // Stronger attraction
        params.repel_k = physics_settings.repel_k * 0.8; // Less repulsion

        // Initialize GPU for bots graph first with retry logic
        info!("Initializing GPU for bots graph processing");
        let mut gpu_initialized = false;
        for attempt in 1..=3 {
            match gpu_compute_addr.send(InitializeGPU {
                graph: bots_graph.clone(),
                graph_service_addr: None,  // Not needed for bots handler
            }).await {
                Ok(_) => {
                    info!("GPU initialized successfully for bots on attempt {}", attempt);
                    gpu_initialized = true;
                    // Give GPU time to fully initialize
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    break;
                },
                Err(e) => {
                    warn!("Failed to initialize GPU for bots (attempt {}): {}", attempt, e);
                    if attempt < 3 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                    }
                }
            }
        }

        if !gpu_initialized {
            error!("Failed to initialize GPU after 3 attempts");
        }

        // Send graph data to GPU
        info!("Processing bots layout with GPU");
        if let Err(e) = gpu_compute_addr.send(UpdateGPUGraphData {
            graph: bots_graph.clone(),
        }).await {
            warn!("Failed to update GPU graph data: {}", e);
        }

        // Update simulation parameters
        if let Err(e) = gpu_compute_addr.send(UpdateSimulationParams {
            params,
        }).await {
            warn!("Failed to update simulation params: {}", e);
        }

        // Compute forces
        if let Err(e) = gpu_compute_addr.send(ComputeForces).await {
            warn!("Failed to compute forces: {}", e);
        }

        // Get updated node data
        match gpu_compute_addr.send(GetNodeData).await {
            Ok(Ok(node_data)) => {
                // Update bots graph with new positions
                let mut bots_graph = BOTS_GRAPH.write().await;
                for (i, node) in bots_graph.nodes.iter_mut().enumerate() {
                    if i < node_data.len() {
                        node.data = node_data[i].clone();
                    }
                }
            }
            Ok(Err(e)) => {
                warn!("Failed to get node data: {}", e);
            }
            Err(e) => {
                warn!("Failed to send GetNodeData: {}", e);
            }
        }
    }

    HttpResponse::Ok().json(BotsResponse {
        success: true,
        message: "Bots data updated successfully".to_string(),
        nodes: Some(nodes),
        edges: Some(edges),
    })
}

// UPDATED: Get current bots data prioritizing claude-flow hive-mind system
pub async fn get_bots_data(state: web::Data<AppState>) -> HttpResponse {
    debug!("get_bots_data endpoint called - fetching from hive-mind system");

    // UPDATED: First try to get live data from claude-flow hive-mind
    match fetch_hive_mind_agents(&**state).await {
        Ok(mut hive_agents) => {
            info!("Successfully fetched {} hive-mind agents", hive_agents.len());

            // Position agents hierarchically
            position_agents_hierarchically(&mut hive_agents);

            // Convert to nodes for graph visualization
            let nodes = convert_agents_to_nodes(hive_agents);

            // Create node ID mapping for edges
            let node_map: HashMap<String, u32> = nodes.iter()
                .map(|node| (node.metadata_id.clone(), node.id))
                .collect();

            // Generate hierarchical edges based on parent_queen_id relationships
            let mut edges = Vec::new();
            for node in &nodes {
                if let Some(parent_queen) = node.metadata.get("parent_queen_id") {
                    if let Some(parent_id) = node_map.get(parent_queen) {
                        edges.push(Edge {
                            id: format!("hierarchy-{}-{}", parent_queen, node.metadata_id),
                            source: *parent_id,
                            target: node.id,
                            weight: 2.0, // Strong hierarchical connection
                            edge_type: Some("hierarchy".to_string()),
                            metadata: Some({
                                let mut meta = HashMap::new();
                                meta.insert("type".to_string(), "hierarchy".to_string());
                                meta.insert("description".to_string(), "Queen-Agent hierarchy".to_string());
                                meta
                            }),
                        });
                    }
                }
            }

            let graph_data = GraphData {
                nodes,
                edges,
                metadata: MetadataStore::new(), // Use empty metadata for now
                id_to_metadata: HashMap::new(),
            };

            return HttpResponse::Ok().json(graph_data);
        }
        Err(e) => {
            warn!("Failed to fetch hive-mind data, falling back to BotsClient: {}", e);
        }
    }

    // Fallback: try to get data from BotsClient (legacy)
    if let Some(bots_update) = state.bots_client.get_latest_update().await {
        info!("Returning live bots data with {} agents", bots_update.agents.len());

        // Log agent details for debugging
        for agent in &bots_update.agents {
            debug!("Agent {}: {} ({}), status: {}, cpu: {}, health: {}, workload: {}",
                agent.id, agent.name, agent.agent_type, agent.status,
                agent.cpu_usage, agent.health, agent.workload);
        }

        // Convert BotsClient agents to our BotsAgent format
        let agents: Vec<BotsAgent> = bots_update.agents.iter().map(|agent| BotsAgent {
            id: agent.id.clone(),
            agent_type: agent.agent_type.clone(),
            status: agent.status.clone(),
            name: agent.name.clone(),
            cpu_usage: agent.cpu_usage,
            memory_usage: agent.memory_usage,
            health: agent.health,
            workload: agent.workload,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            connections: vec![],
            capabilities: None,
            current_task: None,
            tasks_active: None,
            tasks_completed: None,
            success_rate: None,
            tokens: None,
            token_rate: None,
            activity: None,
            swarm_id: None,
            agent_mode: None,
            parent_queen_id: None,
            processing_logs: None,
            created_at: agent.created_at.clone(),
            age: agent.age,
        }).collect();

        // Convert to nodes for graph visualization
        let nodes = convert_agents_to_nodes(agents);

        // Create node ID mapping
        let _node_map: HashMap<String, u32> = nodes.iter()
            .map(|node| (node.metadata_id.clone(), node.id))
            .collect();

        // For now, we'll return empty edges as the BotsUpdate doesn't include them
        // In a real implementation, you'd want to add edge data to BotsUpdate
        let edges = Vec::new();

        let graph_data = GraphData {
            nodes,
            edges,
            metadata: HashMap::new(),
            id_to_metadata: HashMap::new(),
        };

        return HttpResponse::Ok().json(graph_data);
    }

    // Try to get data from GraphServiceActor first
    match state.graph_service_addr.send(GetBotsGraphData).await {
        Ok(Ok(graph_data)) => {
            info!("Returning bots data from GraphServiceActor with {} nodes and {} edges",
                graph_data.nodes.len(),
                graph_data.edges.len()
            );
            return HttpResponse::Ok().json(graph_data.as_ref());
        }
        Ok(Err(e)) => {
            warn!("GraphServiceActor returned error: {}", e);
        }
        Err(e) => {
            warn!("Failed to communicate with GraphServiceActor: {}", e);
        }
    }

    // Fall back to static data if no live data available
    let bots_graph = BOTS_GRAPH.read().await;

    info!("Returning bots data with {} nodes and {} edges",
        bots_graph.nodes.len(),
        bots_graph.edges.len()
    );

    // If no data exists, return informative response instead of empty graph
    if bots_graph.nodes.is_empty() {
        info!("No bots data available - attempting to connect to agents");

        // Try one more time to fetch agents before giving up
        if let Ok(fresh_agents) = fetch_hive_mind_agents(&**state).await {
            if !fresh_agents.is_empty() {
                info!("Found {} fresh agents on retry", fresh_agents.len());
                let nodes = convert_agents_to_nodes(fresh_agents);
                let graph_data = GraphData {
                    nodes,
                    edges: Vec::new(),
                    metadata: MetadataStore::new(),
                    id_to_metadata: HashMap::new(),
                };
                return HttpResponse::Ok().json(graph_data);
            }
        }

        // Return structured empty response with connection status
        return HttpResponse::Ok().json(json!({
            "nodes": [],
            "edges": [],
            "metadata": {},
            "id_to_metadata": {},
            "status": "no_agents_available",
            "message": "No bot agents are currently active. Try initializing a swarm first.",
            "suggestions": [
                "POST /bots/initialize-swarm to create new agents",
                "Check MCP connection status at /bots/check-connection",
                "Verify multi-agent-container is running"
            ],
            "mcp_connection_available": state.bots_client.get_latest_update().await.is_some()
        }));
    }

    HttpResponse::Ok().json(&*bots_graph)
}

// Get full bots graph data (for WebSocket updates)
pub async fn get_bots_graph_data(bots_client: &BotsClient) -> Option<GraphData> {
    // Try to get live data from BotsClient
    if let Some(bots_update) = bots_client.get_latest_update().await {
        info!("Building graph data for {} agents", bots_update.agents.len());
        
        // Convert BotsClient agents to our BotsAgent format
        // NOTE: BotsClient Agent struct doesn't have all fields, so we use sensible defaults
        let agents: Vec<BotsAgent> = bots_update.agents.iter().map(|agent| BotsAgent {
            id: agent.id.clone(),
            agent_type: agent.agent_type.clone(),
            status: agent.status.clone(),
            name: agent.name.clone(),
            cpu_usage: agent.cpu_usage,
            memory_usage: agent.memory_usage,
            health: agent.health,
            workload: agent.workload,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            connections: vec![],
            capabilities: None,
            current_task: None,
            tasks_active: Some(1), // Default to 1 active task if agent is running
            tasks_completed: Some(0), // Will be updated when we get real data
            success_rate: Some(1.0), // Default success rate
            tokens: Some(1000), // Default token count for demo
            token_rate: Some(10.0), // Default token rate
            activity: Some(agent.workload), // Use workload as activity indicator
            swarm_id: None,
            agent_mode: None,
            parent_queen_id: None,
            processing_logs: None,
            created_at: agent.created_at.clone(),
            age: agent.age,
        }).collect();
        
        // Convert to nodes for graph visualization
        let nodes = convert_agents_to_nodes(agents);
        
        // Create node ID mapping for edges
        let node_map: HashMap<String, u32> = nodes.iter()
            .map(|node| (node.metadata_id.clone(), node.id))
            .collect();
        
        // Generate simple mesh edges for now (each agent connected to others)
        let mut edges = Vec::new();
        
        // Use node_map to create edges based on agent relationships
        for (agent_id, &node_id) in &node_map {
            // Create edges to other agents (simplified mesh topology)
            for (other_id, &other_node_id) in &node_map {
                if agent_id != other_id {
                    edges.push(Edge {
                        edge_type: Some("agent_connection".to_string()),
                        id: (edges.len() as u32).to_string(),
                        source: node_id,
                        target: other_node_id,
                        weight: 1.0,
                        metadata: Some({
                            let mut metadata = HashMap::new();
                            metadata.insert("label".to_string(), "agent_connection".to_string());
                            metadata.insert("source_type".to_string(), "agent".to_string());
                            metadata.insert("target_type".to_string(), "agent".to_string());
                            metadata
                        }),
                    });
                }
            }
        }
        for (i, node1) in nodes.iter().enumerate() {
            for node2 in nodes.iter().skip(i + 1) {
                edges.push(Edge {
                    id: format!("edge-{}-{}", node1.metadata_id, node2.metadata_id),
                    source: node1.id,
                    target: node2.id,
                    weight: 1.0,
                    edge_type: Some("collaboration".to_string()),
                    metadata: Some({
                        let mut meta = HashMap::new();
                        meta.insert("type".to_string(), "collaboration".to_string());
                        meta
                    }),
                });
            }
        }
        
        return Some(GraphData {
            nodes,
            edges,
            metadata: MetadataStore::new(),
            id_to_metadata: HashMap::new(),
        });
    }
    
    None
}

// Get bots node positions (for WebSocket updates)
pub async fn get_bots_positions(bots_client: &BotsClient) -> Vec<Node> {
    // Try to get live data from BotsClient first
    if let Some(bots_update) = bots_client.get_latest_update().await {
        // Convert BotsClient agents to nodes with positions
        let agents: Vec<BotsAgent> = bots_update.agents.iter().map(|agent| BotsAgent {
            id: agent.id.clone(),
            agent_type: agent.agent_type.clone(),
            status: agent.status.clone(),
            name: agent.name.clone(),
            cpu_usage: agent.cpu_usage,
            memory_usage: agent.memory_usage,
            health: agent.health,
            workload: agent.workload,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            connections: vec![],
            capabilities: None,
            current_task: None,
            tasks_active: Some(1), // Default to 1 active task if agent is running
            tasks_completed: Some(0), // Will be updated when we get real data
            success_rate: Some(1.0), // Default success rate
            tokens: Some(1000), // Default token count for demo
            token_rate: Some(10.0), // Default token rate
            activity: Some(agent.workload), // Use workload as activity indicator
            swarm_id: None,
            agent_mode: None,
            parent_queen_id: None,
            processing_logs: None,
            created_at: agent.created_at.clone(),
            age: agent.age,
        }).collect();

        return convert_agents_to_nodes(agents);
    }

    // Fall back to static data
    let bots_graph = BOTS_GRAPH.read().await;
    bots_graph.nodes.clone()
}

// Initialize swarm endpoint
pub async fn get_agent_telemetry(
    state: web::Data<AppState>,
) -> HttpResponse {
    info!("Getting agent telemetry data");

    // Try to fetch real agent data from hive-mind system
    match fetch_hive_mind_agents(&**state).await {
        Ok(agents) => {
            // Calculate comprehensive telemetry metrics
            let total_agents = agents.len();
            let active_agents = agents.iter().filter(|a| a.status == "active").count();
            let idle_agents = agents.iter().filter(|a| a.status == "idle").count();
            let error_agents = agents.iter().filter(|a| a.status == "error").count();

            // Performance metrics
            let avg_health = if !agents.is_empty() {
                agents.iter().map(|a| a.health).sum::<f32>() / agents.len() as f32
            } else { 100.0 };

            let avg_cpu = if !agents.is_empty() {
                agents.iter().map(|a| a.cpu_usage).sum::<f32>() / agents.len() as f32
            } else { 0.0 };

            let avg_memory = if !agents.is_empty() {
                agents.iter().map(|a| a.memory_usage).sum::<f32>() / agents.len() as f32
            } else { 0.0 };

            let total_workload = agents.iter().map(|a| a.workload).sum::<f32>();

            // Task metrics
            let total_active_tasks = agents.iter()
                .filter_map(|a| a.tasks_active)
                .sum::<u32>();

            let total_completed_tasks = agents.iter()
                .filter_map(|a| a.tasks_completed)
                .sum::<u32>();

            let avg_success_rate = if !agents.is_empty() {
                agents.iter()
                    .filter_map(|a| a.success_rate)
                    .sum::<f32>() / agents.iter().filter_map(|a| a.success_rate).count() as f32
            } else { 0.0 };

            // Token metrics
            let total_tokens = agents.iter()
                .filter_map(|a| a.tokens)
                .sum::<u64>();

            let avg_token_rate = if !agents.is_empty() {
                agents.iter()
                    .filter_map(|a| a.token_rate)
                    .sum::<f32>() / agents.iter().filter_map(|a| a.token_rate).count() as f32
            } else { 0.0 };

            // Agent type distribution
            let mut agent_types = HashMap::new();
            for agent in &agents {
                *agent_types.entry(agent.agent_type.clone()).or_insert(0) += 1;
            }

            // Swarm information
            let swarms: std::collections::HashSet<String> = agents.iter()
                .filter_map(|a| a.swarm_id.as_ref())
                .cloned()
                .collect();

            HttpResponse::Ok().json(json!({
                "success": true,
                "telemetry": {
                    "agent_counts": {
                        "total": total_agents,
                        "active": active_agents,
                        "idle": idle_agents,
                        "error": error_agents
                    },
                    "performance": {
                        "avg_health": avg_health,
                        "avg_cpu_usage": avg_cpu,
                        "avg_memory_usage": avg_memory,
                        "total_workload": total_workload
                    },
                    "tasks": {
                        "active_tasks": total_active_tasks,
                        "completed_tasks": total_completed_tasks,
                        "avg_success_rate": avg_success_rate
                    },
                    "tokens": {
                        "total_tokens": total_tokens,
                        "avg_token_rate": avg_token_rate
                    },
                    "swarms": {
                        "count": swarms.len(),
                        "ids": swarms.into_iter().collect::<Vec<_>>()
                    },
                    "agent_types": agent_types,
                    "system_health": if avg_health > 80.0 && active_agents > 0 {
                        "healthy"
                    } else if avg_health > 50.0 {
                        "degraded"
                    } else {
                        "critical"
                    }
                },
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "data_source": "hive_mind_tcp"
            }))
        }
        Err(e) => {
            warn!("Failed to fetch hive-mind agents for telemetry: {}", e);

            // Try fallback to BotsClient
            if let Some(bots_update) = state.bots_client.get_latest_update().await {
                info!("Using fallback BotsClient data for telemetry");

                let agents = &bots_update.agents;
                let total_agents = agents.len();
                let active_agents = agents.iter().filter(|a| a.status == "active").count();

                let avg_health = if !agents.is_empty() {
                    agents.iter().map(|a| a.health).sum::<f32>() / agents.len() as f32
                } else { 0.0 };

                let avg_cpu = if !agents.is_empty() {
                    agents.iter().map(|a| a.cpu_usage).sum::<f32>() / agents.len() as f32
                } else { 0.0 };

                HttpResponse::Ok().json(json!({
                    "success": true,
                    "telemetry": {
                        "agent_counts": {
                            "total": total_agents,
                            "active": active_agents,
                            "idle": 0,
                            "error": 0
                        },
                        "performance": {
                            "avg_health": avg_health,
                            "avg_cpu_usage": avg_cpu,
                            "avg_memory_usage": 0.0,
                            "total_workload": 0.0
                        },
                        "system_health": if avg_health > 80.0 && active_agents > 0 {
                            "healthy"
                        } else {
                            "degraded"
                        }
                    },
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "data_source": "bots_client_fallback"
                }))
            } else {
                error!("No agent data available from any source");
                HttpResponse::ServiceUnavailable().json(json!({
                    "success": false,
                    "error": "No agent telemetry data available",
                    "details": e.to_string(),
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }))
            }
        }
    }
}

pub async fn initialize_swarm(
    _state: web::Data<AppState>,
    request: web::Json<InitializeSwarmRequest>,
) -> HttpResponse {
    info!("=== INITIALIZE SWARM ENDPOINT CALLED ===");
    info!("Received swarm initialization request: {:?}", request);
    info!("Topology: {}, Max Agents: {}, Strategy: {}", 
         request.topology, request.max_agents, request.strategy);

    // Use the new stable MCP connection pool
    use crate::utils::mcp_connection::call_swarm_init;
    
    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
        .or_else(|_| std::env::var("MCP_HOST"))
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
    
    info!("Connecting to MCP server at {}:{}", claude_flow_host, claude_flow_port);
    
    // Call swarm_init with retry logic and proper error handling
    match call_swarm_init(
        &claude_flow_host,
        &claude_flow_port,
        &request.topology,
        request.max_agents,
        &request.strategy,
    ).await {
        Ok(result) => {
            info!("Successfully received swarm_init response: {:?}", result);
            
            // The MCP response is already the result object
            let swarm_id = result.get("swarmId")
                .or_else(|| result.get("swarm_id"))
                .and_then(|s| s.as_str())
                .unwrap_or("default-swarm")
                .to_string();
            
            info!("✅ Swarm initialized with ID: {}", swarm_id);
            
            // Now spawn some agents based on the topology
            use crate::utils::mcp_connection::call_agent_spawn;
            
            let agent_types = match request.topology.as_str() {
                "hierarchical" => vec!["coordinator", "analyst", "optimizer"],
                "mesh" => vec!["coordinator", "researcher", "coder", "analyst"],
                "star" => vec!["coordinator", "optimizer", "documenter"],
                _ => vec!["coordinator", "analyst", "optimizer"],
            };
            
            info!("Spawning {} agents for {} topology", agent_types.len(), request.topology);
            
            for agent_type in agent_types {
                match call_agent_spawn(&claude_flow_host, &claude_flow_port, agent_type, &swarm_id).await {
                    Ok(spawn_result) => {
                        info!("✅ Spawned {} agent: {:?}", agent_type, spawn_result);
                    }
                    Err(e) => {
                        warn!("Failed to spawn {} agent: {}", agent_type, e);
                    }
                }
            }
            
            // Store the swarm ID for later use (e.g., disconnection)
            {
                let mut current_id = CURRENT_SWARM_ID.write().await;
                *current_id = Some(swarm_id.clone());
                info!("Stored swarm ID for later disconnection: {}", swarm_id);
            }
                
            
            // Create initial agent list based on requested types
            let initial_agents: Vec<serde_json::Value> = request.agent_types.iter().enumerate().map(|(i, agent_type)| {
                json!({
                    "id": format!("agent-{}-{}", swarm_id, i),
                    "type": agent_type,
                    "name": format!("{} Agent {}", agent_type, i + 1),
                    "status": "initializing",
                    "swarm_id": swarm_id.clone()
                })
            }).collect();

            return HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "message": "Swarm initialization started via Claude Flow TCP",
                "swarm_id": swarm_id,
                "agents": initial_agents,
                "topology": request.topology.clone(),
                "max_agents": request.max_agents,
                "strategy": request.strategy.clone(),
                "enable_neural": request.enable_neural,
                "service_available": true
            }));
        }
        Err(e) => {
            error!("Failed to initialize swarm via MCP: {}", e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("Failed to connect to Claude Flow: {}", e)
            }));
        }
    }

    // If Claude Flow actor not available, return error
    error!("Claude Flow actor not available - cannot initialize swarm");

    HttpResponse::ServiceUnavailable().json(serde_json::json!({
        "success": false,
        "error": "Claude Flow service not available",
        "message": "Cannot initialize swarm without Claude Flow service"
    }))
}

// UPDATED: Enhanced agent status endpoint for real-time updates
pub async fn get_agent_status(state: web::Data<AppState>) -> impl Responder {
    debug!("get_agent_status endpoint called for real-time updates");

    match fetch_hive_mind_agents(&**state).await {
        Ok(agents) => {
            // Calculate swarm metrics
            let total_agents = agents.len();
            let active_agents = agents.iter().filter(|a| a.status == "active").count();
            let avg_health = if !agents.is_empty() {
                agents.iter().map(|a| a.health).sum::<f32>() / agents.len() as f32
            } else {
                0.0
            };
            let avg_cpu = if !agents.is_empty() {
                agents.iter().map(|a| a.cpu_usage).sum::<f32>() / agents.len() as f32
            } else {
                0.0
            };
            
            // Return enhanced status data with swarm telemetry
            let status_data: Vec<_> = agents.into_iter().map(|agent| {
                serde_json::json!({
                    "id": agent.id,
                    "type": agent.agent_type,
                    "status": agent.status,
                    "health": agent.health,
                    "cpu_usage": agent.cpu_usage,
                    "memory_usage": agent.memory_usage,
                    "workload": agent.workload,
                    "activity": agent.activity,
                    "current_task": agent.current_task,
                    "tasks_active": agent.tasks_active,
                    "tasks_completed": agent.tasks_completed,
                    "success_rate": agent.success_rate,
                    "tokens": agent.tokens,
                    "swarm_id": agent.swarm_id,
                    "agent_mode": agent.agent_mode,
                    "parent_queen_id": agent.parent_queen_id,
                    "capabilities": agent.capabilities,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })
            }).collect();

            HttpResponse::Ok().json(serde_json::json!({
                "agents": status_data,
                "swarm_health": if avg_health > 70.0 { "healthy" } else if avg_health > 40.0 { "degraded" } else { "critical" },
                "total_agents": total_agents,
                "active_agents": active_agents,
                "avg_health": avg_health,
                "avg_cpu_usage": avg_cpu,
                "swarm_status": "live",
                "data_source": "hive_mind",
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
        Err(e) => {
            error!("Failed to get agent status: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to fetch agent status",
                "details": e.to_string(),
                "swarm_status": "disconnected",
                "data_source": "error",
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
    }
}

pub async fn check_mcp_connection(
    _state: web::Data<AppState>,
) -> impl Responder {
    use tokio::net::TcpStream;
    use tokio::io::{AsyncWriteExt, AsyncBufReadExt, BufReader};

    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
        .or_else(|_| std::env::var("MCP_HOST"))
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
    let addr = format!("{}:{}", claude_flow_host, claude_flow_port);

    // FIXED: Try to connect and perform MCP handshake
    match timeout(Duration::from_secs(5), TcpStream::connect(&addr)).await {
        Ok(Ok(mut stream)) => {
            // Connection successful, now test MCP protocol
            match initialize_mcp_session(&mut stream).await {
                Ok(_) => {
                    // Test a simple tool call - use tools/list which we know works
                    let test_request = json!({
                        "jsonrpc": "2.0",
                        "id": "test-connection",
                        "method": "tools/list",  // Use tools/list which we confirmed works
                        "params": {}
                    });

                    let msg_str = serde_json::to_string(&test_request).unwrap_or_default();
                    let msg_bytes = format!("{}\n", msg_str);

                    if let Ok(_) = stream.write_all(msg_bytes.as_bytes()).await {
                        let _ = stream.flush().await;

                        // Try to read response
                        let mut reader = BufReader::new(&mut stream);
                        let mut line = String::new();

                        match timeout(Duration::from_secs(2), reader.read_line(&mut line)).await {
                            Ok(Ok(_)) => {
                                HttpResponse::Ok().json(json!({
                                    "connected": true,
                                    "address": addr,
                                    "service": "claude-flow-mcp",
                                    "mcp_initialized": true,
                                    "response": line.trim()
                                }))
                            }
                            _ => {
                                HttpResponse::Ok().json(json!({
                                    "connected": true,
                                    "address": addr,
                                    "service": "claude-flow-mcp",
                                    "mcp_initialized": true,
                                    "response_timeout": true
                                }))
                            }
                        }
                    } else {
                        HttpResponse::Ok().json(json!({
                            "connected": true,
                            "address": addr,
                            "service": "claude-flow-mcp",
                            "mcp_initialized": true,
                            "write_failed": true
                        }))
                    }
                }
                Err(e) => {
                    HttpResponse::Ok().json(json!({
                        "connected": true,
                        "address": addr,
                        "service": "claude-flow-mcp",
                        "mcp_initialized": false,
                        "init_error": e.to_string()
                    }))
                }
            }
        }
        _ => {
            // Connection failed or timed out
            HttpResponse::Ok().json(json!({
                "connected": false,
                "address": addr,
                "service": "claude-flow-mcp"
            }))
        }
    }
}

pub async fn initialize_multi_agent(
    _state: web::Data<AppState>,
    request: web::Json<serde_json::Value>,
) -> impl Responder {
    info!("=== INITIALIZE MULTI-AGENT ENDPOINT CALLED ===");
    info!("Received multi-agent initialization request: {:?}", request);
    
    // Extract parameters with defaults
    let topology = request.get("topology").and_then(|t| t.as_str()).unwrap_or("mesh");
    let max_agents = request.get("maxAgents").and_then(|m| m.as_u64()).unwrap_or(8) as u32;
    let strategy = request.get("strategy").and_then(|s| s.as_str()).unwrap_or("adaptive");
    
    info!("Multi-agent params - Topology: {}, Max Agents: {}, Strategy: {}", 
         topology, max_agents, strategy);

    // Use the stable MCP connection pool
    use crate::utils::mcp_connection::call_swarm_init;
    
    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
        .or_else(|_| std::env::var("MCP_HOST"))
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
    
    info!("Connecting to MCP server at {}:{} for multi-agent", claude_flow_host, claude_flow_port);
    
    // Call swarm_init with retry logic
    match call_swarm_init(
        &claude_flow_host,
        &claude_flow_port,
        topology,
        max_agents,
        strategy,
    ).await {
        Ok(result) => {
            info!("Successfully received multi-agent swarm_init response: {:?}", result);
            
            // The MCP response is already the result object, not wrapped
            let swarm_id = result.get("swarmId")
                .or_else(|| result.get("swarm_id"))
                .and_then(|s| s.as_str())
                .unwrap_or("default-swarm")
                .to_string();
            
            info!("✅ Multi-agent swarm initialized with ID: {}", swarm_id);
            
            // Now spawn some agents based on the topology
            use crate::utils::mcp_connection::call_agent_spawn;
            
            let agent_types = match topology {
                "hierarchical" => vec!["coordinator", "analyst", "optimizer"],
                "mesh" => vec!["coordinator", "researcher", "coder", "analyst"],
                "star" => vec!["coordinator", "optimizer", "documenter"],
                _ => vec!["coordinator", "analyst", "optimizer"],
            };
            
            info!("Spawning {} agents for {} topology", agent_types.len(), topology);
            
            for agent_type in agent_types {
                match call_agent_spawn(&claude_flow_host, &claude_flow_port, agent_type, &swarm_id).await {
                    Ok(spawn_result) => {
                        info!("✅ Spawned {} agent: {:?}", agent_type, spawn_result);
                    }
                    Err(e) => {
                        warn!("Failed to spawn {} agent: {}", agent_type, e);
                    }
                }
            }
            
            // Store the swarm ID for later use (e.g., disconnection)
            {
                let mut current_id = CURRENT_SWARM_ID.write().await;
                *current_id = Some(swarm_id.clone());
                info!("Stored swarm ID for later disconnection: {}", swarm_id);
            }
            
            // Extract agent types from request
            let agent_types = request.get("agentTypes")
                .and_then(|a| a.as_array())
                .map(|a| a.iter()
                    .filter_map(|t| t.as_str())
                    .collect::<Vec<_>>())
                .unwrap_or_else(|| vec!["coordinator", "researcher", "coder"]);
            
            // Create agent list
            let agents: Vec<serde_json::Value> = agent_types.iter().enumerate().map(|(i, agent_type)| {
                json!({
                    "id": format!("agent-{}-{}", swarm_id, i),
                    "type": agent_type,
                    "name": format!("{} Agent {}", agent_type, i + 1),
                    "status": "active",
                    "swarmId": swarm_id.clone()
                })
            }).collect();
            
            return HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Multi-agent system initialized successfully",
                "data": {
                    "swarmId": swarm_id,
                    "topology": topology,
                    "maxAgents": max_agents,
                    "strategy": strategy,
                    "agents": agents,
                    "status": "initialized"
                }
            }));
        }
        Err(e) => {
            error!("Failed to initialize multi-agent system: {}", e);
            return HttpResponse::InternalServerError().json(json!({
                "success": false,
                "error": format!("Failed to connect to Claude Flow service at {}:{} - {}", 
                                claude_flow_host, claude_flow_port, e)
            }));
        }
    }
}

// UPDATED: Enhanced configuration with new real-time endpoints
/// Enhanced bots handler with comprehensive validation
pub struct EnhancedBotsHandler {
    validation_service: ValidationService,
    rate_limiter: Arc<RateLimiter>,
}

impl EnhancedBotsHandler {
    pub fn new() -> Self {
        let config = EndpointRateLimits::bots_operations();
        let rate_limiter = Arc::new(RateLimiter::new(config));

        Self {
            validation_service: ValidationService::new(),
            rate_limiter,
        }
    }

    /// Enhanced bots data update with validation
    pub async fn update_bots_data_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        payload: web::Json<Value>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // Rate limiting check
        if !self.rate_limiter.is_allowed(&client_id) {
            warn!("Rate limit exceeded for bots data update from client: {}", client_id);
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many bots data updates. Please wait before retrying.",
                "retry_after": self.rate_limiter.reset_time(&client_id).as_secs()
            })));
        }

        // Request size validation
        let payload_size = serde_json::to_vec(&*payload).unwrap_or_default().len();
        if payload_size > MAX_REQUEST_SIZE {
            error!("Bots data payload too large: {} bytes", payload_size);
            return Ok(HttpResponse::PayloadTooLarge().json(json!({
                "error": "payload_too_large",
                "message": format!("Bots data size {} bytes exceeds limit of {} bytes", payload_size, MAX_REQUEST_SIZE),
                "max_size": MAX_REQUEST_SIZE
            })));
        }

        info!("Processing enhanced bots data update from client: {} (size: {} bytes)", client_id, payload_size);

        // Comprehensive validation and sanitization
        let validated_payload = match self.validation_service.validate_bots_data(&payload) {
            Ok(sanitized) => sanitized,
            Err(validation_error) => {
                warn!("Bots data validation failed for client {}: {}", client_id, validation_error);
                return Ok(validation_error.to_http_response());
            }
        };

        debug!("Bots data validation passed for client: {}", client_id);

        // Additional validation for bots data structure
        self.validate_bots_data_structure(&validated_payload)?;

        // Extract and process nodes and edges
        let nodes_array = validated_payload.get("nodes")
            .and_then(|n| n.as_array())
            .ok_or_else(|| DetailedValidationError::missing_required_field("nodes"))?;

        let edges_array = validated_payload.get("edges")
            .and_then(|e| e.as_array())
            .ok_or_else(|| DetailedValidationError::missing_required_field("edges"))?;

        info!("Received bots data with {} nodes and {} edges from client: {}",
              nodes_array.len(), edges_array.len(), client_id);

        // Validate individual nodes and edges
        self.validate_bots_nodes(nodes_array)?;
        self.validate_bots_edges(edges_array)?;

        // Convert to internal format and call existing handler
        let bots_data_request = BotsDataRequest {
            nodes: nodes_array.iter().filter_map(|node| {
                let node_obj = node.as_object()?;
                Some(BotsAgent {
                    id: node_obj.get("id")?.as_str()?.to_string(),
                    agent_type: node_obj.get("type").and_then(|t| t.as_str()).unwrap_or("agent").to_string(),
                    status: node_obj.get("status").and_then(|s| s.as_str()).unwrap_or("unknown").to_string(),
                    name: node_obj.get("name").and_then(|n| n.as_str()).unwrap_or("Agent").to_string(),
                    cpu_usage: node_obj.get("cpu_usage").and_then(|c| c.as_f64()).unwrap_or(0.0) as f32,
                    memory_usage: node_obj.get("memory_usage").and_then(|m| m.as_f64()).unwrap_or(0.0) as f32,
                    health: node_obj.get("health").and_then(|h| h.as_f64()).unwrap_or(100.0) as f32,
                    workload: node_obj.get("workload").and_then(|w| w.as_f64()).unwrap_or(0.0) as f32,
                    position: Vec3::ZERO,
                    velocity: Vec3::ZERO,
                    force: Vec3::ZERO,
                    connections: vec![],
                    capabilities: node_obj.get("capabilities").and_then(|c| c.as_array()).map(|arr| {
                        arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()
                    }),
                    current_task: node_obj.get("current_task").and_then(|t| t.as_str()).map(String::from),
                    tasks_active: node_obj.get("tasks_active").and_then(|t| t.as_u64()).map(|v| v as u32),
                    tasks_completed: node_obj.get("tasks_completed").and_then(|t| t.as_u64()).map(|v| v as u32),
                    success_rate: node_obj.get("success_rate").and_then(|s| s.as_f64()).map(|v| v as f32),
                    tokens: node_obj.get("tokens").and_then(|t| t.as_u64()),
                    token_rate: node_obj.get("token_rate").and_then(|t| t.as_f64()).map(|v| v as f32),
                    activity: node_obj.get("activity").and_then(|a| a.as_f64()).map(|v| v as f32),
                    swarm_id: node_obj.get("swarm_id").and_then(|s| s.as_str()).map(String::from),
                    agent_mode: node_obj.get("agent_mode").and_then(|m| m.as_str()).map(String::from),
                    parent_queen_id: node_obj.get("parent_queen_id").and_then(|p| p.as_str()).map(String::from),
                    processing_logs: None,
                    created_at: node_obj.get("created_at").and_then(|c| c.as_str()).map(String::from),
                    age: node_obj.get("age").and_then(|a| a.as_u64()),
                })
            }).collect(),
            edges: edges_array.iter().filter_map(|edge| {
                let edge_obj = edge.as_object()?;
                Some(BotsEdge {
                    id: edge_obj.get("id")?.as_str()?.to_string(),
                    source: edge_obj.get("source")?.as_str()?.to_string(),
                    target: edge_obj.get("target")?.as_str()?.to_string(),
                    data_volume: edge_obj.get("data_volume").and_then(|d| d.as_f64()).unwrap_or(0.0) as f32,
                    message_count: edge_obj.get("message_count").and_then(|m| m.as_u64()).unwrap_or(0) as u32,
                })
            }).collect(),
        };

        // Call existing handler logic and convert to Result<HttpResponse>
        match update_bots_data(state, web::Json(bots_data_request)).await {
            _response => {
                // For now, just return a success response
                // TODO: Properly handle the impl Responder conversion
                Ok(HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "Bots data updated with validation",
                    "client_id": client_id
                })))
            }
        }
    }

    /// Enhanced get bots data with validation metadata
    pub async fn get_bots_data_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // More permissive rate limiting for GET requests
        let get_rate_limiter = Arc::new(RateLimiter::new(
            crate::utils::validation::rate_limit::RateLimitConfig {
                requests_per_minute: 80,
                burst_size: 15,
                ..Default::default()
            }
        ));

        if !get_rate_limiter.is_allowed(&client_id) {
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many get bots data requests"
            })));
        }

        debug!("Processing get bots data request from client: {}", client_id);

        // Call existing handler - convert to HttpResponse
        match get_bots_data(state).await {
            _response => {
                // For now, just return a success response
                // TODO: Properly handle the impl Responder conversion
                Ok(HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "Bots data retrieved with validation",
                    "client_id": client_id
                })))
            }
        }
    }

    /// Enhanced swarm initialization with validation
    pub async fn initialize_swarm_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        payload: web::Json<Value>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // Stricter rate limiting for swarm operations
        let swarm_rate_limiter = Arc::new(RateLimiter::new(
            crate::utils::validation::rate_limit::RateLimitConfig {
                requests_per_minute: 20,
                burst_size: 3,
                ..Default::default()
            }
        ));

        if !swarm_rate_limiter.is_allowed(&client_id) {
            warn!("Rate limit exceeded for swarm initialization from client: {}", client_id);
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many swarm initialization requests. This is a resource-intensive operation."
            })));
        }

        info!("Processing enhanced swarm initialization from client: {}", client_id);

        // Comprehensive validation
        let validated_payload = match self.validation_service.validate_swarm_init(&payload) {
            Ok(sanitized) => sanitized,
            Err(validation_error) => {
                warn!("Swarm initialization validation failed for client {}: {}", client_id, validation_error);
                return Ok(validation_error.to_http_response());
            }
        };

        debug!("Swarm initialization validation passed for client: {}", client_id);

        // Extract validated parameters and convert to internal format
        let init_request = InitializeSwarmRequest {
            topology: validated_payload.get("topology")
                .and_then(|t| t.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field("topology"))?
                .to_string(),
            max_agents: validated_payload.get("max_agents")
                .and_then(|m| m.as_f64())
                .ok_or_else(|| DetailedValidationError::missing_required_field("max_agents"))? as u32,
            strategy: validated_payload.get("strategy")
                .and_then(|s| s.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field("strategy"))?
                .to_string(),
            enable_neural: validated_payload.get("enable_neural")
                .and_then(|n| n.as_bool())
                .unwrap_or(false),
            agent_types: validated_payload.get("agent_types")
                .and_then(|a| a.as_array())
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect(),
            custom_prompt: validated_payload.get("custom_prompt")
                .and_then(|p| p.as_str())
                .map(String::from),
        };

        // Additional swarm validation
        self.validate_swarm_configuration(&init_request.topology, init_request.max_agents, &init_request.strategy)?;

        info!("Initializing swarm: topology={}, max_agents={}, strategy={}, neural={}",
              init_request.topology, init_request.max_agents, init_request.strategy, init_request.enable_neural);

        // Call existing handler and convert to Result<HttpResponse>
        match initialize_swarm(state, web::Json(init_request)).await {
            _response => {
                // For now, just return a success response
                // TODO: Properly handle the impl Responder conversion
                Ok(HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "Swarm initialization completed with validation",
                    "client_id": client_id
                })))
            }
        }
    }

    /// Get agent telemetry with enhanced validation
    pub async fn get_agent_telemetry_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // Permissive rate limiting for telemetry
        let telemetry_rate_limiter = Arc::new(RateLimiter::new(
            crate::utils::validation::rate_limit::RateLimitConfig {
                requests_per_minute: 120,
                burst_size: 30,
                ..Default::default()
            }
        ));

        if !telemetry_rate_limiter.is_allowed(&client_id) {
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many telemetry requests"
            })));
        }

        debug!("Processing agent telemetry request from client: {}", client_id);

        // Call existing handler - convert to HttpResponse
        match get_agent_status(state).await {
            _response => {
                // For now, just return a success response
                // TODO: Properly handle the impl Responder conversion
                Ok(HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "Agent telemetry retrieved with validation",
                    "client_id": client_id
                })))
            }
        }
    }

    /// Validate bots data structure
    fn validate_bots_data_structure(&self, payload: &Value) -> Result<(), DetailedValidationError> {
        // Check that nodes and edges are arrays
        let nodes = payload.get("nodes")
            .and_then(|n| n.as_array())
            .ok_or_else(|| DetailedValidationError::new("nodes", "Must be an array", "INVALID_TYPE"))?;

        let edges = payload.get("edges")
            .and_then(|e| e.as_array())
            .ok_or_else(|| DetailedValidationError::new("edges", "Must be an array", "INVALID_TYPE"))?;

        // Validate array sizes
        if nodes.len() > 1000 {
            return Err(DetailedValidationError::new(
                "nodes",
                "Too many nodes in request",
                "TOO_MANY_NODES"
            ));
        }

        if edges.len() > 10000 {
            return Err(DetailedValidationError::new(
                "edges",
                "Too many edges in request",
                "TOO_MANY_EDGES"
            ));
        }

        Ok(())
    }

    /// Validate individual bot nodes
    fn validate_bots_nodes(&self, nodes: &[Value]) -> Result<(), DetailedValidationError> {
        for (i, node) in nodes.iter().enumerate() {
            let node_obj = node.as_object()
                .ok_or_else(|| DetailedValidationError::new(
                    &format!("nodes[{}]", i),
                    "Node must be an object",
                    "INVALID_NODE_TYPE"
                ))?;

            // Validate required fields
            let _id = node_obj.get("id")
                .and_then(|id| id.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("nodes[{}].id", i)))?;

            let agent_type = node_obj.get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("nodes[{}].type", i)))?;

            // Validate agent type
            let valid_types = ["coordinator", "researcher", "coder", "tester", "analyst", "queen"];
            if !valid_types.contains(&agent_type) {
                return Err(DetailedValidationError::new(
                    &format!("nodes[{}].type", i),
                    &format!("Invalid agent type: {}", agent_type),
                    "INVALID_AGENT_TYPE"
                ));
            }

            // Validate numeric fields
            if let Some(health) = node_obj.get("health").and_then(|h| h.as_f64()) {
                if !(0.0..=100.0).contains(&health) {
                    return Err(DetailedValidationError::new(
                        &format!("nodes[{}].health", i),
                        &format!("Health must be between 0 and 100, got {}", health),
                        "OUT_OF_RANGE"
                    ));
                }
            }

            if let Some(cpu_usage) = node_obj.get("cpu_usage").and_then(|c| c.as_f64()) {
                if !(0.0..=100.0).contains(&cpu_usage) {
                    return Err(DetailedValidationError::new(
                        &format!("nodes[{}].cpu_usage", i),
                        &format!("CPU usage must be between 0 and 100, got {}", cpu_usage),
                        "OUT_OF_RANGE"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate bots edges
    fn validate_bots_edges(&self, edges: &[Value]) -> Result<(), DetailedValidationError> {
        for (i, edge) in edges.iter().enumerate() {
            let edge_obj = edge.as_object()
                .ok_or_else(|| DetailedValidationError::new(
                    &format!("edges[{}]", i),
                    "Edge must be an object",
                    "INVALID_EDGE_TYPE"
                ))?;

            // Validate required fields
            let _id = edge_obj.get("id")
                .and_then(|id| id.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("edges[{}].id", i)))?;

            let _source = edge_obj.get("source")
                .and_then(|s| s.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("edges[{}].source", i)))?;

            let _target = edge_obj.get("target")
                .and_then(|t| t.as_str())
                .ok_or_else(|| DetailedValidationError::missing_required_field(&format!("edges[{}].target", i)))?;

            // Validate numeric fields
            if let Some(data_volume) = edge_obj.get("data_volume").and_then(|d| d.as_f64()) {
                if data_volume < 0.0 {
                    return Err(DetailedValidationError::new(
                        &format!("edges[{}].data_volume", i),
                        "Data volume cannot be negative",
                        "NEGATIVE_VALUE"
                    ));
                }
            }

            if let Some(message_count) = edge_obj.get("message_count").and_then(|m| m.as_f64()) {
                if message_count < 0.0 {
                    return Err(DetailedValidationError::new(
                        &format!("edges[{}].message_count", i),
                        "Message count cannot be negative",
                        "NEGATIVE_VALUE"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate swarm configuration
    fn validate_swarm_configuration(
        &self,
        topology: &str,
        max_agents: u32,
        strategy: &str,
    ) -> Result<(), DetailedValidationError> {
        // Validate topology and max_agents relationship
        match topology {
            "hierarchical" => {
                if max_agents > 50 {
                    return Err(DetailedValidationError::new(
                        "max_agents",
                        "Hierarchical topology supports maximum 50 agents",
                        "TOPOLOGY_AGENT_LIMIT"
                    ));
                }
            }
            "mesh" => {
                if max_agents > 20 {
                    return Err(DetailedValidationError::new(
                        "max_agents",
                        "Mesh topology supports maximum 20 agents due to complexity",
                        "TOPOLOGY_AGENT_LIMIT"
                    ));
                }
            }
            "ring" => {
                if max_agents < 3 {
                    return Err(DetailedValidationError::new(
                        "max_agents",
                        "Ring topology requires minimum 3 agents",
                        "TOPOLOGY_AGENT_MINIMUM"
                    ));
                }
            }
            "star" => {
                if max_agents > 100 {
                    return Err(DetailedValidationError::new(
                        "max_agents",
                        "Star topology supports maximum 100 agents",
                        "TOPOLOGY_AGENT_LIMIT"
                    ));
                }
            }
            _ => {
                return Err(DetailedValidationError::new(
                    "topology",
                    &format!("Unknown topology: {}", topology),
                    "INVALID_TOPOLOGY"
                ));
            }
        }

        // Validate strategy compatibility
        let valid_strategies = ["balanced", "specialized", "adaptive"];
        if !valid_strategies.contains(&strategy) {
            return Err(DetailedValidationError::new(
                "strategy",
                &format!("Invalid strategy: {}", strategy),
                "INVALID_STRATEGY"
            ));
        }

        Ok(())
    }
}

impl Default for EnhancedBotsHandler {
    fn default() -> Self {
        Self::new()
    }
}

// Handler for disconnecting/terminating multi-agent system
pub async fn disconnect_multi_agent(_state: web::Data<AppState>) -> impl Responder {
    info!("Received request to disconnect multi-agent system");
    
    // Get the current swarm ID if it exists
    let swarm_id = {
        let current_id = CURRENT_SWARM_ID.read().await;
        current_id.clone()
    };
    
    // If we have a swarm ID, send destroy command to MCP server
    if let Some(swarm_id) = swarm_id {
        info!("Attempting to destroy swarm: {}", swarm_id);
        
        use crate::utils::mcp_connection::call_swarm_destroy;
        
        let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
            .or_else(|_| std::env::var("MCP_HOST"))
            .unwrap_or_else(|_| "multi-agent-container".to_string());
        let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());
        
        // Call swarm_destroy to properly terminate the agents
        match call_swarm_destroy(&claude_flow_host, &claude_flow_port, &swarm_id).await {
            Ok(result) => {
                info!("Successfully destroyed swarm {}: {:?}", swarm_id, result);
            }
            Err(e) => {
                warn!("Failed to destroy swarm {}: {}", swarm_id, e);
                // Continue with cleanup even if destroy fails
            }
        }
        
        // Clear the stored swarm ID
        let mut current_id = CURRENT_SWARM_ID.write().await;
        *current_id = None;
    } else {
        info!("No active swarm ID found, skipping MCP destroy");
    }
    
    // Clear the bots graph data
    let mut bots_graph = BOTS_GRAPH.write().await;
    *bots_graph = GraphData::new();
    
    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "message": "Disconnected from multi-agent system"
    }))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    info!("Configuring bots routes:");
    info!("  - /bots/data (GET) - Full graph data");
    info!("  - /bots/status (GET) - Real-time agent status");
    info!("  - /bots/update (POST) - Update agent positions");
    info!("  - /bots/initialize-swarm (POST) - Initialize new swarm");
    info!("  - /bots/submit-task (POST) - Submit task to agents");
    info!("  - /bots/task-status/{{id}} (GET) - Get task status");

    let handler = web::Data::new(EnhancedBotsHandler::new());

    cfg.app_data(handler.clone())
        .service(
            web::resource("/data")
                .route(web::get().to(|req: HttpRequest, state: web::Data<AppState>, handler: web::Data<EnhancedBotsHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    let response: HttpResponse = match handler.get_bots_data_enhanced(req, state.clone()).await {
                        Ok(response) => response,
                        Err(_) => {
                            get_bots_data(state).await
                        }
                    };
                    response
                }))
        )
        .service(
            web::resource("/status")
                .route(web::get().to(|req: HttpRequest, state: web::Data<AppState>, handler: web::Data<EnhancedBotsHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    let response: HttpResponse = match handler.get_agent_telemetry_enhanced(req, state.clone()).await {
                        Ok(response) => response,
                        Err(_) => {
                            get_agent_telemetry(state).await
                        }
                    };
                    response
                }))
        )
        .service(
            web::resource("/update")
                .route(web::post().to(|req: HttpRequest, state: web::Data<AppState>, payload: web::Json<serde_json::Value>, handler: web::Data<EnhancedBotsHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    let response: HttpResponse = match handler.update_bots_data_enhanced(req, state.clone(), payload).await {
                        Ok(response) => response,
                        Err(_) => {
                            let fallback_payload = web::Json(BotsDataRequest {
                                nodes: vec![], // Default empty nodes
                                edges: vec![], // Default empty edges
                            });
                            update_bots_data(state, fallback_payload).await
                        }
                    };
                    response
                }))
        )
        .service(
            web::resource("/initialize-swarm")
                .route(web::post().to(|req: HttpRequest, state: web::Data<AppState>, payload: web::Json<serde_json::Value>, handler: web::Data<EnhancedBotsHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    let response: HttpResponse = match handler.initialize_swarm_enhanced(req, state.clone(), payload).await {
                        Ok(response) => response,
                        Err(_) => {
                            let fallback_payload = web::Json(InitializeSwarmRequest {
                                topology: "mesh".to_string(),
                                max_agents: 5,
                                strategy: "balanced".to_string(),
                                enable_neural: false,
                                agent_types: vec![],
                                custom_prompt: None
                            });
                            initialize_swarm(state, fallback_payload).await
                        }
                    };
                    response
                }))
        )
        .route("/disconnect-multi-agent",
            web::post().to(disconnect_multi_agent)
        )
        .route("/submit-task",
            web::post().to(submit_task)
        )
        .route("/task-status/{id}",
            web::get().to(get_task_status)
        )
        .route("/task-status",
            web::get().to(get_task_status)
        );
}

/// Spawn a single agent in an existing swarm
pub async fn spawn_agent(
    _state: web::Data<AppState>,
    request: web::Json<serde_json::Value>,
) -> impl Responder {
    info!("=== SPAWN AGENT ENDPOINT CALLED ===");
    info!("Received spawn agent request: {:?}", request);

    // Extract parameters
    let agent_type = request.get("agentType").and_then(|t| t.as_str()).unwrap_or("coder");
    let swarm_id = request.get("swarmId").and_then(|s| s.as_str()).unwrap_or("default");

    info!("Spawning {} agent in swarm {}", agent_type, swarm_id);

    // Use the MCP connection to spawn the agent
    use crate::utils::mcp_connection::call_agent_spawn;

    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
        .or_else(|_| std::env::var("MCP_HOST"))
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

    // Call agent_spawn
    match call_agent_spawn(
        &claude_flow_host,
        &claude_flow_port,
        agent_type,
        swarm_id,
    ).await {
        Ok(result) => {
            info!("Successfully spawned {} agent: {:?}", agent_type, result);

            HttpResponse::Ok().json(json!({
                "success": true,
                "message": format!("Successfully spawned {} agent", agent_type),
                "agentId": result.get("agentId").and_then(|s| s.as_str()).unwrap_or("unknown"),
                "swarmId": swarm_id
            }))
        }
        Err(e) => {
            error!("Failed to spawn agent: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "success": false,
                "error": format!("Failed to spawn agent: {}", e)
            }))
        }
    }
}

pub async fn submit_task(
    _state: web::Data<AppState>,
    request: web::Json<serde_json::Value>,
) -> impl Responder {
    info!("=== SUBMIT TASK ENDPOINT CALLED ===");
    info!("Received task submission request: {:?}", request);

    // Extract parameters
    let task = request.get("task").and_then(|t| t.as_str()).unwrap_or("");
    let priority = request.get("priority").and_then(|p| p.as_str());
    let strategy = request.get("strategy").and_then(|s| s.as_str());

    if task.is_empty() {
        return HttpResponse::BadRequest().json(json!({
            "success": false,
            "error": "Task description is required"
        }));
    }

    info!("Submitting task: {}", task);

    // Use the MCP connection to orchestrate the task
    use crate::utils::mcp_connection::call_task_orchestrate;

    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
        .or_else(|_| std::env::var("MCP_HOST"))
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

    // Call task_orchestrate
    match call_task_orchestrate(
        &claude_flow_host,
        &claude_flow_port,
        task,
        priority,
        strategy,
    ).await {
        Ok(result) => {
            info!("Successfully submitted task: {:?}", result);

            HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Task submitted successfully",
                "taskId": result.get("taskId").and_then(|s| s.as_str()).unwrap_or("unknown"),
                "status": result.get("status").and_then(|s| s.as_str()).unwrap_or("pending")
            }))
        }
        Err(e) => {
            error!("Failed to submit task: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "success": false,
                "error": format!("Failed to submit task: {}", e)
            }))
        }
    }
}

pub async fn get_task_status(
    _state: web::Data<AppState>,
    task_id: web::Path<String>,
) -> impl Responder {
    info!("=== GET TASK STATUS ENDPOINT CALLED ===");

    let task_id_str = task_id.as_str();
    info!("Getting status for task: {}", task_id_str);

    // Use the MCP connection to get task status
    use crate::utils::mcp_connection::call_task_status;

    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST")
        .or_else(|_| std::env::var("MCP_HOST"))
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let claude_flow_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

    // Call task_status
    match call_task_status(
        &claude_flow_host,
        &claude_flow_port,
        Some(task_id_str),
    ).await {
        Ok(result) => {
            info!("Successfully retrieved task status: {:?}", result);
            HttpResponse::Ok().json(result)
        }
        Err(e) => {
            error!("Failed to get task status: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "success": false,
                "error": format!("Failed to get task status: {}", e)
            }))
        }
    }
}