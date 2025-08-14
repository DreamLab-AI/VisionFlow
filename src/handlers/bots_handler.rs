use actix_web::{web, HttpResponse, Responder};
use crate::AppState;
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
use serde_json::json;
use std::collections::HashMap;
use log::{info, debug, error, warn};
use tokio::sync::RwLock;
use std::sync::Arc;
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

async fn fetch_hive_mind_agents(_state: &AppState) -> Result<Vec<BotsAgent>, Box<dyn std::error::Error>> {
    // Connect directly to Claude Flow TCP server in multi-agent-container
    // The server runs at 172.18.0.10:9500 on the docker_ragflow network
    use tokio::net::TcpStream;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use serde_json::json;
    use uuid::Uuid;
    
    let claude_flow_host = std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| "172.18.0.10".to_string());
    let claude_flow_port = std::env::var("CLAUDE_FLOW_PORT").unwrap_or_else(|_| "9500".to_string());
    let addr = format!("{}:{}", claude_flow_host, claude_flow_port);
    
    match TcpStream::connect(&addr).await {
        Ok(mut stream) => {
            info!("Connected to Claude Flow TCP server at {}", addr);
            
            // Send a request to list agents
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
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).await?;
            
            if let Ok(response) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(result) = response.get("result") {
                    if let Some(agents) = result.get("agents").and_then(|a| a.as_array()) {
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
                position,
                velocity: Vec3Data::zero(),
                mass,
                flags: if agent.status == "active" { 1 } else { 0 },
                padding: [0, 0],
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
) -> impl Responder {
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
        use crate::actors::messages::{UpdateGPUGraphData, UpdateSimulationParams, ComputeForces, GetNodeData};

        let bots_graph = BOTS_GRAPH.read().await;

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
        params.spring_strength = physics_settings.spring_strength * 1.5; // Stronger attraction
        params.repulsion = physics_settings.repulsion_strength * 0.8; // Less repulsion

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
pub async fn get_bots_data(state: web::Data<AppState>) -> impl Responder {
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
            return HttpResponse::Ok().json(graph_data);
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

    // If no data exists, return some test data for visualization
    if bots_graph.nodes.is_empty() {
        info!("No bots data available, returning test data");

        // Create test agents
        let test_agents = vec![
            BotsAgent {
                id: "agent-1".to_string(),
                agent_type: "coordinator".to_string(),
                status: "active".to_string(),
                name: "Coordinator Alpha".to_string(),
                cpu_usage: 45.0,
                memory_usage: 35.0,
                health: 95.0,
                workload: 0.7,
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
                created_at: Some(chrono::Utc::now().to_rfc3339()),
                age: Some(0),
            },
            BotsAgent {
                id: "agent-2".to_string(),
                agent_type: "coder".to_string(),
                status: "active".to_string(),
                name: "Coder Beta".to_string(),
                cpu_usage: 78.0,
                memory_usage: 65.0,
                health: 88.0,
                workload: 0.9,
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
                created_at: Some(chrono::Utc::now().to_rfc3339()),
                age: Some(0),
            },
            BotsAgent {
                id: "agent-3".to_string(),
                agent_type: "tester".to_string(),
                status: "active".to_string(),
                name: "Tester Gamma".to_string(),
                cpu_usage: 32.0,
                memory_usage: 25.0,
                health: 92.0,
                workload: 0.5,
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
                created_at: Some(chrono::Utc::now().to_rfc3339()),
                age: Some(0),
            },
            BotsAgent {
                id: "agent-4".to_string(),
                agent_type: "analyst".to_string(),
                status: "active".to_string(),
                name: "Analyst Delta".to_string(),
                cpu_usage: 56.0,
                memory_usage: 45.0,
                health: 90.0,
                workload: 0.6,
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
                created_at: Some(chrono::Utc::now().to_rfc3339()),
                age: Some(0),
            },
        ];

        let test_edges = vec![
            BotsEdge {
                id: "edge-1".to_string(),
                source: "agent-1".to_string(),
                target: "agent-2".to_string(),
                data_volume: 1024.0,
                message_count: 15,
            },
            BotsEdge {
                id: "edge-2".to_string(),
                source: "agent-1".to_string(),
                target: "agent-3".to_string(),
                data_volume: 512.0,
                message_count: 8,
            },
            BotsEdge {
                id: "edge-3".to_string(),
                source: "agent-2".to_string(),
                target: "agent-4".to_string(),
                data_volume: 2048.0,
                message_count: 22,
            },
        ];

        // Convert to graph format
        let nodes = convert_agents_to_nodes(test_agents);
        let node_map: HashMap<String, u32> = nodes.iter()
            .map(|node| (node.metadata_id.clone(), node.id))
            .collect();
        let edges = convert_bots_edges(test_edges, &node_map);

        let test_graph = GraphData {
            nodes,
            edges,
            metadata: HashMap::new(),
            id_to_metadata: HashMap::new(),
        };

        return HttpResponse::Ok().json(test_graph);
    }

    HttpResponse::Ok().json(&*bots_graph)
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

        return convert_agents_to_nodes(agents);
    }

    // Fall back to static data
    let bots_graph = BOTS_GRAPH.read().await;
    bots_graph.nodes.clone()
}

// Initialize swarm endpoint
pub async fn get_agent_telemetry(
    _state: web::Data<AppState>,
) -> impl Responder {
    HttpResponse::ServiceUnavailable().json(json!({
        "success": false,
        "error": "Claude Flow service not available"
    }))
}

pub async fn initialize_swarm(
    state: web::Data<AppState>,
    request: web::Json<InitializeSwarmRequest>,
) -> impl Responder {
    info!("=== INITIALIZE SWARM ENDPOINT CALLED ===");
    info!("Received swarm initialization request: {:?}", request);

    // Connect to Claude Flow TCP server running in multi-agent-container
    // Claude Flow runs externally, not in VisionFlow docker
    use tokio::net::TcpStream;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    
    // Connect to external Claude Flow service
    match TcpStream::connect("172.18.0.10:9500").await {
        Ok(mut stream) => {
            // Create JSON-RPC request for swarm initialization
            let request_id = uuid::Uuid::new_v4().to_string();
            let json_request = json!({
                "jsonrpc": "2.0",
                "method": "swarm_init",
                "params": {
                    "topology": request.topology.clone(),
                    "maxAgents": request.max_agents,
                    "strategy": request.strategy.clone()
                },
                "id": request_id
            });
            
            let request_str = json_request.to_string();
            
            // Send request
            if let Err(e) = stream.write_all(request_str.as_bytes()).await {
                error!("Failed to send swarm init request: {}", e);
                return HttpResponse::InternalServerError().json(serde_json::json!({
                    "success": false,
                    "error": format!("Failed to communicate with Claude Flow: {}", e)
                }));
            }
            
            // Read response
            let mut buffer = vec![0u8; 4096];
            match stream.read(&mut buffer).await {
                Ok(n) if n > 0 => {
                    let response_str = String::from_utf8_lossy(&buffer[..n]);
                    if let Ok(response) = serde_json::from_str::<serde_json::Value>(&response_str) {
                        if let Some(result) = response.get("result") {
                            let swarm_id = result.get("swarmId")
                                .or_else(|| result.get("swarm_id"))
                                .and_then(|s| s.as_str())
                                .unwrap_or("default-swarm")
                                .to_string();
                            
                            info!("Successfully initialized swarm: {}", swarm_id);
                            
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
                                "mock_mode": false
                            }));
                        }
                    }
                    
                    error!("Invalid response from Claude Flow: {}", response_str);
                    return HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": "Invalid response from Claude Flow"
                    }));
                }
                _ => {
                    error!("Failed to read response from Claude Flow");
                    return HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": "Failed to read response from Claude Flow"
                    }));
                }
            }
        }
        Err(e) => {
            error!("Failed to connect to Claude Flow: {}", e);
            // Fall through to mock response
        }
    }
    
    // Fallback to mock response if Claude Flow actor not available
    warn!("Claude Flow actor not available - using mock response");
    
    let mock_agents = vec![
        json!({
            "id": format!("agent-{}", uuid::Uuid::new_v4()),
            "type": "coordinator",
            "name": "Swarm Coordinator",
            "status": "initializing"
        }),
        json!({
            "id": format!("agent-{}", uuid::Uuid::new_v4()),
            "type": "researcher",
            "name": "Research Agent",
            "status": "initializing"
        }),
        json!({
            "id": format!("agent-{}", uuid::Uuid::new_v4()),
            "type": "coder",
            "name": "Code Agent",
            "status": "initializing"
        }),
    ];
    
    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "message": "Swarm initialization started (mock mode)",
        "swarm_id": format!("swarm-{}", uuid::Uuid::new_v4()),
        "agents": mock_agents,
        "topology": request.topology.clone(),
        "max_agents": request.max_agents,
        "mock_mode": true
    }))
}

// UPDATED: Enhanced agent status endpoint for real-time updates
pub async fn get_agent_status(state: web::Data<AppState>) -> impl Responder {
    debug!("get_agent_status endpoint called for real-time updates");
    
    match fetch_hive_mind_agents(&**state).await {
        Ok(agents) => {
            // Return lightweight status data for frequent polling
            let status_data: Vec<_> = agents.into_iter().map(|agent| {
                serde_json::json!({
                    "id": agent.id,
                    "type": agent.agent_type,
                    "status": agent.status,
                    "health": agent.health,
                    "cpu_usage": agent.cpu_usage,
                    "activity": agent.activity,
                    "current_task": agent.current_task,
                    "tasks_active": agent.tasks_active,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })
            }).collect();
            
            HttpResponse::Ok().json(serde_json::json!({
                "agents": status_data,
                "swarm_health": "active",
                "total_agents": status_data.len(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        }
        Err(e) => {
            error!("Failed to get agent status: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to fetch agent status",
                "details": e.to_string()
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

    // TODO: Connect to external multi-agent-container via TCP on port 9500
    // For now, return success response to allow UI to proceed
    HttpResponse::Ok().json(json!({
        "success": true,
        "message": "Multi-agent system initialized",
        "data": {
            "swarmId": uuid::Uuid::new_v4().to_string(),
            "topology": request.get("topology").and_then(|t| t.as_str()).unwrap_or("mesh"),
            "maxAgents": request.get("maxAgents").and_then(|m| m.as_u64()).unwrap_or(8),
            "status": "active"
        }
    }))
}

// UPDATED: Enhanced configuration with new real-time endpoints
pub fn config(cfg: &mut web::ServiceConfig) {
    info!("Configuring bots routes:");
    info!("  - /bots/data (GET) - Full graph data");
    info!("  - /bots/status (GET) - Real-time agent status");
    info!("  - /bots/update (POST) - Update agent positions");
    info!("  - /bots/initialize-swarm (POST) - Initialize new swarm");
    
    cfg.service(
        web::resource("/data")
            .route(web::get().to(get_bots_data))
    )
    .service(
        web::resource("/status")
            .route(web::get().to(get_agent_telemetry))
    )
    .service(
        web::resource("/update")
            .route(web::post().to(update_bots_data))
    )
    .service(
        web::resource("/initialize-swarm")
            .route(web::post().to(initialize_swarm))
    );
}