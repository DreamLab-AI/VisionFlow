use crate::models::metadata::Metadata;
use crate::models::node::Node; // Changed from socket_flow_messages::Node
use crate::services::file_service::FileService;
use crate::types::vec3::Vec3Data;
use crate::AppState;
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
// GraphService direct import is no longer needed as we use actors
// use crate::services::graph_service::GraphService;
use crate::actors::messages::{AddNodesFromMetadata, GetSettings};
use crate::application::graph::queries::{
    GetAutoBalanceNotifications, GetGraphData, GetNodeMap, GetPhysicsState,
};
use crate::handlers::utils::execute_in_thread;
use hexser::QueryHandler;

/// Settlement state information for client optimization
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SettlementState {
    pub is_settled: bool,
    pub stable_frame_count: u32,
    pub kinetic_energy: f32,
}

/// Node with physics position data included for immediate client rendering
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct NodeWithPosition {
    // Core node data
    pub id: u32,
    pub metadata_id: String,
    pub label: String,

    // Physics state (NEW - prevents client-side random positioning)
    pub position: Vec3Data,
    pub velocity: Vec3Data,

    // Metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,

    // Rendering properties
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
}

/// Original GraphResponse for backward compatibility
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<crate::models::edge::Edge>,
    pub metadata: HashMap<String, Metadata>,
}

/// NEW: Enhanced response with physics positions for optimized client initialization
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphResponseWithPositions {
    pub nodes: Vec<NodeWithPosition>,
    pub edges: Vec<crate::models::edge::Edge>,
    pub metadata: HashMap<String, Metadata>,
    pub settlement_state: SettlementState,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PaginatedGraphResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<crate::models::edge::Edge>,
    pub metadata: HashMap<String, Metadata>,
    pub total_pages: usize,
    pub current_page: usize,
    pub total_items: usize,
    pub page_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphQuery {
    pub query: Option<String>,
    pub page: Option<usize>,
    pub page_size: Option<usize>,
    pub sort: Option<String>,
    pub filter: Option<String>,
}

pub async fn get_graph_data(state: web::Data<AppState>, _req: HttpRequest) -> impl Responder {
    info!("Received request for graph data (CQRS Phase 1D)");

    // Use CQRS query handlers instead of actor messages
    let graph_handler = state.graph_query_handlers.get_graph_data.clone();
    let node_map_handler = state.graph_query_handlers.get_node_map.clone();
    let physics_handler = state.graph_query_handlers.get_physics_state.clone();

    // Execute queries in separate OS threads to avoid Tokio runtime blocking
    let graph_future = execute_in_thread(move || graph_handler.handle(GetGraphData));
    let node_map_future = execute_in_thread(move || node_map_handler.handle(GetNodeMap));
    let physics_future = execute_in_thread(move || physics_handler.handle(GetPhysicsState));

    let (graph_result, node_map_result, physics_result) =
        tokio::join!(graph_future, node_map_future, physics_future);

    match (graph_result, node_map_result, physics_result) {
        (Ok(Ok(graph_data)), Ok(Ok(node_map)), Ok(Ok(physics_state))) => {
            debug!(
                "Preparing enhanced graph response with {} nodes, {} edges, physics state: {:?}",
                graph_data.nodes.len(),
                graph_data.edges.len(),
                physics_state
            );

            // Build nodes with CURRENT physics positions from node_map
            let nodes_with_positions: Vec<NodeWithPosition> = graph_data
                .nodes
                .iter()
                .map(|node| {
                    // Get current position from physics-simulated node_map
                    let (position, velocity) = if let Some(physics_node) = node_map.get(&node.id) {
                        (physics_node.data.position(), physics_node.data.velocity())
                    } else {
                        // Fallback to original position if not in node_map yet
                        (node.data.position(), node.data.velocity())
                    };

                    NodeWithPosition {
                        id: node.id,
                        metadata_id: node.metadata_id.clone(),
                        label: node.label.clone(),
                        position,
                        velocity,
                        metadata: node.metadata.clone(),
                        node_type: node.node_type.clone(),
                        size: node.size,
                        color: node.color.clone(),
                        weight: node.weight,
                        group: node.group.clone(),
                    }
                })
                .collect();

            let response = GraphResponseWithPositions {
                nodes: nodes_with_positions,
                edges: graph_data.edges.clone(),
                metadata: graph_data.metadata.clone(),
                settlement_state: SettlementState {
                    is_settled: physics_state.is_settled,
                    stable_frame_count: physics_state.stable_frame_count,
                    kinetic_energy: physics_state.kinetic_energy,
                },
            };

            info!(
                "Sending graph data with {} nodes (CQRS query handlers)",
                response.nodes.len()
            );

            HttpResponse::Ok().json(response)
        }
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
            error!("Thread execution error: {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "Internal server error"}))
        }
        (Ok(Err(e)), _, _) | (_, Ok(Err(e)), _) | (_, _, Ok(Err(e))) => {
            error!("Failed to fetch graph data (CQRS): {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "Failed to retrieve graph data"}))
        }
    }
}

pub async fn get_paginated_graph_data(
    state: web::Data<AppState>,
    query: web::Query<GraphQuery>,
) -> impl Responder {
    info!(
        "Received request for paginated graph data (CQRS Phase 1D): {:?}",
        query
    );

    let page = query.page.map(|p| p.saturating_sub(1)).unwrap_or(0);
    let page_size = query.page_size.unwrap_or(100);

    if page_size == 0 {
        error!("Invalid page size: {}", page_size);
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": "Page size must be greater than 0"
        }));
    }

    // Use CQRS query handler
    let graph_handler = state.graph_query_handlers.get_graph_data.clone();
    let graph_result = execute_in_thread(move || graph_handler.handle(GetGraphData)).await;

    let graph_data_owned = match graph_result {
        Ok(Ok(g_owned)) => g_owned,
        Ok(Err(e)) => {
            error!("Failed to get graph data for pagination (CQRS): {}", e);
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "Failed to retrieve graph data"}));
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "Internal server error"}));
        }
    };

    let total_items = graph_data_owned.nodes.len();

    if total_items == 0 {
        debug!("Graph is empty");
        return HttpResponse::Ok().json(PaginatedGraphResponse {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
            total_pages: 0,
            current_page: 1,
            total_items: 0,
            page_size,
        });
    }

    let total_pages = (total_items + page_size - 1) / page_size;

    if page >= total_pages {
        warn!(
            "Requested page {} exceeds total pages {}",
            page + 1,
            total_pages
        );
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": format!("Page {} exceeds total available pages {}", page + 1, total_pages)
        }));
    }

    let start = page * page_size;
    let end = std::cmp::min(start + page_size, total_items);

    debug!(
        "Calculating slice from {} to {} out of {} total items",
        start, end, total_items
    );

    let page_nodes = graph_data_owned.nodes[start..end].to_vec();

    let node_ids: std::collections::HashSet<_> = page_nodes.iter().map(|node| node.id).collect();

    let relevant_edges: Vec<_> = graph_data_owned
        .edges
        .iter()
        .filter(|edge| node_ids.contains(&edge.source) || node_ids.contains(&edge.target))
        .cloned()
        .collect();

    debug!(
        "Found {} relevant edges for {} nodes (CQRS)",
        relevant_edges.len(),
        page_nodes.len()
    );

    let response = PaginatedGraphResponse {
        nodes: page_nodes,
        edges: relevant_edges,
        metadata: graph_data_owned.metadata.clone(),
        total_pages,
        current_page: page + 1,
        total_items,
        page_size,
    };

    HttpResponse::Ok().json(response)
}

pub async fn refresh_graph(state: web::Data<AppState>) -> impl Responder {
    info!("Received request to refresh graph (CQRS Phase 1D)");

    // Use CQRS query handler
    let graph_handler = state.graph_query_handlers.get_graph_data.clone();
    let graph_result = execute_in_thread(move || graph_handler.handle(GetGraphData)).await;

    match graph_result {
        Ok(Ok(graph_data_owned)) => {
            debug!(
                "Returning current graph state with {} nodes and {} edges (CQRS)",
                graph_data_owned.nodes.len(),
                graph_data_owned.edges.len()
            );

            let response = GraphResponse {
                nodes: graph_data_owned.nodes.clone(),
                edges: graph_data_owned.edges.clone(),
                metadata: graph_data_owned.metadata.clone(),
            };

            HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "message": "Graph data retrieved successfully",
                "data": response
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to get current graph data (CQRS): {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Failed to retrieve current graph data"
            }))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Internal server error"
            }))
        }
    }
}

pub async fn update_graph(state: web::Data<AppState>) -> impl Responder {
    info!("Received request to update graph");

    let mut metadata = match FileService::load_or_create_metadata() {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to load metadata: {}", e);
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("Failed to load metadata: {}", e)
            }));
        }
    };

    let settings_result = state.settings_addr.send(GetSettings).await;
    let settings = match settings_result {
        Ok(Ok(s)) => Arc::new(tokio::sync::RwLock::new(s)),
        _ => {
            error!("Failed to retrieve settings for FileService in update_graph");
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Failed to retrieve application settings"
            }));
        }
    };

    let file_service = FileService::new(settings.clone());
    match file_service
        .fetch_and_process_files(state.content_api.clone(), settings.clone(), &mut metadata)
        .await
    {
        Ok(processed_files) => {
            if processed_files.is_empty() {
                debug!("No new files to process");
                return HttpResponse::Ok().json(serde_json::json!({
                    "success": true,
                    "message": "No updates needed"
                }));
            }

            debug!("Processing {} new files", processed_files.len());

            {
                // Send UpdateMetadata message to MetadataActor
                if let Err(e) = state
                    .metadata_addr
                    .send(crate::actors::messages::UpdateMetadata {
                        metadata: metadata.clone(),
                    })
                    .await
                {
                    error!("Failed to send UpdateMetadata to MetadataActor: {}", e);
                    // Potentially return error if this is critical
                }
            }

            // Send AddNodesFromMetadata for incremental updates instead of full rebuild
            match state
                .graph_service_addr
                .send(AddNodesFromMetadata { metadata })
                .await
            {
                Ok(Ok(())) => {
                    // Position preservation logic would need to be handled by the actor or subsequent messages.
                    debug!(
                        "Graph updated successfully via GraphServiceActor after file processing"
                    );
                    HttpResponse::Ok().json(serde_json::json!({
                        "success": true,
                        "message": format!("Graph updated with {} new files", processed_files.len())
                    }))
                }
                Ok(Err(e)) => {
                    error!(
                        "GraphServiceActor failed to build graph from metadata: {}",
                        e
                    );
                    HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to build graph: {}", e)
                    }))
                }
                Err(e) => {
                    error!("Failed to build new graph: {}", e);
                    HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": format!("Failed to build new graph: {}", e)
                    }))
                }
            }
        }
        Err(e) => {
            error!("Failed to fetch and process files: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("Failed to fetch and process files: {}", e)
            }))
        }
    }
}

// Auto-balance notifications endpoint
pub async fn get_auto_balance_notifications(
    state: web::Data<AppState>,
    query: web::Query<serde_json::Value>,
) -> impl Responder {
    let since_timestamp = query.get("since").and_then(|v| v.as_i64());

    info!("Fetching auto-balance notifications (CQRS Phase 1D)");

    // Use CQRS query handler
    let handler = state
        .graph_query_handlers
        .get_auto_balance_notifications
        .clone();
    let query_obj = GetAutoBalanceNotifications { since_timestamp };

    let result = execute_in_thread(move || handler.handle(query_obj)).await;

    match result {
        Ok(Ok(notifications)) => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "notifications": notifications
        })),
        Ok(Err(e)) => {
            error!("Failed to get auto-balance notifications (CQRS): {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Failed to retrieve notifications"
            }))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": "Internal server error"
            }))
        }
    }
}

// Configure routes using snake_case
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/graph")
            // Match client's endpoint pattern exactly
            .route("/data", web::get().to(get_graph_data))
            .route("/data/paginated", web::get().to(get_paginated_graph_data))
            .route("/update", web::post().to(update_graph))
            // Keep refresh endpoint for admin/maintenance
            .route("/refresh", web::post().to(refresh_graph))
            // Auto-balance notifications
            .route(
                "/auto-balance-notifications",
                web::get().to(get_auto_balance_notifications),
            ),
    );
}
