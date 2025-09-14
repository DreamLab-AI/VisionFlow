use actix_web::{web, HttpResponse, Responder};
use crate::AppState;
use serde::Serialize;
use log::{info, debug, error};
use crate::actors::messages::{GetGraphData, GetSettings};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphStateResponse {
    pub nodes_count: usize,
    pub edges_count: usize,
    pub metadata_count: usize,
    pub positions: Vec<NodePosition>,
    pub settings_version: String,
    pub timestamp: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodePosition {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Get complete graph state including counts, positions, and settings version
pub async fn get_graph_state(state: web::Data<AppState>) -> impl Responder {
    info!("Received request for complete graph state");
    
    // Get graph data from GraphServiceActor
    let graph_data_result = state.graph_service_addr.send(GetGraphData).await;
    
    match graph_data_result {
        Ok(Ok(graph_data)) => {
            // Get current settings to include version info
            let settings_version = match state.settings_addr.send(GetSettings).await {
                Ok(Ok(_settings)) => {
                    // In a real implementation, you'd extract version from settings
                    // For now, using a static version
                    "1.0.0".to_string()
                }
                _ => "unknown".to_string()
            };
            
            // Extract node positions
            let positions: Vec<NodePosition> = graph_data.nodes.iter()
                .map(|node| NodePosition {
                    id: node.id,
                    x: node.data.position.x,
                    y: node.data.position.y,
                    z: node.data.position.z,
                })
                .collect();
            
            let response = GraphStateResponse {
                nodes_count: graph_data.nodes.len(),
                edges_count: graph_data.edges.len(),
                metadata_count: graph_data.metadata.len(),
                positions,
                settings_version,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            
            debug!(
                "Returning graph state: {} nodes, {} edges, {} metadata entries",
                response.nodes_count,
                response.edges_count,
                response.metadata_count
            );
            
            HttpResponse::Ok().json(response)
        }
        Ok(Err(e)) => {
            error!("Failed to get graph data from actor: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to retrieve graph state"
            }))
        }
        Err(e) => {
            error!("Mailbox error getting graph data: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Graph service unavailable"
            }))
        }
    }
}

/// Configure routes for graph state endpoints
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/graph")
            .route("/state", web::get().to(get_graph_state))
    );
}