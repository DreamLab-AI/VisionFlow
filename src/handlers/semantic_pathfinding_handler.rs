//! Semantic Pathfinding Handler - API endpoints for intelligent graph traversal

use actix_web::{web, HttpResponse, Responder};
use log::{info, debug};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::services::semantic_pathfinding_service::{
    SemanticPathfindingService, PathResult, PathfindingConfig
};
use crate::actors::graph_state_actor::GraphStateActor;
use actix::Addr;
use crate::{ok_json, error_json};
use crate::utils::handler_commons::HandlerResponse;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindPathRequest {
    pub start_id: u32,
    pub end_id: u32,
    pub query: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraversalRequest {
    pub start_id: u32,
    pub query: Option<String>,
    pub max_nodes: Option<usize>,
}

pub async fn find_semantic_path(
    pathfinding_service: web::Data<Arc<SemanticPathfindingService>>,
    graph_state_actor: web::Data<Addr<GraphStateActor>>,
    request: web::Json<FindPathRequest>,
) -> impl Responder {
    info!("Finding semantic path from {} to {}", request.start_id, request.end_id);

    let graph_result = graph_state_actor
        .send(crate::actors::graph_messages::GetGraphState)
        .await;

    match graph_result {
        Ok(Ok(graph_state)) => {
            match pathfinding_service.find_semantic_path(
                &graph_state.graph,
                request.start_id,
                request.end_id,
                request.query.as_deref(),
            ) {
                Some(path) => ok_json!(path),
                None => error_json!("No path found", "Could not find path between nodes"),
            }
        }
        Ok(Err(e)) => error_json!("Graph error", e.to_string()),
        Err(e) => error_json!("Actor error", e.to_string()),
    }
}

pub async fn query_traversal(
    pathfinding_service: web::Data<Arc<SemanticPathfindingService>>,
    graph_state_actor: web::Data<Addr<GraphStateActor>>,
    request: web::Json<TraversalRequest>,
) -> impl Responder {
    info!("Query traversal from {}", request.start_id);

    let graph_result = graph_state_actor
        .send(crate::actors::graph_messages::GetGraphState)
        .await;

    match graph_result {
        Ok(Ok(graph_state)) => {
            if let Some(ref query) = request.query {
                let results = pathfinding_service.query_traversal(
                    &graph_state.graph,
                    request.start_id,
                    query,
                    request.max_nodes.unwrap_or(50),
                );
                ok_json!(serde_json::json!({ "results": results }))
            } else {
                error_json!("Missing query", "Query parameter required")
            }
        }
        Ok(Err(e)) => error_json!("Graph error", e.to_string()),
        Err(e) => error_json!("Actor error", e.to_string()),
    }
}

pub async fn chunk_traversal(
    pathfinding_service: web::Data<Arc<SemanticPathfindingService>>,
    graph_state_actor: web::Data<Addr<GraphStateActor>>,
    request: web::Json<TraversalRequest>,
) -> impl Responder {
    debug!("Chunk traversal from {}", request.start_id);

    let graph_result = graph_state_actor
        .send(crate::actors::graph_messages::GetGraphState)
        .await;

    match graph_result {
        Ok(Ok(graph_state)) => {
            let results = pathfinding_service.chunk_traversal(
                &graph_state.graph,
                request.start_id,
                request.max_nodes.unwrap_or(50),
            );
            ok_json!(serde_json::json!({ "results": results }))
        }
        Ok(Err(e)) => error_json!("Graph error", e.to_string()),
        Err(e) => error_json!("Actor error", e.to_string()),
    }
}

pub fn configure_pathfinding_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/pathfinding")
            .route("/semantic-path", web::post().to(find_semantic_path))
            .route("/query-traversal", web::post().to(query_traversal))
            .route("/chunk-traversal", web::post().to(chunk_traversal))
    );
}
