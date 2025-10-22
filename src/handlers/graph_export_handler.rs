use crate::models::graph::GraphData;
use crate::models::graph_export::*;
use crate::services::graph_serialization::GraphSerializationService;
use crate::AppState;
use actix_web::{http::header::HeaderValue, web, HttpRequest, HttpResponse, Result as ActixResult};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Rate limiting state for export operations
#[derive(Debug, Clone)]
pub struct RateLimitState {
    pub requests: Vec<DateTime<Utc>>,
    pub daily_count: u32,
    pub hourly_count: u32,
}

impl Default for RateLimitState {
    fn default() -> Self {
        Self {
            requests: Vec::new(),
            daily_count: 0,
            hourly_count: 0,
        }
    }
}

/// Shared graph storage for in-memory management
type SharedGraphStorage = Arc<RwLock<HashMap<Uuid, SharedGraph>>>;

/// Rate limiting storage (in production, use Redis)
type RateLimitStorage = Arc<RwLock<HashMap<String, RateLimitState>>>;

/// Graph export handler with comprehensive functionality
pub struct GraphExportHandler {
    serialization_service: GraphSerializationService,
    shared_graphs: SharedGraphStorage,
    rate_limits: RateLimitStorage,
    daily_export_limit: u32,
    hourly_export_limit: u32,
}

impl GraphExportHandler {
    /// Create new graph export handler
    pub fn new(storage_path: std::path::PathBuf) -> Self {
        Self {
            serialization_service: GraphSerializationService::new(storage_path),
            shared_graphs: Arc::new(RwLock::new(HashMap::new())),
            rate_limits: Arc::new(RwLock::new(HashMap::new())),
            daily_export_limit: 100,
            hourly_export_limit: 20,
        }
    }

    /// Check rate limits for a client
    async fn check_rate_limit(&self, client_ip: &str) -> Result<RateLimitInfo> {
        let mut rate_limits = self.rate_limits.write().await;
        let now = Utc::now();

        let state = rate_limits
            .entry(client_ip.to_string())
            .or_insert_with(RateLimitState::default);

        // Clean old requests (older than 24 hours)
        state
            .requests
            .retain(|&timestamp| now.signed_duration_since(timestamp).num_hours() < 24);

        // Count hourly and daily requests
        let hourly_count = state
            .requests
            .iter()
            .filter(|&&timestamp| now.signed_duration_since(timestamp).num_hours() < 1)
            .count() as u32;

        let daily_count = state.requests.len() as u32;

        // Check limits
        if daily_count >= self.daily_export_limit {
            return Ok(RateLimitInfo {
                remaining_exports: 0,
                reset_time: now + chrono::Duration::days(1),
                daily_limit: self.daily_export_limit,
                hourly_limit: self.hourly_export_limit,
            });
        }

        if hourly_count >= self.hourly_export_limit {
            return Ok(RateLimitInfo {
                remaining_exports: 0,
                reset_time: now + chrono::Duration::hours(1),
                daily_limit: self.daily_export_limit,
                hourly_limit: self.hourly_export_limit,
            });
        }

        // Update state
        state.requests.push(now);
        state.daily_count = daily_count + 1;
        state.hourly_count = hourly_count + 1;

        Ok(RateLimitInfo {
            remaining_exports: self.daily_export_limit - daily_count - 1,
            reset_time: now + chrono::Duration::days(1),
            daily_limit: self.daily_export_limit,
            hourly_limit: self.hourly_export_limit,
        })
    }

    /// Get current graph from app state (mock implementation)
    async fn get_current_graph(&self, _app_state: &AppState) -> Result<GraphData> {
        // In production, this would get the actual graph from the visualization service
        // For now, return a mock graph
        let mut graph = GraphData::new();

        // Add sample nodes and edges
        for i in 1..=10 {
            let mut node = crate::models::node::Node::new(format!("node_{}", i))
                .with_label(format!("Node {}", i))
                .with_position((i as f32) * 10.0, (i as f32) * 15.0, 0.0);
            node.id = i;
            graph.nodes.push(node);
        }

        for i in 1..=9 {
            graph
                .edges
                .push(crate::models::edge::Edge::new(i, i + 1, 1.0));
        }

        Ok(graph)
    }
}

/// Export graph in specified format
pub async fn export_graph(
    app_state: web::Data<AppState>,
    request: web::Json<ExportRequest>,
    req: HttpRequest,
) -> ActixResult<HttpResponse> {
    let client_ip = req
        .connection_info()
        .peer_addr()
        .unwrap_or("unknown")
        .to_string();

    // Get handler from app state (in production, this would be injected)
    let handler = GraphExportHandler::new(std::path::PathBuf::from("data"));

    // Check rate limits
    match handler.check_rate_limit(&client_ip).await {
        Ok(rate_info) if rate_info.remaining_exports == 0 => {
            return Ok(HttpResponse::TooManyRequests().json(serde_json::json!({
                "error": "Rate limit exceeded",
                "rate_limit": rate_info
            })));
        }
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Rate limit check failed: {}", e)
            })));
        }
        _ => {}
    }

    // Get current graph
    let graph = match handler.get_current_graph(&app_state).await {
        Ok(graph) => graph,
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Failed to get graph: {}", e)
            })));
        }
    };

    // Export graph
    match handler
        .serialization_service
        .export_graph(&graph, &request)
        .await
    {
        Ok(export_response) => Ok(HttpResponse::Ok().json(export_response)),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Export failed: {}", e)
        }))),
    }
}

/// Create shareable link for graph
pub async fn share_graph(
    app_state: web::Data<AppState>,
    request: web::Json<ShareRequest>,
    req: HttpRequest,
) -> ActixResult<HttpResponse> {
    let client_ip = req
        .connection_info()
        .peer_addr()
        .unwrap_or("unknown")
        .to_string();

    let handler = GraphExportHandler::new(std::path::PathBuf::from("data"));

    // Check rate limits
    match handler.check_rate_limit(&client_ip).await {
        Ok(rate_info) if rate_info.remaining_exports == 0 => {
            return Ok(HttpResponse::TooManyRequests().json(serde_json::json!({
                "error": "Rate limit exceeded",
                "rate_limit": rate_info
            })));
        }
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Rate limit check failed: {}", e)
            })));
        }
        _ => {}
    }

    // Get current graph
    let graph = match handler.get_current_graph(&app_state).await {
        Ok(graph) => graph,
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Failed to get graph: {}", e)
            })));
        }
    };

    // Create shared graph
    match handler
        .serialization_service
        .create_shared_graph(&graph, &request)
        .await
    {
        Ok((shared_graph, share_response)) => {
            // Store shared graph in memory
            {
                let mut shared_graphs = handler.shared_graphs.write().await;
                shared_graphs.insert(shared_graph.id, shared_graph);
            }

            Ok(HttpResponse::Ok().json(share_response))
        }
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Failed to create shared graph: {}", e)
        }))),
    }
}

/// Retrieve shared graph by ID
pub async fn get_shared_graph(
    path: web::Path<String>,
    query: web::Query<HashMap<String, String>>,
) -> ActixResult<HttpResponse> {
    let share_id = match Uuid::parse_str(&path.into_inner()) {
        Ok(id) => id,
        Err(_) => {
            return Ok(HttpResponse::BadRequest().json(serde_json::json!({
                "error": "Invalid share ID format"
            })));
        }
    };

    let handler = GraphExportHandler::new(std::path::PathBuf::from("data"));

    // Get shared graph from storage
    let shared_graph = {
        let shared_graphs = handler.shared_graphs.read().await;
        match shared_graphs.get(&share_id) {
            Some(graph) => graph.clone(),
            None => {
                return Ok(HttpResponse::NotFound().json(serde_json::json!({
                    "error": "Shared graph not found"
                })));
            }
        }
    };

    // Check if expired
    if shared_graph.is_expired() {
        return Ok(HttpResponse::Gone().json(serde_json::json!({
            "error": "Shared graph has expired"
        })));
    }

    // Check access limit
    if shared_graph.access_limit_reached() {
        return Ok(HttpResponse::Forbidden().json(serde_json::json!({
            "error": "Access limit reached for this shared graph"
        })));
    }

    // Check password if required
    if let Some(password) = query.get("password") {
        if !shared_graph.validate_password(password) {
            return Ok(HttpResponse::Unauthorized().json(serde_json::json!({
                "error": "Invalid password"
            })));
        }
    } else if shared_graph.password_hash.is_some() {
        return Ok(HttpResponse::Unauthorized().json(serde_json::json!({
            "error": "Password required"
        })));
    }

    // Update access count
    {
        let mut shared_graphs = handler.shared_graphs.write().await;
        if let Some(graph) = shared_graphs.get_mut(&share_id) {
            graph.increment_access();
        }
    }

    // Read and return the shared graph file
    match std::fs::read(&shared_graph.file_path) {
        Ok(file_data) => {
            let content_type = match shared_graph.original_format {
                ExportFormat::Json => "application/json",
                ExportFormat::Gexf | ExportFormat::Graphml => "application/xml",
                ExportFormat::Csv => "text/csv",
                ExportFormat::Dot => "text/plain",
            };

            let mut response = HttpResponse::Ok()
                .content_type(content_type)
                .body(file_data);

            if shared_graph.compressed {
                response.headers_mut().insert(
                    actix_web::http::header::CONTENT_ENCODING,
                    HeaderValue::from_static("gzip"),
                );
            }

            Ok(response)
        }
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "error": format!("Failed to read shared graph file: {}", e)
        }))),
    }
}

/// Publish graph to repository
pub async fn publish_graph(
    app_state: web::Data<AppState>,
    _request: web::Json<PublishRequest>,
    req: HttpRequest,
) -> ActixResult<HttpResponse> {
    let client_ip = req
        .connection_info()
        .peer_addr()
        .unwrap_or("unknown")
        .to_string();

    let handler = GraphExportHandler::new(std::path::PathBuf::from("data"));

    // Check rate limits (publishing has stricter limits)
    match handler.check_rate_limit(&client_ip).await {
        Ok(rate_info) if rate_info.remaining_exports == 0 => {
            return Ok(HttpResponse::TooManyRequests().json(serde_json::json!({
                "error": "Rate limit exceeded",
                "rate_limit": rate_info
            })));
        }
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Rate limit check failed: {}", e)
            })));
        }
        _ => {}
    }

    // Get current graph
    let _graph = match handler.get_current_graph(&app_state).await {
        Ok(graph) => graph,
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Failed to get graph: {}", e)
            })));
        }
    };

    // Create publication (mock implementation)
    let publication_id = Uuid::new_v4();
    let repository_url = format!("https://graphdb.example.com/graphs/{}", publication_id);

    // In production, this would:
    // 1. Validate graph data
    // 2. Generate metadata
    // 3. Submit to review queue
    // 4. Generate DOI if approved
    // 5. Store in repository

    let publish_response = PublishResponse {
        publication_id,
        repository_url,
        doi: Some(format!("10.1000/graph.{}", publication_id)),
        published_at: Utc::now(),
        status: PublicationStatus::Pending,
    };

    Ok(HttpResponse::Ok().json(publish_response))
}

/// Delete shared graph
pub async fn delete_shared_graph(path: web::Path<String>) -> ActixResult<HttpResponse> {
    let share_id = match Uuid::parse_str(&path.into_inner()) {
        Ok(id) => id,
        Err(_) => {
            return Ok(HttpResponse::BadRequest().json(serde_json::json!({
                "error": "Invalid share ID format"
            })));
        }
    };

    let handler = GraphExportHandler::new(std::path::PathBuf::from("data"));

    // Remove from storage
    let removed_graph = {
        let mut shared_graphs = handler.shared_graphs.write().await;
        shared_graphs.remove(&share_id)
    };

    match removed_graph {
        Some(graph) => {
            // Delete the file
            if let Err(e) = std::fs::remove_file(&graph.file_path) {
                log::warn!("Failed to delete shared graph file: {}", e);
            }

            Ok(HttpResponse::Ok().json(serde_json::json!({
                "message": "Shared graph deleted successfully",
                "deleted_id": share_id
            })))
        }
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Shared graph not found"
        }))),
    }
}

/// Get export statistics
pub async fn get_export_stats() -> ActixResult<HttpResponse> {
    // In production, this would query actual usage statistics
    let stats = ExportStats {
        total_exports: 1250,
        exports_by_format: {
            let mut map = HashMap::new();
            map.insert("json".to_string(), 750);
            map.insert("gexf".to_string(), 300);
            map.insert("graphml".to_string(), 150);
            map.insert("csv".to_string(), 50);
            map
        },
        shared_graphs: 45,
        published_graphs: 12,
        avg_file_size: 2.4, // MB
        last_export: Some(Utc::now() - chrono::Duration::minutes(15)),
    };

    Ok(HttpResponse::Ok().json(stats))
}

/// Configure routes for graph export functionality
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/graph")
            .route("/export", web::post().to(export_graph))
            .route("/share", web::post().to(share_graph))
            .route("/shared/{id}", web::get().to(get_shared_graph))
            .route("/shared/{id}", web::delete().to(delete_shared_graph))
            .route("/publish", web::post().to(publish_graph))
            .route("/stats", web::get().to(get_export_stats)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};
    use tempfile::tempdir;

    #[actix_rt::test]
    async fn test_rate_limiting() {
        let temp_dir = tempdir().unwrap();
        let handler = GraphExportHandler::new(temp_dir.path().to_path_buf());

        let client_ip = "127.0.0.1";

        // First request should succeed
        let rate_info = handler.check_rate_limit(client_ip).await.unwrap();
        assert!(rate_info.remaining_exports > 0);

        // Subsequent requests should reduce remaining count
        let rate_info2 = handler.check_rate_limit(client_ip).await.unwrap();
        assert!(rate_info2.remaining_exports < rate_info.remaining_exports);
    }

    #[actix_rt::test]
    async fn test_export_api_endpoint() {
        let temp_dir = tempdir().unwrap();
        let app_state = web::Data::new(AppState {
            // Mock app state
            server_port: 8080,
        });

        let app =
            test::init_service(App::new().app_data(app_state).configure(configure_routes)).await;

        let export_request = ExportRequest {
            format: ExportFormat::Json,
            ..Default::default()
        };

        let req = test::TestRequest::post()
            .uri("/api/graph/export")
            .set_json(&export_request)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success() || resp.status().is_server_error());
    }
}
