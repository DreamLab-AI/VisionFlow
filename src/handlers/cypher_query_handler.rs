// src/handlers/cypher_query_handler.rs
//! Cypher Query Handler
//!
//! Provides REST API endpoints for executing Cypher queries against Neo4j.
//! Enables complex graph analytics and multi-hop reasoning.
//!
//! Safety features:
//! - Query timeout limits
//! - Result size limits
//! - Parameterized queries to prevent injection
//! - Read-only by default

use actix_web::{web, HttpResponse, Responder};
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use crate::utils::response_macros::*;

use crate::adapters::neo4j_adapter::Neo4jAdapter;

// Response macros - Task 1.4 HTTP Standardization
use crate::{ok_json, error_json, bad_request, not_found, created_json};
use crate::utils::handler_commons::HandlerResponse;


/// Cypher query request
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CypherQueryRequest {
    /// Cypher query string
    pub query: String,

    /// Query parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,

    /// Maximum number of results to return (default: 100, max: 10000)
    #[serde(default = "default_limit")]
    pub limit: usize,

    /// Query timeout in seconds (default: 30, max: 300)
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_limit() -> usize {
    100
}

fn default_timeout() -> u64 {
    30
}

/// Cypher query response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CypherQueryResponse {
    /// Query results
    pub results: Vec<HashMap<String, serde_json::Value>>,

    /// Number of results returned
    pub count: usize,

    /// Whether results were truncated due to limit
    pub truncated: bool,

    /// Query execution time in milliseconds
    pub execution_time_ms: u128,
}

/// Error response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

/// Execute a Cypher query
///
/// POST /api/query/cypher
///
/// # Safety Features
/// - Query timeouts
/// - Result size limits
/// - Parameterized queries
/// - Read-only enforcement (optional)
///
/// # Example
/// ```json
/// {
///   "query": "MATCH (n:GraphNode)-[:EDGE*1..3]-(m:GraphNode) WHERE n.id = $start_id RETURN m.label, m.owl_class_iri LIMIT 10",
///   "parameters": {"start_id": 42},
///   "limit": 100,
///   "timeout": 30
/// }
/// ```
pub async fn execute_cypher_query(
    neo4j: web::Data<Arc<Neo4jAdapter>>,
    request: web::Json<CypherQueryRequest>,
) -> impl Responder {
    let start_time = std::time::Instant::now();

    // Validate limits
    let limit = request.limit.min(10000); // Max 10k results
    let timeout = request.timeout.min(300); // Max 5 minutes

    // Check for unsafe operations (optional safety check)
    let query_upper = request.query.to_uppercase();
    if query_upper.contains("DELETE") || query_upper.contains("REMOVE") ||
       query_upper.contains("SET") || query_upper.contains("CREATE") ||
       query_upper.contains("MERGE") {
        warn!("🚨 Attempted to execute write operation via Cypher query endpoint");
        return HttpResponse::Forbidden().json(ErrorResponse {
            error: "Forbidden".to_string(),
            message: "Write operations are not allowed via this endpoint. Use dedicated mutation endpoints.".to_string(),
        });
    }

    // Add LIMIT clause if not present
    let query_with_limit = if !query_upper.contains("LIMIT") {
        format!("{} LIMIT {}", request.query, limit + 1) // +1 to detect truncation
    } else {
        request.query.clone()
    };
use crate::utils::json::{from_json, to_json};

    debug!("Executing Cypher query: {}", query_with_limit);

    // Convert parameters to Neo4j BoltType
    let mut bolt_params = HashMap::new();
    for (key, value) in &request.parameters {
        // Convert serde_json::Value to neo4rs::BoltType
        match value {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    bolt_params.insert(key.clone(), neo4rs::BoltType::Integer(neo4rs::BoltInteger::new(i)));
                } else if let Some(f) = n.as_f64() {
                    bolt_params.insert(key.clone(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(f)));
                }
            }
            serde_json::Value::String(s) => {
                bolt_params.insert(key.clone(), neo4rs::BoltType::String(neo4rs::BoltString::from(s.clone())));
            }
            serde_json::Value::Bool(b) => {
                bolt_params.insert(key.clone(), neo4rs::BoltType::Boolean(neo4rs::BoltBoolean::new(*b)));
            }
            serde_json::Value::Null => {
                bolt_params.insert(key.clone(), neo4rs::BoltType::Null(neo4rs::BoltNull));
            }
            _ => {
                // Complex types - serialize to string
                if let Ok(json_str) = to_json(value) {
                    bolt_params.insert(key.clone(), neo4rs::BoltType::String(neo4rs::BoltString::from(json_str)));
                }
            }
        }
    }

    // Execute query with timeout
    match tokio::time::timeout(
        std::time::Duration::from_secs(timeout),
        neo4j.execute_cypher(&query_with_limit, bolt_params)
    ).await {
        Ok(Ok(mut results)) => {
            let truncated = results.len() > limit;
            if truncated {
                results.truncate(limit);
            }

            let count = results.len();
            let execution_time_ms = start_time.elapsed().as_millis();

            debug!("✅ Cypher query executed successfully: {} results in {}ms", count, execution_time_ms);

            ok_json!(CypherQueryResponse {
                results,
                count,
                truncated,
                execution_time_ms,
            })
        }
        Ok(Err(e)) => {
            warn!("❌ Cypher query failed: {}", e);
            bad_request!("Internal server error"),
                message: format!("Failed to execute query: {}", e),
            })
        }
        Err(_) => {
            warn!("⏱️  Cypher query timeout after {}s", timeout);
            HttpResponse::RequestTimeout().json(ErrorResponse {
                error: "Timeout".to_string(),
                message: format!("Query exceeded timeout of {} seconds", timeout),
            })
        }
    }
}

/// Get example Cypher queries
///
/// GET /api/query/cypher/examples
///
/// Returns a list of common Cypher query patterns for reference
pub async fn get_cypher_examples() -> impl Responder {
    let examples = vec![
        CypherExample {
            title: "Find all neighbors of a node".to_string(),
            description: "Get all directly connected nodes".to_string(),
            query: "MATCH (n:GraphNode {id: $node_id})-[:EDGE]-(m:GraphNode) RETURN m".to_string(),
            parameters: vec!["node_id (integer)".to_string()],
        },
        CypherExample {
            title: "Multi-hop path analysis".to_string(),
            description: "Find nodes within 3 hops of a starting node".to_string(),
            query: "MATCH (n:GraphNode {id: $node_id})-[:EDGE*1..3]-(m:GraphNode) RETURN DISTINCT m.id, m.label".to_string(),
            parameters: vec!["node_id (integer)".to_string()],
        },
        CypherExample {
            title: "Shortest path between two nodes".to_string(),
            description: "Find the shortest path connecting two nodes".to_string(),
            query: "MATCH p=shortestPath((n:GraphNode {id: $start_id})-[:EDGE*]-(m:GraphNode {id: $end_id})) RETURN p, length(p) AS hops".to_string(),
            parameters: vec!["start_id (integer)".to_string(), "end_id (integer)".to_string()],
        },
        CypherExample {
            title: "Nodes by OWL class".to_string(),
            description: "Find all nodes with a specific OWL class IRI".to_string(),
            query: "MATCH (n:GraphNode {owl_class_iri: $iri}) RETURN n.id, n.label, n.metadata".to_string(),
            parameters: vec!["iri (string)".to_string()],
        },
        CypherExample {
            title: "High-degree nodes (hubs)".to_string(),
            description: "Find nodes with the most connections".to_string(),
            query: "MATCH (n:GraphNode)-[r:EDGE]-() WITH n, count(r) AS degree ORDER BY degree DESC LIMIT $limit RETURN n.id, n.label, degree".to_string(),
            parameters: vec!["limit (integer)".to_string()],
        },
        CypherExample {
            title: "Semantic path by OWL properties".to_string(),
            description: "Traverse edges with specific OWL property IRIs".to_string(),
            query: "MATCH (n:GraphNode {id: $start_id})-[r:EDGE*1..5]->(m:GraphNode) WHERE ALL(rel IN r WHERE rel.owl_property_iri = $property_iri) RETURN m.id, m.label".to_string(),
            parameters: vec!["start_id (integer)".to_string(), "property_iri (string)".to_string()],
        },
        CypherExample {
            title: "Cluster detection".to_string(),
            description: "Find densely connected subgraphs".to_string(),
            query: "MATCH (n:GraphNode)-[:EDGE]-(m:GraphNode) WHERE n.group_name = $group RETURN n, m".to_string(),
            parameters: vec!["group (string)".to_string()],
        },
    ];

    ok_json!(examples)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CypherExample {
    title: String,
    description: String,
    query: String,
    parameters: Vec<String>,
}

/// Configure Cypher query routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/api/query/cypher")
            .route(web::post().to(execute_cypher_query))
    )
    .service(
        web::resource("/api/query/cypher/examples")
            .route(web::get().to(get_cypher_examples))
    );
}
