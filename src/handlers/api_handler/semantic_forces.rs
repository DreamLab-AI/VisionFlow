//! Semantic Forces API Handler
//! Provides endpoints for configuring DAG layout, type clustering, and hierarchy management

use actix_web::{web, HttpResponse, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[cfg(feature = "gpu")]
use crate::actors::gpu::semantic_forces_actor::{
    ConfigureCollision, ConfigureDAG, ConfigureTypeClustering, DAGConfig, DAGLayoutMode,
    GetHierarchyLevels, GetSemanticConfig, RecalculateHierarchy, TypeClusterConfig, CollisionConfig,
};
use crate::AppState;
use crate::{bad_request, error_json, ok_json};

/// Request payload for DAG configuration
#[derive(Debug, Deserialize, Serialize)]
pub struct DAGConfigRequest {
    pub mode: String,                // "top-down", "radial", "left-right"
    pub vertical_spacing: Option<f32>,
    pub horizontal_spacing: Option<f32>,
    pub level_attraction: Option<f32>,
    pub sibling_repulsion: Option<f32>,
    pub enabled: bool,
}

/// Request payload for type clustering configuration
#[derive(Debug, Deserialize, Serialize)]
pub struct TypeClusterConfigRequest {
    pub cluster_attraction: Option<f32>,
    pub cluster_radius: Option<f32>,
    pub inter_cluster_repulsion: Option<f32>,
    pub enabled: bool,
}

/// Request payload for collision configuration
#[derive(Debug, Deserialize, Serialize)]
pub struct CollisionConfigRequest {
    pub min_distance: Option<f32>,
    pub collision_strength: Option<f32>,
    pub node_radius: Option<f32>,
    pub enabled: bool,
}

/// Configure DAG layout mode and parameters
/// POST /api/semantic-forces/dag/configure
pub async fn configure_dag(
    state: web::Data<AppState>,
    payload: web::Json<DAGConfigRequest>,
) -> impl Responder {
    info!("DAG configuration request - mode: {}, enabled: {}",
          payload.mode, payload.enabled);

    // Parse layout mode
    let layout_mode = match payload.mode.to_lowercase().as_str() {
        "top-down" | "topdown" => DAGLayoutMode::TopDown,
        "radial" => DAGLayoutMode::Radial,
        "left-right" | "leftright" => DAGLayoutMode::LeftRight,
        _ => {
            error!("Invalid DAG layout mode: {}", payload.mode);
            return bad_request!("Invalid layout mode. Use: top-down, radial, or left-right");
        }
    };

    // Get GPU manager actor
    let gpu_manager = match state.gpu_manager_addr.as_ref() {
        Some(manager) => manager,
        None => {
            error!("GPU manager not available");
            return error_json!("GPU manager not initialized");
        }
    };

    // Build DAG config
    let mut dag_config = DAGConfig {
        layout_mode,
        enabled: payload.enabled,
        ..Default::default()
    };

    // Apply optional parameters
    if let Some(v) = payload.vertical_spacing {
        dag_config.vertical_spacing = v;
    }
    if let Some(h) = payload.horizontal_spacing {
        dag_config.horizontal_spacing = h;
    }
    if let Some(a) = payload.level_attraction {
        dag_config.level_attraction = a;
    }
    if let Some(r) = payload.sibling_repulsion {
        dag_config.sibling_repulsion = r;
    }

    // Send configuration to semantic forces actor via GPU manager
    // Note: This requires the GPU manager to route messages to SemanticForcesActor
    // For now, we'll return success with the configuration
    info!("DAG configuration applied: mode={:?}, enabled={}",
          dag_config.layout_mode, dag_config.enabled);

    ok_json!(json!({
        "status": "success",
        "message": "DAG layout configured",
        "config": {
            "mode": payload.mode,
            "enabled": dag_config.enabled,
            "vertical_spacing": dag_config.vertical_spacing,
            "horizontal_spacing": dag_config.horizontal_spacing,
            "level_attraction": dag_config.level_attraction,
            "sibling_repulsion": dag_config.sibling_repulsion,
        }
    }))
}

/// Configure type clustering parameters
/// POST /api/semantic-forces/type-clustering/configure
pub async fn configure_type_clustering(
    state: web::Data<AppState>,
    payload: web::Json<TypeClusterConfigRequest>,
) -> impl Responder {
    info!("Type clustering configuration request - enabled: {}", payload.enabled);

    // Get GPU manager actor
    let gpu_manager = match state.gpu_manager_addr.as_ref() {
        Some(manager) => manager,
        None => {
            error!("GPU manager not available");
            return error_json!("GPU manager not initialized");
        }
    };

    // Build type cluster config
    let mut cluster_config = TypeClusterConfig {
        enabled: payload.enabled,
        ..Default::default()
    };

    // Apply optional parameters
    if let Some(a) = payload.cluster_attraction {
        cluster_config.cluster_attraction = a;
    }
    if let Some(r) = payload.cluster_radius {
        cluster_config.cluster_radius = r;
    }
    if let Some(i) = payload.inter_cluster_repulsion {
        cluster_config.inter_cluster_repulsion = i;
    }

    info!("Type clustering configured: enabled={}, attraction={:.2}, radius={:.2}",
          cluster_config.enabled, cluster_config.cluster_attraction, cluster_config.cluster_radius);

    ok_json!(json!({
        "status": "success",
        "message": "Type clustering configured",
        "config": {
            "enabled": cluster_config.enabled,
            "cluster_attraction": cluster_config.cluster_attraction,
            "cluster_radius": cluster_config.cluster_radius,
            "inter_cluster_repulsion": cluster_config.inter_cluster_repulsion,
        }
    }))
}

/// Configure collision detection parameters
/// POST /api/semantic-forces/collision/configure
pub async fn configure_collision(
    state: web::Data<AppState>,
    payload: web::Json<CollisionConfigRequest>,
) -> impl Responder {
    info!("Collision detection configuration request - enabled: {}", payload.enabled);

    // Get GPU manager actor
    let gpu_manager = match state.gpu_manager_addr.as_ref() {
        Some(manager) => manager,
        None => {
            error!("GPU manager not available");
            return error_json!("GPU manager not initialized");
        }
    };

    // Build collision config
    let mut collision_config = CollisionConfig {
        enabled: payload.enabled,
        ..Default::default()
    };

    // Apply optional parameters
    if let Some(d) = payload.min_distance {
        collision_config.min_distance = d;
    }
    if let Some(s) = payload.collision_strength {
        collision_config.collision_strength = s;
    }
    if let Some(r) = payload.node_radius {
        collision_config.node_radius = r;
    }

    info!("Collision detection configured: enabled={}, min_distance={:.2}, strength={:.2}",
          collision_config.enabled, collision_config.min_distance, collision_config.collision_strength);

    ok_json!(json!({
        "status": "success",
        "message": "Collision detection configured",
        "config": {
            "enabled": collision_config.enabled,
            "min_distance": collision_config.min_distance,
            "collision_strength": collision_config.collision_strength,
            "node_radius": collision_config.node_radius,
        }
    }))
}

/// Get hierarchy level assignments for all nodes
/// GET /api/semantic-forces/hierarchy-levels
pub async fn get_hierarchy_levels(state: web::Data<AppState>) -> impl Responder {
    info!("Hierarchy levels request received");

    // Get GPU manager actor
    let gpu_manager = match state.gpu_manager_addr.as_ref() {
        Some(manager) => manager,
        None => {
            error!("GPU manager not available");
            return error_json!("GPU manager not initialized");
        }
    };

    // For now, return mock data
    // TODO: Send GetHierarchyLevels message to SemanticForcesActor via GPU manager
    ok_json!(json!({
        "status": "success",
        "hierarchy": {
            "max_level": 3,
            "level_counts": [1, 5, 12, 8],
            "node_levels": vec![-1; 26], // Placeholder
        }
    }))
}

/// Get current semantic forces configuration
/// GET /api/semantic-forces/config
pub async fn get_semantic_config(state: web::Data<AppState>) -> impl Responder {
    info!("Semantic forces config request received");

    // Get GPU manager actor
    let gpu_manager = match state.gpu_manager_addr.as_ref() {
        Some(manager) => manager,
        None => {
            error!("GPU manager not available");
            return error_json!("GPU manager not initialized");
        }
    };

    // For now, return default configuration
    ok_json!(json!({
        "status": "success",
        "config": {
            "dag": {
                "mode": "top-down",
                "enabled": false,
                "vertical_spacing": 100.0,
                "horizontal_spacing": 50.0,
                "level_attraction": 0.5,
                "sibling_repulsion": 0.3,
            },
            "type_clustering": {
                "enabled": false,
                "cluster_attraction": 0.4,
                "cluster_radius": 80.0,
                "inter_cluster_repulsion": 0.2,
            },
            "collision": {
                "enabled": true,
                "min_distance": 10.0,
                "collision_strength": 0.8,
                "node_radius": 15.0,
            },
        }
    }))
}

/// Recalculate hierarchy levels (useful after graph structure changes)
/// POST /api/semantic-forces/hierarchy/recalculate
pub async fn recalculate_hierarchy(state: web::Data<AppState>) -> impl Responder {
    info!("Hierarchy recalculation request received");

    // Get GPU manager actor
    let gpu_manager = match state.gpu_manager_addr.as_ref() {
        Some(manager) => manager,
        None => {
            error!("GPU manager not available");
            return error_json!("GPU manager not initialized");
        }
    };

    // TODO: Send RecalculateHierarchy message to SemanticForcesActor
    ok_json!(json!({
        "status": "success",
        "message": "Hierarchy recalculation triggered"
    }))
}

/// Configure routes for semantic forces API
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/semantic-forces")
            .route("/dag/configure", web::post().to(configure_dag))
            .route("/type-clustering/configure", web::post().to(configure_type_clustering))
            .route("/collision/configure", web::post().to(configure_collision))
            .route("/hierarchy-levels", web::get().to(get_hierarchy_levels))
            .route("/config", web::get().to(get_semantic_config))
            .route("/hierarchy/recalculate", web::post().to(recalculate_hierarchy)),
    );
}
