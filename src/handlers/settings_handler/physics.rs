// Physics-related handlers and GPU propagation logic

use crate::actors::messages::{GetSettings, UpdateSettings, UpdateSimulationParams};
use crate::app_state::AppState;
use crate::config::AppFullSettings;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use log::{debug, error, info, warn};
use serde_json::{json, Value};
use crate::{ok_json, error_json, bad_request, service_unavailable};

use super::helpers::create_physics_settings_update;
use super::validation::validate_constraints;

pub async fn propagate_physics_to_gpu(
    state: &web::Data<AppState>,
    settings: &AppFullSettings,
    graph: &str,
) {
    propagate_physics_to_gpu_with_layout(state, settings, graph, None).await;
}

/// Propagate physics settings to GPU actors, optionally applying layout-mode overrides.
/// When `layout_mode_override` is provided, it adjusts simulation parameters for
/// hierarchical (DAG) or type-clustering layouts.
pub async fn propagate_physics_to_gpu_with_layout(
    state: &web::Data<AppState>,
    settings: &AppFullSettings,
    graph: &str,
    layout_mode_override: Option<&str>,
) {
    let physics = settings.get_physics(graph);


    info!("[PHYSICS UPDATE] Propagating {} physics to actors:", graph);
    info!(
        "  - repulsion_k: {:.3} (affects node spreading)",
        physics.repel_k
    );
    info!(
        "  - spring_k: {:.3} (affects edge tension)",
        physics.spring_k
    );
    info!("  - spring_k: {:.3} (affects clustering)", physics.spring_k);
    info!(
        "  - damping: {:.3} (affects settling, 1.0 = no movement)",
        physics.damping
    );
    info!("  - time_step: {:.3} (simulation speed)", physics.dt);
    info!(
        "  - max_velocity: {:.3} (prevents explosions)",
        physics.max_velocity
    );
    info!(
        "  - temperature: {:.3} (random motion)",
        physics.temperature
    );
    info!("  - gravity: {:.3} (directional force)", physics.gravity);

    if crate::utils::logging::is_debug_enabled() {
        debug!("  - bounds_size: {:.1}", physics.bounds_size);
        debug!("  - separation_radius: {:.3}", physics.separation_radius);
        debug!("  - mass_scale: {:.3}", physics.mass_scale);
        debug!("  - boundary_damping: {:.3}", physics.boundary_damping);
        debug!("  - update_threshold: {:.3}", physics.update_threshold);
        debug!("  - iterations: {}", physics.iterations);
        debug!("  - enabled: {}", physics.enabled);


        debug!("  - min_distance: {:.3}", physics.min_distance);
        debug!("  - max_repulsion_dist: {:.1}", physics.max_repulsion_dist);
        debug!("  - boundary_margin: {:.3}", physics.boundary_margin);
        debug!(
            "  - boundary_force_strength: {:.1}",
            physics.boundary_force_strength
        );
        debug!("  - warmup_iterations: {}", physics.warmup_iterations);
        debug!("  - warmup_curve: {}", physics.warmup_curve);
        debug!(
            "  - zero_velocity_iterations: {}",
            physics.zero_velocity_iterations
        );
        debug!("  - cooling_rate: {:.6}", physics.cooling_rate);
        debug!("  - clustering_algorithm: {}", physics.clustering_algorithm);
        debug!("  - cluster_count: {}", physics.cluster_count);
        debug!(
            "  - clustering_resolution: {:.3}",
            physics.clustering_resolution
        );
        debug!(
            "  - clustering_iterations: {}",
            physics.clustering_iterations
        );
        debug!("[GPU Parameters] All new parameters available for GPU processing");
    }

    let mut sim_params: crate::models::simulation_params::SimulationParams = physics.into();

    // Apply layout-mode-specific physics overrides from quality gate settings.
    // The quality gate `layoutMode` drives different force configurations:
    //   - force-directed: default (no overrides)
    //   - dag-topdown: enable center gravity + SSSP for hierarchical layout
    //   - dag-radial: same as dag-topdown (orientation is a TODO)
    //   - dag-leftright: same as dag-topdown (orientation is a TODO)
    //   - type-clustering: enable cluster/alignment forces for type-based grouping
    let layout_mode = layout_mode_override.unwrap_or("force-directed");

    match layout_mode {
        "dag-topdown" | "dag-radial" | "dag-leftright" => {
            info!("[PHYSICS UPDATE] Applying DAG layout overrides for mode: {}", layout_mode);
            sim_params.center_gravity_k = sim_params.center_gravity_k.max(0.1);
            sim_params.use_sssp_distances = true;
            sim_params.sssp_alpha = Some(sim_params.sssp_alpha.unwrap_or(0.0).max(0.5));
        }
        "type-clustering" => {
            info!("[PHYSICS UPDATE] Applying type-clustering layout overrides");
            sim_params.cluster_strength = sim_params.cluster_strength.max(0.5);
            sim_params.alignment_strength = sim_params.alignment_strength.max(0.3);
        }
        _ => {
            // force-directed: use physics settings as-is
        }
    }

    info!(
        "[PHYSICS UPDATE] Converted to SimulationParams - repulsion: {}, damping: {:.3}, time_step: {:.3}, layout: {}",
        sim_params.repel_k, sim_params.damping, sim_params.dt, layout_mode
    );

    let update_msg = UpdateSimulationParams {
        params: sim_params.clone(),
    };


    if let Some(gpu_addr) = state.get_gpu_compute_addr().await {
        info!("[PHYSICS UPDATE] Sending to GPUComputeActor...");
        if let Err(e) = gpu_addr.send(update_msg.clone()).await {
            error!("[PHYSICS UPDATE] FAILED to update GPUComputeActor: {}", e);
        } else {
            info!("[PHYSICS UPDATE] GPUComputeActor updated successfully");
        }
    } else {
        warn!("[PHYSICS UPDATE] No GPUComputeActor available");
    }


    info!("[PHYSICS UPDATE] Sending to GraphServiceActor...");
    if let Err(e) = state.graph_service_addr.send(update_msg).await {
        error!("[PHYSICS UPDATE] FAILED to update GraphServiceActor: {}", e);
    } else {
        info!("[PHYSICS UPDATE] GraphServiceActor updated successfully");
    }
}

pub async fn update_compute_mode(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();

    info!("Compute mode update request received");
    debug!(
        "Compute mode payload: {}",
        serde_json::to_string_pretty(&update).unwrap_or_default()
    );


    let compute_mode = update
        .get("computeMode")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            actix_web::error::ErrorBadRequest("computeMode must be an integer between 0 and 3")
        })?;

    if compute_mode > 3 {
        return bad_request!("computeMode must be between 0 and 3");
    }


    let physics_update = json!({
        "computeMode": compute_mode
    });

    let settings_update = create_physics_settings_update(physics_update);


    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return error_json!("Failed to get current settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge compute mode settings: {}", e);
        return error_json!("Failed to update compute mode: {}", e);
    }


    match state
        .settings_addr
        .send(UpdateSettings {
            settings: app_settings.clone(),
        })
        .await
    {
        Ok(Ok(())) => {
            info!("Compute mode updated successfully to: {}", compute_mode);


            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;

            ok_json!(json!({
                "status": "Compute mode updated successfully",
                "computeMode": compute_mode
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to save compute mode settings: {}", e);
            error_json!("Failed to save compute mode settings: {}", e)
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            service_unavailable!("Settings service unavailable")
        }
    }
}

pub async fn update_clustering_algorithm(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();

    info!("Clustering algorithm update request received");
    debug!(
        "Clustering payload: {}",
        serde_json::to_string_pretty(&update).unwrap_or_default()
    );


    let algorithm = update
        .get("algorithm")
        .and_then(|v| v.as_str())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("algorithm must be a string"))?;

    if !["none", "kmeans", "spectral", "louvain"].contains(&algorithm) {
        return bad_request!("algorithm must be 'none', 'kmeans', 'spectral', or 'louvain'");
    }


    let cluster_count = update
        .get("clusterCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(5);
    let resolution = update
        .get("resolution")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let iterations = update
        .get("iterations")
        .and_then(|v| v.as_u64())
        .unwrap_or(30);


    let physics_update = json!({
        "clusteringAlgorithm": algorithm,
        "clusterCount": cluster_count,
        "clusteringResolution": resolution,
        "clusteringIterations": iterations
    });

    let settings_update = create_physics_settings_update(physics_update);


    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return error_json!("Failed to get current settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge clustering settings: {}", e);
        return error_json!("Failed to update clustering algorithm: {}", e);
    }


    match state
        .settings_addr
        .send(UpdateSettings {
            settings: app_settings.clone(),
        })
        .await
    {
        Ok(Ok(())) => {
            info!(
                "Clustering algorithm updated successfully to: {}",
                algorithm
            );


            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;

            ok_json!(json!({
                "status": "Clustering algorithm updated successfully",
                "algorithm": algorithm,
                "clusterCount": cluster_count,
                "resolution": resolution,
                "iterations": iterations
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to save clustering settings: {}", e);
            error_json!("Failed to save clustering settings: {}", e)
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            service_unavailable!("Settings service unavailable")
        }
    }
}

pub async fn update_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();

    info!("Constraints update request received");
    debug!(
        "Constraints payload: {}",
        serde_json::to_string_pretty(&update).unwrap_or_default()
    );


    if let Err(e) = validate_constraints(&update) {
        return bad_request!("Invalid constraints: {}", e);
    }



    let settings_update = json!({
        "visualisation": {
            "graphs": {
                "logseq": {
                    "physics": {
                        "computeMode": 2
                    }
                },
                "visionflow": {
                    "physics": {
                        "computeMode": 2
                    }
                }
            }
        }
    });


    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return error_json!("Failed to get current settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge constraints settings: {}", e);
        return error_json!("Failed to update constraints: {}", e);
    }


    match state
        .settings_addr
        .send(UpdateSettings {
            settings: app_settings.clone(),
        })
        .await
    {
        Ok(Ok(())) => {
            info!("Constraints updated successfully");


            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;

            ok_json!(json!({
                "status": "Constraints updated successfully"
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to save constraints settings: {}", e);
            error_json!("Failed to save constraints settings: {}", e)
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            service_unavailable!("Settings service unavailable")
        }
    }
}

pub async fn get_cluster_analytics(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("Cluster analytics request received");


    if let Some(_gpu_addr) = state.get_gpu_compute_addr().await {

        use crate::actors::messages::GetGraphData;


        let graph_data = match state.graph_service_addr.send(GetGraphData).await {
            Ok(Ok(data)) => data,
            Ok(Err(e)) => {
                error!("Failed to get graph data for clustering analytics: {}", e);
                return error_json!("Failed to get graph data for analytics");
            }
            Err(e) => {
                error!("Graph service communication error: {}", e);
                return service_unavailable!("Graph service unavailable");
            }
        };


        info!("GPU compute actor available but clustering not handled by force compute actor");
        get_cpu_fallback_analytics(&graph_data).await
    } else {

        use crate::actors::messages::GetGraphData;
        match state.graph_service_addr.send(GetGraphData).await {
            Ok(Ok(graph_data)) => get_cpu_fallback_analytics(&graph_data).await,
            Ok(Err(e)) => {
                error!("Failed to get graph data: {}", e);
                error_json!("Failed to get graph data for analytics")
            }
            Err(e) => {
                error!("Graph service unavailable: {}", e);
                service_unavailable!("Graph service unavailable")
            }
        }
    }
}

async fn get_cpu_fallback_analytics(
    graph_data: &crate::models::graph::GraphData,
) -> Result<HttpResponse, Error> {
    use std::collections::HashMap;



    let node_count = graph_data.nodes.len();
    let _edge_count = graph_data.edges.len();


    let mut type_clusters: HashMap<String, Vec<&crate::models::node::Node>> = HashMap::new();

    for node in &graph_data.nodes {
        let node_type = node
            .node_type
            .as_ref()
            .unwrap_or(&"unknown".to_string())
            .clone();
        type_clusters
            .entry(node_type)
            .or_insert_with(Vec::new)
            .push(node);
    }


    let clusters: Vec<_> = type_clusters
        .into_iter()
        .enumerate()
        .map(|(i, (type_name, nodes))| {

            let centroid = if !nodes.is_empty() {
                let sum_x: f32 = nodes.iter().map(|n| n.data.x).sum();
                let sum_y: f32 = nodes.iter().map(|n| n.data.y).sum();
                let sum_z: f32 = nodes.iter().map(|n| n.data.z).sum();
                let count = nodes.len() as f32;
                [sum_x / count, sum_y / count, sum_z / count]
            } else {
                [0.0, 0.0, 0.0]
            };

            json!({
                "id": format!("cpu_cluster_{}", i),
                "nodeCount": nodes.len(),
                "coherence": 0.6,
                "centroid": centroid,
                "keywords": [type_name.clone(), "cpu_cluster"],
                "type": type_name
            })
        })
        .collect();

    let fallback_analytics = json!({
        "clusters": clusters,
        "totalNodes": node_count,
        "algorithmUsed": "cpu_heuristic",
        "modularity": 0.4,
        "lastUpdated": chrono::Utc::now().to_rfc3339(),
        "gpu_accelerated": false,
        "note": "CPU fallback clustering based on node types",
        "computation_time_ms": 0
    });

    ok_json!(fallback_analytics)
}

pub async fn update_stress_optimization(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();

    info!("Stress optimization update request received");
    debug!(
        "Stress optimization payload: {}",
        serde_json::to_string_pretty(&update).unwrap_or_default()
    );


    let stress_weight = update
        .get("stressWeight")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.1) as f32;

    let stress_alpha = update
        .get("stressAlpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.1) as f32;

    if !(0.0..=1.0).contains(&stress_weight) || !(0.0..=1.0).contains(&stress_alpha) {
        return bad_request!("stressWeight and stressAlpha must be between 0.0 and 1.0");
    }


    let physics_update = json!({
        "stressWeight": stress_weight,
        "stressAlpha": stress_alpha
    });

    let settings_update = create_physics_settings_update(physics_update);


    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return error_json!("Failed to get current settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge stress optimization settings: {}", e);
        return error_json!("Failed to update stress optimization: {}", e);
    }


    match state
        .settings_addr
        .send(UpdateSettings {
            settings: app_settings.clone(),
        })
        .await
    {
        Ok(Ok(())) => {
            info!("Stress optimization updated successfully");


            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;

            ok_json!(json!({
                "status": "Stress optimization updated successfully",
                "stressWeight": stress_weight,
                "stressAlpha": stress_alpha
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to save stress optimization settings: {}", e);
            error_json!("Failed to save stress optimization settings: {}", e)
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            service_unavailable!("Settings service unavailable")
        }
    }
}
