use actix::prelude::*;
use actix_web_actors::ws;
use log::{debug, info, warn};

use crate::utils::binary_protocol;
use crate::utils::socket_flow_messages::{BinaryNodeData, BinaryNodeDataClient};
use crate::utils::validation::rate_limit::EndpointRateLimits;

use super::types::SocketFlowServer;

/// Fetch nodes from the graph service for streaming position updates.
pub(crate) async fn fetch_nodes(
    app_state: std::sync::Arc<crate::app_state::AppState>,
    _settings_addr: actix::Addr<crate::actors::optimized_settings_actor::OptimizedSettingsActor>,
) -> Option<(Vec<(u32, BinaryNodeData)>, bool)> {
    use crate::actors::messages::GetGraphData;
    use log::error;

    let graph_data = match app_state.graph_service_addr.send(GetGraphData).await {
        Ok(Ok(data)) => data,
        Ok(Err(e)) => {
            error!("[WebSocket] Failed to get graph data: {}", e);
            return None;
        }
        Err(e) => {
            error!(
                "[WebSocket] Failed to send message to GraphServiceActor: {}",
                e
            );
            return None;
        }
    };

    if graph_data.nodes.is_empty() {
        debug!("[WebSocket] No nodes to send! Empty graph data.");
        return None;
    }

    let debug_enabled = crate::utils::logging::is_debug_enabled();
    let debug_websocket = debug_enabled;
    let detailed_debug = debug_enabled && debug_websocket;

    if detailed_debug {
        debug!(
            "Raw nodes count: {}, showing first 5 nodes IDs:",
            graph_data.nodes.len()
        );
        for (i, node) in graph_data.nodes.iter().take(5).enumerate() {
            debug!(
                "  Node {}: id={} (numeric), metadata_id={} (filename)",
                i, node.id, node.metadata_id
            );
        }
    }

    let mut nodes = Vec::with_capacity(graph_data.nodes.len());
    for node in &graph_data.nodes {
        let node_id = node.id;
        let node_data =
            BinaryNodeDataClient::new(node_id, node.data.position(), node.data.velocity());
        nodes.push((node_id, node_data));
    }

    if nodes.is_empty() {
        return None;
    }

    Some((nodes, detailed_debug))
}

pub(crate) fn handle_request_full_snapshot(
    _act: &mut SocketFlowServer,
    msg: &serde_json::Value,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client requested full position snapshot");

    let graphs = msg.get("graphs").and_then(|g| g.as_array());
    let include_knowledge = graphs.map_or(true, |arr| arr.iter().any(|v| v.as_str() == Some("knowledge")));
    let include_agent = graphs.map_or(true, |arr| arr.iter().any(|v| v.as_str() == Some("agent")));

    let fut = async move {
        debug!(
            "RequestPositionSnapshot: include_knowledge={}, include_agent={}",
            include_knowledge, include_agent
        );
        crate::actors::messages::PositionSnapshot {
            knowledge_nodes: Vec::new(),
            agent_nodes: Vec::new(),
            timestamp: std::time::Instant::now(),
        }
    };

    let fut = actix::fut::wrap_future::<_, SocketFlowServer>(fut);
    ctx.spawn(fut.map(move |snapshot, _act, ctx| {
        let mut all_nodes = Vec::new();

        for (id, data) in snapshot.knowledge_nodes {
            all_nodes.push((binary_protocol::set_knowledge_flag(id), data));
        }

        for (id, data) in snapshot.agent_nodes {
            all_nodes.push((binary_protocol::set_agent_flag(id), data));
        }

        if !all_nodes.is_empty() {
            let binary_data = binary_protocol::encode_node_data(&all_nodes);
            ctx.binary(binary_data);
            info!("Sent position snapshot with {} nodes", all_nodes.len());
        }
    }));
}

pub(crate) fn handle_request_initial_data(
    act: &mut SocketFlowServer,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client requested initial data - unified init flow expects REST call first");

    let response = serde_json::json!({
        "type": "initialDataInfo",
        "message": "Please call REST endpoint /api/graph/data first, which will trigger WebSocket sync",
        "flow": "unified_init",
        "timestamp": chrono::Utc::now().timestamp_millis()
    });

    if let Ok(msg_str) = serde_json::to_string(&response) {
        act.last_activity = std::time::Instant::now();
        ctx.text(msg_str);
    }
}

pub(crate) fn handle_enable_randomization(msg: &serde_json::Value) {
    let enabled = msg.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false);
    info!(
        "Client requested to {} node position randomization (server-side removed, client-side used instead)",
        if enabled { "enable" } else { "disable" }
    );
}

pub(crate) fn handle_request_bots_graph(
    act: &mut SocketFlowServer,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client requested bots graph - returning optimized position data only");

    let graph_addr = act.app_state.graph_service_addr.clone();

    ctx.spawn(
        actix::fut::wrap_future::<_, SocketFlowServer>(async move {
            use crate::actors::messages::GetBotsGraphData;
            match graph_addr.send(GetBotsGraphData).await {
                Ok(Ok(graph_data)) => Some(graph_data),
                _ => None,
            }
        })
        .map(|graph_data_opt, _act, ctx| {
            if let Some(graph_data) = graph_data_opt {
                let minimal_nodes: Vec<serde_json::Value> = graph_data
                    .nodes
                    .iter()
                    .map(|node| {
                        serde_json::json!({
                            "id": node.id,
                            "metadata_id": node.metadata_id,
                            "x": node.data.x,
                            "y": node.data.y,
                            "z": node.data.z,
                            "vx": node.data.vx,
                            "vy": node.data.vy,
                            "vz": node.data.vz
                        })
                    })
                    .collect();

                let minimal_edges: Vec<serde_json::Value> = graph_data
                    .edges
                    .iter()
                    .map(|edge| {
                        serde_json::json!({
                            "id": edge.id,
                            "source": edge.source,
                            "target": edge.target,
                            "weight": edge.weight
                        })
                    })
                    .collect();

                let response = serde_json::json!({
                    "type": "botsGraphUpdate",
                    "data": {
                        "nodes": minimal_nodes,
                        "edges": minimal_edges,
                    },
                    "meta": {
                        "optimized": true,
                        "message": "This response contains only position data. For full agent details:",
                        "api_endpoints": {
                            "full_agent_data": "/api/bots/data",
                            "agent_status": "/api/bots/status",
                            "individual_agent": "/api/agents/{id}"
                        }
                    },
                    "timestamp": chrono::Utc::now().timestamp_millis()
                });

                if let Ok(msg_str) = serde_json::to_string(&response) {
                    let original_size = graph_data.nodes.len() * 500;
                    let optimized_size = msg_str.len();
                    info!(
                        "Sending optimized bots graph: {} nodes, {} edges ({} bytes, est. {}% reduction)",
                        minimal_nodes.len(),
                        minimal_edges.len(),
                        optimized_size,
                        if original_size > 0 {
                            100 - (optimized_size * 100 / original_size)
                        } else {
                            0
                        }
                    );
                    ctx.text(msg_str);
                }
            } else {
                warn!("No bots graph data available");
                let response = serde_json::json!({
                    "type": "botsGraphUpdate",
                    "error": "No data available",
                    "meta": {
                        "api_endpoints": {
                            "full_agent_data": "/api/bots/data",
                            "agent_status": "/api/bots/status"
                        }
                    },
                    "timestamp": chrono::Utc::now().timestamp_millis()
                });
                if let Ok(msg_str) = serde_json::to_string(&response) {
                    ctx.text(msg_str);
                }
            }
        }),
    );
}

pub(crate) fn handle_request_bots_positions(
    act: &mut SocketFlowServer,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client requested bots position updates");

    let app_state = act.app_state.clone();

    ctx.spawn(
        actix::fut::wrap_future::<_, SocketFlowServer>(async move {
            let bots_nodes =
                crate::handlers::bots_handler::get_bots_positions(&app_state.bots_client).await;

            if bots_nodes.is_empty() {
                return vec![];
            }

            let mut nodes_data = Vec::new();
            for node in bots_nodes {
                let node_data = BinaryNodeData {
                    node_id: node.id,
                    x: node.data.x,
                    y: node.data.y,
                    z: node.data.z,
                    vx: node.data.vx,
                    vy: node.data.vy,
                    vz: node.data.vz,
                };
                nodes_data.push((node.id, node_data));
            }

            nodes_data
        })
        .map(|nodes_data, _act, ctx| {
            if !nodes_data.is_empty() {
                let binary_data = binary_protocol::encode_node_data(&nodes_data);

                info!(
                    "Sending bots positions: {} nodes, {} bytes",
                    nodes_data.len(),
                    binary_data.len()
                );

                ctx.binary(binary_data);
            }
        }),
    );

    let response = serde_json::json!({
        "type": "botsUpdatesStarted",
        "timestamp": chrono::Utc::now().timestamp_millis()
    });
    if let Ok(msg_str) = serde_json::to_string(&response) {
        ctx.text(msg_str);
    }
}

pub(crate) fn handle_subscribe_position_updates(
    act: &mut SocketFlowServer,
    msg: &serde_json::Value,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client requested position update subscription");

    let interval = msg
        .get("data")
        .and_then(|data| data.get("interval"))
        .and_then(|interval| interval.as_u64())
        .unwrap_or(60);

    let binary = msg
        .get("data")
        .and_then(|data| data.get("binary"))
        .and_then(|binary| binary.as_bool())
        .unwrap_or(true);

    let min_allowed_interval =
        1000 / (EndpointRateLimits::socket_flow_updates().requests_per_minute / 60);
    let actual_interval = interval.max(min_allowed_interval as u64);

    if actual_interval != interval {
        info!(
            "Adjusted position update interval from {}ms to {}ms to comply with rate limits",
            interval, actual_interval
        );
    }

    info!(
        "Starting position updates with interval: {}ms, binary: {}",
        actual_interval, binary
    );

    let update_interval = std::time::Duration::from_millis(actual_interval);
    let app_state = act.app_state.clone();
    let settings_addr = act.app_state.settings_addr.clone();

    let response = serde_json::json!({
        "type": "subscription_confirmed",
        "subscription": "position_updates",
        "interval": actual_interval,
        "binary": binary,
        "timestamp": chrono::Utc::now().timestamp_millis(),
        "rate_limit": {
            "requests_per_minute": EndpointRateLimits::socket_flow_updates().requests_per_minute,
            "min_interval_ms": min_allowed_interval
        }
    });
    if let Ok(msg_str) = serde_json::to_string(&response) {
        ctx.text(msg_str);
    }

    ctx.run_later(update_interval, move |_act, ctx| {
        let fut = fetch_nodes(app_state.clone(), settings_addr.clone());
        let fut = actix::fut::wrap_future::<_, SocketFlowServer>(fut);

        ctx.spawn(fut.map(move |result, act, ctx| {
            if let Some((nodes, detailed_debug)) = result {
                let mut filtered_nodes = Vec::new();
                for (node_id, node_data) in &nodes {
                    let node_id_str = node_id.to_string();
                    let position = node_data.position();
                    let velocity = node_data.velocity();

                    if act.has_node_changed_significantly(
                        &node_id_str,
                        position.clone(),
                        velocity.clone(),
                    ) {
                        filtered_nodes.push((*node_id, node_data.clone()));
                    }
                }

                if !filtered_nodes.is_empty() {
                    let binary_data = binary_protocol::encode_node_data(&filtered_nodes);

                    act.total_node_count = filtered_nodes.len();
                    let moving_nodes = filtered_nodes
                        .iter()
                        .filter(|(_, node_data)| {
                            let vel = node_data.velocity();
                            vel.x.abs() > 0.001 || vel.y.abs() > 0.001 || vel.z.abs() > 0.001
                        })
                        .count();
                    act.nodes_in_motion = moving_nodes;

                    act.last_transfer_size = binary_data.len();
                    act.total_bytes_sent += binary_data.len();
                    act.update_count += 1;
                    act.nodes_sent_count += filtered_nodes.len();

                    if detailed_debug {
                        debug!(
                            "[Position Updates] Sending {} nodes, {} bytes",
                            filtered_nodes.len(),
                            binary_data.len()
                        );
                    }

                    ctx.binary(binary_data);
                }

                let next_interval = std::time::Duration::from_millis(actual_interval);
                ctx.run_later(next_interval, move |act, ctx| {
                    let subscription_msg = format!(
                        "{{\"type\":\"subscribe_position_updates\",\"data\":{{\"interval\":{},\"binary\":{}}}}}",
                        actual_interval, binary
                    );
                    <SocketFlowServer as StreamHandler<
                        Result<ws::Message, ws::ProtocolError>,
                    >>::handle(
                        act,
                        Ok(ws::Message::Text(subscription_msg.into())),
                        ctx,
                    );
                });
            }
        }));
    });
}

pub(crate) fn handle_request_swarm_telemetry(
    act: &mut SocketFlowServer,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client requested enhanced swarm telemetry");

    let app_state = act.app_state.clone();

    ctx.spawn(
        actix::fut::wrap_future::<_, SocketFlowServer>(async move {
            match crate::handlers::bots_handler::fetch_hive_mind_agents(&app_state, None).await {
                Ok(agents) => {
                    let mut nodes_data = Vec::new();
                    let mut swarm_metrics = serde_json::json!({
                        "total_agents": agents.len(),
                        "active_agents": 0,
                        "avg_health": 0.0,
                        "avg_cpu": 0.0,
                        "avg_workload": 0.0,
                        "total_tokens": 0,
                        "swarm_ids": std::collections::HashSet::<String>::new(),
                    });

                    let (mut active_count, mut total_health, mut total_cpu, mut total_workload) = (0u32, 0.0f32, 0.0f32, 0.0f32);

                    for (idx, agent) in agents.iter().enumerate() {
                        if agent.status == "active" { active_count += 1; }
                        total_health += agent.health;
                        total_cpu += agent.cpu_usage;
                        total_workload += agent.workload;

                        let id = (1000 + idx) as u32;
                        let node_data = BinaryNodeData {
                            node_id: id,
                            x: (idx as f32 * 100.0).sin() * 500.0,
                            y: (idx as f32 * 100.0).cos() * 500.0,
                            z: 0.0, vx: 0.0, vy: 0.0, vz: 0.0,
                        };
                        nodes_data.push((id, node_data));
                    }

                    let n = agents.len() as f32;
                    if n > 0.0 {
                        swarm_metrics["active_agents"] = serde_json::json!(active_count);
                        swarm_metrics["avg_health"] = serde_json::json!(total_health / n);
                        swarm_metrics["avg_cpu"] = serde_json::json!(total_cpu / n);
                        swarm_metrics["avg_workload"] = serde_json::json!(total_workload / n);
                        swarm_metrics["total_tokens"] = serde_json::json!(0);
                        swarm_metrics["swarm_count"] = serde_json::json!(0);
                    }

                    (nodes_data, swarm_metrics)
                }
                Err(_) => (vec![], serde_json::json!({})),
            }
        })
        .map(|(nodes_data, swarm_metrics), _act, ctx| {
            if !nodes_data.is_empty() {
                let binary_data = binary_protocol::encode_node_data(&nodes_data);
                ctx.binary(binary_data);
            }

            let telemetry_response = serde_json::json!({
                "type": "swarmTelemetry",
                "timestamp": chrono::Utc::now().timestamp_millis(),
                "data_source": "live",
                "metrics": swarm_metrics,
                "node_count": nodes_data.len()
            });

            if let Ok(msg_str) = serde_json::to_string(&telemetry_response) {
                ctx.text(msg_str);
            }
        }),
    );
}
