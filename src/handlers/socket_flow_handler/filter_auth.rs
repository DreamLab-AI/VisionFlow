use actix::prelude::*;
use log::{error, info, warn};

use super::types::SocketFlowServer;

/// Handle "authenticate" message -- NIP-98 and legacy token/pubkey paths.
pub(crate) fn handle_authenticate(
    act: &mut SocketFlowServer,
    msg: &serde_json::Value,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client sent authenticate message");

    if let Some(event_b64) = msg.get("event").and_then(|e| e.as_str()) {
        // --- NIP-98 path: { type: "authenticate", event: "<base64>" } ---
        let nostr_service = act.app_state.nostr_service.clone();
        let client_id = act.client_id;
        let cm_addr = act.client_manager_addr.clone();
        let auth_header = format!("Nostr {}", event_b64);
        let ws_url = act.connection_url.clone();

        ctx.spawn(
            actix::fut::wrap_future::<_, SocketFlowServer>(async move {
                if let Some(ref ns) = nostr_service {
                    match ns
                        .verify_nip98_auth(&auth_header, &ws_url, "GET", None)
                        .await
                    {
                        Ok(user) => return (Some(user), client_id, cm_addr),
                        Err(e) => {
                            warn!("NIP-98 WS auth failed: {}", e);
                        }
                    }
                }
                (None, client_id, cm_addr)
            })
            .map(|(user_opt, client_id, cm_addr), act, ctx| {
                if let Some(user) = user_opt {
                    act.pubkey = Some(user.pubkey.clone());
                    act.is_power_user = user.is_power_user;

                    if let Some(cid) = client_id {
                        use crate::actors::messages::AuthenticateClient;
                        cm_addr.do_send(AuthenticateClient {
                            client_id: cid,
                            pubkey: user.pubkey.clone(),
                            is_power_user: user.is_power_user,
                        });
                    }

                    let response = serde_json::json!({
                        "type": "authenticate_success",
                        "pubkey": user.pubkey,
                        "is_power_user": user.is_power_user,
                        "timestamp": chrono::Utc::now().timestamp_millis()
                    });
                    if let Ok(msg_str) = serde_json::to_string(&response) {
                        ctx.text(msg_str);
                    }
                    info!(
                        "NIP-98 WS authenticated: pubkey={}, power_user={}",
                        user.pubkey, user.is_power_user
                    );
                } else {
                    let error_msg = serde_json::json!({
                        "type": "error",
                        "message": "NIP-98 WebSocket authentication failed"
                    });
                    if let Ok(msg_str) = serde_json::to_string(&error_msg) {
                        ctx.text(msg_str);
                    }
                    warn!("NIP-98 WS authentication failed for client");
                }
            }),
        );
    } else {
        // --- Legacy path: { type: "authenticate", token, pubkey } ---
        let token = msg
            .get("token")
            .and_then(|t| t.as_str())
            .map(String::from);
        let pubkey = msg
            .get("pubkey")
            .and_then(|p| p.as_str())
            .map(String::from);

        if let (Some(token), Some(pubkey)) = (token, pubkey) {
            let nostr_service = act.app_state.nostr_service.clone();
            let client_id = act.client_id;
            let cm_addr = act.client_manager_addr.clone();

            ctx.spawn(
                actix::fut::wrap_future::<_, SocketFlowServer>(async move {
                    if let Some(ref ns) = nostr_service {
                        if let Some(user) = ns.get_session(&token).await {
                            if user.pubkey == pubkey {
                                return (Some(user), client_id, cm_addr);
                            }
                        }
                    }
                    (None, client_id, cm_addr)
                })
                .map(|(user_opt, client_id, cm_addr), act, ctx| {
                    if let Some(user) = user_opt {
                        act.pubkey = Some(user.pubkey.clone());
                        act.is_power_user = user.is_power_user;

                        if let Some(cid) = client_id {
                            use crate::actors::messages::AuthenticateClient;
                            cm_addr.do_send(AuthenticateClient {
                                client_id: cid,
                                pubkey: user.pubkey.clone(),
                                is_power_user: user.is_power_user,
                            });
                        }

                        let response = serde_json::json!({
                            "type": "authenticate_success",
                            "pubkey": user.pubkey,
                            "is_power_user": user.is_power_user,
                            "timestamp": chrono::Utc::now().timestamp_millis()
                        });
                        if let Ok(msg_str) = serde_json::to_string(&response) {
                            ctx.text(msg_str);
                        }
                        info!(
                            "Client authenticated: pubkey={}, power_user={}",
                            user.pubkey, user.is_power_user
                        );
                    } else {
                        let error_msg = serde_json::json!({
                            "type": "error",
                            "message": "Authentication failed: invalid token or pubkey mismatch"
                        });
                        if let Ok(msg_str) = serde_json::to_string(&error_msg) {
                            ctx.text(msg_str);
                        }
                        warn!("Authentication failed for client");
                    }
                }),
            );
        } else {
            let error_msg = serde_json::json!({
                "type": "error",
                "message": "Authentication requires 'event' (NIP-98) or both 'token' and 'pubkey'"
            });
            if let Ok(msg_str) = serde_json::to_string(&error_msg) {
                ctx.text(msg_str);
            }
        }
    }
}

/// Handle "filter_update" message -- per-client node filtering with optional Neo4j persistence.
pub(crate) fn handle_filter_update(
    act: &mut SocketFlowServer,
    msg: &serde_json::Value,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("Client sent filter_update message");

    if let Some(client_id) = act.client_id {
        // Check both nested "filter" key and "data" key (client sends in data)
        let filter_data = msg.get("filter").or_else(|| msg.get("data")).unwrap_or(msg);

        use crate::actors::messages::UpdateClientFilter;
        let update = UpdateClientFilter {
            client_id,
            enabled: filter_data
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            quality_threshold: filter_data
                .get("quality_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.7),
            authority_threshold: filter_data
                .get("authority_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5),
            filter_by_quality: filter_data
                .get("filter_by_quality")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            filter_by_authority: filter_data
                .get("filter_by_authority")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            filter_mode: filter_data
                .get("filter_mode")
                .and_then(|v| v.as_str())
                .unwrap_or("or")
                .to_string(),
            max_nodes: filter_data
                .get("max_nodes")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32),
        };

        info!(
            "Processing filter update: enabled={}, quality_threshold={}, filter_by_quality={}",
            update.enabled, update.quality_threshold, update.filter_by_quality
        );

        let cm_addr = act.client_manager_addr.clone();
        let pubkey = act.pubkey.clone();

        ctx.spawn(
            actix::fut::wrap_future::<_, SocketFlowServer>(async move {
                match cm_addr.send(update.clone()).await {
                    Ok(Ok(())) => (true, pubkey, update),
                    Ok(Err(e)) => {
                        error!("Failed to update client filter: {}", e);
                        (false, pubkey, update)
                    }
                    Err(e) => {
                        error!("Failed to send filter update: {}", e);
                        (false, pubkey, update)
                    }
                }
            })
            .map(|(success, pubkey_opt, update), act, ctx| {
                if success {
                    // Persist to Neo4j if authenticated
                    if let Some(pubkey) = pubkey_opt {
                        info!(
                            "Filter updated for pubkey {}: enabled={}, max_nodes={:?}",
                            pubkey, update.enabled, update.max_nodes
                        );

                        let neo4j_repo = act.app_state.neo4j_settings_repository.clone();
                        let pubkey_clone = pubkey.clone();
                        use crate::adapters::neo4j_settings_repository::UserFilter;
                        let filter = UserFilter {
                            pubkey: pubkey_clone.clone(),
                            enabled: update.enabled,
                            quality_threshold: update.quality_threshold,
                            authority_threshold: update.authority_threshold,
                            filter_by_quality: update.filter_by_quality,
                            filter_by_authority: update.filter_by_authority,
                            filter_mode: update.filter_mode.clone(),
                            max_nodes: update.max_nodes,
                            updated_at: chrono::Utc::now(),
                        };

                        ctx.spawn(
                            actix::fut::wrap_future::<_, SocketFlowServer>(async move {
                                match neo4j_repo
                                    .save_user_filter(&pubkey_clone, &filter)
                                    .await
                                {
                                    Ok(()) => {
                                        info!(
                                            "Filter persisted to Neo4j for pubkey: {}",
                                            pubkey_clone
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to persist filter to Neo4j: {}", e);
                                    }
                                }
                            })
                            .map(|_result, _act, _ctx| ()),
                        );
                    }

                    let response = serde_json::json!({
                        "type": "filter_update_success",
                        "enabled": update.enabled,
                        "timestamp": chrono::Utc::now().timestamp_millis()
                    });
                    if let Ok(msg_str) = serde_json::to_string(&response) {
                        ctx.text(msg_str);
                    }
                } else {
                    let error_msg = serde_json::json!({
                        "type": "error",
                        "message": "Failed to update filter"
                    });
                    if let Ok(msg_str) = serde_json::to_string(&error_msg) {
                        ctx.text(msg_str);
                    }
                }
            }),
        );
    } else {
        warn!("filter_update received but client_id not yet assigned - registration may still be in progress");
        let error_msg = serde_json::json!({
            "type": "error",
            "message": "Client registration in progress, please retry filter update in a moment"
        });
        if let Ok(msg_str) = serde_json::to_string(&error_msg) {
            ctx.text(msg_str);
        }
    }
}

/// Handle ontology validation requests.
pub(crate) fn handle_ontology_validation(
    act: &mut SocketFlowServer,
    msg: &serde_json::Value,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("[WebSocket] Ontology validation request received");
    let ontology_id = msg
        .get("ontologyId")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    if let Some(ref ontology_addr) = act.app_state.ontology_actor_addr {
        let addr = ontology_addr.clone();
        let fut = async move {
            use crate::actors::messages::GetOntologyReport;
            match addr
                .send(GetOntologyReport {
                    report_id: Some(ontology_id.clone()),
                })
                .await
            {
                Ok(Ok(Some(report))) => {
                    serde_json::json!({
                        "type": "ontology_validation_update",
                        "ontologyId": ontology_id,
                        "status": "completed",
                        "violations": report.violations.len(),
                        "inferredTriples": report.inferred_triples.len(),
                        "constraints": report.constraint_summary.total_constraints,
                        "timestamp": chrono::Utc::now().timestamp_millis()
                    })
                }
                Ok(Ok(None)) => {
                    serde_json::json!({
                        "type": "ontology_validation_update",
                        "ontologyId": ontology_id,
                        "status": "not_found",
                        "message": "No validation report available. Run validation first.",
                        "timestamp": chrono::Utc::now().timestamp_millis()
                    })
                }
                _ => {
                    serde_json::json!({
                        "type": "ontology_validation_update",
                        "status": "error",
                        "message": "Failed to retrieve validation report",
                        "timestamp": chrono::Utc::now().timestamp_millis()
                    })
                }
            }
        };

        let fut = actix::fut::wrap_future::<_, SocketFlowServer>(fut);
        ctx.spawn(fut.map(|response, _act, ctx| {
            if let Ok(msg_str) = serde_json::to_string(&response) {
                ctx.text(msg_str);
            }
        }));
    } else {
        let response = serde_json::json!({
            "type": "ontology_validation_update",
            "status": "unavailable",
            "message": "Ontology system not initialized"
        });
        if let Ok(msg_str) = serde_json::to_string(&response) {
            ctx.text(msg_str);
        }
    }
}

/// Handle ontology constraint update/toggle requests.
pub(crate) fn handle_ontology_constraint_update(
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("[WebSocket] Ontology constraint update request");
    let response = serde_json::json!({
        "type": "ontology_constraint_update",
        "status": "acknowledged",
        "message": "Use REST API /api/ontology-physics/enable for constraint management",
        "timestamp": chrono::Utc::now().timestamp_millis()
    });
    if let Ok(msg_str) = serde_json::to_string(&response) {
        ctx.text(msg_str);
    }
}

/// Handle ontology reasoning requests.
pub(crate) fn handle_ontology_reasoning(
    act: &mut SocketFlowServer,
    msg: &serde_json::Value,
    ctx: &mut <SocketFlowServer as Actor>::Context,
) {
    info!("[WebSocket] Ontology reasoning request received");
    if let Some(ref ontology_addr) = act.app_state.ontology_actor_addr {
        let addr = ontology_addr.clone();
        let ontology_id = msg
            .get("ontologyId")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let source = msg
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("websocket")
            .to_string();

        let fut = async move {
            use crate::actors::ontology_actor::TriggerReasoning;
            match addr
                .send(TriggerReasoning {
                    ontology_id,
                    source,
                })
                .await
            {
                Ok(Ok(job_id)) => serde_json::json!({
                    "type": "ontology_reasoning_started",
                    "jobId": job_id,
                    "timestamp": chrono::Utc::now().timestamp_millis()
                }),
                _ => serde_json::json!({
                    "type": "ontology_reasoning_error",
                    "message": "Failed to trigger reasoning",
                    "timestamp": chrono::Utc::now().timestamp_millis()
                }),
            }
        };
        let fut = actix::fut::wrap_future::<_, SocketFlowServer>(fut);
        ctx.spawn(fut.map(|response, _act, ctx| {
            if let Ok(msg_str) = serde_json::to_string(&response) {
                ctx.text(msg_str);
            }
        }));
    }
}
