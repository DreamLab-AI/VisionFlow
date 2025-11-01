// Real-Time WebSocket Handler for All Feature Updates
// Handles workspace events, analysis progress, optimization status, and export notifications

use crate::app_state::AppState;
use actix::prelude::*;
use actix_web_actors::ws;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// Enhanced WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeWebSocketMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: Value,
    pub timestamp: u64,
    pub client_id: Option<String>,
    pub session_id: Option<String>,
}

// Workspace event messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceUpdateEvent {
    pub workspace_id: String,
    pub changes: Value,
    pub operation: String, 
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceDeletedEvent {
    pub workspace_id: String,
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceCollaborationEvent {
    pub workspace_id: String,
    pub action: String, 
    pub user_id: String,
    pub user_name: Option<String>,
    pub permissions: Option<Vec<String>>,
}

// Analysis event messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisProgressEvent {
    pub analysis_id: String,
    pub graph_id: Option<String>,
    pub progress: f64, 
    pub stage: String,
    pub estimated_time_remaining: Option<u64>,
    pub current_operation: String,
    pub metrics: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCompleteEvent {
    pub analysis_id: String,
    pub graph_id: Option<String>,
    pub results: Value,
    pub success: bool,
    pub error: Option<String>,
    pub processing_time: f64,
}

// Optimization event messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationUpdateEvent {
    pub optimization_id: String,
    pub graph_id: Option<String>,
    pub progress: f64, 
    pub algorithm: String,
    pub current_iteration: u64,
    pub total_iterations: u64,
    pub metrics: Value,
    pub recommendations: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResultEvent {
    pub optimization_id: String,
    pub graph_id: Option<String>,
    pub algorithm: String,
    pub confidence: f64,
    pub performance_gain: f64,
    pub clusters: u64,
    pub recommendations: Vec<Value>,
    pub layout_changes: Option<Value>,
    pub success: bool,
    pub error: Option<String>,
}

// Export event messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportProgressEvent {
    pub export_id: String,
    pub graph_id: Option<String>,
    pub format: String,
    pub progress: f64, 
    pub stage: String, 
    pub size: Option<u64>,
    pub estimated_time_remaining: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportReadyEvent {
    pub export_id: String,
    pub graph_id: Option<String>,
    pub format: String,
    pub download_url: String,
    pub size: u64,
    pub expires_at: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareCreatedEvent {
    pub share_id: String,
    pub graph_id: Option<String>,
    pub share_url: String,
    pub expires_at: Option<String>,
    pub password_protected: bool,
    pub permissions: Vec<String>,
    pub description: Option<String>,
}

// System notification messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemNotificationEvent {
    pub level: String, 
    pub title: String,
    pub message: String,
    pub actions: Option<Vec<Value>>,
    pub persistent: Option<bool>,
}

// Connection and subscription management
#[derive(Debug, Clone)]
pub struct ClientSubscription {
    pub client_id: String,
    pub subscriptions: HashSet<String>,   
    pub filters: HashMap<String, String>, 
    pub last_activity: Instant,
}

pub struct RealtimeWebSocketHandler {
    client_id: String,
    session_id: String,
    app_state: actix_web::web::Data<AppState>,
    subscriptions: HashSet<String>,
    filters: HashMap<String, String>,
    heartbeat: Instant,
    last_ping: Instant,
    message_count: u64,
    bytes_sent: u64,
    bytes_received: u64,
}

// Global connection manager for broadcasting
pub struct ConnectionManager {
    connections: HashMap<String, Addr<RealtimeWebSocketHandler>>,
    subscriptions: HashMap<String, HashSet<String>>, 
}

impl ConnectionManager {
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
            subscriptions: HashMap::new(),
        }
    }

    pub fn add_connection(&mut self, client_id: String, addr: Addr<RealtimeWebSocketHandler>) {
        self.connections.insert(client_id.clone(), addr);
        info!("Added WebSocket connection for client: {}", client_id);
    }

    pub fn remove_connection(&mut self, client_id: &str) {
        self.connections.remove(client_id);
        
        for (_, client_ids) in self.subscriptions.iter_mut() {
            client_ids.remove(client_id);
        }
        info!("Removed WebSocket connection for client: {}", client_id);
    }

    pub fn subscribe(&mut self, client_id: String, event_type: String) {
        self.subscriptions
            .entry(event_type.clone())
            .or_insert_with(HashSet::new)
            .insert(client_id.clone());
        debug!("Client {} subscribed to {}", client_id, event_type);
    }

    pub fn unsubscribe(&mut self, client_id: &str, event_type: &str) {
        if let Some(client_ids) = self.subscriptions.get_mut(event_type) {
            client_ids.remove(client_id);
            debug!("Client {} unsubscribed from {}", client_id, event_type);
        }
    }

    pub async fn broadcast(&self, event_type: &str, message: RealtimeWebSocketMessage) {
        if let Some(client_ids) = self.subscriptions.get(event_type) {
            for client_id in client_ids {
                if let Some(addr) = self.connections.get(client_id) {
                    addr.do_send(BroadcastMessage {
                        message: message.clone(),
                    });
                }
            }
        }
    }
}

// Static connection manager instance
use lazy_static::lazy_static;
use tokio::sync::Mutex;

lazy_static! {
    static ref CONNECTION_MANAGER: Mutex<ConnectionManager> = Mutex::new(ConnectionManager::new());
}

impl RealtimeWebSocketHandler {
    pub fn new(app_state: actix_web::web::Data<AppState>) -> Self {
        let client_id = Uuid::new_v4().to_string();
        let session_id = Uuid::new_v4().to_string();

        Self {
            client_id,
            session_id,
            app_state,
            subscriptions: HashSet::new(),
            filters: HashMap::new(),
            heartbeat: Instant::now(),
            last_ping: Instant::now(),
            message_count: 0,
            bytes_sent: 0,
            bytes_received: 0,
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn send_message(
        &mut self,
        ctx: &mut ws::WebsocketContext<Self>,
        message: RealtimeWebSocketMessage,
    ) {
        match serde_json::to_string(&message) {
            Ok(json_str) => {
                ctx.text(json_str.clone());
                self.message_count += 1;
                self.bytes_sent += json_str.len() as u64;

                if log::log_enabled!(log::Level::Debug) {
                    debug!("Sent message to {}: {}", self.client_id, message.msg_type);
                }
            }
            Err(e) => {
                error!("Failed to serialize message: {}", e);
            }
        }
    }

    fn handle_subscription(
        &mut self,
        ctx: &mut ws::WebsocketContext<Self>,
        event_type: String,
        filters: Option<HashMap<String, String>>,
    ) {
        self.subscriptions.insert(event_type.clone());

        if let Some(filter_map) = filters {
            for (key, value) in filter_map {
                self.filters
                    .insert(format!("{}:{}", event_type, key), value);
            }
        }

        
        let client_id = self.client_id.clone();
        let event_type_clone = event_type.clone();
        tokio::spawn(async move {
            let mut manager = CONNECTION_MANAGER.lock().await;
            manager.subscribe(client_id, event_type_clone);
        });

        
        let confirmation = RealtimeWebSocketMessage {
            msg_type: "subscription_confirmed".to_string(),
            data: json!({
                "event_type": event_type,
                "client_id": self.client_id,
                "filters_applied": !self.filters.is_empty()
            }),
            timestamp: Self::current_timestamp(),
            client_id: Some(self.client_id.clone()),
            session_id: Some(self.session_id.clone()),
        };

        self.send_message(ctx, confirmation);
        info!("Client {} subscribed to {}", self.client_id, event_type);
    }

    fn handle_unsubscription(&mut self, _ctx: &mut ws::WebsocketContext<Self>, event_type: String) {
        self.subscriptions.remove(&event_type);

        
        self.filters
            .retain(|key, _| !key.starts_with(&format!("{}:", event_type)));

        
        let client_id = self.client_id.clone();
        let event_type_clone = event_type.clone();
        tokio::spawn(async move {
            let mut manager = CONNECTION_MANAGER.lock().await;
            manager.unsubscribe(&client_id, &event_type_clone);
        });

        info!("Client {} unsubscribed from {}", self.client_id, event_type);
    }

    fn send_heartbeat(&mut self, ctx: &mut ws::WebsocketContext<Self>) {
        let heartbeat_msg = RealtimeWebSocketMessage {
            msg_type: "heartbeat".to_string(),
            data: json!({
                "server_time": Self::current_timestamp(),
                "client_id": self.client_id,
                "message_count": self.message_count,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "active_subscriptions": self.subscriptions.len(),
                "uptime": self.heartbeat.elapsed().as_secs()
            }),
            timestamp: Self::current_timestamp(),
            client_id: Some(self.client_id.clone()),
            session_id: Some(self.session_id.clone()),
        };

        self.send_message(ctx, heartbeat_msg);
        self.last_ping = Instant::now();
    }
}

impl Actor for RealtimeWebSocketHandler {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!(
            "Real-time WebSocket handler started for client: {}",
            self.client_id
        );

        
        let client_id = self.client_id.clone();
        let ctx_address = ctx.address();
        tokio::spawn(async move {
            let mut manager = CONNECTION_MANAGER.lock().await;
            manager.add_connection(client_id, ctx_address);
        });

        
        ctx.run_interval(Duration::from_secs(30), |act, ctx| {
            act.send_heartbeat(ctx);
        });

        
        ctx.run_interval(Duration::from_secs(10), |act, ctx| {
            if Instant::now().duration_since(act.heartbeat) > Duration::from_secs(120) {
                warn!(
                    "Client {} heartbeat timeout, closing connection",
                    act.client_id
                );
                ctx.stop();
                return;
            }
        });

        
        let welcome_message = RealtimeWebSocketMessage {
            msg_type: "connection_established".to_string(),
            data: json!({
                "client_id": self.client_id,
                "session_id": self.session_id,
                "server_time": Self::current_timestamp(),
                "features": [
                    "workspace_events",
                    "analysis_progress",
                    "optimization_updates",
                    "export_notifications",
                    "system_notifications",
                    "real_time_collaboration"
                ]
            }),
            timestamp: Self::current_timestamp(),
            client_id: Some(self.client_id.clone()),
            session_id: Some(self.session_id.clone()),
        };

        self.send_message(ctx, welcome_message);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!(
            "Real-time WebSocket handler stopped for client: {}",
            self.client_id
        );

        
        let client_id = self.client_id.clone();
        tokio::spawn(async move {
            let mut manager = CONNECTION_MANAGER.lock().await;
            manager.remove_connection(&client_id);
        });

        
        info!(
            "Final statistics for client {}: {} messages sent, {} bytes sent, {} bytes received",
            self.client_id, self.message_count, self.bytes_sent, self.bytes_received
        );
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for RealtimeWebSocketHandler {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                self.heartbeat = Instant::now();
                self.bytes_received += text.len() as u64;

                match serde_json::from_str::<RealtimeWebSocketMessage>(&text) {
                    Ok(ws_message) => match ws_message.msg_type.as_str() {
                        "subscribe" => {
                            if let (Ok(event_type), filters) = (
                                serde_json::from_value::<String>(
                                    ws_message
                                        .data
                                        .get("event_type")
                                        .unwrap_or(&Value::Null)
                                        .clone(),
                                ),
                                ws_message.data.get("filters").and_then(|f| {
                                    serde_json::from_value::<HashMap<String, String>>(f.clone())
                                        .ok()
                                }),
                            ) {
                                self.handle_subscription(ctx, event_type, filters);
                            }
                        }
                        "unsubscribe" => {
                            if let Ok(event_type) = serde_json::from_value::<String>(
                                ws_message
                                    .data
                                    .get("event_type")
                                    .unwrap_or(&Value::Null)
                                    .clone(),
                            ) {
                                self.handle_unsubscription(ctx, event_type);
                            }
                        }
                        "ping" => {
                            let pong = RealtimeWebSocketMessage {
                                msg_type: "pong".to_string(),
                                data: json!({
                                    "server_time": Self::current_timestamp(),
                                    "client_time": ws_message.timestamp
                                }),
                                timestamp: Self::current_timestamp(),
                                client_id: Some(self.client_id.clone()),
                                session_id: Some(self.session_id.clone()),
                            };
                            self.send_message(ctx, pong);
                        }
                        "get_subscriptions" => {
                            let subscriptions_msg = RealtimeWebSocketMessage {
                                msg_type: "subscriptions".to_string(),
                                data: json!({
                                    "subscriptions": self.subscriptions.iter().collect::<Vec<_>>(),
                                    "filters": self.filters
                                }),
                                timestamp: Self::current_timestamp(),
                                client_id: Some(self.client_id.clone()),
                                session_id: Some(self.session_id.clone()),
                            };
                            self.send_message(ctx, subscriptions_msg);
                        }
                        _ => {
                            debug!("Unhandled message type: {}", ws_message.msg_type);
                        }
                    },
                    Err(e) => {
                        error!("Failed to parse WebSocket message: {}", e);
                    }
                }
            }

            Ok(ws::Message::Ping(msg)) => {
                self.heartbeat = Instant::now();
                ctx.pong(&msg);
            }

            Ok(ws::Message::Pong(_)) => {
                self.heartbeat = Instant::now();
            }

            Ok(ws::Message::Close(reason)) => {
                info!(
                    "WebSocket closing for client {}: {:?}",
                    self.client_id, reason
                );
                ctx.stop();
            }

            Err(e) => {
                error!(
                    "WebSocket protocol error for client {}: {}",
                    self.client_id, e
                );
                ctx.stop();
            }

            _ => {
                debug!(
                    "Unhandled WebSocket message type for client {}",
                    self.client_id
                );
            }
        }
    }
}

// Message for broadcasting to specific client
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastMessage {
    pub message: RealtimeWebSocketMessage,
}

impl Handler<BroadcastMessage> for RealtimeWebSocketHandler {
    type Result = ();

    fn handle(&mut self, msg: BroadcastMessage, ctx: &mut Self::Context) {
        
        let should_send = if self.filters.is_empty() {
            true
        } else {
            
            let event_type = &msg.message.msg_type;
            let data = &msg.message.data;

            
            self.filters.iter().any(|(key, filter_value)| {
                if let Some((filter_event_type, filter_key)) = key.split_once(':') {
                    if filter_event_type == event_type {
                        if let Some(data_value) = data.get(filter_key) {
                            return data_value.as_str() == Some(filter_value);
                        }
                    }
                }
                false
            }) || !self.subscriptions.contains(event_type)
        };

        if should_send || self.filters.is_empty() {
            self.send_message(ctx, msg.message);
        }
    }
}

// Public API functions for broadcasting events
pub async fn broadcast_workspace_update(
    workspace_id: String,
    changes: Value,
    operation: String,
    user_id: Option<String>,
) {
    let event = WorkspaceUpdateEvent {
        workspace_id,
        changes,
        operation,
        user_id,
    };

    let message = RealtimeWebSocketMessage {
        msg_type: "workspace_update".to_string(),
        data: serde_json::to_value(&event).unwrap_or_default(),
        timestamp: RealtimeWebSocketHandler::current_timestamp(),
        client_id: None,
        session_id: None,
    };

    
    let msg_to_send = message.clone();
    tokio::spawn(async move {
        let manager = CONNECTION_MANAGER.lock().await;
        manager.broadcast("workspace_update", msg_to_send).await;
    });
}

pub async fn broadcast_analysis_progress(
    analysis_id: String,
    graph_id: Option<String>,
    progress: f64,
    stage: String,
    current_operation: String,
    metrics: Option<Value>,
) {
    let event = AnalysisProgressEvent {
        analysis_id,
        graph_id,
        progress,
        stage,
        estimated_time_remaining: None,
        current_operation,
        metrics,
    };

    let message = RealtimeWebSocketMessage {
        msg_type: "analysis_progress".to_string(),
        data: serde_json::to_value(&event).unwrap_or_default(),
        timestamp: RealtimeWebSocketHandler::current_timestamp(),
        client_id: None,
        session_id: None,
    };

    
    let msg = message.clone();
    tokio::spawn(async move {
        let manager = CONNECTION_MANAGER.lock().await;
        manager.broadcast("analysis_progress", msg).await;
    });
}

pub async fn broadcast_optimization_update(
    optimization_id: String,
    graph_id: Option<String>,
    progress: f64,
    algorithm: String,
    current_iteration: u64,
    total_iterations: u64,
    metrics: Value,
) {
    let event = OptimizationUpdateEvent {
        optimization_id,
        graph_id,
        progress,
        algorithm,
        current_iteration,
        total_iterations,
        metrics,
        recommendations: None,
    };

    let message = RealtimeWebSocketMessage {
        msg_type: "optimization_update".to_string(),
        data: serde_json::to_value(&event).unwrap_or_default(),
        timestamp: RealtimeWebSocketHandler::current_timestamp(),
        client_id: None,
        session_id: None,
    };

    
    let msg = message.clone();
    tokio::spawn(async move {
        let manager = CONNECTION_MANAGER.lock().await;
        manager.broadcast("optimization_update", msg).await;
    });
}

pub async fn broadcast_export_progress(
    export_id: String,
    graph_id: Option<String>,
    format: String,
    progress: f64,
    stage: String,
) {
    let event = ExportProgressEvent {
        export_id,
        graph_id,
        format,
        progress,
        stage,
        size: None,
        estimated_time_remaining: None,
    };

    let message = RealtimeWebSocketMessage {
        msg_type: "export_progress".to_string(),
        data: serde_json::to_value(&event).unwrap_or_default(),
        timestamp: RealtimeWebSocketHandler::current_timestamp(),
        client_id: None,
        session_id: None,
    };

    
    let msg = message.clone();
    tokio::spawn(async move {
        let manager = CONNECTION_MANAGER.lock().await;
        manager.broadcast("export_progress", msg).await;
    });
}

pub async fn broadcast_export_ready(
    export_id: String,
    graph_id: Option<String>,
    format: String,
    download_url: String,
    size: u64,
) {
    let event = ExportReadyEvent {
        export_id,
        graph_id,
        format,
        download_url,
        size,
        expires_at: chrono::Utc::now()
            .checked_add_signed(chrono::Duration::hours(24))
            .unwrap_or_else(chrono::Utc::now)
            .to_rfc3339(),
        metadata: json!({}),
    };

    let message = RealtimeWebSocketMessage {
        msg_type: "export_ready".to_string(),
        data: serde_json::to_value(&event).unwrap_or_default(),
        timestamp: RealtimeWebSocketHandler::current_timestamp(),
        client_id: None,
        session_id: None,
    };

    
    let msg = message.clone();
    tokio::spawn(async move {
        let manager = CONNECTION_MANAGER.lock().await;
        manager.broadcast("export_ready", msg).await;
    });
}

// WebSocket route handler
pub async fn realtime_websocket(
    req: actix_web::HttpRequest,
    stream: actix_web::web::Payload,
    app_state: actix_web::web::Data<AppState>,
) -> Result<actix_web::HttpResponse, actix_web::Error> {
    let resp = ws::start(RealtimeWebSocketHandler::new(app_state), &req, stream);

    info!("New real-time WebSocket connection established");
    resp
}
