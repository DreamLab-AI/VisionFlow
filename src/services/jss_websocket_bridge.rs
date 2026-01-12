//! JSS WebSocket Bridge
//!
//! Bridges JSS WebSocket notifications (solid-0.1 protocol) to VisionFlow clients.
//! Connects to JSS at ws://jss:3030/.notifications and forwards resource update
//! notifications to connected VisionFlow clients via both JSON and binary protocols.
//!
//! Protocol:
//! - JSS sends: "pub <url>" when a resource is published/updated
//! - JSS sends: "ack <url>" when a subscription is acknowledged
//! - VisionFlow clients receive:
//!   - JSON messages via RealtimeWebSocketHandler (for general updates)
//!   - Binary messages via ClientCoordinatorActor (for ontology/agent updates)
//!
//! Event Filtering:
//! - Ontology changes (/ontology/, /public/ontology/) -> Binary + JSON
//! - Agent updates (/agents/, /contributions/) -> Binary + JSON
//! - Profile changes (/profile/) -> JSON only
//! - Other resources -> JSON only

use actix::Addr;
use futures_util::{SinkExt, StreamExt};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

use crate::actors::client_coordinator_actor::ClientCoordinatorActor;
use crate::actors::messages::BroadcastNodePositions;
use crate::handlers::solid_proxy_handler::JssConfig;
use crate::handlers::realtime_websocket_handler::{
    RealtimeWebSocketMessage, CONNECTION_MANAGER,
};

/// JSS WebSocket Bridge errors
#[derive(Debug, Error)]
pub enum JssBridgeError {
    #[error("WebSocket connection failed: {0}")]
    ConnectionError(String),

    #[error("WebSocket send failed: {0}")]
    SendError(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("Reconnection limit exceeded")]
    ReconnectionLimitExceeded,

    #[error("Bridge already running")]
    AlreadyRunning,

    #[error("Bridge not running")]
    NotRunning,
}

pub type Result<T> = std::result::Result<T, JssBridgeError>;

/// JSS notification message types
#[derive(Debug, Clone, PartialEq)]
pub enum JssNotificationType {
    /// Resource published/updated: "pub <url>"
    Publish(String),
    /// Subscription acknowledged: "ack <url>"
    Acknowledge(String),
    /// Ping message for keepalive
    Ping,
    /// Pong response
    Pong,
    /// Unknown message
    Unknown(String),
}

impl JssNotificationType {
    /// Parse a JSS notification message
    pub fn parse(message: &str) -> Self {
        let trimmed = message.trim();

        if trimmed.starts_with("pub ") {
            JssNotificationType::Publish(trimmed[4..].to_string())
        } else if trimmed.starts_with("ack ") {
            JssNotificationType::Acknowledge(trimmed[4..].to_string())
        } else if trimmed == "ping" {
            JssNotificationType::Ping
        } else if trimmed == "pong" {
            JssNotificationType::Pong
        } else {
            JssNotificationType::Unknown(trimmed.to_string())
        }
    }
}

/// VisionFlow notification message for clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JssResourceNotification {
    /// URL of the affected resource
    pub resource_url: String,
    /// Type of notification (publish, acknowledge)
    pub notification_type: String,
    /// Timestamp when the notification was received
    pub timestamp: u64,
    /// Extracted resource path
    pub resource_path: Option<String>,
    /// Resource type if identifiable (ontology, contribution, etc.)
    pub resource_type: Option<String>,
}

/// Event relevance categories for binary protocol routing
#[derive(Debug, Clone, PartialEq)]
pub enum EventRelevance {
    /// Ontology changes - high priority, triggers graph reload
    OntologyChange,
    /// Agent/contribution updates - high priority, triggers position updates
    AgentUpdate,
    /// Profile changes - medium priority, JSON only
    ProfileChange,
    /// Pod resource changes - medium priority, JSON only
    PodResource,
    /// Other/unknown - low priority, JSON only
    Other,
}

impl EventRelevance {
    /// Determine relevance from resource URL
    pub fn from_url(url: &str) -> Self {
        let url_lower = url.to_lowercase();

        if url_lower.contains("/ontology/") || url_lower.contains("/public/ontology/") {
            EventRelevance::OntologyChange
        } else if url_lower.contains("/agents/") || url_lower.contains("/contributions/") {
            EventRelevance::AgentUpdate
        } else if url_lower.contains("/profile/") {
            EventRelevance::ProfileChange
        } else if url_lower.contains("/pods/") {
            EventRelevance::PodResource
        } else {
            EventRelevance::Other
        }
    }

    /// Check if this event should trigger binary protocol broadcast
    pub fn requires_binary_broadcast(&self) -> bool {
        matches!(self, EventRelevance::OntologyChange | EventRelevance::AgentUpdate)
    }

    /// Get priority level (higher = more urgent)
    pub fn priority(&self) -> u8 {
        match self {
            EventRelevance::OntologyChange => 10,
            EventRelevance::AgentUpdate => 8,
            EventRelevance::ProfileChange => 5,
            EventRelevance::PodResource => 3,
            EventRelevance::Other => 1,
        }
    }
}

/// Bridge connection state
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Stopped,
}

/// Bridge status for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStatus {
    pub state: String,
    pub connected: bool,
    pub last_message_at: Option<u64>,
    pub messages_received: u64,
    pub messages_forwarded: u64,
    pub reconnect_count: u64,
    pub subscribed_resources: Vec<String>,
    pub uptime_secs: u64,
}

/// Configuration for JSS WebSocket bridge
#[derive(Debug, Clone)]
pub struct JssBridgeConfig {
    pub jss_config: JssConfig,
    pub reconnect_initial_delay_ms: u64,
    pub reconnect_max_delay_ms: u64,
    pub reconnect_max_attempts: u32,
    pub ping_interval_secs: u64,
    pub connection_timeout_secs: u64,
    pub auto_subscribe_paths: Vec<String>,
    /// Enable binary protocol broadcasts for relevant events
    pub enable_binary_broadcast: bool,
    /// Event types to broadcast via binary protocol
    pub binary_event_types: Vec<String>,
}

impl Default for JssBridgeConfig {
    fn default() -> Self {
        Self {
            jss_config: JssConfig::from_env(),
            reconnect_initial_delay_ms: 1000,
            reconnect_max_delay_ms: 60000,
            reconnect_max_attempts: 0, // 0 = unlimited
            ping_interval_secs: 30,
            connection_timeout_secs: 30,
            auto_subscribe_paths: vec![
                "/public/ontology/*".to_string(),
                "/pods/*/contributions/*".to_string(),
                "/pods/*/agents/*".to_string(),
            ],
            enable_binary_broadcast: true,
            binary_event_types: vec![
                "ontology".to_string(),
                "contribution".to_string(),
                "agent".to_string(),
            ],
        }
    }
}

/// JSS WebSocket Bridge
/// Maintains a persistent WebSocket connection to JSS and bridges
/// notifications to VisionFlow clients via both JSON and binary protocols.
pub struct JssWebSocketBridge {
    config: JssBridgeConfig,
    state: Arc<RwLock<BridgeState>>,
    status: Arc<RwLock<BridgeStatus>>,
    subscriptions: Arc<RwLock<HashSet<String>>>,
    shutdown_tx: Arc<RwLock<Option<mpsc::Sender<()>>>>,
    started_at: Arc<RwLock<Option<Instant>>>,
    /// ClientCoordinatorActor address for binary protocol broadcasts
    client_coordinator_addr: Arc<RwLock<Option<Addr<ClientCoordinatorActor>>>>,
    /// Binary broadcast statistics
    binary_broadcasts_sent: Arc<RwLock<u64>>,
}

impl JssWebSocketBridge {
    /// Create a new JSS WebSocket bridge
    pub fn new(config: JssBridgeConfig) -> Self {
        info!(
            "JSS WebSocket Bridge initialized - WS URL: {}, binary_broadcast: {}",
            config.jss_config.ws_url,
            config.enable_binary_broadcast
        );

        Self {
            config,
            state: Arc::new(RwLock::new(BridgeState::Disconnected)),
            status: Arc::new(RwLock::new(BridgeStatus {
                state: "disconnected".to_string(),
                connected: false,
                last_message_at: None,
                messages_received: 0,
                messages_forwarded: 0,
                reconnect_count: 0,
                subscribed_resources: Vec::new(),
                uptime_secs: 0,
            })),
            subscriptions: Arc::new(RwLock::new(HashSet::new())),
            shutdown_tx: Arc::new(RwLock::new(None)),
            started_at: Arc::new(RwLock::new(None)),
            client_coordinator_addr: Arc::new(RwLock::new(None)),
            binary_broadcasts_sent: Arc::new(RwLock::new(0)),
        }
    }

    /// Create with default configuration from environment
    pub fn from_env() -> Self {
        Self::new(JssBridgeConfig::default())
    }

    /// Set the ClientCoordinatorActor address for binary protocol broadcasts
    pub async fn set_client_coordinator(&self, addr: Addr<ClientCoordinatorActor>) {
        info!("JSS Bridge: ClientCoordinatorActor address set for binary broadcasts");
        *self.client_coordinator_addr.write().await = Some(addr);
    }

    /// Get binary broadcast count
    pub async fn get_binary_broadcast_count(&self) -> u64 {
        *self.binary_broadcasts_sent.read().await
    }

    /// Get current bridge state
    pub async fn get_state(&self) -> BridgeState {
        self.state.read().await.clone()
    }

    /// Get bridge status for monitoring
    pub async fn get_status(&self) -> BridgeStatus {
        let mut status = self.status.read().await.clone();

        // Update uptime
        if let Some(started) = *self.started_at.read().await {
            status.uptime_secs = started.elapsed().as_secs();
        }

        // Update subscriptions list
        status.subscribed_resources = self.subscriptions.read().await.iter().cloned().collect();

        status
    }

    /// Start the WebSocket bridge
    pub async fn start(&self) -> Result<()> {
        // Check if already running
        {
            let state = self.state.read().await;
            if *state != BridgeState::Disconnected && *state != BridgeState::Stopped {
                return Err(JssBridgeError::AlreadyRunning);
            }
        }

        info!("Starting JSS WebSocket Bridge...");

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        *self.shutdown_tx.write().await = Some(shutdown_tx);
        *self.started_at.write().await = Some(Instant::now());

        // Clone state for async task
        let config = self.config.clone();
        let state = self.state.clone();
        let status = self.status.clone();
        let subscriptions = self.subscriptions.clone();
        let client_coordinator_addr = self.client_coordinator_addr.clone();
        let binary_broadcasts_sent = self.binary_broadcasts_sent.clone();

        // Spawn the main connection loop
        tokio::spawn(async move {
            let mut reconnect_attempts = 0u32;
            let mut current_delay = config.reconnect_initial_delay_ms;

            loop {
                // Update state to connecting
                {
                    let mut s = state.write().await;
                    *s = if reconnect_attempts > 0 {
                        BridgeState::Reconnecting
                    } else {
                        BridgeState::Connecting
                    };
                }

                Self::update_status_state(&status, "connecting").await;

                // Attempt connection
                match Self::connect_and_run(
                    &config,
                    &state,
                    &status,
                    &subscriptions,
                    &client_coordinator_addr,
                    &binary_broadcasts_sent,
                    &mut shutdown_rx,
                ).await {
                    Ok(_) => {
                        // Clean shutdown
                        info!("JSS WebSocket Bridge stopped cleanly");
                        break;
                    }
                    Err(e) => {
                        error!("JSS WebSocket connection error: {}", e);

                        // Check if we should retry
                        reconnect_attempts += 1;

                        if config.reconnect_max_attempts > 0
                            && reconnect_attempts >= config.reconnect_max_attempts
                        {
                            error!("Max reconnection attempts exceeded");
                            break;
                        }

                        // Update status
                        {
                            let mut st = status.write().await;
                            st.reconnect_count += 1;
                        }

                        // Exponential backoff
                        info!(
                            "Reconnecting in {}ms (attempt {})",
                            current_delay, reconnect_attempts
                        );

                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_millis(current_delay)) => {}
                            _ = shutdown_rx.recv() => {
                                info!("Shutdown received during reconnect wait");
                                break;
                            }
                        }

                        // Increase delay with exponential backoff
                        current_delay = (current_delay * 2).min(config.reconnect_max_delay_ms);
                    }
                }
            }

            // Final state update
            *state.write().await = BridgeState::Stopped;
            Self::update_status_state(&status, "stopped").await;
        });

        Ok(())
    }

    /// Stop the WebSocket bridge
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping JSS WebSocket Bridge...");

        let shutdown_tx = self.shutdown_tx.write().await.take();

        if let Some(tx) = shutdown_tx {
            let _ = tx.send(()).await;
        }

        Ok(())
    }

    /// Subscribe to resource notifications
    pub async fn subscribe(&self, resource_path: &str) -> Result<()> {
        self.subscriptions.write().await.insert(resource_path.to_string());
        debug!("Added subscription for: {}", resource_path);
        Ok(())
    }

    /// Unsubscribe from resource notifications
    pub async fn unsubscribe(&self, resource_path: &str) -> Result<()> {
        self.subscriptions.write().await.remove(resource_path);
        debug!("Removed subscription for: {}", resource_path);
        Ok(())
    }

    // Private methods

    async fn connect_and_run(
        config: &JssBridgeConfig,
        state: &Arc<RwLock<BridgeState>>,
        status: &Arc<RwLock<BridgeStatus>>,
        subscriptions: &Arc<RwLock<HashSet<String>>>,
        client_coordinator_addr: &Arc<RwLock<Option<Addr<ClientCoordinatorActor>>>>,
        binary_broadcasts_sent: &Arc<RwLock<u64>>,
        shutdown_rx: &mut mpsc::Receiver<()>,
    ) -> Result<()> {
        let ws_url = &config.jss_config.ws_url;

        info!("Connecting to JSS WebSocket: {}", ws_url);

        // Connect with timeout
        let connect_result = timeout(
            Duration::from_secs(config.connection_timeout_secs),
            connect_async(ws_url),
        ).await;

        let (ws_stream, response) = match connect_result {
            Ok(Ok((stream, resp))) => (stream, resp),
            Ok(Err(e)) => {
                return Err(JssBridgeError::ConnectionError(e.to_string()));
            }
            Err(_) => {
                return Err(JssBridgeError::ConnectionError(
                    "Connection timeout".to_string(),
                ));
            }
        };

        info!(
            "Connected to JSS WebSocket (HTTP status: {})",
            response.status()
        );

        // Update state to connected
        *state.write().await = BridgeState::Connected;
        Self::update_status_state(status, "connected").await;
        {
            let mut st = status.write().await;
            st.connected = true;
        }

        let (mut ws_tx, mut ws_rx) = ws_stream.split();

        // Subscribe to auto-subscribe paths
        for path in &config.auto_subscribe_paths {
            let sub_msg = format!("sub {}", path);
            if let Err(e) = ws_tx.send(WsMessage::Text(sub_msg.clone())).await {
                warn!("Failed to send subscription: {}", e);
            } else {
                debug!("Sent subscription: {}", sub_msg);
                subscriptions.write().await.insert(path.clone());
            }
        }

        // Set up ping interval
        let mut ping_interval = interval(Duration::from_secs(config.ping_interval_secs));

        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                msg = ws_rx.next() => {
                    match msg {
                        Some(Ok(WsMessage::Text(text))) => {
                            Self::handle_message(
                                &text,
                                config,
                                status,
                                client_coordinator_addr,
                                binary_broadcasts_sent,
                            ).await;
                        }
                        Some(Ok(WsMessage::Ping(data))) => {
                            debug!("Received WebSocket ping");
                            if let Err(e) = ws_tx.send(WsMessage::Pong(data)).await {
                                warn!("Failed to send pong: {}", e);
                            }
                        }
                        Some(Ok(WsMessage::Pong(_))) => {
                            debug!("Received WebSocket pong");
                        }
                        Some(Ok(WsMessage::Close(frame))) => {
                            info!("JSS WebSocket closed: {:?}", frame);
                            return Err(JssBridgeError::ConnectionError(
                                "Connection closed by server".to_string(),
                            ));
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error: {}", e);
                            return Err(JssBridgeError::ConnectionError(e.to_string()));
                        }
                        None => {
                            info!("WebSocket stream ended");
                            return Err(JssBridgeError::ConnectionError(
                                "Stream ended".to_string(),
                            ));
                        }
                        _ => {}
                    }
                }

                // Send periodic pings
                _ = ping_interval.tick() => {
                    debug!("Sending WebSocket ping");
                    if let Err(e) = ws_tx.send(WsMessage::Ping(vec![])).await {
                        warn!("Failed to send ping: {}", e);
                    }
                }

                // Handle shutdown
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received");

                    // Send close frame
                    let _ = ws_tx.send(WsMessage::Close(None)).await;

                    *state.write().await = BridgeState::Stopped;
                    return Ok(());
                }
            }
        }
    }

    async fn handle_message(
        message: &str,
        config: &JssBridgeConfig,
        status: &Arc<RwLock<BridgeStatus>>,
        client_coordinator_addr: &Arc<RwLock<Option<Addr<ClientCoordinatorActor>>>>,
        binary_broadcasts_sent: &Arc<RwLock<u64>>,
    ) {
        debug!("Received JSS message: {}", message);

        // Update message count
        {
            let mut st = status.write().await;
            st.messages_received += 1;
            st.last_message_at = Some(Self::current_timestamp());
        }

        let notification_type = JssNotificationType::parse(message);

        match notification_type {
            JssNotificationType::Publish(url) => {
                info!("JSS resource published: {}", url);
                Self::forward_to_visionflow(
                    "publish",
                    &url,
                    config,
                    status,
                    client_coordinator_addr,
                    binary_broadcasts_sent,
                ).await;
            }
            JssNotificationType::Acknowledge(url) => {
                debug!("JSS subscription acknowledged: {}", url);
                Self::forward_to_visionflow(
                    "acknowledge",
                    &url,
                    config,
                    status,
                    client_coordinator_addr,
                    binary_broadcasts_sent,
                ).await;
            }
            JssNotificationType::Ping => {
                debug!("Received JSS ping");
            }
            JssNotificationType::Pong => {
                debug!("Received JSS pong");
            }
            JssNotificationType::Unknown(msg) => {
                warn!("Unknown JSS message: {}", msg);
            }
        }
    }

    async fn forward_to_visionflow(
        notification_type: &str,
        resource_url: &str,
        config: &JssBridgeConfig,
        status: &Arc<RwLock<BridgeStatus>>,
        client_coordinator_addr: &Arc<RwLock<Option<Addr<ClientCoordinatorActor>>>>,
        binary_broadcasts_sent: &Arc<RwLock<u64>>,
    ) {
        // Determine event relevance for routing
        let relevance = EventRelevance::from_url(resource_url);
        let resource_type = Self::determine_resource_type(resource_url);

        // Create VisionFlow notification
        let notification = JssResourceNotification {
            resource_url: resource_url.to_string(),
            notification_type: notification_type.to_string(),
            timestamp: Self::current_timestamp(),
            resource_path: Self::extract_path(resource_url),
            resource_type: resource_type.clone(),
        };

        // Create realtime message for JSON broadcast
        let message = RealtimeWebSocketMessage {
            msg_type: "jss_notification".to_string(),
            data: serde_json::to_value(&notification).unwrap_or(json!({})),
            timestamp: Self::current_timestamp(),
            client_id: None,
            session_id: None,
        };

        // Broadcast to all subscribed VisionFlow clients via JSON (always)
        let msg_clone = message.clone();
        tokio::spawn(async move {
            let manager = CONNECTION_MANAGER.lock().await;
            manager.broadcast("jss_notification", msg_clone).await;
        });

        // Binary protocol broadcast for high-priority events (ontology/agent changes)
        if config.enable_binary_broadcast && relevance.requires_binary_broadcast() {
            if let Some(ref addr) = *client_coordinator_addr.read().await {
                // Create binary notification message
                let binary_msg = Self::create_binary_notification(
                    notification_type,
                    resource_url,
                    &relevance,
                    &resource_type,
                );

                if !binary_msg.is_empty() {
                    info!(
                        "JSS Bridge: Sending binary broadcast for {:?} event: {} ({} bytes)",
                        relevance,
                        resource_url,
                        binary_msg.len()
                    );

                    // Send via ClientCoordinatorActor
                    addr.do_send(BroadcastNodePositions {
                        positions: binary_msg,
                    });

                    // Update binary broadcast count
                    {
                        let mut count = binary_broadcasts_sent.write().await;
                        *count += 1;
                    }
                }
            } else {
                debug!(
                    "JSS Bridge: Binary broadcast skipped - no ClientCoordinatorActor address (event: {:?})",
                    relevance
                );
            }
        }

        // Update forwarded count
        {
            let mut st = status.write().await;
            st.messages_forwarded += 1;
        }

        debug!(
            "Forwarded JSS notification to VisionFlow clients (relevance: {:?}, binary: {})",
            relevance,
            relevance.requires_binary_broadcast()
        );
    }

    /// Create a binary notification message for the SocketFlowHandler binary protocol
    fn create_binary_notification(
        notification_type: &str,
        resource_url: &str,
        relevance: &EventRelevance,
        resource_type: &Option<String>,
    ) -> Vec<u8> {
        // Binary message format:
        // [message_type: u8][relevance_priority: u8][timestamp: u64][url_len: u16][url: bytes][type_len: u8][type: bytes]
        let mut buffer = Vec::with_capacity(256);

        // Message type: 0x20 = JSS notification (custom type for JSS events)
        let msg_type: u8 = match relevance {
            EventRelevance::OntologyChange => 0x21, // Ontology update
            EventRelevance::AgentUpdate => 0x22,    // Agent/contribution update
            _ => 0x20,                              // Generic JSS notification
        };
        buffer.push(msg_type);

        // Relevance priority (1-10)
        buffer.push(relevance.priority());

        // Notification type (0 = publish, 1 = acknowledge)
        buffer.push(if notification_type == "publish" { 0 } else { 1 });

        // Timestamp (8 bytes, big endian)
        let timestamp = Self::current_timestamp();
        buffer.extend_from_slice(&timestamp.to_be_bytes());

        // Resource URL length and data
        let url_bytes = resource_url.as_bytes();
        let url_len = url_bytes.len().min(u16::MAX as usize) as u16;
        buffer.extend_from_slice(&url_len.to_be_bytes());
        buffer.extend_from_slice(&url_bytes[..url_len as usize]);

        // Resource type length and data
        if let Some(ref res_type) = resource_type {
            let type_bytes = res_type.as_bytes();
            let type_len = type_bytes.len().min(u8::MAX as usize) as u8;
            buffer.push(type_len);
            buffer.extend_from_slice(&type_bytes[..type_len as usize]);
        } else {
            buffer.push(0);
        }

        buffer
    }

    fn extract_path(url: &str) -> Option<String> {
        // Extract path from full URL
        url::Url::parse(url)
            .ok()
            .map(|u| u.path().to_string())
    }

    fn determine_resource_type(url: &str) -> Option<String> {
        if url.contains("/ontology/") {
            Some("ontology".to_string())
        } else if url.contains("/contributions/") {
            Some("contribution".to_string())
        } else if url.contains("/profile/") {
            Some("profile".to_string())
        } else if url.contains("/pods/") {
            Some("pod_resource".to_string())
        } else {
            None
        }
    }

    async fn update_status_state(status: &Arc<RwLock<BridgeStatus>>, state: &str) {
        let mut st = status.write().await;
        st.state = state.to_string();
        st.connected = state == "connected";
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

impl Default for JssWebSocketBridge {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_notification_pub() {
        let msg = "pub http://jss:3030/pods/alice/ontology/Person.ttl";
        let notification = JssNotificationType::parse(msg);

        assert_eq!(
            notification,
            JssNotificationType::Publish("http://jss:3030/pods/alice/ontology/Person.ttl".to_string())
        );
    }

    #[test]
    fn test_parse_notification_ack() {
        let msg = "ack /public/ontology/*";
        let notification = JssNotificationType::parse(msg);

        assert_eq!(
            notification,
            JssNotificationType::Acknowledge("/public/ontology/*".to_string())
        );
    }

    #[test]
    fn test_parse_notification_ping() {
        let msg = "ping";
        let notification = JssNotificationType::parse(msg);

        assert_eq!(notification, JssNotificationType::Ping);
    }

    #[test]
    fn test_parse_notification_unknown() {
        let msg = "unknown message";
        let notification = JssNotificationType::parse(msg);

        assert_eq!(
            notification,
            JssNotificationType::Unknown("unknown message".to_string())
        );
    }

    #[test]
    fn test_extract_path() {
        let url = "http://jss:3030/pods/alice/ontology/Person.ttl";
        let path = JssWebSocketBridge::extract_path(url);

        assert_eq!(path, Some("/pods/alice/ontology/Person.ttl".to_string()));
    }

    #[test]
    fn test_determine_resource_type() {
        assert_eq!(
            JssWebSocketBridge::determine_resource_type("http://jss:3030/public/ontology/Person.ttl"),
            Some("ontology".to_string())
        );

        assert_eq!(
            JssWebSocketBridge::determine_resource_type("http://jss:3030/pods/alice/contributions/prop1.jsonld"),
            Some("contribution".to_string())
        );

        assert_eq!(
            JssWebSocketBridge::determine_resource_type("http://jss:3030/pods/alice/profile/card"),
            Some("profile".to_string())
        );
    }

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = JssWebSocketBridge::from_env();
        let state = bridge.get_state().await;

        assert_eq!(state, BridgeState::Disconnected);
    }

    #[tokio::test]
    async fn test_subscription_management() {
        let bridge = JssWebSocketBridge::from_env();

        bridge.subscribe("/test/path/*").await.unwrap();

        let status = bridge.get_status().await;
        assert!(status.subscribed_resources.contains(&"/test/path/*".to_string()));

        bridge.unsubscribe("/test/path/*").await.unwrap();

        let status = bridge.get_status().await;
        assert!(!status.subscribed_resources.contains(&"/test/path/*".to_string()));
    }

    #[test]
    fn test_event_relevance_ontology() {
        let url = "http://jss:3030/public/ontology/Person.ttl";
        let relevance = EventRelevance::from_url(url);

        assert_eq!(relevance, EventRelevance::OntologyChange);
        assert!(relevance.requires_binary_broadcast());
        assert_eq!(relevance.priority(), 10);
    }

    #[test]
    fn test_event_relevance_agent() {
        let url = "http://jss:3030/pods/alice/contributions/prop1.jsonld";
        let relevance = EventRelevance::from_url(url);

        assert_eq!(relevance, EventRelevance::AgentUpdate);
        assert!(relevance.requires_binary_broadcast());
        assert_eq!(relevance.priority(), 8);
    }

    #[test]
    fn test_event_relevance_profile() {
        let url = "http://jss:3030/pods/alice/profile/card";
        let relevance = EventRelevance::from_url(url);

        assert_eq!(relevance, EventRelevance::ProfileChange);
        assert!(!relevance.requires_binary_broadcast());
        assert_eq!(relevance.priority(), 5);
    }

    #[test]
    fn test_event_relevance_pod_resource() {
        let url = "http://jss:3030/pods/alice/documents/readme.txt";
        let relevance = EventRelevance::from_url(url);

        assert_eq!(relevance, EventRelevance::PodResource);
        assert!(!relevance.requires_binary_broadcast());
        assert_eq!(relevance.priority(), 3);
    }

    #[test]
    fn test_event_relevance_other() {
        let url = "http://example.com/some/random/path";
        let relevance = EventRelevance::from_url(url);

        assert_eq!(relevance, EventRelevance::Other);
        assert!(!relevance.requires_binary_broadcast());
        assert_eq!(relevance.priority(), 1);
    }

    #[test]
    fn test_create_binary_notification() {
        let binary_msg = JssWebSocketBridge::create_binary_notification(
            "publish",
            "http://jss:3030/public/ontology/Person.ttl",
            &EventRelevance::OntologyChange,
            &Some("ontology".to_string()),
        );

        // Verify message structure
        assert!(!binary_msg.is_empty());
        assert_eq!(binary_msg[0], 0x21); // OntologyChange message type
        assert_eq!(binary_msg[1], 10);   // Priority
        assert_eq!(binary_msg[2], 0);    // Publish = 0

        // Timestamp is 8 bytes
        // URL length is 2 bytes, then URL
        // Resource type length is 1 byte, then type
    }

    #[test]
    fn test_create_binary_notification_agent() {
        let binary_msg = JssWebSocketBridge::create_binary_notification(
            "acknowledge",
            "http://jss:3030/pods/alice/contributions/test.jsonld",
            &EventRelevance::AgentUpdate,
            &Some("contribution".to_string()),
        );

        assert!(!binary_msg.is_empty());
        assert_eq!(binary_msg[0], 0x22); // AgentUpdate message type
        assert_eq!(binary_msg[1], 8);    // Priority
        assert_eq!(binary_msg[2], 1);    // Acknowledge = 1
    }

    #[tokio::test]
    async fn test_bridge_binary_broadcast_count() {
        let bridge = JssWebSocketBridge::from_env();

        // Initially should be 0
        assert_eq!(bridge.get_binary_broadcast_count().await, 0);
    }
}
