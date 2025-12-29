//! JSS WebSocket Bridge
//!
//! Bridges JSS WebSocket notifications (solid-0.1 protocol) to VisionFlow clients.
//! Connects to JSS at ws://jss:3030/.notifications and forwards resource update
//! notifications to connected VisionFlow clients.
//!
//! Protocol:
//! - JSS sends: "pub <url>" when a resource is published/updated
//! - JSS sends: "ack <url>" when a subscription is acknowledged
//! - VisionFlow clients receive: structured JSON messages via realtime WebSocket

use futures_util::{SinkExt, StreamExt};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use uuid::Uuid;

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
            ],
        }
    }
}

/// JSS WebSocket Bridge
///
/// Maintains a persistent WebSocket connection to JSS and bridges
/// notifications to VisionFlow clients.
pub struct JssWebSocketBridge {
    config: JssBridgeConfig,
    state: Arc<RwLock<BridgeState>>,
    status: Arc<RwLock<BridgeStatus>>,
    subscriptions: Arc<RwLock<HashSet<String>>>,
    shutdown_tx: Arc<RwLock<Option<mpsc::Sender<()>>>>,
    started_at: Arc<RwLock<Option<Instant>>>,
}

impl JssWebSocketBridge {
    /// Create a new JSS WebSocket bridge
    pub fn new(config: JssBridgeConfig) -> Self {
        info!(
            "JSS WebSocket Bridge initialized - WS URL: {}",
            config.jss_config.ws_url
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
        }
    }

    /// Create with default configuration from environment
    pub fn from_env() -> Self {
        Self::new(JssBridgeConfig::default())
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
                            Self::handle_message(&text, status).await;
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

    async fn handle_message(message: &str, status: &Arc<RwLock<BridgeStatus>>) {
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
                Self::forward_to_visionflow("publish", &url, status).await;
            }
            JssNotificationType::Acknowledge(url) => {
                debug!("JSS subscription acknowledged: {}", url);
                Self::forward_to_visionflow("acknowledge", &url, status).await;
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
        status: &Arc<RwLock<BridgeStatus>>,
    ) {
        // Create VisionFlow notification
        let notification = JssResourceNotification {
            resource_url: resource_url.to_string(),
            notification_type: notification_type.to_string(),
            timestamp: Self::current_timestamp(),
            resource_path: Self::extract_path(resource_url),
            resource_type: Self::determine_resource_type(resource_url),
        };

        // Create realtime message
        let message = RealtimeWebSocketMessage {
            msg_type: "jss_notification".to_string(),
            data: serde_json::to_value(&notification).unwrap_or(json!({})),
            timestamp: Self::current_timestamp(),
            client_id: None,
            session_id: None,
        };

        // Broadcast to all subscribed VisionFlow clients
        let msg_clone = message.clone();
        tokio::spawn(async move {
            let manager = CONNECTION_MANAGER.lock().await;
            manager.broadcast("jss_notification", msg_clone).await;
        });

        // Update forwarded count
        {
            let mut st = status.write().await;
            st.messages_forwarded += 1;
        }

        debug!("Forwarded JSS notification to VisionFlow clients");
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
}
