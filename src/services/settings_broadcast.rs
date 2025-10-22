use actix::prelude::*;
use actix_web_actors::ws;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// WebSocket broadcast service for real-time settings synchronization
///
/// Features:
/// - Broadcast setting changes to all connected clients
/// - Connection management with heartbeat
/// - Message batching for efficiency
/// - Type-safe message serialization
/// - Automatic reconnection support
///
/// Usage:
/// ```rust
/// // In settings actor handler
/// let broadcast = SettingsBroadcast::from_registry();
/// broadcast.send(BroadcastSettingChange {
///     key: "physics.damping".to_string(),
///     value: serde_json::json!(0.95),
///     timestamp: chrono::Utc::now().timestamp()
/// }).await;
/// ```

/// Message types for settings broadcast
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SettingsBroadcastMessage {
    /// Single setting changed
    SettingChanged {
        key: String,
        value: serde_json::Value,
        timestamp: i64,
    },
    /// Multiple settings changed (batch)
    SettingsBatchChanged {
        changes: Vec<SettingChange>,
        timestamp: i64,
    },
    /// Settings reloaded (hot-reload triggered)
    SettingsReloaded {
        timestamp: i64,
        reason: String,
    },
    /// Preset applied
    PresetApplied {
        preset_id: String,
        settings_count: usize,
        timestamp: i64,
    },
    /// Heartbeat ping
    Ping {
        timestamp: i64,
    },
    /// Heartbeat pong
    Pong {
        timestamp: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingChange {
    pub key: String,
    pub value: serde_json::Value,
}

/// WebSocket session for settings synchronization
pub struct SettingsWebSocket {
    /// Client ID for tracking
    id: String,
    /// Last heartbeat time
    hb: Instant,
    /// Broadcast manager address
    broadcast_addr: Addr<SettingsBroadcastManager>,
}

impl SettingsWebSocket {
    pub fn new(id: String, broadcast_addr: Addr<SettingsBroadcastManager>) -> Self {
        Self {
            id,
            hb: Instant::now(),
            broadcast_addr,
        }
    }

    /// Send heartbeat to client
    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_secs(5), |act, ctx| {
            // Check client heartbeat
            if Instant::now().duration_since(act.hb) > Duration::from_secs(30) {
                // Heartbeat timeout - disconnect
                log::warn!("Settings WebSocket heartbeat timeout for client {}", act.id);
                ctx.stop();
                return;
            }

            // Send ping
            let msg = SettingsBroadcastMessage::Ping {
                timestamp: chrono::Utc::now().timestamp(),
            };
            if let Ok(json) = serde_json::to_string(&msg) {
                ctx.text(json);
            }
        });
    }
}

impl Actor for SettingsWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        log::info!("Settings WebSocket connected: {}", self.id);

        // Start heartbeat
        self.hb(ctx);

        // Register with broadcast manager
        self.broadcast_addr.do_send(RegisterClient {
            id: self.id.clone(),
            addr: ctx.address(),
        });
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        log::info!("Settings WebSocket disconnected: {}", self.id);

        // Unregister from broadcast manager
        self.broadcast_addr.do_send(UnregisterClient {
            id: self.id.clone(),
        });
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SettingsWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.hb = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                self.hb = Instant::now();

                // Parse incoming message
                if let Ok(msg) = serde_json::from_str::<SettingsBroadcastMessage>(&text) {
                    match msg {
                        SettingsBroadcastMessage::Pong { .. } => {
                            // Client acknowledged ping
                        }
                        _ => {
                            log::debug!("Received settings message: {:?}", msg);
                        }
                    }
                }
            }
            Ok(ws::Message::Binary(_)) => {
                log::warn!("Binary messages not supported for settings WebSocket");
            }
            Ok(ws::Message::Close(reason)) => {
                log::info!("Settings WebSocket closing: {:?}", reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

/// Broadcast message to all connected clients
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastToClients {
    pub message: SettingsBroadcastMessage,
}

impl Handler<BroadcastToClients> for SettingsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: BroadcastToClients, ctx: &mut Self::Context) {
        if let Ok(json) = serde_json::to_string(&msg.message) {
            ctx.text(json);
        }
    }
}

/// Central broadcast manager for all settings WebSocket connections
pub struct SettingsBroadcastManager {
    /// Connected clients
    clients: Arc<RwLock<HashMap<String, Addr<SettingsWebSocket>>>>,
    /// Message buffer for batching
    message_buffer: Vec<SettingChange>,
    /// Last batch send time
    last_batch_send: Instant,
}

impl SettingsBroadcastManager {
    pub fn new() -> Self {
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            message_buffer: Vec::new(),
            last_batch_send: Instant::now(),
        }
    }

    /// Broadcast message to all connected clients
    fn broadcast(&self, message: SettingsBroadcastMessage) {
        if let Ok(clients) = self.clients.read() {
            let message_for_send = BroadcastToClients { message };

            for (id, addr) in clients.iter() {
                if let Err(e) = addr.try_send(message_for_send.clone()) {
                    log::warn!("Failed to send to client {}: {:?}", id, e);
                }
            }

            log::debug!("Broadcast to {} clients", clients.len());
        }
    }

    /// Flush message buffer if needed
    fn flush_buffer(&mut self) {
        if !self.message_buffer.is_empty()
            && Instant::now().duration_since(self.last_batch_send) > Duration::from_millis(100)
        {
            let changes = std::mem::take(&mut self.message_buffer);
            self.broadcast(SettingsBroadcastMessage::SettingsBatchChanged {
                changes,
                timestamp: chrono::Utc::now().timestamp(),
            });
            self.last_batch_send = Instant::now();
        }
    }
}

impl Actor for SettingsBroadcastManager {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        log::info!("Settings broadcast manager started");

        // Periodic buffer flush
        ctx.run_interval(Duration::from_millis(100), |act, _ctx| {
            act.flush_buffer();
        });
    }
}

impl Supervised for SettingsBroadcastManager {}

impl SystemService for SettingsBroadcastManager {}

impl Default for SettingsBroadcastManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Register a new client
#[derive(Message)]
#[rtype(result = "()")]
pub struct RegisterClient {
    pub id: String,
    pub addr: Addr<SettingsWebSocket>,
}

impl Handler<RegisterClient> for SettingsBroadcastManager {
    type Result = ();

    fn handle(&mut self, msg: RegisterClient, _ctx: &mut Self::Context) {
        if let Ok(mut clients) = self.clients.write() {
            clients.insert(msg.id.clone(), msg.addr);
            log::info!("Client registered: {} (total: {})", msg.id, clients.len());
        }
    }
}

/// Unregister a client
#[derive(Message)]
#[rtype(result = "()")]
pub struct UnregisterClient {
    pub id: String,
}

impl Handler<UnregisterClient> for SettingsBroadcastManager {
    type Result = ();

    fn handle(&mut self, msg: UnregisterClient, _ctx: &mut Self::Context) {
        if let Ok(mut clients) = self.clients.write() {
            clients.remove(&msg.id);
            log::info!("Client unregistered: {} (total: {})", msg.id, clients.len());
        }
    }
}

/// Broadcast a single setting change
#[derive(Message, Clone)]
#[rtype(result = "()")]
pub struct BroadcastSettingChange {
    pub key: String,
    pub value: serde_json::Value,
}

impl Handler<BroadcastSettingChange> for SettingsBroadcastManager {
    type Result = ();

    fn handle(&mut self, msg: BroadcastSettingChange, _ctx: &mut Self::Context) {
        // Add to buffer for batching
        self.message_buffer.push(SettingChange {
            key: msg.key.clone(),
            value: msg.value.clone(),
        });

        // If buffer is large, flush immediately
        if self.message_buffer.len() >= 10 {
            self.flush_buffer();
        }
    }
}

/// Broadcast settings reload notification
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastSettingsReload {
    pub reason: String,
}

impl Handler<BroadcastSettingsReload> for SettingsBroadcastManager {
    type Result = ();

    fn handle(&mut self, msg: BroadcastSettingsReload, _ctx: &mut Self::Context) {
        self.broadcast(SettingsBroadcastMessage::SettingsReloaded {
            timestamp: chrono::Utc::now().timestamp(),
            reason: msg.reason,
        });
    }
}

/// Broadcast preset application
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastPresetApplied {
    pub preset_id: String,
    pub settings_count: usize,
}

impl Handler<BroadcastPresetApplied> for SettingsBroadcastManager {
    type Result = ();

    fn handle(&mut self, msg: BroadcastPresetApplied, _ctx: &mut Self::Context) {
        self.broadcast(SettingsBroadcastMessage::PresetApplied {
            preset_id: msg.preset_id,
            settings_count: msg.settings_count,
            timestamp: chrono::Utc::now().timestamp(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = SettingsBroadcastMessage::SettingChanged {
            key: "physics.damping".to_string(),
            value: serde_json::json!(0.95),
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("SettingChanged"));
        assert!(json.contains("physics.damping"));
    }

    #[test]
    fn test_batch_message() {
        let msg = SettingsBroadcastMessage::SettingsBatchChanged {
            changes: vec![
                SettingChange {
                    key: "physics.damping".to_string(),
                    value: serde_json::json!(0.95),
                },
                SettingChange {
                    key: "physics.springK".to_string(),
                    value: serde_json::json!(0.05),
                },
            ],
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("SettingsBatchChanged"));
        assert!(json.contains("physics.damping"));
        assert!(json.contains("physics.springK"));
    }
}
