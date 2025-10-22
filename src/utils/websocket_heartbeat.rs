use actix::{Actor, AsyncContext};
use actix_web_actors::ws;
use chrono::Utc;
use log::warn;
use serde_json::json;
use std::time::{Duration, Instant};

/// Shared WebSocket heartbeat functionality
pub trait WebSocketHeartbeat: Actor<Context = ws::WebsocketContext<Self>>
where
    Self: Sized,
{
    /// Get the client ID for this WebSocket connection
    fn get_client_id(&self) -> &str;

    /// Get the last heartbeat timestamp
    fn get_last_heartbeat(&self) -> Instant;

    /// Update the last heartbeat timestamp
    fn update_last_heartbeat(&mut self);

    /// Start the heartbeat mechanism with configurable intervals
    fn start_heartbeat(
        &self,
        ctx: &mut ws::WebsocketContext<Self>,
        ping_interval_secs: u64,
        timeout_secs: u64,
    ) where
        Self: actix::Actor<Context = ws::WebsocketContext<Self>> + 'static,
    {
        let ping_duration = Duration::from_secs(ping_interval_secs);
        let timeout_duration = Duration::from_secs(timeout_secs);

        ctx.run_interval(ping_duration, move |act, ctx| {
            if Instant::now().duration_since(act.get_last_heartbeat()) > timeout_duration {
                warn!(
                    "WebSocket client {} heartbeat timeout, disconnecting",
                    act.get_client_id()
                );
                // Close the WebSocket connection - this will trigger cleanup
                ctx.close(Some(ws::CloseReason {
                    code: ws::CloseCode::Abnormal,
                    description: Some("Heartbeat timeout".to_string()),
                }));
                return;
            }

            ctx.ping(b"heartbeat");
        });
    }

    /// Send a ping message
    fn send_ping(&self, ctx: &mut ws::WebsocketContext<Self>)
    where
        Self: actix::Actor<Context = ws::WebsocketContext<Self>>,
    {
        let ping_message = json!({
            "type": "ping",
            "timestamp": Utc::now(),
            "client_id": self.get_client_id()
        });

        if let Ok(msg) = serde_json::to_string(&ping_message) {
            ctx.text(msg);
        }
    }

    /// Send a pong response
    fn send_pong(&self, ctx: &mut ws::WebsocketContext<Self>)
    where
        Self: actix::Actor<Context = ws::WebsocketContext<Self>>,
    {
        let pong_message = json!({
            "type": "pong",
            "timestamp": Utc::now(),
            "client_id": self.get_client_id()
        });

        if let Ok(msg) = serde_json::to_string(&pong_message) {
            ctx.text(msg);
        }
    }

    /// Handle standard heartbeat messages
    fn handle_heartbeat_message(
        &mut self,
        msg: Result<ws::Message, ws::ProtocolError>,
        ctx: &mut ws::WebsocketContext<Self>,
    ) -> bool
    where
        Self: actix::Actor<Context = ws::WebsocketContext<Self>>,
    {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.update_last_heartbeat();
                ctx.pong(&msg);
                true // Message handled
            }
            Ok(ws::Message::Pong(_)) => {
                self.update_last_heartbeat();
                true // Message handled
            }
            _ => false, // Message not handled
        }
    }
}

/// Default heartbeat settings
pub struct HeartbeatConfig {
    pub ping_interval_secs: u64,
    pub timeout_secs: u64,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            ping_interval_secs: 5, // Send ping every 5 seconds
            timeout_secs: 30,      // Disconnect after 30 seconds of no response
        }
    }
}

impl HeartbeatConfig {
    pub fn new(ping_interval_secs: u64, timeout_secs: u64) -> Self {
        Self {
            ping_interval_secs,
            timeout_secs,
        }
    }

    pub fn fast() -> Self {
        Self::new(2, 10) // Fast heartbeat for critical connections
    }

    pub fn slow() -> Self {
        Self::new(15, 60) // Slow heartbeat for background connections
    }
}

/// Common WebSocket message types
#[derive(serde::Serialize, serde::Deserialize, Debug)]
#[serde(tag = "type")]
pub enum CommonWebSocketMessage {
    #[serde(rename = "ping")]
    Ping {
        timestamp: chrono::DateTime<Utc>,
        client_id: String,
    },

    #[serde(rename = "pong")]
    Pong {
        timestamp: chrono::DateTime<Utc>,
        client_id: String,
    },

    #[serde(rename = "connection_established")]
    ConnectionEstablished {
        client_id: String,
        timestamp: chrono::DateTime<Utc>,
    },

    #[serde(rename = "error")]
    Error {
        message: String,
        timestamp: chrono::DateTime<Utc>,
    },
}
