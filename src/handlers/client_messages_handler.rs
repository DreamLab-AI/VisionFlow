use actix_web::{web, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use actix::{Actor, StreamHandler, AsyncContext, ActorContext};
use serde_json::json;
use std::time::{Duration, Instant};
use log::{info, debug, warn};

use crate::AppState;
use crate::utils::client_message_extractor::ClientMessage;

/// WebSocket actor for streaming client messages from agents to frontend
pub struct ClientMessagesWs {
    app_state: web::Data<AppState>,
    last_heartbeat: Instant,
}

impl ClientMessagesWs {
    pub fn new(app_state: web::Data<AppState>) -> Self {
        Self {
            app_state,
            last_heartbeat: Instant::now(),
        }
    }

    fn start_heartbeat(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(Duration::from_secs(30), |act, ctx| {
            if Instant::now().duration_since(act.last_heartbeat) > Duration::from_secs(90) {
                warn!("Client messages WebSocket heartbeat timeout, disconnecting");
                ctx.stop();
                return;
            }
            ctx.ping(b"");
        });
    }

    fn start_message_stream(&self, ctx: &mut ws::WebsocketContext<Self>) {
        let rx = self.app_state.client_message_rx.clone();

        ctx.run_interval(Duration::from_millis(100), move |_act, ctx| {
            // Try to receive messages without blocking
            if let Ok(mut receiver) = rx.try_lock() {
                while let Ok(msg) = receiver.try_recv() {
                    let json = json!({
                        "type": "client_message",
                        "content": msg.content,
                        "timestamp": msg.timestamp.to_rfc3339(),
                        "session_id": msg.session_id,
                        "agent_id": msg.agent_id
                    });

                    ctx.text(json.to_string());
                    debug!("Forwarded client message to WebSocket: {}", msg.content);
                }
            }
        });
    }
}

impl Actor for ClientMessagesWs {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Client messages WebSocket connection established");
        self.start_heartbeat(ctx);
        self.start_message_stream(ctx);

        // Send initial connection confirmation
        let init_json = json!({
            "type": "init",
            "status": "connected",
            "message": "Client message stream ready"
        });
        ctx.text(init_json.to_string());
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Client messages WebSocket connection closed");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for ClientMessagesWs {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.last_heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.last_heartbeat = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                debug!("Received WebSocket text: {}", text);
                // Client can send commands here if needed
            }
            Ok(ws::Message::Close(reason)) => {
                info!("Client messages WebSocket closing: {:?}", reason);
                ctx.stop();
            }
            Ok(ws::Message::Binary(_)) => {
                warn!("Binary messages not supported on client messages stream");
            }
            Err(e) => {
                warn!("WebSocket protocol error: {}", e);
                ctx.stop();
            }
            _ => {}
        }
    }
}

/// WebSocket route handler
pub async fn websocket_client_messages(
    req: HttpRequest,
    stream: web::Payload,
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, actix_web::Error> {
    info!("New client messages WebSocket connection request");

    let resp = ws::start(
        ClientMessagesWs::new(app_state),
        &req,
        stream,
    );

    match resp {
        Ok(response) => Ok(response),
        Err(e) => {
            warn!("Failed to establish client messages WebSocket: {}", e);
            Err(e)
        }
    }
}
