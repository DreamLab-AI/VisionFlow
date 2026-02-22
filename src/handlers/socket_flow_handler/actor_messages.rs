use actix::{Handler, Message};
use log::{error, info, trace};

use crate::utils::binary_protocol;
use crate::utils::socket_flow_messages::BinaryNodeData;

use super::types::SocketFlowServer;

// Message to set client ID after registration
#[derive(Message)]
#[rtype(result = "()")]
pub(crate) struct SetClientId(pub usize);

impl Handler<SetClientId> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: SetClientId, _ctx: &mut Self::Context) -> Self::Result {
        self.client_id = Some(msg.0);
        info!("[WebSocket] Client assigned ID: {}", msg.0);
    }
}

// Broadcast position update to the client
#[derive(Message, Clone)]
#[rtype(result = "()")]
pub struct BroadcastPositionUpdate(pub Vec<(u32, BinaryNodeData)>);

impl Handler<BroadcastPositionUpdate> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: BroadcastPositionUpdate, ctx: &mut Self::Context) -> Self::Result {
        if !msg.0.is_empty() {
            let binary_data = binary_protocol::encode_node_data(&msg.0);
            ctx.binary(binary_data);

            if self.should_log_update() {
                trace!("[WebSocket] Position update sent: {} nodes", msg.0.len());
            }
        }
    }
}

// Import the actor messages for binary/text send
use crate::actors::messages::{SendToClientBinary, SendToClientText};

impl Handler<SendToClientBinary> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: SendToClientBinary, ctx: &mut Self::Context) {
        ctx.binary(msg.0);
    }
}

impl Handler<SendToClientText> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: SendToClientText, ctx: &mut Self::Context) {
        ctx.text(msg.0);
    }
}

// Handler for initial graph load - sends all nodes and edges as JSON
use crate::actors::messages::{SendInitialGraphLoad, SendPositionUpdate};

impl Handler<SendInitialGraphLoad> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: SendInitialGraphLoad, ctx: &mut Self::Context) -> Self::Result {
        use crate::utils::socket_flow_messages::Message;

        let initial_load = Message::InitialGraphLoad {
            nodes: msg.nodes,
            edges: msg.edges,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };

        if let Ok(json) = serde_json::to_string(&initial_load) {
            ctx.text(json);
            if let Message::InitialGraphLoad { nodes, edges, .. } = &initial_load {
                info!(
                    "[WebSocket] Sent initial graph load: {} nodes, {} edges",
                    nodes.len(),
                    edges.len()
                );
            }
        } else {
            error!("[WebSocket] Failed to serialize initial graph load message");
        }
    }
}

// Handler for streamed position updates
impl Handler<SendPositionUpdate> for SocketFlowServer {
    type Result = ();

    fn handle(&mut self, msg: SendPositionUpdate, ctx: &mut Self::Context) -> Self::Result {
        use crate::utils::socket_flow_messages::Message;

        let position_update = Message::PositionUpdate {
            node_id: msg.node_id,
            x: msg.x,
            y: msg.y,
            z: msg.z,
            vx: msg.vx,
            vy: msg.vy,
            vz: msg.vz,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        };

        if let Ok(json) = serde_json::to_string(&position_update) {
            ctx.text(json);
            if self.should_log_update() {
                trace!("[WebSocket] Sent position update for node {}", msg.node_id);
            }
        } else {
            error!(
                "[WebSocket] Failed to serialize position update for node {}",
                msg.node_id
            );
        }
    }
}
