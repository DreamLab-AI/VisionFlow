use actix::{Handler, Message};
use log::{debug, error, info, trace};

use crate::utils::delta_encoding;
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
        if msg.0.is_empty() {
            return;
        }

        let frame = self.delta_frame_counter;
        let is_full_sync = frame == 0; // Frame 0 and every 60th frame triggers full sync

        // Filter to only nodes that have actually moved (epsilon check using squared distance)
        let epsilon_sq = self.delta_epsilon_sq;
        let changed_nodes: Vec<&(u32, BinaryNodeData)> = msg.0.iter().filter(|(node_id, node)| {
            if let Some(prev) = self.delta_previous_nodes.get(node_id) {
                let dx = node.x - prev.x;
                let dy = node.y - prev.y;
                let dz = node.z - prev.z;
                (dx * dx + dy * dy + dz * dz) > epsilon_sq
            } else {
                true // New node, always include
            }
        }).collect();

        // If graph has fully converged (0 nodes changed) and not a full sync frame, skip broadcast
        if changed_nodes.is_empty() && !is_full_sync {
            // Still advance frame counter so full sync cadence is maintained
            self.delta_frame_counter = (frame + 1) % 60;
            return;
        }

        // Encode using delta encoding (V4 for delta frames, V3 for full sync frames)
        let binary_data = delta_encoding::encode_node_data_delta(
            &msg.0,
            &self.delta_previous_nodes,
            frame,
            &[], // agent_node_ids -- flags are already set on the node IDs by callers
            &[], // knowledge_node_ids
        );
        ctx.binary(binary_data);

        // Update previous state for next delta computation.
        // On full sync frames, clear and repopulate to prevent stale entries for
        // deleted nodes from accumulating (VULN-09 / Code Review #2).
        if is_full_sync {
            self.delta_previous_nodes.clear();
        }
        for (node_id, node) in &msg.0 {
            self.delta_previous_nodes.insert(*node_id, *node);
        }

        // Advance frame counter (wraps at 60 for periodic full sync)
        self.delta_frame_counter = (frame + 1) % 60;

        if self.should_log_update() {
            if is_full_sync {
                debug!(
                    "[WebSocket] Full sync sent (frame {}): {} nodes",
                    frame, msg.0.len()
                );
            } else {
                trace!(
                    "[WebSocket] Delta update sent (frame {}): {} changed of {} total nodes",
                    frame, changed_nodes.len(), msg.0.len()
                );
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
