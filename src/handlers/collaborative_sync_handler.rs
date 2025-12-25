// Collaborative Multi-User Sync Handler
// Real-time WebSocket sync for graph operations, user presence, and annotations

use actix::prelude::*;
use actix_web_actors::ws;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// Message types matching client-side TypeScript
const SYNC_UPDATE: u8 = 0x50;
const ANNOTATION_UPDATE: u8 = 0x51;
const SELECTION_UPDATE: u8 = 0x52;
const USER_POSITION: u8 = 0x53;
const VR_PRESENCE: u8 = 0x54;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOperation {
    pub id: String,
    pub op_type: u8, // 0=move, 1=add, 2=delete, 3=edge_add, 4=edge_delete
    pub user_id: String,
    pub node_id: String,
    pub position: Option<(f32, f32, f32)>,
    pub timestamp: u64,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSelection {
    pub agent_id: String,
    pub username: String,
    pub node_ids: Vec<String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnnotation {
    pub id: String,
    pub agent_id: String,
    pub username: String,
    pub node_id: String,
    pub text: String,
    pub position: (f32, f32, f32),
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct UserPresence {
    pub user_id: String,
    pub position: (f32, f32, f32),
    pub rotation: (f32, f32, f32, f32), // quaternion
    pub head_position: Option<(f32, f32, f32)>,
    pub head_rotation: Option<(f32, f32, f32, f32)>,
    pub left_hand_position: Option<(f32, f32, f32)>,
    pub left_hand_rotation: Option<(f32, f32, f32, f32)>,
    pub right_hand_position: Option<(f32, f32, f32)>,
    pub right_hand_rotation: Option<(f32, f32, f32, f32)>,
    pub last_update: Instant,
}

// Global sync state manager
pub struct SyncManager {
    operations: HashMap<String, GraphOperation>,
    selections: HashMap<String, UserSelection>,
    annotations: HashMap<String, GraphAnnotation>,
    user_presence: HashMap<String, UserPresence>,
    active_connections: HashMap<String, Addr<CollaborativeSyncActor>>,
    subscriptions: HashMap<String, HashSet<String>>, // channel -> user_ids
}

impl SyncManager {
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            selections: HashMap::new(),
            annotations: HashMap::new(),
            user_presence: HashMap::new(),
            active_connections: HashMap::new(),
            subscriptions: HashMap::new(),
        }
    }

    pub fn add_connection(&mut self, user_id: String, addr: Addr<CollaborativeSyncActor>) {
        self.active_connections.insert(user_id.clone(), addr);
        info!("Added sync connection for user: {}", user_id);
    }

    pub fn remove_connection(&mut self, user_id: &str) {
        self.active_connections.remove(user_id);
        self.user_presence.remove(user_id);

        // Remove from all subscriptions
        for (_, subscribers) in self.subscriptions.iter_mut() {
            subscribers.remove(user_id);
        }

        info!("Removed sync connection for user: {}", user_id);
    }

    pub fn subscribe(&mut self, user_id: String, channel: String) {
        self.subscriptions
            .entry(channel.clone())
            .or_insert_with(HashSet::new)
            .insert(user_id.clone());
        debug!("User {} subscribed to {}", user_id, channel);
    }

    pub fn broadcast_operation(&mut self, operation: GraphOperation) {
        let payload = self.encode_operation(&operation);

        if let Some(subscribers) = self.subscriptions.get("graph_sync") {
            for user_id in subscribers {
                if user_id != &operation.user_id {
                    if let Some(addr) = self.active_connections.get(user_id) {
                        addr.do_send(BroadcastMessage {
                            message_type: SYNC_UPDATE,
                            payload: payload.clone(),
                        });
                    }
                }
            }
        }

        self.operations.insert(operation.id.clone(), operation);
    }

    pub fn broadcast_selection(&mut self, selection: UserSelection) {
        let message = serde_json::to_string(&selection).unwrap_or_default();
        let payload = message.into_bytes();

        if let Some(subscribers) = self.subscriptions.get("user_presence") {
            for user_id in subscribers {
                if user_id != &selection.agent_id {
                    if let Some(addr) = self.active_connections.get(user_id) {
                        addr.do_send(BroadcastMessage {
                            message_type: SELECTION_UPDATE,
                            payload: payload.clone(),
                        });
                    }
                }
            }
        }

        self.selections.insert(selection.agent_id.clone(), selection);
    }

    pub fn broadcast_annotation(&mut self, annotation: GraphAnnotation) {
        let message = serde_json::to_string(&annotation).unwrap_or_default();
        let payload = message.into_bytes();

        if let Some(subscribers) = self.subscriptions.get("annotations") {
            for user_id in subscribers {
                if user_id != &annotation.agent_id {
                    if let Some(addr) = self.active_connections.get(user_id) {
                        addr.do_send(BroadcastMessage {
                            message_type: ANNOTATION_UPDATE,
                            payload: payload.clone(),
                        });
                    }
                }
            }
        }

        self.annotations.insert(annotation.id.clone(), annotation);
    }

    pub fn broadcast_user_position(&mut self, presence: UserPresence) {
        let payload = self.encode_user_position(&presence);

        if let Some(subscribers) = self.subscriptions.get("user_presence") {
            for user_id in subscribers {
                if user_id != &presence.user_id {
                    if let Some(addr) = self.active_connections.get(user_id) {
                        addr.do_send(BroadcastMessage {
                            message_type: USER_POSITION,
                            payload: payload.clone(),
                        });
                    }
                }
            }
        }

        self.user_presence.insert(presence.user_id.clone(), presence);
    }

    fn encode_operation(&self, op: &GraphOperation) -> Vec<u8> {
        let mut payload = Vec::new();

        // Operation type (1 byte)
        payload.push(op.op_type);

        // User ID (36 bytes for UUID)
        let mut user_id_bytes = op.user_id.as_bytes().to_vec();
        user_id_bytes.resize(36, 0);
        payload.extend_from_slice(&user_id_bytes);

        // Node ID (variable length with u16 prefix)
        let node_id_bytes = op.node_id.as_bytes();
        payload.extend_from_slice(&(node_id_bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(node_id_bytes);

        // Position (12 bytes if present)
        if let Some((x, y, z)) = op.position {
            payload.extend_from_slice(&x.to_le_bytes());
            payload.extend_from_slice(&y.to_le_bytes());
            payload.extend_from_slice(&z.to_le_bytes());
        }

        payload
    }

    fn encode_user_position(&self, presence: &UserPresence) -> Vec<u8> {
        let mut payload = Vec::new();

        // Position (12 bytes)
        payload.extend_from_slice(&presence.position.0.to_le_bytes());
        payload.extend_from_slice(&presence.position.1.to_le_bytes());
        payload.extend_from_slice(&presence.position.2.to_le_bytes());

        // Rotation (16 bytes)
        payload.extend_from_slice(&presence.rotation.0.to_le_bytes());
        payload.extend_from_slice(&presence.rotation.1.to_le_bytes());
        payload.extend_from_slice(&presence.rotation.2.to_le_bytes());
        payload.extend_from_slice(&presence.rotation.3.to_le_bytes());

        // User ID (4 bytes as hash)
        let user_id_hash = presence.user_id.as_bytes()
            .iter()
            .fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
        payload.extend_from_slice(&user_id_hash.to_le_bytes());

        payload
    }
}

// Actor for handling individual WebSocket connections
pub struct CollaborativeSyncActor {
    user_id: String,
    heartbeat: Instant,
}

impl CollaborativeSyncActor {
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            heartbeat: Instant::now(),
        }
    }

    fn handle_sync_message(&mut self, ctx: &mut ws::WebsocketContext<Self>, msg: &[u8]) {
        if msg.is_empty() {
            return;
        }

        let msg_type = msg[0];
        let payload = &msg[1..];

        match msg_type {
            SYNC_UPDATE => self.handle_operation(payload),
            SELECTION_UPDATE => self.handle_selection(payload),
            ANNOTATION_UPDATE => self.handle_annotation(payload),
            USER_POSITION => self.handle_user_position(payload),
            VR_PRESENCE => self.handle_vr_presence(payload),
            0xFF => self.handle_subscription(payload), // Subscription message
            _ => {
                warn!("Unknown sync message type: {}", msg_type);
            }
        }
    }

    fn handle_operation(&mut self, payload: &[u8]) {
        if payload.len() < 39 {
            return;
        }

        let op_type = payload[0];

        let mut user_id_bytes = vec![0u8; 36];
        user_id_bytes.copy_from_slice(&payload[1..37]);
        let user_id = String::from_utf8_lossy(&user_id_bytes).trim_end_matches('\0').to_string();

        let node_id_len = u16::from_le_bytes([payload[37], payload[38]]) as usize;
        if payload.len() < 39 + node_id_len {
            return;
        }

        let node_id = String::from_utf8_lossy(&payload[39..39 + node_id_len]).to_string();

        let position = if payload.len() >= 39 + node_id_len + 12 {
            let offset = 39 + node_id_len;
            Some((
                f32::from_le_bytes([
                    payload[offset],
                    payload[offset + 1],
                    payload[offset + 2],
                    payload[offset + 3],
                ]),
                f32::from_le_bytes([
                    payload[offset + 4],
                    payload[offset + 5],
                    payload[offset + 6],
                    payload[offset + 7],
                ]),
                f32::from_le_bytes([
                    payload[offset + 8],
                    payload[offset + 9],
                    payload[offset + 10],
                    payload[offset + 11],
                ]),
            ))
        } else {
            None
        };

        let operation = GraphOperation {
            id: format!("{}_{}", user_id, chrono::Utc::now().timestamp_millis()),
            op_type,
            user_id,
            node_id,
            position,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            version: 0,
        };

        debug!("Received graph operation: {:?}", operation.op_type);
        // TODO: Send to SyncManager for broadcasting
    }

    fn handle_selection(&mut self, payload: &[u8]) {
        if let Ok(text) = String::from_utf8(payload.to_vec()) {
            if let Ok(selection) = serde_json::from_str::<UserSelection>(&text) {
                debug!("Received selection: {} nodes", selection.node_ids.len());
                // TODO: Send to SyncManager for broadcasting
            }
        }
    }

    fn handle_annotation(&mut self, payload: &[u8]) {
        if let Ok(text) = String::from_utf8(payload.to_vec()) {
            if let Ok(annotation) = serde_json::from_str::<GraphAnnotation>(&text) {
                debug!("Received annotation: {}", annotation.text);
                // TODO: Send to SyncManager for broadcasting
            }
        }
    }

    fn handle_user_position(&mut self, payload: &[u8]) {
        if payload.len() < 32 {
            return;
        }

        let position = (
            f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]),
            f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]),
            f32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]),
        );

        let rotation = (
            f32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]),
            f32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]),
            f32::from_le_bytes([payload[20], payload[21], payload[22], payload[23]]),
            f32::from_le_bytes([payload[24], payload[25], payload[26], payload[27]]),
        );

        debug!("User position updated: {:?}", position);
        // TODO: Send to SyncManager for broadcasting
    }

    fn handle_vr_presence(&mut self, payload: &[u8]) {
        if payload.len() < 87 {
            return;
        }

        // Parse VR presence data (87 bytes total)
        let head_position = (
            f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]),
            f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]),
            f32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]),
        );

        let head_rotation = (
            f32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]),
            f32::from_le_bytes([payload[16], payload[17], payload[18], payload[19]]),
            f32::from_le_bytes([payload[20], payload[21], payload[22], payload[23]]),
            f32::from_le_bytes([payload[24], payload[25], payload[26], payload[27]]),
        );

        let hand_flags = payload[28];
        let has_left_hand = (hand_flags & 1) != 0;
        let has_right_hand = (hand_flags & 2) != 0;

        debug!(
            "VR presence: head={:?}, hands: L={}, R={}",
            head_position, has_left_hand, has_right_hand
        );
        // TODO: Send to SyncManager for broadcasting
    }

    fn handle_subscription(&mut self, payload: &[u8]) {
        if let Ok(text) = String::from_utf8(payload.to_vec()) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(channel) = value.get("channel").and_then(|c| c.as_str()) {
                    debug!("User {} subscribed to: {}", self.user_id, channel);
                    // TODO: Send to SyncManager for subscription
                }
            }
        }
    }
}

impl Actor for CollaborativeSyncActor {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("CollaborativeSyncActor started for user: {}", self.user_id);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("CollaborativeSyncActor stopped for user: {}", self.user_id);
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for CollaborativeSyncActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Binary(bin)) => {
                self.heartbeat = Instant::now();
                self.handle_sync_message(ctx, &bin);
            }
            Ok(ws::Message::Ping(msg)) => {
                self.heartbeat = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.heartbeat = Instant::now();
            }
            Ok(ws::Message::Close(reason)) => {
                info!("WebSocket closed for user {}: {:?}", self.user_id, reason);
                ctx.stop();
            }
            _ => {}
        }
    }
}

// Message for broadcasting to connected clients
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastMessage {
    pub message_type: u8,
    pub payload: Vec<u8>,
}

impl Handler<BroadcastMessage> for CollaborativeSyncActor {
    type Result = ();

    fn handle(&mut self, msg: BroadcastMessage, ctx: &mut Self::Context) {
        let mut packet = Vec::with_capacity(1 + msg.payload.len());
        packet.push(msg.message_type);
        packet.extend_from_slice(&msg.payload);

        ctx.binary(packet);
    }
}
