//! Client-domain messages: WebSocket client registration, broadcast,
//! authentication, filtering, initial graph load, and position streaming.

use actix::prelude::*;
use serde::{Deserialize, Serialize};

use crate::utils::socket_flow_messages::{InitialEdgeData, InitialNodeData};

// ---------------------------------------------------------------------------
// Client Manager Actor Messages
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "Result<usize, String>")]
pub struct RegisterClient {
    pub addr: actix::Addr<crate::handlers::socket_flow_handler::SocketFlowServer>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UnregisterClient {
    pub client_id: usize,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BroadcastNodePositions {
    pub positions: Vec<u8>,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastPositions {
    pub positions: Vec<crate::utils::socket_flow_messages::BinaryNodeDataClient>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct BroadcastMessage {
    pub message: String,
}

#[derive(Message)]
#[rtype(result = "Result<usize, String>")]
pub struct GetClientCount;

// WEBSOCKET SETTLING FIX: Message to force immediate position broadcast for new clients
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ForcePositionBroadcast {
    pub reason: String,
}

// UNIFIED INIT: Message to coordinate REST-triggered broadcasts
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitialClientSync {
    pub client_identifier: String,
    pub trigger_source: String,
}

// WEBSOCKET SETTLING FIX: Message to set graph service address in client manager
#[derive(Message)]
#[rtype(result = "()")]
pub struct SetGraphServiceAddress {
    pub addr: actix::Addr<crate::actors::GraphServiceSupervisor>,
}

// Messages for ClientManagerActor to send to individual SocketFlowServer clients
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendToClientBinary(pub Vec<u8>);

#[derive(Message)]
#[rtype(result = "()")]
pub struct SendToClientText(pub String);

// ---------------------------------------------------------------------------
// Client authentication
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct AuthenticateClient {
    pub client_id: usize,
    pub pubkey: String,
    pub is_power_user: bool,
}

// ---------------------------------------------------------------------------
// Client filter settings
// ---------------------------------------------------------------------------

#[derive(Message, Clone, Serialize, Deserialize)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateClientFilter {
    pub client_id: usize,
    pub enabled: bool,
    pub quality_threshold: f64,
    pub authority_threshold: f64,
    pub filter_by_quality: bool,
    pub filter_by_authority: bool,
    pub filter_mode: String,
    pub max_nodes: Option<i32>,
}

// ---------------------------------------------------------------------------
// WebSocket protocol: Initial graph load
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "()")]
pub struct SendInitialGraphLoad {
    pub nodes: Vec<InitialNodeData>,
    pub edges: Vec<InitialEdgeData>,
}

// WebSocket protocol: Streamed position updates indexed by node ID
#[derive(Message)]
#[rtype(result = "()")]
pub struct SendPositionUpdate {
    pub node_id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

// ---------------------------------------------------------------------------
// Client broadcast acknowledgement (end-to-end flow control)
// ---------------------------------------------------------------------------

/// Client-originated broadcast acknowledgement for true end-to-end flow control
/// Sent by WebSocket clients after they process position updates
#[derive(Message, Debug, Clone)]
#[rtype(result = "()")]
pub struct ClientBroadcastAck {
    /// Sequence ID from the original broadcast (correlates with GPU broadcast sequence)
    pub sequence_id: u64,
    /// Number of nodes the client actually processed
    pub nodes_received: u32,
    /// Client receive timestamp (ms since epoch)
    pub timestamp: u64,
    /// Client ID that sent this ACK (set by handler)
    pub client_id: Option<usize>,
}
