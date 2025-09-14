use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use crate::types::vec3::Vec3Data;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use glam::Vec3;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
/// Binary node data structure for server-side processing and GPU computation
///
/// **Server format (28 bytes):**
/// - position: Vec3Data (12 bytes)
/// - velocity: Vec3Data (12 bytes)
/// - mass: u8 (1 byte) - Server-side only, not transmitted over wire
/// - flags: u8 (1 byte) - Server-side only, not transmitted over wire
/// - padding: [u8; 2] (2 bytes) - Server-side only, not transmitted over wire
///
/// **Wire format (26 bytes) is handled separately by `WireNodeDataItem` in `binary_protocol.rs`:**
/// - id: u16 (2 bytes) - With node type flags in high bits
/// - position: Vec3Data (12 bytes)
/// - velocity: Vec3Data (12 bytes)
///
/// The wire and server formats are distinct to optimize bandwidth while preserving
/// server-side physics properties (mass, flags) that are not needed on the client.
pub struct BinaryNodeData {
    pub position: Vec3Data,
    pub velocity: Vec3Data,
    pub mass: u8,      // Server-side only, not transmitted over wire
    pub flags: u8,     // Server-side only, not transmitted over wire
    pub padding: [u8; 2], // Server-side only, not transmitted over wire
}

// Compile-time assertion to ensure server format is exactly 28 bytes
static_assertions::const_assert_eq!(std::mem::size_of::<BinaryNodeData>(), 28);

// Implement DeviceRepr for BinaryNodeData
unsafe impl DeviceRepr for BinaryNodeData {}

// Implement ValidAsZeroBits for BinaryNodeData
unsafe impl ValidAsZeroBits for BinaryNodeData {}

#[derive(Debug, Serialize, Deserialize)]
pub struct PingMessage {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(default = "default_timestamp")]
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PongMessage {
    #[serde(rename = "type")]
    pub type_: String,
    pub timestamp: u64,
}

fn default_timestamp() -> u64 {
    chrono::Utc::now().timestamp_millis() as u64
}

// SocketNode has been consolidated into models::node::Node
// All socket communication now uses the canonical Node type with conversion helpers

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    #[serde(rename = "ping")]
    Ping { timestamp: u64 },
    
    #[serde(rename = "pong")]
    Pong { timestamp: u64 },
    
    #[serde(rename = "enableRandomization")]
    EnableRandomization { enabled: bool },
}

// Helper functions to convert between Vec3Data and [f32; 3] for GPU computations
#[inline]
pub fn vec3data_to_array(vec: &Vec3Data) -> [f32; 3] {
    [vec.x, vec.y, vec.z]
}

#[inline]
pub fn array_to_vec3data(arr: [f32; 3]) -> Vec3Data {
    Vec3Data::new(arr[0], arr[1], arr[2])
}

#[inline]
pub fn vec3data_to_glam(vec: &Vec3Data) -> Vec3 {
    Vec3::new(vec.x, vec.y, vec.z)
}

#[inline]
pub fn glam_to_vec3data(vec: glam::Vec3) -> Vec3Data {
    Vec3Data::new(vec.x, vec.y, vec.z)
}
