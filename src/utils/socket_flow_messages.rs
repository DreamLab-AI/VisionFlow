use crate::types::vec3::Vec3Data;
use bytemuck::{Pod, Zeroable};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use glam::Vec3;
use serde::{Deserialize, Serialize};

// ===== CLIENT-SIDE BINARY DATA (28 bytes) =====
// Optimized for network transmission - contains only what clients need

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
///
///
///
pub struct BinaryNodeDataClient {
    pub node_id: u32, 
    pub x: f32,       
    pub y: f32,       
    pub z: f32,       
    pub vx: f32,      
    pub vy: f32,      
    pub vz: f32,      
}

// Compile-time assertion to ensure client format is exactly 28 bytes
static_assertions::const_assert_eq!(std::mem::size_of::<BinaryNodeDataClient>(), 28);

// Backwards compatibility alias - will be deprecated
pub type BinaryNodeData = BinaryNodeDataClient;

// ===== GPU COMPUTE BINARY DATA (48 bytes) =====
// Extended format for server-side GPU computations

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
///
///
pub struct BinaryNodeDataGPU {
    pub node_id: u32,       
    pub x: f32,             
    pub y: f32,             
    pub z: f32,             
    pub vx: f32,            
    pub vy: f32,            
    pub vz: f32,            
    pub sssp_distance: f32, 
    pub sssp_parent: i32,   
    pub cluster_id: i32,    
    pub centrality: f32,    
    pub mass: f32,          
}

// Compile-time assertion to ensure GPU format is exactly 48 bytes
static_assertions::const_assert_eq!(std::mem::size_of::<BinaryNodeDataGPU>(), 48);

// Implement DeviceRepr for GPU data
unsafe impl DeviceRepr for BinaryNodeDataGPU {}
unsafe impl ValidAsZeroBits for BinaryNodeDataGPU {}

// Helper conversion functions
impl BinaryNodeDataClient {
    pub fn new(node_id: u32, position: Vec3Data, velocity: Vec3Data) -> Self {
        Self {
            node_id,
            x: position.x,
            y: position.y,
            z: position.z,
            vx: velocity.x,
            vy: velocity.y,
            vz: velocity.z,
        }
    }

    pub fn position(&self) -> Vec3Data {
        Vec3Data::new(self.x, self.y, self.z)
    }

    pub fn velocity(&self) -> Vec3Data {
        Vec3Data::new(self.vx, self.vy, self.vz)
    }
}

impl BinaryNodeDataGPU {
    pub fn to_client(&self) -> BinaryNodeDataClient {
        BinaryNodeDataClient {
            node_id: self.node_id,
            x: self.x,
            y: self.y,
            z: self.z,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
        }
    }

    pub fn from_client(client: &BinaryNodeDataClient) -> Self {
        Self {
            node_id: client.node_id,
            x: client.x,
            y: client.y,
            z: client.z,
            vx: client.vx,
            vy: client.vy,
            vz: client.vz,
            sssp_distance: f32::INFINITY,
            sssp_parent: -1,
            cluster_id: -1,
            centrality: 0.0,
            mass: 1.0,
        }
    }
}

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
