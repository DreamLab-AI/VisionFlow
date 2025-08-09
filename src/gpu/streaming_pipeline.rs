//! Streaming Pipeline - Optimized for headless GPU compute to lightweight clients
//! Designed for Quest 3 and other limited-capability clients

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use bytes::{Bytes, BytesMut, BufMut};
use serde::{Serialize, Deserialize};
use log::info;

/// Simplified render packet for bandwidth-constrained clients
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimplifiedNode {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color_index: u8,      // Palette index (0-255)
    pub size: u8,              // Quantized size (0-255)
    pub importance: u8,        // Quantized importance (0-255)
    pub flags: u8,             // Bit flags for properties
}

/// Compressed edge format for streaming
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompressedEdge {
    pub source: u16,           // Support up to 65K nodes
    pub target: u16,
    pub weight: u8,            // Quantized weight
    pub bundling_id: u8,       // Edge bundle group
}

/// Level-of-detail settings based on client capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientLOD {
    /// Quest 3 - highly optimized
    Mobile {
        max_nodes: usize,      // ~1000 nodes
        max_edges: usize,      // ~2000 edges
        update_rate: u32,      // 30 FPS
        compression: bool,     // Heavy compression
    },
    /// Desktop VR - medium quality
    DesktopVR {
        max_nodes: usize,      // ~10000 nodes
        max_edges: usize,      // ~20000 edges
        update_rate: u32,      // 60 FPS
        compression: bool,     // Moderate compression
    },
    /// High-end workstation - full quality
    Workstation {
        max_nodes: usize,      // 100000+ nodes
        max_edges: usize,      // 200000+ edges
        update_rate: u32,      // 120 FPS
        compression: bool,     // Minimal compression
    },
}

impl ClientLOD {
    pub fn quest3() -> Self {
        ClientLOD::Mobile {
            max_nodes: 1000,
            max_edges: 2000,
            update_rate: 30,
            compression: true,
        }
    }
}

/// Streaming pipeline from GPU to clients
pub struct StreamingPipeline {
    /// GPU compute results channel
    gpu_receiver: mpsc::Receiver<RenderData>,
    
    /// Client connections with their LOD settings
    clients: Arc<RwLock<Vec<ClientConnection>>>,
    
    /// Frame buffer for temporal coherence
    frame_buffer: Arc<RwLock<FrameBuffer>>,
    
    /// Importance-based culling threshold
    importance_threshold: f32,
}

struct ClientConnection {
    id: String,
    lod: ClientLOD,
    sender: mpsc::Sender<Bytes>,
    last_frame: u32,
    position: Option<[f32; 3]>,  // Client camera position for culling
}

struct FrameBuffer {
    current_frame: u32,
    full_positions: Vec<f32>,
    full_colors: Vec<f32>,
    full_importance: Vec<f32>,
    node_count: usize,
}

/// Render data from GPU
pub struct RenderData {
    pub positions: Vec<f32>,
    pub colors: Vec<f32>,
    pub importance: Vec<f32>,
    pub frame: u32,
}

impl StreamingPipeline {
    pub fn new(gpu_receiver: mpsc::Receiver<RenderData>) -> Self {
        Self {
            gpu_receiver,
            clients: Arc::new(RwLock::new(Vec::new())),
            frame_buffer: Arc::new(RwLock::new(FrameBuffer {
                current_frame: 0,
                full_positions: Vec::new(),
                full_colors: Vec::new(),
                full_importance: Vec::new(),
                node_count: 0,
            })),
            importance_threshold: 0.1,
        }
    }
    
    /// Add a new client connection
    pub async fn add_client(&self, id: String, lod: ClientLOD) -> mpsc::Receiver<Bytes> {
        let (tx, rx) = mpsc::channel(10);
        
        let mut clients = self.clients.write().await;
        clients.push(ClientConnection {
            id,
            lod,
            sender: tx,
            last_frame: 0,
            position: None,
        });
        
        rx
    }
    
    /// Main streaming loop
    pub async fn run(&mut self) {
        while let Some(render_data) = self.gpu_receiver.recv().await {
            // Update frame buffer
            {
                let mut buffer = self.frame_buffer.write().await;
                buffer.current_frame = render_data.frame;
                buffer.full_positions = render_data.positions;
                buffer.full_colors = render_data.colors;
                buffer.full_importance = render_data.importance;
                buffer.node_count = buffer.full_positions.len() / 4;
            }
            
            // Stream to each client based on their LOD
            let clients = self.clients.read().await;
            let buffer = self.frame_buffer.read().await;
            
            for client in clients.iter() {
                // Skip if client is not ready for update
                if !should_update_client(client, buffer.current_frame) {
                    continue;
                }
                
                // Generate LOD-appropriate packet
                let packet = match &client.lod {
                    ClientLOD::Mobile { max_nodes, .. } => {
                        self.create_mobile_packet(&buffer, *max_nodes, client.position).await
                    },
                    ClientLOD::DesktopVR { max_nodes, .. } => {
                        self.create_desktop_packet(&buffer, *max_nodes, client.position).await
                    },
                    ClientLOD::Workstation { .. } => {
                        self.create_workstation_packet(&buffer).await
                    },
                };
                
                // Send packet to client
                let _ = client.sender.send(packet).await;
            }
        }
    }
    
    /// Create highly optimized packet for mobile/Quest 3
    async fn create_mobile_packet(
        &self,
        buffer: &FrameBuffer,
        max_nodes: usize,
        client_position: Option<[f32; 3]>,
    ) -> Bytes {
        let mut packet = BytesMut::new();
        
        // Header
        packet.put_u8(1);  // Packet version
        packet.put_u32_le(buffer.current_frame);
        
        // Importance-based culling
        let mut nodes: Vec<(usize, f32)> = buffer.full_importance
            .iter()
            .enumerate()
            .filter(|(_, &imp)| imp > self.importance_threshold)
            .map(|(i, &imp)| (i, imp))
            .collect();
        
        // Sort by importance and take top N
        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        nodes.truncate(max_nodes);
        
        // If client has position, apply distance culling
        if let Some(cam_pos) = client_position {
            nodes.retain(|(idx, _)| {
                let x = buffer.full_positions[idx * 4];
                let y = buffer.full_positions[idx * 4 + 1];
                let z = buffer.full_positions[idx * 4 + 2];
                
                let dist_sq = (x - cam_pos[0]).powi(2) + 
                             (y - cam_pos[1]).powi(2) + 
                             (z - cam_pos[2]).powi(2);
                
                dist_sq < 10000.0  // 100 unit radius
            });
        }
        
        // Write node count
        packet.put_u16_le(nodes.len() as u16);
        
        // Write simplified nodes
        for (idx, importance) in nodes {
            let x = buffer.full_positions[idx * 4];
            let y = buffer.full_positions[idx * 4 + 1];
            let z = buffer.full_positions[idx * 4 + 2];
            
            // Quantize position to 16-bit per component (6 bytes instead of 12)
            packet.put_i16_le((x * 100.0) as i16);
            packet.put_i16_le((y * 100.0) as i16);
            packet.put_i16_le((z * 100.0) as i16);
            
            // Quantize color to palette index
            let hue = buffer.full_colors[idx * 4];
            packet.put_u8((hue * 255.0) as u8);
            
            // Quantize size and importance
            packet.put_u8((importance * 255.0) as u8);
        }
        
        packet.freeze()
    }
    
    /// Create medium-quality packet for desktop VR
    async fn create_desktop_packet(
        &self,
        buffer: &FrameBuffer,
        max_nodes: usize,
        client_position: Option<[f32; 3]>,
    ) -> Bytes {
        let mut packet = BytesMut::new();
        
        // Header
        packet.put_u8(2);  // Packet version
        packet.put_u32_le(buffer.current_frame);
        
        // Apply LOD based on importance
        let mut nodes: Vec<usize> = (0..buffer.node_count.min(max_nodes))
            .filter(|&i| buffer.full_importance[i] > self.importance_threshold * 0.5)
            .collect();
        
        // Spatial culling if camera position available
        if let Some(cam_pos) = client_position {
            nodes.retain(|&idx| {
                let x = buffer.full_positions[idx * 4];
                let y = buffer.full_positions[idx * 4 + 1];
                let z = buffer.full_positions[idx * 4 + 2];
                
                let dist_sq = (x - cam_pos[0]).powi(2) + 
                             (y - cam_pos[1]).powi(2) + 
                             (z - cam_pos[2]).powi(2);
                
                dist_sq < 40000.0  // 200 unit radius
            });
        }
        
        packet.put_u32_le(nodes.len() as u32);
        
        // Write nodes with moderate compression
        for idx in nodes {
            // Position (full precision)
            packet.put_f32_le(buffer.full_positions[idx * 4]);
            packet.put_f32_le(buffer.full_positions[idx * 4 + 1]);
            packet.put_f32_le(buffer.full_positions[idx * 4 + 2]);
            
            // Color (HSV, 3 bytes)
            packet.put_u8((buffer.full_colors[idx * 4] * 255.0) as u8);
            packet.put_u8((buffer.full_colors[idx * 4 + 1] * 255.0) as u8);
            packet.put_u8((buffer.full_colors[idx * 4 + 2] * 255.0) as u8);
            
            // Importance
            packet.put_u8((buffer.full_importance[idx] * 255.0) as u8);
        }
        
        packet.freeze()
    }
    
    /// Create full-quality packet for workstations
    async fn create_workstation_packet(&self, buffer: &FrameBuffer) -> Bytes {
        let mut packet = BytesMut::new();
        
        // Header
        packet.put_u8(3);  // Packet version
        packet.put_u32_le(buffer.current_frame);
        packet.put_u32_le(buffer.node_count as u32);
        
        // Write full precision data
        for i in 0..buffer.node_count {
            // Position (x, y, z, t)
            packet.put_f32_le(buffer.full_positions[i * 4]);
            packet.put_f32_le(buffer.full_positions[i * 4 + 1]);
            packet.put_f32_le(buffer.full_positions[i * 4 + 2]);
            packet.put_f32_le(buffer.full_positions[i * 4 + 3]);
            
            // Color (HSVA)
            packet.put_f32_le(buffer.full_colors[i * 4]);
            packet.put_f32_le(buffer.full_colors[i * 4 + 1]);
            packet.put_f32_le(buffer.full_colors[i * 4 + 2]);
            packet.put_f32_le(buffer.full_colors[i * 4 + 3]);
            
            // Importance
            packet.put_f32_le(buffer.full_importance[i]);
        }
        
        packet.freeze()
    }
}

fn should_update_client(client: &ClientConnection, current_frame: u32) -> bool {
    let frame_delta = current_frame - client.last_frame;
    
    match &client.lod {
        ClientLOD::Mobile { update_rate, .. } => {
            frame_delta >= (120 / update_rate)  // Skip frames for lower rates
        },
        ClientLOD::DesktopVR { update_rate, .. } => {
            frame_delta >= (120 / update_rate)
        },
        ClientLOD::Workstation { .. } => true,  // Always update
    }
}

/// Delta compression for bandwidth optimization
pub struct DeltaCompressor {
    previous_frame: Option<Vec<SimplifiedNode>>,
    keyframe_interval: u32,
    current_frame: u32,
}

impl DeltaCompressor {
    pub fn new(keyframe_interval: u32) -> Self {
        Self {
            previous_frame: None,
            keyframe_interval,
            current_frame: 0,
        }
    }
    
    pub fn compress(&mut self, nodes: Vec<SimplifiedNode>) -> Bytes {
        self.current_frame += 1;
        
        let mut packet = BytesMut::new();
        
        // Check if keyframe needed
        if self.current_frame % self.keyframe_interval == 0 || self.previous_frame.is_none() {
            // Send keyframe
            packet.put_u8(0xFF);  // Keyframe marker
            packet.put_u32_le(nodes.len() as u32);
            
            for node in &nodes {
                packet.put_f32_le(node.x);
                packet.put_f32_le(node.y);
                packet.put_f32_le(node.z);
                packet.put_u8(node.color_index);
                packet.put_u8(node.size);
                packet.put_u8(node.importance);
                packet.put_u8(node.flags);
            }
            
            self.previous_frame = Some(nodes);
        } else {
            // Send delta frame
            packet.put_u8(0xFE);  // Delta marker
            
            let prev = self.previous_frame.as_ref().unwrap();
            let mut deltas = Vec::new();
            
            for (i, (curr, prev)) in nodes.iter().zip(prev.iter()).enumerate() {
                let dx = curr.x - prev.x;
                let dy = curr.y - prev.y;
                let dz = curr.z - prev.z;
                
                // Only send if changed significantly
                if dx.abs() > 0.01 || dy.abs() > 0.01 || dz.abs() > 0.01 ||
                   curr.color_index != prev.color_index ||
                   curr.importance != prev.importance {
                    deltas.push((i as u16, dx, dy, dz, curr.color_index, curr.importance));
                }
            }
            
            packet.put_u16_le(deltas.len() as u16);
            
            for (idx, dx, dy, dz, color, importance) in deltas {
                packet.put_u16_le(idx);
                packet.put_i16_le((dx * 1000.0) as i16);  // Quantized delta
                packet.put_i16_le((dy * 1000.0) as i16);
                packet.put_i16_le((dz * 1000.0) as i16);
                packet.put_u8(color);
                packet.put_u8(importance);
            }
            
            self.previous_frame = Some(nodes);
        }
        
        packet.freeze()
    }
}

/// WebSocket message for control and data
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamMessage {
    /// Client capability declaration
    ClientCapability {
        device: String,
        lod: ClientLOD,
        position: Option<[f32; 3]>,
    },
    
    /// Focus request from client
    FocusRequest {
        node_id: Option<u32>,
        position: [f32; 3],
        radius: f32,
    },
    
    /// Performance metrics
    Metrics {
        fps: f32,
        latency_ms: f32,
        bandwidth_kbps: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simplified_node_size() {
        assert_eq!(std::mem::size_of::<SimplifiedNode>(), 8);
    }
    
    #[test]
    fn test_compressed_edge_size() {
        assert_eq!(std::mem::size_of::<CompressedEdge>(), 6);
    }
    
    #[test]
    fn test_delta_compression() {
        let mut compressor = DeltaCompressor::new(30);
        
        let nodes = vec![SimplifiedNode {
            x: 1.0, y: 2.0, z: 3.0,
            color_index: 10,
            size: 50,
            importance: 128,
            flags: 0,
        }];
        
        let compressed = compressor.compress(nodes);
        assert!(compressed.len() > 0);
    }
}