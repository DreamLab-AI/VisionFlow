//! Safe Streaming Pipeline with Comprehensive Bounds Checking
//! 
//! Enhanced version of the streaming pipeline with comprehensive GPU safety measures,
//! memory bounds checking, and overflow protection.

use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use bytes::{Bytes, BytesMut, BufMut};
use serde::{Serialize, Deserialize};
use log::{debug, warn, error, info};
use std::time::Instant;

use crate::utils::gpu_safety::{GPUSafetyValidator, GPUSafetyConfig, GPUSafetyError};
use crate::utils::memory_bounds::{ThreadSafeMemoryBoundsChecker, MemoryBounds, SafeArrayAccess};

/// Safe simplified render packet with validation
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SafeSimplifiedNode {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color_index: u8,
    pub size: u8,
    pub importance: u8,
    pub flags: u8,
}

impl SafeSimplifiedNode {
    /// Validate node data for safety
    pub fn validate(&self) -> Result<(), GPUSafetyError> {
        // Check for infinite or NaN values
        if !self.x.is_finite() || !self.y.is_finite() || !self.z.is_finite() {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Invalid position coordinates: ({}, {}, {})", self.x, self.y, self.z),
            });
        }

        // Check for reasonable bounds (prevent extreme values)
        const MAX_COORD: f32 = 1e6;
        if self.x.abs() > MAX_COORD || self.y.abs() > MAX_COORD || self.z.abs() > MAX_COORD {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Coordinates exceed safe bounds: ({}, {}, {})", self.x, self.y, self.z),
            });
        }

        Ok(())
    }

    /// Create from raw values with validation
    pub fn new(x: f32, y: f32, z: f32, color_index: u8, size: u8, importance: u8, flags: u8) -> Result<Self, GPUSafetyError> {
        let node = Self { x, y, z, color_index, size, importance, flags };
        node.validate()?;
        Ok(node)
    }
}

/// Safe compressed edge with validation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SafeCompressedEdge {
    pub source: u16,
    pub target: u16,
    pub weight: u8,
    pub bundling_id: u8,
}

impl SafeCompressedEdge {
    /// Validate edge indices against node count
    pub fn validate(&self, max_nodes: usize) -> Result<(), GPUSafetyError> {
        if self.source as usize >= max_nodes {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: self.source as usize,
                size: max_nodes,
            });
        }

        if self.target as usize >= max_nodes {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: self.target as usize,
                size: max_nodes,
            });
        }

        // Prevent self-loops in compressed format
        if self.source == self.target {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Self-loop detected in compressed edge: {} -> {}", self.source, self.target),
            });
        }

        Ok(())
    }
}

/// Safe client LOD with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafeClientLOD {
    Mobile {
        max_nodes: usize,
        max_edges: usize,
        update_rate: u32,
        compression: bool,
    },
    DesktopVR {
        max_nodes: usize,
        max_edges: usize,
        update_rate: u32,
        compression: bool,
    },
    Workstation {
        max_nodes: usize,
        max_edges: usize,
        update_rate: u32,
        compression: bool,
    },
}

impl SafeClientLOD {
    pub fn validate(&self) -> Result<(), GPUSafetyError> {
        let (max_nodes, max_edges, update_rate) = match self {
            SafeClientLOD::Mobile { max_nodes, max_edges, update_rate, .. } => (*max_nodes, *max_edges, *update_rate),
            SafeClientLOD::DesktopVR { max_nodes, max_edges, update_rate, .. } => (*max_nodes, *max_edges, *update_rate),
            SafeClientLOD::Workstation { max_nodes, max_edges, update_rate, .. } => (*max_nodes, *max_edges, *update_rate),
        };

        // Validate reasonable limits
        if max_nodes > 10_000_000 {
            return Err(GPUSafetyError::ResourceExhaustion {
                resource: "max_nodes".to_string(),
                current: max_nodes,
                limit: 10_000_000,
            });
        }

        if max_edges > 50_000_000 {
            return Err(GPUSafetyError::ResourceExhaustion {
                resource: "max_edges".to_string(),
                current: max_edges,
                limit: 50_000_000,
            });
        }

        if update_rate > 240 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Update rate {} exceeds maximum of 240 FPS", update_rate),
            });
        }

        if update_rate == 0 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: "Update rate must be greater than 0".to_string(),
            });
        }

        Ok(())
    }

    pub fn quest3() -> Result<Self, GPUSafetyError> {
        let lod = SafeClientLOD::Mobile {
            max_nodes: 1000,
            max_edges: 2000,
            update_rate: 30,
            compression: true,
        };
        lod.validate()?;
        Ok(lod)
    }

    pub fn max_nodes(&self) -> usize {
        match self {
            SafeClientLOD::Mobile { max_nodes, .. } => *max_nodes,
            SafeClientLOD::DesktopVR { max_nodes, .. } => *max_nodes,
            SafeClientLOD::Workstation { max_nodes, .. } => *max_nodes,
        }
    }

    pub fn max_edges(&self) -> usize {
        match self {
            SafeClientLOD::Mobile { max_edges, .. } => *max_edges,
            SafeClientLOD::DesktopVR { max_edges, .. } => *max_edges,
            SafeClientLOD::Workstation { max_edges, .. } => *max_edges,
        }
    }
}

/// Safe frame buffer with bounds checking
pub struct SafeFrameBuffer {
    current_frame: u32,
    positions: SafeArrayAccess<f32>,
    colors: SafeArrayAccess<f32>,
    importance: SafeArrayAccess<f32>,
    node_count: usize,
    bounds_checker: Arc<ThreadSafeMemoryBoundsChecker>,
}

impl SafeFrameBuffer {
    pub fn new(max_nodes: usize, bounds_checker: Arc<ThreadSafeMemoryBoundsChecker>) -> Result<Self, GPUSafetyError> {
        // Validate buffer sizes
        if max_nodes > 10_000_000 {
            return Err(GPUSafetyError::ResourceExhaustion {
                resource: "max_nodes".to_string(),
                current: max_nodes,
                limit: 10_000_000,
            });
        }

        // Check for potential overflow in allocation size
        let positions_size = max_nodes.checked_mul(4)
            .ok_or_else(|| GPUSafetyError::InvalidBufferSize {
                requested: max_nodes,
                max_allowed: usize::MAX / 4,
            })?;

        let colors_size = max_nodes.checked_mul(4)
            .ok_or_else(|| GPUSafetyError::InvalidBufferSize {
                requested: max_nodes,
                max_allowed: usize::MAX / 4,
            })?;

        // Register memory allocations
        bounds_checker.register_allocation(MemoryBounds::new(
            "frame_buffer_positions".to_string(),
            positions_size * std::mem::size_of::<f32>(),
            std::mem::size_of::<f32>(),
            std::mem::align_of::<f32>(),
        ))?;

        bounds_checker.register_allocation(MemoryBounds::new(
            "frame_buffer_colors".to_string(),
            colors_size * std::mem::size_of::<f32>(),
            std::mem::size_of::<f32>(),
            std::mem::align_of::<f32>(),
        ))?;

        bounds_checker.register_allocation(MemoryBounds::new(
            "frame_buffer_importance".to_string(),
            max_nodes * std::mem::size_of::<f32>(),
            std::mem::size_of::<f32>(),
            std::mem::align_of::<f32>(),
        ))?;

        let positions = SafeArrayAccess::new(vec![0.0f32; positions_size], "frame_buffer_positions".to_string())
            .with_bounds_checker(bounds_checker.clone());

        let colors = SafeArrayAccess::new(vec![0.0f32; colors_size], "frame_buffer_colors".to_string())
            .with_bounds_checker(bounds_checker.clone());

        let importance = SafeArrayAccess::new(vec![0.0f32; max_nodes], "frame_buffer_importance".to_string())
            .with_bounds_checker(bounds_checker.clone());

        Ok(Self {
            current_frame: 0,
            positions,
            colors,
            importance,
            node_count: 0,
            bounds_checker,
        })
    }

    pub fn update_data(&mut self, positions: &[f32], colors: &[f32], importance: &[f32], frame: u32) -> Result<(), GPUSafetyError> {
        // Validate input sizes
        if positions.len() % 4 != 0 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Position array length {} is not divisible by 4", positions.len()),
            });
        }

        if colors.len() % 4 != 0 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Color array length {} is not divisible by 4", colors.len()),
            });
        }

        let node_count = positions.len() / 4;
        
        if colors.len() / 4 != node_count {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Color array represents {} nodes but position array represents {} nodes", 
                               colors.len() / 4, node_count),
            });
        }

        if importance.len() != node_count {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Importance array length {} doesn't match node count {}", 
                               importance.len(), node_count),
            });
        }

        // Validate buffer capacity
        if positions.len() > self.positions.len() {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: positions.len(),
                size: self.positions.len(),
            });
        }

        if colors.len() > self.colors.len() {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: colors.len(),
                size: self.colors.len(),
            });
        }

        if importance.len() > self.importance.len() {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: importance.len(),
                size: self.importance.len(),
            });
        }

        // Validate data values
        for (i, &val) in positions.iter().enumerate() {
            if !val.is_finite() {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Invalid position value at index {}: {}", i, val),
                });
            }
        }

        for (i, &val) in colors.iter().enumerate() {
            if !val.is_finite() {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Invalid color value at index {}: {}", i, val),
                });
            }
        }

        for (i, &val) in importance.iter().enumerate() {
            if !val.is_finite() || val < 0.0 {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Invalid importance value at index {}: {}", i, val),
                });
            }
        }

        // Update frame data safely
        self.current_frame = frame;
        self.node_count = node_count;

        // Copy data with bounds checking (this is done implicitly by SafeArrayAccess)
        for i in 0..positions.len() {
            *self.positions.get_mut(i).map_err(|e| GPUSafetyError::DeviceError {
                message: format!("Failed to update position {}: {}", i, e),
            })? = positions[i];
        }

        for i in 0..colors.len() {
            *self.colors.get_mut(i).map_err(|e| GPUSafetyError::DeviceError {
                message: format!("Failed to update color {}: {}", i, e),
            })? = colors[i];
        }

        for i in 0..importance.len() {
            *self.importance.get_mut(i).map_err(|e| GPUSafetyError::DeviceError {
                message: format!("Failed to update importance {}: {}", i, e),
            })? = importance[i];
        }

        debug!("Frame buffer updated: frame={}, nodes={}", frame, node_count);
        Ok(())
    }

    pub fn get_current_frame(&self) -> u32 {
        self.current_frame
    }

    pub fn get_node_count(&self) -> usize {
        self.node_count
    }

    pub fn get_position(&self, node_index: usize, component: usize) -> Result<f32, GPUSafetyError> {
        if component >= 4 {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: component,
                size: 4,
            });
        }

        let pos_index = node_index * 4 + component;
        self.positions.get(pos_index)
            .map(|&val| val)
            .map_err(|e| GPUSafetyError::DeviceError {
                message: format!("Failed to get position: {}", e),
            })
    }

    pub fn get_importance(&self, node_index: usize) -> Result<f32, GPUSafetyError> {
        self.importance.get(node_index)
            .map(|&val| val)
            .map_err(|e| GPUSafetyError::DeviceError {
                message: format!("Failed to get importance: {}", e),
            })
    }
}

/// Safe client connection with validation
pub struct SafeClientConnection {
    id: String,
    lod: SafeClientLOD,
    sender: mpsc::Sender<Bytes>,
    last_frame: u32,
    position: Option<[f32; 3]>,
    packet_count: u64,
    bytes_sent: u64,
    last_packet_time: Option<Instant>,
}

impl SafeClientConnection {
    pub fn new(id: String, lod: SafeClientLOD, sender: mpsc::Sender<Bytes>) -> Result<Self, GPUSafetyError> {
        lod.validate()?;
        
        if id.is_empty() {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: "Client ID cannot be empty".to_string(),
            });
        }

        Ok(Self {
            id,
            lod,
            sender,
            last_frame: 0,
            position: None,
            packet_count: 0,
            bytes_sent: 0,
            last_packet_time: None,
        })
    }

    pub fn update_position(&mut self, position: [f32; 3]) -> Result<(), GPUSafetyError> {
        // Validate position values
        for &coord in &position {
            if !coord.is_finite() {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Invalid position coordinate: {}", coord),
                });
            }
        }

        self.position = Some(position);
        debug!("Updated client {} position: {:?}", self.id, position);
        Ok(())
    }

    pub async fn send_packet(&mut self, packet: Bytes) -> Result<(), GPUSafetyError> {
        // Validate packet size
        const MAX_PACKET_SIZE: usize = 10 * 1024 * 1024; // 10MB limit
        if packet.len() > MAX_PACKET_SIZE {
            return Err(GPUSafetyError::ResourceExhaustion {
                resource: "packet_size".to_string(),
                current: packet.len(),
                limit: MAX_PACKET_SIZE,
            });
        }

        // Check send queue capacity
        if self.sender.capacity() == 0 && self.sender.try_send(packet.clone()).is_err() {
            warn!("Client {} send queue full, dropping packet", self.id);
            return Ok(()); // Don't fail, just drop the packet
        }

        match self.sender.send(packet.clone()).await {
            Ok(()) => {
                self.packet_count += 1;
                self.bytes_sent += packet.len() as u64;
                self.last_packet_time = Some(Instant::now());
                debug!("Sent packet to client {}: {} bytes (total: {} packets, {} bytes)", 
                       self.id, packet.len(), self.packet_count, self.bytes_sent);
                Ok(())
            },
            Err(e) => {
                error!("Failed to send packet to client {}: {}", self.id, e);
                Err(GPUSafetyError::DeviceError {
                    message: format!("Failed to send packet: {}", e),
                })
            }
        }
    }

    pub fn should_update(&self, current_frame: u32) -> bool {
        let frame_delta = current_frame.saturating_sub(self.last_frame);
        
        match &self.lod {
            SafeClientLOD::Mobile { update_rate, .. } => {
                frame_delta >= (120 / update_rate.max(1))
            },
            SafeClientLOD::DesktopVR { update_rate, .. } => {
                frame_delta >= (120 / update_rate.max(1))
            },
            SafeClientLOD::Workstation { .. } => true,
        }
    }

    pub fn mark_frame_sent(&mut self, frame: u32) {
        self.last_frame = frame;
    }

    pub fn get_stats(&self) -> ClientStats {
        ClientStats {
            id: self.id.clone(),
            packet_count: self.packet_count,
            bytes_sent: self.bytes_sent,
            last_frame: self.last_frame,
            position: self.position,
            lod_type: match self.lod {
                SafeClientLOD::Mobile { .. } => "Mobile".to_string(),
                SafeClientLOD::DesktopVR { .. } => "DesktopVR".to_string(),
                SafeClientLOD::Workstation { .. } => "Workstation".to_string(),
            },
        }
    }
}

/// Client statistics
#[derive(Debug, Clone, Serialize)]
pub struct ClientStats {
    pub id: String,
    pub packet_count: u64,
    pub bytes_sent: u64,
    pub last_frame: u32,
    pub position: Option<[f32; 3]>,
    pub lod_type: String,
}

/// Safe streaming pipeline with comprehensive bounds checking
pub struct SafeStreamingPipeline {
    gpu_receiver: mpsc::Receiver<RenderData>,
    clients: Arc<RwLock<Vec<SafeClientConnection>>>,
    frame_buffer: Arc<RwLock<SafeFrameBuffer>>,
    importance_threshold: f32,
    safety_validator: Arc<GPUSafetyValidator>,
    bounds_checker: Arc<ThreadSafeMemoryBoundsChecker>,
    stats: Arc<RwLock<PipelineStats>>,
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub frames_processed: u64,
    pub total_packets_sent: u64,
    pub total_bytes_sent: u64,
    pub active_clients: usize,
    pub last_frame_time: Option<Instant>,
    pub average_frame_time_ms: f64,
    pub errors_count: u64,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            frames_processed: 0,
            total_packets_sent: 0,
            total_bytes_sent: 0,
            active_clients: 0,
            last_frame_time: None,
            average_frame_time_ms: 0.0,
            errors_count: 0,
        }
    }
}

/// Render data from GPU
#[derive(Debug)]
pub struct RenderData {
    pub positions: Vec<f32>,
    pub colors: Vec<f32>,
    pub importance: Vec<f32>,
    pub frame: u32,
}

impl RenderData {
    pub fn validate(&self) -> Result<(), GPUSafetyError> {
        // Check array sizes
        if self.positions.len() % 4 != 0 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Position array length {} is not divisible by 4", self.positions.len()),
            });
        }

        if self.colors.len() % 4 != 0 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Color array length {} is not divisible by 4", self.colors.len()),
            });
        }

        let node_count = self.positions.len() / 4;
        
        if self.colors.len() / 4 != node_count {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Color array represents {} nodes but position array represents {} nodes", 
                               self.colors.len() / 4, node_count),
            });
        }

        if self.importance.len() != node_count {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Importance array length {} doesn't match node count {}", 
                               self.importance.len(), node_count),
            });
        }

        // Validate data values
        for (i, &val) in self.positions.iter().enumerate() {
            if !val.is_finite() {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Invalid position value at index {}: {}", i, val),
                });
            }
        }

        for (i, &val) in self.importance.iter().enumerate() {
            if !val.is_finite() || val < 0.0 {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Invalid importance value at index {}: {}", i, val),
                });
            }
        }

        Ok(())
    }
}

impl SafeStreamingPipeline {
    pub fn new(
        gpu_receiver: mpsc::Receiver<RenderData>,
        max_nodes: usize,
        safety_config: GPUSafetyConfig,
    ) -> Result<Self, GPUSafetyError> {
        let bounds_checker = Arc::new(ThreadSafeMemoryBoundsChecker::new(safety_config.max_memory_bytes));
        let safety_validator = Arc::new(GPUSafetyValidator::new(safety_config));
        
        let frame_buffer = Arc::new(RwLock::new(SafeFrameBuffer::new(max_nodes, bounds_checker.clone())?));
        
        Ok(Self {
            gpu_receiver,
            clients: Arc::new(RwLock::new(Vec::new())),
            frame_buffer,
            importance_threshold: 0.1,
            safety_validator,
            bounds_checker,
            stats: Arc::new(RwLock::new(PipelineStats::default())),
        })
    }

    pub async fn add_client(&self, id: String, lod: SafeClientLOD) -> Result<mpsc::Receiver<Bytes>, GPUSafetyError> {
        let (tx, rx) = mpsc::channel(10);
        
        let client = SafeClientConnection::new(id.clone(), lod, tx)?;
        
        let mut clients = self.clients.write().await;
        clients.push(client);
        
        info!("Added safe client: {}", id);
        Ok(rx)
    }

    pub async fn run(&mut self) -> Result<(), GPUSafetyError> {
        info!("Starting safe streaming pipeline");
        
        while let Some(render_data) = self.gpu_receiver.recv().await {
            let frame_start = Instant::now();
            
            // Validate incoming data
            if let Err(e) = render_data.validate() {
                error!("Invalid render data received: {}", e);
                self.record_error().await;
                continue;
            }

            // Update frame buffer safely
            {
                let mut buffer = self.frame_buffer.write().await;
                if let Err(e) = buffer.update_data(
                    &render_data.positions,
                    &render_data.colors,
                    &render_data.importance,
                    render_data.frame,
                ) {
                    error!("Failed to update frame buffer: {}", e);
                    self.record_error().await;
                    continue;
                }
            }

            // Process clients safely
            if let Err(e) = self.process_clients().await {
                error!("Error processing clients: {}", e);
                self.record_error().await;
            }

            // Update statistics
            self.update_stats(frame_start).await;
        }

        info!("Safe streaming pipeline stopped");
        Ok(())
    }

    async fn process_clients(&self) -> Result<(), GPUSafetyError> {
        let mut clients = self.clients.write().await;
        let buffer = self.frame_buffer.read().await;
        
        let current_frame = buffer.get_current_frame();
        let node_count = buffer.get_node_count();

        for client in clients.iter_mut() {
            if !client.should_update(current_frame) {
                continue;
            }

            let packet = match &client.lod {
                SafeClientLOD::Mobile { max_nodes, .. } => {
                    self.create_mobile_packet(&*buffer, *max_nodes, client.position, node_count).await?
                },
                SafeClientLOD::DesktopVR { max_nodes, .. } => {
                    self.create_desktop_packet(&*buffer, *max_nodes, client.position, node_count).await?
                },
                SafeClientLOD::Workstation { .. } => {
                    self.create_workstation_packet(&*buffer, node_count).await?
                },
            };

            if let Err(e) = client.send_packet(packet).await {
                warn!("Failed to send packet to client {}: {}", client.id, e);
                continue;
            }

            client.mark_frame_sent(current_frame);
        }

        Ok(())
    }

    async fn create_mobile_packet(
        &self,
        buffer: &SafeFrameBuffer,
        max_nodes: usize,
        client_position: Option<[f32; 3]>,
        node_count: usize,
    ) -> Result<Bytes, GPUSafetyError> {
        let mut packet = BytesMut::new();
        
        // Header with validation
        packet.put_u8(1);  // Packet version
        packet.put_u32_le(buffer.get_current_frame());
        
        // Safely collect importance-based culling
        let mut nodes: Vec<(usize, f32)> = Vec::new();
        
        for i in 0..node_count {
            let importance = buffer.get_importance(i)?;
            if importance > self.importance_threshold {
                nodes.push((i, importance));
            }
        }
        
        // Sort by importance and take top N
        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        nodes.truncate(max_nodes);
        
        // Apply distance culling if client position available
        if let Some(cam_pos) = client_position {
            nodes.retain(|(idx, _)| {
                let x = buffer.get_position(*idx, 0).unwrap_or(0.0);
                let y = buffer.get_position(*idx, 1).unwrap_or(0.0);
                let z = buffer.get_position(*idx, 2).unwrap_or(0.0);
                
                let dist_sq = (x - cam_pos[0]).powi(2) + 
                             (y - cam_pos[1]).powi(2) + 
                             (z - cam_pos[2]).powi(2);
                
                dist_sq < 10000.0  // 100 unit radius
            });
        }
        
        // Validate node count fits in packet
        if nodes.len() > u16::MAX as usize {
            return Err(GPUSafetyError::ResourceExhaustion {
                resource: "packet_nodes".to_string(),
                current: nodes.len(),
                limit: u16::MAX as usize,
            });
        }
        
        packet.put_u16_le(nodes.len() as u16);
        
        // Write simplified nodes with bounds checking
        for (idx, importance) in nodes {
            let x = buffer.get_position(idx, 0)?;
            let y = buffer.get_position(idx, 1)?;
            let z = buffer.get_position(idx, 2)?;
            
            // Quantize position to 16-bit per component (6 bytes instead of 12)
            let quantized_x = (x * 100.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let quantized_y = (y * 100.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let quantized_z = (z * 100.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            
            packet.put_i16_le(quantized_x);
            packet.put_i16_le(quantized_y);
            packet.put_i16_le(quantized_z);
            
            // Quantize color to palette index (simplified)
            let hue = buffer.get_position(idx, 0).unwrap_or(0.0);
            let color_index = (hue.abs() * 255.0).clamp(0.0, 255.0) as u8;
            packet.put_u8(color_index);
            
            // Quantize importance
            let importance_quantized = (importance * 255.0).clamp(0.0, 255.0) as u8;
            packet.put_u8(importance_quantized);
        }
        
        Ok(packet.freeze())
    }

    async fn create_desktop_packet(
        &self,
        buffer: &SafeFrameBuffer,
        max_nodes: usize,
        client_position: Option<[f32; 3]>,
        node_count: usize,
    ) -> Result<Bytes, GPUSafetyError> {
        let mut packet = BytesMut::new();
        
        // Header
        packet.put_u8(2);  // Packet version
        packet.put_u32_le(buffer.get_current_frame());
        
        // Apply LOD based on importance
        let mut nodes: Vec<usize> = (0..node_count.min(max_nodes))
            .filter(|&i| buffer.get_importance(i).unwrap_or(0.0) > self.importance_threshold * 0.5)
            .collect();
        
        // Spatial culling if camera position available
        if let Some(cam_pos) = client_position {
            nodes.retain(|&idx| {
                let x = buffer.get_position(idx, 0).unwrap_or(0.0);
                let y = buffer.get_position(idx, 1).unwrap_or(0.0);
                let z = buffer.get_position(idx, 2).unwrap_or(0.0);
                
                let dist_sq = (x - cam_pos[0]).powi(2) + 
                             (y - cam_pos[1]).powi(2) + 
                             (z - cam_pos[2]).powi(2);
                
                dist_sq < 40000.0  // 200 unit radius
            });
        }
        
        // Validate node count
        if nodes.len() > u32::MAX as usize {
            return Err(GPUSafetyError::ResourceExhaustion {
                resource: "packet_nodes".to_string(),
                current: nodes.len(),
                limit: u32::MAX as usize,
            });
        }
        
        packet.put_u32_le(nodes.len() as u32);
        
        // Write nodes with moderate compression
        for idx in nodes {
            // Position (full precision)
            packet.put_f32_le(buffer.get_position(idx, 0)?);
            packet.put_f32_le(buffer.get_position(idx, 1)?);
            packet.put_f32_le(buffer.get_position(idx, 2)?);
            
            // Color (simplified - using position as hue)
            let hue = buffer.get_position(idx, 0).unwrap_or(0.0);
            packet.put_u8((hue.abs() * 255.0).clamp(0.0, 255.0) as u8);
            packet.put_u8(128); // Fixed saturation
            packet.put_u8(255); // Fixed value
            
            // Importance
            let importance = buffer.get_importance(idx)?;
            packet.put_u8((importance * 255.0).clamp(0.0, 255.0) as u8);
        }
        
        Ok(packet.freeze())
    }

    async fn create_workstation_packet(
        &self,
        buffer: &SafeFrameBuffer,
        node_count: usize,
    ) -> Result<Bytes, GPUSafetyError> {
        let mut packet = BytesMut::new();
        
        // Header
        packet.put_u8(3);  // Packet version
        packet.put_u32_le(buffer.get_current_frame());
        packet.put_u32_le(node_count as u32);
        
        // Write full precision data
        for i in 0..node_count {
            // Position (x, y, z, w)
            packet.put_f32_le(buffer.get_position(i, 0)?);
            packet.put_f32_le(buffer.get_position(i, 1)?);
            packet.put_f32_le(buffer.get_position(i, 2)?);
            packet.put_f32_le(buffer.get_position(i, 3).unwrap_or(1.0)?); // w component
            
            // Color (simplified RGBA)
            let hue = buffer.get_position(i, 0).unwrap_or(0.0);
            packet.put_f32_le(hue.abs()); // R
            packet.put_f32_le(0.5); // G
            packet.put_f32_le(1.0); // B
            packet.put_f32_le(1.0); // A
            
            // Importance
            packet.put_f32_le(buffer.get_importance(i)?);
        }
        
        Ok(packet.freeze())
    }

    async fn update_stats(&self, frame_start: Instant) {
        if let Ok(mut stats) = self.stats.write().await {
            stats.frames_processed += 1;
            
            let frame_time = frame_start.elapsed();
            let frame_time_ms = frame_time.as_secs_f64() * 1000.0;
            
            if stats.frames_processed == 1 {
                stats.average_frame_time_ms = frame_time_ms;
            } else {
                // Exponential moving average
                stats.average_frame_time_ms = stats.average_frame_time_ms * 0.9 + frame_time_ms * 0.1;
            }
            
            stats.last_frame_time = Some(frame_start);
            
            // Update client count
            if let Ok(clients) = self.clients.read().await {
                stats.active_clients = clients.len();
            }
        }
    }

    async fn record_error(&self) {
        if let Ok(mut stats) = self.stats.write().await {
            stats.errors_count += 1;
        }
        self.safety_validator.record_failure();
    }

    pub async fn get_pipeline_stats(&self) -> Option<PipelineStats> {
        self.stats.read().await.ok().map(|stats| stats.clone())
    }

    pub async fn get_client_stats(&self) -> Vec<ClientStats> {
        if let Ok(clients) = self.clients.read().await {
            clients.iter().map(|client| client.get_stats()).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_memory_usage(&self) -> Option<crate::utils::memory_bounds::MemoryUsageReport> {
        self.bounds_checker.get_usage_report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_safe_simplified_node_validation() {
        // Valid node
        let valid_node = SafeSimplifiedNode::new(1.0, 2.0, 3.0, 10, 20, 30, 0);
        assert!(valid_node.is_ok());

        // Invalid coordinates
        let invalid_node = SafeSimplifiedNode::new(f32::NAN, 2.0, 3.0, 10, 20, 30, 0);
        assert!(invalid_node.is_err());

        // Extreme coordinates
        let extreme_node = SafeSimplifiedNode::new(1e7, 2.0, 3.0, 10, 20, 30, 0);
        assert!(extreme_node.is_err());
    }

    #[test]
    fn test_safe_client_lod_validation() {
        // Valid LOD
        let valid_lod = SafeClientLOD::Mobile {
            max_nodes: 1000,
            max_edges: 2000,
            update_rate: 30,
            compression: true,
        };
        assert!(valid_lod.validate().is_ok());

        // Invalid update rate
        let invalid_lod = SafeClientLOD::Mobile {
            max_nodes: 1000,
            max_edges: 2000,
            update_rate: 0,
            compression: true,
        };
        assert!(invalid_lod.validate().is_err());

        // Excessive node count
        let excessive_lod = SafeClientLOD::Mobile {
            max_nodes: 20_000_000,
            max_edges: 2000,
            update_rate: 30,
            compression: true,
        };
        assert!(excessive_lod.validate().is_err());
    }

    #[tokio::test]
    async fn test_safe_frame_buffer() {
        let bounds_checker = Arc::new(ThreadSafeMemoryBoundsChecker::new(1024 * 1024 * 1024));
        let mut buffer = SafeFrameBuffer::new(100, bounds_checker).unwrap();

        let positions = vec![1.0f32; 400]; // 100 nodes * 4 components
        let colors = vec![0.5f32; 400];
        let importance = vec![0.8f32; 100];

        assert!(buffer.update_data(&positions, &colors, &importance, 1).is_ok());
        assert_eq!(buffer.get_current_frame(), 1);
        assert_eq!(buffer.get_node_count(), 100);

        // Test out of bounds access
        assert!(buffer.get_position(150, 0).is_err());
        assert!(buffer.get_importance(150).is_err());

        // Test valid access
        assert!(buffer.get_position(50, 0).is_ok());
        assert!(buffer.get_importance(50).is_ok());
    }

    #[tokio::test]
    async fn test_render_data_validation() {
        // Valid render data
        let valid_data = RenderData {
            positions: vec![1.0f32; 40], // 10 nodes
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(valid_data.validate().is_ok());

        // Invalid positions length
        let invalid_data = RenderData {
            positions: vec![1.0f32; 39], // Not divisible by 4
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 10],
            frame: 1,
        };
        assert!(invalid_data.validate().is_err());

        // Mismatched array sizes
        let mismatched_data = RenderData {
            positions: vec![1.0f32; 40], // 10 nodes
            colors: vec![0.5f32; 40],
            importance: vec![0.8f32; 15], // Wrong count
            frame: 1,
        };
        assert!(mismatched_data.validate().is_err());
    }
}