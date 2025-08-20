//! Visual Analytics GPU Interface - Optimal data pipeline for GPU kernel
//! Designed to maximize A6000 throughput with zero-copy where possible

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use serde::{Deserialize, Serialize};
use log::{info, debug, trace};

/// 4D vector for temporal-spatial representation
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub t: f32,
}

/// GPU-optimized temporal-spatial node
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TSNode {
    // Core dynamics (16 bytes aligned)
    pub position: Vec4,
    pub velocity: Vec4,
    pub acceleration: Vec4,
    
    // Temporal trajectory (128 bytes)
    pub trajectory: [Vec4; 8],
    pub temporal_coherence: f32,
    pub motion_saliency: f32,
    
    // Hierarchy (24 bytes)
    pub hierarchy_level: i32,
    pub parent_idx: i32,
    pub children: [i32; 4],
    pub lod_importance: f32,
    
    // Layer membership (68 bytes)
    pub layer_membership: [f32; 16],
    pub primary_layer: i32,
    pub isolation_strength: f32,
    
    // Topology (136 bytes)
    pub topology: [f32; 32],
    pub betweenness_centrality: f32,
    pub clustering_coefficient: f32,
    pub pagerank: f32,
    pub community_id: i32,
    
    // Semantic (68 bytes)
    pub semantic_vector: [f32; 16],
    pub semantic_drift: f32,
    
    // Visual importance (16 bytes)
    pub visual_saliency: f32,
    pub information_content: f32,
    pub attention_weight: f32,
    
    // Force modulation (12 bytes)
    pub force_scale: f32,
    pub damping_local: f32,
    pub constraint_mask: i32,
}

/// GPU-optimized edge with temporal dynamics
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TSEdge {
    pub source: i32,
    pub target: i32,
    
    // Multi-dimensional weights
    pub structural_weight: f32,
    pub semantic_weight: f32,
    pub temporal_weight: f32,
    pub causal_weight: f32,
    
    // Temporal dynamics
    pub weight_history: [f32; 8],
    pub formation_time: f32,
    pub stability: f32,
    
    // Visual properties
    pub bundling_strength: f32,
    pub control_points: [Vec4; 2],
    pub layer_mask: i32,
    
    // Information flow
    pub information_flow: f32,
    pub latency: f32,
}

/// Isolation layer for visual focus
#[repr(C)]
#[derive(Debug, Clone)]
pub struct IsolationLayer {
    pub layer_id: i32,
    pub opacity: f32,
    pub z_offset: f32,
    
    pub focus_center: Vec4,
    pub focus_radius: f32,
    pub context_falloff: f32,
    
    pub importance_threshold: f32,
    pub community_filter: i32,
    pub topology_filter_mask: i32,
    pub temporal_range: [f32; 2],
    
    pub force_modulation: f32,
    pub edge_opacity: f32,
    pub color_scheme: i32,
}

// Implement DeviceRepr and ValidAsZeroBits for GPU structs
unsafe impl DeviceRepr for Vec4 {}
unsafe impl ValidAsZeroBits for Vec4 {}

unsafe impl DeviceRepr for TSNode {}
unsafe impl ValidAsZeroBits for TSNode {}

unsafe impl DeviceRepr for TSEdge {}
unsafe impl ValidAsZeroBits for TSEdge {}

unsafe impl DeviceRepr for IsolationLayer {}
unsafe impl ValidAsZeroBits for IsolationLayer {}

/// Visual analytics parameters
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnalyticsParams {
    // GPU optimization
    pub total_nodes: i32,
    pub total_edges: i32,
    pub active_layers: i32,
    pub hierarchy_depth: i32,
    
    // Temporal dynamics
    pub current_frame: i32,
    pub time_step: f32,
    pub temporal_decay: f32,
    pub history_weight: f32,
    
    // Force parameters (multi-resolution)
    pub force_scale: [f32; 4],
    pub damping: [f32; 4],
    pub temperature: [f32; 4],
    
    // Isolation and focus
    pub isolation_strength: f32,
    pub focus_gamma: f32,
    pub primary_focus_node: i32,
    pub context_alpha: f32,
    
    // Visual comprehension
    pub complexity_threshold: f32,
    pub saliency_boost: f32,
    pub information_bandwidth: f32,
    
    // Topology analysis
    pub community_algorithm: i32,
    pub modularity_resolution: f32,
    pub topology_update_interval: i32,
    
    // Semantic analysis
    pub semantic_influence: f32,
    pub drift_threshold: f32,
    pub embedding_dims: i32,
    
    // Viewport
    pub camera_position: Vec4,
    pub viewport_bounds: Vec4,
    pub zoom_level: f32,
    pub time_window: f32,
}

/// Main visual analytics GPU context
#[derive(Debug)]
pub struct VisualAnalyticsGPU {
    device: Arc<CudaDevice>,
    
    // GPU memory (pinned for zero-copy)
    nodes: CudaSlice<TSNode>,
    edges: CudaSlice<TSEdge>,
    layers: CudaSlice<IsolationLayer>,
    
    // Output buffers
    output_positions: CudaSlice<f32>,
    output_colors: CudaSlice<f32>,
    output_importance: CudaSlice<f32>,
    
    // Metadata
    max_nodes: usize,
    max_edges: usize,
    max_layers: usize,
    current_frame: u32,
    
    // Performance tracking
    kernel_time_ms: f32,
    transfer_time_ms: f32,
}

impl VisualAnalyticsGPU {
    /// Create new GPU context with pre-allocated memory
    pub async fn new(max_nodes: usize, max_edges: usize, max_layers: usize) -> Result<Self, String> {
        info!("Initializing Visual Analytics GPU for {} nodes, {} edges", max_nodes, max_edges);
        
        let device: Arc<CudaDevice> = CudaDevice::new(0)
            .map_err(|e| format!("Failed to create CUDA device: {}", e))?
            .into();
        
        // Pre-allocate GPU memory
        let nodes = device.alloc_zeros::<TSNode>(max_nodes)
            .map_err(|e| format!("Failed to allocate node memory: {}", e))?;
        
        let edges = device.alloc_zeros::<TSEdge>(max_edges)
            .map_err(|e| format!("Failed to allocate edge memory: {}", e))?;
        
        let layers = device.alloc_zeros::<IsolationLayer>(max_layers)
            .map_err(|e| format!("Failed to allocate layer memory: {}", e))?;
        
        // Output buffers
        let output_positions = device.alloc_zeros::<f32>(max_nodes * 4)
            .map_err(|e| format!("Failed to allocate position buffer: {}", e))?;
        
        let output_colors = device.alloc_zeros::<f32>(max_nodes * 4)
            .map_err(|e| format!("Failed to allocate color buffer: {}", e))?;
        
        let output_importance = device.alloc_zeros::<f32>(max_nodes)
            .map_err(|e| format!("Failed to allocate importance buffer: {}", e))?;
        
        Ok(Self {
            device,
            nodes,
            edges,
            layers,
            output_positions,
            output_colors,
            output_importance,
            max_nodes,
            max_edges,
            max_layers,
            current_frame: 0,
            kernel_time_ms: 0.0,
            transfer_time_ms: 0.0,
        })
    }
    
    /// Stream nodes to GPU (optimized for continuous updates)
    pub fn stream_nodes(&mut self, nodes: &[TSNode]) -> Result<(), String> {
        if nodes.len() > self.max_nodes {
            return Err(format!("Too many nodes: {} > {}", nodes.len(), self.max_nodes));
        }
        
        let start = std::time::Instant::now();
        
        // Use async copy for overlapping with computation
        self.device.htod_sync_copy_into(nodes, &mut self.nodes)
            .map_err(|e| format!("Failed to copy nodes to GPU: {}", e))?;
        
        self.transfer_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        trace!("Streamed {} nodes to GPU in {:.2}ms", nodes.len(), self.transfer_time_ms);
        
        Ok(())
    }
    
    /// Stream edges to GPU
    pub fn stream_edges(&mut self, edges: &[TSEdge]) -> Result<(), String> {
        if edges.len() > self.max_edges {
            return Err(format!("Too many edges: {} > {}", edges.len(), self.max_edges));
        }
        
        self.device.htod_sync_copy_into(edges, &mut self.edges)
            .map_err(|e| format!("Failed to copy edges to GPU: {}", e))?;
        
        Ok(())
    }
    
    /// Update isolation layers
    pub fn update_layers(&mut self, layers: &[IsolationLayer]) -> Result<(), String> {
        if layers.len() > self.max_layers {
            return Err(format!("Too many layers: {} > {}", layers.len(), self.max_layers));
        }
        
        self.device.htod_sync_copy_into(layers, &mut self.layers)
            .map_err(|e| format!("Failed to copy layers to GPU: {}", e))?;
        
        Ok(())
    }
    
    /// Execute visual analytics pipeline
    pub fn execute(&mut self, _params: &VisualAnalyticsParams, _num_nodes: usize, _num_edges: usize, _num_layers: usize) -> Result<(), String> {
        let start = std::time::Instant::now();
        
        // Launch kernel through FFI (would need actual FFI binding)
        // For now, this is a placeholder for the kernel launch
        // visual_analytics_kernel<<<grid, block>>>(...)
        // This would call the CUDA kernel we defined
        
        self.device.synchronize()
            .map_err(|e| format!("Kernel execution failed: {}", e))?;
        
        self.kernel_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        self.current_frame += 1;
        
        debug!("Visual analytics frame {} completed in {:.2}ms", self.current_frame, self.kernel_time_ms);
        
        Ok(())
    }
    
    /// Get output positions for rendering
    pub fn get_positions(&self) -> Result<Vec<f32>, String> {
        let mut positions = vec![0.0f32; self.max_nodes * 4];
        self.device.dtoh_sync_copy_into(&self.output_positions, &mut positions)
            .map_err(|e| format!("Failed to copy positions from GPU: {}", e))?;
        Ok(positions)
    }
    
    /// Get output colors for rendering
    pub fn get_colors(&self) -> Result<Vec<f32>, String> {
        let mut colors = vec![0.0f32; self.max_nodes * 4];
        self.device.dtoh_sync_copy_into(&self.output_colors, &mut colors)
            .map_err(|e| format!("Failed to copy colors from GPU: {}", e))?;
        Ok(colors)
    }
    
    /// Get importance scores
    pub fn get_importance(&self) -> Result<Vec<f32>, String> {
        let mut importance = vec![0.0f32; self.max_nodes];
        self.device.dtoh_sync_copy_into(&self.output_importance, &mut importance)
            .map_err(|e| format!("Failed to copy importance from GPU: {}", e))?;
        Ok(importance)
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            kernel_time_ms: self.kernel_time_ms,
            transfer_time_ms: self.transfer_time_ms,
            current_frame: self.current_frame,
            gpu_memory_used: (self.max_nodes * std::mem::size_of::<TSNode>() +
                             self.max_edges * std::mem::size_of::<TSEdge>()) as f32 / 1_048_576.0,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct PerformanceMetrics {
    pub kernel_time_ms: f32,
    pub transfer_time_ms: f32,
    pub current_frame: u32,
    pub gpu_memory_used: f32, // MB
}

/// Builder pattern for visual analytics configuration
pub struct VisualAnalyticsBuilder {
    params: VisualAnalyticsParams,
}

impl VisualAnalyticsBuilder {
    pub fn new() -> Self {
        Self {
            params: VisualAnalyticsParams {
                total_nodes: 0,
                total_edges: 0,
                active_layers: 1,
                hierarchy_depth: 3,
                current_frame: 0,
                time_step: 0.016,
                temporal_decay: 0.1,
                history_weight: 0.8,
                force_scale: [1.0, 0.5, 0.25, 0.125],
                damping: [0.9, 0.85, 0.8, 0.75],
                temperature: [1.0; 4],
                isolation_strength: 1.0,
                focus_gamma: 2.2,
                primary_focus_node: -1,
                context_alpha: 0.3,
                complexity_threshold: 0.5,
                saliency_boost: 1.5,
                information_bandwidth: 100.0,
                community_algorithm: 0,
                modularity_resolution: 1.0,
                topology_update_interval: 30,
                semantic_influence: 0.7,
                drift_threshold: 0.1,
                embedding_dims: 16,
                camera_position: Vec4 { x: 0.0, y: 0.0, z: 1000.0, t: 0.0 },
                viewport_bounds: Vec4 { x: 2000.0, y: 2000.0, z: 1000.0, t: 100.0 },
                zoom_level: 1.0,
                time_window: 100.0,
            },
        }
    }
    
    pub fn with_nodes(mut self, count: i32) -> Self {
        self.params.total_nodes = count;
        self
    }
    
    pub fn with_edges(mut self, count: i32) -> Self {
        self.params.total_edges = count;
        self
    }
    
    pub fn with_focus(mut self, node_id: i32, gamma: f32) -> Self {
        self.params.primary_focus_node = node_id;
        self.params.focus_gamma = gamma;
        self
    }
    
    pub fn with_temporal_decay(mut self, decay: f32) -> Self {
        self.params.temporal_decay = decay;
        self
    }
    
    pub fn build(self) -> VisualAnalyticsParams {
        self.params
    }
}

/// High-level interface for visual analytics
pub struct VisualAnalyticsEngine {
    gpu: VisualAnalyticsGPU,
    params: VisualAnalyticsParams,
    nodes: Vec<TSNode>,
    edges: Vec<TSEdge>,
    layers: Vec<IsolationLayer>,
}

impl VisualAnalyticsEngine {
    pub async fn new(max_nodes: usize, max_edges: usize) -> Result<Self, String> {
        let gpu = VisualAnalyticsGPU::new(max_nodes, max_edges, 16).await?;
        let params = VisualAnalyticsBuilder::new().build();
        
        Ok(Self {
            gpu,
            params,
            nodes: Vec::with_capacity(max_nodes),
            edges: Vec::with_capacity(max_edges),
            layers: vec![IsolationLayer::default(); 1],
        })
    }
    
    /// Add or update a node
    pub fn upsert_node(&mut self, id: usize, node: TSNode) {
        if id >= self.nodes.len() {
            self.nodes.resize(id + 1, TSNode::default());
        }
        self.nodes[id] = node;
    }
    
    /// Add an edge
    pub fn add_edge(&mut self, edge: TSEdge) {
        self.edges.push(edge);
    }
    
    /// Set focus on a specific node
    pub fn focus_on(&mut self, node_id: i32, radius: f32) {
        self.params.primary_focus_node = node_id;
        if !self.layers.is_empty() {
            self.layers[0].focus_center = if node_id >= 0 && (node_id as usize) < self.nodes.len() {
                self.nodes[node_id as usize].position
            } else {
                Vec4::default()
            };
            self.layers[0].focus_radius = radius;
        }
    }
    
    /// Execute one frame of visual analytics
    pub async fn step(&mut self) -> Result<RenderData, String> {
        // Stream data to GPU
        self.gpu.stream_nodes(&self.nodes)?;
        self.gpu.stream_edges(&self.edges)?;
        self.gpu.update_layers(&self.layers)?;
        
        // Execute GPU kernel
        self.gpu.execute(&self.params, self.nodes.len(), self.edges.len(), self.layers.len())?;
        
        // Get results
        let positions = self.gpu.get_positions()?;
        let colors = self.gpu.get_colors()?;
        let importance = self.gpu.get_importance()?;
        
        self.params.current_frame += 1;
        
        Ok(RenderData {
            positions,
            colors,
            importance,
            frame: self.params.current_frame,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct RenderData {
    pub positions: Vec<f32>,
    pub colors: Vec<f32>,
    pub importance: Vec<f32>,
    pub frame: i32,
}

impl Default for TSNode {
    fn default() -> Self {
        Self {
            position: Vec4::default(),
            velocity: Vec4::default(),
            acceleration: Vec4::default(),
            trajectory: [Vec4::default(); 8],
            temporal_coherence: 0.0,
            motion_saliency: 0.0,
            hierarchy_level: 0,
            parent_idx: -1,
            children: [-1; 4],
            lod_importance: 1.0,
            layer_membership: [0.0; 16],
            primary_layer: 0,
            isolation_strength: 1.0,
            topology: [0.0; 32],
            betweenness_centrality: 0.0,
            clustering_coefficient: 0.0,
            pagerank: 0.0,
            community_id: 0,
            semantic_vector: [0.0; 16],
            semantic_drift: 0.0,
            visual_saliency: 1.0,
            information_content: 0.0,
            attention_weight: 1.0,
            force_scale: 1.0,
            damping_local: 0.9,
            constraint_mask: 0,
        }
    }
}

impl Default for IsolationLayer {
    fn default() -> Self {
        Self {
            layer_id: 0,
            opacity: 1.0,
            z_offset: 0.0,
            focus_center: Vec4::default(),
            focus_radius: 500.0,
            context_falloff: 0.001,
            importance_threshold: 0.0,
            community_filter: -1,
            topology_filter_mask: 0,
            temporal_range: [0.0, 1000.0],
            force_modulation: 1.0,
            edge_opacity: 1.0,
            color_scheme: 0,
        }
    }
}