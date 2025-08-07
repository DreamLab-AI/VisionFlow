// Minimal, clean settings structure aligned with actual usage
// This replaces the overcomplicated multi-layer system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CORE PHYSICS - What actually matters for GPU simulation
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhysicsSettings {
    pub enabled: bool,
    pub iterations: u32,           // GPU kernel uses this
    pub damping: f32,              // GPU kernel uses this
    pub spring_strength: f32,      // GPU kernel uses this
    pub repulsion_strength: f32,   // GPU kernel uses this (maps to repulsion)
    pub repulsion_distance: f32,   // GPU kernel uses this (maps to max_repulsion_distance)
    pub max_velocity: f32,         // Used for clamping
    pub bounds_size: f32,          // GPU kernel uses this (maps to viewport_bounds)
    pub enable_bounds: bool,       // Controls viewport_bounds application
    pub mass_scale: f32,           // GPU kernel uses this
    pub boundary_damping: f32,     // GPU kernel uses this
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            iterations: 200,
            damping: 0.85,
            spring_strength: 0.02,
            repulsion_strength: 15.0,
            repulsion_distance: 50.0,
            max_velocity: 0.5,
            bounds_size: 150.0,
            enable_bounds: true,
            mass_scale: 1.5,
            boundary_damping: 0.9,
        }
    }
}

// Direct conversion to GPU params - no complex mapping needed
impl PhysicsSettings {
    pub fn to_simulation_params(&self) -> crate::models::simulation_params::SimulationParams {
        crate::models::simulation_params::SimulationParams {
            iterations: self.iterations,
            time_step: 0.2, // Fixed for stability
            spring_strength: self.spring_strength,
            repulsion: self.repulsion_strength,
            max_repulsion_distance: self.repulsion_distance,
            mass_scale: self.mass_scale,
            damping: self.damping,
            boundary_damping: self.boundary_damping,
            viewport_bounds: self.bounds_size,
            enable_bounds: self.enable_bounds,
            phase: crate::models::simulation_params::SimulationPhase::Dynamic,
            mode: crate::models::simulation_params::SimulationMode::Remote,
        }
    }
}

// ============================================================================
// VISUAL SETTINGS - What users actually see and control
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSettings {
    pub base_color: String,
    pub size: f32,
    pub opacity: f32,
    pub metalness: f32,
    pub roughness: f32,
    pub enable_hologram: bool,
}

impl Default for NodeSettings {
    fn default() -> Self {
        Self {
            base_color: "#66d9ef".to_string(),
            size: 1.2,
            opacity: 0.95,
            metalness: 0.85,
            roughness: 0.15,
            enable_hologram: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSettings {
    pub color: String,
    pub width: f32,
    pub opacity: f32,
    pub enable_arrows: bool,
    pub arrow_size: f32,
}

impl Default for EdgeSettings {
    fn default() -> Self {
        Self {
            color: "#56b6c2".to_string(),
            width: 0.5,
            opacity: 0.25,
            enable_arrows: false,
            arrow_size: 0.02,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSettings {
    pub enabled: bool,
    pub font_size: f32,
    pub color: String,
    pub outline_color: String,
    pub outline_width: f32,
}

impl Default for LabelSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            font_size: 0.5,
            color: "#f8f8f2".to_string(),
            outline_color: "#181c28".to_string(),
            outline_width: 0.005,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderingSettings {
    pub background_color: String,
    pub ambient_light: f32,
    pub directional_light: f32,
    pub enable_bloom: bool,
    pub bloom_strength: f32,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            background_color: "#0a0e1a".to_string(),
            ambient_light: 1.2,
            directional_light: 1.5,
            enable_bloom: true,
            bloom_strength: 1.5,
        }
    }
}

// ============================================================================
// SYSTEM SETTINGS - Backend configuration
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSettings {
    pub port: u16,
    pub bind_address: String,
}

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            port: 3001,
            bind_address: "0.0.0.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketSettings {
    pub update_rate: u32,
    pub binary_chunk_size: usize,
    pub heartbeat_interval: u64,
}

impl Default for WebSocketSettings {
    fn default() -> Self {
        Self {
            update_rate: 30,
            binary_chunk_size: 2048,
            heartbeat_interval: 10000,
        }
    }
}

// ============================================================================
// GRAPH SETTINGS - Support for multiple graphs
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSettings {
    pub physics: PhysicsSettings,
    pub nodes: NodeSettings,
    pub edges: EdgeSettings,
    pub labels: LabelSettings,
}

impl Default for GraphSettings {
    fn default() -> Self {
        Self {
            physics: PhysicsSettings::default(),
            nodes: NodeSettings::default(),
            edges: EdgeSettings::default(),
            labels: LabelSettings::default(),
        }
    }
}

// ============================================================================
// ROOT SETTINGS - The complete minimal structure
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalSettings {
    // Active graph being displayed
    pub active_graph: String,
    
    // Graph-specific settings
    pub graphs: HashMap<String, GraphSettings>,
    
    // Global rendering settings
    pub rendering: RenderingSettings,
    
    // System configuration
    pub network: NetworkSettings,
    pub websocket: WebSocketSettings,
    
    // Debug mode
    pub debug: bool,
}

impl Default for MinimalSettings {
    fn default() -> Self {
        let mut graphs = HashMap::new();
        
        // Default logseq graph
        graphs.insert("logseq".to_string(), GraphSettings::default());
        
        // Visionflow graph with different defaults
        let mut visionflow = GraphSettings::default();
        visionflow.nodes.base_color = "#ff8800".to_string();
        visionflow.edges.color = "#ffaa00".to_string();
        visionflow.physics.spring_strength = 0.5;
        visionflow.physics.repulsion_strength = 150.0;
        graphs.insert("visionflow".to_string(), visionflow);
        
        Self {
            active_graph: "logseq".to_string(),
            graphs,
            rendering: RenderingSettings::default(),
            network: NetworkSettings::default(),
            websocket: WebSocketSettings::default(),
            debug: false,
        }
    }
}

impl MinimalSettings {
    // Load from YAML with minimal validation
    pub fn load() -> Result<Self, String> {
        let path = std::env::var("SETTINGS_FILE_PATH")
            .unwrap_or_else(|_| "/app/settings.yaml".to_string());
        
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read settings: {}", e))?;
        
        serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse settings: {}", e))
    }
    
    // Save to YAML
    pub fn save(&self) -> Result<(), String> {
        let path = std::env::var("SETTINGS_FILE_PATH")
            .unwrap_or_else(|_| "/app/settings.yaml".to_string());
        
        let yaml = serde_yaml::to_string(self)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;
        
        std::fs::write(&path, yaml)
            .map_err(|e| format!("Failed to write settings: {}", e))
    }
    
    // Get active graph settings
    pub fn active_graph_settings(&self) -> &GraphSettings {
        self.graphs.get(&self.active_graph)
            .unwrap_or_else(|| self.graphs.get("logseq").unwrap())
    }
    
    // Get mutable active graph settings
    pub fn active_graph_settings_mut(&mut self) -> &mut GraphSettings {
        let active = self.active_graph.clone();
        // First try to get the active graph, fallback to logseq
        if self.graphs.contains_key(&active) {
            self.graphs.get_mut(&active).unwrap()
        } else {
            self.graphs.get_mut("logseq").unwrap()
        }
    }
    
    // Update physics for active graph
    pub fn update_physics(&mut self, physics: PhysicsSettings) {
        self.active_graph_settings_mut().physics = physics;
    }
}

// ============================================================================
// CLIENT UPDATE PAYLOAD - What the UI sends
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SettingsUpdate {
    Physics(PhysicsSettings),
    Nodes(NodeSettings),
    Edges(EdgeSettings),
    Labels(LabelSettings),
    Rendering(RenderingSettings),
    Full(MinimalSettings),
}

impl SettingsUpdate {
    pub fn apply_to(&self, settings: &mut MinimalSettings) {
        match self {
            Self::Physics(p) => settings.active_graph_settings_mut().physics = p.clone(),
            Self::Nodes(n) => settings.active_graph_settings_mut().nodes = n.clone(),
            Self::Edges(e) => settings.active_graph_settings_mut().edges = e.clone(),
            Self::Labels(l) => settings.active_graph_settings_mut().labels = l.clone(),
            Self::Rendering(r) => settings.rendering = r.clone(),
            Self::Full(s) => *settings = s.clone(),
        }
    }
}