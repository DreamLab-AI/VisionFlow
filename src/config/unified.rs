// Unified Settings Model - Single source of truth for all settings
// This replaces the complex multi-layer system with a clean, maintainable structure

use serde::{Deserialize, Serialize};
use config::{ConfigBuilder, ConfigError, Environment};
use log::debug;

// ============================================================================
// CORE PHYSICS SETTINGS - Critical for GPU simulation
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PhysicsSettings {
    pub enabled: bool,
    pub iterations: u32,
    pub damping: f32,
    pub spring_strength: f32,
    pub repulsion_strength: f32,
    pub repulsion_distance: f32,
    pub attraction_strength: f32,
    pub max_velocity: f32,
    pub collision_radius: f32,
    pub bounds_size: f32,
    pub enable_bounds: bool,
    pub mass_scale: f32,
    pub boundary_damping: f32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            iterations: 100,
            damping: 0.95,
            spring_strength: 0.3,
            repulsion_strength: 100.0,
            repulsion_distance: 150.0,
            attraction_strength: 0.001,
            max_velocity: 10.0,
            collision_radius: 50.0,
            bounds_size: 2000.0,
            enable_bounds: false,
            mass_scale: 1.0,
            boundary_damping: 0.5,
        }
    }
}

// Direct conversion to GPU simulation parameters
impl From<&PhysicsSettings> for crate::models::simulation_params::SimulationParams {
    fn from(physics: &PhysicsSettings) -> Self {
        Self {
            iterations: physics.iterations,
            spring_strength: physics.spring_strength,
            repulsion: physics.repulsion_strength,  // Note: field name difference
            damping: physics.damping,
            max_repulsion_distance: physics.repulsion_distance,
            viewport_bounds: physics.bounds_size,  // Note: field name difference
            mass_scale: physics.mass_scale,
            boundary_damping: physics.boundary_damping,
            enable_bounds: physics.enable_bounds,
            time_step: 0.016,  // 60 FPS
            phase: crate::models::simulation_params::SimulationPhase::Dynamic,
            mode: crate::models::simulation_params::SimulationMode::Remote,
        }
    }
}

// ============================================================================
// VISUAL SETTINGS - Node, Edge, Label appearance
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeSettings {
    pub base_color: String,
    pub metalness: f32,
    pub opacity: f32,
    pub roughness: f32,
    pub node_size: f32,
    pub quality: String,
    pub enable_instancing: bool,
    pub enable_hologram: bool,
    pub enable_metadata_shape: bool,
    pub enable_metadata_visualisation: bool,
}

impl Default for NodeSettings {
    fn default() -> Self {
        Self {
            base_color: "#00ff88".to_string(),
            metalness: 0.5,
            opacity: 1.0,
            roughness: 0.5,
            node_size: 1.0,
            quality: "high".to_string(),
            enable_instancing: true,
            enable_hologram: false,
            enable_metadata_shape: true,
            enable_metadata_visualisation: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EdgeSettings {
    pub arrow_size: f32,
    pub base_width: f32,
    pub color: String,
    pub enable_arrows: bool,
    pub opacity: f32,
    pub width_range: Vec<f32>,
    pub quality: String,
}

impl Default for EdgeSettings {
    fn default() -> Self {
        Self {
            arrow_size: 0.5,
            base_width: 0.1,
            color: "#ffffff".to_string(),
            enable_arrows: true,
            opacity: 0.5,
            width_range: vec![0.1, 1.0],
            quality: "high".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LabelSettings {
    pub desktop_font_size: f32,
    pub enable_labels: bool,
    pub text_color: String,
    pub text_outline_color: String,
    pub text_outline_width: f32,
    pub text_resolution: u32,
    pub text_padding: f32,
    pub billboard_mode: String,
}

impl Default for LabelSettings {
    fn default() -> Self {
        Self {
            desktop_font_size: 14.0,
            enable_labels: true,
            text_color: "#ffffff".to_string(),
            text_outline_color: "#000000".to_string(),
            text_outline_width: 2.0,
            text_resolution: 256,
            text_padding: 4.0,
            billboard_mode: "screen".to_string(),
        }
    }
}

// ============================================================================
// GRAPH SETTINGS - Individual graph configuration (logseq, visionflow)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GraphConfig {
    pub nodes: NodeSettings,
    pub edges: EdgeSettings,
    pub labels: LabelSettings,
    pub physics: PhysicsSettings,
}

// ============================================================================
// RENDERING & EFFECTS
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub background_color: String,
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub environment_intensity: f32,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            ambient_light_intensity: 0.4,
            background_color: "#000000".to_string(),
            directional_light_intensity: 1.0,
            enable_ambient_occlusion: false,
            enable_antialiasing: true,
            enable_shadows: false,
            environment_intensity: 1.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnimationSettings {
    pub enable_motion_blur: bool,
    pub enable_node_animations: bool,
    pub motion_blur_strength: f32,
    pub selection_wave_enabled: bool,
    pub pulse_enabled: bool,
    pub pulse_speed: f32,
    pub pulse_strength: f32,
    pub wave_speed: f32,
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            enable_motion_blur: false,
            enable_node_animations: true,
            motion_blur_strength: 0.5,
            selection_wave_enabled: true,
            pulse_enabled: true,
            pulse_speed: 1.0,
            pulse_strength: 0.1,
            wave_speed: 1.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BloomSettings {
    pub enabled: bool,
    pub strength: f32,
    pub radius: f32,
    pub node_bloom_strength: f32,
    pub edge_bloom_strength: f32,
    pub environment_bloom_strength: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 1.0,
            radius: 0.5,
            node_bloom_strength: 1.0,
            edge_bloom_strength: 1.0,
            environment_bloom_strength: 1.0,
        }
    }
}

// ============================================================================
// SYSTEM SETTINGS
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebSocketSettings {
    pub update_rate: u32,
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub binary_chunk_size: usize,
    pub heartbeat_interval: u64,
    pub max_message_size: usize,
}

impl Default for WebSocketSettings {
    fn default() -> Self {
        Self {
            update_rate: 30,
            compression_enabled: false,
            compression_threshold: 512,
            reconnect_attempts: 5,
            reconnect_delay: 1000,
            binary_chunk_size: 2048,
            heartbeat_interval: 10000,
            max_message_size: 10485760,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DebugSettings {
    pub enabled: bool,
    pub enable_physics_debug: bool,
    pub enable_websocket_debug: bool,
    pub log_level: String,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            enable_physics_debug: false,
            enable_websocket_debug: false,
            log_level: "info".to_string(),
        }
    }
}

// ============================================================================
// MAIN SETTINGS STRUCTURE - Single source of truth
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct UnifiedSettings {
    // Multi-graph configuration
    pub graphs: Graphs,
    
    // Global visual settings
    pub rendering: RenderingSettings,
    pub animations: AnimationSettings,
    pub bloom: BloomSettings,
    
    // System configuration
    pub websocket: WebSocketSettings,
    pub debug: DebugSettings,
    
    // Network settings (server only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkSettings>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Graphs {
    pub logseq: GraphConfig,
    pub visionflow: GraphConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkSettings {
    pub bind_address: String,
    pub port: u16,
    pub domain: String,
}

impl UnifiedSettings {
    /// Load settings from YAML file
    pub fn load() -> Result<Self, ConfigError> {
        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .unwrap_or_else(|_| "/app/settings.yaml".to_string());
        
        debug!("Loading settings from: {}", settings_path);
        
        ConfigBuilder::<config::builder::DefaultState>::default()
            .add_source(config::File::with_name(&settings_path))
            .add_source(Environment::default().separator("__"))
            .build()?
            .try_deserialize()
    }
    
    /// Save settings to YAML file
    pub fn save(&self) -> Result<(), String> {
        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .unwrap_or_else(|_| "/app/settings.yaml".to_string());
        
        let yaml = serde_yaml::to_string(self)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;
        
        std::fs::write(&settings_path, yaml)
            .map_err(|e| format!("Failed to write settings file: {}", e))?;
        
        debug!("Settings saved to: {}", settings_path);
        Ok(())
    }
    
    /// Get physics settings for a specific graph
    pub fn get_physics(&self, graph: &str) -> &PhysicsSettings {
        match graph {
            "logseq" => &self.graphs.logseq.physics,
            "visionflow" => &self.graphs.visionflow.physics,
            _ => &self.graphs.logseq.physics,  // Default to logseq
        }
    }
    
    /// Update physics settings for a specific graph
    pub fn update_physics(&mut self, graph: &str, update: PhysicsUpdate) {
        let physics = match graph {
            "logseq" => &mut self.graphs.logseq.physics,
            "visionflow" => &mut self.graphs.visionflow.physics,
            _ => &mut self.graphs.logseq.physics,
        };
        
        // Direct field updates, no macros needed
        if let Some(v) = update.damping { physics.damping = v; }
        if let Some(v) = update.spring_strength { physics.spring_strength = v; }
        if let Some(v) = update.repulsion_strength { physics.repulsion_strength = v; }
        if let Some(v) = update.iterations { physics.iterations = v; }
        if let Some(v) = update.enabled { physics.enabled = v; }
        if let Some(v) = update.bounds_size { physics.bounds_size = v; }
        if let Some(v) = update.enable_bounds { physics.enable_bounds = v; }
        if let Some(v) = update.max_velocity { physics.max_velocity = v; }
        if let Some(v) = update.collision_radius { physics.collision_radius = v; }
        if let Some(v) = update.attraction_strength { physics.attraction_strength = v; }
        if let Some(v) = update.repulsion_distance { physics.repulsion_distance = v; }
        if let Some(v) = update.mass_scale { physics.mass_scale = v; }
        if let Some(v) = update.boundary_damping { physics.boundary_damping = v; }
    }
    
    /// Merge partial settings update
    pub fn merge(&mut self, update: SettingsUpdate) {
        // Graph-specific updates
        if let Some(graphs) = update.graphs {
            if let Some(logseq) = graphs.logseq {
                if let Some(physics) = logseq.physics {
                    self.update_physics("logseq", physics);
                }
                // Merge other logseq settings...
            }
            if let Some(visionflow) = graphs.visionflow {
                if let Some(physics) = visionflow.physics {
                    self.update_physics("visionflow", physics);
                }
                // Merge other visionflow settings...
            }
        }
        
        // Global settings updates
        if let Some(_rendering) = update.rendering {
            // TODO: Merge rendering settings when needed
        }
        
        // System settings updates
        if let Some(_websocket) = update.websocket {
            // TODO: Merge websocket settings when needed
        }
    }
}

// ============================================================================
// UPDATE STRUCTURES - For partial updates from clients
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SettingsUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graphs: Option<GraphsUpdate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendering: Option<RenderingUpdate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket: Option<WebSocketUpdate>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GraphsUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logseq: Option<GraphUpdate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visionflow: Option<GraphUpdate>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GraphUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physics: Option<PhysicsUpdate>,
    // Add other graph setting updates as needed
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PhysicsUpdate {
    pub damping: Option<f32>,
    pub spring_strength: Option<f32>,
    pub repulsion_strength: Option<f32>,
    pub iterations: Option<u32>,
    pub enabled: Option<bool>,
    pub bounds_size: Option<f32>,
    pub enable_bounds: Option<bool>,
    pub max_velocity: Option<f32>,
    pub collision_radius: Option<f32>,
    pub attraction_strength: Option<f32>,
    pub repulsion_distance: Option<f32>,
    pub mass_scale: Option<f32>,
    pub boundary_damping: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RenderingUpdate {
    // Add rendering update fields as needed
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WebSocketUpdate {
    // Add websocket update fields as needed
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_physics_to_simulation_params() {
        let physics = PhysicsSettings::default();
        let sim_params: crate::models::simulation_params::SimulationParams = (&physics).into();
        
        assert_eq!(sim_params.damping, physics.damping);
        assert_eq!(sim_params.spring_strength, physics.spring_strength);
        assert_eq!(sim_params.repulsion, physics.repulsion_strength);
        assert_eq!(sim_params.viewport_bounds, physics.bounds_size);
    }
    
    #[test]
    fn test_physics_update() {
        let mut settings = UnifiedSettings::default();
        let update = PhysicsUpdate {
            damping: Some(0.8),
            spring_strength: Some(0.5),
            ..Default::default()
        };
        
        settings.update_physics("logseq", update);
        
        assert_eq!(settings.graphs.logseq.physics.damping, 0.8);
        assert_eq!(settings.graphs.logseq.physics.spring_strength, 0.5);
    }
}