// Settings Model - Single source of truth for all settings
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
#[serde(rename_all = "camelCase")]
pub struct Settings {
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

impl Settings {
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
// BACKWARD COMPATIBILITY CONVERSIONS
// ============================================================================

// Conversion from Settings to AppFullSettings for backward compatibility during transition
impl From<Settings> for crate::config::AppFullSettings {
    fn from(settings: Settings) -> Self {
        // Convert the new Settings structure back to AppFullSettings
        // This creates the proper structure mapping
        
        Self {
            visualisation: crate::config::VisualisationSettings {
                // Legacy flat fields (using logseq as default)
                nodes: crate::config::NodeSettings {
                    base_color: settings.graphs.logseq.nodes.base_color.clone(),
                    metalness: settings.graphs.logseq.nodes.metalness,
                    opacity: settings.graphs.logseq.nodes.opacity,
                    roughness: settings.graphs.logseq.nodes.roughness,
                    node_size: settings.graphs.logseq.nodes.node_size,
                    quality: settings.graphs.logseq.nodes.quality.clone(),
                    enable_instancing: settings.graphs.logseq.nodes.enable_instancing,
                    enable_hologram: settings.graphs.logseq.nodes.enable_hologram,
                    enable_metadata_shape: settings.graphs.logseq.nodes.enable_metadata_shape,
                    enable_metadata_visualisation: settings.graphs.logseq.nodes.enable_metadata_visualisation,
                },
                edges: crate::config::EdgeSettings {
                    arrow_size: settings.graphs.logseq.edges.arrow_size,
                    base_width: settings.graphs.logseq.edges.base_width,
                    color: settings.graphs.logseq.edges.color.clone(),
                    enable_arrows: settings.graphs.logseq.edges.enable_arrows,
                    opacity: settings.graphs.logseq.edges.opacity,
                    width_range: settings.graphs.logseq.edges.width_range.clone(),
                    quality: settings.graphs.logseq.edges.quality.clone(),
                },
                physics: crate::config::PhysicsSettings {
                    attraction_strength: settings.graphs.logseq.physics.attraction_strength,
                    bounds_size: settings.graphs.logseq.physics.bounds_size,
                    collision_radius: settings.graphs.logseq.physics.collision_radius,
                    damping: settings.graphs.logseq.physics.damping,
                    enable_bounds: settings.graphs.logseq.physics.enable_bounds,
                    enabled: settings.graphs.logseq.physics.enabled,
                    iterations: settings.graphs.logseq.physics.iterations,
                    max_velocity: settings.graphs.logseq.physics.max_velocity,
                    repulsion_strength: settings.graphs.logseq.physics.repulsion_strength,
                    spring_strength: settings.graphs.logseq.physics.spring_strength,
                    repulsion_distance: settings.graphs.logseq.physics.repulsion_distance,
                    mass_scale: settings.graphs.logseq.physics.mass_scale,
                    boundary_damping: settings.graphs.logseq.physics.boundary_damping,
                },
                labels: crate::config::LabelSettings {
                    desktop_font_size: settings.graphs.logseq.labels.desktop_font_size,
                    enable_labels: settings.graphs.logseq.labels.enable_labels,
                    text_color: settings.graphs.logseq.labels.text_color.clone(),
                    text_outline_color: settings.graphs.logseq.labels.text_outline_color.clone(),
                    text_outline_width: settings.graphs.logseq.labels.text_outline_width,
                    text_resolution: settings.graphs.logseq.labels.text_resolution,
                    text_padding: settings.graphs.logseq.labels.text_padding,
                    billboard_mode: settings.graphs.logseq.labels.billboard_mode.clone(),
                },
                rendering: crate::config::RenderingSettings {
                    ambient_light_intensity: settings.rendering.ambient_light_intensity,
                    background_color: settings.rendering.background_color.clone(),
                    directional_light_intensity: settings.rendering.directional_light_intensity,
                    enable_ambient_occlusion: settings.rendering.enable_ambient_occlusion,
                    enable_antialiasing: settings.rendering.enable_antialiasing,
                    enable_shadows: settings.rendering.enable_shadows,
                    environment_intensity: settings.rendering.environment_intensity,
                },
                animations: crate::config::AnimationSettings {
                    enable_motion_blur: settings.animations.enable_motion_blur,
                    enable_node_animations: settings.animations.enable_node_animations,
                    motion_blur_strength: settings.animations.motion_blur_strength,
                    selection_wave_enabled: settings.animations.selection_wave_enabled,
                    pulse_enabled: settings.animations.pulse_enabled,
                    pulse_speed: settings.animations.pulse_speed,
                    pulse_strength: settings.animations.pulse_strength,
                    wave_speed: settings.animations.wave_speed,
                },
                bloom: crate::config::BloomSettings {
                    edge_bloom_strength: settings.bloom.edge_bloom_strength,
                    enabled: settings.bloom.enabled,
                    environment_bloom_strength: settings.bloom.environment_bloom_strength,
                    node_bloom_strength: settings.bloom.node_bloom_strength,
                    radius: settings.bloom.radius,
                    strength: settings.bloom.strength,
                },
                hologram: crate::config::HologramSettings::default(), // Use defaults for hologram
                graphs: crate::config::GraphsSettings {
                    logseq: crate::config::GraphSettings {
                        nodes: crate::config::NodeSettings {
                            base_color: settings.graphs.logseq.nodes.base_color.clone(),
                            metalness: settings.graphs.logseq.nodes.metalness,
                            opacity: settings.graphs.logseq.nodes.opacity,
                            roughness: settings.graphs.logseq.nodes.roughness,
                            node_size: settings.graphs.logseq.nodes.node_size,
                            quality: settings.graphs.logseq.nodes.quality.clone(),
                            enable_instancing: settings.graphs.logseq.nodes.enable_instancing,
                            enable_hologram: settings.graphs.logseq.nodes.enable_hologram,
                            enable_metadata_shape: settings.graphs.logseq.nodes.enable_metadata_shape,
                            enable_metadata_visualisation: settings.graphs.logseq.nodes.enable_metadata_visualisation,
                        },
                        edges: crate::config::EdgeSettings {
                            arrow_size: settings.graphs.logseq.edges.arrow_size,
                            base_width: settings.graphs.logseq.edges.base_width,
                            color: settings.graphs.logseq.edges.color.clone(),
                            enable_arrows: settings.graphs.logseq.edges.enable_arrows,
                            opacity: settings.graphs.logseq.edges.opacity,
                            width_range: settings.graphs.logseq.edges.width_range.clone(),
                            quality: settings.graphs.logseq.edges.quality.clone(),
                        },
                        labels: crate::config::LabelSettings {
                            desktop_font_size: settings.graphs.logseq.labels.desktop_font_size,
                            enable_labels: settings.graphs.logseq.labels.enable_labels,
                            text_color: settings.graphs.logseq.labels.text_color.clone(),
                            text_outline_color: settings.graphs.logseq.labels.text_outline_color.clone(),
                            text_outline_width: settings.graphs.logseq.labels.text_outline_width,
                            text_resolution: settings.graphs.logseq.labels.text_resolution,
                            text_padding: settings.graphs.logseq.labels.text_padding,
                            billboard_mode: settings.graphs.logseq.labels.billboard_mode.clone(),
                        },
                        physics: crate::config::PhysicsSettings {
                            attraction_strength: settings.graphs.logseq.physics.attraction_strength,
                            bounds_size: settings.graphs.logseq.physics.bounds_size,
                            collision_radius: settings.graphs.logseq.physics.collision_radius,
                            damping: settings.graphs.logseq.physics.damping,
                            enable_bounds: settings.graphs.logseq.physics.enable_bounds,
                            enabled: settings.graphs.logseq.physics.enabled,
                            iterations: settings.graphs.logseq.physics.iterations,
                            max_velocity: settings.graphs.logseq.physics.max_velocity,
                            repulsion_strength: settings.graphs.logseq.physics.repulsion_strength,
                            spring_strength: settings.graphs.logseq.physics.spring_strength,
                            repulsion_distance: settings.graphs.logseq.physics.repulsion_distance,
                            mass_scale: settings.graphs.logseq.physics.mass_scale,
                            boundary_damping: settings.graphs.logseq.physics.boundary_damping,
                        },
                    },
                    visionflow: crate::config::GraphSettings {
                        nodes: crate::config::NodeSettings {
                            base_color: settings.graphs.visionflow.nodes.base_color.clone(),
                            metalness: settings.graphs.visionflow.nodes.metalness,
                            opacity: settings.graphs.visionflow.nodes.opacity,
                            roughness: settings.graphs.visionflow.nodes.roughness,
                            node_size: settings.graphs.visionflow.nodes.node_size,
                            quality: settings.graphs.visionflow.nodes.quality.clone(),
                            enable_instancing: settings.graphs.visionflow.nodes.enable_instancing,
                            enable_hologram: settings.graphs.visionflow.nodes.enable_hologram,
                            enable_metadata_shape: settings.graphs.visionflow.nodes.enable_metadata_shape,
                            enable_metadata_visualisation: settings.graphs.visionflow.nodes.enable_metadata_visualisation,
                        },
                        edges: crate::config::EdgeSettings {
                            arrow_size: settings.graphs.visionflow.edges.arrow_size,
                            base_width: settings.graphs.visionflow.edges.base_width,
                            color: settings.graphs.visionflow.edges.color.clone(),
                            enable_arrows: settings.graphs.visionflow.edges.enable_arrows,
                            opacity: settings.graphs.visionflow.edges.opacity,
                            width_range: settings.graphs.visionflow.edges.width_range.clone(),
                            quality: settings.graphs.visionflow.edges.quality.clone(),
                        },
                        labels: crate::config::LabelSettings {
                            desktop_font_size: settings.graphs.visionflow.labels.desktop_font_size,
                            enable_labels: settings.graphs.visionflow.labels.enable_labels,
                            text_color: settings.graphs.visionflow.labels.text_color.clone(),
                            text_outline_color: settings.graphs.visionflow.labels.text_outline_color.clone(),
                            text_outline_width: settings.graphs.visionflow.labels.text_outline_width,
                            text_resolution: settings.graphs.visionflow.labels.text_resolution,
                            text_padding: settings.graphs.visionflow.labels.text_padding,
                            billboard_mode: settings.graphs.visionflow.labels.billboard_mode.clone(),
                        },
                        physics: crate::config::PhysicsSettings {
                            attraction_strength: settings.graphs.visionflow.physics.attraction_strength,
                            bounds_size: settings.graphs.visionflow.physics.bounds_size,
                            collision_radius: settings.graphs.visionflow.physics.collision_radius,
                            damping: settings.graphs.visionflow.physics.damping,
                            enable_bounds: settings.graphs.visionflow.physics.enable_bounds,
                            enabled: settings.graphs.visionflow.physics.enabled,
                            iterations: settings.graphs.visionflow.physics.iterations,
                            max_velocity: settings.graphs.visionflow.physics.max_velocity,
                            repulsion_strength: settings.graphs.visionflow.physics.repulsion_strength,
                            spring_strength: settings.graphs.visionflow.physics.spring_strength,
                            repulsion_distance: settings.graphs.visionflow.physics.repulsion_distance,
                            mass_scale: settings.graphs.visionflow.physics.mass_scale,
                            boundary_damping: settings.graphs.visionflow.physics.boundary_damping,
                        },
                    },
                },
            },
            system: crate::config::ServerSystemConfigFromFile {
                network: crate::config::NetworkSettings::default(),
                websocket: crate::config::ServerFullWebSocketSettings {
                    binary_chunk_size: settings.websocket.binary_chunk_size,
                    binary_update_rate: 30,
                    min_update_rate: 5,
                    max_update_rate: 60,
                    motion_threshold: 0.05,
                    motion_damping: 0.9,
                    binary_message_version: 1,
                    compression_enabled: settings.websocket.compression_enabled,
                    compression_threshold: settings.websocket.compression_threshold,
                    heartbeat_interval: settings.websocket.heartbeat_interval,
                    heartbeat_timeout: 600000,
                    max_connections: 100,
                    max_message_size: settings.websocket.max_message_size,
                    reconnect_attempts: settings.websocket.reconnect_attempts,
                    reconnect_delay: settings.websocket.reconnect_delay,
                    update_rate: settings.websocket.update_rate,
                },
                security: crate::config::SecuritySettings::default(),
                debug: crate::config::DebugSettings {
                    enabled: settings.debug.enabled,
                    enable_data_debug: settings.debug.enable_physics_debug,
                    enable_websocket_debug: settings.debug.enable_websocket_debug,
                    log_binary_headers: false,
                    log_full_json: false,
                    log_level: settings.debug.log_level.clone(),
                    log_format: "json".to_string(),
                },
                persist_settings: true,
            },
            xr: crate::config::XRSettings::default(),
            auth: crate::config::AuthSettings::default(),
            ragflow: None,
            perplexity: None,
            openai: None,
            kokoro: None,
            whisper: None,
        }
    }
}

// Conversion from AppFullSettings to Settings (client-facing)
impl From<crate::config::AppFullSettings> for Settings {
    fn from(app_settings: crate::config::AppFullSettings) -> Self {
        // Convert AppFullSettings to the new Settings structure
        // Extract the visualisation settings and convert them appropriately
        
        Self {
            graphs: Graphs {
                logseq: GraphConfig {
                    nodes: NodeSettings {
                        base_color: app_settings.visualisation.graphs.logseq.nodes.base_color,
                        metalness: app_settings.visualisation.graphs.logseq.nodes.metalness,
                        opacity: app_settings.visualisation.graphs.logseq.nodes.opacity,
                        roughness: app_settings.visualisation.graphs.logseq.nodes.roughness,
                        node_size: app_settings.visualisation.graphs.logseq.nodes.node_size,
                        quality: app_settings.visualisation.graphs.logseq.nodes.quality,
                        enable_instancing: app_settings.visualisation.graphs.logseq.nodes.enable_instancing,
                        enable_hologram: app_settings.visualisation.graphs.logseq.nodes.enable_hologram,
                        enable_metadata_shape: app_settings.visualisation.graphs.logseq.nodes.enable_metadata_shape,
                        enable_metadata_visualisation: app_settings.visualisation.graphs.logseq.nodes.enable_metadata_visualisation,
                    },
                    edges: EdgeSettings {
                        arrow_size: app_settings.visualisation.graphs.logseq.edges.arrow_size,
                        base_width: app_settings.visualisation.graphs.logseq.edges.base_width,
                        color: app_settings.visualisation.graphs.logseq.edges.color,
                        enable_arrows: app_settings.visualisation.graphs.logseq.edges.enable_arrows,
                        opacity: app_settings.visualisation.graphs.logseq.edges.opacity,
                        width_range: app_settings.visualisation.graphs.logseq.edges.width_range,
                        quality: app_settings.visualisation.graphs.logseq.edges.quality,
                    },
                    labels: LabelSettings {
                        desktop_font_size: app_settings.visualisation.graphs.logseq.labels.desktop_font_size,
                        enable_labels: app_settings.visualisation.graphs.logseq.labels.enable_labels,
                        text_color: app_settings.visualisation.graphs.logseq.labels.text_color,
                        text_outline_color: app_settings.visualisation.graphs.logseq.labels.text_outline_color,
                        text_outline_width: app_settings.visualisation.graphs.logseq.labels.text_outline_width,
                        text_resolution: app_settings.visualisation.graphs.logseq.labels.text_resolution,
                        text_padding: app_settings.visualisation.graphs.logseq.labels.text_padding,
                        billboard_mode: app_settings.visualisation.graphs.logseq.labels.billboard_mode,
                    },
                    physics: PhysicsSettings {
                        enabled: app_settings.visualisation.graphs.logseq.physics.enabled,
                        iterations: app_settings.visualisation.graphs.logseq.physics.iterations,
                        damping: app_settings.visualisation.graphs.logseq.physics.damping,
                        spring_strength: app_settings.visualisation.graphs.logseq.physics.spring_strength,
                        repulsion_strength: app_settings.visualisation.graphs.logseq.physics.repulsion_strength,
                        repulsion_distance: app_settings.visualisation.graphs.logseq.physics.repulsion_distance,
                        attraction_strength: app_settings.visualisation.graphs.logseq.physics.attraction_strength,
                        max_velocity: app_settings.visualisation.graphs.logseq.physics.max_velocity,
                        collision_radius: app_settings.visualisation.graphs.logseq.physics.collision_radius,
                        bounds_size: app_settings.visualisation.graphs.logseq.physics.bounds_size,
                        enable_bounds: app_settings.visualisation.graphs.logseq.physics.enable_bounds,
                        mass_scale: app_settings.visualisation.graphs.logseq.physics.mass_scale,
                        boundary_damping: app_settings.visualisation.graphs.logseq.physics.boundary_damping,
                    },
                },
                visionflow: GraphConfig {
                    nodes: NodeSettings {
                        base_color: app_settings.visualisation.graphs.visionflow.nodes.base_color,
                        metalness: app_settings.visualisation.graphs.visionflow.nodes.metalness,
                        opacity: app_settings.visualisation.graphs.visionflow.nodes.opacity,
                        roughness: app_settings.visualisation.graphs.visionflow.nodes.roughness,
                        node_size: app_settings.visualisation.graphs.visionflow.nodes.node_size,
                        quality: app_settings.visualisation.graphs.visionflow.nodes.quality,
                        enable_instancing: app_settings.visualisation.graphs.visionflow.nodes.enable_instancing,
                        enable_hologram: app_settings.visualisation.graphs.visionflow.nodes.enable_hologram,
                        enable_metadata_shape: app_settings.visualisation.graphs.visionflow.nodes.enable_metadata_shape,
                        enable_metadata_visualisation: app_settings.visualisation.graphs.visionflow.nodes.enable_metadata_visualisation,
                    },
                    edges: EdgeSettings {
                        arrow_size: app_settings.visualisation.graphs.visionflow.edges.arrow_size,
                        base_width: app_settings.visualisation.graphs.visionflow.edges.base_width,
                        color: app_settings.visualisation.graphs.visionflow.edges.color,
                        enable_arrows: app_settings.visualisation.graphs.visionflow.edges.enable_arrows,
                        opacity: app_settings.visualisation.graphs.visionflow.edges.opacity,
                        width_range: app_settings.visualisation.graphs.visionflow.edges.width_range,
                        quality: app_settings.visualisation.graphs.visionflow.edges.quality,
                    },
                    labels: LabelSettings {
                        desktop_font_size: app_settings.visualisation.graphs.visionflow.labels.desktop_font_size,
                        enable_labels: app_settings.visualisation.graphs.visionflow.labels.enable_labels,
                        text_color: app_settings.visualisation.graphs.visionflow.labels.text_color,
                        text_outline_color: app_settings.visualisation.graphs.visionflow.labels.text_outline_color,
                        text_outline_width: app_settings.visualisation.graphs.visionflow.labels.text_outline_width,
                        text_resolution: app_settings.visualisation.graphs.visionflow.labels.text_resolution,
                        text_padding: app_settings.visualisation.graphs.visionflow.labels.text_padding,
                        billboard_mode: app_settings.visualisation.graphs.visionflow.labels.billboard_mode,
                    },
                    physics: PhysicsSettings {
                        enabled: app_settings.visualisation.graphs.visionflow.physics.enabled,
                        iterations: app_settings.visualisation.graphs.visionflow.physics.iterations,
                        damping: app_settings.visualisation.graphs.visionflow.physics.damping,
                        spring_strength: app_settings.visualisation.graphs.visionflow.physics.spring_strength,
                        repulsion_strength: app_settings.visualisation.graphs.visionflow.physics.repulsion_strength,
                        repulsion_distance: app_settings.visualisation.graphs.visionflow.physics.repulsion_distance,
                        attraction_strength: app_settings.visualisation.graphs.visionflow.physics.attraction_strength,
                        max_velocity: app_settings.visualisation.graphs.visionflow.physics.max_velocity,
                        collision_radius: app_settings.visualisation.graphs.visionflow.physics.collision_radius,
                        bounds_size: app_settings.visualisation.graphs.visionflow.physics.bounds_size,
                        enable_bounds: app_settings.visualisation.graphs.visionflow.physics.enable_bounds,
                        mass_scale: app_settings.visualisation.graphs.visionflow.physics.mass_scale,
                        boundary_damping: app_settings.visualisation.graphs.visionflow.physics.boundary_damping,
                    },
                },
            },
            rendering: RenderingSettings {
                ambient_light_intensity: app_settings.visualisation.rendering.ambient_light_intensity,
                background_color: app_settings.visualisation.rendering.background_color,
                directional_light_intensity: app_settings.visualisation.rendering.directional_light_intensity,
                enable_ambient_occlusion: app_settings.visualisation.rendering.enable_ambient_occlusion,
                enable_antialiasing: app_settings.visualisation.rendering.enable_antialiasing,
                enable_shadows: app_settings.visualisation.rendering.enable_shadows,
                environment_intensity: app_settings.visualisation.rendering.environment_intensity,
            },
            animations: AnimationSettings {
                enable_motion_blur: app_settings.visualisation.animations.enable_motion_blur,
                enable_node_animations: app_settings.visualisation.animations.enable_node_animations,
                motion_blur_strength: app_settings.visualisation.animations.motion_blur_strength,
                selection_wave_enabled: app_settings.visualisation.animations.selection_wave_enabled,
                pulse_enabled: app_settings.visualisation.animations.pulse_enabled,
                pulse_speed: app_settings.visualisation.animations.pulse_speed,
                pulse_strength: app_settings.visualisation.animations.pulse_strength,
                wave_speed: app_settings.visualisation.animations.wave_speed,
            },
            bloom: BloomSettings {
                enabled: app_settings.visualisation.bloom.enabled,
                strength: app_settings.visualisation.bloom.strength,
                radius: app_settings.visualisation.bloom.radius,
                node_bloom_strength: app_settings.visualisation.bloom.node_bloom_strength,
                edge_bloom_strength: app_settings.visualisation.bloom.edge_bloom_strength,
                environment_bloom_strength: app_settings.visualisation.bloom.environment_bloom_strength,
            },
            websocket: WebSocketSettings {
                update_rate: app_settings.system.websocket.update_rate,
                compression_enabled: app_settings.system.websocket.compression_enabled,
                compression_threshold: app_settings.system.websocket.compression_threshold,
                reconnect_attempts: app_settings.system.websocket.reconnect_attempts,
                reconnect_delay: app_settings.system.websocket.reconnect_delay,
                binary_chunk_size: app_settings.system.websocket.binary_chunk_size,
                heartbeat_interval: app_settings.system.websocket.heartbeat_interval,
                max_message_size: app_settings.system.websocket.max_message_size,
            },
            debug: DebugSettings {
                enabled: app_settings.system.debug.enabled,
                enable_physics_debug: app_settings.system.debug.enable_data_debug,
                enable_websocket_debug: app_settings.system.debug.enable_websocket_debug,
                log_level: app_settings.system.debug.log_level,
            },
            network: None, // Network settings are not part of client-facing Settings
        }
    }
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
        let mut settings = Settings::default();
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