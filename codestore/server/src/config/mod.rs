use config::{ConfigBuilder, ConfigError, Environment};
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use specta::Type;
use std::path::PathBuf;
use validator::{Validate, ValidationError};
use std::collections::HashMap;
use regex::Regex;
use lazy_static::lazy_static;

pub mod dev_config;
pub mod path_access;
pub mod unified_access;
pub mod api_wrapper;

// Import the macro
use crate::impl_unified_path_accessible;

// Validation regex patterns
lazy_static! {
    static ref HEX_COLOR_REGEX: Regex = Regex::new(r"^#[0-9A-Fa-f]{6}$").unwrap();
    static ref URL_REGEX: Regex = Regex::new(r"^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?::[0-9]+)?(?:/.*)?$").unwrap();
    static ref FILE_PATH_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9._/-]+$").unwrap();
    static ref DOMAIN_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
}

// Custom validation functions
fn validate_hex_color(color: &str) -> Result<(), ValidationError> {
    if HEX_COLOR_REGEX.is_match(color) {
        Ok(())
    } else {
        let mut error = ValidationError::new("invalid_hex_color");
        error.message = Some("Must be a valid hex color (e.g., #ff0000)".into());
        Err(error)
    }
}

fn validate_width_range(range: &[f32]) -> Result<(), ValidationError> {
    if range.len() != 2 {
        let mut error = ValidationError::new("invalid_range_length");
        error.message = Some("widthRange must have exactly 2 values [min, max]".into());
        return Err(error);
    }
    if range[0] >= range[1] {
        let mut error = ValidationError::new("invalid_range_order");
        error.message = Some("Minimum widthRange value must be less than maximum value".into());
        return Err(error);
    }
    if range[0] < 0.0 || range[1] < 0.0 {
        let mut error = ValidationError::new("negative_width");
        error.message = Some("widthRange values must be positive".into());
        return Err(error);
    }
    Ok(())
}

fn validate_port(port: u16) -> Result<(), ValidationError> {
    if port == 0 {
        let mut error = ValidationError::new("invalid_port");
        error.message = Some("Port must be between 1 and 65535".into());
        return Err(error);
    }
    Ok(())
}

fn validate_percentage(value: f32) -> Result<(), ValidationError> {
    if !(0.0..=100.0).contains(&value) {
        let mut error = ValidationError::new("invalid_percentage");
        error.message = Some("Value must be between 0 and 100".into());
        return Err(error);
    }
    Ok(())
}

fn default_auto_balance_interval() -> u32 {
    500
}

fn default_glow_color() -> String {
    "#00ffff".to_string()
}

fn default_glow_opacity() -> f32 {
    0.8
}

pub mod feature_access;

// Types are already public in this module, no need to re-export


// Helper function to merge two JSON values
fn merge_json_values(base: Value, update: Value) -> Value {
    use serde_json::map::Entry;
    
    match (base, update) {
        (Value::Object(mut base_map), Value::Object(update_map)) => {
            for (key, update_value) in update_map {
                match base_map.entry(key) {
                    Entry::Occupied(mut entry) => {
                        let merged = merge_json_values(entry.get().clone(), update_value);
                        entry.insert(merged);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(update_value);
                    }
                }
            }
            Value::Object(base_map)
        }
        (_, update) => update, // For non-objects, update overwrites base
    }
}


#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct MovementAxes {
    #[validate(range(min = -100, max = 100, message = "Movement axis must be between -100 and 100"))]
    pub horizontal: i32,
    #[validate(range(min = -100, max = 100, message = "Movement axis must be between -100 and 100"))]
    pub vertical: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct NodeSettings {
    #[validate(custom(function = "validate_hex_color", message = "Must be a valid hex color (e.g., #ff0000)"))]
    pub base_color: String,
    #[validate(range(min = 0.0, max = 1.0, message = "Metalness must be between 0.0 and 1.0"))]
    pub metalness: f32,
    #[validate(range(min = 0.0, max = 1.0, message = "Opacity must be between 0.0 and 1.0"))]
    pub opacity: f32,
    #[validate(range(min = 0.0, max = 1.0, message = "Roughness must be between 0.0 and 1.0"))]
    pub roughness: f32,
    #[validate(range(min = 0.1, max = 100.0, message = "Node size must be between 0.1 and 100.0"))]
    pub node_size: f32,
    #[validate(length(min = 1, message = "Quality cannot be empty"))]
    pub quality: String,
    pub enable_instancing: bool,
    pub enable_hologram: bool,
    pub enable_metadata_shape: bool,
    pub enable_metadata_visualisation: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct EdgeSettings {
    #[validate(range(min = 0.1, max = 10.0, message = "Arrow size must be between 0.1 and 10.0"))]
    pub arrow_size: f32,
    #[validate(range(min = 0.1, max = 20.0, message = "Base width must be between 0.1 and 20.0"))]
    pub base_width: f32,
    #[validate(custom(function = "validate_hex_color", message = "Must be a valid hex color (e.g., #ff0000)"))]
    pub color: String,
    pub enable_arrows: bool,
    #[validate(range(min = 0.0, max = 1.0, message = "Opacity must be between 0.0 and 1.0"))]
    pub opacity: f32,
    #[validate(length(min = 2, max = 2, message = "Width range must have exactly 2 values [min, max]"))]
    #[validate(custom(function = "validate_width_range"))]
    pub width_range: Vec<f32>,
    #[validate(length(min = 1, message = "Quality cannot be empty"))]
    pub quality: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct AutoBalanceConfig {
    pub stability_variance_threshold: f32,
    pub stability_frame_count: u32,
    pub clustering_distance_threshold: f32,
    pub bouncing_node_percentage: f32,
    pub boundary_min_distance: f32,
    pub boundary_max_distance: f32,
    pub extreme_distance_threshold: f32,
    pub explosion_distance_threshold: f32,
    pub spreading_distance_threshold: f32,
    pub oscillation_detection_frames: usize,
    pub oscillation_change_threshold: f32,
    pub min_oscillation_changes: usize,
    
    // New CUDA kernel parameter tuning thresholds
    pub grid_cell_size_min: f32,
    pub grid_cell_size_max: f32,
    pub repulsion_cutoff_min: f32,
    pub repulsion_cutoff_max: f32,
    pub repulsion_softening_min: f32,
    pub repulsion_softening_max: f32,
    pub center_gravity_min: f32,
    pub center_gravity_max: f32,
    
    // Spatial hashing effectiveness thresholds
    pub spatial_hash_efficiency_threshold: f32,
    pub cluster_density_threshold: f32,
    pub numerical_instability_threshold: f32,
}

impl AutoBalanceConfig {
    pub fn default() -> Self {
        Self {
            stability_variance_threshold: 100.0,
            stability_frame_count: 180,
            clustering_distance_threshold: 20.0,
            bouncing_node_percentage: 0.33,
            boundary_min_distance: 90.0,
            boundary_max_distance: 110.0,
            extreme_distance_threshold: 1000.0,
            explosion_distance_threshold: 10000.0,
            spreading_distance_threshold: 500.0,
            oscillation_detection_frames: 10,
            oscillation_change_threshold: 5.0,
            min_oscillation_changes: 5,
            
            // New CUDA kernel parameter defaults for tuning ranges
            grid_cell_size_min: 1.0,
            grid_cell_size_max: 50.0,
            repulsion_cutoff_min: 5.0,
            repulsion_cutoff_max: 200.0,
            repulsion_softening_min: 1e-6,
            repulsion_softening_max: 1.0,
            center_gravity_min: 0.0,
            center_gravity_max: 0.1,
            
            // Spatial hashing and numerical stability thresholds
            spatial_hash_efficiency_threshold: 0.3, // Below 30% efficiency triggers grid_cell_size adjustment
            cluster_density_threshold: 50.0, // Nodes per unit area that indicates clustering
            numerical_instability_threshold: 1e-3, // Threshold for detecting numerical issues
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct PhysicsSettings {
    #[serde(default)]
    pub auto_balance: bool,
    #[serde(default = "default_auto_balance_interval")]
    pub auto_balance_interval_ms: u32,
    #[serde(default)]
    pub auto_balance_config: AutoBalanceConfig,
    #[validate(range(min = 0.0, max = 10.0, message = "Attraction k must be between 0.0 and 10.0"))]
    pub attraction_k: f32,
    #[validate(range(min = 10.0, max = 10000.0, message = "Bounds size must be between 10.0 and 10000.0"))]
    pub bounds_size: f32,
    #[validate(range(min = 0.1, max = 100.0, message = "Separation radius must be between 0.1 and 100.0"))]
    pub separation_radius: f32,
    #[validate(range(min = 0.0, max = 1.0, message = "Damping must be between 0.0 and 1.0"))]
    pub damping: f32,
    pub enable_bounds: bool,
    pub enabled: bool,
    #[validate(range(min = 1, max = 10000, message = "Iterations must be between 1 and 10000"))]
    pub iterations: u32,
    #[validate(range(min = 0.1, max = 1000.0, message = "Max velocity must be between 0.1 and 1000.0"))]
    pub max_velocity: f32,
    #[validate(range(min = 0.1, max = 10000.0, message = "Max force must be between 0.1 and 10000.0"))]
    pub max_force: f32,
    #[validate(range(min = 0.0, max = 1000.0, message = "Repel k must be between 0.0 and 1000.0"))]
    pub repel_k: f32,
    #[validate(range(min = 0.0, max = 10.0, message = "Spring k must be between 0.0 and 10.0"))]
    pub spring_k: f32,
    #[validate(range(min = 0.1, max = 10.0, message = "Mass scale must be between 0.1 and 10.0"))]
    pub mass_scale: f32,
    #[validate(range(min = 0.0, max = 1.0, message = "Boundary damping must be between 0.0 and 1.0"))]
    pub boundary_damping: f32,
    #[validate(range(min = 0.001, max = 1.0, message = "Update threshold must be between 0.001 and 1.0"))]
    pub update_threshold: f32,
    #[validate(range(min = 0.001, max = 1.0, message = "Delta time must be between 0.001 and 1.0"))]
    pub dt: f32,
    #[validate(range(min = 0.0, max = 10.0, message = "Temperature must be between 0.0 and 10.0"))]
    pub temperature: f32,
    #[validate(range(min = 0.0, max = 10.0, message = "Gravity must be between 0.0 and 10.0"))]
    pub gravity: f32,
    // New GPU-aligned fields
    pub stress_weight: f32,
    pub stress_alpha: f32,
    pub boundary_limit: f32,
    pub alignment_strength: f32,
    pub cluster_strength: f32,
    pub compute_mode: i32,
    
    // CUDA kernel parameters from dev_config.toml
    pub rest_length: f32,
    pub repulsion_cutoff: f32,
    pub repulsion_softening_epsilon: f32,
    pub center_gravity_k: f32,
    pub grid_cell_size: f32,
    pub warmup_iterations: u32,
    pub cooling_rate: f32,
    pub boundary_extreme_multiplier: f32,
    pub boundary_extreme_force_multiplier: f32,
    pub boundary_velocity_damping: f32,
    // Additional GPU parameters from documentation
    pub min_distance: f32,
    pub max_repulsion_dist: f32,
    pub boundary_margin: f32,
    pub boundary_force_strength: f32,
    pub warmup_curve: String,
    pub zero_velocity_iterations: u32,
    // Clustering parameters
    pub clustering_algorithm: String,
    pub cluster_count: u32,
    pub clustering_resolution: f32,
    pub clustering_iterations: u32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            auto_balance: false,
            auto_balance_interval_ms: 500,
            auto_balance_config: AutoBalanceConfig::default(),
            attraction_k: 0.0001,
            bounds_size: 500.0,
            separation_radius: 2.0,
            damping: 0.95,
            enable_bounds: true,
            enabled: true,
            iterations: 100,
            max_velocity: 1.0,
            max_force: 100.0,
            repel_k: 50.0,
            spring_k: 0.005,
            mass_scale: 1.0,
            boundary_damping: 0.95,
            update_threshold: 0.01,
            dt: 0.016,
            temperature: 0.01,
            gravity: 0.0001,
            stress_weight: 0.1,
            stress_alpha: 0.1,
            boundary_limit: 490.0,
            alignment_strength: 0.0,
            cluster_strength: 0.0,
            compute_mode: 0,
            // CUDA kernel parameter defaults from dev_config.toml
            rest_length: 50.0,
            repulsion_cutoff: 50.0,
            repulsion_softening_epsilon: 0.0001,
            center_gravity_k: 0.0,
            grid_cell_size: 50.0,
            warmup_iterations: 100,
            cooling_rate: 0.001,
            boundary_extreme_multiplier: 2.0,
            boundary_extreme_force_multiplier: 10.0,
            boundary_velocity_damping: 0.5,
            // Additional GPU parameter defaults
            min_distance: 0.15,
            max_repulsion_dist: 50.0,
            boundary_margin: 0.85,
            boundary_force_strength: 2.0,
            warmup_curve: "quadratic".to_string(),
            zero_velocity_iterations: 5,
            // Clustering defaults
            clustering_algorithm: "none".to_string(),
            cluster_count: 5,
            clustering_resolution: 1.0,
            clustering_iterations: 30,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct RenderingSettings {
    #[validate(range(min = 0.0, max = 10.0, message = "Ambient light intensity must be between 0.0 and 10.0"))]
    pub ambient_light_intensity: f32,
    #[validate(custom(function = "validate_hex_color", message = "Must be a valid hex color (e.g., #ff0000)"))]
    pub background_color: String,
    #[validate(range(min = 0.0, max = 10.0, message = "Directional light intensity must be between 0.0 and 10.0"))]
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    #[validate(range(min = 0.0, max = 10.0, message = "Environment intensity must be between 0.0 and 10.0"))]
    pub environment_intensity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_map_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_bias: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct AnimationSettings {
    pub enable_motion_blur: bool,
    pub enable_node_animations: bool,
    #[validate(range(min = 0.0, max = 5.0, message = "Motion blur strength must be between 0.0 and 5.0"))]
    pub motion_blur_strength: f32,
    pub selection_wave_enabled: bool,
    pub pulse_enabled: bool,
    #[validate(range(min = 0.0, max = 10.0, message = "Pulse speed must be between 0.0 and 10.0"))]
    pub pulse_speed: f32,
    #[validate(range(min = 0.0, max = 5.0, message = "Pulse strength must be between 0.0 and 5.0"))]
    pub pulse_strength: f32,
    #[validate(range(min = 0.0, max = 10.0, message = "Wave speed must be between 0.0 and 10.0"))]
    pub wave_speed: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct LabelSettings {
    #[validate(range(min = 8.0, max = 72.0, message = "Desktop font size must be between 8.0 and 72.0"))]
    pub desktop_font_size: f32,
    pub enable_labels: bool,
    #[validate(custom(function = "validate_hex_color", message = "Must be a valid hex color (e.g., #ff0000)"))]
    pub text_color: String,
    #[validate(custom(function = "validate_hex_color", message = "Must be a valid hex color (e.g., #ff0000)"))]
    pub text_outline_color: String,
    #[validate(range(min = 0.0, max = 10.0, message = "Text outline width must be between 0.0 and 10.0"))]
    pub text_outline_width: f32,
    #[validate(range(min = 64, max = 2048, message = "Text resolution must be between 64 and 2048"))]
    pub text_resolution: u32,
    #[validate(range(min = 0.0, max = 50.0, message = "Text padding must be between 0.0 and 50.0"))]
    pub text_padding: f32,
    pub billboard_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_metadata: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_label_width: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct GlowSettings {
    pub enabled: bool,
    #[validate(range(min = 0.0, max = 10.0, message = "Glow intensity must be between 0.0 and 10.0"))]
    pub intensity: f32,
    #[validate(range(min = 0.0, max = 50.0, message = "Glow radius must be between 0.0 and 50.0"))]
    pub radius: f32,
    #[validate(range(min = 0.0, max = 1.0, message = "Glow threshold must be between 0.0 and 1.0"))]
    pub threshold: f32,
    #[serde(default)]
    pub diffuse_strength: f32,
    #[serde(default)]
    pub atmospheric_density: f32,
    #[serde(default)]
    pub volumetric_intensity: f32,
    #[serde(skip_serializing_if = "String::is_empty", default = "default_glow_color")]
    pub base_color: String,
    #[serde(skip_serializing_if = "String::is_empty", default = "default_glow_color")]
    pub emission_color: String,
    #[serde(default = "default_glow_opacity")]
    pub opacity: f32,
    #[serde(default)]
    pub pulse_speed: f32,
    #[serde(default)]
    pub flow_speed: f32,
    #[serde(default)]
    pub node_glow_strength: f32,
    #[serde(default)]
    pub edge_glow_strength: f32,
    #[serde(default)]
    pub environment_glow_strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct HologramSettings {
    pub ring_count: u32,
    pub ring_color: String,
    pub ring_opacity: f32,
    pub sphere_sizes: Vec<f32>,
    pub ring_rotation_speed: f32,
    pub enable_buckminster: bool,
    pub buckminster_size: f32,
    pub buckminster_opacity: f32,
    pub enable_geodesic: bool,
    pub geodesic_size: f32,
    pub geodesic_opacity: f32,
    pub enable_triangle_sphere: bool,
    pub triangle_sphere_size: f32,
    pub triangle_sphere_opacity: f32,
    pub global_rotation_speed: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct CameraSettings {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub position: Position,
    pub look_at: Position,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct SpacePilotSettings {
    pub enabled: bool,
    pub mode: String,
    pub sensitivity: Sensitivity,
    pub smoothing: f32,
    pub deadzone: f32,
    pub button_functions: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct Sensitivity {
    pub translation: f32,
    pub rotation: f32,
}

// Graph-specific settings
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct GraphSettings {
    #[validate(nested)]
    pub nodes: NodeSettings,
    #[validate(nested)]
    pub edges: EdgeSettings,
    #[validate(nested)]
    pub labels: LabelSettings,
    #[validate(nested)]
    pub physics: PhysicsSettings,
}

// Multi-graph container
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct GraphsSettings {
    #[validate(nested)]
    pub logseq: GraphSettings,
    #[validate(nested)]
    pub visionflow: GraphSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct VisualisationSettings {
    
    // Global settings
    #[validate(nested)]
    pub rendering: RenderingSettings,
    #[validate(nested)]
    pub animations: AnimationSettings,
    #[validate(nested)]
    pub glow: GlowSettings,
    pub hologram: HologramSettings,
    #[validate(nested)]
    pub graphs: GraphsSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space_pilot: Option<SpacePilotSettings>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct NetworkSettings {
    #[validate(length(min = 7, max = 45, message = "Bind address must be between 7 and 45 characters"))]
    pub bind_address: String,
    #[validate(length(min = 1, max = 253, message = "Domain must be between 1 and 253 characters"))]
    pub domain: String,
    pub enable_http2: bool,
    pub enable_rate_limiting: bool,
    pub enable_tls: bool,
    #[validate(range(min = 1024, max = 104857600, message = "Max request size must be between 1KB and 100MB"))]
    pub max_request_size: usize,
    #[validate(length(min = 1, message = "TLS version cannot be empty"))]
    pub min_tls_version: String,
    #[validate(range(min = 1, max = 65535, message = "Port must be between 1 and 65535"))]
    pub port: u16,
    #[validate(range(min = 1, max = 10000, message = "Rate limit requests must be between 1 and 10000"))]
    pub rate_limit_requests: u32,
    #[validate(range(min = 1, max = 3600, message = "Rate limit window must be between 1 and 3600 seconds"))]
    pub rate_limit_window: u32,
    pub tunnel_id: String,
    #[validate(range(min = 1, max = 300, message = "API client timeout must be between 1 and 300 seconds"))]
    pub api_client_timeout: u64,
    pub enable_metrics: bool,
    #[validate(range(min = 1, max = 1000, message = "Max concurrent requests must be between 1 and 1000"))]
    pub max_concurrent_requests: u32,
    #[validate(range(min = 0, max = 10, message = "Max retries must be between 0 and 10"))]
    pub max_retries: u32,
    #[validate(range(min = 1, max = 65535, message = "Metrics port must be between 1 and 65535"))]
    pub metrics_port: u16,
    #[validate(range(min = 100, max = 30000, message = "Retry delay must be between 100ms and 30 seconds"))]
    pub retry_delay: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct WebSocketSettings {
    #[validate(range(min = 64, max = 65536, message = "Binary chunk size must be between 64 bytes and 64KB"))]
    pub binary_chunk_size: usize,
    #[validate(range(min = 1, max = 120, message = "Binary update rate must be between 1 and 120 FPS"))]
    pub binary_update_rate: u32,
    #[validate(range(min = 1, max = 30, message = "Min update rate must be between 1 and 30 FPS"))]
    pub min_update_rate: u32,
    #[validate(range(min = 30, max = 120, message = "Max update rate must be between 30 and 120 FPS"))]
    pub max_update_rate: u32,
    #[validate(range(min = 0.001, max = 1.0, message = "Motion threshold must be between 0.001 and 1.0"))]
    pub motion_threshold: f32,
    #[validate(range(min = 0.1, max = 1.0, message = "Motion damping must be between 0.1 and 1.0"))]
    pub motion_damping: f32,
    #[validate(range(min = 1, max = 255, message = "Binary message version must be between 1 and 255"))]
    pub binary_message_version: u32,
    pub compression_enabled: bool,
    #[validate(range(min = 64, max = 8192, message = "Compression threshold must be between 64 bytes and 8KB"))]
    pub compression_threshold: usize,
    #[validate(range(min = 1000, max = 60000, message = "Heartbeat interval must be between 1 and 60 seconds"))]
    pub heartbeat_interval: u64,
    #[validate(range(min = 5000, max = 600000, message = "Heartbeat timeout must be between 5 seconds and 10 minutes"))]
    pub heartbeat_timeout: u64,
    #[validate(range(min = 1, max = 10000, message = "Max connections must be between 1 and 10000"))]
    pub max_connections: usize,
    #[validate(range(min = 1024, max = 104857600, message = "Max message size must be between 1KB and 100MB"))]
    pub max_message_size: usize,
    #[validate(range(min = 0, max = 20, message = "Reconnect attempts must be between 0 and 20"))]
    pub reconnect_attempts: u32,
    #[validate(range(min = 100, max = 30000, message = "Reconnect delay must be between 100ms and 30 seconds"))]
    pub reconnect_delay: u64,
    #[validate(range(min = 1, max = 120, message = "Update rate must be between 1 and 120 FPS"))]
    pub update_rate: u32,
}

impl Default for WebSocketSettings {
    fn default() -> Self {
        Self {
            binary_chunk_size: 2048,
            binary_update_rate: 30,
            min_update_rate: 5,
            max_update_rate: 60,
            motion_threshold: 0.05,
            motion_damping: 0.9,
            binary_message_version: 1,
            compression_enabled: false,
            compression_threshold: 512,
            heartbeat_interval: 10000,
            heartbeat_timeout: 600000,
            max_connections: 100,
            max_message_size: 10485760,
            reconnect_attempts: 5,
            reconnect_delay: 1000,
            update_rate: 60,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct SecuritySettings {
    pub allowed_origins: Vec<String>,
    pub audit_log_path: String,
    pub cookie_httponly: bool,
    pub cookie_samesite: String,
    pub cookie_secure: bool,
    pub csrf_token_timeout: u32,
    pub enable_audit_logging: bool,
    pub enable_request_validation: bool,
    pub session_timeout: u32,
}

// Simple debug settings for server-side control
#[derive(Debug, Serialize, Deserialize, Clone, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct DebugSettings {
    #[serde(default)]
    pub enabled: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            enabled: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct SystemSettings {
    pub network: NetworkSettings,
    pub websocket: WebSocketSettings,
    pub security: SecuritySettings,
    pub debug: DebugSettings,
    #[serde(default)]
    pub persist_settings: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_backend_url: Option<String>,
}

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            network: NetworkSettings::default(),
            websocket: WebSocketSettings::default(),
            security: SecuritySettings::default(),
            debug: DebugSettings::default(),
            persist_settings: false,
            custom_backend_url: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct XRSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_side_enable_xr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    pub room_scale: f32,
    pub space_type: String,
    pub quality: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render_scale: Option<f32>,
    pub interaction_distance: f32,
    pub locomotion_method: String,
    pub teleport_ray_color: String,
    pub controller_ray_color: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controller_model: Option<String>,
    
    pub enable_hand_tracking: bool,
    pub hand_mesh_enabled: bool,
    pub hand_mesh_color: String,
    pub hand_mesh_opacity: f32,
    pub hand_point_size: f32,
    pub hand_ray_enabled: bool,
    pub hand_ray_color: String,
    pub hand_ray_width: f32,
    pub gesture_smoothing: f32,
    
    pub enable_haptics: bool,
    pub haptic_intensity: f32,
    pub drag_threshold: f32,
    pub pinch_threshold: f32,
    pub rotation_threshold: f32,
    pub interaction_radius: f32,
    pub movement_speed: f32,
    pub dead_zone: f32,
    pub movement_axes: MovementAxes,
    
    pub enable_light_estimation: bool,
    pub enable_plane_detection: bool,
    pub enable_scene_understanding: bool,
    pub plane_color: String,
    pub plane_opacity: f32,
    pub plane_detection_distance: f32,
    pub show_plane_overlay: bool,
    pub snap_to_floor: bool,
    
    pub enable_passthrough_portal: bool,
    pub passthrough_opacity: f32,
    pub passthrough_brightness: f32,
    pub passthrough_contrast: f32,
    pub portal_size: f32,
    pub portal_edge_color: String,
    pub portal_edge_width: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct AuthSettings {
    pub enabled: bool,
    pub provider: String,
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct RagFlowSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct PerplexitySettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct OpenAISettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct KokoroSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct WhisperSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vad_filter: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
}

// Constraint system structures
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct ConstraintData {
    pub constraint_type: i32,  // 0=none, 1=separation, 2=boundary, 3=alignment, 4=cluster
    pub strength: f32,
    pub param1: f32,
    pub param2: f32,
    pub node_mask: i32,  // Bit mask for selective application
    pub enabled: bool,
}

impl Default for ConstraintData {
    fn default() -> Self {
        Self {
            constraint_type: 0,
            strength: 1.0,
            param1: 0.0,
            param2: 0.0,
            node_mask: 0,
            enabled: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct ConstraintSystem {
    pub separation: ConstraintData,
    pub boundary: ConstraintData,
    pub alignment: ConstraintData,
    pub cluster: ConstraintData,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct ClusteringConfiguration {
    pub algorithm: String,
    pub num_clusters: u32,
    pub resolution: f32,
    pub iterations: u32,
    pub export_assignments: bool,
    pub auto_update: bool,
}

// Helper struct for physics updates
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct PhysicsUpdate {
    pub damping: Option<f32>,
    pub spring_k: Option<f32>,
    pub repel_k: Option<f32>,
    pub iterations: Option<u32>,
    pub enabled: Option<bool>,
    pub bounds_size: Option<f32>,
    pub enable_bounds: Option<bool>,
    pub max_velocity: Option<f32>,
    pub max_force: Option<f32>,
    pub separation_radius: Option<f32>,
    pub attraction_k: Option<f32>,
    pub mass_scale: Option<f32>,
    pub boundary_damping: Option<f32>,
    pub dt: Option<f32>,
    pub temperature: Option<f32>,
    pub gravity: Option<f32>,
    pub update_threshold: Option<f32>,
    // New GPU-aligned fields
    pub stress_weight: Option<f32>,
    pub stress_alpha: Option<f32>,
    pub boundary_limit: Option<f32>,
    pub alignment_strength: Option<f32>,
    pub cluster_strength: Option<f32>,
    pub compute_mode: Option<i32>,
    // Additional GPU parameters
    pub min_distance: Option<f32>,
    pub max_repulsion_dist: Option<f32>,
    pub boundary_margin: Option<f32>,
    pub boundary_force_strength: Option<f32>,
    pub warmup_iterations: Option<u32>,
    pub warmup_curve: Option<String>,
    pub zero_velocity_iterations: Option<u32>,
    pub cooling_rate: Option<f32>,
    // Clustering parameters
    pub clustering_algorithm: Option<String>,
    pub cluster_count: Option<u32>,
    pub clustering_resolution: Option<f32>,
    pub clustering_iterations: Option<u32>,
    // New CUDA kernel parameters
    pub repulsion_softening_epsilon: Option<f32>,
    pub center_gravity_k: Option<f32>,
    pub grid_cell_size: Option<f32>,
    pub rest_length: Option<f32>,
}

// Single unified settings struct
#[derive(Debug, Clone, Deserialize, Serialize, Type, Validate)]
// Using default snake_case for YAML compatibility, API handles conversion
pub struct AppFullSettings {
    #[validate(nested)]
    pub visualisation: VisualisationSettings,
    #[validate(nested)]
    pub system: SystemSettings,
    #[validate(nested)]
    pub xr: XRSettings,
    pub auth: AuthSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ragflow: Option<RagFlowSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perplexity: Option<PerplexitySettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAISettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kokoro: Option<KokoroSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub whisper: Option<WhisperSettings>,
}

impl Default for AppFullSettings {
    fn default() -> Self {
        Self {
            visualisation: VisualisationSettings::default(),
            system: SystemSettings::default(),
            xr: XRSettings::default(),
            auth: AuthSettings::default(),
            ragflow: None,
            perplexity: None,
            openai: None,
            kokoro: None,
            whisper: None,
        }
    }
}

impl AppFullSettings {
    pub fn new() -> Result<Self, ConfigError> {
        debug!("Initializing AppFullSettings from YAML");
        dotenvy::dotenv().ok();

        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/settings.yaml"));
        debug!("Loading AppFullSettings from YAML file: {:?}", settings_path);

        // Try direct YAML deserialization first (respects serde attributes properly)
        if let Ok(yaml_content) = std::fs::read_to_string(&settings_path) {
            debug!("Attempting direct YAML deserialization...");
            match serde_yaml::from_str::<AppFullSettings>(&yaml_content) {
                Ok(settings) => {
                    info!("Successfully loaded settings using direct YAML deserialization");
                    return Ok(settings);
                }
                Err(yaml_err) => {
                    debug!("Direct YAML deserialization failed: {}, trying config crate fallback", yaml_err);
                }
            }
        }

        // Fallback to config crate approach (with environment variable support)
        let builder = ConfigBuilder::<config::builder::DefaultState>::default()
            .add_source(config::File::from(settings_path.clone()).required(true))
            .add_source(
                Environment::default()
                    .separator("_")
                    .list_separator(",")
            );
        let config = builder.build()?;
        debug!("Configuration built successfully. Deserializing AppFullSettings...");

        let settings: AppFullSettings = match config.clone().try_deserialize() {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to deserialize AppFullSettings from {:?}: {}", settings_path, e);
                match config.try_deserialize::<Value>() {
                    Ok(raw_value) => error!("Raw settings structure from YAML: {:?}", raw_value),
                    Err(val_err) => error!("Failed to deserialize into raw Value as well: {:?}", val_err),
                }
                return Err(e);
            }
        };
        
        
        Ok(settings)
    }
    

    pub fn save(&self) -> Result<(), String> {
        // Check if persist_settings is enabled
        if !self.system.persist_settings {
            debug!("Settings persistence is disabled (persist_settings: false), skipping save");
            return Ok(());
        }

        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/settings.yaml"));
        info!("Saving AppFullSettings to YAML file: {:?}", settings_path);

        let yaml = serde_yaml::to_string(&self)
            .map_err(|e| format!("Failed to serialize AppFullSettings to YAML: {}", e))?;

        std::fs::write(&settings_path, yaml)
            .map_err(|e| format!("Failed to write settings file {:?}: {}", settings_path, e))?;
        info!("Successfully saved AppFullSettings to {:?}", settings_path);
        Ok(())
    }
    
    /// Get physics settings for a specific graph
    /// - "logseq": Knowledge graph (primary) - for visualizing knowledge/data relationships
    /// - "visionflow": Agent graph (secondary) - for visualizing AI agents and their interactions
    /// - "bots": Alias for visionflow/agent graph
    /// Default: logseq (knowledge graph)
    pub fn get_physics(&self, graph: &str) -> &PhysicsSettings {
        match graph {
            "logseq" | "knowledge" => &self.visualisation.graphs.logseq.physics,
            "visionflow" | "agent" | "bots" => &self.visualisation.graphs.visionflow.physics,
            _ => {
                log::debug!("Unknown graph type '{}', defaulting to logseq (knowledge graph)", graph);
                &self.visualisation.graphs.logseq.physics
            }
        }
    }
    
    // Physics updates now handled through the general merge_update method
    // which provides better validation and consistency
    
    /// Convert settings to JSON for client consumption
    /// Note: serde with // Using default snake_case for YAML compatibility, API handles conversion handles case conversion automatically
    pub fn to_json(&self) -> Result<Value, String> {
        serde_json::to_value(self)
            .map_err(|e| format!("Failed to serialize settings: {}", e))
    }
    
    /// Validate the entire configuration
    pub fn validate_config(&self) -> Result<(), HashMap<String, String>> {
        match self.validate() {
            Ok(()) => Ok(()),
            Err(errors) => {
                let mut error_map = HashMap::new();
                for (field, field_errors) in errors.field_errors() {
                    for error in field_errors {
                        let message = error.message
                            .as_ref()
                            .map(|m| m.to_string())
                            .unwrap_or_else(|| format!("Validation error in field: {}", field));
                        error_map.insert(field.to_string(), message);
                    }
                }
                Err(error_map)
            }
        }
    }

    /// Validate the entire configuration and return camelCase field names
    pub fn validate_config_camel_case(&self) -> Result<(), HashMap<String, String>> {
        match self.validate_config() {
            Ok(()) => Ok(()),
            Err(errors) => {
                // Return errors as-is since serde structs with // Using default snake_case for YAML compatibility, API handles conversion
                // will handle field name conversion automatically
                Err(errors)
            }
        }
    }

    /// Deep merge partial update into settings
    pub fn merge_update(&mut self, update: serde_json::Value) -> Result<(), String> {
        // Debug: Log the incoming update (only if debug is enabled)
        if crate::utils::logging::is_debug_enabled() {
            debug!("merge_update: Incoming update (camelCase): {}", serde_json::to_string_pretty(&update).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        
        // Direct merge without empty string conversion - frontend should send proper values
        let current_value = serde_json::to_value(&self)
            .map_err(|e| format!("Failed to serialize current settings: {}", e))?;
        
        let merged = merge_json_values(current_value, update.clone());
        if crate::utils::logging::is_debug_enabled() {
            debug!("merge_update: After merge: {}", serde_json::to_string_pretty(&merged).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        
        // Deserialize back to AppFullSettings
        *self = serde_json::from_value(merged.clone())
            .map_err(|e| {
                if crate::utils::logging::is_debug_enabled() {
                    error!("merge_update: Failed to deserialize merged JSON: {}", serde_json::to_string_pretty(&merged).unwrap_or_else(|_| "Could not serialize".to_string()));
                    error!("merge_update: Original update was: {}", serde_json::to_string_pretty(&update).unwrap_or_else(|_| "Could not serialize".to_string()));
                }
                format!("Failed to deserialize merged settings: {}", e)
            })?;
        
        // Validate the merged settings with camelCase field names
        if let Err(validation_errors) = self.validate_config_camel_case() {
            let error_summary = validation_errors
                .iter()
                .map(|(field, message)| format!("{}: {}", field, message))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(format!("Validation failed after merge: {}", error_summary));
        }
        
        Ok(())
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_errors_use_camel_case_field_names() {
        let mut settings = AppFullSettings::default();
        
        // Set invalid physics values that should trigger validation errors
        settings.visualisation.graphs.logseq.physics.attraction_k = -1.0; // Invalid: should be between 0.0 and 10.0
        settings.visualisation.graphs.logseq.physics.bounds_size = 5.0;   // Invalid: should be between 10.0 and 10000.0  
        settings.visualisation.graphs.logseq.physics.damping = 2.0;       // Invalid: should be between 0.0 and 1.0

        // Test camelCase validation
        let result = settings.validate_config_camel_case();
        assert!(result.is_err(), "Validation should fail with invalid values");

        let errors = result.unwrap_err();
        
        // Check that field names are in camelCase
        let error_keys: Vec<&String> = errors.keys().collect();
        println!("Validation error fields: {:?}", error_keys);
        
        // Look for camelCase field patterns
        let has_camel_case_fields = error_keys.iter().any(|key| {
            key.contains("attractionK") || 
            key.contains("boundsSize") || 
            key.contains("damping") ||
            // These are nested paths, so they might look like:
            // visualisation.graphs.logseq.physics.attractionK
            key.ends_with(".attractionK") ||
            key.ends_with(".boundsSize")
        });
        
        assert!(has_camel_case_fields, "Field names should be in camelCase format. Found: {:?}", error_keys);
    }

    #[test]
    fn test_snake_case_validation_still_works() {
        let mut settings = AppFullSettings::default();
        
        // Set invalid physics values
        settings.visualisation.graphs.logseq.physics.attraction_k = -1.0;

        // Test snake_case validation (old method)
        let result = settings.validate_config();
        assert!(result.is_err(), "Validation should fail with invalid values");

        let errors = result.unwrap_err();
        let error_keys: Vec<&String> = errors.keys().collect();
        
        // This should contain snake_case field names
        let has_snake_case_fields = error_keys.iter().any(|key| {
            key.contains("attraction_k") || key.ends_with(".attraction_k")
        });
        
        assert!(has_snake_case_fields, "Snake case validation should still work. Found: {:?}", error_keys);
    }
}

// Use unified PathAccessible for PhysicsSettings
impl_unified_path_accessible!(PhysicsSettings);

// The macro is already available from the module

// Implement unified path access for main structs
impl_unified_path_accessible!(GraphSettings);
impl_unified_path_accessible!(GraphsSettings);
impl_unified_path_accessible!(VisualisationSettings);
impl_unified_path_accessible!(SystemSettings);

// Use unified PathAccessible for XRSettings
impl_unified_path_accessible!(XRSettings);

// Use unified PathAccessible for AuthSettings
impl_unified_path_accessible!(AuthSettings);

impl_unified_path_accessible!(AppFullSettings);


// Apply unified path access to remaining structs
impl_unified_path_accessible!(EdgeSettings);
impl_unified_path_accessible!(LabelSettings);
impl_unified_path_accessible!(RenderingSettings);
impl_unified_path_accessible!(AnimationSettings);
impl_unified_path_accessible!(GlowSettings);
impl_unified_path_accessible!(HologramSettings);
impl_unified_path_accessible!(NetworkSettings);
impl_unified_path_accessible!(WebSocketSettings);
impl_unified_path_accessible!(SecuritySettings);
impl_unified_path_accessible!(DebugSettings);
impl_unified_path_accessible!(NodeSettings);
impl_unified_path_accessible!(MovementAxes);
impl_unified_path_accessible!(AutoBalanceConfig);
impl_unified_path_accessible!(CameraSettings);
impl_unified_path_accessible!(Position);
impl_unified_path_accessible!(SpacePilotSettings);
impl_unified_path_accessible!(Sensitivity);
impl_unified_path_accessible!(RagFlowSettings);
impl_unified_path_accessible!(PerplexitySettings);
impl_unified_path_accessible!(OpenAISettings);
impl_unified_path_accessible!(KokoroSettings);
impl_unified_path_accessible!(WhisperSettings);
impl_unified_path_accessible!(ConstraintData);
impl_unified_path_accessible!(ConstraintSystem);
impl_unified_path_accessible!(ClusteringConfiguration);
impl_unified_path_accessible!(PhysicsUpdate);



