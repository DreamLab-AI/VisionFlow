use config::{ConfigBuilder, ConfigError, Environment};
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use std::path::PathBuf;

pub mod dev_config;

fn default_auto_balance_interval() -> u32 {
    500
}

pub mod feature_access;

// Types are already public in this module, no need to re-export

// Helper function to convert empty strings to null for Option<String> fields
fn convert_empty_strings_to_null(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let new_map = map.into_iter().map(|(k, v)| {
                let new_v = match v {
                    Value::String(s) if s.is_empty() => {
                        // For required String fields, keep empty strings
                        // For optional fields, convert to null
                        // List of required string fields that should NOT be null
                        let required_string_fields = vec![
                            "base_color", "color", "background_color", "text_color",
                            "text_outline_color", "billboard_mode", "quality", "mode",
                            "context", "cookie_samesite", "audit_log_path", "bind_address",
                            "domain", "min_tls_version", "tunnel_id", "provider",
                            "ring_color", "hand_mesh_color", "hand_ray_color",
                            "teleport_ray_color", "controller_ray_color", "plane_color",
                            "portal_edge_color", "space_type", "locomotion_method"
                        ];
                        
                        if required_string_fields.contains(&k.as_str()) {
                            // Keep empty string for required fields
                            Value::String(s)
                        } else {
                            // Convert to null for optional fields
                            Value::Null
                        }
                    },
                    Value::Object(_) => convert_empty_strings_to_null(v),
                    Value::Array(_) => convert_empty_strings_to_null(v),
                    _ => v,
                };
                (k, new_v)
            }).collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(convert_empty_strings_to_null).collect())
        }
        _ => value,
    }
}

// Recursive function to convert JSON Value keys to snake_case
fn keys_to_snake_case(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let new_map = map.into_iter().map(|(k, v)| {
                let snake_key = k.chars().fold(String::new(), |mut acc, c| {
                    if c.is_ascii_uppercase() {
                        if !acc.is_empty() {
                            acc.push('_');
                        }
                        acc.push(c.to_ascii_lowercase());
                    } else {
                        acc.push(c);
                    }
                    acc
                });
                (snake_key, keys_to_snake_case(v))
            }).collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(keys_to_snake_case).collect())
        }
        _ => value,
    }
}

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

// Helper to convert camelCase to snake_case
fn keys_to_camel_case(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let new_map = map.into_iter().map(|(k, v)| {
                let camel_key = k.split('_').enumerate().map(|(i, part)| {
                    if i == 0 {
                        part.to_string()
                    } else {
                        part.chars().next().map_or(String::new(), |c| {
                            c.to_uppercase().collect::<String>() + &part[1..]
                        })
                    }
                }).collect::<String>();
                (camel_key, keys_to_camel_case(v))
            }).collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(keys_to_camel_case).collect())
        }
        _ => value,
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MovementAxes {
    pub horizontal: i32,
    pub vertical: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct EdgeSettings {
    pub arrow_size: f32,
    pub base_width: f32,
    pub color: String,
    pub enable_arrows: bool,
    pub opacity: f32,
    pub width_range: Vec<f32>,
    pub quality: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PhysicsSettings {
    #[serde(default)]
    pub auto_balance: bool,
    #[serde(default = "default_auto_balance_interval")]
    pub auto_balance_interval_ms: u32,
    #[serde(default)]
    pub auto_balance_config: AutoBalanceConfig,
    pub attraction_k: f32,
    pub bounds_size: f32,
    pub separation_radius: f32,
    pub damping: f32,
    pub enable_bounds: bool,
    pub enabled: bool,
    pub iterations: u32,
    pub max_velocity: f32,
    pub max_force: f32,
    pub repel_k: f32,
    pub spring_k: f32,
    pub mass_scale: f32,
    pub boundary_damping: f32,
    pub update_threshold: f32,
    pub dt: f32,
    pub temperature: f32,
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub background_color: String,
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub environment_intensity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_map_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_bias: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct LabelSettings {
    pub desktop_font_size: f32,
    pub enable_labels: bool,
    pub text_color: String,
    pub text_outline_color: String,
    pub text_outline_width: f32,
    pub text_resolution: u32,
    pub text_padding: f32,
    pub billboard_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_metadata: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_label_width: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BloomSettings {
    pub edge_bloom_strength: f32,
    pub enabled: bool,
    pub environment_bloom_strength: f32,
    pub node_bloom_strength: f32,
    pub radius: f32,
    pub strength: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct CameraSettings {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub position: Position,
    pub look_at: Position,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SpacePilotSettings {
    pub enabled: bool,
    pub mode: String,
    pub sensitivity: Sensitivity,
    pub smoothing: f32,
    pub deadzone: f32,
    pub button_functions: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Sensitivity {
    pub translation: f32,
    pub rotation: f32,
}

// Graph-specific settings
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GraphSettings {
    pub nodes: NodeSettings,
    pub edges: EdgeSettings,
    pub labels: LabelSettings,
    pub physics: PhysicsSettings,
}

// Multi-graph container
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GraphsSettings {
    pub logseq: GraphSettings,
    pub visionflow: GraphSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct VisualisationSettings {
    
    // Global settings
    pub rendering: RenderingSettings,
    pub animations: AnimationSettings,
    pub bloom: BloomSettings,
    pub hologram: HologramSettings,
    pub graphs: GraphsSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space_pilot: Option<SpacePilotSettings>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct NetworkSettings {
    pub bind_address: String,
    pub domain: String,
    pub enable_http2: bool,
    pub enable_rate_limiting: bool,
    pub enable_tls: bool,
    pub max_request_size: usize,
    pub min_tls_version: String,
    pub port: u16,
    pub rate_limit_requests: u32,
    pub rate_limit_window: u32,
    pub tunnel_id: String,
    pub api_client_timeout: u64,
    pub enable_metrics: bool,
    pub max_concurrent_requests: u32,
    pub max_retries: u32,
    pub metrics_port: u16,
    pub retry_delay: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebSocketSettings {
    pub binary_chunk_size: usize,
    pub binary_update_rate: u32,
    pub min_update_rate: u32,
    pub max_update_rate: u32,
    pub motion_threshold: f32,
    pub motion_damping: f32,
    pub binary_message_version: u32,
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    pub heartbeat_interval: u64,
    pub heartbeat_timeout: u64,
    pub max_connections: usize,
    pub max_message_size: usize,
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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
#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AuthSettings {
    pub enabled: bool,
    pub provider: String,
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ConstraintSystem {
    pub separation: ConstraintData,
    pub boundary: ConstraintData,
    pub alignment: ConstraintData,
    pub cluster: ConstraintData,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ClusteringConfiguration {
    pub algorithm: String,
    pub num_clusters: u32,
    pub resolution: f32,
    pub iterations: u32,
    pub export_assignments: bool,
    pub auto_update: bool,
}

// Helper struct for physics updates
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings,
    pub system: SystemSettings,
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
    
    /// Deep merge partial update into settings
    pub fn merge_update(&mut self, update: serde_json::Value) -> Result<(), String> {
        // Debug: Log the incoming update (only if debug is enabled)
        if crate::utils::logging::is_debug_enabled() {
            debug!("merge_update: Incoming update (camelCase): {}", serde_json::to_string_pretty(&update).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        
        // Convert from camelCase to snake_case
        let mut snake_update = keys_to_snake_case(update.clone());
        if crate::utils::logging::is_debug_enabled() {
            debug!("merge_update: After snake_case conversion: {}", serde_json::to_string_pretty(&snake_update).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        
        // Convert empty strings to null for Option<String> fields
        snake_update = convert_empty_strings_to_null(snake_update);
        if crate::utils::logging::is_debug_enabled() {
            debug!("merge_update: After null conversion: {}", serde_json::to_string_pretty(&snake_update).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        
        // Merge the update into self
        let current_value = serde_json::to_value(&self)
            .map_err(|e| format!("Failed to serialize current settings: {}", e))?;
        
        let merged = merge_json_values(current_value, snake_update);
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
        
        Ok(())
    }
    
    /// Convert to camelCase JSON for client
    pub fn to_camel_case_json(&self) -> Result<Value, String> {
        let snake_json = serde_json::to_value(&self)
            .map_err(|e| format!("Failed to serialize settings: {}", e))?;
        Ok(keys_to_camel_case(snake_json))
    }
}

#[cfg(test)]
mod tests {
    // mod feature_access_test;
}