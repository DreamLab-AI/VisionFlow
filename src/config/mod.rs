use config::{ConfigBuilder, ConfigError, Environment};
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use std::path::PathBuf;
use std::collections::HashMap;

// New imports for enhanced validation and type generation
use specta::Type;
use validator::{Validate, ValidationError};
use regex::Regex;
use lazy_static::lazy_static;

pub mod dev_config;
pub mod path_access;

// Import the trait and functions we need
use path_access::{PathAccessible, parse_path};

// Centralized validation patterns
lazy_static! {
    /// Validates hex color format (#RRGGBB or #RRGGBBAA)
    static ref HEX_COLOR_REGEX: Regex = Regex::new(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$").unwrap();
    
    /// Validates URL format (http/https)
    static ref URL_REGEX: Regex = Regex::new(r"^https?://[^\s/$.?#].[^\s]*$").unwrap();
    
    /// Validates file path format (Unix/Windows compatible)
    static ref FILE_PATH_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9._/\\-]+$").unwrap();
    
    /// Validates domain name format
    static ref DOMAIN_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$").unwrap();
}

/// Custom validation functions for specific business logic
/// Validates hex color format (#RRGGBB or #RRGGBBAA)
pub fn validate_hex_color(color: &str) -> Result<(), ValidationError> {
    if !HEX_COLOR_REGEX.is_match(color) {
        return Err(ValidationError::new("invalid_hex_color"));
    }
    Ok(())
}

/// Validates width range has exactly 2 elements with proper min/max order
pub fn validate_width_range(range: &[f32]) -> Result<(), ValidationError> {
    if range.len() != 2 {
        return Err(ValidationError::new("width_range_length"));
    }
    if range[0] >= range[1] {
        return Err(ValidationError::new("width_range_order"));
    }
    Ok(())
}

/// Validates port number is in valid range (1-65535)
pub fn validate_port(port: u16) -> Result<(), ValidationError> {
    if port == 0 {
        return Err(ValidationError::new("invalid_port"));
    }
    Ok(())
}

/// Validates percentage is between 0 and 100
pub fn validate_percentage(value: f32) -> Result<(), ValidationError> {
    if !(0.0..=100.0).contains(&value) {
        return Err(ValidationError::new("invalid_percentage"));
    }
    Ok(())
}

/// Validates bloom/glow settings to prevent GPU kernel crashes
/// This function checks ranges for intensity, threshold, radius, and validates hex colors
pub fn validate_bloom_glow_settings(glow: &GlowSettings, bloom: &BloomSettings) -> Result<(), ValidationError> {
    // Validate glow settings
    if glow.intensity < 0.0 || glow.intensity > 10.0 {
        return Err(ValidationError::new("glow_intensity_out_of_range"));
    }
    if glow.radius < 0.0 || glow.radius > 10.0 {
        return Err(ValidationError::new("glow_radius_out_of_range"));
    }
    if glow.threshold < 0.0 || glow.threshold > 1.0 {
        return Err(ValidationError::new("glow_threshold_out_of_range"));
    }
    if glow.opacity < 0.0 || glow.opacity > 1.0 {
        return Err(ValidationError::new("glow_opacity_out_of_range"));
    }

    // Validate glow colors
    validate_hex_color(&glow.base_color)?;
    validate_hex_color(&glow.emission_color)?;

    // Check for NaN/Infinity values that would crash GPU kernel
    if !glow.intensity.is_finite() {
        return Err(ValidationError::new("glow_intensity_not_finite"));
    }
    if !glow.radius.is_finite() {
        return Err(ValidationError::new("glow_radius_not_finite"));
    }
    if !glow.threshold.is_finite() {
        return Err(ValidationError::new("glow_threshold_not_finite"));
    }

    // Validate bloom settings
    if bloom.intensity < 0.0 || bloom.intensity > 10.0 {
        return Err(ValidationError::new("bloom_intensity_out_of_range"));
    }
    if bloom.radius < 0.0 || bloom.radius > 10.0 {
        return Err(ValidationError::new("bloom_radius_out_of_range"));
    }
    if bloom.threshold < 0.0 || bloom.threshold > 1.0 {
        return Err(ValidationError::new("bloom_threshold_out_of_range"));
    }
    if bloom.strength < 0.0 || bloom.strength > 1.0 {
        return Err(ValidationError::new("bloom_strength_out_of_range"));
    }
    if bloom.knee < 0.0 || bloom.knee > 2.0 {
        return Err(ValidationError::new("bloom_knee_out_of_range"));
    }

    // Validate bloom colors
    validate_hex_color(&bloom.color)?;
    validate_hex_color(&bloom.tint_color)?;

    // Check for NaN/Infinity values in bloom settings
    if !bloom.intensity.is_finite() {
        return Err(ValidationError::new("bloom_intensity_not_finite"));
    }
    if !bloom.radius.is_finite() {
        return Err(ValidationError::new("bloom_radius_not_finite"));
    }
    if !bloom.threshold.is_finite() {
        return Err(ValidationError::new("bloom_threshold_not_finite"));
    }

    Ok(())
}

/// Converts snake_case to camelCase
fn to_camel_case(snake_str: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    
    for ch in snake_str.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    
    result
}

fn default_auto_balance_interval() -> u32 {
    500
}

fn default_constraint_ramp_frames() -> u32 {
    60  // 1 second at 60 FPS for full activation
}

fn default_constraint_max_force_per_node() -> f32 {
    50.0  // Default constraint force limit per node
}

fn default_glow_color() -> String {
    "#00ffff".to_string()
}

fn default_glow_opacity() -> f32 {
    0.8
}

pub mod feature_access;
// pub mod tests;

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


#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
#[serde(rename_all = "camelCase")]
pub struct MovementAxes {
    #[serde(alias = "horizontal")]
    pub horizontal: i32,
    #[serde(alias = "vertical")]
    pub vertical: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct NodeSettings {
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(alias = "base_color")]
    pub base_color: String,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(alias = "metalness")]
    pub metalness: f32,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(alias = "opacity")]
    pub opacity: f32,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(alias = "roughness")]
    pub roughness: f32,
    #[validate(range(min = 0.1, max = 100.0))]
    #[serde(alias = "node_size")]
    pub node_size: f32,
    #[serde(alias = "quality")]
    pub quality: String,
    #[serde(alias = "enable_instancing")]
    pub enable_instancing: bool,
    #[serde(alias = "enable_hologram")]
    pub enable_hologram: bool,
    #[serde(alias = "enable_metadata_shape")]
    pub enable_metadata_shape: bool,
    #[serde(alias = "enable_metadata_visualisation")]
    pub enable_metadata_visualisation: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct EdgeSettings {
    #[validate(range(min = 0.01, max = 5.0))]
    #[serde(alias = "arrow_size")]
    pub arrow_size: f32,
    #[validate(range(min = 0.01, max = 5.0))]
    #[serde(alias = "base_width")]
    pub base_width: f32,
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(alias = "color")]
    pub color: String,
    #[serde(alias = "enable_arrows")]
    pub enable_arrows: bool,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(alias = "opacity")]
    pub opacity: f32,
    #[validate(custom(function = "validate_width_range"))]
    #[serde(alias = "width_range")]
    pub width_range: Vec<f32>,
    #[serde(alias = "quality")]
    pub quality: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AutoPauseConfig {
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(alias = "equilibrium_velocity_threshold")]
    pub equilibrium_velocity_threshold: f32,
    #[validate(range(min = 1, max = 300))]
    #[serde(alias = "equilibrium_check_frames")]
    pub equilibrium_check_frames: u32,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(alias = "equilibrium_energy_threshold")]
    pub equilibrium_energy_threshold: f32,
    #[serde(alias = "pause_on_equilibrium")]
    pub pause_on_equilibrium: bool,
    #[serde(alias = "resume_on_interaction")]
    pub resume_on_interaction: bool,
}

impl AutoPauseConfig {
    pub fn default() -> Self {
        Self {
            enabled: true,
            equilibrium_velocity_threshold: 0.1,
            equilibrium_check_frames: 30,
            equilibrium_energy_threshold: 0.01,
            pause_on_equilibrium: true,
            resume_on_interaction: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AutoBalanceConfig {
    #[serde(alias = "stability_variance_threshold")]
    pub stability_variance_threshold: f32,
    #[serde(alias = "stability_frame_count")]
    pub stability_frame_count: u32,
    #[serde(alias = "clustering_distance_threshold")]
    pub clustering_distance_threshold: f32,
    #[serde(alias = "clustering_hysteresis_buffer")]
    pub clustering_hysteresis_buffer: f32,
    #[serde(alias = "bouncing_node_percentage")]
    pub bouncing_node_percentage: f32,
    #[serde(alias = "boundary_min_distance")]
    pub boundary_min_distance: f32,
    #[serde(alias = "boundary_max_distance")]
    pub boundary_max_distance: f32,
    #[serde(alias = "extreme_distance_threshold")]
    pub extreme_distance_threshold: f32,
    #[serde(alias = "explosion_distance_threshold")]
    pub explosion_distance_threshold: f32,
    #[serde(alias = "spreading_distance_threshold")]
    pub spreading_distance_threshold: f32,
    #[serde(alias = "spreading_hysteresis_buffer")]
    pub spreading_hysteresis_buffer: f32,
    #[serde(alias = "oscillation_detection_frames")]
    pub oscillation_detection_frames: usize,
    #[serde(alias = "oscillation_change_threshold")]
    pub oscillation_change_threshold: f32,
    #[serde(alias = "min_oscillation_changes")]
    pub min_oscillation_changes: usize,
    
    // Parameter adjustment and cooldown configuration
    #[serde(alias = "parameter_adjustment_rate")]
    pub parameter_adjustment_rate: f32,
    #[serde(alias = "max_adjustment_factor")]
    pub max_adjustment_factor: f32,
    #[serde(alias = "min_adjustment_factor")]
    pub min_adjustment_factor: f32,
    #[serde(alias = "adjustment_cooldown_ms")]
    pub adjustment_cooldown_ms: u64,
    #[serde(alias = "state_change_cooldown_ms")]
    pub state_change_cooldown_ms: u64,
    #[serde(alias = "parameter_dampening_factor")]
    pub parameter_dampening_factor: f32,
    #[serde(alias = "hysteresis_delay_frames")]
    pub hysteresis_delay_frames: u32,
    
    // New CUDA kernel parameter tuning thresholds
    #[serde(alias = "grid_cell_size_min")]
    pub grid_cell_size_min: f32,
    #[serde(alias = "grid_cell_size_max")]
    pub grid_cell_size_max: f32,
    #[serde(alias = "repulsion_cutoff_min")]
    pub repulsion_cutoff_min: f32,
    #[serde(alias = "repulsion_cutoff_max")]
    pub repulsion_cutoff_max: f32,
    #[serde(alias = "repulsion_softening_min")]
    pub repulsion_softening_min: f32,
    #[serde(alias = "repulsion_softening_max")]
    pub repulsion_softening_max: f32,
    #[serde(alias = "center_gravity_min")]
    pub center_gravity_min: f32,
    #[serde(alias = "center_gravity_max")]
    pub center_gravity_max: f32,
    
    // Spatial hashing effectiveness thresholds
    #[serde(alias = "spatial_hash_efficiency_threshold")]
    pub spatial_hash_efficiency_threshold: f32,
    #[serde(alias = "cluster_density_threshold")]
    pub cluster_density_threshold: f32,
    #[serde(alias = "numerical_instability_threshold")]
    pub numerical_instability_threshold: f32,
}

impl AutoBalanceConfig {
    pub fn default() -> Self {
        Self {
            stability_variance_threshold: 100.0,
            stability_frame_count: 180,
            clustering_distance_threshold: 20.0,
            clustering_hysteresis_buffer: 5.0,
            bouncing_node_percentage: 0.33,
            boundary_min_distance: 90.0,
            boundary_max_distance: 110.0,
            extreme_distance_threshold: 1000.0,
            explosion_distance_threshold: 10000.0,
            spreading_distance_threshold: 500.0,
            spreading_hysteresis_buffer: 50.0,
            oscillation_detection_frames: 20,
            oscillation_change_threshold: 10.0,
            min_oscillation_changes: 8,
            
            // Parameter adjustment and cooldown defaults
            parameter_adjustment_rate: 0.1,
            max_adjustment_factor: 0.2,
            min_adjustment_factor: -0.2,
            adjustment_cooldown_ms: 2000,
            state_change_cooldown_ms: 1000,
            parameter_dampening_factor: 0.05,
            hysteresis_delay_frames: 30,
            
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
#[serde(rename_all = "camelCase")]
pub struct PhysicsSettings {
    #[serde(default, alias = "auto_balance")]
    pub auto_balance: bool,
    #[serde(default = "default_auto_balance_interval", alias = "auto_balance_interval_ms")]
    pub auto_balance_interval_ms: u32,
    #[serde(default, alias = "auto_balance_config")]
    #[validate(nested)]
    pub auto_balance_config: AutoBalanceConfig,
    #[serde(default, alias = "auto_pause")]
    #[validate(nested)]
    pub auto_pause: AutoPauseConfig,
    #[serde(alias = "attraction_k")]
    pub attraction_k: f32,
    #[serde(alias = "bounds_size")]
    pub bounds_size: f32,
    #[serde(alias = "separation_radius")]
    pub separation_radius: f32,
    #[serde(alias = "damping")]
    pub damping: f32,
    #[serde(alias = "enable_bounds")]
    pub enable_bounds: bool,
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[serde(alias = "iterations")]
    pub iterations: u32,
    #[serde(alias = "max_velocity")]
    pub max_velocity: f32,
    #[serde(alias = "max_force")]
    pub max_force: f32,
    #[serde(alias = "repel_k")]
    pub repel_k: f32,
    #[serde(alias = "spring_k")]
    pub spring_k: f32,
    #[serde(alias = "mass_scale")]
    pub mass_scale: f32,
    #[serde(alias = "boundary_damping")]
    pub boundary_damping: f32,
    #[serde(alias = "update_threshold")]
    pub update_threshold: f32,
    #[serde(alias = "dt")]
    pub dt: f32,
    #[serde(alias = "temperature")]
    pub temperature: f32,
    #[serde(alias = "gravity")]
    pub gravity: f32,
    // New GPU-aligned fields
    #[serde(alias = "stress_weight")]
    pub stress_weight: f32,
    #[serde(alias = "stress_alpha")]
    pub stress_alpha: f32,
    #[serde(alias = "boundary_limit")]
    pub boundary_limit: f32,
    #[serde(alias = "alignment_strength")]
    pub alignment_strength: f32,
    #[serde(alias = "cluster_strength")]
    pub cluster_strength: f32,
    #[serde(alias = "compute_mode")]
    pub compute_mode: i32,
    
    // CUDA kernel parameters from dev_config.toml
    #[serde(alias = "rest_length")]
    pub rest_length: f32,
    #[serde(alias = "repulsion_cutoff")]
    pub repulsion_cutoff: f32,
    #[serde(alias = "repulsion_softening_epsilon")]
    pub repulsion_softening_epsilon: f32,
    #[serde(alias = "center_gravity_k")]
    pub center_gravity_k: f32,
    #[serde(alias = "grid_cell_size")]
    pub grid_cell_size: f32,
    #[serde(alias = "warmup_iterations")]
    pub warmup_iterations: u32,
    #[serde(alias = "cooling_rate")]
    pub cooling_rate: f32,
    #[serde(alias = "boundary_extreme_multiplier")]
    pub boundary_extreme_multiplier: f32,
    #[serde(alias = "boundary_extreme_force_multiplier")]
    pub boundary_extreme_force_multiplier: f32,
    #[serde(alias = "boundary_velocity_damping")]
    pub boundary_velocity_damping: f32,
    // Additional GPU parameters from documentation
    #[serde(alias = "min_distance")]
    pub min_distance: f32,
    #[serde(alias = "max_repulsion_dist")]
    pub max_repulsion_dist: f32,
    #[serde(alias = "boundary_margin")]
    pub boundary_margin: f32,
    #[serde(alias = "boundary_force_strength")]
    pub boundary_force_strength: f32,
    #[serde(alias = "warmup_curve")]
    pub warmup_curve: String,
    #[serde(alias = "zero_velocity_iterations")]
    pub zero_velocity_iterations: u32,
    
    // Constraint progressive activation parameters
    #[serde(alias = "constraint_ramp_frames", default = "default_constraint_ramp_frames")]
    pub constraint_ramp_frames: u32,
    #[serde(alias = "constraint_max_force_per_node", default = "default_constraint_max_force_per_node")]
    pub constraint_max_force_per_node: f32,
    
    // Clustering parameters
    #[serde(alias = "clustering_algorithm")]
    pub clustering_algorithm: String,
    #[serde(alias = "cluster_count")]
    pub cluster_count: u32,
    #[serde(alias = "clustering_resolution")]
    pub clustering_resolution: f32,
    #[serde(alias = "clustering_iterations")]
    pub clustering_iterations: u32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            auto_balance: false,
            auto_balance_interval_ms: 500,
            auto_balance_config: AutoBalanceConfig::default(),
            auto_pause: AutoPauseConfig::default(),
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
            // Constraint progressive activation defaults
            constraint_ramp_frames: default_constraint_ramp_frames(),
            constraint_max_force_per_node: default_constraint_max_force_per_node(),
            // Clustering defaults
            clustering_algorithm: "none".to_string(),
            cluster_count: 5,
            clustering_resolution: 1.0,
            clustering_iterations: 30,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct RenderingSettings {
    #[serde(alias = "ambient_light_intensity")]
    pub ambient_light_intensity: f32,
    #[serde(alias = "background_color")]
    pub background_color: String,
    #[serde(alias = "directional_light_intensity")]
    pub directional_light_intensity: f32,
    #[serde(alias = "enable_ambient_occlusion")]
    pub enable_ambient_occlusion: bool,
    #[serde(alias = "enable_antialiasing")]
    pub enable_antialiasing: bool,
    #[serde(alias = "enable_shadows")]
    pub enable_shadows: bool,
    #[serde(alias = "environment_intensity")]
    pub environment_intensity: f32,
    #[serde(skip_serializing_if = "Option::is_none", alias = "shadow_map_size")]
    pub shadow_map_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "shadow_bias")]
    pub shadow_bias: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "context")]
    pub context: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AnimationSettings {
    #[serde(alias = "enable_motion_blur")]
    pub enable_motion_blur: bool,
    #[serde(alias = "enable_node_animations")]
    pub enable_node_animations: bool,
    #[serde(alias = "motion_blur_strength")]
    pub motion_blur_strength: f32,
    #[serde(alias = "selection_wave_enabled")]
    pub selection_wave_enabled: bool,
    #[serde(alias = "pulse_enabled")]
    pub pulse_enabled: bool,
    #[serde(alias = "pulse_speed")]
    pub pulse_speed: f32,
    #[serde(alias = "pulse_strength")]
    pub pulse_strength: f32,
    #[serde(alias = "wave_speed")]
    pub wave_speed: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct LabelSettings {
    #[serde(alias = "desktop_font_size")]
    pub desktop_font_size: f32,
    #[serde(alias = "enable_labels")]
    pub enable_labels: bool,
    #[serde(alias = "text_color")]
    pub text_color: String,
    #[serde(alias = "text_outline_color")]
    pub text_outline_color: String,
    #[serde(alias = "text_outline_width")]
    pub text_outline_width: f32,
    #[serde(alias = "text_resolution")]
    pub text_resolution: u32,
    #[serde(alias = "text_padding")]
    pub text_padding: f32,
    #[serde(alias = "billboard_mode")]
    pub billboard_mode: String,
    #[serde(skip_serializing_if = "Option::is_none", alias = "show_metadata")]
    pub show_metadata: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "max_label_width")]
    pub max_label_width: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct GlowSettings {
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(alias = "intensity")]
    pub intensity: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(alias = "radius")]
    pub radius: f32,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(alias = "threshold")]
    pub threshold: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "diffuse_strength")]
    pub diffuse_strength: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "atmospheric_density")]
    pub atmospheric_density: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "volumetric_intensity")]
    pub volumetric_intensity: f32,
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(skip_serializing_if = "String::is_empty", default = "default_glow_color", alias = "base_color")]
    pub base_color: String,
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(skip_serializing_if = "String::is_empty", default = "default_glow_color", alias = "emission_color")]
    pub emission_color: String,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(default = "default_glow_opacity", alias = "opacity")]
    pub opacity: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "pulse_speed")]
    pub pulse_speed: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "flow_speed")]
    pub flow_speed: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "node_glow_strength")]
    pub node_glow_strength: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "edge_glow_strength")]
    pub edge_glow_strength: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default, alias = "environment_glow_strength")]
    pub environment_glow_strength: f32,
}

/// Default function for bloom intensity
fn default_bloom_intensity() -> f32 {
    1.0
}

/// Default function for bloom radius  
fn default_bloom_radius() -> f32 {
    0.8
}

/// Default function for bloom threshold
fn default_bloom_threshold() -> f32 {
    0.15
}

/// Default function for bloom color
fn default_bloom_color() -> String {
    "#ffffff".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct BloomSettings {
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default = "default_bloom_intensity", alias = "intensity")]
    pub intensity: f32,
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(default = "default_bloom_radius", alias = "radius")]
    pub radius: f32,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(default = "default_bloom_threshold", alias = "threshold")]
    pub threshold: f32,
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(skip_serializing_if = "String::is_empty", default = "default_bloom_color", alias = "color")]
    pub color: String,
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(skip_serializing_if = "String::is_empty", default = "default_bloom_color", alias = "tint_color")]
    pub tint_color: String,
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(default, alias = "strength")]
    pub strength: f32,
    #[validate(range(min = 0.0, max = 5.0))]
    #[serde(default, alias = "blur_passes")]
    pub blur_passes: f32,
    #[validate(range(min = 0.0, max = 2.0))]
    #[serde(default, alias = "knee")]
    pub knee: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            intensity: default_bloom_intensity(),
            radius: default_bloom_radius(), 
            threshold: default_bloom_threshold(),
            color: default_bloom_color(),
            tint_color: default_bloom_color(),
            strength: 0.8,
            blur_passes: 1.0,
            knee: 0.7,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct HologramSettings {
    #[serde(alias = "ring_count")]
    pub ring_count: u32,
    #[serde(alias = "ring_color")]
    pub ring_color: String,
    #[serde(alias = "ring_opacity")]
    pub ring_opacity: f32,
    #[serde(alias = "sphere_sizes")]
    pub sphere_sizes: Vec<f32>,
    #[serde(alias = "ring_rotation_speed")]
    pub ring_rotation_speed: f32,
    #[serde(alias = "enable_buckminster")]
    pub enable_buckminster: bool,
    #[serde(alias = "buckminster_size")]
    pub buckminster_size: f32,
    #[serde(alias = "buckminster_opacity")]
    pub buckminster_opacity: f32,
    #[serde(alias = "enable_geodesic")]
    pub enable_geodesic: bool,
    #[serde(alias = "geodesic_size")]
    pub geodesic_size: f32,
    #[serde(alias = "geodesic_opacity")]
    pub geodesic_opacity: f32,
    #[serde(alias = "enable_triangle_sphere")]
    pub enable_triangle_sphere: bool,
    #[serde(alias = "triangle_sphere_size")]
    pub triangle_sphere_size: f32,
    #[serde(alias = "triangle_sphere_opacity")]
    pub triangle_sphere_opacity: f32,
    #[serde(alias = "global_rotation_speed")]
    pub global_rotation_speed: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct CameraSettings {
    #[serde(alias = "fov")]
    pub fov: f32,
    #[serde(alias = "near")]
    pub near: f32,
    #[serde(alias = "far")]
    pub far: f32,
    #[serde(alias = "position")]
    pub position: Position,
    #[serde(alias = "look_at")]
    pub look_at: Position,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    #[serde(alias = "x")]
    pub x: f32,
    #[serde(alias = "y")]
    pub y: f32,
    #[serde(alias = "z")]
    pub z: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct SpacePilotSettings {
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[serde(alias = "mode")]
    pub mode: String,
    #[serde(alias = "sensitivity")]
    pub sensitivity: Sensitivity,
    #[serde(alias = "smoothing")]
    pub smoothing: f32,
    #[serde(alias = "deadzone")]
    pub deadzone: f32,
    #[serde(alias = "button_functions")]
    pub button_functions: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
#[serde(rename_all = "camelCase")]
pub struct Sensitivity {
    #[serde(alias = "translation")]
    pub translation: f32,
    #[serde(alias = "rotation")]
    pub rotation: f32,
}

// Graph-specific settings
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
pub struct GraphsSettings {
    #[validate(nested)]
    pub logseq: GraphSettings,
    #[validate(nested)]
    pub visionflow: GraphSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct VisualisationSettings {
    
    // Global settings
    #[validate(nested)]
    pub rendering: RenderingSettings,
    #[validate(nested)]
    pub animations: AnimationSettings,
    #[validate(nested)]
    pub glow: GlowSettings,
    #[validate(nested)]
    pub bloom: BloomSettings,
    #[validate(nested)]
    pub hologram: HologramSettings,
    #[validate(nested)]
    pub graphs: GraphsSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space_pilot: Option<SpacePilotSettings>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct NetworkSettings {
    #[serde(alias = "bind_address")]
    pub bind_address: String,
    #[serde(alias = "domain")]
    pub domain: String,
    #[serde(alias = "enable_http2")]
    pub enable_http2: bool,
    #[serde(alias = "enable_rate_limiting")]
    pub enable_rate_limiting: bool,
    #[serde(alias = "enable_tls")]
    pub enable_tls: bool,
    #[serde(alias = "max_request_size")]
    pub max_request_size: usize,
    #[serde(alias = "min_tls_version")]
    pub min_tls_version: String,
    #[serde(alias = "port")]
    pub port: u16,
    #[serde(alias = "rate_limit_requests")]
    pub rate_limit_requests: u32,
    #[serde(alias = "rate_limit_window")]
    pub rate_limit_window: u32,
    #[serde(alias = "tunnel_id")]
    pub tunnel_id: String,
    #[serde(alias = "api_client_timeout")]
    pub api_client_timeout: u64,
    #[serde(alias = "enable_metrics")]
    pub enable_metrics: bool,
    #[serde(alias = "max_concurrent_requests")]
    pub max_concurrent_requests: u32,
    #[serde(alias = "max_retries")]
    pub max_retries: u32,
    #[serde(alias = "metrics_port")]
    pub metrics_port: u16,
    #[serde(alias = "retry_delay")]
    pub retry_delay: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct WebSocketSettings {
    #[serde(alias = "binary_chunk_size")]
    pub binary_chunk_size: usize,
    #[serde(alias = "binary_update_rate")]
    pub binary_update_rate: u32,
    #[serde(alias = "min_update_rate")]
    pub min_update_rate: u32,
    #[serde(alias = "max_update_rate")]
    pub max_update_rate: u32,
    #[serde(alias = "motion_threshold")]
    pub motion_threshold: f32,
    #[serde(alias = "motion_damping")]
    pub motion_damping: f32,
    #[serde(alias = "binary_message_version")]
    pub binary_message_version: u32,
    #[serde(alias = "compression_enabled")]
    pub compression_enabled: bool,
    #[serde(alias = "compression_threshold")]
    pub compression_threshold: usize,
    #[serde(alias = "heartbeat_interval")]
    pub heartbeat_interval: u64,
    #[serde(alias = "heartbeat_timeout")]
    pub heartbeat_timeout: u64,
    #[serde(alias = "max_connections")]
    pub max_connections: usize,
    #[serde(alias = "max_message_size")]
    pub max_message_size: usize,
    #[serde(alias = "reconnect_attempts")]
    pub reconnect_attempts: u32,
    #[serde(alias = "reconnect_delay")]
    pub reconnect_delay: u64,
    #[serde(alias = "update_rate")]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct SecuritySettings {
    #[serde(alias = "allowed_origins")]
    pub allowed_origins: Vec<String>,
    #[serde(alias = "audit_log_path")]
    pub audit_log_path: String,
    #[serde(alias = "cookie_httponly")]
    pub cookie_httponly: bool,
    #[serde(alias = "cookie_samesite")]
    pub cookie_samesite: String,
    #[serde(alias = "cookie_secure")]
    pub cookie_secure: bool,
    #[serde(alias = "csrf_token_timeout")]
    pub csrf_token_timeout: u32,
    #[serde(alias = "enable_audit_logging")]
    pub enable_audit_logging: bool,
    #[serde(alias = "enable_request_validation")]
    pub enable_request_validation: bool,
    #[serde(alias = "session_timeout")]
    pub session_timeout: u32,
}

// Simple debug settings for server-side control
#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct DebugSettings {
    #[serde(default, alias = "enabled")]
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
#[serde(rename_all = "camelCase")]
pub struct SystemSettings {
    #[validate(nested)]
    #[serde(alias = "network")]
    pub network: NetworkSettings,
    #[validate(nested)]
    #[serde(alias = "websocket")]
    pub websocket: WebSocketSettings,
    #[validate(nested)]
    #[serde(alias = "security")]
    pub security: SecuritySettings,
    #[validate(nested)]
    #[serde(alias = "debug")]
    pub debug: DebugSettings,
    #[serde(default, alias = "persist_settings")]
    pub persist_settings: bool,
    #[serde(skip_serializing_if = "Option::is_none", alias = "custom_backend_url")]
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
#[serde(rename_all = "camelCase")]
pub struct XRSettings {
    #[serde(skip_serializing_if = "Option::is_none", alias = "enabled")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "client_side_enable_xr")]
    pub client_side_enable_xr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "mode")]
    pub mode: Option<String>,
    #[serde(alias = "room_scale")]
    pub room_scale: f32,
    #[serde(alias = "space_type")]
    pub space_type: String,
    #[serde(alias = "quality")]
    pub quality: String,
    #[serde(skip_serializing_if = "Option::is_none", alias = "render_scale")]
    pub render_scale: Option<f32>,
    #[serde(alias = "interaction_distance")]
    pub interaction_distance: f32,
    #[serde(alias = "locomotion_method")]
    pub locomotion_method: String,
    #[serde(alias = "teleport_ray_color")]
    pub teleport_ray_color: String,
    #[serde(alias = "controller_ray_color")]
    pub controller_ray_color: String,
    #[serde(skip_serializing_if = "Option::is_none", alias = "controller_model")]
    pub controller_model: Option<String>,
    
    #[serde(alias = "enable_hand_tracking")]
    pub enable_hand_tracking: bool,
    #[serde(alias = "hand_mesh_enabled")]
    pub hand_mesh_enabled: bool,
    #[serde(alias = "hand_mesh_color")]
    pub hand_mesh_color: String,
    #[serde(alias = "hand_mesh_opacity")]
    pub hand_mesh_opacity: f32,
    #[serde(alias = "hand_point_size")]
    pub hand_point_size: f32,
    #[serde(alias = "hand_ray_enabled")]
    pub hand_ray_enabled: bool,
    #[serde(alias = "hand_ray_color")]
    pub hand_ray_color: String,
    #[serde(alias = "hand_ray_width")]
    pub hand_ray_width: f32,
    #[serde(alias = "gesture_smoothing")]
    pub gesture_smoothing: f32,
    
    #[serde(alias = "enable_haptics")]
    pub enable_haptics: bool,
    #[serde(alias = "haptic_intensity")]
    pub haptic_intensity: f32,
    #[serde(alias = "drag_threshold")]
    pub drag_threshold: f32,
    #[serde(alias = "pinch_threshold")]
    pub pinch_threshold: f32,
    #[serde(alias = "rotation_threshold")]
    pub rotation_threshold: f32,
    #[serde(alias = "interaction_radius")]
    pub interaction_radius: f32,
    #[serde(alias = "movement_speed")]
    pub movement_speed: f32,
    #[serde(alias = "dead_zone")]
    pub dead_zone: f32,
    #[serde(alias = "movement_axes")]
    pub movement_axes: MovementAxes,
    
    #[serde(alias = "enable_light_estimation")]
    pub enable_light_estimation: bool,
    #[serde(alias = "enable_plane_detection")]
    pub enable_plane_detection: bool,
    #[serde(alias = "enable_scene_understanding")]
    pub enable_scene_understanding: bool,
    #[serde(alias = "plane_color")]
    pub plane_color: String,
    #[serde(alias = "plane_opacity")]
    pub plane_opacity: f32,
    #[serde(alias = "plane_detection_distance")]
    pub plane_detection_distance: f32,
    #[serde(alias = "show_plane_overlay")]
    pub show_plane_overlay: bool,
    #[serde(alias = "snap_to_floor")]
    pub snap_to_floor: bool,
    
    #[serde(alias = "enable_passthrough_portal")]
    pub enable_passthrough_portal: bool,
    #[serde(alias = "passthrough_opacity")]
    pub passthrough_opacity: f32,
    #[serde(alias = "passthrough_brightness")]
    pub passthrough_brightness: f32,
    #[serde(alias = "passthrough_contrast")]
    pub passthrough_contrast: f32,
    #[serde(alias = "portal_size")]
    pub portal_size: f32,
    #[serde(alias = "portal_edge_color")]
    pub portal_edge_color: String,
    #[serde(alias = "portal_edge_width")]
    pub portal_edge_width: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AuthSettings {
    #[serde(alias = "enabled")]
    pub enabled: bool,
    #[serde(alias = "provider")]
    pub provider: String,
    #[serde(alias = "required")]
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct RagFlowSettings {
    #[serde(skip_serializing_if = "Option::is_none", alias = "api_key")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "agent_id")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "api_base_url")]
    pub api_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "timeout")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "max_retries")]
    pub max_retries: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "chat_id")]
    pub chat_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct PerplexitySettings {
    #[serde(skip_serializing_if = "Option::is_none", alias = "api_key")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "model")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "api_url")]
    pub api_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "max_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "temperature")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "top_p")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "presence_penalty")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "frequency_penalty")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "timeout")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "rate_limit")]
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct OpenAISettings {
    #[serde(skip_serializing_if = "Option::is_none", alias = "api_key")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "base_url")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "timeout")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "rate_limit")]
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct KokoroSettings {
    #[serde(skip_serializing_if = "Option::is_none", alias = "api_url")]
    pub api_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "default_voice")]
    pub default_voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "default_format")]
    pub default_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "default_speed")]
    pub default_speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "timeout")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "stream")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "return_timestamps")]
    pub return_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "sample_rate")]
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct WhisperSettings {
    #[serde(skip_serializing_if = "Option::is_none", alias = "api_url")]
    pub api_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "default_model")]
    pub default_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "default_language")]
    pub default_language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "timeout")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "temperature")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "return_timestamps")]
    pub return_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "vad_filter")]
    pub vad_filter: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "word_timestamps")]
    pub word_timestamps: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "initial_prompt")]
    pub initial_prompt: Option<String>,
}

// Constraint system structures
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintData {
    #[serde(alias = "constraint_type")]
    pub constraint_type: i32,  // 0=none, 1=separation, 2=boundary, 3=alignment, 4=cluster
    #[serde(alias = "strength")]
    pub strength: f32,
    #[serde(alias = "param1")]
    pub param1: f32,
    #[serde(alias = "param2")]
    pub param2: f32,
    #[serde(alias = "node_mask")]
    pub node_mask: i32,  // Bit mask for selective application
    #[serde(alias = "enabled")]
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

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintSystem {
    #[serde(alias = "separation")]
    pub separation: ConstraintData,
    #[serde(alias = "boundary")]
    pub boundary: ConstraintData,
    #[serde(alias = "alignment")]
    pub alignment: ConstraintData,
    #[serde(alias = "cluster")]
    pub cluster: ConstraintData,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct ClusteringConfiguration {
    #[serde(alias = "algorithm")]
    pub algorithm: String,
    #[serde(alias = "num_clusters")]
    pub num_clusters: u32,
    #[serde(alias = "resolution")]
    pub resolution: f32,
    #[serde(alias = "iterations")]
    pub iterations: u32,
    #[serde(alias = "export_assignments")]
    pub export_assignments: bool,
    #[serde(alias = "auto_update")]
    pub auto_update: bool,
}

// Helper struct for physics updates
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct PhysicsUpdate {
    #[serde(alias = "damping")]
    pub damping: Option<f32>,
    #[serde(alias = "spring_k")]
    pub spring_k: Option<f32>,
    #[serde(alias = "repel_k")]
    pub repel_k: Option<f32>,
    #[serde(alias = "iterations")]
    pub iterations: Option<u32>,
    #[serde(alias = "enabled")]
    pub enabled: Option<bool>,
    #[serde(alias = "bounds_size")]
    pub bounds_size: Option<f32>,
    #[serde(alias = "enable_bounds")]
    pub enable_bounds: Option<bool>,
    #[serde(alias = "max_velocity")]
    pub max_velocity: Option<f32>,
    #[serde(alias = "max_force")]
    pub max_force: Option<f32>,
    #[serde(alias = "separation_radius")]
    pub separation_radius: Option<f32>,
    #[serde(alias = "attraction_k")]
    pub attraction_k: Option<f32>,
    #[serde(alias = "mass_scale")]
    pub mass_scale: Option<f32>,
    #[serde(alias = "boundary_damping")]
    pub boundary_damping: Option<f32>,
    #[serde(alias = "dt")]
    pub dt: Option<f32>,
    #[serde(alias = "temperature")]
    pub temperature: Option<f32>,
    #[serde(alias = "gravity")]
    pub gravity: Option<f32>,
    #[serde(alias = "update_threshold")]
    pub update_threshold: Option<f32>,
    // New GPU-aligned fields
    #[serde(alias = "stress_weight")]
    pub stress_weight: Option<f32>,
    #[serde(alias = "stress_alpha")]
    pub stress_alpha: Option<f32>,
    #[serde(alias = "boundary_limit")]
    pub boundary_limit: Option<f32>,
    #[serde(alias = "alignment_strength")]
    pub alignment_strength: Option<f32>,
    #[serde(alias = "cluster_strength")]
    pub cluster_strength: Option<f32>,
    #[serde(alias = "compute_mode")]
    pub compute_mode: Option<i32>,
    // Additional GPU parameters
    #[serde(alias = "min_distance")]
    pub min_distance: Option<f32>,
    #[serde(alias = "max_repulsion_dist")]
    pub max_repulsion_dist: Option<f32>,
    #[serde(alias = "boundary_margin")]
    pub boundary_margin: Option<f32>,
    #[serde(alias = "boundary_force_strength")]
    pub boundary_force_strength: Option<f32>,
    #[serde(alias = "warmup_iterations")]
    pub warmup_iterations: Option<u32>,
    #[serde(alias = "warmup_curve")]
    pub warmup_curve: Option<String>,
    #[serde(alias = "zero_velocity_iterations")]
    pub zero_velocity_iterations: Option<u32>,
    #[serde(alias = "cooling_rate")]
    pub cooling_rate: Option<f32>,
    // Clustering parameters
    #[serde(alias = "clustering_algorithm")]
    pub clustering_algorithm: Option<String>,
    #[serde(alias = "cluster_count")]
    pub cluster_count: Option<u32>,
    #[serde(alias = "clustering_resolution")]
    pub clustering_resolution: Option<f32>,
    #[serde(alias = "clustering_iterations")]
    pub clustering_iterations: Option<u32>,
    // New CUDA kernel parameters
    #[serde(alias = "repulsion_softening_epsilon")]
    pub repulsion_softening_epsilon: Option<f32>,
    #[serde(alias = "center_gravity_k")]
    pub center_gravity_k: Option<f32>,
    #[serde(alias = "grid_cell_size")]
    pub grid_cell_size: Option<f32>,
    #[serde(alias = "rest_length")]
    pub rest_length: Option<f32>,
}

// Single unified settings struct
#[derive(Debug, Clone, Deserialize, Serialize, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AppFullSettings {
    #[validate(nested)]
    #[serde(alias = "visualisation")]
    pub visualisation: VisualisationSettings,
    #[validate(nested)]
    #[serde(alias = "system")]
    pub system: SystemSettings,
    #[validate(nested)]
    #[serde(alias = "xr")]
    pub xr: XRSettings,
    #[validate(nested)]
    #[serde(alias = "auth")]
    pub auth: AuthSettings,
    #[serde(skip_serializing_if = "Option::is_none", alias = "ragflow")]
    pub ragflow: Option<RagFlowSettings>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "perplexity")]
    pub perplexity: Option<PerplexitySettings>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "openai")]
    pub openai: Option<OpenAISettings>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "kokoro")]
    pub kokoro: Option<KokoroSettings>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "whisper")]
    pub whisper: Option<WhisperSettings>,
    #[serde(default = "default_version", alias = "version")]
    pub version: String,
}

fn default_version() -> String {
    "1.0.0".to_string()
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
            version: default_version(),
        }
    }
}

impl AppFullSettings {
    /// Load AppFullSettings from a YAML file with proper snake_case to camelCase conversion
    pub fn from_yaml_file(path: &PathBuf) -> Result<Self, ConfigError> {
        let yaml_content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::Message(format!("Failed to read YAML file {:?}: {}", path, e)))?;
            
        // Try direct YAML deserialization first
        match serde_yaml::from_str::<AppFullSettings>(&yaml_content) {
            Ok(settings) => {
                debug!("Successfully loaded settings using direct YAML deserialization");
                return Ok(settings);
            }
            Err(yaml_err) => {
                debug!("Direct YAML deserialization failed: {}, trying YAML->JSON conversion", yaml_err);
                
                // Parse as raw Value first
                let raw_value = serde_yaml::from_str::<Value>(&yaml_content)
                    .map_err(|e| ConfigError::Message(format!("Failed to parse YAML as Value: {}", e)))?;
                
                // Convert to JSON string (this preserves the structure)
                let json_str = serde_json::to_string(&raw_value)
                    .map_err(|e| ConfigError::Message(format!("Failed to convert YAML to JSON: {}", e)))?;
                
                // Deserialize from JSON (this will respect serde rename attributes)
                serde_json::from_str::<AppFullSettings>(&json_str)
                    .map_err(|e| ConfigError::Message(format!("Failed to deserialize JSON: {}", e)))
            }
        }
    }

    pub fn new() -> Result<Self, ConfigError> {
        debug!("Initializing AppFullSettings from YAML");
        dotenvy::dotenv().ok();

        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/settings.yaml"));
        debug!("Loading AppFullSettings from YAML file: {:?}", settings_path);

        // Use our improved YAML loading method
        match Self::from_yaml_file(&settings_path) {
            Ok(settings) => {
                info!("Successfully loaded settings from YAML file");
                return Ok(settings);
            }
            Err(yaml_err) => {
                error!("YAML loading failed: {}", yaml_err);
                debug!("Trying config crate fallback (this may fail due to case conversion issues)");
            }
        }

        // Fallback to config crate approach (with environment variable support)
        // NOTE: This fallback doesn't respect serde rename attributes, so it will likely fail
        // with "missing field ambientLightIntensity" type errors
        let builder = ConfigBuilder::<config::builder::DefaultState>::default()
            .add_source(config::File::from(settings_path.clone()).required(true))
            .add_source(
                Environment::default()
                    .separator("_")
                    .list_separator(",")
            );
        let config = builder.build()?;
        debug!("Configuration built successfully. Attempting deserialization...");

        let settings: AppFullSettings = config.clone().try_deserialize()
            .map_err(|e| {
                error!("Config crate deserialization failed as expected: {}", e);
                error!("This is because config crate doesn't respect #[serde(rename_all = \"camelCase\")] attributes");
                e
            })?;
        
        debug!("Unexpectedly succeeded with config crate fallback");
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

        // Create parent directory if it doesn't exist
        if let Some(parent) = settings_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {:?}: {}", parent, e))?;
        }

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
        
        // Convert empty strings to null for Option<String> fields
        let processed_update = convert_empty_strings_to_null(update.clone());
        if crate::utils::logging::is_debug_enabled() {
            debug!("merge_update: After null conversion: {}", serde_json::to_string_pretty(&processed_update).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        
        // Merge the update into self
        let current_value = serde_json::to_value(&self)
            .map_err(|e| format!("Failed to serialize current settings: {}", e))?;
        
        let merged = merge_json_values(current_value, processed_update);
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
    
    /// Validates the entire configuration with camelCase field names for frontend compatibility
    pub fn validate_config_camel_case(&self) -> Result<(), validator::ValidationErrors> {
        // Validate the entire struct
        self.validate()?;
        
        // Additional cross-field validation
        self.validate_cross_field_constraints()?;
        
        Ok(())
    }
    
    /// Validates constraints that span multiple fields
    fn validate_cross_field_constraints(&self) -> Result<(), validator::ValidationErrors> {
        let mut errors = validator::ValidationErrors::new();
        
        // Example: Check that physics simulation is enabled if physics settings are configured
        if self.visualisation.graphs.logseq.physics.gravity != 0.0 && !self.visualisation.graphs.logseq.physics.enabled {
            errors.add("physics", ValidationError::new("physics_enabled_required"));
        }
        
        // Validate bloom/glow settings to prevent GPU kernel crashes
        if let Err(validation_error) = validate_bloom_glow_settings(&self.visualisation.glow, &self.visualisation.bloom) {
            errors.add("visualisation.bloom_glow", validation_error);
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    
    /// Gets user-friendly error messages in camelCase format
    pub fn get_validation_errors_camel_case(
        errors: &validator::ValidationErrors
    ) -> HashMap<String, Vec<String>> {
        let mut result = HashMap::new();
        
        for (field, field_errors) in errors.field_errors() {
            let camel_case_field = to_camel_case(field);
            let messages: Vec<String> = field_errors
                .iter()
                .map(|error| match error.code.as_ref() {
                    "invalid_hex_color" => "Must be a valid hex color (#RRGGBB or #RRGGBBAA)".to_string(),
                    "width_range_length" => "Width range must have exactly 2 values".to_string(),
                    "width_range_order" => "Width range minimum must be less than maximum".to_string(),
                    "invalid_port" => "Port must be between 1 and 65535".to_string(),
                    "invalid_percentage" => "Value must be between 0 and 100".to_string(),
                    "physics_enabled_required" => "Physics must be enabled when gravity is configured".to_string(),
                    _ => format!("Invalid value for {}", camel_case_field),
                })
                .collect();
            
            result.insert(camel_case_field, messages);
        }
        
        result
    }
    
}

// PathAccessible implementation for AppFullSettings
impl PathAccessible for AppFullSettings {
    fn get_by_path(&self, path: &str) -> Result<Box<dyn std::any::Any>, String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "visualisation" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.visualisation.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.visualisation.get_by_path(&remaining)
                }
            }
            "system" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.system.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.system.get_by_path(&remaining)
                }
            }
            "xr" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.xr.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.xr.get_by_path(&remaining)
                }
            }
            "auth" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.auth.clone()))
                } else {
                    Err("Auth fields are not deeply accessible".to_string())
                }
            }
            _ => Err(format!("Unknown top-level field: {}", segments[0]))
        }
    }
    
    fn set_by_path(&mut self, path: &str, value: Box<dyn std::any::Any>) -> Result<(), String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "visualisation" => {
                if segments.len() == 1 {
                    match value.downcast::<VisualisationSettings>() {
                        Ok(v) => {
                            self.visualisation = *v;
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for visualisation field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    self.visualisation.set_by_path(&remaining, value)
                }
            }
            "system" => {
                if segments.len() == 1 {
                    match value.downcast::<SystemSettings>() {
                        Ok(v) => {
                            self.system = *v;
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for system field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    self.system.set_by_path(&remaining, value)
                }
            }
            "xr" => {
                if segments.len() == 1 {
                    match value.downcast::<XRSettings>() {
                        Ok(v) => {
                            self.xr = *v;
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for xr field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    self.xr.set_by_path(&remaining, value)
                }
            }
            "auth" => {
                if segments.len() == 1 {
                    match value.downcast::<AuthSettings>() {
                        Ok(v) => {
                            self.auth = *v;
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for auth field".to_string())
                    }
                } else {
                    Err("Auth nested fields are not modifiable".to_string())
                }
            }
            _ => Err(format!("Unknown top-level field: {}", segments[0]))
        }
    }
}

// Basic PathAccessible implementations for nested structures
impl PathAccessible for VisualisationSettings {
    fn get_by_path(&self, path: &str) -> Result<Box<dyn std::any::Any>, String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "graphs" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.graphs.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.graphs.get_by_path(&remaining)
                }
            }
            _ => Err(format!("Only graphs field is currently supported: {}", segments[0]))
        }
    }
    
    fn set_by_path(&mut self, path: &str, value: Box<dyn std::any::Any>) -> Result<(), String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "graphs" => {
                if segments.len() == 1 {
                    match value.downcast::<GraphsSettings>() {
                        Ok(v) => { self.graphs = *v; Ok(()) }
                        Err(_) => Err("Type mismatch for graphs field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    self.graphs.set_by_path(&remaining, value)
                }
            }
            _ => Err("Only graphs field is currently supported for modification".to_string())
        }
    }
}

impl PathAccessible for GraphsSettings {
    fn get_by_path(&self, path: &str) -> Result<Box<dyn std::any::Any>, String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "logseq" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.logseq.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.logseq.get_by_path(&remaining)
                }
            }
            "visionflow" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.visionflow.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.visionflow.get_by_path(&remaining)
                }
            }
            _ => Err(format!("Unknown graph type: {}", segments[0]))
        }
    }
    
    fn set_by_path(&mut self, path: &str, value: Box<dyn std::any::Any>) -> Result<(), String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "logseq" => {
                if segments.len() == 1 {
                    match value.downcast::<GraphSettings>() {
                        Ok(v) => { self.logseq = *v; Ok(()) }
                        Err(_) => Err("Type mismatch for logseq field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    self.logseq.set_by_path(&remaining, value)
                }
            }
            "visionflow" => {
                if segments.len() == 1 {
                    match value.downcast::<GraphSettings>() {
                        Ok(v) => { self.visionflow = *v; Ok(()) }
                        Err(_) => Err("Type mismatch for visionflow field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    self.visionflow.set_by_path(&remaining, value)
                }
            }
            _ => Err(format!("Unknown graph type: {}", segments[0]))
        }
    }
}

impl PathAccessible for GraphSettings {
    fn get_by_path(&self, path: &str) -> Result<Box<dyn std::any::Any>, String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "physics" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.physics.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.physics.get_by_path(&remaining)
                }
            }
            _ => Err(format!("Only physics is supported currently: {}", segments[0]))
        }
    }
    
    fn set_by_path(&mut self, path: &str, value: Box<dyn std::any::Any>) -> Result<(), String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "physics" => {
                if segments.len() == 1 {
                    match value.downcast::<PhysicsSettings>() {
                        Ok(v) => { self.physics = *v; Ok(()) }
                        Err(_) => Err("Type mismatch for physics field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    self.physics.set_by_path(&remaining, value)
                }
            }
            _ => Err("Only physics field is currently supported for modification".to_string())
        }
    }
}

// Critical: PhysicsSettings PathAccessible implementation for performance fix
impl PathAccessible for PhysicsSettings {
    fn get_by_path(&self, path: &str) -> Result<Box<dyn std::any::Any>, String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "damping" => Ok(Box::new(self.damping)),
            "springK" => Ok(Box::new(self.spring_k)),
            "repelK" => Ok(Box::new(self.repel_k)),
            "enabled" => Ok(Box::new(self.enabled)),
            "iterations" => Ok(Box::new(self.iterations)),
            "maxVelocity" => Ok(Box::new(self.max_velocity)),
            "boundsSize" => Ok(Box::new(self.bounds_size)),
            "gravity" => Ok(Box::new(self.gravity)),
            "temperature" => Ok(Box::new(self.temperature)),
            _ => Err(format!("Unknown physics field: {}", segments[0]))
        }
    }
    
    fn set_by_path(&mut self, path: &str, value: Box<dyn std::any::Any>) -> Result<(), String> {
        let segments = parse_path(path)?;
        
        match segments[0] {
            "damping" => match value.downcast::<f32>() {
                Ok(v) => { self.damping = *v; Ok(()) }
                Err(_) => Err("Type mismatch for damping field".to_string())
            },
            "springK" => match value.downcast::<f32>() {
                Ok(v) => { self.spring_k = *v; Ok(()) }
                Err(_) => Err("Type mismatch for springK field".to_string())
            },
            "repelK" => match value.downcast::<f32>() {
                Ok(v) => { self.repel_k = *v; Ok(()) }
                Err(_) => Err("Type mismatch for repelK field".to_string())
            },
            "enabled" => match value.downcast::<bool>() {
                Ok(v) => { self.enabled = *v; Ok(()) }
                Err(_) => Err("Type mismatch for enabled field".to_string())
            },
            "iterations" => match value.downcast::<u32>() {
                Ok(v) => { self.iterations = *v; Ok(()) }
                Err(_) => Err("Type mismatch for iterations field".to_string())
            },
            "maxVelocity" => match value.downcast::<f32>() {
                Ok(v) => { self.max_velocity = *v; Ok(()) }
                Err(_) => Err("Type mismatch for maxVelocity field".to_string())
            },
            "boundsSize" => match value.downcast::<f32>() {
                Ok(v) => { self.bounds_size = *v; Ok(()) }
                Err(_) => Err("Type mismatch for boundsSize field".to_string())
            },
            "gravity" => match value.downcast::<f32>() {
                Ok(v) => { self.gravity = *v; Ok(()) }
                Err(_) => Err("Type mismatch for gravity field".to_string())
            },
            "temperature" => match value.downcast::<f32>() {
                Ok(v) => { self.temperature = *v; Ok(()) }
                Err(_) => Err("Type mismatch for temperature field".to_string())
            },
            _ => Err(format!("Unknown physics field: {}", segments[0]))
        }
    }
}

// Placeholder implementations for other structures
impl PathAccessible for SystemSettings {
    fn get_by_path(&self, _path: &str) -> Result<Box<dyn std::any::Any>, String> {
        Err("SystemSettings path access not yet implemented".to_string())
    }
    
    fn set_by_path(&mut self, _path: &str, _value: Box<dyn std::any::Any>) -> Result<(), String> {
        Err("SystemSettings path modification not yet implemented".to_string())
    }
}

impl PathAccessible for XRSettings {
    fn get_by_path(&self, _path: &str) -> Result<Box<dyn std::any::Any>, String> {
        Err("XRSettings path access not yet implemented".to_string())
    }
    
    fn set_by_path(&mut self, _path: &str, _value: Box<dyn std::any::Any>) -> Result<(), String> {
        Err("XRSettings path modification not yet implemented".to_string())
    }
}

