use serde::{Deserialize, Serialize};
use specta::Type;
use std::collections::HashMap;
use validator::Validate;

use super::physics::PhysicsSettings;
use super::validation::{validate_hex_color, validate_width_range};

fn default_glow_color() -> String {
    "#00ffff".to_string()
}

fn default_glow_opacity() -> f32 {
    0.8
}

fn default_bloom_intensity() -> f32 {
    1.0
}

fn default_bloom_radius() -> f32 {
    0.8
}

fn default_bloom_threshold() -> f32 {
    0.15
}

fn default_bloom_color() -> String {
    "#ffffff".to_string()
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
    #[serde(
        skip_serializing_if = "String::is_empty",
        default = "default_glow_color",
        alias = "base_color"
    )]
    pub base_color: String,
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(
        skip_serializing_if = "String::is_empty",
        default = "default_glow_color",
        alias = "emission_color"
    )]
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
    #[serde(
        skip_serializing_if = "String::is_empty",
        default = "default_bloom_color",
        alias = "color"
    )]
    pub color: String,
    #[validate(custom(function = "validate_hex_color"))]
    #[serde(
        skip_serializing_if = "String::is_empty",
        default = "default_bloom_color",
        alias = "tint_color"
    )]
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
    pub button_functions: HashMap<String, String>,
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
