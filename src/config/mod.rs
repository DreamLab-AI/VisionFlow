pub mod dev_config;
pub mod path_access;
pub mod feature_access;

// Submodules split from the original monolithic mod.rs
pub mod validation;
pub mod field_mappings;
pub mod physics;
pub mod visualisation;
pub mod system;
pub mod xr;
pub mod services;
pub mod app_settings;
mod path_accessible_impls;

/// Canonical default for max_velocity across the entire codebase.
/// All modules MUST use this constant instead of hardcoded values.
pub const CANONICAL_MAX_VELOCITY: f32 = 200.0;

/// Canonical default for max_force across the entire codebase.
/// All modules MUST use this constant instead of hardcoded values.
pub const CANONICAL_MAX_FORCE: f32 = 50.0;

// Re-export all public types to preserve the existing pub API paths
// (i.e., `crate::config::AppFullSettings` still works)

pub use validation::{
    validate_bloom_glow_settings, validate_hex_color, validate_percentage, validate_port,
    validate_width_range,
};

pub use physics::{
    AutoBalanceConfig, AutoPauseConfig, ClusteringConfiguration, ConstraintSystem,
    LegacyConstraintData, PhysicsSettings, PhysicsUpdate,
};

pub use visualisation::{
    AnimationSettings, BloomSettings, CameraSettings, EdgeSettings, GlowSettings,
    GraphSettings, GraphsSettings, HologramSettings, LabelSettings, NodeSettings, Position,
    RenderingSettings, Sensitivity, SpacePilotSettings, VisualisationSettings,
};

pub use system::{
    DebugSettings, NetworkSettings, SecuritySettings, SystemSettings, WebSocketSettings,
};

pub use xr::{MovementAxes, XRSettings};

pub use services::{
    AgentVoicePreset, AuthSettings, KokoroSettings, LiveKitSettings, OntologyAgentSettings,
    OpenAISettings, PerplexitySettings, RagFlowSettings, TurboWhisperSettings,
    VoiceRoutingSettings, WhisperSettings,
};

pub use app_settings::{
    AppFullSettings, DeveloperConfig, FeatureFlags, UserPreferences,
};

// Re-export path_access trait
pub use path_access::{JsonPathAccessible, PathAccessible};
