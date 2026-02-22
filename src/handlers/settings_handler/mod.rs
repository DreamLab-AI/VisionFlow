// Unified Settings Handler - Single source of truth: AppFullSettings
// Split into submodules for maintainability.

pub mod types;
pub mod conversions;
pub mod enhanced;
pub mod routes;
pub mod write_handlers;
pub mod validation;
pub mod physics;
pub mod helpers;

// Re-export all public types from types module
pub use types::{
    value_type_name,
    SettingsResponseDTO,
    SettingsValidationError,
    SettingsUpdateDTO,
    VisualisationSettingsDTO,
    RenderingSettingsDTO,
    AgentColorsDTO,
    AnimationSettingsDTO,
    GlowSettingsDTO,
    HologramSettingsDTO,
    GraphsSettingsDTO,
    GraphSettingsDTO,
    NodeSettingsDTO,
    EdgeSettingsDTO,
    LabelSettingsDTO,
    PhysicsSettingsDTO,
    AutoBalanceConfigDTO,
    CameraSettingsDTO,
    PositionDTO,
    SpacePilotSettingsDTO,
    SensitivityDTO,
    SystemSettingsDTO,
    NetworkSettingsDTO,
    WebSocketSettingsDTO,
    SecuritySettingsDTO,
    DebugSettingsDTO,
    XRSettingsDTO,
    MovementAxesDTO,
    AuthSettingsDTO,
    RagFlowSettingsDTO,
    PerplexitySettingsDTO,
    OpenAISettingsDTO,
    KokoroSettingsDTO,
    WhisperSettingsDTO,
};

// Re-export enhanced handler
pub use enhanced::EnhancedSettingsHandler;

// Re-export route configuration
pub use routes::config;

// Re-export batch operations (used externally)
pub use write_handlers::{batch_get_settings, batch_update_settings};

// Re-export physics propagation (used by other handlers)
pub use physics::propagate_physics_to_gpu;

// Re-export helpers (used externally)
pub use helpers::{
    get_field_variant,
    count_fields,
    extract_physics_updates,
    extract_failed_field,
    create_physics_settings_update,
};

// Re-export validation functions (used externally)
pub use validation::{validate_settings_update, validate_xr_settings, validate_constraints};
