// src/application/settings/mod.rs
//! Settings Domain Application Layer
//!
//! Contains all directives (write operations) and queries (read operations)
//! for settings management following CQRS patterns.

pub mod directives;
pub mod queries;

// Re-export directives
pub use directives::{
    ClearSettingsCache, ClearSettingsCacheHandler, DeletePhysicsProfile,
    DeletePhysicsProfileHandler, SaveAllSettings, SaveAllSettingsHandler, UpdatePhysicsSettings,
    UpdatePhysicsSettingsHandler, UpdateSetting, UpdateSettingHandler, UpdateSettingsBatch,
    UpdateSettingsBatchHandler,
};

// Re-export queries
pub use queries::{
    GetPhysicsSettings, GetPhysicsSettingsHandler, GetSetting, GetSettingHandler, GetSettingsBatch,
    GetSettingsBatchHandler, ListPhysicsProfiles, ListPhysicsProfilesHandler, LoadAllSettings,
    LoadAllSettingsHandler,
};
