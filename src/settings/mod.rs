// src/settings/mod.rs
//! Settings Management Module
//!
//! Provides persistent settings management for the control center including:
//! - Database persistence layer (settings_repository)
//! - Runtime settings actor (settings_actor)
//! - REST API endpoints (api/settings_routes)

pub mod settings_actor;
pub mod api;
pub mod models;

pub use settings_actor::{SettingsActor, UpdatePhysicsSettings, GetPhysicsSettings, LoadProfile, SaveProfile};
pub use models::{ConstraintSettings, PriorityWeighting, AllSettings, SettingsProfile};
