// src/settings/models.rs
//! Settings data models

use serde::{Deserialize, Serialize};
use crate::config::{PhysicsSettings, RenderingSettings};

/// Priority weighting strategy for constraint LOD
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum PriorityWeighting {
    /// Linear weighting (priority 1 = 1x, priority 2 = 2x, etc.)
    Linear,
    /// Exponential weighting (priority 1 = 1x, priority 2 = 2x, priority 3 = 4x, etc.)
    Exponential,
    /// Quadratic weighting (priority 1 = 1x, priority 2 = 4x, priority 3 = 9x, etc.)
    Quadratic,
}

impl Default for PriorityWeighting {
    fn default() -> Self {
        Self::Exponential
    }
}

/// Constraint Level-of-Detail settings for progressive constraint activation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintSettings {
    /// Enable LOD-based constraint culling
    pub lod_enabled: bool,

    /// Distance threshold for far view (only priority 1-3 constraints)
    pub far_threshold: f32,

    /// Distance threshold for medium view (priority 1-5 constraints)
    pub medium_threshold: f32,

    /// Distance threshold for near view (all constraints)
    pub near_threshold: f32,

    /// Priority weighting strategy
    pub priority_weighting: PriorityWeighting,

    /// Enable progressive constraint activation (ramp up over time)
    pub progressive_activation: bool,

    /// Number of frames to fully activate constraints
    pub activation_frames: u32,
}

impl Default for ConstraintSettings {
    fn default() -> Self {
        Self {
            lod_enabled: true,
            far_threshold: 1000.0,
            medium_threshold: 100.0,
            near_threshold: 10.0,
            priority_weighting: PriorityWeighting::Exponential,
            progressive_activation: true,
            activation_frames: 60,
        }
    }
}

/// Combined settings container for profile management
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AllSettings {
    pub physics: PhysicsSettings,
    pub constraints: ConstraintSettings,
    pub rendering: RenderingSettings,
}

impl Default for AllSettings {
    fn default() -> Self {
        Self {
            physics: PhysicsSettings::default(),
            constraints: ConstraintSettings::default(),
            rendering: RenderingSettings::default(),
        }
    }
}

/// Settings profile metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsProfile {
    pub id: i64,
    pub name: String,
    pub created_at: String,
    pub updated_at: String,
}
