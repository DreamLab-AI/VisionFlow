// src/settings/models.rs
//! Settings data models

use serde::{Deserialize, Serialize};
use crate::config::{PhysicsSettings, RenderingSettings};

///
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum PriorityWeighting {
    
    Linear,
    
    Exponential,
    
    Quadratic,
}

impl Default for PriorityWeighting {
    fn default() -> Self {
        Self::Exponential
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintSettings {
    
    pub lod_enabled: bool,

    
    pub far_threshold: f32,

    
    pub medium_threshold: f32,

    
    pub near_threshold: f32,

    
    pub priority_weighting: PriorityWeighting,

    
    pub progressive_activation: bool,

    
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

///
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

///
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsProfile {
    pub id: i64,
    pub name: String,
    pub created_at: String,
    pub updated_at: String,
}
