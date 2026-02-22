//! Settings-domain messages: CRUD for AppFullSettings, path-based access,
//! priority updates, and physics auto-balance propagation.

use actix::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

use crate::config::AppFullSettings;
use crate::errors::VisionFlowError;

// ---------------------------------------------------------------------------
// Settings Actor Messages
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "Result<AppFullSettings, VisionFlowError>")]
pub struct GetSettings;

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct UpdateSettings {
    pub settings: AppFullSettings,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ReloadSettings;

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct MergeSettingsUpdate {
    pub update: serde_json::Value,
}

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct PartialSettingsUpdate {
    pub partial_settings: serde_json::Value,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdatePhysicsFromAutoBalance {
    pub physics_update: serde_json::Value,
}

#[derive(Message)]
#[rtype(result = "Result<Value, VisionFlowError>")]
pub struct GetSettingByPath {
    pub path: String,
}

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct SetSettingByPath {
    pub path: String,
    pub value: Value,
}

// Batch path-based settings messages for performance
#[derive(Message)]
#[rtype(result = "Result<HashMap<String, Value>, VisionFlowError>")]
pub struct GetSettingsByPaths {
    pub paths: Vec<String>,
}

#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct SetSettingsByPaths {
    pub updates: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Priority-based updates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UpdatePriority {
    Critical = 1,
    High = 2,
    Normal = 3,
    Low = 4,
}

impl PartialOrd for UpdatePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UpdatePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PriorityUpdate {
    pub path: String,
    pub value: Value,
    pub priority: UpdatePriority,
    pub timestamp: std::time::Instant,
    pub client_id: Option<String>,
}

impl Eq for PriorityUpdate {}

impl PartialOrd for PriorityUpdate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityUpdate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => self.timestamp.cmp(&other.timestamp),
            other => other,
        }
    }
}

impl PriorityUpdate {
    pub fn new(path: String, value: Value) -> Self {
        let priority = Self::determine_priority(&path);
        Self {
            path,
            value,
            priority,
            timestamp: std::time::Instant::now(),
            client_id: None,
        }
    }

    pub fn with_client_id(mut self, client_id: String) -> Self {
        self.client_id = Some(client_id);
        self
    }

    fn determine_priority(path: &str) -> UpdatePriority {
        if path.contains(".physics.") {
            UpdatePriority::Critical
        } else if path.contains(".bloom.") || path.contains(".glow.") || path.contains(".visual") {
            UpdatePriority::High
        } else if path.contains(".system.") || path.contains(".security.") {
            UpdatePriority::Normal
        } else {
            UpdatePriority::Low
        }
    }
}
