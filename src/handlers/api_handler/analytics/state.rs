use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::types::{AnomalyState, ClusteringTask, FeatureFlags};

// Global state for clustering operations
pub static CLUSTERING_TASKS: Lazy<Arc<Mutex<HashMap<String, ClusteringTask>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

pub static ANOMALY_STATE: Lazy<Arc<Mutex<AnomalyState>>> =
    Lazy::new(|| Arc::new(Mutex::new(AnomalyState::default())));

pub static FEATURE_FLAGS: Lazy<Arc<Mutex<FeatureFlags>>> =
    Lazy::new(|| Arc::new(Mutex::new(FeatureFlags::default())));
