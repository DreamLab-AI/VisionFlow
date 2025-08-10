use std::sync::Arc;
use tokio::sync::RwLock;
use crate::config::AppFullSettings;

#[derive(Debug)]  // Only Debug derive, remove Clone
pub struct AppState {
    pub settings: Arc<RwLock<AppFullSettings>>,
}

impl AppState {
    pub fn new(settings: AppFullSettings) -> Self {
        Self {
            settings: Arc::new(RwLock::new(settings)),
        }
    }

    pub fn clone_settings(&self) -> Arc<RwLock<AppFullSettings>> {
        self.settings.clone()
    }
} 