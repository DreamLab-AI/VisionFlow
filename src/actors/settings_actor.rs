// Simplified Settings Actor using Settings
// Clean actor implementation without complex conversions

use actix::prelude::*;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, GetSettingByPath};
use log::{info, error, debug};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct SettingsActor {
    settings: Arc<RwLock<AppFullSettings>>,
}

impl SettingsActor {
    pub fn new() -> Self {
        // Load settings from file or use defaults
        let settings = AppFullSettings::new()
            .unwrap_or_else(|e| {
                error!("Failed to load settings from file: {}", e);
                panic!("Failed to create AppFullSettings: {}", e)
            });
        
        info!("Settings actor initialized with configuration");
        debug!("Logseq physics: damping={}, spring={}, repulsion={}", 
            settings.visualisation.graphs.logseq.physics.damping,
            settings.visualisation.graphs.logseq.physics.spring_k,
            settings.visualisation.graphs.logseq.physics.repel_k
        );
        
        Self {
            settings: Arc::new(RwLock::new(settings)),
        }
    }
    
    pub async fn get_settings(&self) -> AppFullSettings {
        self.settings.read().await.clone()
    }
    
    pub async fn update_settings(&self, new_settings: AppFullSettings) -> Result<(), String> {
        let mut settings = self.settings.write().await;
        *settings = new_settings;
        
        // Persist to file
        if let Err(e) = settings.save() {
            error!("Failed to save settings to file: {}", e);
            return Err(format!("Failed to persist settings: {}", e));
        }
        
        info!("Settings updated and saved successfully");
        Ok(())
    }
}

impl Actor for SettingsActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("SettingsActor started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("SettingsActor stopped");
    }
}

// Handle GetSettings message
impl Handler<GetSettings> for SettingsActor {
    type Result = ResponseFuture<Result<AppFullSettings, String>>;
    
    fn handle(&mut self, _msg: GetSettings, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        
        Box::pin(async move {
            Ok(settings.read().await.clone())
        })
    }
}

// Handle UpdateSettings message  
impl Handler<UpdateSettings> for SettingsActor {
    type Result = ResponseFuture<Result<(), String>>;
    
    fn handle(&mut self, msg: UpdateSettings, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        
        Box::pin(async move {
            let mut current = settings.write().await;
            *current = msg.settings;
            
            // Save to file
            if let Err(e) = current.save() {
                error!("Failed to save settings: {}", e);
                Err(format!("Failed to save settings: {}", e))
            } else {
                info!("Settings updated successfully");
                Ok(())
            }
        })
    }
}

// Handler for getting settings by path (for socket_flow_handler compatibility)
impl Handler<GetSettingByPath> for SettingsActor {
    type Result = ResponseFuture<Result<serde_json::Value, String>>;
    
    fn handle(&mut self, msg: GetSettingByPath, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let path = msg.path.clone();
        
        Box::pin(async move {
            let current = settings.read().await;
            
            // Convert settings to JSON for path traversal
            let json = serde_json::to_value(&*current)
                .map_err(|e| format!("Failed to serialize settings: {}", e))?;
            
            // Navigate the path
            let parts: Vec<&str> = path.split('.').collect();
            let mut value = &json;
            
            for part in parts {
                match value.get(part) {
                    Some(v) => value = v,
                    None => return Err(format!("Path not found: {}", path)),
                }
            }
            
            Ok(value.clone())
        })
    }
}

