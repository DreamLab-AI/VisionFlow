// Simplified Settings Actor using UnifiedSettings
// Clean actor implementation without complex conversions

use actix::prelude::*;
use crate::config::unified::UnifiedSettings;
use crate::actors::messages::{GetSettings, UpdateSettings};
use log::{info, error, debug};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct UnifiedSettingsActor {
    settings: Arc<RwLock<UnifiedSettings>>,
}

impl UnifiedSettingsActor {
    pub fn new() -> Self {
        // Load settings from file or use defaults
        let settings = UnifiedSettings::load()
            .unwrap_or_else(|e| {
                error!("Failed to load settings from file: {}. Using defaults.", e);
                UnifiedSettings::default()
            });
        
        info!("Settings actor initialized with configuration");
        debug!("Logseq physics: damping={}, spring={}, repulsion={}", 
            settings.graphs.logseq.physics.damping,
            settings.graphs.logseq.physics.spring_strength,
            settings.graphs.logseq.physics.repulsion_strength
        );
        
        Self {
            settings: Arc::new(RwLock::new(settings)),
        }
    }
    
    pub async fn get_settings(&self) -> UnifiedSettings {
        self.settings.read().await.clone()
    }
    
    pub async fn update_settings(&self, new_settings: UnifiedSettings) -> Result<(), String> {
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

impl Actor for UnifiedSettingsActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("UnifiedSettingsActor started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("UnifiedSettingsActor stopped");
    }
}

// Handle GetSettings message
impl Handler<GetSettings> for UnifiedSettingsActor {
    type Result = ResponseFuture<Result<UnifiedSettings, String>>;
    
    fn handle(&mut self, _msg: GetSettings, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        
        Box::pin(async move {
            Ok(settings.read().await.clone())
        })
    }
}

// Handle UpdateSettings message  
impl Handler<UpdateSettings<UnifiedSettings>> for UnifiedSettingsActor {
    type Result = ResponseFuture<Result<(), String>>;
    
    fn handle(&mut self, msg: UpdateSettings<UnifiedSettings>, _ctx: &mut Self::Context) -> Self::Result {
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

// For backward compatibility during migration
use crate::config::AppFullSettings;

impl Handler<UpdateSettings<AppFullSettings>> for UnifiedSettingsActor {
    type Result = ResponseFuture<Result<(), String>>;
    
    fn handle(&mut self, msg: UpdateSettings<AppFullSettings>, _ctx: &mut Self::Context) -> Self::Result {
        // Convert old format to new
        let unified: UnifiedSettings = msg.settings.into();
        let settings = self.settings.clone();
        
        Box::pin(async move {
            let mut current = settings.write().await;
            *current = unified;
            
            if let Err(e) = current.save() {
                error!("Failed to save migrated settings: {}", e);
                Err(format!("Failed to save settings: {}", e))
            } else {
                info!("Settings migrated and saved successfully");
                Ok(())
            }
        })
    }
}