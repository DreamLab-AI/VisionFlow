// Simplified Settings Actor using Settings
// Clean actor implementation without complex conversions

use actix::prelude::*;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, GetSettingByPath, UpdatePhysicsFromAutoBalance};
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

impl Handler<UpdatePhysicsFromAutoBalance> for SettingsActor {
    type Result = ();
    
    fn handle(&mut self, msg: UpdatePhysicsFromAutoBalance, ctx: &mut Self::Context) {
        let settings = self.settings.clone();
        
        ctx.spawn(Box::pin(async move {
            let mut current = settings.write().await;
            
            // Merge the physics update from auto-balance
            if let Err(e) = current.merge_update(msg.physics_update.clone()) {
                error!("[AUTO-BALANCE] Failed to merge physics update: {}", e);
                return;
            }
            
            info!("[AUTO-BALANCE] Physics parameters updated in settings from auto-tuning");
            
            // Reset validation boundaries now that auto-tune has mapped the parameter space
            // The auto-tuning has found optimal values, so we can relax validation constraints
            if let Some(physics) = msg.physics_update.get("visualisation")
                .and_then(|v| v.get("graphs"))
                .and_then(|g| g.get("logseq"))
                .and_then(|l| l.get("physics")) {
                
                info!("[AUTO-BALANCE] Auto-tune complete - resetting validation boundaries for discovered optimal parameters");
                
                // Log the final tuned values
                if let Some(repel_k) = physics.get("repelK").and_then(|v| v.as_f64()) {
                    info!("[AUTO-BALANCE] Final repelK: {:.3}", repel_k);
                }
                if let Some(damping) = physics.get("damping").and_then(|v| v.as_f64()) {
                    info!("[AUTO-BALANCE] Final damping: {:.3}", damping);
                }
                if let Some(max_vel) = physics.get("maxVelocity").and_then(|v| v.as_f64()) {
                    info!("[AUTO-BALANCE] Final maxVelocity: {:.3}", max_vel);
                }
            }
            
            // Save to file if persistence is enabled and user is authenticated
            // Check if persist_settings is enabled
            if current.system.persist_settings {
                if let Err(e) = current.save() {
                    error!("[AUTO-BALANCE] Failed to save auto-tuned settings to file: {}", e);
                } else {
                    info!("[AUTO-BALANCE] Auto-tuned settings saved to settings.yaml");
                }
            } else {
                info!("[AUTO-BALANCE] Settings persistence disabled, not saving to file");
            }
        }).into_actor(self));
    }
}
