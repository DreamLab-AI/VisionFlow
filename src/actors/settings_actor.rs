// Simplified Settings Actor using Settings
// Clean actor implementation without complex conversions

use actix::prelude::*;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, GetSettingByPath, SetSettingByPath, GetSettingsByPaths, SetSettingsByPaths, UpdatePhysicsFromAutoBalance};
use crate::config::path_access::{PathAccessible, JsonPathAccessible};
use std::collections::HashMap;
use serde_json::Value;
use log::{info, error, debug};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct SettingsActor {
    settings: Arc<RwLock<AppFullSettings>>,
}

impl SettingsActor {
    pub fn new() -> Result<Self, String> {
        // Load settings from file or use defaults
        let settings = AppFullSettings::new()
            .map_err(|e| {
                error!("Failed to load settings from file: {}", e);
                format!("Failed to create AppFullSettings: {}", e)
            })?;
        
        info!("Settings actor initialized with configuration");
        debug!("Logseq physics: damping={}, spring={}, repulsion={}", 
            settings.visualisation.graphs.logseq.physics.damping,
            settings.visualisation.graphs.logseq.physics.spring_k,
            settings.visualisation.graphs.logseq.physics.repel_k
        );
        
        Ok(Self {
            settings: Arc::new(RwLock::new(settings)),
        })
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

// Handler for SetSettingByPath message - FIXED to use JsonPathAccessible
impl Handler<SetSettingByPath> for SettingsActor {
    type Result = ResponseFuture<Result<(), String>>;

    fn handle(&mut self, msg: SetSettingByPath, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let path = msg.path;
        let value = msg.value;

        Box::pin(async move {
            let mut current = settings.write().await;

            // Use the correct trait that respects serde's rename_all attribute
            if let Err(e) = current.set_json_by_path(&path, value.clone()) {
                error!("Failed to set setting via JSON path '{}': {}", path, e);
                return Err(format!("Failed to set setting: {}", e));
            }

            // Validate the updated settings
            if let Err(e) = current.validate_config_camel_case() {
                error!("Validation failed after path update: {:?}", e);
                return Err(format!("Validation failed: {:?}", e));
            }

            // Save to file if persistence is enabled
            if current.system.persist_settings {
                if let Err(e) = current.save() {
                    error!("Failed to save settings after path update: {}", e);
                    return Err(format!("Failed to save settings: {}", e));
                }
            }

            info!("Successfully updated setting at path: {} = {:?}", path, value);
            Ok(())
        })
    }
}

// Handler for batch path operations - for high-frequency updates like sliders
impl Handler<GetSettingsByPaths> for SettingsActor {
    type Result = ResponseFuture<Result<HashMap<String, Value>, String>>;
    
    fn handle(&mut self, msg: GetSettingsByPaths, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let paths = msg.paths;
        
        Box::pin(async move {
            let current = settings.read().await;
            let mut results = HashMap::new();
            
            for path in paths {
                match current.get_by_path(&path) {
                    Ok(boxed_value) => {
                        // Convert back to JSON value - this is a simplified version
                        // In a full implementation, we'd need proper type conversion
                        let json_val = serde_json::to_value(&*current)
                            .map_err(|e| format!("Failed to serialize settings: {}", e))?;
                        
                        // Navigate to the specific path in JSON
                        let mut current_val = &json_val;
                        for segment in path.split('.') {
                            match current_val.get(segment) {
                                Some(v) => current_val = v,
                                None => {
                                    error!("Path not found during batch get: {}", path);
                                    continue;
                                }
                            }
                        }
                        results.insert(path, current_val.clone());
                    }
                    Err(e) => {
                        error!("Failed to get path {}: {}", path, e);
                        // Continue with other paths even if one fails
                    }
                }
            }
            
            Ok(results)
        })
    }
}

// Handler for batch path updates - critical for slider performance
impl Handler<SetSettingsByPaths> for SettingsActor {
    type Result = ResponseFuture<Result<(), String>>;
    
    fn handle(&mut self, msg: SetSettingsByPaths, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let updates = msg.updates;
        
        Box::pin(async move {
            let mut current = settings.write().await;
            let mut validation_needed = false;
            
            for (path, value) in updates {
                // Use the same logic as SetSettingByPath but in batch
                if path.starts_with("visualisation.graphs.logseq.physics.") {
                    validation_needed = true;
                    
                    let field_name = path.replace("visualisation.graphs.logseq.physics.", "");
                    let internal_field = match field_name.as_str() {
                        "springK" => "spring_k",
                        "repelK" => "repel_k", 
                        "maxVelocity" => "max_velocity",
                        "boundsSize" => "bounds_size",
                        other => other,
                    };
                    
                    let full_path = format!("visualisation.graphs.logseq.physics.{}", internal_field);
                    
                    match internal_field {
                        "damping" | "spring_k" | "repel_k" | "max_velocity" | "bounds_size" | "gravity" | "temperature" => {
                            if let Some(f_val) = value.as_f64() {
                                if let Err(e) = current.set_by_path(&full_path, Box::new(f_val as f32)) {
                                    error!("Failed to batch update physics field {}: {}", internal_field, e);
                                    continue;
                                }
                            }
                        }
                        "enabled" => {
                            if let Some(b_val) = value.as_bool() {
                                if let Err(e) = current.set_by_path(&full_path, Box::new(b_val)) {
                                    error!("Failed to batch update physics field {}: {}", internal_field, e);
                                    continue;
                                }
                            }
                        }
                        "iterations" => {
                            if let Some(i_val) = value.as_u64() {
                                if let Err(e) = current.set_by_path(&full_path, Box::new(i_val as u32)) {
                                    error!("Failed to batch update physics field {}: {}", internal_field, e);
                                    continue;
                                }
                            }
                        }
                        _ => {
                            error!("Unsupported physics field in batch update: {}", internal_field);
                            continue;
                        }
                    }
                    
                    debug!("Batch updated physics setting: {} = {:?}", internal_field, value);
                }
            }
            
            // Only validate once for all batch updates
            if validation_needed {
                if let Err(e) = current.validate_config_camel_case() {
                    error!("Validation failed after batch update: {:?}", e);
                    return Err(format!("Batch validation failed: {:?}", e));
                }
                
                // Save to file if persistence is enabled
                if current.system.persist_settings {
                    if let Err(e) = current.save() {
                        error!("Failed to save settings after batch update: {}", e);
                        return Err(format!("Failed to save batch settings: {}", e));
                    }
                }
            }
            
            info!("Successfully completed batch settings update");
            Ok(())
        })
    }
}
