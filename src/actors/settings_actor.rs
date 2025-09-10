// Simplified Settings Actor using Settings
// Clean actor implementation without complex conversions

use actix::prelude::*;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, GetSettingByPath, SetSettingByPath, GetSettingsByPaths, SetSettingsByPaths, UpdatePhysicsFromAutoBalance, UpdateSimulationParams, BatchedUpdate, PriorityUpdate, UpdatePriority};
use crate::actors::{GraphServiceActor, GPUComputeActor};
use crate::config::path_access::{PathAccessible, JsonPathAccessible};
use crate::errors::{VisionFlowError, VisionFlowResult, SettingsError};
use std::collections::{HashMap, BinaryHeap};
use serde_json::Value;
use log::{info, error, debug, warn};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Instant, Duration};
use std::cmp::Reverse;

// Batching configuration constants
const MAX_BATCH_SIZE: usize = 50;
const BATCH_TIMEOUT_MS: u64 = 100;
const MAX_MAILBOX_SIZE: usize = 1000;

// Mailbox monitoring and overflow protection
#[derive(Debug, Clone)]
pub struct MailboxMetrics {
    pub pending_messages: usize,
    pub batched_updates: usize,
    pub priority_queue_size: usize,
    pub overflow_count: u64,
    pub last_batch_time: Instant,
}

impl Default for MailboxMetrics {
    fn default() -> Self {
        Self {
            pending_messages: 0,
            batched_updates: 0,
            priority_queue_size: 0,
            overflow_count: 0,
            last_batch_time: Instant::now(),
        }
    }
}

pub struct SettingsActor {
    settings: Arc<RwLock<AppFullSettings>>,
    graph_service_addr: Option<Addr<GraphServiceActor>>,
    gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    // Batching and concurrent update handling
    update_queue: BinaryHeap<Reverse<PriorityUpdate>>,
    batch_timer: Option<actix::SpawnHandle>,
    metrics: MailboxMetrics,
    last_batch_process: Instant,
}

impl SettingsActor {
    pub fn new() -> VisionFlowResult<Self> {
        // Load settings from file or use defaults
        let settings = AppFullSettings::new()
            .map_err(|e| {
                error!("Failed to load settings from file: {}", e);
                VisionFlowError::Settings(SettingsError::ParseError {
                    file_path: "settings".to_string(),
                    reason: e.to_string(),
                })
            })?;
        
        info!("Settings actor initialized with configuration");
        debug!("Logseq physics: damping={}, spring={}, repulsion={}", 
            settings.visualisation.graphs.logseq.physics.damping,
            settings.visualisation.graphs.logseq.physics.spring_k,
            settings.visualisation.graphs.logseq.physics.repel_k
        );
        
        Ok(Self {
            settings: Arc::new(RwLock::new(settings)),
            graph_service_addr: None,
            gpu_compute_addr: None,
            update_queue: BinaryHeap::new(),
            batch_timer: None,
            metrics: MailboxMetrics::default(),
            last_batch_process: Instant::now(),
        })
    }
    
    pub fn with_actors(
        graph_service_addr: Option<Addr<GraphServiceActor>>,
        gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    ) -> VisionFlowResult<Self> {
        let mut actor = Self::new()?;
        actor.graph_service_addr = graph_service_addr;
        actor.gpu_compute_addr = gpu_compute_addr;
        info!("SettingsActor initialized with GPU and Graph actor addresses for physics forwarding and concurrent update batching");
        Ok(actor)
    }
    
    pub async fn get_settings(&self) -> AppFullSettings {
        self.settings.read().await.clone()
    }
    
    pub async fn update_settings(&self, new_settings: AppFullSettings) -> VisionFlowResult<()> {
        let mut settings = self.settings.write().await;
        *settings = new_settings;
        
        // Persist to file
        settings.save().map_err(|e| {
            error!("Failed to save settings to file: {}", e);
            VisionFlowError::Settings(SettingsError::SaveFailed {
                file_path: "settings".to_string(),
                reason: e.to_string(),
            })
        })?;
        
        // Propagate physics updates after settings update
        self.propagate_physics_updates(&settings, "logseq").await;
        
        info!("Settings updated, saved, and physics parameters propagated successfully");
        Ok(())
    }
    
    /// Check if mailbox is approaching overflow and take protective action
    fn check_mailbox_overflow(&mut self, ctx: &mut Context<Self>) -> bool {
        if self.update_queue.len() >= MAX_MAILBOX_SIZE {
            warn!(
                "[CONCURRENT UPDATES] Mailbox overflow detected! Queue size: {}, processing emergency batch",
                self.update_queue.len()
            );
            self.metrics.overflow_count += 1;
            
            // Emergency batch processing - process critical updates first
            let critical_updates = self.extract_priority_updates(UpdatePriority::Critical, 25);
            if !critical_updates.is_empty() {
                info!("[CONCURRENT UPDATES] Emergency processing {} critical updates", critical_updates.len());
                ctx.spawn(Self::process_emergency_batch(self.settings.clone(), critical_updates, 
                    self.graph_service_addr.clone(), self.gpu_compute_addr.clone()).into_actor(self));
            }
            
            // Drop low priority updates to make room
            let dropped_count = self.drop_low_priority_updates(50);
            warn!("[CONCURRENT UPDATES] Dropped {} low-priority updates to prevent mailbox overflow", dropped_count);
            
            true
        } else {
            false
        }
    }
    
    /// Extract updates of specific priority from the queue
    fn extract_priority_updates(&mut self, target_priority: UpdatePriority, max_count: usize) -> Vec<PriorityUpdate> {
        let mut extracted = Vec::with_capacity(max_count);
        let mut remaining = BinaryHeap::new();
        
        while let Some(Reverse(update)) = self.update_queue.pop() {
            if update.priority == target_priority && extracted.len() < max_count {
                extracted.push(update);
            } else {
                remaining.push(Reverse(update));
            }
        }
        
        self.update_queue = remaining;
        extracted
    }
    
    /// Drop low priority updates from queue to prevent overflow
    fn drop_low_priority_updates(&mut self, max_drops: usize) -> usize {
        let mut kept = BinaryHeap::new();
        let mut dropped = 0;
        
        while let Some(Reverse(update)) = self.update_queue.pop() {
            if update.priority == UpdatePriority::Low && dropped < max_drops {
                dropped += 1;
            } else {
                kept.push(Reverse(update));
            }
        }
        
        self.update_queue = kept;
        dropped
    }
    
    /// Schedule batch processing with timeout
    fn schedule_batch_processing(&mut self, ctx: &mut Context<Self>) {
        if self.batch_timer.is_none() && !self.update_queue.is_empty() {
            let handle = ctx.run_later(Duration::from_millis(BATCH_TIMEOUT_MS), |actor, ctx| {
                actor.process_pending_batches(ctx);
            });
            self.batch_timer = Some(handle);
        }
    }
    
    /// Process pending batches from the queue
    fn process_pending_batches(&mut self, ctx: &mut Context<Self>) {
        self.batch_timer = None;
        let now = Instant::now();
        
        if self.update_queue.is_empty() {
            return;
        }
        
        info!("[CONCURRENT UPDATES] Processing pending batches, queue size: {}", self.update_queue.len());
        
        // Extract updates for processing, prioritizing by priority level
        let mut batch_updates = Vec::with_capacity(MAX_BATCH_SIZE);
        while batch_updates.len() < MAX_BATCH_SIZE && !self.update_queue.is_empty() {
            if let Some(Reverse(update)) = self.update_queue.pop() {
                batch_updates.push(update);
            }
        }
        
        if !batch_updates.is_empty() {
            self.metrics.batched_updates += batch_updates.len();
            self.metrics.last_batch_time = now;
            self.last_batch_process = now;
            
            info!(
                "[CONCURRENT UPDATES] Processing batch of {} updates, priority breakdown: Critical: {}, High: {}, Normal: {}, Low: {}",
                batch_updates.len(),
                batch_updates.iter().filter(|u| u.priority == UpdatePriority::Critical).count(),
                batch_updates.iter().filter(|u| u.priority == UpdatePriority::High).count(),
                batch_updates.iter().filter(|u| u.priority == UpdatePriority::Normal).count(),
                batch_updates.iter().filter(|u| u.priority == UpdatePriority::Low).count(),
            );
            
            ctx.spawn(Self::process_priority_batch(self.settings.clone(), batch_updates, 
                self.graph_service_addr.clone(), self.gpu_compute_addr.clone()).into_actor(self));
        }
        
        // Schedule next batch if queue still has items
        if !self.update_queue.is_empty() {
            self.schedule_batch_processing(ctx);
        }
        
        // Update metrics
        self.metrics.priority_queue_size = self.update_queue.len();
    }
    
    /// Process emergency batch for critical updates during overflow
    async fn process_emergency_batch(
        settings: Arc<RwLock<AppFullSettings>>,
        updates: Vec<PriorityUpdate>,
        graph_service_addr: Option<Addr<GraphServiceActor>>,
        gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    ) {
        let start = Instant::now();
        let mut current = settings.write().await;
        let mut physics_updated = false;
        
        for update in updates {
            // Process only critical updates (physics parameters)
            if let Err(e) = current.set_json_by_path(&update.path, update.value.clone()) {
                error!("[EMERGENCY BATCH] Failed to set critical path {}: {}", update.path, e);
                continue;
            }
            
            if update.path.contains(".physics.") {
                physics_updated = true;
            }
        }
        
        // Forward physics updates immediately
        if physics_updated {
            let physics = current.get_physics("logseq");
            let sim_params = crate::models::simulation_params::SimulationParams::from(physics);
            let update_msg = UpdateSimulationParams { params: sim_params };
            
            if let Some(graph_addr) = &graph_service_addr {
                graph_addr.do_send(update_msg.clone());
            }
            if let Some(gpu_addr) = &gpu_compute_addr {
                gpu_addr.do_send(update_msg);
            }
        }
        
        // Skip validation and persistence during emergency batch for performance
        info!(
            "[EMERGENCY BATCH] Completed emergency processing in {:?}", 
            start.elapsed()
        );
    }
    
    /// Process priority batch with full validation
    async fn process_priority_batch(
        settings: Arc<RwLock<AppFullSettings>>,
        mut updates: Vec<PriorityUpdate>,
        graph_service_addr: Option<Addr<GraphServiceActor>>,
        gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    ) {
        let start = Instant::now();
        let mut current = settings.write().await;
        
        // Sort by priority to ensure critical updates are processed first
        updates.sort();
        
        let mut physics_updated = false;
        let mut bloom_glow_updated = false;
        let mut validation_needed = false;
        
        for update in updates {
            if let Err(e) = current.set_json_by_path(&update.path, update.value.clone()) {
                error!("[PRIORITY BATCH] Failed to set path {}: {}", update.path, e);
                continue;
            }
            
            if update.path.contains(".physics.") {
                physics_updated = true;
                validation_needed = true;
            } else if update.path.contains(".bloom.") || update.path.contains(".glow.") {
                bloom_glow_updated = true;
                validation_needed = true;
            }
        }
        
        // Validate if needed
        if validation_needed {
            if let Err(e) = current.validate_config_camel_case() {
                error!("[PRIORITY BATCH] Validation failed: {:?}", e);
                return;
            }
        }
        
        // Forward physics updates
        if physics_updated {
            let physics = current.get_physics("logseq");
            let sim_params = crate::models::simulation_params::SimulationParams::from(physics);
            let update_msg = UpdateSimulationParams { params: sim_params };
            
            if let Some(graph_addr) = &graph_service_addr {
                graph_addr.do_send(update_msg.clone());
                info!("[PRIORITY BATCH] Physics updates forwarded to GraphServiceActor");
            }
            if let Some(gpu_addr) = &gpu_compute_addr {
                gpu_addr.do_send(update_msg);
                info!("[PRIORITY BATCH] Physics updates forwarded to GPUComputeActor");
            }
        }
        
        // Persist settings if needed
        if current.system.persist_settings {
            if let Err(e) = current.save() {
                error!("[PRIORITY BATCH] Failed to save settings: {}", e);
            }
        }
        
        info!(
            "[PRIORITY BATCH] Completed batch processing in {:?}, physics_updated: {}, bloom_glow_updated: {}",
            start.elapsed(), physics_updated, bloom_glow_updated
        );
    }
    
    /// Propagate physics updates to GraphServiceActor and GPUComputeActor
    async fn propagate_physics_updates(&self, settings: &AppFullSettings, graph_name: &str) {
        let physics = settings.get_physics(graph_name);
        
        // Always log critical physics values for debugging
        info!("[SETTINGS ACTOR] Propagating {} physics parameters:", graph_name);
        info!("  - repel_k: {:.3} (node repulsion)", physics.repel_k);
        info!("  - spring_k: {:.3} (edge tension)", physics.spring_k);
        info!("  - damping: {:.3} (velocity damping)", physics.damping);
        info!("  - max_velocity: {:.3} (speed limit)", physics.max_velocity);
        
        // Convert physics settings to simulation parameters
        let sim_params = crate::models::simulation_params::SimulationParams::from(physics);
        let update_msg = UpdateSimulationParams { params: sim_params };
        
        // Send to GraphServiceActor if available
        if let Some(graph_addr) = &self.graph_service_addr {
            graph_addr.do_send(update_msg.clone());
            info!("[SETTINGS ACTOR] Physics parameters sent to GraphServiceActor");
        } else {
            debug!("[SETTINGS ACTOR] GraphServiceActor not available for physics forwarding");
        }
        
        // Send to GPUComputeActor if available
        if let Some(gpu_addr) = &self.gpu_compute_addr {
            gpu_addr.do_send(update_msg);
            info!("[SETTINGS ACTOR] Physics parameters sent to GPUComputeActor");
        } else {
            debug!("[SETTINGS ACTOR] GPUComputeActor not available for physics forwarding");
        }
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
    type Result = ResponseFuture<VisionFlowResult<AppFullSettings>>;
    
    fn handle(&mut self, _msg: GetSettings, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        
        Box::pin(async move {
            Ok(settings.read().await.clone())
        })
    }
}

// Handle UpdateSettings message  
impl Handler<UpdateSettings> for SettingsActor {
    type Result = ResponseFuture<VisionFlowResult<()>>;
    
    fn handle(&mut self, msg: UpdateSettings, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        
        Box::pin(async move {
            let mut current = settings.write().await;
            *current = msg.settings;
            
            // Save to file
            current.save().map_err(|e| {
                error!("Failed to save settings: {}", e);
                VisionFlowError::Settings(SettingsError::SaveFailed {
                    file_path: "settings".to_string(),
                    reason: e,
                })
            })?;
            
            info!("Settings updated successfully");
            Ok(())
        })
    }
}

// Handler for getting settings by path (for socket_flow_handler compatibility)
impl Handler<GetSettingByPath> for SettingsActor {
    type Result = ResponseFuture<VisionFlowResult<serde_json::Value>>;
    
    fn handle(&mut self, msg: GetSettingByPath, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let path = msg.path.clone();
        
        Box::pin(async move {
            let current = settings.read().await;
            
            // Convert settings to JSON for path traversal
            let json = serde_json::to_value(&*current)
                .map_err(|e| VisionFlowError::Serialization(format!("Failed to serialize settings: {}", e)))?;
            
            // Navigate the path
            let parts: Vec<&str> = path.split('.').collect();
            let mut value = &json;
            
            for part in parts {
                match value.get(part) {
                    Some(v) => value = v,
                    None => return Err(VisionFlowError::Settings(SettingsError::ValidationFailed {
                        setting_path: path.clone(),
                        reason: "Path not found".to_string(),
                    })),
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
        let graph_service_addr = self.graph_service_addr.clone();
        let gpu_compute_addr = self.gpu_compute_addr.clone();
        
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
            
            // CRITICAL: Forward physics updates to GPU actors
            // This is the missing piece that caused the original issue
            let physics = current.get_physics("logseq");
            let sim_params = crate::models::simulation_params::SimulationParams::from(physics);
            let update_msg = UpdateSimulationParams { params: sim_params };
            
            // Send to GraphServiceActor
            if let Some(graph_addr) = &graph_service_addr {
                graph_addr.do_send(update_msg.clone());
                info!("[AUTO-BALANCE] Physics forwarded to GraphServiceActor");
            } else {
                error!("[AUTO-BALANCE] GraphServiceActor not available - GPU will not receive physics updates!");
            }
            
            // Send to GPUComputeActor
            if let Some(gpu_addr) = &gpu_compute_addr {
                gpu_addr.do_send(update_msg);
                info!("[AUTO-BALANCE] Physics forwarded to GPUComputeActor");
            } else {
                error!("[AUTO-BALANCE] GPUComputeActor not available - GPU will not receive physics updates!");
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

// Handler for SetSettingByPath message - Enhanced with concurrent update handling
impl Handler<SetSettingByPath> for SettingsActor {
    type Result = ResponseFuture<VisionFlowResult<()>>;

    fn handle(&mut self, msg: SetSettingByPath, ctx: &mut Self::Context) -> Self::Result {
        // Convert single update to priority update and route through batching system
        let priority_update = PriorityUpdate::new(msg.path.clone(), msg.value.clone());
        
        // For critical updates (physics), process immediately
        if priority_update.priority == UpdatePriority::Critical {
            info!("[CONCURRENT UPDATES] Critical update detected, processing immediately: {}", msg.path);
            // Process immediately for physics parameters to maintain responsiveness
            return self.handle_immediate_update(msg);
        } else {
            // Route through batching system for non-critical updates
            let batch_msg = BatchedUpdate::new(vec![priority_update]);
            return self.handle(batch_msg, ctx);
        }
    }
}

impl SettingsActor {
    /// Handle immediate update for critical settings (bypasses batching for responsiveness)
    fn handle_immediate_update(&self, msg: SetSettingByPath) -> ResponseFuture<VisionFlowResult<()>> {
        let settings = self.settings.clone();
        let path = msg.path.clone();
        let value = msg.value;
        let graph_service_addr = self.graph_service_addr.clone();
        let gpu_compute_addr = self.gpu_compute_addr.clone();

        Box::pin(async move {
            let mut current = settings.write().await;

            // Use the correct trait that respects serde's rename_all attribute
            current.set_json_by_path(&path, value.clone()).map_err(|e| {
                error!("Failed to set setting via JSON path '{}': {}", path, e);
                VisionFlowError::Settings(SettingsError::ValidationFailed {
                    setting_path: path.clone(),
                    reason: e,
                })
            })?;

            // Check if this is a physics update that needs forwarding
            let is_physics_update = path.starts_with("visualisation.graphs.") && 
                                  (path.contains(".physics.repelK") || 
                                   path.contains(".physics.springK") ||
                                   path.contains(".physics.damping") ||
                                   path.contains(".physics.maxVelocity") ||
                                   path.contains(".physics.gravity") ||
                                   path.contains(".physics.temperature") ||
                                   path.contains(".physics.dt") ||
                                   path.contains(".physics.timeStep") ||
                                   path.contains(".physics.attractionK"));

            // Check if this is a bloom/glow update that needs validation
            let is_bloom_glow_update = path.starts_with("visualisation.") &&
                                     (path.contains(".bloom.") ||
                                      path.contains(".glow."));

            // Validate the updated settings with special handling for bloom/glow errors
            current.validate_config_camel_case().map_err(|e| {
                if is_bloom_glow_update {
                    error!("Bloom/Glow validation failed after path update '{}': {:?}", path, e);
                    // Return detailed error message for bloom/glow validation failures
                    let error_details = if path.contains(".intensity") {
                        "Intensity must be between 0.0 and 10.0"
                    } else if path.contains(".radius") {
                        "Radius must be between 0.0 and 10.0" 
                    } else if path.contains(".threshold") {
                        "Threshold must be between 0.0 and 1.0"
                    } else if path.contains(".color") || path.contains(".tintColor") || path.contains(".baseColor") || path.contains(".emissionColor") {
                        "Color must be a valid hex color format (#RRGGBB or #RRGGBBAA)"
                    } else if path.contains(".strength") || path.contains(".opacity") {
                        "Strength/Opacity must be between 0.0 and 1.0"
                    } else {
                        "Invalid bloom/glow parameter value"
                    };
                    VisionFlowError::Settings(SettingsError::ValidationFailed {
                        setting_path: path.clone(),
                        reason: format!("Bloom/Glow validation failed: {}. This validation prevents GPU kernel crashes.", error_details),
                    })
                } else {
                    error!("General validation failed after path update: {:?}", e);
                    VisionFlowError::Settings(SettingsError::ValidationFailed {
                        setting_path: path.clone(),
                        reason: format!("Validation failed: {:?}", e),
                    })
                }
            })?;

            // Log bloom/glow updates for monitoring
            if is_bloom_glow_update {
                info!("[SETTINGS ACTOR] Bloom/Glow parameter updated via path: {} = {:?} (validation passed)", path, value);
            }

            // Forward physics updates to GPU actors if this is a physics parameter change
            if is_physics_update {
                info!("[SETTINGS ACTOR] Physics parameter updated via path: {} = {:?}", path, value);
                
                // Extract graph name from path (e.g. "visualisation.graphs.logseq.physics.repelK" -> "logseq")
                let graph_name = if path.contains(".logseq.") {
                    "logseq"
                } else if path.contains(".visionflow.") {
                    "visionflow"
                } else {
                    "logseq" // default to logseq for backward compatibility
                };
                
                let physics = current.get_physics(graph_name);
                let sim_params = crate::models::simulation_params::SimulationParams::from(physics);
                let update_msg = UpdateSimulationParams { params: sim_params };
                
                // Send to GraphServiceActor
                if let Some(graph_addr) = &graph_service_addr {
                    graph_addr.do_send(update_msg.clone());
                    info!("[SETTINGS ACTOR] Physics update forwarded to GraphServiceActor for {}", graph_name);
                } else {
                    error!("[SETTINGS ACTOR] GraphServiceActor not available - GPU will not receive physics updates!");
                }
                
                // Send to GPUComputeActor
                if let Some(gpu_addr) = &gpu_compute_addr {
                    gpu_addr.do_send(update_msg);
                    info!("[SETTINGS ACTOR] Critical physics update forwarded immediately to GPUComputeActor for {}", graph_name);
                } else {
                    error!("[SETTINGS ACTOR] GPUComputeActor not available - GPU will not receive physics updates!");
                }
            }

            // Save to file if persistence is enabled
            if current.system.persist_settings {
                current.save().map_err(|e| {
                    error!("Failed to save settings after immediate update: {}", e);
                    VisionFlowError::Settings(SettingsError::SaveFailed {
                        file_path: "immediate_update".to_string(),
                        reason: e,
                    })
                })?;
            }

            info!("[CONCURRENT UPDATES] Successfully processed immediate critical update at path: {} = {:?}", path, value);
            Ok(())
        })
    }
}

// Handler for batch path operations - for high-frequency updates like sliders
impl Handler<GetSettingsByPaths> for SettingsActor {
    type Result = ResponseFuture<VisionFlowResult<HashMap<String, Value>>>;
    
    fn handle(&mut self, msg: GetSettingsByPaths, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let paths = msg.paths;
        
        Box::pin(async move {
            let current = settings.read().await;
            let mut results = HashMap::new();
            
            for path in paths {
                match current.get_by_path(&path) {
                    Ok(_boxed_value) => {
                        // The boxed_value is retrieved successfully, now get the JSON representation
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
                        warn!("Unable to retrieve setting at path '{}', skipping", path);
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
    type Result = ResponseFuture<VisionFlowResult<()>>;
    
    fn handle(&mut self, msg: SetSettingsByPaths, _ctx: &mut Self::Context) -> Self::Result {
        let settings = self.settings.clone();
        let updates = msg.updates;
        let graph_service_addr = self.graph_service_addr.clone();
        let gpu_compute_addr = self.gpu_compute_addr.clone();
        
        Box::pin(async move {
            let mut current = settings.write().await;
            let mut validation_needed = false;
            let mut physics_updated = false;
            let mut bloom_glow_updated = false;
            
            for (path, value) in updates {
                // Use the same logic as SetSettingByPath but in batch
                if path.starts_with("visualisation.graphs.logseq.physics.") {
                    validation_needed = true;
                    physics_updated = true;
                    
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
                // Check for bloom/glow updates
                else if path.starts_with("visualisation.") && (path.contains(".bloom.") || path.contains(".glow.")) {
                    validation_needed = true;
                    bloom_glow_updated = true;
                    
                    // Use JSON path access for bloom/glow fields  
                    if let Err(e) = current.set_json_by_path(&path, value.clone()) {
                        error!("Failed to batch update bloom/glow field {}: {}", path, e);
                        continue;
                    }
                    
                    debug!("Batch updated bloom/glow setting: {} = {:?}", path, value);
                }
            }
            
            // Only validate once for all batch updates
            if validation_needed {
                current.validate_config_camel_case().map_err(|e| {
                    if bloom_glow_updated {
                        error!("Bloom/Glow batch validation failed: {:?}", e);
                        VisionFlowError::Settings(SettingsError::ValidationFailed {
                            setting_path: "batch_settings".to_string(),
                            reason: format!("Batch validation failed for bloom/glow settings: {:?}. This validation prevents GPU kernel crashes from invalid values like NaN, negative intensities, or malformed hex colors.", e),
                        })
                    } else {
                        error!("Validation failed after batch update: {:?}", e);
                        VisionFlowError::Settings(SettingsError::ValidationFailed {
                            setting_path: "batch_settings".to_string(),
                            reason: format!("Batch validation failed: {:?}", e),
                        })
                    }
                })?;
            }
            
            // Log successful bloom/glow batch updates
            if bloom_glow_updated {
                info!("[SETTINGS ACTOR] Batch bloom/glow parameters updated successfully (validation passed)");
            }
            
            // CRITICAL: Forward physics updates to GPU actors if any physics parameters changed
            // This is especially important for batch slider updates from the UI
            if physics_updated {
                info!("[SETTINGS ACTOR] Batch physics parameters updated - forwarding to GPU actors");
                
                let physics = current.get_physics("logseq");
                let sim_params = crate::models::simulation_params::SimulationParams::from(physics);
                let update_msg = UpdateSimulationParams { params: sim_params };
                
                // Send to GraphServiceActor
                if let Some(graph_addr) = &graph_service_addr {
                    graph_addr.do_send(update_msg.clone());
                    info!("[SETTINGS ACTOR] Batch physics updates forwarded to GraphServiceActor");
                } else {
                    error!("[SETTINGS ACTOR] GraphServiceActor not available - GPU will not receive batch physics updates!");
                }
                
                // Send to GPUComputeActor
                if let Some(gpu_addr) = &gpu_compute_addr {
                    gpu_addr.do_send(update_msg);
                    info!("[SETTINGS ACTOR] Batch physics updates forwarded to GPUComputeActor");
                } else {
                    error!("[SETTINGS ACTOR] GPUComputeActor not available - GPU will not receive batch physics updates!");
                }
            }
            
            // Save to file if persistence is enabled and validation was needed
            if validation_needed && current.system.persist_settings {
                current.save().map_err(|e| {
                    error!("Failed to save settings after batch update: {}", e);
                    VisionFlowError::Settings(SettingsError::SaveFailed {
                        file_path: "batch_settings".to_string(),
                        reason: e,
                    })
                })?;
            }
            
            info!("Successfully completed batch settings update");
            Ok(())
        })
    }
}

// Handler for BatchedUpdate message - concurrent update handling
impl Handler<BatchedUpdate> for SettingsActor {
    type Result = ResponseFuture<VisionFlowResult<()>>;

    fn handle(&mut self, msg: BatchedUpdate, ctx: &mut Self::Context) -> Self::Result {
        // Check for mailbox overflow first
        if self.check_mailbox_overflow(ctx) {
            // Return early if emergency processing was triggered
            let future = async { Ok(()) };
            return Box::pin(future);
        }
        
        // Add updates to priority queue
        let update_count = msg.updates.len();
        for update in msg.updates {
            self.update_queue.push(Reverse(update));
        }
        
        self.metrics.pending_messages += update_count;
        self.metrics.priority_queue_size = self.update_queue.len();
        
        info!(
            "[CONCURRENT UPDATES] Queued {} updates, total queue size: {}, priority breakdown in queue: Critical: {}, High: {}, Normal: {}, Low: {}",
            update_count,
            self.update_queue.len(),
            self.update_queue.iter().filter(|Reverse(u)| u.priority == UpdatePriority::Critical).count(),
            self.update_queue.iter().filter(|Reverse(u)| u.priority == UpdatePriority::High).count(),
            self.update_queue.iter().filter(|Reverse(u)| u.priority == UpdatePriority::Normal).count(),
            self.update_queue.iter().filter(|Reverse(u)| u.priority == UpdatePriority::Low).count(),
        );
        
        // Check if we should process immediately (critical updates or queue full)
        let has_critical = self.update_queue.iter().any(|Reverse(u)| u.priority == UpdatePriority::Critical);
        let queue_full = self.update_queue.len() >= MAX_BATCH_SIZE;
        
        if has_critical || queue_full {
            info!("[CONCURRENT UPDATES] Processing immediately due to: critical_updates: {}, queue_full: {}", has_critical, queue_full);
            self.process_pending_batches(ctx);
        } else {
            // Schedule batch processing with timeout
            self.schedule_batch_processing(ctx);
        }
        
        let future = async { Ok(()) };
        Box::pin(future)
    }
}
