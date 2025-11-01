// src/settings/settings_actor.rs
//! Settings Actor - Actix runtime settings management

use actix::prelude::*;
use actix::dev::{MessageResponse, OneshotSender};
use std::sync::Arc;
use anyhow::Result;
use log::{info, error};
use crate::config::{PhysicsSettings, RenderingSettings};
use super::settings_repository::SettingsRepository;
use super::models::{ConstraintSettings, AllSettings, SettingsProfile};

/// Settings Actor for managing runtime settings
pub struct SettingsActor {
    repository: Arc<SettingsRepository>,
    current_physics: PhysicsSettings,
    current_constraints: ConstraintSettings,
    current_rendering: RenderingSettings,
}

impl SettingsActor {
    pub fn new(repository: Arc<SettingsRepository>) -> Self {
        Self {
            repository,
            current_physics: PhysicsSettings::default(),
            current_constraints: ConstraintSettings::default(),
            current_rendering: RenderingSettings::default(),
        }
    }

    /// Initialize actor by loading settings from database
    pub fn initialize(&mut self) -> Result<()> {
        match self.repository.load_all_settings() {
            Ok(settings) => {
                self.current_physics = settings.physics;
                self.current_constraints = settings.constraints;
                self.current_rendering = settings.rendering;
                info!("Settings actor initialized with persisted settings");
                Ok(())
            }
            Err(e) => {
                error!("Failed to load persisted settings, using defaults: {}", e);
                Ok(()) // Use defaults on error
            }
        }
    }
}

impl Actor for SettingsActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("SettingsActor started");

        // Load settings synchronously (repository methods are sync)
        match self.repository.load_all_settings() {
            Ok(settings) => {
                self.current_physics = settings.physics;
                self.current_constraints = settings.constraints;
                self.current_rendering = settings.rendering;
                info!("Settings loaded successfully");
            }
            Err(e) => {
                error!("Failed to load settings: {}", e);
            }
        }
    }
}

// ============================================================================
// Message Types
// ============================================================================

/// Update physics settings
#[derive(Message)]
#[rtype(result = "Result<()>")]
pub struct UpdatePhysicsSettings(pub PhysicsSettings);

/// Get current physics settings
#[derive(Message)]
#[rtype(result = "PhysicsSettings")]
pub struct GetPhysicsSettings;

/// Update constraint settings
#[derive(Message)]
#[rtype(result = "Result<()>")]
pub struct UpdateConstraintSettings(pub ConstraintSettings);

/// Get current constraint settings
#[derive(Message)]
#[rtype(result = "ConstraintSettings")]
pub struct GetConstraintSettings;

/// Update rendering settings
#[derive(Message)]
#[rtype(result = "Result<()>")]
pub struct UpdateRenderingSettings(pub RenderingSettings);

/// Get current rendering settings
#[derive(Message)]
#[rtype(result = "RenderingSettings")]
pub struct GetRenderingSettings;

/// Load a settings profile
#[derive(Message)]
#[rtype(result = "Result<AllSettings>")]
pub struct LoadProfile(pub i64);

/// Save current settings as a profile
#[derive(Message)]
#[rtype(result = "Result<i64>")]
pub struct SaveProfile {
    pub name: String,
}

/// List all settings profiles
#[derive(Message)]
#[rtype(result = "Result<Vec<SettingsProfile>>")]
pub struct ListProfiles;

/// Delete a settings profile
#[derive(Message)]
#[rtype(result = "Result<()>")]
pub struct DeleteProfile(pub i64);

/// Get all current settings
#[derive(Message)]
#[rtype(result = "AllSettings")]
pub struct GetAllSettings;

// ============================================================================
// MessageResponse Implementations
// ============================================================================

impl<A, M> MessageResponse<A, M> for PhysicsSettings
where
    A: Actor,
    M: Message<Result = PhysicsSettings>,
{
    fn handle(self, _ctx: &mut A::Context, tx: Option<OneshotSender<M::Result>>) {
        if let Some(tx) = tx {
            let _ = tx.send(self);
        }
    }
}

impl<A, M> MessageResponse<A, M> for ConstraintSettings
where
    A: Actor,
    M: Message<Result = ConstraintSettings>,
{
    fn handle(self, _ctx: &mut A::Context, tx: Option<OneshotSender<M::Result>>) {
        if let Some(tx) = tx {
            let _ = tx.send(self);
        }
    }
}

impl<A, M> MessageResponse<A, M> for RenderingSettings
where
    A: Actor,
    M: Message<Result = RenderingSettings>,
{
    fn handle(self, _ctx: &mut A::Context, tx: Option<OneshotSender<M::Result>>) {
        if let Some(tx) = tx {
            let _ = tx.send(self);
        }
    }
}

impl<A, M> MessageResponse<A, M> for AllSettings
where
    A: Actor,
    M: Message<Result = AllSettings>,
{
    fn handle(self, _ctx: &mut A::Context, tx: Option<OneshotSender<M::Result>>) {
        if let Some(tx) = tx {
            let _ = tx.send(self);
        }
    }
}

// ============================================================================
// Message Handlers
// ============================================================================

impl Handler<UpdatePhysicsSettings> for SettingsActor {
    type Result = ResponseFuture<Result<()>>;

    fn handle(&mut self, msg: UpdatePhysicsSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.current_physics = msg.0.clone();
        let repository = self.repository.clone();
        let settings = msg.0;

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                repository.save_physics_settings(&settings)?;
                info!("Physics settings updated and persisted");

                // TODO: Notify GpuPhysicsActor of settings change
                // This would require access to the GpuPhysicsActor address
                // For now, settings will be picked up on next physics iteration

                Ok(())
            }).await.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        })
    }
}

impl Handler<GetPhysicsSettings> for SettingsActor {
    type Result = PhysicsSettings;

    fn handle(&mut self, _msg: GetPhysicsSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.current_physics.clone()
    }
}

impl Handler<UpdateConstraintSettings> for SettingsActor {
    type Result = ResponseFuture<Result<()>>;

    fn handle(&mut self, msg: UpdateConstraintSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.current_constraints = msg.0.clone();
        let repository = self.repository.clone();
        let settings = msg.0;

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                repository.save_constraint_settings(&settings)?;
                info!("Constraint settings updated and persisted");
                Ok(())
            }).await.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        })
    }
}

impl Handler<GetConstraintSettings> for SettingsActor {
    type Result = ConstraintSettings;

    fn handle(&mut self, _msg: GetConstraintSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.current_constraints.clone()
    }
}

impl Handler<UpdateRenderingSettings> for SettingsActor {
    type Result = ResponseFuture<Result<()>>;

    fn handle(&mut self, msg: UpdateRenderingSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.current_rendering = msg.0.clone();
        let repository = self.repository.clone();
        let settings = msg.0;

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                repository.save_rendering_settings(&settings)?;
                info!("Rendering settings updated and persisted");
                Ok(())
            }).await.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        })
    }
}

impl Handler<GetRenderingSettings> for SettingsActor {
    type Result = RenderingSettings;

    fn handle(&mut self, _msg: GetRenderingSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.current_rendering.clone()
    }
}

impl Handler<LoadProfile> for SettingsActor {
    type Result = ResponseFuture<Result<AllSettings>>;

    fn handle(&mut self, msg: LoadProfile, _ctx: &mut Self::Context) -> Self::Result {
        let repository = self.repository.clone();
        let profile_id = msg.0;

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let settings = repository.load_profile(profile_id)?;
                info!("Loaded settings profile {}", profile_id);
                Ok(settings)
            }).await.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        })
    }
}

impl Handler<SaveProfile> for SettingsActor {
    type Result = ResponseFuture<Result<i64>>;

    fn handle(&mut self, msg: SaveProfile, _ctx: &mut Self::Context) -> Self::Result {
        let repository = self.repository.clone();
        let settings = AllSettings {
            physics: self.current_physics.clone(),
            constraints: self.current_constraints.clone(),
            rendering: self.current_rendering.clone(),
        };

        Box::pin(async move {
            let name = msg.name.clone();
            tokio::task::spawn_blocking(move || {
                let profile_id = repository.save_profile(&name, &settings)?;
                info!("Saved settings profile '{}' with ID {}", name, profile_id);
                Ok(profile_id)
            }).await.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        })
    }
}

impl Handler<ListProfiles> for SettingsActor {
    type Result = ResponseFuture<Result<Vec<SettingsProfile>>>;

    fn handle(&mut self, _msg: ListProfiles, _ctx: &mut Self::Context) -> Self::Result {
        let repository = self.repository.clone();

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                repository.list_profiles()
            }).await.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        })
    }
}

impl Handler<DeleteProfile> for SettingsActor {
    type Result = ResponseFuture<Result<()>>;

    fn handle(&mut self, msg: DeleteProfile, _ctx: &mut Self::Context) -> Self::Result {
        let repository = self.repository.clone();
        let profile_id = msg.0;

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                repository.delete_profile(profile_id)?;
                info!("Deleted settings profile {}", profile_id);
                Ok(())
            }).await.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        })
    }
}

impl Handler<GetAllSettings> for SettingsActor {
    type Result = AllSettings;

    fn handle(&mut self, _msg: GetAllSettings, _ctx: &mut Self::Context) -> Self::Result {
        AllSettings {
            physics: self.current_physics.clone(),
            constraints: self.current_constraints.clone(),
            rendering: self.current_rendering.clone(),
        }
    }
}
