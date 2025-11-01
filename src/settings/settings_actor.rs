// src/settings/settings_actor.rs
//! Settings Actor - Actix runtime settings management

use actix::prelude::*;
use actix::dev::{MessageResponse, OneshotSender};
use std::sync::Arc;
use anyhow::Result;
use log::{info, error};
use crate::config::{PhysicsSettings, RenderingSettings};
use crate::ports::settings_repository::SettingsRepository;
use super::models::{ConstraintSettings, AllSettings, SettingsProfile};

/// Settings Actor for managing runtime settings
pub struct SettingsActor {
    repository: Arc<dyn SettingsRepository>,
    current_physics: PhysicsSettings,
    current_constraints: ConstraintSettings,
    current_rendering: RenderingSettings,
}

impl SettingsActor {
    pub fn new(repository: Arc<dyn SettingsRepository>) -> Self {
        Self {
            repository,
            current_physics: PhysicsSettings::default(),
            current_constraints: ConstraintSettings::default(),
            current_rendering: RenderingSettings::default(),
        }
    }

    /// Initialize actor by loading settings from database
    /// Note: This is a stub - actual initialization happens in Actor::started
    pub fn initialize(&mut self) -> Result<()> {
        info!("Settings actor initialized with defaults (async load will occur on start)");
        Ok(())
    }
}

impl Actor for SettingsActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("SettingsActor started with default settings");
        // Settings are loaded on-demand via message handlers
        // Repository stub returns defaults, so we use in-memory defaults
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
            // Use generic save_physics_settings from trait
            repository.save_physics_settings("default", &settings).await?;
            info!("Physics settings updated and persisted");
            Ok(())
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
        info!("Constraint settings updated (in-memory only)");
        Box::pin(async move { Ok(()) })
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
        info!("Rendering settings updated (in-memory only)");
        Box::pin(async move { Ok(()) })
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
        let _profile_id = msg.0;
        info!("Profile loading not implemented, returning defaults");
        Box::pin(async move { Ok(AllSettings::default()) })
    }
}

impl Handler<SaveProfile> for SettingsActor {
    type Result = ResponseFuture<Result<i64>>;

    fn handle(&mut self, msg: SaveProfile, _ctx: &mut Self::Context) -> Self::Result {
        let name = msg.name.clone();
        info!("Profile saving not implemented for '{}'", name);
        Box::pin(async move { Ok(1) }) // Return dummy ID
    }
}

impl Handler<ListProfiles> for SettingsActor {
    type Result = ResponseFuture<Result<Vec<SettingsProfile>>>;

    fn handle(&mut self, _msg: ListProfiles, _ctx: &mut Self::Context) -> Self::Result {
        info!("Profile listing not implemented");
        Box::pin(async move { Ok(Vec::new()) })
    }
}

impl Handler<DeleteProfile> for SettingsActor {
    type Result = ResponseFuture<Result<()>>;

    fn handle(&mut self, msg: DeleteProfile, _ctx: &mut Self::Context) -> Self::Result {
        let profile_id = msg.0;
        info!("Profile deletion not implemented for ID {}", profile_id);
        Box::pin(async move { Ok(()) })
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
