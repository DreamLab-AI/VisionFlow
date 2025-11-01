// src/cqrs/handlers/settings_handlers.rs
//! Settings Command and Query Handlers

use crate::cqrs::commands::*;
use crate::cqrs::queries::*;
use crate::cqrs::types::{Command, CommandHandler, Query, QueryHandler, Result};
use crate::ports::SettingsRepository;
use async_trait::async_trait;
use std::sync::Arc;

///
pub struct SettingsCommandHandler {
    repository: Arc<dyn SettingsRepository>,
}

impl SettingsCommandHandler {
    pub fn new(repository: Arc<dyn SettingsRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl CommandHandler<UpdateSettingCommand> for SettingsCommandHandler {
    async fn handle(&self, command: UpdateSettingCommand) -> Result<()> {
        command.validate()?;
        Ok(self
            .repository
            .set_setting(&command.key, command.value, command.description.as_deref())
            .await?)
    }
}

#[async_trait]
impl CommandHandler<UpdateBatchSettingsCommand> for SettingsCommandHandler {
    async fn handle(&self, command: UpdateBatchSettingsCommand) -> Result<()> {
        command.validate()?;
        Ok(self.repository.set_settings_batch(command.updates).await?)
    }
}

#[async_trait]
impl CommandHandler<DeleteSettingCommand> for SettingsCommandHandler {
    async fn handle(&self, command: DeleteSettingCommand) -> Result<()> {
        command.validate()?;
        Ok(self.repository.delete_setting(&command.key).await?)
    }
}

#[async_trait]
impl CommandHandler<SaveAllSettingsCommand> for SettingsCommandHandler {
    async fn handle(&self, command: SaveAllSettingsCommand) -> Result<()> {
        Ok(self.repository.save_all_settings(&command.settings).await?)
    }
}

#[async_trait]
impl CommandHandler<SavePhysicsSettingsCommand> for SettingsCommandHandler {
    async fn handle(&self, command: SavePhysicsSettingsCommand) -> Result<()> {
        command.validate()?;
        Ok(self
            .repository
            .save_physics_settings(&command.profile_name, &command.settings)
            .await?)
    }
}

#[async_trait]
impl CommandHandler<DeletePhysicsProfileCommand> for SettingsCommandHandler {
    async fn handle(&self, command: DeletePhysicsProfileCommand) -> Result<()> {
        command.validate()?;
        Ok(self
            .repository
            .delete_physics_profile(&command.profile_name)
            .await?)
    }
}

#[async_trait]
impl CommandHandler<ImportSettingsCommand> for SettingsCommandHandler {
    async fn handle(&self, command: ImportSettingsCommand) -> Result<()> {
        command.validate()?;
        Ok(self
            .repository
            .import_settings(&command.settings_json)
            .await?)
    }
}

#[async_trait]
impl CommandHandler<ClearSettingsCacheCommand> for SettingsCommandHandler {
    async fn handle(&self, _command: ClearSettingsCacheCommand) -> Result<()> {
        Ok(self.repository.clear_cache().await?)
    }
}

///
pub struct SettingsQueryHandler {
    repository: Arc<dyn SettingsRepository>,
}

impl SettingsQueryHandler {
    pub fn new(repository: Arc<dyn SettingsRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl QueryHandler<GetSettingQuery> for SettingsQueryHandler {
    async fn handle(
        &self,
        query: GetSettingQuery,
    ) -> Result<Option<crate::ports::settings_repository::SettingValue>> {
        query.validate()?;
        Ok(self.repository.get_setting(&query.key).await?)
    }
}

#[async_trait]
impl QueryHandler<GetBatchSettingsQuery> for SettingsQueryHandler {
    async fn handle(
        &self,
        query: GetBatchSettingsQuery,
    ) -> Result<std::collections::HashMap<String, crate::ports::settings_repository::SettingValue>>
    {
        query.validate()?;
        Ok(self.repository.get_settings_batch(&query.keys).await?)
    }
}

#[async_trait]
impl QueryHandler<GetAllSettingsQuery> for SettingsQueryHandler {
    async fn handle(
        &self,
        _query: GetAllSettingsQuery,
    ) -> Result<Option<crate::config::AppFullSettings>> {
        Ok(self.repository.load_all_settings().await?)
    }
}

#[async_trait]
impl QueryHandler<ListSettingsQuery> for SettingsQueryHandler {
    async fn handle(&self, query: ListSettingsQuery) -> Result<Vec<String>> {
        Ok(self
            .repository
            .list_settings(query.prefix.as_deref())
            .await?)
    }
}

#[async_trait]
impl QueryHandler<HasSettingQuery> for SettingsQueryHandler {
    async fn handle(&self, query: HasSettingQuery) -> Result<bool> {
        query.validate()?;
        Ok(self.repository.has_setting(&query.key).await?)
    }
}

#[async_trait]
impl QueryHandler<GetPhysicsSettingsQuery> for SettingsQueryHandler {
    async fn handle(
        &self,
        query: GetPhysicsSettingsQuery,
    ) -> Result<crate::config::PhysicsSettings> {
        query.validate()?;
        Ok(self
            .repository
            .get_physics_settings(&query.profile_name)
            .await?)
    }
}

#[async_trait]
impl QueryHandler<ListPhysicsProfilesQuery> for SettingsQueryHandler {
    async fn handle(&self, _query: ListPhysicsProfilesQuery) -> Result<Vec<String>> {
        Ok(self.repository.list_physics_profiles().await?)
    }
}

#[async_trait]
impl QueryHandler<ExportSettingsQuery> for SettingsQueryHandler {
    async fn handle(&self, _query: ExportSettingsQuery) -> Result<serde_json::Value> {
        Ok(self.repository.export_settings().await?)
    }
}

#[async_trait]
impl QueryHandler<SettingsHealthCheckQuery> for SettingsQueryHandler {
    async fn handle(&self, _query: SettingsHealthCheckQuery) -> Result<bool> {
        Ok(self.repository.health_check().await?)
    }
}
