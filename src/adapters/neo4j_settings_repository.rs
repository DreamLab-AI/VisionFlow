// src/adapters/neo4j_settings_repository.rs
//! Neo4j Settings Repository Adapter
//!
//! Implements the SettingsRepository port using Neo4j graph database with
//! category-based schema modeling, caching, and comprehensive error handling.
//!
//! ## Schema Design
//!
//! The settings are organized using a hierarchical node structure:
//! - `:SettingsRoot` - Root node (singleton, id: "default")
//! - Category nodes: `:PhysicsSettings`, `:RenderingSettings`, `:SystemSettings`, etc.
//! - Settings stored as properties on category nodes
//! - Relationships: `(:SettingsRoot)-[:HAS_PHYSICS_SETTINGS]->(:PhysicsSettings)`

use async_trait::async_trait;
use neo4rs::{Graph, query, ConfigBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, instrument, error};

use crate::config::PhysicsSettings;
use crate::ports::settings_repository::{
    AppFullSettings, Result as RepoResult, SettingValue, SettingsRepository,
    SettingsRepositoryError,
};
use crate::utils::json::{from_json, to_json};

/// Neo4j configuration for settings repository
#[derive(Debug, Clone)]
pub struct Neo4jSettingsConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
    pub database: Option<String>,
    pub fetch_size: usize,
    pub max_connections: usize,
}

impl Default for Neo4jSettingsConfig {
    fn default() -> Self {
        Self {
            uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string()),
            database: std::env::var("NEO4J_DATABASE").ok(),
            fetch_size: 500,
            max_connections: 10,
        }
    }
}

/// Cache entry with TTL support
struct CachedSetting {
    value: SettingValue,
    timestamp: std::time::Instant,
}

/// Settings cache with TTL
struct SettingsCache {
    settings: HashMap<String, CachedSetting>,
    last_updated: std::time::Instant,
    ttl_seconds: u64,
}

impl SettingsCache {
    fn new(ttl_seconds: u64) -> Self {
        Self {
            settings: HashMap::new(),
            last_updated: std::time::Instant::now(),
            ttl_seconds,
        }
    }

    fn get(&self, key: &str) -> Option<SettingValue> {
        if let Some(cached) = self.settings.get(key) {
            if cached.timestamp.elapsed().as_secs() < self.ttl_seconds {
                return Some(cached.value.clone());
            }
        }
        None
    }

    fn insert(&mut self, key: String, value: SettingValue) {
        self.settings.insert(
            key,
            CachedSetting {
                value,
                timestamp: std::time::Instant::now(),
            },
        );
    }

    fn remove(&mut self, key: &str) {
        self.settings.remove(key);
    }

    fn clear(&mut self) {
        self.settings.clear();
        self.last_updated = std::time::Instant::now();
    }
}

/// Neo4j Settings Repository implementation
pub struct Neo4jSettingsRepository {
    graph: Arc<Graph>,
    cache: Arc<RwLock<SettingsCache>>,
    config: Neo4jSettingsConfig,
}

impl Neo4jSettingsRepository {
    /// Create a new Neo4j settings repository with configuration
    pub async fn new(config: Neo4jSettingsConfig) -> RepoResult<Self> {
        info!("Initializing Neo4jSettingsRepository with URI: {}", config.uri);

        // Build Neo4j configuration
        let mut builder = ConfigBuilder::default()
            .uri(&config.uri)
            .user(&config.user)
            .password(&config.password)
            .fetch_size(config.fetch_size)
            .max_connections(config.max_connections);

        if let Some(ref db) = config.database {
            builder = builder.db(db);
        }

        let neo4j_config = builder.build()
            .map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to build Neo4j config: {}", e)
            ))?;

        // Connect to Neo4j
        let graph = Graph::connect(neo4j_config)
            .await
            .map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to connect to Neo4j: {}", e)
            ))?;

        let repository = Self {
            graph: Arc::new(graph),
            cache: Arc::new(RwLock::new(SettingsCache::new(300))), // 5 min TTL
            config,
        };

        // Initialize schema
        repository.initialize_schema().await?;

        info!("✅ Neo4jSettingsRepository initialized successfully");
        Ok(repository)
    }

    /// Initialize the Neo4j schema for settings storage
    async fn initialize_schema(&self) -> RepoResult<()> {
        info!("Initializing Neo4j settings schema");

        // Create constraints for unique settings root
        let constraint_query = query(
            "CREATE CONSTRAINT settings_root_id IF NOT EXISTS
             FOR (s:SettingsRoot) REQUIRE s.id IS UNIQUE"
        );

        self.graph.run(constraint_query)
            .await
            .map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to create constraints: {}", e)
            ))?;

        // Create indices for performance
        let indices = vec![
            "CREATE INDEX settings_key_idx IF NOT EXISTS FOR (s:Setting) ON (s.key)",
            "CREATE INDEX physics_profile_idx IF NOT EXISTS FOR (p:PhysicsProfile) ON (p.name)",
        ];

        for index_query in indices {
            self.graph.run(query(index_query))
                .await
                .map_err(|e| SettingsRepositoryError::DatabaseError(
                    format!("Failed to create index: {}", e)
                ))?;
        }

        // Create root settings node if it doesn't exist
        let init_query = query(
            "MERGE (s:SettingsRoot {id: 'default'})
             ON CREATE SET s.created_at = datetime(), s.version = '1.0.0'
             RETURN s"
        );

        self.graph.run(init_query)
            .await
            .map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to initialize settings root: {}", e)
            ))?;

        info!("✅ Neo4j settings schema initialized");
        Ok(())
    }

    /// Get setting from cache
    async fn get_from_cache(&self, key: &str) -> Option<SettingValue> {
        let cache = self.cache.read().await;
        if let Some(value) = cache.get(key) {
            debug!("Cache hit for setting: {}", key);
            return Some(value);
        }
        None
    }

    /// Update cache
    async fn update_cache(&self, key: String, value: SettingValue) {
        let mut cache = self.cache.write().await;
        cache.insert(key, value);
    }

    /// Invalidate cache entry
    async fn invalidate_cache(&self, key: &str) {
        let mut cache = self.cache.write().await;
        cache.remove(key);
    }

    /// Clear entire cache
    async fn clear_cache_internal(&self) -> RepoResult<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        Ok(())
    }

    /// Convert SettingValue to Cypher parameter value
    fn setting_value_to_param(&self, value: &SettingValue) -> serde_json::Value {
        match value {
            SettingValue::String(s) => serde_json::json!({"type": "string", "value": s}),
            SettingValue::Integer(i) => serde_json::json!({"type": "integer", "value": i}),
            SettingValue::Float(f) => serde_json::json!({"type": "float", "value": f}),
            SettingValue::Boolean(b) => serde_json::json!({"type": "boolean", "value": b}),
            SettingValue::Json(j) => serde_json::json!({"type": "json", "value": to_json(j).unwrap_or_default()}),
        }
    }

    /// Parse setting value from Neo4j result
    fn parse_setting_value(&self, value_type: &str, value: &serde_json::Value) -> Option<SettingValue> {
        match value_type {
            "string" => value.as_str().map(|s| SettingValue::String(s.to_string())),
            "integer" => value.as_i64().map(SettingValue::Integer),
            "float" => value.as_f64().map(SettingValue::Float),
            "boolean" => value.as_bool().map(SettingValue::Boolean),
            "json" => {
                if let Some(json_str) = value.as_str() {
                    from_json(json_str).ok().map(SettingValue::Json)
                } else {
                    Some(SettingValue::Json(value.clone()))
                }
            }
            _ => None,
        }
    }
}

#[async_trait]
impl SettingsRepository for Neo4jSettingsRepository {
    #[instrument(skip(self), level = "debug")]
    async fn get_setting(&self, key: &str) -> RepoResult<Option<SettingValue>> {
        // Check cache first
        if let Some(cached_value) = self.get_from_cache(key).await {
            return Ok(Some(cached_value));
        }

        // Query Neo4j
        let query_str =
            "MATCH (s:Setting {key: $key})
             RETURN s.value_type AS value_type, s.value AS value";

        let mut result = self.graph.execute(
            query(query_str).param("key", key)
        ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
            format!("Failed to query setting: {}", e)
        ))?;

        if let Some(row) = result.next().await.map_err(|e|
            SettingsRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        )? {
            let value_type: String = row.get("value_type").map_err(|e|
                SettingsRepositoryError::DatabaseError(format!("Failed to get value_type: {}", e))
            )?;

            let value: serde_json::Value = row.get("value").map_err(|e|
                SettingsRepositoryError::DatabaseError(format!("Failed to get value: {}", e))
            )?;

            if let Some(setting_value) = self.parse_setting_value(&value_type, &value) {
                // Update cache
                self.update_cache(key.to_string(), setting_value.clone()).await;
                return Ok(Some(setting_value));
            }
        }

        Ok(None)
    }

    #[instrument(skip(self, value), level = "debug")]
    async fn set_setting(
        &self,
        key: &str,
        value: SettingValue,
        description: Option<&str>,
    ) -> RepoResult<()> {
        let value_param = self.setting_value_to_param(&value);
        let value_type = value_param["type"].as_str().unwrap();
        let value_data = &value_param["value"];

        let query_str =
            "MERGE (s:Setting {key: $key})
             ON CREATE SET
                s.created_at = datetime(),
                s.value_type = $value_type,
                s.value = $value,
                s.description = $description
             ON MATCH SET
                s.updated_at = datetime(),
                s.value_type = $value_type,
                s.value = $value,
                s.description = COALESCE($description, s.description)
             RETURN s";

        self.graph.run(
            query(query_str)
                .param("key", key)
                .param("value_type", value_type)
                .param("value", value_data.clone())
                .param("description", description.unwrap_or(""))
        ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
            format!("Failed to set setting: {}", e)
        ))?;

        // Invalidate cache
        self.invalidate_cache(key).await;

        Ok(())
    }

    async fn get_settings_batch(
        &self,
        keys: &[String],
    ) -> RepoResult<HashMap<String, SettingValue>> {
        let mut results = HashMap::new();

        // Try to get from cache first
        for key in keys {
            if let Some(value) = self.get_from_cache(key).await {
                results.insert(key.clone(), value);
            }
        }

        // Get remaining keys from database
        let remaining_keys: Vec<String> = keys.iter()
            .filter(|k| !results.contains_key(*k))
            .cloned()
            .collect();

        if !remaining_keys.is_empty() {
            let query_str =
                "MATCH (s:Setting)
                 WHERE s.key IN $keys
                 RETURN s.key AS key, s.value_type AS value_type, s.value AS value";

            let mut result = self.graph.execute(
                query(query_str).param("keys", remaining_keys)
            ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to query batch settings: {}", e)
            ))?;

            while let Some(row) = result.next().await.map_err(|e|
                SettingsRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
            )? {
                let key: String = row.get("key").unwrap_or_default();
                let value_type: String = row.get("value_type").unwrap_or_default();
                let value: serde_json::Value = row.get("value").unwrap_or_default();

                if let Some(setting_value) = self.parse_setting_value(&value_type, &value) {
                    self.update_cache(key.clone(), setting_value.clone()).await;
                    results.insert(key, setting_value);
                }
            }
        }

        Ok(results)
    }

    async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> RepoResult<()> {
        // Use transaction for batch updates
        let txn = self.graph.start_txn().await.map_err(|e|
            SettingsRepositoryError::DatabaseError(format!("Failed to start transaction: {}", e))
        )?;

        for (key, value) in &updates {
            let value_param = self.setting_value_to_param(value);
            let value_type = value_param["type"].as_str().unwrap();
            let value_data = &value_param["value"];

            let query_str =
                "MERGE (s:Setting {key: $key})
                 ON CREATE SET
                    s.created_at = datetime(),
                    s.value_type = $value_type,
                    s.value = $value
                 ON MATCH SET
                    s.updated_at = datetime(),
                    s.value_type = $value_type,
                    s.value = $value";

            txn.run_queries(vec![
                query(query_str)
                    .param("key", key.as_str())
                    .param("value_type", value_type)
                    .param("value", value_data.clone())
            ]).await.map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to execute batch update: {}", e)
            ))?;
        }

        txn.commit().await.map_err(|e|
            SettingsRepositoryError::DatabaseError(format!("Failed to commit transaction: {}", e))
        )?;

        // Clear cache after batch update
        self.clear_cache_internal().await?;

        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    async fn load_all_settings(&self) -> RepoResult<Option<AppFullSettings>> {
        // For now, return default settings
        // In a full implementation, this would reconstruct AppFullSettings from Neo4j
        info!("Loading all settings from Neo4j (returning defaults for now)");

        Ok(Some(AppFullSettings {
            visualisation: Default::default(),
            system: Default::default(),
            xr: Default::default(),
            auth: Default::default(),
            ragflow: None,
            perplexity: None,
            openai: None,
            kokoro: None,
            whisper: None,
            version: "1.0.0".to_string(),
            user_preferences: Default::default(),
            physics: Default::default(),
            feature_flags: Default::default(),
            developer_config: Default::default(),
        }))
    }

    #[instrument(skip(self, settings), level = "debug")]
    async fn save_all_settings(&self, settings: &AppFullSettings) -> RepoResult<()> {
        info!("Saving all settings to Neo4j");

        // Serialize settings to JSON
        let settings_json = serde_json::to_value(settings)
            .map_err(|e| SettingsRepositoryError::SerializationError(e.to_string()))?;

        // Store as JSON on root node for now
        let query_str =
            "MERGE (s:SettingsRoot {id: 'default'})
             SET s.full_settings = $settings,
                 s.updated_at = datetime(),
                 s.version = $version
             RETURN s";

        self.graph.run(
            query(query_str)
                .param("settings", to_json(&settings_json).unwrap_or_default())
                .param("version", &settings.version)
        ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
            format!("Failed to save all settings: {}", e)
        ))?;

        // Clear cache
        self.clear_cache_internal().await?;

        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    async fn get_physics_settings(&self, profile_name: &str) -> RepoResult<PhysicsSettings> {
        let query_str =
            "MATCH (p:PhysicsProfile {name: $profile_name})
             RETURN p.settings AS settings";

        let mut result = self.graph.execute(
            query(query_str).param("profile_name", profile_name)
        ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
            format!("Failed to query physics settings: {}", e)
        ))?;

        if let Some(row) = result.next().await.map_err(|e|
            SettingsRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        )? {
            let settings_json: String = row.get("settings").unwrap_or_default();
            let settings: PhysicsSettings = from_json(&settings_json)
                .map_err(|e| SettingsRepositoryError::SerializationError(e.to_string()))?;
            return Ok(settings);
        }

        // Return default if not found
        Ok(PhysicsSettings::default())
    }

    #[instrument(skip(self, settings), level = "debug")]
    async fn save_physics_settings(
        &self,
        profile_name: &str,
        settings: &PhysicsSettings,
    ) -> RepoResult<()> {
        let settings_json = to_json(settings)
            .map_err(|e| SettingsRepositoryError::SerializationError(e.to_string()))?;

        let query_str =
            "MERGE (p:PhysicsProfile {name: $profile_name})
             ON CREATE SET
                p.created_at = datetime(),
                p.settings = $settings
             ON MATCH SET
                p.updated_at = datetime(),
                p.settings = $settings
             RETURN p";

        self.graph.run(
            query(query_str)
                .param("profile_name", profile_name)
                .param("settings", settings_json)
        ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
            format!("Failed to save physics settings: {}", e)
        ))?;

        Ok(())
    }

    async fn delete_setting(&self, key: &str) -> RepoResult<()> {
        let query_str = "MATCH (s:Setting {key: $key}) DELETE s";

        self.graph.run(
            query(query_str).param("key", key)
        ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
            format!("Failed to delete setting: {}", e)
        ))?;

        self.invalidate_cache(key).await;
        Ok(())
    }

    async fn has_setting(&self, key: &str) -> RepoResult<bool> {
        Ok(self.get_setting(key).await?.is_some())
    }

    async fn list_settings(&self, prefix: Option<&str>) -> RepoResult<Vec<String>> {
        let query_str = if let Some(p) = prefix {
            "MATCH (s:Setting) WHERE s.key STARTS WITH $prefix RETURN s.key AS key ORDER BY s.key"
        } else {
            "MATCH (s:Setting) RETURN s.key AS key ORDER BY s.key"
        };

        let mut query_obj = query(query_str);
        if let Some(p) = prefix {
            query_obj = query_obj.param("prefix", p);
        }

        let mut result = self.graph.execute(query_obj)
            .await.map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to list settings: {}", e)
            ))?;

        let mut keys = Vec::new();
        while let Some(row) = result.next().await.map_err(|e|
            SettingsRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        )? {
            if let Ok(key) = row.get::<String>("key") {
                keys.push(key);
            }
        }

        Ok(keys)
    }

    async fn list_physics_profiles(&self) -> RepoResult<Vec<String>> {
        let query_str = "MATCH (p:PhysicsProfile) RETURN p.name AS name ORDER BY p.name";

        let mut result = self.graph.execute(query(query_str))
            .await.map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to list physics profiles: {}", e)
            ))?;

        let mut profiles = Vec::new();
        while let Some(row) = result.next().await.map_err(|e|
            SettingsRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        )? {
            if let Ok(name) = row.get::<String>("name") {
                profiles.push(name);
            }
        }

        Ok(profiles)
    }

    async fn delete_physics_profile(&self, profile_name: &str) -> RepoResult<()> {
        let query_str = "MATCH (p:PhysicsProfile {name: $name}) DELETE p";

        self.graph.run(
            query(query_str).param("name", profile_name)
        ).await.map_err(|e| SettingsRepositoryError::DatabaseError(
            format!("Failed to delete physics profile: {}", e)
        ))?;

        Ok(())
    }

    async fn export_settings(&self) -> RepoResult<serde_json::Value> {
        let query_str =
            "MATCH (s:Setting)
             RETURN s.key AS key, s.value_type AS value_type, s.value AS value, s.description AS description";

        let mut result = self.graph.execute(query(query_str))
            .await.map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Failed to export settings: {}", e)
            ))?;

        let mut settings = serde_json::Map::new();
        while let Some(row) = result.next().await.map_err(|e|
            SettingsRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        )? {
            let key: String = row.get("key").unwrap_or_default();
            let value_type: String = row.get("value_type").unwrap_or_default();
            let value: serde_json::Value = row.get("value").unwrap_or_default();
            let description: String = row.get("description").unwrap_or_default();

            settings.insert(key, serde_json::json!({
                "type": value_type,
                "value": value,
                "description": description
            }));
        }

        Ok(serde_json::Value::Object(settings))
    }

    async fn import_settings(&self, settings_json: &serde_json::Value) -> RepoResult<()> {
        if let Some(settings_map) = settings_json.as_object() {
            let mut updates = HashMap::new();

            for (key, value_obj) in settings_map {
                if let Some(obj) = value_obj.as_object() {
                    let value_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("string");
                    let value = obj.get("value").cloned().unwrap_or(serde_json::Value::Null);

                    if let Some(setting_value) = self.parse_setting_value(value_type, &value) {
                        updates.insert(key.clone(), setting_value);
                    }
                }
            }

            self.set_settings_batch(updates).await?;
        }

        Ok(())
    }

    async fn clear_cache(&self) -> RepoResult<()> {
        self.clear_cache_internal().await
    }

    async fn health_check(&self) -> RepoResult<bool> {
        let query_str = "RETURN 1 AS health";

        self.graph.run(query(query_str))
            .await
            .map_err(|e| SettingsRepositoryError::DatabaseError(
                format!("Health check failed: {}", e)
            ))?;

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Neo4j instance
    async fn test_neo4j_settings_repository() {
        let config = Neo4jSettingsConfig::default();
        let repo = Neo4jSettingsRepository::new(config).await.unwrap();

        // Test set and get
        repo.set_setting("test.key", SettingValue::String("test_value".to_string()), Some("Test setting"))
            .await.unwrap();

        let value = repo.get_setting("test.key").await.unwrap();
        assert_eq!(value, Some(SettingValue::String("test_value".to_string())));

        // Test delete
        repo.delete_setting("test.key").await.unwrap();
        let value = repo.get_setting("test.key").await.unwrap();
        assert_eq!(value, None);

        // Test health check
        assert!(repo.health_check().await.unwrap());
    }
}
