// tests/adapters/sqlite_settings_repository_tests.rs
//! Integration tests for SqliteSettingsRepository
//!
//! Tests all 18 port methods with comprehensive coverage including:
//! - Basic CRUD operations
//! - Batch operations
//! - Cache behavior
//! - Physics settings
//! - Import/export
//! - Error handling
//! - Concurrent access

use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

use visionflow::adapters::sqlite_settings_repository::SqliteSettingsRepository;
use visionflow::config::PhysicsSettings;
use visionflow::ports::settings_repository::{
    AppFullSettings, SettingValue, SettingsRepository,
};
use visionflow::services::database_service::DatabaseService;

/// Create a temporary SQLite database for testing
async fn setup_test_db() -> Result<(TempDir, Arc<SqliteSettingsRepository>)> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_settings.db");
    let db_service = Arc::new(DatabaseService::new(db_path.to_str().unwrap())?);
    let repo = Arc::new(SqliteSettingsRepository::new(db_service));
    Ok((temp_dir, repo))
}

#[tokio::test]
async fn test_get_set_setting() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Test string value
    repo.set_setting(
        "test.string",
        SettingValue::String("hello".to_string()),
        Some("Test string setting"),
    )
    .await?;

    let value = repo.get_setting("test.string").await?;
    assert!(value.is_some());
    assert_eq!(
        value.unwrap().as_string(),
        Some("hello")
    );

    // Test integer value
    repo.set_setting(
        "test.integer",
        SettingValue::Integer(42),
        None,
    )
    .await?;

    let value = repo.get_setting("test.integer").await?;
    assert_eq!(value.unwrap().as_i64(), Some(42));

    // Test float value
    repo.set_setting(
        "test.float",
        SettingValue::Float(3.14),
        None,
    )
    .await?;

    let value = repo.get_setting("test.float").await?;
    assert_eq!(value.unwrap().as_f64(), Some(3.14));

    // Test boolean value
    repo.set_setting(
        "test.boolean",
        SettingValue::Boolean(true),
        None,
    )
    .await?;

    let value = repo.get_setting("test.boolean").await?;
    assert_eq!(value.unwrap().as_bool(), Some(true));

    Ok(())
}

#[tokio::test]
async fn test_delete_setting() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    repo.set_setting(
        "test.delete",
        SettingValue::String("value".to_string()),
        None,
    )
    .await?;

    assert!(repo.has_setting("test.delete").await?);

    repo.delete_setting("test.delete").await?;

    assert!(!repo.has_setting("test.delete").await?);
    assert!(repo.get_setting("test.delete").await?.is_none());

    Ok(())
}

#[tokio::test]
async fn test_has_setting() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    assert!(!repo.has_setting("nonexistent").await?);

    repo.set_setting(
        "test.exists",
        SettingValue::String("value".to_string()),
        None,
    )
    .await?;

    assert!(repo.has_setting("test.exists").await?);

    Ok(())
}

#[tokio::test]
async fn test_batch_operations() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Batch set
    let mut updates = HashMap::new();
    updates.insert("batch.1".to_string(), SettingValue::Integer(1));
    updates.insert("batch.2".to_string(), SettingValue::Integer(2));
    updates.insert("batch.3".to_string(), SettingValue::Integer(3));

    repo.set_settings_batch(updates).await?;

    // Batch get
    let keys = vec![
        "batch.1".to_string(),
        "batch.2".to_string(),
        "batch.3".to_string(),
    ];

    let results = repo.get_settings_batch(&keys).await?;

    assert_eq!(results.len(), 3);
    assert_eq!(results.get("batch.1").unwrap().as_i64(), Some(1));
    assert_eq!(results.get("batch.2").unwrap().as_i64(), Some(2));
    assert_eq!(results.get("batch.3").unwrap().as_i64(), Some(3));

    Ok(())
}

#[tokio::test]
async fn test_list_settings() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Add settings with different prefixes
    repo.set_setting("app.theme", SettingValue::String("dark".to_string()), None).await?;
    repo.set_setting("app.language", SettingValue::String("en".to_string()), None).await?;
    repo.set_setting("system.debug", SettingValue::Boolean(true), None).await?;

    // List all settings
    let all = repo.list_settings(None).await?;
    assert!(all.len() >= 3);

    // List with prefix
    let app_settings = repo.list_settings(Some("app")).await?;
    assert!(app_settings.len() >= 2);
    assert!(app_settings.contains(&"app.theme".to_string()));
    assert!(app_settings.contains(&"app.language".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_cache_behavior() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Set a value
    repo.set_setting(
        "cache.test",
        SettingValue::String("initial".to_string()),
        None,
    )
    .await?;

    // First read - should cache
    let value1 = repo.get_setting("cache.test").await?;
    assert_eq!(value1.unwrap().as_string(), Some("initial"));

    // Second read - should hit cache
    let value2 = repo.get_setting("cache.test").await?;
    assert_eq!(value2.unwrap().as_string(), Some("initial"));

    // Clear cache
    repo.clear_cache().await?;

    // Read after cache clear
    let value3 = repo.get_setting("cache.test").await?;
    assert_eq!(value3.unwrap().as_string(), Some("initial"));

    Ok(())
}

#[tokio::test]
async fn test_physics_settings() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    let settings = PhysicsSettings {
        enabled: true,
        physics_type: "force_directed".to_string(),
        iterations_per_frame: 5,
        target_fps: 60.0,
        damping: 0.95,
        repulsion_strength: 2000.0,
        attraction_strength: 0.1,
        center_gravity: 0.01,
        edge_weight_influence: 1.0,
        boundary_box_size: 5000.0,
        boundary_type: "soft".to_string(),
        time_step: 0.016,
        min_velocity_threshold: 0.01,
        use_gpu_acceleration: true,
    };

    // Save physics settings
    repo.save_physics_settings("test_profile", &settings).await?;

    // List profiles
    let profiles = repo.list_physics_profiles().await?;
    assert!(profiles.contains(&"test_profile".to_string()));

    // Get physics settings
    let loaded = repo.get_physics_settings("test_profile").await?;
    assert_eq!(loaded.enabled, settings.enabled);
    assert_eq!(loaded.physics_type, settings.physics_type);
    assert_eq!(loaded.repulsion_strength, settings.repulsion_strength);

    // Delete profile
    repo.delete_physics_profile("test_profile").await?;
    let profiles_after = repo.list_physics_profiles().await?;
    assert!(!profiles_after.contains(&"test_profile".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_export_import_settings() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Create some settings
    repo.set_setting("export.test1", SettingValue::String("value1".to_string()), None).await?;
    repo.set_setting("export.test2", SettingValue::Integer(42), None).await?;
    repo.set_setting("export.test3", SettingValue::Boolean(true), None).await?;

    // Export settings
    let exported = repo.export_settings().await?;
    assert!(exported.is_object());

    // Clear and import
    repo.delete_setting("export.test1").await?;
    repo.delete_setting("export.test2").await?;
    repo.delete_setting("export.test3").await?;

    repo.import_settings(&exported).await?;

    // Verify imported settings
    assert_eq!(
        repo.get_setting("export.test1").await?.unwrap().as_string(),
        Some("value1")
    );
    assert_eq!(
        repo.get_setting("export.test2").await?.unwrap().as_i64(),
        Some(42)
    );
    assert_eq!(
        repo.get_setting("export.test3").await?.unwrap().as_bool(),
        Some(true)
    );

    Ok(())
}

#[tokio::test]
async fn test_app_full_settings() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Load default settings
    let settings = repo.load_all_settings().await?;
    assert!(settings.is_some());

    let loaded = settings.unwrap();
    assert_eq!(loaded.version, "1.0.0");

    // Save settings (currently a stub)
    repo.save_all_settings(&loaded).await?;

    Ok(())
}

#[tokio::test]
async fn test_health_check() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    let healthy = repo.health_check().await?;
    assert!(healthy);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_access() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Spawn multiple concurrent operations
    let repo_clone1 = repo.clone();
    let repo_clone2 = repo.clone();
    let repo_clone3 = repo.clone();

    let handle1 = tokio::spawn(async move {
        for i in 0..10 {
            repo_clone1
                .set_setting(
                    &format!("concurrent.1.{}", i),
                    SettingValue::Integer(i as i64),
                    None,
                )
                .await
                .unwrap();
        }
    });

    let handle2 = tokio::spawn(async move {
        for i in 0..10 {
            repo_clone2
                .set_setting(
                    &format!("concurrent.2.{}", i),
                    SettingValue::Integer(i as i64),
                    None,
                )
                .await
                .unwrap();
        }
    });

    let handle3 = tokio::spawn(async move {
        for i in 0..10 {
            let _ = repo_clone3.get_setting(&format!("concurrent.1.{}", i)).await;
        }
    });

    handle1.await?;
    handle2.await?;
    handle3.await?;

    // Verify all settings were written
    let all_settings = repo.list_settings(Some("concurrent")).await?;
    assert!(all_settings.len() >= 20);

    Ok(())
}

#[tokio::test]
async fn test_json_setting_value() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    let json_value = json!({
        "nested": {
            "key": "value",
            "array": [1, 2, 3]
        }
    });

    repo.set_setting(
        "test.json",
        SettingValue::Json(json_value.clone()),
        None,
    )
    .await?;

    let loaded = repo.get_setting("test.json").await?;
    assert!(loaded.is_some());

    if let Some(SettingValue::Json(loaded_json)) = loaded {
        assert_eq!(loaded_json, json_value);
    } else {
        panic!("Expected JSON value");
    }

    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Test getting nonexistent setting
    let result = repo.get_setting("nonexistent.key").await?;
    assert!(result.is_none());

    // Test deleting nonexistent setting (should succeed silently)
    let result = repo.delete_setting("nonexistent.key").await;
    assert!(result.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_cache_invalidation() -> Result<()> {
    let (_temp, repo) = setup_test_db().await?;

    // Set initial value
    repo.set_setting(
        "invalidate.test",
        SettingValue::String("value1".to_string()),
        None,
    )
    .await?;

    // Read to populate cache
    let value1 = repo.get_setting("invalidate.test").await?;
    assert_eq!(value1.unwrap().as_string(), Some("value1"));

    // Update value (should invalidate cache)
    repo.set_setting(
        "invalidate.test",
        SettingValue::String("value2".to_string()),
        None,
    )
    .await?;

    // Read again - should get updated value
    let value2 = repo.get_setting("invalidate.test").await?;
    assert_eq!(value2.unwrap().as_string(), Some("value2"));

    // Delete (should invalidate cache)
    repo.delete_setting("invalidate.test").await?;

    // Read after delete
    let value3 = repo.get_setting("invalidate.test").await?;
    assert!(value3.is_none());

    Ok(())
}
