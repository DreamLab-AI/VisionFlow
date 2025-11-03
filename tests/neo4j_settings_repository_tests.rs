// tests/neo4j_settings_repository_tests.rs
//! Comprehensive integration tests for Neo4jSettingsRepository
//!
//! These tests require a running Neo4j instance.
//! Run with: cargo test --features neo4j --test neo4j_settings_repository_tests

#![cfg(feature = "neo4j")]

use std::collections::HashMap;
use webxr::adapters::neo4j_settings_repository::{Neo4jSettingsRepository, Neo4jSettingsConfig};
use webxr::config::PhysicsSettings;
use webxr::ports::settings_repository::{SettingsRepository, SettingValue};

/// Test helper to create a unique repository instance
async fn create_test_repo() -> Neo4jSettingsRepository {
    let config = Neo4jSettingsConfig {
        uri: std::env::var("NEO4J_TEST_URI")
            .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
        user: std::env::var("NEO4J_TEST_USER")
            .unwrap_or_else(|_| "neo4j".to_string()),
        password: std::env::var("NEO4J_TEST_PASSWORD")
            .unwrap_or_else(|_| "password".to_string()),
        database: Some("test".to_string()),
        fetch_size: 100,
        max_connections: 5,
    };

    Neo4jSettingsRepository::new(config)
        .await
        .expect("Failed to create test repository")
}

/// Clean up test data after tests
async fn cleanup_test_repo(repo: &Neo4jSettingsRepository, prefix: &str) {
    let keys = repo.list_settings(Some(prefix)).await.unwrap();
    for key in keys {
        let _ = repo.delete_setting(&key).await;
    }
}

#[tokio::test]
#[ignore] // Requires Neo4j instance
async fn test_connection_and_health_check() {
    let repo = create_test_repo().await;

    // Test health check
    let healthy = repo.health_check().await.expect("Health check failed");
    assert!(healthy, "Repository should be healthy");
}

#[tokio::test]
#[ignore]
async fn test_set_and_get_string_setting() {
    let repo = create_test_repo().await;
    let test_key = "test.string.key";

    // Set a string setting
    repo.set_setting(
        test_key,
        SettingValue::String("test_value".to_string()),
        Some("Test string setting"),
    )
    .await
    .expect("Failed to set setting");

    // Get the setting
    let value = repo.get_setting(test_key).await.expect("Failed to get setting");
    assert_eq!(
        value,
        Some(SettingValue::String("test_value".to_string())),
        "Retrieved value should match"
    );

    // Clean up
    cleanup_test_repo(&repo, "test.string").await;
}

#[tokio::test]
#[ignore]
async fn test_set_and_get_integer_setting() {
    let repo = create_test_repo().await;
    let test_key = "test.integer.key";

    repo.set_setting(test_key, SettingValue::Integer(42), Some("Test integer"))
        .await
        .expect("Failed to set integer");

    let value = repo.get_setting(test_key).await.expect("Failed to get integer");
    assert_eq!(value, Some(SettingValue::Integer(42)));

    cleanup_test_repo(&repo, "test.integer").await;
}

#[tokio::test]
#[ignore]
async fn test_set_and_get_float_setting() {
    let repo = create_test_repo().await;
    let test_key = "test.float.key";

    repo.set_setting(test_key, SettingValue::Float(3.14159), Some("Test float"))
        .await
        .expect("Failed to set float");

    let value = repo.get_setting(test_key).await.expect("Failed to get float");
    assert_eq!(value, Some(SettingValue::Float(3.14159)));

    cleanup_test_repo(&repo, "test.float").await;
}

#[tokio::test]
#[ignore]
async fn test_set_and_get_boolean_setting() {
    let repo = create_test_repo().await;
    let test_key = "test.boolean.key";

    repo.set_setting(test_key, SettingValue::Boolean(true), Some("Test boolean"))
        .await
        .expect("Failed to set boolean");

    let value = repo.get_setting(test_key).await.expect("Failed to get boolean");
    assert_eq!(value, Some(SettingValue::Boolean(true)));

    cleanup_test_repo(&repo, "test.boolean").await;
}

#[tokio::test]
#[ignore]
async fn test_set_and_get_json_setting() {
    let repo = create_test_repo().await;
    let test_key = "test.json.key";

    let json_value = serde_json::json!({
        "name": "test",
        "value": 42,
        "nested": {
            "key": "value"
        }
    });

    repo.set_setting(test_key, SettingValue::Json(json_value.clone()), Some("Test JSON"))
        .await
        .expect("Failed to set JSON");

    let value = repo.get_setting(test_key).await.expect("Failed to get JSON");
    assert_eq!(value, Some(SettingValue::Json(json_value)));

    cleanup_test_repo(&repo, "test.json").await;
}

#[tokio::test]
#[ignore]
async fn test_update_setting() {
    let repo = create_test_repo().await;
    let test_key = "test.update.key";

    // Set initial value
    repo.set_setting(test_key, SettingValue::String("initial".to_string()), None)
        .await
        .expect("Failed to set initial value");

    // Update value
    repo.set_setting(test_key, SettingValue::String("updated".to_string()), None)
        .await
        .expect("Failed to update value");

    // Verify update
    let value = repo.get_setting(test_key).await.expect("Failed to get updated value");
    assert_eq!(value, Some(SettingValue::String("updated".to_string())));

    cleanup_test_repo(&repo, "test.update").await;
}

#[tokio::test]
#[ignore]
async fn test_delete_setting() {
    let repo = create_test_repo().await;
    let test_key = "test.delete.key";

    // Set a setting
    repo.set_setting(test_key, SettingValue::String("to_delete".to_string()), None)
        .await
        .expect("Failed to set setting");

    // Verify it exists
    assert!(repo.has_setting(test_key).await.unwrap());

    // Delete it
    repo.delete_setting(test_key).await.expect("Failed to delete setting");

    // Verify deletion
    let value = repo.get_setting(test_key).await.expect("Failed to get deleted setting");
    assert_eq!(value, None, "Setting should be deleted");
    assert!(!repo.has_setting(test_key).await.unwrap());
}

#[tokio::test]
#[ignore]
async fn test_batch_operations() {
    let repo = create_test_repo().await;

    // Create batch of settings
    let mut updates = HashMap::new();
    updates.insert("test.batch.1".to_string(), SettingValue::String("value1".to_string()));
    updates.insert("test.batch.2".to_string(), SettingValue::Integer(2));
    updates.insert("test.batch.3".to_string(), SettingValue::Boolean(true));

    // Set batch
    repo.set_settings_batch(updates.clone())
        .await
        .expect("Failed to set batch");

    // Get batch
    let keys: Vec<String> = updates.keys().cloned().collect();
    let retrieved = repo.get_settings_batch(&keys)
        .await
        .expect("Failed to get batch");

    // Verify all settings retrieved
    assert_eq!(retrieved.len(), 3, "Should retrieve all batch settings");
    assert_eq!(retrieved.get("test.batch.1"), Some(&SettingValue::String("value1".to_string())));
    assert_eq!(retrieved.get("test.batch.2"), Some(&SettingValue::Integer(2)));
    assert_eq!(retrieved.get("test.batch.3"), Some(&SettingValue::Boolean(true)));

    cleanup_test_repo(&repo, "test.batch").await;
}

#[tokio::test]
#[ignore]
async fn test_list_settings_with_prefix() {
    let repo = create_test_repo().await;

    // Create multiple settings with same prefix
    repo.set_setting("test.prefix.alpha", SettingValue::String("a".to_string()), None).await.unwrap();
    repo.set_setting("test.prefix.beta", SettingValue::String("b".to_string()), None).await.unwrap();
    repo.set_setting("test.prefix.gamma", SettingValue::String("c".to_string()), None).await.unwrap();
    repo.set_setting("test.other.delta", SettingValue::String("d".to_string()), None).await.unwrap();

    // List with prefix
    let keys = repo.list_settings(Some("test.prefix"))
        .await
        .expect("Failed to list settings");

    // Verify results
    assert_eq!(keys.len(), 3, "Should find 3 settings with prefix");
    assert!(keys.contains(&"test.prefix.alpha".to_string()));
    assert!(keys.contains(&"test.prefix.beta".to_string()));
    assert!(keys.contains(&"test.prefix.gamma".to_string()));
    assert!(!keys.contains(&"test.other.delta".to_string()));

    cleanup_test_repo(&repo, "test.prefix").await;
    cleanup_test_repo(&repo, "test.other").await;
}

#[tokio::test]
#[ignore]
async fn test_list_all_settings() {
    let repo = create_test_repo().await;

    // Create test settings
    repo.set_setting("test.all.1", SettingValue::String("1".to_string()), None).await.unwrap();
    repo.set_setting("test.all.2", SettingValue::String("2".to_string()), None).await.unwrap();

    // List all
    let keys = repo.list_settings(None).await.expect("Failed to list all settings");

    // Should include our test settings
    assert!(keys.contains(&"test.all.1".to_string()));
    assert!(keys.contains(&"test.all.2".to_string()));

    cleanup_test_repo(&repo, "test.all").await;
}

#[tokio::test]
#[ignore]
async fn test_physics_settings() {
    let repo = create_test_repo().await;
    let profile_name = "test_physics_profile";

    // Create test physics settings
    let mut physics = PhysicsSettings::default();
    physics.damping = 0.95;
    physics.spring_k = 0.02;

    // Save physics settings
    repo.save_physics_settings(profile_name, &physics)
        .await
        .expect("Failed to save physics settings");

    // Load physics settings
    let loaded = repo.get_physics_settings(profile_name)
        .await
        .expect("Failed to load physics settings");

    // Verify
    assert_eq!(loaded.damping, 0.95);
    assert_eq!(loaded.spring_k, 0.02);

    // List physics profiles
    let profiles = repo.list_physics_profiles()
        .await
        .expect("Failed to list physics profiles");
    assert!(profiles.contains(&profile_name.to_string()));

    // Delete profile
    repo.delete_physics_profile(profile_name)
        .await
        .expect("Failed to delete physics profile");

    // Verify deletion
    let profiles = repo.list_physics_profiles().await.unwrap();
    assert!(!profiles.contains(&profile_name.to_string()));
}

#[tokio::test]
#[ignore]
async fn test_cache_functionality() {
    let repo = create_test_repo().await;
    let test_key = "test.cache.key";

    // Set a setting
    repo.set_setting(test_key, SettingValue::String("cached_value".to_string()), None)
        .await
        .expect("Failed to set setting");

    // First read (cache miss)
    let value1 = repo.get_setting(test_key).await.unwrap();

    // Second read (cache hit - should be faster)
    let value2 = repo.get_setting(test_key).await.unwrap();

    assert_eq!(value1, value2);
    assert_eq!(value1, Some(SettingValue::String("cached_value".to_string())));

    // Clear cache
    repo.clear_cache().await.expect("Failed to clear cache");

    // Third read (cache miss again after clear)
    let value3 = repo.get_setting(test_key).await.unwrap();
    assert_eq!(value1, value3);

    cleanup_test_repo(&repo, "test.cache").await;
}

#[tokio::test]
#[ignore]
async fn test_export_import_settings() {
    let repo = create_test_repo().await;

    // Create test settings
    repo.set_setting("test.export.a", SettingValue::String("alpha".to_string()), Some("First")).await.unwrap();
    repo.set_setting("test.export.b", SettingValue::Integer(42), Some("Second")).await.unwrap();

    // Export settings
    let exported = repo.export_settings()
        .await
        .expect("Failed to export settings");

    // Verify export contains our settings
    assert!(exported.get("test.export.a").is_some());
    assert!(exported.get("test.export.b").is_some());

    // Delete original settings
    repo.delete_setting("test.export.a").await.unwrap();
    repo.delete_setting("test.export.b").await.unwrap();

    // Import settings
    repo.import_settings(&exported)
        .await
        .expect("Failed to import settings");

    // Verify import
    let value_a = repo.get_setting("test.export.a").await.unwrap();
    let value_b = repo.get_setting("test.export.b").await.unwrap();

    assert_eq!(value_a, Some(SettingValue::String("alpha".to_string())));
    assert_eq!(value_b, Some(SettingValue::Integer(42)));

    cleanup_test_repo(&repo, "test.export").await;
}

#[tokio::test]
#[ignore]
async fn test_concurrent_access() {
    let repo = create_test_repo().await;

    // Spawn multiple concurrent tasks
    let mut handles = vec![];

    for i in 0..10 {
        let repo_clone = create_test_repo().await;
        let handle = tokio::spawn(async move {
            let key = format!("test.concurrent.{}", i);
            repo_clone.set_setting(&key, SettingValue::Integer(i as i64), None).await.unwrap();
            let value = repo_clone.get_setting(&key).await.unwrap();
            assert_eq!(value, Some(SettingValue::Integer(i as i64)));
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.expect("Task failed");
    }

    cleanup_test_repo(&repo, "test.concurrent").await;
}

#[tokio::test]
#[ignore]
async fn test_error_handling_invalid_key() {
    let repo = create_test_repo().await;

    // Try to get non-existent setting
    let value = repo.get_setting("non.existent.key").await.unwrap();
    assert_eq!(value, None, "Non-existent setting should return None");
}

#[tokio::test]
#[ignore]
async fn test_performance_batch_vs_individual() {
    let repo = create_test_repo().await;

    // Create test data
    let mut test_data = HashMap::new();
    for i in 0..100 {
        test_data.insert(
            format!("test.perf.{}", i),
            SettingValue::String(format!("value_{}", i)),
        );
    }

    // Batch insert
    let batch_start = std::time::Instant::now();
    repo.set_settings_batch(test_data.clone()).await.unwrap();
    let batch_duration = batch_start.elapsed();

    println!("Batch insert of 100 settings: {:?}", batch_duration);

    // Individual delete (for comparison)
    let individual_start = std::time::Instant::now();
    for key in test_data.keys() {
        repo.delete_setting(key).await.unwrap();
    }
    let individual_duration = individual_start.elapsed();

    println!("Individual delete of 100 settings: {:?}", individual_duration);

    // Batch should be significantly faster
    // (Note: This may not always be true for deletes, but demonstrates the pattern)

    cleanup_test_repo(&repo, "test.perf").await;
}
