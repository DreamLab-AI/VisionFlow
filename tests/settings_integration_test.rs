// tests/settings_integration_test.rs
//! Integration tests for settings persistence
//!
//! NOTE: This test file is DISABLED - it was written for the old SQLite-based
//! settings architecture (turbo_flow_control crate). The project has since:
//! 1. Been renamed from turbo_flow_control to webxr
//! 2. Migrated from SQLite to Neo4j for persistence
//! 3. Changed the SettingsRepository to use the port-based Neo4j adapter
//!
//! These tests would need to be rewritten to:
//! - Use the webxr crate
//! - Use Neo4j test containers or mocked repositories
//! - Match the new SettingsRepository port interface
//!
//! The settings actor and messages (UpdatePhysicsSettings, GetPhysicsSettings, etc.)
//! still exist and work, but they now use Neo4j-backed persistence via
//! src/adapters/neo4j_settings_repository.rs

// DISABLED: Old SQLite-based tests incompatible with Neo4j architecture
// TODO: Rewrite using Neo4j test containers if integration tests are needed
/*
use actix::Actor;
use sqlx::SqlitePool;
use std::sync::Arc;

// Import from the main crate
use webxr::{
    config::{PhysicsSettings, RenderingSettings},
    settings::{
        SettingsActor,
        UpdatePhysicsSettings, GetPhysicsSettings,
        UpdateConstraintSettings, GetConstraintSettings,
        SaveProfile, LoadProfile, ListProfiles, DeleteProfile,
        ConstraintSettings, PriorityWeighting,
    },
};

async fn setup_test_db() -> SqlitePool {
    // Create in-memory SQLite database for testing
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create test database");

    // Run migrations
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS physics_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            settings_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS constraint_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            settings_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rendering_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            settings_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            physics_json TEXT NOT NULL,
            constraints_json TEXT NOT NULL,
            rendering_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to run migrations");

    pool
}

#[actix_rt::test]
async fn test_physics_settings_persistence() {
    let pool = setup_test_db().await;
    let repository = Arc::new(SettingsRepository::new(pool));
    let actor = SettingsActor::new(repository.clone()).start();

    // Create custom physics settings
    let mut physics = PhysicsSettings::default();
    physics.damping = 0.85;
    physics.spring_k = 0.01;
    physics.repel_k = 75.0;

    // Update settings
    actor
        .send(UpdatePhysicsSettings(physics.clone()))
        .await
        .expect("Failed to send message")
        .expect("Failed to update settings");

    // Retrieve settings
    let retrieved = actor
        .send(GetPhysicsSettings)
        .await
        .expect("Failed to get settings");

    assert_eq!(retrieved.damping, 0.85);
    assert_eq!(retrieved.spring_k, 0.01);
    assert_eq!(retrieved.repel_k, 75.0);

    // Verify persistence by loading directly from repository
    let persisted = repository
        .load_physics_settings()
        .await
        .expect("Failed to load from repository");

    assert_eq!(persisted.damping, 0.85);
}

#[actix_rt::test]
async fn test_constraint_settings_persistence() {
    let pool = setup_test_db().await;
    let repository = Arc::new(SettingsRepository::new(pool));
    let actor = SettingsActor::new(repository.clone()).start();

    // Create custom constraint settings
    let constraints = ConstraintSettings {
        lod_enabled: true,
        far_threshold: 500.0,
        medium_threshold: 50.0,
        near_threshold: 5.0,
        priority_weighting: PriorityWeighting::Quadratic,
        progressive_activation: false,
        activation_frames: 30,
    };

    // Update settings
    actor
        .send(UpdateConstraintSettings(constraints.clone()))
        .await
        .expect("Failed to send message")
        .expect("Failed to update settings");

    // Retrieve settings
    let retrieved = actor
        .send(GetConstraintSettings)
        .await
        .expect("Failed to get settings");

    assert_eq!(retrieved.far_threshold, 500.0);
    assert_eq!(retrieved.medium_threshold, 50.0);
    assert_eq!(retrieved.priority_weighting, PriorityWeighting::Quadratic);
    assert!(!retrieved.progressive_activation);
}

#[actix_rt::test]
async fn test_profile_management() {
    let pool = setup_test_db().await;
    let repository = Arc::new(SettingsRepository::new(pool));
    let actor = SettingsActor::new(repository.clone()).start();

    // Set up custom settings
    let mut physics = PhysicsSettings::default();
    physics.damping = 0.8;

    let constraints = ConstraintSettings {
        far_threshold: 2000.0,
        ..Default::default()
    };

    actor
        .send(UpdatePhysicsSettings(physics))
        .await
        .unwrap()
        .unwrap();

    actor
        .send(UpdateConstraintSettings(constraints))
        .await
        .unwrap()
        .unwrap();

    // Save as profile
    let profile_id = actor
        .send(SaveProfile {
            name: "test_profile".to_string(),
        })
        .await
        .expect("Failed to send message")
        .expect("Failed to save profile");

    assert!(profile_id > 0);

    // Load profile
    let loaded = actor
        .send(LoadProfile(profile_id))
        .await
        .expect("Failed to send message")
        .expect("Failed to load profile");

    assert_eq!(loaded.physics.damping, 0.8);
    assert_eq!(loaded.constraints.far_threshold, 2000.0);

    // List profiles
    let profiles = actor
        .send(ListProfiles)
        .await
        .expect("Failed to send message")
        .expect("Failed to list profiles");

    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].name, "test_profile");

    // Delete profile
    actor
        .send(DeleteProfile(profile_id))
        .await
        .expect("Failed to send message")
        .expect("Failed to delete profile");

    // Verify deletion
    let profiles = actor
        .send(ListProfiles)
        .await
        .expect("Failed to send message")
        .expect("Failed to list profiles");

    assert_eq!(profiles.len(), 0);
}

#[actix_rt::test]
async fn test_default_settings_on_empty_database() {
    let pool = setup_test_db().await;
    let repository = Arc::new(SettingsRepository::new(pool));

    // Load physics settings from empty database - should return defaults
    let physics = repository
        .load_physics_settings()
        .await
        .expect("Failed to load default physics settings");

    assert_eq!(physics.damping, PhysicsSettings::default().damping);

    // Load constraint settings from empty database - should return defaults
    let constraints = repository
        .load_constraint_settings()
        .await
        .expect("Failed to load default constraint settings");

    assert_eq!(constraints.lod_enabled, true);
    assert_eq!(constraints.far_threshold, 1000.0);
}

#[actix_rt::test]
async fn test_all_settings_save_and_load() {
    let pool = setup_test_db().await;
    let repository = Arc::new(SettingsRepository::new(pool));

    let mut physics = PhysicsSettings::default();
    physics.damping = 0.92;

    let constraints = ConstraintSettings {
        far_threshold: 1500.0,
        ..Default::default()
    };

    let mut rendering = RenderingSettings::default();
    rendering.ambient_light_intensity = 0.7;

    let all_settings = webxr::settings::AllSettings {
        physics,
        constraints,
        rendering,
    };

    // Save all settings
    repository
        .save_all_settings(&all_settings)
        .await
        .expect("Failed to save all settings");

    // Load all settings
    let loaded = repository
        .load_all_settings()
        .await
        .expect("Failed to load all settings");

    assert_eq!(loaded.physics.damping, 0.92);
    assert_eq!(loaded.constraints.far_threshold, 1500.0);
    assert_eq!(loaded.rendering.ambient_light_intensity, 0.7);
}
*/
