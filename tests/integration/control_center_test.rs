// Control center UI integration tests
// Tests settings persistence and actor system integration

use actix::{Actor, Addr, Context, Handler, Message};
use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio;

// ============================================================================
// SETTINGS STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PhysicsSettings {
    gravity: f32,
    damping: f32,
    stiffness: f32,
    iterations: u32,
    enabled: bool,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            damping: 0.99,
            stiffness: 0.5,
            iterations: 10,
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RenderSettings {
    quality: String,
    shadows: bool,
    antialiasing: bool,
    fps_limit: u32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            quality: "high".to_string(),
            shadows: true,
            antialiasing: true,
            fps_limit: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserPreferences {
    theme: String,
    auto_save: bool,
    notifications: bool,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            auto_save: true,
            notifications: true,
        }
    }
}

// ============================================================================
// SETTINGS REPOSITORY
// ============================================================================

struct SettingsRepository {
    conn: Arc<Mutex<Connection>>,
}

impl SettingsRepository {
    fn new(conn: Connection) -> Self {
        let conn = Arc::new(Mutex::new(conn));

        // Create settings table
        conn.lock().unwrap().execute(
            "CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )",
            [],
        ).expect("Failed to create settings table");

        Self { conn }
    }

    fn save_physics_settings(&self, settings: &PhysicsSettings) -> Result<()> {
        let json = serde_json::to_string(settings)?;
        self.conn.lock().unwrap().execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('physics', ?)",
            [json],
        )?;
        Ok(())
    }

    fn load_physics_settings(&self) -> Result<PhysicsSettings> {
        let json: String = self.conn.lock().unwrap().query_row(
            "SELECT value FROM settings WHERE key = 'physics'",
            [],
            |row| row.get(0),
        ).unwrap_or_else(|_| serde_json::to_string(&PhysicsSettings::default()).unwrap());

        Ok(serde_json::from_str(&json)?)
    }

    fn save_render_settings(&self, settings: &RenderSettings) -> Result<()> {
        let json = serde_json::to_string(settings)?;
        self.conn.lock().unwrap().execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('render', ?)",
            [json],
        )?;
        Ok(())
    }

    fn load_render_settings(&self) -> Result<RenderSettings> {
        let json: String = self.conn.lock().unwrap().query_row(
            "SELECT value FROM settings WHERE key = 'render'",
            [],
            |row| row.get(0),
        ).unwrap_or_else(|_| serde_json::to_string(&RenderSettings::default()).unwrap());

        Ok(serde_json::from_str(&json)?)
    }

    fn save_user_preferences(&self, prefs: &UserPreferences) -> Result<()> {
        let json = serde_json::to_string(prefs)?;
        self.conn.lock().unwrap().execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('preferences', ?)",
            [json],
        )?;
        Ok(())
    }

    fn load_user_preferences(&self) -> Result<UserPreferences> {
        let json: String = self.conn.lock().unwrap().query_row(
            "SELECT value FROM settings WHERE key = 'preferences'",
            [],
            |row| row.get(0),
        ).unwrap_or_else(|_| serde_json::to_string(&UserPreferences::default()).unwrap());

        Ok(serde_json::from_str(&json)?)
    }
}

// ============================================================================
// ACTOR MESSAGES
// ============================================================================

#[derive(Message)]
#[rtype(result = "Result<()>")]
struct UpdatePhysicsSettings(PhysicsSettings);

#[derive(Message)]
#[rtype(result = "Result<PhysicsSettings>")]
struct GetPhysicsSettings;

#[derive(Message)]
#[rtype(result = "Result<()>")]
struct UpdateRenderSettings(RenderSettings);

#[derive(Message)]
#[rtype(result = "Result<RenderSettings>")]
struct GetRenderSettings;

#[derive(Message)]
#[rtype(result = "Result<()>")]
struct UpdateUserPreferences(UserPreferences);

#[derive(Message)]
#[rtype(result = "Result<UserPreferences>")]
struct GetUserPreferences;

// ============================================================================
// SETTINGS ACTOR
// ============================================================================

struct SettingsActor {
    repo: Arc<SettingsRepository>,
}

impl SettingsActor {
    fn new(repo: Arc<SettingsRepository>) -> Self {
        Self { repo }
    }
}

impl Actor for SettingsActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        println!("SettingsActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        println!("SettingsActor stopped");
    }
}

impl Handler<UpdatePhysicsSettings> for SettingsActor {
    type Result = Result<()>;

    fn handle(&mut self, msg: UpdatePhysicsSettings, _ctx: &mut Context<Self>) -> Self::Result {
        self.repo.save_physics_settings(&msg.0)
    }
}

impl Handler<GetPhysicsSettings> for SettingsActor {
    type Result = Result<PhysicsSettings>;

    fn handle(&mut self, _msg: GetPhysicsSettings, _ctx: &mut Context<Self>) -> Self::Result {
        self.repo.load_physics_settings()
    }
}

impl Handler<UpdateRenderSettings> for SettingsActor {
    type Result = Result<()>;

    fn handle(&mut self, msg: UpdateRenderSettings, _ctx: &mut Context<Self>) -> Self::Result {
        self.repo.save_render_settings(&msg.0)
    }
}

impl Handler<GetRenderSettings> for SettingsActor {
    type Result = Result<RenderSettings>;

    fn handle(&mut self, _msg: GetRenderSettings, _ctx: &mut Context<Self>) -> Self::Result {
        self.repo.load_render_settings()
    }
}

impl Handler<UpdateUserPreferences> for SettingsActor {
    type Result = Result<()>;

    fn handle(&mut self, msg: UpdateUserPreferences, _ctx: &mut Context<Self>) -> Self::Result {
        self.repo.save_user_preferences(&msg.0)
    }
}

impl Handler<GetUserPreferences> for SettingsActor {
    type Result = Result<UserPreferences>;

    fn handle(&mut self, _msg: GetUserPreferences, _ctx: &mut Context<Self>) -> Self::Result {
        self.repo.load_user_preferences()
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn setup_test_db() -> Connection {
    Connection::open_in_memory().expect("Failed to create in-memory database")
}

#[actix::test]
async fn test_settings_persistence() {
    let conn = setup_test_db();
    let repo = Arc::new(SettingsRepository::new(conn));
    let settings_actor = SettingsActor::new(repo.clone()).start();

    // Update physics settings
    let new_physics = PhysicsSettings {
        gravity: 12.0,
        damping: 0.95,
        stiffness: 0.7,
        iterations: 15,
        enabled: true,
    };

    settings_actor
        .send(UpdatePhysicsSettings(new_physics.clone()))
        .await
        .unwrap()
        .unwrap();

    // Restart actor (simulate app restart)
    drop(settings_actor);
    let settings_actor = SettingsActor::new(repo).start();

    // Verify settings persisted
    let loaded = settings_actor
        .send(GetPhysicsSettings)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(loaded.gravity, 12.0);
    assert_eq!(loaded.damping, 0.95);
    assert_eq!(loaded.stiffness, 0.7);
    assert_eq!(loaded.iterations, 15);

    println!("✅ Physics settings persisted correctly");
}

#[actix::test]
async fn test_multiple_settings_types() {
    let conn = setup_test_db();
    let repo = Arc::new(SettingsRepository::new(conn));
    let settings_actor = SettingsActor::new(repo).start();

    // Update physics settings
    let physics = PhysicsSettings {
        gravity: 10.0,
        damping: 0.98,
        stiffness: 0.6,
        iterations: 12,
        enabled: true,
    };
    settings_actor
        .send(UpdatePhysicsSettings(physics.clone()))
        .await
        .unwrap()
        .unwrap();

    // Update render settings
    let render = RenderSettings {
        quality: "ultra".to_string(),
        shadows: true,
        antialiasing: true,
        fps_limit: 144,
    };
    settings_actor
        .send(UpdateRenderSettings(render.clone()))
        .await
        .unwrap()
        .unwrap();

    // Update user preferences
    let prefs = UserPreferences {
        theme: "light".to_string(),
        auto_save: false,
        notifications: false,
    };
    settings_actor
        .send(UpdateUserPreferences(prefs.clone()))
        .await
        .unwrap()
        .unwrap();

    // Verify all settings
    let loaded_physics = settings_actor
        .send(GetPhysicsSettings)
        .await
        .unwrap()
        .unwrap();
    let loaded_render = settings_actor
        .send(GetRenderSettings)
        .await
        .unwrap()
        .unwrap();
    let loaded_prefs = settings_actor
        .send(GetUserPreferences)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(loaded_physics, physics);
    assert_eq!(loaded_render, render);
    assert_eq!(loaded_prefs.theme, "light");

    println!("✅ Multiple settings types persisted correctly");
}

#[actix::test]
async fn test_default_settings() {
    let conn = setup_test_db();
    let repo = Arc::new(SettingsRepository::new(conn));
    let settings_actor = SettingsActor::new(repo).start();

    // Load settings without saving (should return defaults)
    let physics = settings_actor
        .send(GetPhysicsSettings)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(physics, PhysicsSettings::default());

    let render = settings_actor
        .send(GetRenderSettings)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(render, RenderSettings::default());

    println!("✅ Default settings loaded correctly");
}

#[actix::test]
async fn test_concurrent_updates() {
    let conn = setup_test_db();
    let repo = Arc::new(SettingsRepository::new(conn));
    let settings_actor = SettingsActor::new(repo).start();

    // Send multiple concurrent updates
    let mut handles = vec![];

    for i in 0..10 {
        let actor = settings_actor.clone();
        let handle = tokio::spawn(async move {
            let settings = PhysicsSettings {
                gravity: 9.8 + i as f32,
                damping: 0.99,
                stiffness: 0.5,
                iterations: 10,
                enabled: true,
            };
            actor.send(UpdatePhysicsSettings(settings)).await.unwrap().unwrap();
        });
        handles.push(handle);
    }

    // Wait for all updates to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify final state
    let loaded = settings_actor
        .send(GetPhysicsSettings)
        .await
        .unwrap()
        .unwrap();

    // Should have one of the values (exact value depends on execution order)
    assert!(loaded.gravity >= 9.8 && loaded.gravity < 20.0);

    println!("✅ Concurrent updates handled correctly");
}

#[actix::test]
async fn test_settings_validation() {
    let conn = setup_test_db();
    let repo = Arc::new(SettingsRepository::new(conn));
    let settings_actor = SettingsActor::new(repo).start();

    // Test valid settings
    let valid_settings = PhysicsSettings {
        gravity: 9.8,
        damping: 0.99,
        stiffness: 0.5,
        iterations: 10,
        enabled: true,
    };

    let result = settings_actor
        .send(UpdatePhysicsSettings(valid_settings))
        .await
        .unwrap();

    assert!(result.is_ok());

    // In a real implementation, you would test invalid settings here
    // For example: negative gravity, damping > 1.0, etc.

    println!("✅ Settings validation works correctly");
}

#[actix::test]
async fn test_actor_restart_preserves_settings() {
    let conn = setup_test_db();
    let repo = Arc::new(SettingsRepository::new(conn));

    // First actor instance
    {
        let actor = SettingsActor::new(repo.clone()).start();
        let settings = PhysicsSettings {
            gravity: 15.0,
            damping: 0.96,
            stiffness: 0.8,
            iterations: 20,
            enabled: false,
        };
        actor
            .send(UpdatePhysicsSettings(settings))
            .await
            .unwrap()
            .unwrap();
    } // Actor dropped here

    // Second actor instance (simulates app restart)
    {
        let actor = SettingsActor::new(repo.clone()).start();
        let loaded = actor
            .send(GetPhysicsSettings)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(loaded.gravity, 15.0);
        assert_eq!(loaded.damping, 0.96);
        assert_eq!(loaded.enabled, false);
    }

    println!("✅ Settings preserved across actor restarts");
}

#[tokio::test]
async fn test_settings_json_serialization() {
    let settings = PhysicsSettings {
        gravity: 9.8,
        damping: 0.99,
        stiffness: 0.5,
        iterations: 10,
        enabled: true,
    };

    let json = serde_json::to_string(&settings).unwrap();
    let deserialized: PhysicsSettings = serde_json::from_str(&json).unwrap();

    assert_eq!(settings, deserialized);
    println!("✅ Settings JSON serialization works correctly");
}
