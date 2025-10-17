use rusqlite::{params, Connection, Result as SqliteResult};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::Utc;
use thiserror::Error;
use log::{debug, error, info, warn};

#[derive(Debug, Error)]
pub enum UserServiceError {
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("User not found")]
    UserNotFound,
    #[error("Invalid setting value")]
    InvalidSettingValue,
    #[error("Permission denied")]
    PermissionDenied,
    #[error("User already exists")]
    UserAlreadyExists,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub nostr_pubkey: String,
    pub username: Option<String>,
    pub is_power_user: bool,
    pub created_at: i64,
    pub last_seen: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(JsonValue),
}

impl SettingValue {
    fn value_type(&self) -> &'static str {
        match self {
            SettingValue::String(_) => "string",
            SettingValue::Integer(_) => "integer",
            SettingValue::Float(_) => "float",
            SettingValue::Boolean(_) => "boolean",
            SettingValue::Json(_) => "json",
        }
    }

    fn to_json(&self) -> String {
        match self {
            SettingValue::String(s) => s.clone(),
            SettingValue::Integer(i) => i.to_string(),
            SettingValue::Float(f) => f.to_string(),
            SettingValue::Boolean(b) => b.to_string(),
            SettingValue::Json(v) => serde_json::to_string(v).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSetting {
    pub id: i64,
    pub user_id: i64,
    pub key: String,
    pub value: SettingValue,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: i64,
    pub user_id: Option<i64>,
    pub key: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub action: String,
    pub timestamp: i64,
}

pub struct UserService {
    db_path: String,
    conn: Arc<RwLock<Connection>>,
}

impl UserService {
    pub async fn new(db_path: String) -> Result<Self, UserServiceError> {
        let conn = Connection::open(&db_path)
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        Ok(Self {
            db_path,
            conn: Arc::new(RwLock::new(conn)),
        })
    }

    pub async fn get_user_by_nostr_pubkey(&self, pubkey: &str) -> Result<User, UserServiceError> {
        let conn = self.conn.read().await;
        let mut stmt = conn
            .prepare("SELECT id, nostr_pubkey, username, is_power_user, strftime('%s', created_at) as created_at, strftime('%s', last_seen) as last_seen FROM users WHERE nostr_pubkey = ?1")
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let user = stmt
            .query_row(params![pubkey], |row| {
                Ok(User {
                    id: row.get(0)?,
                    nostr_pubkey: row.get(1)?,
                    username: row.get(2)?,
                    is_power_user: row.get::<_, i64>(3)? == 1,
                    created_at: row.get::<_, String>(4)?.parse().unwrap_or(0),
                    last_seen: row.get::<_, String>(5)?.parse().unwrap_or(0),
                })
            })
            .map_err(|_| UserServiceError::UserNotFound)?;

        Ok(user)
    }

    pub async fn get_user_by_id(&self, user_id: i64) -> Result<User, UserServiceError> {
        let conn = self.conn.read().await;
        let mut stmt = conn
            .prepare("SELECT id, nostr_pubkey, username, is_power_user, strftime('%s', created_at) as created_at, strftime('%s', last_seen) as last_seen FROM users WHERE id = ?1")
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let user = stmt
            .query_row(params![user_id], |row| {
                Ok(User {
                    id: row.get(0)?,
                    nostr_pubkey: row.get(1)?,
                    username: row.get(2)?,
                    is_power_user: row.get::<_, i64>(3)? == 1,
                    created_at: row.get::<_, String>(4)?.parse().unwrap_or(0),
                    last_seen: row.get::<_, String>(5)?.parse().unwrap_or(0),
                })
            })
            .map_err(|_| UserServiceError::UserNotFound)?;

        Ok(user)
    }

    pub async fn create_or_update_user(
        &self,
        pubkey: &str,
        username: Option<String>,
    ) -> Result<User, UserServiceError> {
        let mut conn = self.conn.write().await;
        let now = Utc::now().timestamp();

        let tx = conn.transaction()
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let result = tx.execute(
            "INSERT INTO users (nostr_pubkey, username, created_at, last_seen)
             VALUES (?1, ?2, datetime('now'), datetime('now'))
             ON CONFLICT(nostr_pubkey) DO UPDATE SET
             username = COALESCE(?2, username),
             last_seen = datetime('now')",
            params![pubkey, username],
        );

        match result {
            Ok(_) => {
                let user: User = tx.query_row(
                    "SELECT id, nostr_pubkey, username, is_power_user, strftime('%s', created_at) as created_at, strftime('%s', last_seen) as last_seen FROM users WHERE nostr_pubkey = ?1",
                    params![pubkey],
                    |row| {
                        Ok(User {
                            id: row.get(0)?,
                            nostr_pubkey: row.get(1)?,
                            username: row.get(2)?,
                            is_power_user: row.get::<_, i64>(3)? == 1,
                            created_at: row.get::<_, String>(4)?.parse().unwrap_or(0),
                            last_seen: row.get::<_, String>(5)?.parse().unwrap_or(0),
                        })
                    }
                ).map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

                tx.commit().map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;
                info!("Created/updated user: pubkey={}, id={}", pubkey, user.id);
                Ok(user)
            }
            Err(e) => {
                error!("Failed to create/update user {}: {}", pubkey, e);
                Err(UserServiceError::DatabaseError(e.to_string()))
            }
        }
    }

    pub async fn is_power_user(&self, user_id: i64) -> Result<bool, UserServiceError> {
        let conn = self.conn.read().await;
        let mut stmt = conn
            .prepare("SELECT is_power_user FROM users WHERE id = ?1")
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let is_power = stmt
            .query_row(params![user_id], |row| {
                let val: i64 = row.get(0)?;
                Ok(val == 1)
            })
            .map_err(|_| UserServiceError::UserNotFound)?;

        Ok(is_power)
    }

    pub async fn grant_power_user(&self, user_id: i64) -> Result<(), UserServiceError> {
        let mut conn = self.conn.write().await;
        conn.execute(
            "UPDATE users SET is_power_user = 1 WHERE id = ?1",
            params![user_id],
        )
        .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        info!("Granted power user to user_id={}", user_id);
        Ok(())
    }

    pub async fn revoke_power_user(&self, user_id: i64) -> Result<(), UserServiceError> {
        let mut conn = self.conn.write().await;
        conn.execute(
            "UPDATE users SET is_power_user = 0 WHERE id = ?1",
            params![user_id],
        )
        .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        info!("Revoked power user from user_id={}", user_id);
        Ok(())
    }

    pub async fn get_user_settings(&self, user_id: i64) -> Result<Vec<UserSetting>, UserServiceError> {
        let conn = self.conn.read().await;
        let mut stmt = conn
            .prepare(
                "SELECT id, user_id, key, value_type, value_text, value_integer, value_float, value_boolean, value_json,
                 strftime('%s', created_at) as created_at, strftime('%s', updated_at) as updated_at
                 FROM user_settings WHERE user_id = ?1"
            )
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let settings = stmt
            .query_map(params![user_id], |row| {
                let value_type: String = row.get(3)?;
                let value = match value_type.as_str() {
                    "string" => SettingValue::String(row.get(4)?),
                    "integer" => SettingValue::Integer(row.get(5)?),
                    "float" => SettingValue::Float(row.get(6)?),
                    "boolean" => SettingValue::Boolean(row.get::<_, i64>(7)? == 1),
                    "json" => {
                        let json_str: String = row.get(8)?;
                        let json_val: JsonValue = serde_json::from_str(&json_str).unwrap_or(JsonValue::Null);
                        SettingValue::Json(json_val)
                    }
                    _ => SettingValue::String(String::new()),
                };

                Ok(UserSetting {
                    id: row.get(0)?,
                    user_id: row.get(1)?,
                    key: row.get(2)?,
                    value,
                    created_at: row.get::<_, String>(9)?.parse().unwrap_or(0),
                    updated_at: row.get::<_, String>(10)?.parse().unwrap_or(0),
                })
            })
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        Ok(settings)
    }

    pub async fn set_user_setting(
        &self,
        user_id: i64,
        key: &str,
        value: SettingValue,
    ) -> Result<(), UserServiceError> {
        let mut conn = self.conn.write().await;
        let tx = conn.transaction()
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let old_value: Option<String> = tx
            .query_row(
                "SELECT value_text, value_integer, value_float, value_boolean, value_json, value_type FROM user_settings WHERE user_id = ?1 AND key = ?2",
                params![user_id, key],
                |row| {
                    let value_type: String = row.get(5)?;
                    let val = match value_type.as_str() {
                        "string" => row.get::<_, String>(0).ok(),
                        "integer" => row.get::<_, i64>(1).ok().map(|v| v.to_string()),
                        "float" => row.get::<_, f64>(2).ok().map(|v| v.to_string()),
                        "boolean" => row.get::<_, i64>(3).ok().map(|v| (v == 1).to_string()),
                        "json" => row.get::<_, String>(4).ok(),
                        _ => None,
                    };
                    Ok(val)
                }
            )
            .ok()
            .flatten();

        let (value_text, value_integer, value_float, value_boolean, value_json) = match &value {
            SettingValue::String(s) => (Some(s.clone()), None, None, None, None),
            SettingValue::Integer(i) => (None, Some(*i), None, None, None),
            SettingValue::Float(f) => (None, None, Some(*f), None, None),
            SettingValue::Boolean(b) => (None, None, None, Some(if *b { 1 } else { 0 }), None),
            SettingValue::Json(j) => (None, None, None, None, Some(serde_json::to_string(j).unwrap_or_default())),
        };

        let action = if old_value.is_some() { "update" } else { "create" };

        tx.execute(
            "INSERT INTO user_settings (user_id, key, value_type, value_text, value_integer, value_float, value_boolean, value_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, datetime('now'), datetime('now'))
             ON CONFLICT(user_id, key) DO UPDATE SET
             value_type = ?3,
             value_text = ?4,
             value_integer = ?5,
             value_float = ?6,
             value_boolean = ?7,
             value_json = ?8,
             updated_at = datetime('now')",
            params![user_id, key, value.value_type(), value_text, value_integer, value_float, value_boolean, value_json],
        ).map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        tx.execute(
            "INSERT INTO settings_audit_log (user_id, key, old_value, new_value, action, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))",
            params![user_id, key, old_value, value.to_json(), action],
        ).map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        tx.commit().map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        info!("Set user setting: user_id={}, key={}, action={}", user_id, key, action);
        Ok(())
    }

    pub async fn delete_user_setting(&self, user_id: i64, key: &str) -> Result<(), UserServiceError> {
        let mut conn = self.conn.write().await;
        let tx = conn.transaction()
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let old_value: Option<String> = tx
            .query_row(
                "SELECT value_text, value_integer, value_float, value_boolean, value_json, value_type FROM user_settings WHERE user_id = ?1 AND key = ?2",
                params![user_id, key],
                |row| {
                    let value_type: String = row.get(5)?;
                    let val = match value_type.as_str() {
                        "string" => row.get::<_, String>(0).ok(),
                        "integer" => row.get::<_, i64>(1).ok().map(|v| v.to_string()),
                        "float" => row.get::<_, f64>(2).ok().map(|v| v.to_string()),
                        "boolean" => row.get::<_, i64>(3).ok().map(|v| (v == 1).to_string()),
                        "json" => row.get::<_, String>(4).ok(),
                        _ => None,
                    };
                    Ok(val)
                }
            )
            .ok()
            .flatten();

        tx.execute(
            "DELETE FROM user_settings WHERE user_id = ?1 AND key = ?2",
            params![user_id, key],
        ).map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        tx.execute(
            "INSERT INTO settings_audit_log (user_id, key, old_value, new_value, action, timestamp)
             VALUES (?1, ?2, ?3, NULL, 'delete', datetime('now'))",
            params![user_id, key, old_value],
        ).map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        tx.commit().map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        info!("Deleted user setting: user_id={}, key={}", user_id, key);
        Ok(())
    }

    pub async fn get_audit_log(
        &self,
        key: Option<String>,
        user_id: Option<i64>,
        limit: i64,
    ) -> Result<Vec<AuditLogEntry>, UserServiceError> {
        let conn = self.conn.read().await;

        let query = match (key.as_ref(), user_id) {
            (Some(_), Some(_)) => {
                "SELECT id, user_id, key, old_value, new_value, action, strftime('%s', timestamp) as timestamp
                 FROM settings_audit_log WHERE key = ?1 AND user_id = ?2 ORDER BY timestamp DESC LIMIT ?3"
            }
            (Some(_), None) => {
                "SELECT id, user_id, key, old_value, new_value, action, strftime('%s', timestamp) as timestamp
                 FROM settings_audit_log WHERE key = ?1 ORDER BY timestamp DESC LIMIT ?2"
            }
            (None, Some(_)) => {
                "SELECT id, user_id, key, old_value, new_value, action, strftime('%s', timestamp) as timestamp
                 FROM settings_audit_log WHERE user_id = ?1 ORDER BY timestamp DESC LIMIT ?2"
            }
            (None, None) => {
                "SELECT id, user_id, key, old_value, new_value, action, strftime('%s', timestamp) as timestamp
                 FROM settings_audit_log ORDER BY timestamp DESC LIMIT ?1"
            }
        };

        let mut stmt = conn.prepare(query)
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let entries = match (key, user_id) {
            (Some(k), Some(u)) => {
                stmt.query_map(params![k, u, limit], Self::map_audit_row)
            }
            (Some(k), None) => {
                stmt.query_map(params![k, limit], Self::map_audit_row)
            }
            (None, Some(u)) => {
                stmt.query_map(params![u, limit], Self::map_audit_row)
            }
            (None, None) => {
                stmt.query_map(params![limit], Self::map_audit_row)
            }
        }
        .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        Ok(entries)
    }

    fn map_audit_row(row: &rusqlite::Row) -> rusqlite::Result<AuditLogEntry> {
        Ok(AuditLogEntry {
            id: row.get(0)?,
            user_id: row.get(1)?,
            key: row.get(2)?,
            old_value: row.get(3)?,
            new_value: row.get(4)?,
            action: row.get(5)?,
            timestamp: row.get::<_, String>(6)?.parse().unwrap_or(0),
        })
    }

    pub async fn list_all_users(&self) -> Result<Vec<User>, UserServiceError> {
        let conn = self.conn.read().await;
        let mut stmt = conn
            .prepare("SELECT id, nostr_pubkey, username, is_power_user, strftime('%s', created_at) as created_at, strftime('%s', last_seen) as last_seen FROM users ORDER BY created_at DESC")
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        let users = stmt
            .query_map([], |row| {
                Ok(User {
                    id: row.get(0)?,
                    nostr_pubkey: row.get(1)?,
                    username: row.get(2)?,
                    is_power_user: row.get::<_, i64>(3)? == 1,
                    created_at: row.get::<_, String>(4)?.parse().unwrap_or(0),
                    last_seen: row.get::<_, String>(5)?.parse().unwrap_or(0),
                })
            })
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| UserServiceError::DatabaseError(e.to_string()))?;

        Ok(users)
    }
}
