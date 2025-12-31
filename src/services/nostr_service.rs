use crate::config::feature_access::FeatureAccess;
use crate::models::protected_settings::{ApiKeys, NostrUser};
use crate::utils::nip98::{
    parse_auth_header, validate_nip98_token, Nip98ValidationError, Nip98ValidationResult,
};
use chrono::Utc;
use log::{debug, error, info, warn};
use nostr_sdk::{event::Error as EventError, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;
use crate::utils::time;
use crate::utils::json::{from_json, to_json};

#[derive(Debug, Error)]
pub enum NostrError {
    #[error("Invalid event: {0}")]
    InvalidEvent(String),
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("User not found")]
    UserNotFound,
    #[error("Invalid token")]
    InvalidToken,
    #[error("Session expired")]
    SessionExpired,
    #[error("Power user operation not allowed")]
    PowerUserOperation,
    #[error("Nostr event error: {0}")]
    NostrError(#[from] EventError),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("NIP-98 validation error: {0}")]
    Nip98Error(String),
}

impl Serialize for NostrError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NostrError", 2)?;
        match self {
            NostrError::InvalidEvent(msg) => {
                state.serialize_field("type", "InvalidEvent")?;
                state.serialize_field("message", msg)?;
            }
            NostrError::InvalidSignature => {
                state.serialize_field("type", "InvalidSignature")?;
                state.serialize_field("message", "Invalid signature")?;
            }
            NostrError::UserNotFound => {
                state.serialize_field("type", "UserNotFound")?;
                state.serialize_field("message", "User not found")?;
            }
            NostrError::InvalidToken => {
                state.serialize_field("type", "InvalidToken")?;
                state.serialize_field("message", "Invalid token")?;
            }
            NostrError::SessionExpired => {
                state.serialize_field("type", "SessionExpired")?;
                state.serialize_field("message", "Session expired")?;
            }
            NostrError::PowerUserOperation => {
                state.serialize_field("type", "PowerUserOperation")?;
                state.serialize_field("message", "Power user operation not allowed")?;
            }
            NostrError::NostrError(e) => {
                state.serialize_field("type", "NostrError")?;
                state.serialize_field("message", &e.to_string())?;
            }
            NostrError::JsonError(e) => {
                state.serialize_field("type", "JsonError")?;
                state.serialize_field("message", &e.to_string())?;
            }
            NostrError::Nip98Error(msg) => {
                state.serialize_field("type", "Nip98Error")?;
                state.serialize_field("message", msg)?;
            }
        }
        state.end()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthEvent {
    pub id: String,
    pub pubkey: String,
    pub content: String,
    pub sig: String,
    pub created_at: i64,
    pub kind: i32,
    pub tags: Vec<Vec<String>>,
}

#[derive(Clone)]
pub struct NostrService {
    users: Arc<RwLock<HashMap<String, NostrUser>>>,
    power_user_pubkeys: Vec<String>,
    token_expiry: i64,
    feature_access: Arc<RwLock<FeatureAccess>>,
}

impl NostrService {
    pub fn new() -> Self {
        let power_users = std::env::var("POWER_USER_PUBKEYS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let token_expiry = std::env::var("AUTH_TOKEN_EXPIRY")
            .unwrap_or_else(|_| "3600".to_string())
            .parse()
            .unwrap_or(3600);

        let feature_access = Arc::new(RwLock::new(FeatureAccess::from_env()));
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
            power_user_pubkeys: power_users,
            feature_access,
            token_expiry,
        }
    }

    pub async fn verify_auth_event(&self, event: AuthEvent) -> Result<NostrUser, NostrError> {
        
        
        debug!(
            "Verifying auth event with id: {} and pubkey: {}",
            event.id, event.pubkey
        );

        let json_str = match to_json(&event) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to serialize auth event with id {}: {}", event.id, e);
                return Err(NostrError::JsonError(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))));
            }
        };

        debug!(
            "Event JSON for verification (truncated): {}...",
            if json_str.len() > 100 {
                &json_str[0..100]
            } else {
                &json_str
            }
        );

        let nostr_event = match Event::from_json(&json_str) {
            Ok(e) => e,
            Err(e) => {
                error!(
                    "Failed to parse Nostr event for pubkey {}: {}",
                    event.pubkey, e
                );
                return Err(NostrError::InvalidEvent(format!(
                    "Parse error for event {}: {}",
                    event.id, e
                )));
            }
        };

        if let Err(e) = nostr_event.verify() {
            error!(
                "Signature verification failed for pubkey {}: {}",
                event.pubkey, e
            );
            return Err(NostrError::InvalidSignature);
        }

        
        let mut feature_access = self.feature_access.write().await;
        if feature_access.register_new_user(&event.pubkey) {
            info!("Registered new user with basic access: {}", event.pubkey);
        }

        let now = time::now();
        let is_power_user = self.power_user_pubkeys.contains(&event.pubkey);

        
        let session_token = Uuid::new_v4().to_string();

        let user = NostrUser {
            pubkey: event.pubkey.clone(),
            npub: nostr_event
                .pubkey
                .to_bech32()
                .map_err(|_| NostrError::NostrError(EventError::InvalidId))?,
            is_power_user,
            api_keys: ApiKeys::default(),
            last_seen: now.timestamp(),
            session_token: Some(session_token),
        };

        
        info!(
            "Created/updated user: pubkey={}, is_power_user={}",
            user.pubkey, user.is_power_user
        );

        
        let mut users = self.users.write().await;
        users.insert(user.pubkey.clone(), user.clone());

        Ok(user)
    }

    pub async fn get_user(&self, pubkey: &str) -> Option<NostrUser> {
        let users = self.users.read().await;
        users.get(pubkey).cloned()
    }

    pub async fn update_user_api_keys(
        &self,
        pubkey: &str,
        api_keys: ApiKeys,
    ) -> Result<NostrUser, NostrError> {
        let mut users = self.users.write().await;

        if let Some(user) = users.get_mut(pubkey) {
            if user.is_power_user {
                return Err(NostrError::PowerUserOperation);
            }
            user.api_keys = api_keys;
            user.last_seen = time::timestamp_seconds();
            Ok(user.clone())
        } else {
            Err(NostrError::UserNotFound)
        }
    }

    pub async fn validate_session(&self, pubkey: &str, token: &str) -> bool {
        if let Some(user) = self.get_user(pubkey).await {
            if let Some(session_token) = user.session_token {
                let now = time::timestamp_seconds();
                if now - user.last_seen <= self.token_expiry {
                    return session_token == token;
                }
            }
        }
        false
    }

    pub async fn refresh_session(&self, pubkey: &str) -> Result<String, NostrError> {
        let mut users = self.users.write().await;

        if let Some(user) = users.get_mut(pubkey) {
            let now = time::timestamp_seconds();
            let new_token = Uuid::new_v4().to_string();
            user.session_token = Some(new_token.clone());
            user.last_seen = now;
            Ok(new_token)
        } else {
            Err(NostrError::UserNotFound)
        }
    }

    pub async fn logout(&self, pubkey: &str) -> Result<(), NostrError> {
        let mut users = self.users.write().await;

        if let Some(user) = users.get_mut(pubkey) {
            user.session_token = None;
            user.last_seen = time::timestamp_seconds();
            Ok(())
        } else {
            Err(NostrError::UserNotFound)
        }
    }

    pub async fn cleanup_sessions(&self, max_age_hours: i64) {
        let now = time::now();
        let mut users = self.users.write().await;

        users.retain(|_, user| {
            let age = now.timestamp() - user.last_seen;
            age < (max_age_hours * 3600)
        });
    }

    pub async fn is_power_user(&self, pubkey: &str) -> bool {
        if let Some(user) = self.get_user(pubkey).await {
            user.is_power_user
        } else {
            false
        }
    }

    /// Get session by token (looks up user by token)
    pub async fn get_session(&self, token: &str) -> Option<NostrUser> {
        let users = self.users.read().await;
        let token_string = token.to_string();
        users.values()
            .find(|user| {
                user.session_token.as_ref() == Some(&token_string)
            })
            .cloned()
    }

    /// Validate NIP-98 HTTP authentication from Authorization header
    ///
    /// This validates tokens for Solid server requests per NIP-98 spec.
    /// Tokens must be signed, unexpired (60s max), and match the request URL/method.
    ///
    /// # Arguments
    /// * `auth_header` - Full Authorization header (e.g., "Nostr <base64>")
    /// * `request_url` - The URL the request was made to
    /// * `request_method` - The HTTP method (GET, POST, PUT, etc.)
    /// * `request_body` - Optional request body for payload verification
    ///
    /// # Returns
    /// The authenticated NostrUser or an error
    pub async fn verify_nip98_auth(
        &self,
        auth_header: &str,
        request_url: &str,
        request_method: &str,
        request_body: Option<&str>,
    ) -> Result<NostrUser, NostrError> {
        // Parse the Authorization header
        let token = parse_auth_header(auth_header).ok_or_else(|| {
            NostrError::Nip98Error("Invalid Authorization header format".to_string())
        })?;

        // Validate the NIP-98 token
        let validation = validate_nip98_token(token, request_url, request_method, request_body)
            .map_err(|e| NostrError::Nip98Error(e.to_string()))?;

        debug!(
            "NIP-98 token validated for pubkey: {}..., url: {}, method: {}",
            &validation.pubkey[..16.min(validation.pubkey.len())],
            validation.url,
            validation.method
        );

        // Get or create the user
        let user = self.get_or_create_user_from_pubkey(&validation.pubkey).await?;

        info!(
            "NIP-98 auth successful for pubkey: {}, is_power_user: {}",
            user.pubkey, user.is_power_user
        );

        Ok(user)
    }

    /// Get existing user or create a new one from pubkey
    async fn get_or_create_user_from_pubkey(&self, pubkey: &str) -> Result<NostrUser, NostrError> {
        // Check if user exists
        if let Some(user) = self.get_user(pubkey).await {
            // Update last_seen
            let mut users = self.users.write().await;
            if let Some(user) = users.get_mut(pubkey) {
                user.last_seen = time::timestamp_seconds();
            }
            return Ok(users.get(pubkey).cloned().unwrap());
        }

        // Register new user with feature access
        let mut feature_access = self.feature_access.write().await;
        if feature_access.register_new_user(pubkey) {
            info!("Registered new user via NIP-98 with basic access: {}", pubkey);
        }

        let is_power_user = self.power_user_pubkeys.contains(&pubkey.to_string());
        let session_token = Uuid::new_v4().to_string();

        // Convert hex pubkey to npub (bech32)
        let npub = match PublicKey::from_hex(pubkey) {
            Ok(pk) => pk.to_bech32().unwrap_or_else(|_| pubkey.to_string()),
            Err(_) => {
                warn!("Could not convert pubkey to npub: {}", pubkey);
                pubkey.to_string()
            }
        };

        let user = NostrUser {
            pubkey: pubkey.to_string(),
            npub,
            is_power_user,
            api_keys: ApiKeys::default(),
            last_seen: time::timestamp_seconds(),
            session_token: Some(session_token),
        };

        // Store the new user
        let mut users = self.users.write().await;
        users.insert(pubkey.to_string(), user.clone());

        info!(
            "Created new user via NIP-98: pubkey={}, is_power_user={}",
            user.pubkey, user.is_power_user
        );

        Ok(user)
    }

    /// Validate NIP-98 token and return just the validation result (no user creation)
    ///
    /// Use this for stateless validation when you only need to verify the token.
    pub fn validate_nip98_token_only(
        &self,
        auth_header: &str,
        request_url: &str,
        request_method: &str,
        request_body: Option<&str>,
    ) -> Result<Nip98ValidationResult, NostrError> {
        let token = parse_auth_header(auth_header).ok_or_else(|| {
            NostrError::Nip98Error("Invalid Authorization header format".to_string())
        })?;

        validate_nip98_token(token, request_url, request_method, request_body)
            .map_err(|e| NostrError::Nip98Error(e.to_string()))
    }
}

impl Default for NostrService {
    fn default() -> Self {
        Self::new()
    }
}
