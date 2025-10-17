// WebSocket Broadcast Integration for Settings Changes
// Notifies all connected clients when settings are modified

use actix::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{info, debug, warn};

use crate::services::database_service::SettingValue;

/// Manager for broadcasting settings changes to WebSocket clients
#[derive(Clone)]
pub struct SettingsBroadcastManager {
    clients: Arc<RwLock<HashMap<String, SettingsClientInfo>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<String>>>>, // key prefix -> client_ids
}

#[derive(Clone)]
struct SettingsClientInfo {
    client_id: String,
    subscriptions: Vec<String>, // Prefixes the client is subscribed to
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsChangeNotification {
    pub event_type: String,
    pub key: String,
    pub value: JsonValue,
    pub changed_by: Option<String>,
    pub timestamp: u64,
}

impl SettingsBroadcastManager {
    pub fn new() -> Self {
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new WebSocket client
    pub async fn register_client(&self, client_id: String) {
        let mut clients = self.clients.write().await;
        clients.insert(
            client_id.clone(),
            SettingsClientInfo {
                client_id: client_id.clone(),
                subscriptions: vec!["*".to_string()], // Default: subscribe to all
            },
        );
        info!("Settings WebSocket client registered: {}", client_id);
    }

    /// Unregister a WebSocket client
    pub async fn unregister_client(&self, client_id: &str) {
        let mut clients = self.clients.write().await;
        if clients.remove(client_id).is_some() {
            info!("Settings WebSocket client unregistered: {}", client_id);
        }

        // Clean up subscriptions
        let mut subscriptions = self.subscriptions.write().await;
        for (_, client_ids) in subscriptions.iter_mut() {
            client_ids.retain(|id| id != client_id);
        }
    }

    /// Subscribe client to specific setting prefix
    pub async fn subscribe(&self, client_id: &str, prefix: String) {
        let mut clients = self.clients.write().await;
        if let Some(client) = clients.get_mut(client_id) {
            if !client.subscriptions.contains(&prefix) {
                client.subscriptions.push(prefix.clone());
            }
        }

        let mut subscriptions = self.subscriptions.write().await;
        subscriptions
            .entry(prefix.clone())
            .or_insert_with(Vec::new)
            .push(client_id.to_string());

        debug!(
            "Client {} subscribed to settings prefix: {}",
            client_id, prefix
        );
    }

    /// Unsubscribe client from specific setting prefix
    pub async fn unsubscribe(&self, client_id: &str, prefix: &str) {
        let mut clients = self.clients.write().await;
        if let Some(client) = clients.get_mut(client_id) {
            client.subscriptions.retain(|p| p != prefix);
        }

        let mut subscriptions = self.subscriptions.write().await;
        if let Some(client_ids) = subscriptions.get_mut(prefix) {
            client_ids.retain(|id| id != client_id);
        }

        debug!(
            "Client {} unsubscribed from settings prefix: {}",
            client_id, prefix
        );
    }

    /// Broadcast setting change to all interested clients
    pub async fn broadcast_change(&self, key: &str, value: &SettingValue, user_id: Option<&str>) {
        let clients = self.clients.read().await;
        let interested_clients: Vec<String> = clients
            .values()
            .filter(|client| self.is_subscribed_to(client, key))
            .map(|client| client.client_id.clone())
            .collect();

        if interested_clients.is_empty() {
            return;
        }

        let notification = SettingsChangeNotification {
            event_type: "settingChanged".to_string(),
            key: key.to_string(),
            value: convert_setting_value_to_json(value),
            changed_by: user_id.map(|s| s.to_string()),
            timestamp: current_timestamp(),
        };

        info!(
            "Broadcasting setting change for '{}' to {} clients",
            key,
            interested_clients.len()
        );

        // In a real implementation, this would send the notification via WebSocket
        // For now, we just log it
        debug!("Broadcast notification: {:?}", notification);
    }

    /// Check if client is subscribed to a specific key
    fn is_subscribed_to(&self, client: &SettingsClientInfo, key: &str) -> bool {
        for subscription in &client.subscriptions {
            if subscription == "*" || key.starts_with(subscription) {
                return true;
            }
        }
        false
    }

    /// Get all registered clients
    pub async fn get_client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Get subscription stats
    pub async fn get_subscription_stats(&self) -> HashMap<String, usize> {
        let subscriptions = self.subscriptions.read().await;
        subscriptions
            .iter()
            .map(|(prefix, clients)| (prefix.clone(), clients.len()))
            .collect()
    }
}

fn convert_setting_value_to_json(value: &SettingValue) -> JsonValue {
    match value {
        SettingValue::String(s) => JsonValue::String(s.clone()),
        SettingValue::Integer(i) => JsonValue::Number((*i).into()),
        SettingValue::Float(f) => JsonValue::Number(
            serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0)),
        ),
        SettingValue::Boolean(b) => JsonValue::Bool(*b),
        SettingValue::Json(j) => j.clone(),
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_unregister_client() {
        let manager = SettingsBroadcastManager::new();

        manager.register_client("client1".to_string()).await;
        assert_eq!(manager.get_client_count().await, 1);

        manager.unregister_client("client1").await;
        assert_eq!(manager.get_client_count().await, 0);
    }

    #[tokio::test]
    async fn test_subscribe_unsubscribe() {
        let manager = SettingsBroadcastManager::new();

        manager.register_client("client1".to_string()).await;
        manager
            .subscribe("client1", "visualisation.physics".to_string())
            .await;

        let stats = manager.get_subscription_stats().await;
        assert_eq!(stats.get("visualisation.physics"), Some(&1));

        manager.unsubscribe("client1", "visualisation.physics").await;
        let stats = manager.get_subscription_stats().await;
        assert_eq!(stats.get("visualisation.physics"), Some(&0));
    }
}
