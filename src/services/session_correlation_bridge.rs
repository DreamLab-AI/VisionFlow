//! Session Correlation Bridge - Bidirectional mapping between client sessionIds and server CorrelationIds
//!
//! This module provides thread-safe bidirectional mapping to link client-generated session IDs
//! with server-generated correlation IDs for complete distributed tracing.
//!
//! ## Features
//! - Bidirectional HashMap mapping (client_session_id ↔ correlation_id)
//! - Auto-registration when client connects
//! - Thread-safe Arc<RwLock<>> wrapper
//! - Integration with agent_telemetry.rs
//! - HTTP header propagation (X-Session-ID)
//! - Automatic cleanup of stale mappings
//!
//! ## Usage
//! ```rust
//! use session_correlation_bridge::{SessionCorrelationBridge, init_global_bridge};
//!
//! // Initialize the global bridge
//! init_global_bridge();
//!
//! // Register a mapping when client connects
//! let bridge = get_global_bridge().unwrap();
//! bridge.register_session("session_1234567890_abc123", correlation_id);
//!
//! // Later, retrieve correlation_id from client session_id
//! let correlation_id = bridge.get_correlation_id("session_1234567890_abc123");
//! ```

use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use uuid;

use crate::telemetry::agent_telemetry::CorrelationId;

/// Session mapping entry with metadata and timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMapping {
    /// Client-generated session ID (format: session_${timestamp}_${random})
    pub client_session_id: String,

    /// Server-generated correlation ID (format: UUID or agent-{id})
    pub correlation_id: CorrelationId,

    /// Timestamp when mapping was created
    pub created_at: DateTime<Utc>,

    /// Last access timestamp for cleanup purposes
    pub last_accessed: DateTime<Utc>,

    /// Optional metadata about the session
    pub metadata: HashMap<String, String>,
}

impl SessionMapping {
    /// Create a new session mapping
    pub fn new(client_session_id: String, correlation_id: CorrelationId) -> Self {
        let now = Utc::now();
        Self {
            client_session_id,
            correlation_id,
            created_at: now,
            last_accessed: now,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the mapping
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Update last accessed timestamp
    pub fn touch(&mut self) {
        self.last_accessed = Utc::now();
    }

    /// Check if mapping is stale (not accessed in given duration)
    pub fn is_stale(&self, max_age_seconds: i64) -> bool {
        let now = Utc::now();
        (now - self.last_accessed).num_seconds() > max_age_seconds
    }
}

/// Thread-safe bidirectional mapping between client session IDs and server correlation IDs
#[derive(Clone)]
pub struct SessionCorrelationBridge {
    /// Primary mapping: client_session_id -> SessionMapping
    client_to_server: Arc<RwLock<HashMap<String, SessionMapping>>>,

    /// Reverse mapping: correlation_id -> client_session_id for fast lookups
    server_to_client: Arc<RwLock<HashMap<String, String>>>,

    /// Configuration for automatic cleanup
    max_age_seconds: i64,
}

impl SessionCorrelationBridge {
    /// Create a new session correlation bridge with default settings
    pub fn new() -> Self {
        Self::with_max_age(3600) // Default 1 hour TTL
    }

    /// Create a new session correlation bridge with custom max age
    ///
    /// # Arguments
    /// * `max_age_seconds` - Maximum age of mappings before considered stale (default: 3600 = 1 hour)
    pub fn with_max_age(max_age_seconds: i64) -> Self {
        info!("Initialising SessionCorrelationBridge with max_age={}s", max_age_seconds);
        Self {
            client_to_server: Arc::new(RwLock::new(HashMap::new())),
            server_to_client: Arc::new(RwLock::new(HashMap::new())),
            max_age_seconds,
        }
    }

    /// Register a new session mapping
    ///
    /// # Arguments
    /// * `client_session_id` - Client-generated session ID (e.g., "session_1234567890_abc123")
    /// * `correlation_id` - Server-generated correlation ID
    ///
    /// # Returns
    /// * `Ok(())` if registration successful
    /// * `Err(String)` if locks could not be acquired
    pub fn register_session(
        &self,
        client_session_id: String,
        correlation_id: CorrelationId,
    ) -> Result<(), String> {
        debug!(
            "Registering session mapping: client_session_id={}, correlation_id={}",
            client_session_id, correlation_id
        );

        let mapping = SessionMapping::new(client_session_id.clone(), correlation_id.clone());

        // Acquire write locks
        let mut client_map = self.client_to_server
            .write()
            .map_err(|e| format!("Failed to acquire client_to_server write lock: {}", e))?;

        let mut server_map = self.server_to_client
            .write()
            .map_err(|e| format!("Failed to acquire server_to_client write lock: {}", e))?;

        // Insert into both maps
        client_map.insert(client_session_id.clone(), mapping);
        server_map.insert(correlation_id.as_str().to_string(), client_session_id.clone());

        info!(
            "Session registered: {} ↔ {} (total mappings: {})",
            client_session_id,
            correlation_id,
            client_map.len()
        );

        Ok(())
    }

    /// Register with additional metadata
    pub fn register_session_with_metadata(
        &self,
        client_session_id: String,
        correlation_id: CorrelationId,
        metadata: HashMap<String, String>,
    ) -> Result<(), String> {
        debug!(
            "Registering session with metadata: client_session_id={}, correlation_id={}, metadata={:?}",
            client_session_id, correlation_id, metadata
        );

        let (key, value) = metadata.into_iter().next().unwrap_or(("".to_string(), "".to_string()));
        let mapping = SessionMapping::new(client_session_id.clone(), correlation_id.clone())
            .with_metadata(key, value);

        let mut client_map = self.client_to_server
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        let mut server_map = self.server_to_client
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        client_map.insert(client_session_id.clone(), mapping);
        server_map.insert(correlation_id.as_str().to_string(), client_session_id);

        Ok(())
    }

    /// Get correlation ID from client session ID
    ///
    /// # Arguments
    /// * `client_session_id` - Client-generated session ID
    ///
    /// # Returns
    /// * `Some(CorrelationId)` if mapping exists
    /// * `None` if no mapping found
    pub fn get_correlation_id(&self, client_session_id: &str) -> Option<CorrelationId> {
        let mut client_map = self.client_to_server.write().ok()?;

        if let Some(mapping) = client_map.get_mut(client_session_id) {
            // Update last accessed timestamp
            mapping.touch();
            debug!("Retrieved correlation_id={} for client_session_id={}",
                   mapping.correlation_id, client_session_id);
            Some(mapping.correlation_id.clone())
        } else {
            debug!("No correlation_id found for client_session_id={}", client_session_id);
            None
        }
    }

    /// Get client session ID from correlation ID
    ///
    /// # Arguments
    /// * `correlation_id` - Server-generated correlation ID
    ///
    /// # Returns
    /// * `Some(String)` if mapping exists
    /// * `None` if no mapping found
    pub fn get_client_session_id(&self, correlation_id: &CorrelationId) -> Option<String> {
        let server_map = self.server_to_client.read().ok()?;

        if let Some(client_session_id) = server_map.get(correlation_id.as_str()) {
            debug!("Retrieved client_session_id={} for correlation_id={}",
                   client_session_id, correlation_id);

            // Update last accessed in primary map
            if let Ok(mut client_map) = self.client_to_server.write() {
                if let Some(mapping) = client_map.get_mut(client_session_id) {
                    mapping.touch();
                }
            }

            Some(client_session_id.clone())
        } else {
            debug!("No client_session_id found for correlation_id={}", correlation_id);
            None
        }
    }

    /// Get full session mapping with metadata
    pub fn get_session_mapping(&self, client_session_id: &str) -> Option<SessionMapping> {
        let mut client_map = self.client_to_server.write().ok()?;

        if let Some(mapping) = client_map.get_mut(client_session_id) {
            mapping.touch();
            Some(mapping.clone())
        } else {
            None
        }
    }

    /// Remove a session mapping
    ///
    /// # Arguments
    /// * `client_session_id` - Client session ID to remove
    ///
    /// # Returns
    /// * `Ok(bool)` - true if mapping was removed, false if not found
    /// * `Err(String)` if locks could not be acquired
    pub fn remove_session(&self, client_session_id: &str) -> Result<bool, String> {
        debug!("Removing session mapping for client_session_id={}", client_session_id);

        let mut client_map = self.client_to_server
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        if let Some(mapping) = client_map.remove(client_session_id) {
            // Also remove from reverse map
            let mut server_map = self.server_to_client
                .write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            server_map.remove(mapping.correlation_id.as_str());

            info!("Session mapping removed: {} ↔ {}", client_session_id, mapping.correlation_id);
            Ok(true)
        } else {
            debug!("No mapping found for client_session_id={}", client_session_id);
            Ok(false)
        }
    }

    /// Clean up stale mappings based on last accessed timestamp
    ///
    /// # Returns
    /// * Number of mappings removed
    pub fn cleanup_stale_sessions(&self) -> Result<usize, String> {
        debug!("Starting cleanup of stale sessions (max_age={}s)", self.max_age_seconds);

        let mut client_map = self.client_to_server
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        let mut server_map = self.server_to_client
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        // Find stale session IDs
        let stale_sessions: Vec<String> = client_map
            .iter()
            .filter(|(_, mapping)| mapping.is_stale(self.max_age_seconds))
            .map(|(session_id, _)| session_id.clone())
            .collect();

        let count = stale_sessions.len();

        // Remove stale mappings
        for session_id in stale_sessions {
            if let Some(mapping) = client_map.remove(&session_id) {
                server_map.remove(mapping.correlation_id.as_str());
                debug!("Removed stale mapping: {} ↔ {}", session_id, mapping.correlation_id);
            }
        }

        if count > 0 {
            info!("Cleaned up {} stale session mappings", count);
        }

        Ok(count)
    }

    /// Get statistics about current mappings
    pub fn get_stats(&self) -> Result<BridgeStats, String> {
        let client_map = self.client_to_server
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        let now = Utc::now();
        let stale_count = client_map
            .values()
            .filter(|m| m.is_stale(self.max_age_seconds))
            .count();

        Ok(BridgeStats {
            total_mappings: client_map.len(),
            stale_mappings: stale_count,
            active_mappings: client_map.len() - stale_count,
            max_age_seconds: self.max_age_seconds,
        })
    }

    /// Get all active client session IDs
    pub fn get_all_client_sessions(&self) -> Result<Vec<String>, String> {
        let client_map = self.client_to_server
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        Ok(client_map
            .keys()
            .filter(|k| {
                if let Some(mapping) = client_map.get(*k) {
                    !mapping.is_stale(self.max_age_seconds)
                } else {
                    false
                }
            })
            .cloned()
            .collect())
    }

    /// Get all active correlation IDs
    pub fn get_all_correlation_ids(&self) -> Result<Vec<CorrelationId>, String> {
        let client_map = self.client_to_server
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        Ok(client_map
            .values()
            .filter(|m| !m.is_stale(self.max_age_seconds))
            .map(|m| m.correlation_id.clone())
            .collect())
    }

    /// Async wrapper for lookup_correlation - checks if client session exists and returns correlation ID
    /// This is the primary method used by client_log_handler
    pub async fn lookup_correlation(&self, client_session_id: &str) -> Option<uuid::Uuid> {
        self.get_correlation_id(client_session_id)
            .and_then(|corr_id| uuid::Uuid::parse_str(corr_id.as_str()).ok())
    }

    /// Async wrapper for register - creates new bidirectional mapping
    /// This is called when a new client session is detected
    pub async fn register(&self, client_session_id: String, correlation_uuid: uuid::Uuid) {
        let correlation_id = CorrelationId(correlation_uuid.to_string());
        if let Err(e) = self.register_session(client_session_id.clone(), correlation_id) {
            error!("Failed to register session correlation: {}", e);
        } else {
            debug!("Session correlation registered: {} ↔ {}", client_session_id, correlation_uuid);
        }
    }

    /// Async wrapper for cleanup - removes expired mappings based on TTL
    /// Returns (removed_count, remaining_count)
    pub async fn cleanup_expired(&self, max_age: tokio::time::Duration) -> (usize, usize) {
        let max_age_secs = max_age.as_secs() as i64;

        match self.cleanup_stale_sessions_with_age(max_age_secs) {
            Ok(removed) => {
                let remaining = self.get_stats()
                    .map(|s| s.total_mappings)
                    .unwrap_or(0);
                (removed, remaining)
            }
            Err(e) => {
                error!("Failed to cleanup expired sessions: {}", e);
                (0, 0)
            }
        }
    }

    /// Internal helper for cleanup with custom age
    fn cleanup_stale_sessions_with_age(&self, max_age_seconds: i64) -> Result<usize, String> {
        debug!("Starting cleanup of stale sessions (max_age={}s)", max_age_seconds);

        let mut client_map = self.client_to_server
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        let mut server_map = self.server_to_client
            .write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        // Find stale session IDs
        let stale_sessions: Vec<String> = client_map
            .iter()
            .filter(|(_, mapping)| mapping.is_stale(max_age_seconds))
            .map(|(session_id, _)| session_id.clone())
            .collect();

        let count = stale_sessions.len();

        // Remove stale mappings
        for session_id in stale_sessions {
            if let Some(mapping) = client_map.remove(&session_id) {
                server_map.remove(mapping.correlation_id.as_str());
                debug!("Removed stale mapping: {} ↔ {}", session_id, mapping.correlation_id);
            }
        }

        if count > 0 {
            info!("Cleaned up {} stale session mappings", count);
        }

        Ok(count)
    }
}

/// Statistics about the bridge state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStats {
    pub total_mappings: usize,
    pub stale_mappings: usize,
    pub active_mappings: usize,
    pub max_age_seconds: i64,
}

/// Global instance of the session correlation bridge
static mut GLOBAL_BRIDGE: Option<SessionCorrelationBridge> = None;
static BRIDGE_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the global session correlation bridge
///
/// # Arguments
/// * `max_age_seconds` - Maximum age of mappings before considered stale (default: 3600)
pub fn init_global_bridge(max_age_seconds: Option<i64>) {
    BRIDGE_INIT.call_once(|| {
        let max_age = max_age_seconds.unwrap_or(3600);
        let bridge = SessionCorrelationBridge::with_max_age(max_age);
        unsafe {
            GLOBAL_BRIDGE = Some(bridge);
        }
        info!("Global SessionCorrelationBridge initialized with max_age={}s", max_age);
    });
}

/// Get the global session correlation bridge
pub fn get_global_bridge() -> Option<&'static SessionCorrelationBridge> {
    unsafe { GLOBAL_BRIDGE.as_ref() }
}

/// Extract client session ID from HTTP X-Session-ID header
pub fn extract_session_id_from_header(header_value: Option<&str>) -> Option<String> {
    header_value.map(|v| {
        let session_id = v.trim().to_string();
        debug!("Extracted session_id from X-Session-ID header: {}", session_id);
        session_id
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_session_mapping_creation() {
        let session_id = "session_1234567890_abc123".to_string();
        let correlation_id = CorrelationId::new();

        let mapping = SessionMapping::new(session_id.clone(), correlation_id.clone());

        assert_eq!(mapping.client_session_id, session_id);
        assert_eq!(mapping.correlation_id, correlation_id);
        assert!(mapping.metadata.is_empty());
    }

    #[test]
    fn test_bidirectional_mapping() {
        let bridge = SessionCorrelationBridge::with_max_age(3600);
        let session_id = "session_1234567890_abc123".to_string();
        let correlation_id = CorrelationId::new();

        // Register mapping
        bridge.register_session(session_id.clone(), correlation_id.clone())
            .expect("Failed to register session");

        // Test forward lookup
        let retrieved_corr_id = bridge.get_correlation_id(&session_id)
            .expect("Failed to retrieve correlation_id");
        assert_eq!(retrieved_corr_id, correlation_id);

        // Test reverse lookup
        let retrieved_session_id = bridge.get_client_session_id(&correlation_id)
            .expect("Failed to retrieve client_session_id");
        assert_eq!(retrieved_session_id, session_id);
    }

    #[test]
    fn test_session_removal() {
        let bridge = SessionCorrelationBridge::with_max_age(3600);
        let session_id = "session_1234567890_abc123".to_string();
        let correlation_id = CorrelationId::new();

        // Register and then remove
        bridge.register_session(session_id.clone(), correlation_id.clone())
            .expect("Failed to register");

        let removed = bridge.remove_session(&session_id)
            .expect("Failed to remove session");
        assert!(removed);

        // Verify both directions are removed
        assert!(bridge.get_correlation_id(&session_id).is_none());
        assert!(bridge.get_client_session_id(&correlation_id).is_none());
    }

    #[test]
    fn test_stale_session_cleanup() {
        let bridge = SessionCorrelationBridge::with_max_age(1); // 1 second max age
        let session_id = "session_1234567890_abc123".to_string();
        let correlation_id = CorrelationId::new();

        bridge.register_session(session_id.clone(), correlation_id.clone())
            .expect("Failed to register");

        // Wait for session to become stale
        thread::sleep(Duration::from_secs(2));

        let cleaned = bridge.cleanup_stale_sessions()
            .expect("Failed to cleanup");
        assert_eq!(cleaned, 1);

        // Verify session was removed
        assert!(bridge.get_correlation_id(&session_id).is_none());
    }

    #[test]
    fn test_bridge_stats() {
        let bridge = SessionCorrelationBridge::with_max_age(3600);

        // Register multiple sessions
        for i in 0..5 {
            let session_id = format!("session_{}", i);
            let correlation_id = CorrelationId::new();
            bridge.register_session(session_id, correlation_id)
                .expect("Failed to register");
        }

        let stats = bridge.get_stats().expect("Failed to get stats");
        assert_eq!(stats.total_mappings, 5);
        assert_eq!(stats.active_mappings, 5);
        assert_eq!(stats.stale_mappings, 0);
    }

    #[test]
    fn test_concurrent_access() {
        let bridge = Arc::new(SessionCorrelationBridge::with_max_age(3600));
        let mut handles = vec![];

        // Spawn multiple threads to register sessions
        for i in 0..10 {
            let bridge_clone = Arc::clone(&bridge);
            let handle = thread::spawn(move || {
                let session_id = format!("session_{}", i);
                let correlation_id = CorrelationId::new();
                bridge_clone.register_session(session_id, correlation_id)
                    .expect("Failed to register");
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let stats = bridge.get_stats().expect("Failed to get stats");
        assert_eq!(stats.total_mappings, 10);
    }

    #[test]
    fn test_metadata_storage() {
        let bridge = SessionCorrelationBridge::with_max_age(3600);
        let session_id = "session_1234567890_abc123".to_string();
        let correlation_id = CorrelationId::new();

        let mut metadata = HashMap::new();
        metadata.insert("user_agent".to_string(), "Quest3".to_string());
        metadata.insert("ip_address".to_string(), "192.168.1.1".to_string());

        bridge.register_session_with_metadata(session_id.clone(), correlation_id, metadata)
            .expect("Failed to register with metadata");

        let mapping = bridge.get_session_mapping(&session_id)
            .expect("Failed to retrieve mapping");

        assert!(!mapping.metadata.is_empty());
    }
}
