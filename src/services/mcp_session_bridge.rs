use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use log::{debug, info, warn, error};
use chrono::{DateTime, Utc};

use crate::utils::docker_hive_mind::{DockerHiveMind, SwarmConfig, SwarmStatus};
use crate::client::mcp_tcp_client::{McpTelemetryClient, SessionStatus, SessionMetrics, SwarmInfo};

/// Bridge between session manager UUIDs and MCP telemetry swarm IDs
///
/// This component manages the lifecycle mapping:
/// 1. Session UUID (from session manager) -> created
/// 2. Swarm ID (from claude-flow MCP) -> appears after spawn
/// 3. Bidirectional mapping for telemetry queries
#[derive(Clone)]
pub struct McpSessionBridge {
    docker_hive_mind: DockerHiveMind,
    mcp_client: Arc<RwLock<McpTelemetryClient>>,

    // UUID -> Swarm ID mapping
    uuid_to_swarm: Arc<RwLock<HashMap<String, String>>>,

    // Swarm ID -> UUID reverse mapping
    swarm_to_uuid: Arc<RwLock<HashMap<String, String>>>,

    // Session metadata cache
    session_metadata_cache: Arc<RwLock<HashMap<String, MonitoredSessionMetadata>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoredSessionMetadata {
    pub uuid: String,
    pub swarm_id: Option<String>,
    pub task: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub swarm_discovered_at: Option<DateTime<Utc>>,
    pub agent_count: u32,
}

#[derive(Debug, Clone)]
pub struct MonitoredSession {
    pub uuid: String,
    pub swarm_id: Option<String>,
}

impl McpSessionBridge {
    /// Create a new MCP session bridge
    pub fn new(container_name: String) -> Self {
        let docker_hive_mind = DockerHiveMind::new(container_name);
        let mcp_client = Arc::new(RwLock::new(McpTelemetryClient::for_multi_agent_container()));

        Self {
            docker_hive_mind,
            mcp_client,
            uuid_to_swarm: Arc::new(RwLock::new(HashMap::new())),
            swarm_to_uuid: Arc::new(RwLock::new(HashMap::new())),
            session_metadata_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Spawn a new session and monitor until swarm ID is discovered
    pub async fn spawn_and_monitor(
        &self,
        task: &str,
        config: SwarmConfig,
    ) -> Result<MonitoredSession, Box<dyn std::error::Error + Send + Sync>> {
        info!("Spawning and monitoring session for task: {}", task);

        // 1. Create and start session via docker_hive_mind
        let uuid = self.docker_hive_mind.spawn_swarm(task, config).await?;

        info!("Session {} spawned, waiting for swarm ID...", uuid);

        // 2. Initialize metadata cache
        {
            let mut cache = self.session_metadata_cache.write().await;
            cache.insert(uuid.clone(), MonitoredSessionMetadata {
                uuid: uuid.clone(),
                swarm_id: None,
                task: task.to_string(),
                status: "starting".to_string(),
                created_at: Utc::now(),
                last_updated: Utc::now(),
                swarm_discovered_at: None,
                agent_count: 0,
            });
        }

        // 3. Poll MCP for swarm ID (with timeout)
        let swarm_id = self.poll_for_swarm_id(&uuid, Duration::from_secs(60)).await?;

        // 4. Link UUID to swarm ID
        self.link_session_to_swarm(&uuid, &swarm_id).await;

        info!("Session {} mapped to swarm {}", uuid, swarm_id);

        Ok(MonitoredSession {
            uuid,
            swarm_id: Some(swarm_id),
        })
    }

    /// Discover swarm ID from filesystem (fallback when MCP doesn't have mapping)
    async fn discover_swarm_id_from_filesystem(
        &self,
        uuid: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        use tokio::process::Command;

        debug!("Attempting filesystem discovery for session {}", uuid);

        let output = Command::new("docker")
            .args(&[
                "exec",
                "multi-agent-container",
                "find",
                &format!("/workspace/.swarm/sessions/{}/.hive-mind/sessions", uuid),
                "-name",
                "hive-mind-prompt-swarm-*.txt",
                "-type",
                "f",
            ])
            .output()
            .await?;

        if !output.status.success() {
            return Err(format!("Docker exec failed for session {}", uuid).into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let first_line = stdout.lines().next().ok_or("No swarm files found")?;

        // Extract swarm ID from path like: .../hive-mind-prompt-swarm-1759754652788-aygunvow1.txt
        if let Some(filename) = first_line.split('/').last() {
            if let Some(swarm_part) = filename.strip_prefix("hive-mind-prompt-swarm-")
                .and_then(|s| s.strip_suffix(".txt"))
            {
                let swarm_id = format!("swarm-{}", swarm_part);
                info!("Discovered swarm ID {} for session {} via filesystem", swarm_id, uuid);
                return Ok(swarm_id);
            }
        }

        Err(format!("Could not parse swarm ID from filesystem for session {}", uuid).into())
    }

    /// Poll MCP server until swarm ID appears for this session
    async fn poll_for_swarm_id(
        &self,
        uuid: &str,
        timeout: Duration,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        let mut delay = Duration::from_millis(500);

        loop {
            if start_time.elapsed() > timeout {
                // Final attempt: filesystem discovery
                warn!("MCP polling timed out for session {}, trying filesystem discovery", uuid);
                return self.discover_swarm_id_from_filesystem(uuid).await;
            }

            // Try filesystem discovery first (faster and more reliable)
            match self.discover_swarm_id_from_filesystem(uuid).await {
                Ok(swarm_id) => {
                    info!("Found swarm ID {} via filesystem for session {}", swarm_id, uuid);
                    return Ok(swarm_id);
                }
                Err(e) => {
                    debug!("Filesystem discovery failed: {}", e);
                }
            }

            // Query MCP for session status
            let mut mcp = self.mcp_client.write().await;

            match mcp.query_session_status(uuid).await {
                Ok(session_status) => {
                    if let Some(swarm_id) = session_status.swarm_id {
                        debug!("Discovered swarm ID {} for session {}", swarm_id, uuid);
                        return Ok(swarm_id);
                    }
                }
                Err(e) => {
                    debug!("Session {} not yet in MCP: {}", uuid, e);
                }
            }

            // Also try querying swarm list for new swarms
            match mcp.query_swarm_list().await {
                Ok(swarms) => {
                    // Look for swarms with matching session_id
                    for swarm in swarms {
                        if swarm.session_id == uuid {
                            debug!("Found swarm {} for session {} via swarm list", swarm.swarm_id, uuid);
                            return Ok(swarm.swarm_id);
                        }
                    }
                }
                Err(e) => {
                    debug!("Failed to query swarm list: {}", e);
                }
            }

            // Exponential backoff
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_secs(5));
        }
    }

    /// Link a session UUID to its swarm ID
    async fn link_session_to_swarm(&self, uuid: &str, swarm_id: &str) {
        // Forward mapping
        {
            let mut uuid_to_swarm = self.uuid_to_swarm.write().await;
            uuid_to_swarm.insert(uuid.to_string(), swarm_id.to_string());
        }

        // Reverse mapping
        {
            let mut swarm_to_uuid = self.swarm_to_uuid.write().await;
            swarm_to_uuid.insert(swarm_id.to_string(), uuid.to_string());
        }

        // Update metadata cache
        {
            let mut cache = self.session_metadata_cache.write().await;
            if let Some(metadata) = cache.get_mut(uuid) {
                metadata.swarm_id = Some(swarm_id.to_string());
                metadata.swarm_discovered_at = Some(Utc::now());
                metadata.last_updated = Utc::now();
            }
        }

        info!("Linked session {} to swarm {}", uuid, swarm_id);
    }

    /// Get swarm ID for a session UUID
    pub async fn get_swarm_id_for_session(&self, uuid: &str) -> Option<String> {
        let uuid_to_swarm = self.uuid_to_swarm.read().await;
        uuid_to_swarm.get(uuid).cloned()
    }

    /// Get session UUID for a swarm ID
    pub async fn get_session_for_swarm(&self, swarm_id: &str) -> Option<String> {
        let swarm_to_uuid = self.swarm_to_uuid.read().await;
        swarm_to_uuid.get(swarm_id).cloned()
    }

    /// Query session telemetry using session UUID (maps to swarm ID internally)
    pub async fn query_session_telemetry(
        &self,
        uuid: &str,
    ) -> Result<SessionMetrics, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Querying telemetry for session: {}", uuid);

        // 1. Get swarm ID
        let swarm_id = match self.get_swarm_id_for_session(uuid).await {
            Some(id) => id,
            None => {
                // Try to discover swarm ID if not cached
                warn!("Swarm ID not found for session {}, attempting discovery", uuid);
                match self.poll_for_swarm_id(uuid, Duration::from_secs(10)).await {
                    Ok(id) => {
                        self.link_session_to_swarm(uuid, &id).await;
                        id
                    }
                    Err(e) => {
                        return Err(format!("No swarm ID found for session {}: {}", uuid, e).into());
                    }
                }
            }
        };

        // 2. Query MCP using swarm ID
        let mut mcp = self.mcp_client.write().await;
        let metrics = mcp.query_swarm_metrics(&swarm_id).await?;

        // 3. Update metadata cache
        {
            let mut cache = self.session_metadata_cache.write().await;
            if let Some(metadata) = cache.get_mut(uuid) {
                metadata.agent_count = metrics.metrics.active_agents;
                metadata.last_updated = Utc::now();
            }
        }

        let perf_metrics = metrics.metrics.clone();

        Ok(SessionMetrics {
            swarm_id,
            session_id: uuid.to_string(),
            agents: vec![], // Populated from metrics.agents if needed
            tasks: perf_metrics.clone().into(),
            performance: perf_metrics,
        })
    }

    /// Get session status (combines session manager + MCP data)
    pub async fn get_session_status(
        &self,
        uuid: &str,
    ) -> Result<MonitoredSessionMetadata, Box<dyn std::error::Error + Send + Sync>> {
        // Check cache first
        {
            let cache = self.session_metadata_cache.read().await;
            if let Some(metadata) = cache.get(uuid) {
                if metadata.last_updated > Utc::now() - chrono::Duration::seconds(10) {
                    return Ok(metadata.clone());
                }
            }
        }

        // Query session manager for status
        let status = self.docker_hive_mind.get_session_status(uuid).await?;

        // Try to get MCP data if swarm ID is known
        let agent_count = if let Some(swarm_id) = self.get_swarm_id_for_session(uuid).await {
            match self.mcp_client.write().await.query_swarm_metrics(&swarm_id).await {
                Ok(metrics) => metrics.metrics.active_agents,
                Err(_) => 0,
            }
        } else {
            0
        };

        // Get metadata from docker_hive_mind
        let session_metadata = self.docker_hive_mind.get_session_metadata(uuid).await?;

        // Update cache
        let metadata = MonitoredSessionMetadata {
            uuid: uuid.to_string(),
            swarm_id: self.get_swarm_id_for_session(uuid).await,
            task: session_metadata.task,
            status: format!("{:?}", status),
            created_at: session_metadata.created,
            last_updated: Utc::now(),
            swarm_discovered_at: {
                let cache = self.session_metadata_cache.read().await;
                cache.get(uuid).and_then(|m| m.swarm_discovered_at)
            },
            agent_count,
        };

        {
            let mut cache = self.session_metadata_cache.write().await;
            cache.insert(uuid.to_string(), metadata.clone());
        }

        Ok(metadata)
    }

    /// Refresh all session mappings from MCP
    pub async fn refresh_mappings(&self) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Refreshing session-to-swarm mappings");

        let mut discovered = 0u32;

        // Get all swarms from MCP
        let swarms = self.mcp_client.write().await.query_swarm_list().await?;

        for swarm in swarms {
            let uuid = swarm.session_id;
            let swarm_id = swarm.swarm_id;

            // Check if already mapped
            if self.get_swarm_id_for_session(&uuid).await.is_none() {
                self.link_session_to_swarm(&uuid, &swarm_id).await;
                discovered += 1;
            }
        }

        if discovered > 0 {
            info!("Discovered {} new session-to-swarm mappings", discovered);
        }

        Ok(discovered)
    }

    /// List all monitored sessions
    pub async fn list_monitored_sessions(&self) -> Vec<MonitoredSessionMetadata> {
        let cache = self.session_metadata_cache.read().await;
        cache.values().cloned().collect()
    }

    /// Remove completed/failed sessions from cache
    pub async fn cleanup_completed_sessions(&self) -> u32 {
        let mut cleaned = 0u32;

        let mut cache = self.session_metadata_cache.write().await;

        let to_remove: Vec<String> = cache
            .iter()
            .filter(|(_, metadata)| {
                metadata.status == "Completed" || metadata.status == "Failed"
            })
            .map(|(uuid, _)| uuid.clone())
            .collect();

        for uuid in to_remove {
            cache.remove(&uuid);
            cleaned += 1;

            // Also remove from mappings
            if let Some(swarm_id) = {
                let uuid_to_swarm = self.uuid_to_swarm.read().await;
                uuid_to_swarm.get(&uuid).cloned()
            } {
                let mut uuid_to_swarm = self.uuid_to_swarm.write().await;
                uuid_to_swarm.remove(&uuid);

                let mut swarm_to_uuid = self.swarm_to_uuid.write().await;
                swarm_to_uuid.remove(&swarm_id);
            }
        }

        if cleaned > 0 {
            info!("Cleaned up {} completed sessions", cleaned);
        }

        cleaned
    }

    /// Background task to periodically refresh mappings
    pub async fn start_background_refresh(self: Arc<Self>, interval: Duration) {
        info!("Starting background session mapping refresh (interval: {:?})", interval);

        let mut refresh_interval = tokio::time::interval(interval);

        loop {
            refresh_interval.tick().await;

            match self.refresh_mappings().await {
                Ok(count) => {
                    if count > 0 {
                        debug!("Background refresh: discovered {} mappings", count);
                    }
                }
                Err(e) => {
                    error!("Background refresh failed: {}", e);
                }
            }

            // Also cleanup old sessions
            self.cleanup_completed_sessions().await;
        }
    }
}

// Helper conversion
impl From<crate::client::mcp_tcp_client::PerformanceMetrics> for crate::client::mcp_tcp_client::TaskMetrics {
    fn from(perf: crate::client::mcp_tcp_client::PerformanceMetrics) -> Self {
        Self {
            total_tasks: perf.total_tasks,
            completed_tasks: perf.completed_tasks,
            active_tasks: perf.total_tasks - perf.completed_tasks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_bridge_creation() {
        let bridge = McpSessionBridge::new("multi-agent-container".to_string());

        let uuid = "test-uuid-123";
        let swarm_id = "swarm-test-456";

        bridge.link_session_to_swarm(uuid, swarm_id).await;

        assert_eq!(
            bridge.get_swarm_id_for_session(uuid).await,
            Some(swarm_id.to_string())
        );

        assert_eq!(
            bridge.get_session_for_swarm(swarm_id).await,
            Some(uuid.to_string())
        );
    }
}
