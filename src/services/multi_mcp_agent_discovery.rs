//! Multi-MCP Agent Discovery Service
//! 
//! This service discovers and monitors agents across multiple MCP servers:
//! - Claude Flow (claude-flow MCP server)
//! - RuvSwarm (ruv-swarm MCP server) 
//! - DAA (Decentralized Autonomous Agents)
//! - Custom MCP implementations
//!
//! It provides unified agent discovery, real-time monitoring, and topology analysis.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use log::{info, warn, error, debug};

use crate::services::agent_visualization_protocol::{
    McpServerInfo, McpServerType, MultiMcpAgentStatus, AgentExtendedMetadata,
    TopologyPosition, AgentPerformanceData, NeuralAgentData, SwarmTopologyData,
    LayerLoad, CriticalPath, Bottleneck, GlobalPerformanceMetrics
};

/// Configuration for MCP server discovery
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub server_id: String,
    pub server_type: McpServerType,
    pub host: String,
    pub port: u16,
    pub enabled: bool,
    pub discovery_interval_ms: u64,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            server_id: "unknown".to_string(),
            server_type: McpServerType::Custom("unknown".to_string()),
            host: "localhost".to_string(),
            port: 9500,
            enabled: true,
            discovery_interval_ms: 5000, // 5 seconds
            timeout_ms: 10000, // 10 seconds
            retry_attempts: 3,
        }
    }
}

/// Agent discovery and monitoring statistics
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    pub total_discoveries: u64,
    pub successful_discoveries: u64,
    pub failed_discoveries: u64,
    pub total_agents_discovered: u64,
    pub last_discovery_time: Option<DateTime<Utc>>,
    pub average_discovery_time_ms: f64,
    pub servers_online: u32,
    pub servers_offline: u32,
}

/// Multi-MCP Agent Discovery Service
pub struct MultiMcpAgentDiscovery {
    servers: Arc<RwLock<HashMap<String, McpServerConfig>>>,
    discovered_agents: Arc<RwLock<HashMap<String, MultiMcpAgentStatus>>>,
    server_statuses: Arc<RwLock<HashMap<String, McpServerInfo>>>,
    topology_data: Arc<RwLock<HashMap<String, SwarmTopologyData>>>,
    stats: Arc<RwLock<DiscoveryStats>>,
    discovery_running: Arc<RwLock<bool>>,
}

impl MultiMcpAgentDiscovery {
    /// Create new discovery service
    pub fn new() -> Self {
        Self {
            servers: Arc::new(RwLock::new(HashMap::new())),
            discovered_agents: Arc::new(RwLock::new(HashMap::new())),
            server_statuses: Arc::new(RwLock::new(HashMap::new())),
            topology_data: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DiscoveryStats::default())),
            discovery_running: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize with default MCP server configurations
    pub async fn initialize_default_servers(&self) {
        let mut servers = self.servers.write().await;
        
        // Claude Flow MCP server
        servers.insert("claude-flow".to_string(), McpServerConfig {
            server_id: "claude-flow".to_string(),
            server_type: McpServerType::ClaudeFlow,
            host: std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string()).parse().unwrap_or(9500),
            enabled: true,
            discovery_interval_ms: 3000,
            timeout_ms: 10000,
            retry_attempts: 3,
        });

        // RuvSwarm MCP server  
        servers.insert("ruv-swarm".to_string(), McpServerConfig {
            server_id: "ruv-swarm".to_string(),
            server_type: McpServerType::RuvSwarm,
            host: std::env::var("RUV_SWARM_HOST").unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("RUV_SWARM_PORT").unwrap_or_else(|_| "9501".to_string()).parse().unwrap_or(9501),
            enabled: true,
            discovery_interval_ms: 3000,
            timeout_ms: 10000,
            retry_attempts: 3,
        });

        // DAA MCP server
        servers.insert("daa".to_string(), McpServerConfig {
            server_id: "daa".to_string(),
            server_type: McpServerType::Daa,
            host: std::env::var("DAA_HOST").unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("DAA_PORT").unwrap_or_else(|_| "9502".to_string()).parse().unwrap_or(9502),
            enabled: true,
            discovery_interval_ms: 5000,
            timeout_ms: 15000,
            retry_attempts: 2,
        });

        info!("Initialized {} default MCP servers for discovery", servers.len());
    }

    /// Add or update MCP server configuration
    pub async fn add_server(&self, config: McpServerConfig) {
        let mut servers = self.servers.write().await;
        info!("Adding MCP server: {} ({}:{})", config.server_id, config.host, config.port);
        servers.insert(config.server_id.clone(), config);
    }

    /// Remove MCP server from discovery
    pub async fn remove_server(&self, server_id: &str) {
        let mut servers = self.servers.write().await;
        if servers.remove(server_id).is_some() {
            info!("Removed MCP server: {}", server_id);
            
            // Also remove discovered agents from this server
            let mut agents = self.discovered_agents.write().await;
            agents.retain(|_, agent| {
                !matches!(
                    (&agent.server_source, server_id),
                    (McpServerType::ClaudeFlow, "claude-flow") |
                    (McpServerType::RuvSwarm, "ruv-swarm") |
                    (McpServerType::Daa, "daa")
                )
            });
        }
    }

    /// Start continuous agent discovery across all servers
    pub async fn start_discovery(&self) {
        let mut discovery_running = self.discovery_running.write().await;
        if *discovery_running {
            warn!("Discovery already running");
            return;
        }
        *discovery_running = true;
        drop(discovery_running);

        info!("Starting multi-MCP agent discovery service");

        let servers = self.servers.clone();
        let discovered_agents = self.discovered_agents.clone();
        let server_statuses = self.server_statuses.clone();
        let topology_data = self.topology_data.clone();
        let stats = self.stats.clone();
        let discovery_running = self.discovery_running.clone();

        tokio::spawn(async move {
            while *discovery_running.read().await {
                let servers_config = servers.read().await.clone();
                
                for (server_id, config) in servers_config {
                    if !config.enabled {
                        continue;
                    }

                    let start_time = std::time::Instant::now();
                    
                    match Self::discover_server_agents(&config).await {
                        Ok((server_info, agents, topology)) => {
                            let mut server_statuses_guard = server_statuses.write().await;
                            server_statuses_guard.insert(server_id.clone(), server_info);
                            drop(server_statuses_guard);

                            let mut agents_guard = discovered_agents.write().await;
                            for agent in agents {
                                agents_guard.insert(agent.agent_id.clone(), agent);
                            }
                            drop(agents_guard);

                            if let Some(topo) = topology {
                                let mut topology_guard = topology_data.write().await;
                                topology_guard.insert(server_id.clone(), topo);
                                drop(topology_guard);
                            }

                            let discovery_time = start_time.elapsed().as_millis() as f64;
                            let mut stats_guard = stats.write().await;
                            stats_guard.successful_discoveries += 1;
                            stats_guard.last_discovery_time = Some(Utc::now());
                            stats_guard.average_discovery_time_ms = 
                                (stats_guard.average_discovery_time_ms + discovery_time) / 2.0;
                            drop(stats_guard);

                            debug!("Successfully discovered agents from {} in {}ms", server_id, discovery_time);
                        }
                        Err(e) => {
                            error!("Failed to discover agents from {}: {}", server_id, e);
                            let mut stats_guard = stats.write().await;
                            stats_guard.failed_discoveries += 1;
                            drop(stats_guard);

                            // Update server status to offline
                            let mut server_statuses_guard = server_statuses.write().await;
                            if let Some(server_info) = server_statuses_guard.get_mut(&server_id) {
                                server_info.is_connected = false;
                                server_info.last_heartbeat = Utc::now().timestamp();
                            }
                            drop(server_statuses_guard);
                        }
                    }

                    // Wait before next discovery for this server
                    tokio::time::sleep(tokio::time::Duration::from_millis(config.discovery_interval_ms)).await;
                }

                // Overall discovery cycle delay
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
            }

            info!("Multi-MCP agent discovery service stopped");
        });
    }

    /// Stop agent discovery
    pub async fn stop_discovery(&self) {
        let mut discovery_running = self.discovery_running.write().await;
        *discovery_running = false;
        info!("Stopping multi-MCP agent discovery service");
    }

    /// Discover agents from a specific MCP server
    async fn discover_server_agents(
        config: &McpServerConfig,
    ) -> Result<(McpServerInfo, Vec<MultiMcpAgentStatus>, Option<SwarmTopologyData>), Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Utc::now();
        
        debug!("Discovering agents from {} server at {}:{}", config.server_id, config.host, config.port);

        match &config.server_type {
            McpServerType::ClaudeFlow => Self::discover_claude_flow_agents(config).await,
            McpServerType::RuvSwarm => Self::discover_ruv_swarm_agents(config).await,
            McpServerType::Daa => Self::discover_daa_agents(config).await,
            McpServerType::Custom(name) => {
                warn!("Custom MCP server type '{}' not implemented", name);
                Err("Custom server type not implemented".into())
            }
        }
    }

    /// Discover agents from Claude Flow MCP server
    async fn discover_claude_flow_agents(
        config: &McpServerConfig,
    ) -> Result<(McpServerInfo, Vec<MultiMcpAgentStatus>, Option<SwarmTopologyData>), Box<dyn std::error::Error + Send + Sync>> {
        // This would use the existing TCP connection to Claude Flow MCP
        // For now, return mock data
        debug!("Discovering Claude Flow agents...");

        let server_info = McpServerInfo {
            server_id: config.server_id.clone(),
            server_type: config.server_type.clone(),
            host: config.host.clone(),
            port: config.port,
            is_connected: true,
            last_heartbeat: Utc::now().timestamp(),
            supported_tools: vec![
                "swarm_init".to_string(),
                "agent_spawn".to_string(),
                "agent_list".to_string(),
                "task_orchestrate".to_string(),
                "swarm_status".to_string(),
                "neural_status".to_string(),
            ],
            agent_count: 0, // Will be updated with actual count
        };

        // In real implementation, this would call the actual MCP tools
        // For now, return empty results
        let agents = vec![];
        let topology = None;

        Ok((server_info, agents, topology))
    }

    /// Discover agents from RuvSwarm MCP server
    async fn discover_ruv_swarm_agents(
        config: &McpServerConfig,
    ) -> Result<(McpServerInfo, Vec<MultiMcpAgentStatus>, Option<SwarmTopologyData>), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Discovering RuvSwarm agents...");

        let server_info = McpServerInfo {
            server_id: config.server_id.clone(),
            server_type: config.server_type.clone(),
            host: config.host.clone(),
            port: config.port,
            is_connected: true,
            last_heartbeat: Utc::now().timestamp(),
            supported_tools: vec![
                "swarm_init".to_string(),
                "agent_spawn".to_string(),
                "daa_init".to_string(),
                "neural_train".to_string(),
                "benchmark_run".to_string(),
            ],
            agent_count: 0,
        };

        let agents = vec![];
        let topology = None;

        Ok((server_info, agents, topology))
    }

    /// Discover agents from DAA MCP server
    async fn discover_daa_agents(
        config: &McpServerConfig,
    ) -> Result<(McpServerInfo, Vec<MultiMcpAgentStatus>, Option<SwarmTopologyData>), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Discovering DAA agents...");

        let server_info = McpServerInfo {
            server_id: config.server_id.clone(),
            server_type: config.server_type.clone(),
            host: config.host.clone(),
            port: config.port,
            is_connected: true,
            last_heartbeat: Utc::now().timestamp(),
            supported_tools: vec![
                "daa_agent_create".to_string(),
                "daa_workflow_create".to_string(),
                "daa_knowledge_share".to_string(),
                "daa_learning_status".to_string(),
            ],
            agent_count: 0,
        };

        let agents = vec![];
        let topology = None;

        Ok((server_info, agents, topology))
    }

    /// Get all discovered agents
    pub async fn get_all_agents(&self) -> Vec<MultiMcpAgentStatus> {
        self.discovered_agents.read().await.values().cloned().collect()
    }

    /// Get agents by server type
    pub async fn get_agents_by_server(&self, server_type: &McpServerType) -> Vec<MultiMcpAgentStatus> {
        self.discovered_agents.read().await
            .values()
            .filter(|agent| std::mem::discriminant(&agent.server_source) == std::mem::discriminant(server_type))
            .cloned()
            .collect()
    }

    /// Get all server statuses
    pub async fn get_server_statuses(&self) -> Vec<McpServerInfo> {
        self.server_statuses.read().await.values().cloned().collect()
    }

    /// Get discovery statistics
    pub async fn get_discovery_stats(&self) -> DiscoveryStats {
        self.stats.read().await.clone()
    }

    /// Get topology data for all swarms
    pub async fn get_topology_data(&self) -> HashMap<String, SwarmTopologyData> {
        self.topology_data.read().await.clone()
    }

    /// Generate global performance metrics across all servers
    pub async fn get_global_performance_metrics(&self) -> GlobalPerformanceMetrics {
        let agents = self.discovered_agents.read().await;
        let agent_list: Vec<&MultiMcpAgentStatus> = agents.values().collect();

        if agent_list.is_empty() {
            return GlobalPerformanceMetrics {
                total_throughput: 0.0,
                average_latency: 0.0,
                system_efficiency: 0.0,
                resource_utilization: 0.0,
                error_rate: 0.0,
                coordination_overhead: 0.0,
            };
        }

        let total_throughput: f32 = agent_list.iter().map(|a| a.performance.throughput).sum();
        let average_latency: f32 = agent_list.iter().map(|a| a.performance.response_time_ms).sum::<f32>() / agent_list.len() as f32;
        let resource_utilization: f32 = agent_list.iter().map(|a| (a.performance.cpu_usage + a.performance.memory_usage) / 2.0).sum::<f32>() / agent_list.len() as f32;
        
        let total_tasks: u32 = agent_list.iter().map(|a| a.performance.tasks_completed + a.performance.tasks_failed).sum();
        let failed_tasks: u32 = agent_list.iter().map(|a| a.performance.tasks_failed).sum();
        let error_rate = if total_tasks > 0 { failed_tasks as f32 / total_tasks as f32 } else { 0.0 };

        GlobalPerformanceMetrics {
            total_throughput,
            average_latency,
            system_efficiency: (total_throughput / agent_list.len() as f32).min(1.0),
            resource_utilization,
            error_rate,
            coordination_overhead: 0.15, // TODO: Calculate from actual coordination metrics
        }
    }

    /// Check if any server is online
    pub async fn is_any_server_online(&self) -> bool {
        self.server_statuses.read().await
            .values()
            .any(|server| server.is_connected)
    }

    /// Get total agent count across all servers
    pub async fn get_total_agent_count(&self) -> u32 {
        self.discovered_agents.read().await.len() as u32
    }
}

impl Default for MultiMcpAgentDiscovery {
    fn default() -> Self {
        Self::new()
    }
}