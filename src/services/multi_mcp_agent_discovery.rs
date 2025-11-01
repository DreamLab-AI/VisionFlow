//! Multi-MCP Agent Discovery Service
//!
//! This service discovers and monitors agents across multiple MCP servers:
//! - Claude Flow (claude-flow MCP server)
//! - RuvSwarm (ruv-swarm MCP server)
//! - DAA (Decentralized Autonomous Agents)
//! - Custom MCP implementations
//!
//! It provides unified agent discovery, real-time monitoring, and topology analysis.

use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::services::agent_visualization_protocol::{
    GlobalPerformanceMetrics, McpServerInfo, McpServerType, MultiMcpAgentStatus, SwarmTopologyData,
};
use crate::utils::mcp_tcp_client::create_mcp_client;

///
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
            discovery_interval_ms: 5000, 
            timeout_ms: 10000,           
            retry_attempts: 3,
        }
    }
}

///
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

///
pub struct MultiMcpAgentDiscovery {
    servers: Arc<RwLock<HashMap<String, McpServerConfig>>>,
    discovered_agents: Arc<RwLock<HashMap<String, MultiMcpAgentStatus>>>,
    server_statuses: Arc<RwLock<HashMap<String, McpServerInfo>>>,
    topology_data: Arc<RwLock<HashMap<String, SwarmTopologyData>>>,
    stats: Arc<RwLock<DiscoveryStats>>,
    discovery_running: Arc<RwLock<bool>>,
}

impl MultiMcpAgentDiscovery {
    
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

    
    pub async fn initialize_default_servers(&self) {
        let mut servers = self.servers.write().await;

        
        servers.insert(
            "claude-flow".to_string(),
            McpServerConfig {
                server_id: "claude-flow".to_string(),
                server_type: McpServerType::ClaudeFlow,
                host: std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| "localhost".to_string()),
                port: std::env::var("MCP_TCP_PORT")
                    .unwrap_or_else(|_| "9500".to_string())
                    .parse()
                    .unwrap_or(9500),
                enabled: true,
                discovery_interval_ms: 3000,
                timeout_ms: 10000,
                retry_attempts: 3,
            },
        );

        
        servers.insert(
            "ruv-swarm".to_string(),
            McpServerConfig {
                server_id: "ruv-swarm".to_string(),
                server_type: McpServerType::RuvSwarm,
                host: std::env::var("RUV_SWARM_HOST").unwrap_or_else(|_| "localhost".to_string()),
                port: std::env::var("RUV_SWARM_PORT")
                    .unwrap_or_else(|_| "9501".to_string())
                    .parse()
                    .unwrap_or(9501),
                enabled: true,
                discovery_interval_ms: 3000,
                timeout_ms: 10000,
                retry_attempts: 3,
            },
        );

        
        servers.insert(
            "daa".to_string(),
            McpServerConfig {
                server_id: "daa".to_string(),
                server_type: McpServerType::Daa,
                host: std::env::var("DAA_HOST").unwrap_or_else(|_| "localhost".to_string()),
                port: std::env::var("DAA_PORT")
                    .unwrap_or_else(|_| "9502".to_string())
                    .parse()
                    .unwrap_or(9502),
                enabled: true,
                discovery_interval_ms: 5000,
                timeout_ms: 15000,
                retry_attempts: 2,
            },
        );

        info!(
            "Initialized {} default MCP servers for discovery",
            servers.len()
        );
    }

    
    pub async fn add_server(&self, config: McpServerConfig) {
        let mut servers = self.servers.write().await;
        info!(
            "Adding MCP server: {} ({}:{})",
            config.server_id, config.host, config.port
        );
        servers.insert(config.server_id.clone(), config);
    }

    
    pub async fn remove_server(&self, server_id: &str) {
        let mut servers = self.servers.write().await;
        if servers.remove(server_id).is_some() {
            info!("Removed MCP server: {}", server_id);

            
            let mut agents = self.discovered_agents.write().await;
            agents.retain(|_, agent| {
                !matches!(
                    (&agent.server_source, server_id),
                    (McpServerType::ClaudeFlow, "claude-flow")
                        | (McpServerType::RuvSwarm, "ruv-swarm")
                        | (McpServerType::Daa, "daa")
                )
            });
        }
    }

    
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

                            debug!(
                                "Successfully discovered agents from {} in {}ms",
                                server_id, discovery_time
                            );
                        }
                        Err(e) => {
                            error!("Failed to discover agents from {}: {}", server_id, e);
                            let mut stats_guard = stats.write().await;
                            stats_guard.failed_discoveries += 1;
                            drop(stats_guard);

                            
                            let mut server_statuses_guard = server_statuses.write().await;
                            if let Some(server_info) = server_statuses_guard.get_mut(&server_id) {
                                server_info.is_connected = false;
                                server_info.last_heartbeat = Utc::now().timestamp();
                            }
                            drop(server_statuses_guard);
                        }
                    }

                    
                    tokio::time::sleep(tokio::time::Duration::from_millis(
                        config.discovery_interval_ms,
                    ))
                    .await;
                }

                
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
            }

            info!("Multi-MCP agent discovery service stopped");
        });
    }

    
    pub async fn stop_discovery(&self) {
        let mut discovery_running = self.discovery_running.write().await;
        *discovery_running = false;
        info!("Stopping multi-MCP agent discovery service");
    }

    
    async fn discover_server_agents(
        config: &McpServerConfig,
    ) -> Result<
        (
            McpServerInfo,
            Vec<MultiMcpAgentStatus>,
            Option<SwarmTopologyData>,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let start_time = Utc::now();
        
        let _discovery_duration = start_time;

        debug!(
            "Discovering agents from {} server at {}:{}",
            config.server_id, config.host, config.port
        );

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

    
    async fn discover_claude_flow_agents(
        config: &McpServerConfig,
    ) -> Result<
        (
            McpServerInfo,
            Vec<MultiMcpAgentStatus>,
            Option<SwarmTopologyData>,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        debug!(
            "Discovering Claude Flow agents from real MCP server at {}:{}",
            config.host, config.port
        );

        let client = create_mcp_client(&config.server_type, &config.host, config.port);

        
        let is_connected = client.test_connection().await.unwrap_or(false);
        if !is_connected {
            return Err(format!(
                "Cannot connect to Claude Flow MCP server at {}:{}",
                config.host, config.port
            )
            .into());
        }

        
        if let Err(e) = client.initialize_session().await {
            warn!("Failed to initialize MCP session: {}", e);
        }

        
        let mut server_info = match client.query_server_info().await {
            Ok(info) => info,
            Err(e) => {
                warn!("Failed to get server info, using defaults: {}", e);
                McpServerInfo {
                    server_id: config.server_id.clone(),
                    server_type: config.server_type.clone(),
                    host: config.host.clone(),
                    port: config.port,
                    is_connected,
                    last_heartbeat: Utc::now().timestamp(),
                    supported_tools: vec![
                        "agent_list".to_string(),
                        "swarm_status".to_string(),
                        "server_info".to_string(),
                    ],
                    agent_count: 0,
                }
            }
        };

        
        let agents = match client.query_agent_list().await {
            Ok(agent_list) => {
                info!(
                    "Retrieved {} agents from Claude Flow MCP server",
                    agent_list.len()
                );
                
                agent_list
                    .into_iter()
                    .map(|mut agent| {
                        agent.server_source = McpServerType::ClaudeFlow;
                        agent
                    })
                    .collect()
            }
            Err(e) => {
                error!("Failed to query agent list from Claude Flow MCP: {}", e);
                Vec::new()
            }
        };

        
        server_info.agent_count = agents.len() as u32;

        
        let topology = match client.query_swarm_status().await {
            Ok(topology_data) => {
                info!("Retrieved topology data from Claude Flow MCP server");
                Some(topology_data)
            }
            Err(e) => {
                warn!("Failed to query swarm topology from Claude Flow MCP: {}", e);
                None
            }
        };

        Ok((server_info, agents, topology))
    }

    
    async fn discover_ruv_swarm_agents(
        config: &McpServerConfig,
    ) -> Result<
        (
            McpServerInfo,
            Vec<MultiMcpAgentStatus>,
            Option<SwarmTopologyData>,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        debug!(
            "Discovering RuvSwarm agents from real MCP server at {}:{}",
            config.host, config.port
        );

        let client = create_mcp_client(&config.server_type, &config.host, config.port);

        
        let is_connected = client.test_connection().await.unwrap_or(false);
        if !is_connected {
            return Err(format!(
                "Cannot connect to RuvSwarm MCP server at {}:{}",
                config.host, config.port
            )
            .into());
        }

        
        if let Err(e) = client.initialize_session().await {
            warn!("Failed to initialize MCP session: {}", e);
        }

        
        let mut server_info = match client.query_server_info().await {
            Ok(mut info) => {
                info.server_type = McpServerType::RuvSwarm;
                info
            }
            Err(e) => {
                warn!("Failed to get server info, using defaults: {}", e);
                McpServerInfo {
                    server_id: config.server_id.clone(),
                    server_type: config.server_type.clone(),
                    host: config.host.clone(),
                    port: config.port,
                    is_connected,
                    last_heartbeat: Utc::now().timestamp(),
                    supported_tools: vec![
                        "swarm_init".to_string(),
                        "agent_spawn".to_string(),
                        "daa_init".to_string(),
                        "neural_train".to_string(),
                        "benchmark_run".to_string(),
                    ],
                    agent_count: 0,
                }
            }
        };

        
        let agents = match client.query_agent_list().await {
            Ok(agent_list) => {
                info!(
                    "Retrieved {} agents from RuvSwarm MCP server",
                    agent_list.len()
                );
                
                agent_list
                    .into_iter()
                    .map(|mut agent| {
                        agent.server_source = McpServerType::RuvSwarm;
                        agent
                    })
                    .collect()
            }
            Err(e) => {
                error!("Failed to query agent list from RuvSwarm MCP: {}", e);
                Vec::new()
            }
        };

        
        server_info.agent_count = agents.len() as u32;

        
        let topology = match client.query_swarm_status().await {
            Ok(topology_data) => {
                info!("Retrieved topology data from RuvSwarm MCP server");
                Some(topology_data)
            }
            Err(e) => {
                warn!("Failed to query swarm topology from RuvSwarm MCP: {}", e);
                None
            }
        };

        Ok((server_info, agents, topology))
    }

    
    async fn discover_daa_agents(
        config: &McpServerConfig,
    ) -> Result<
        (
            McpServerInfo,
            Vec<MultiMcpAgentStatus>,
            Option<SwarmTopologyData>,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        debug!(
            "Discovering DAA agents from real MCP server at {}:{}",
            config.host, config.port
        );

        let client = create_mcp_client(&config.server_type, &config.host, config.port);

        
        let is_connected = client.test_connection().await.unwrap_or(false);
        if !is_connected {
            return Err(format!(
                "Cannot connect to DAA MCP server at {}:{}",
                config.host, config.port
            )
            .into());
        }

        
        if let Err(e) = client.initialize_session().await {
            warn!("Failed to initialize MCP session: {}", e);
        }

        
        let mut server_info = match client.query_server_info().await {
            Ok(mut info) => {
                info.server_type = McpServerType::Daa;
                info
            }
            Err(e) => {
                warn!("Failed to get server info, using defaults: {}", e);
                McpServerInfo {
                    server_id: config.server_id.clone(),
                    server_type: config.server_type.clone(),
                    host: config.host.clone(),
                    port: config.port,
                    is_connected,
                    last_heartbeat: Utc::now().timestamp(),
                    supported_tools: vec![
                        "daa_agent_create".to_string(),
                        "daa_workflow_create".to_string(),
                        "daa_knowledge_share".to_string(),
                        "daa_learning_status".to_string(),
                    ],
                    agent_count: 0,
                }
            }
        };

        
        let agents = match client.query_agent_list().await {
            Ok(agent_list) => {
                info!("Retrieved {} agents from DAA MCP server", agent_list.len());
                
                agent_list
                    .into_iter()
                    .map(|mut agent| {
                        agent.server_source = McpServerType::Daa;
                        agent
                    })
                    .collect()
            }
            Err(e) => {
                error!("Failed to query agent list from DAA MCP: {}", e);
                Vec::new()
            }
        };

        
        server_info.agent_count = agents.len() as u32;

        
        let topology = match client.query_swarm_status().await {
            Ok(topology_data) => {
                info!("Retrieved topology data from DAA MCP server");
                Some(topology_data)
            }
            Err(e) => {
                warn!("Failed to query swarm topology from DAA MCP: {}", e);
                None
            }
        };

        Ok((server_info, agents, topology))
    }

    
    pub async fn get_all_agents(&self) -> Vec<MultiMcpAgentStatus> {
        self.discovered_agents
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    
    pub async fn get_agents_by_server(
        &self,
        server_type: &McpServerType,
    ) -> Vec<MultiMcpAgentStatus> {
        self.discovered_agents
            .read()
            .await
            .values()
            .filter(|agent| {
                std::mem::discriminant(&agent.server_source) == std::mem::discriminant(server_type)
            })
            .cloned()
            .collect()
    }

    
    pub async fn get_server_statuses(&self) -> Vec<McpServerInfo> {
        self.server_statuses
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    
    pub async fn get_discovery_stats(&self) -> DiscoveryStats {
        self.stats.read().await.clone()
    }

    
    pub async fn get_topology_data(&self) -> HashMap<String, SwarmTopologyData> {
        self.topology_data.read().await.clone()
    }

    
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
        let average_latency: f32 = agent_list
            .iter()
            .map(|a| a.performance.response_time_ms)
            .sum::<f32>()
            / agent_list.len() as f32;
        let resource_utilization: f32 = agent_list
            .iter()
            .map(|a| (a.performance.cpu_usage + a.performance.memory_usage) / 2.0)
            .sum::<f32>()
            / agent_list.len() as f32;

        let total_tasks: u32 = agent_list
            .iter()
            .map(|a| a.performance.tasks_completed + a.performance.tasks_failed)
            .sum();
        let failed_tasks: u32 = agent_list.iter().map(|a| a.performance.tasks_failed).sum();
        let error_rate = if total_tasks > 0 {
            failed_tasks as f32 / total_tasks as f32
        } else {
            0.0
        };

        GlobalPerformanceMetrics {
            total_throughput,
            average_latency,
            system_efficiency: (total_throughput / agent_list.len() as f32).min(1.0),
            resource_utilization,
            error_rate,
            coordination_overhead: self.calculate_coordination_overhead(&agent_list),
        }
    }

    
    fn calculate_coordination_overhead(&self, agents: &[&MultiMcpAgentStatus]) -> f32 {
        if agents.is_empty() {
            return 0.0;
        }

        
        
        
        

        let agent_count = agents.len() as f32;
        let avg_queue_size = agents
            .iter()
            .map(|a| a.metadata.task_queue_size as f32)
            .sum::<f32>()
            / agent_count;

        
        let mut server_types = std::collections::HashSet::new();
        for agent in agents {
            server_types.insert(std::mem::discriminant(&agent.server_source));
        }
        let server_diversity = server_types.len() as f32;

        
        let base_overhead = (agent_count.ln() / 10.0).min(0.3);

        
        let queue_overhead = (avg_queue_size / 10.0).min(0.2);

        
        let cross_server_overhead = if server_diversity > 1.0 {
            (server_diversity - 1.0) * 0.05
        } else {
            0.0
        };

        (base_overhead + queue_overhead + cross_server_overhead).min(0.8)
    }

    
    pub async fn is_any_server_online(&self) -> bool {
        self.server_statuses
            .read()
            .await
            .values()
            .any(|server| server.is_connected)
    }

    
    pub async fn get_total_agent_count(&self) -> u32 {
        self.discovered_agents.read().await.len() as u32
    }
}

impl Default for MultiMcpAgentDiscovery {
    fn default() -> Self {
        Self::new()
    }
}
