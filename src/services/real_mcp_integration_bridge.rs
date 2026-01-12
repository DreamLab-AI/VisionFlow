//! Real MCP Integration Bridge
//!
//! This service provides a bridge between VisionFlow and real MCP (Model Context Protocol)
//! servers, enabling direct communication with agent swarms and execution environments.
//! It handles JSON-RPC communication, agent management, and real-time data synchronization.

use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};

use crate::services::agent_visualization_protocol::{
    McpServerInfo, McpServerType, MultiMcpAgentStatus, SwarmTopologyData,
};
use crate::types::mcp_responses::{McpError, McpResponse};
use crate::utils::time;
use crate::utils::json::{from_json, to_json};

// Debug removed due to non-Debug trait object
pub struct RealMcpIntegrationBridge {
    
    connections: Arc<RwLock<HashMap<String, McpConnection>>>,

    
    pool_config: ConnectionPoolConfig,

    
    message_router: Arc<RwLock<MessageRouter>>,

    
    event_handlers: Arc<RwLock<Vec<Box<dyn McpEventHandler + Send + Sync>>>>,

    
    health_monitor: Arc<Mutex<HealthMonitor>>,

    
    auth_manager: Arc<RwLock<AuthenticationManager>>,

    
    request_tracker: Arc<RwLock<RequestTracker>>,

    
    stats: Arc<RwLock<IntegrationStats>>,
}

#[derive(Debug)]
pub struct McpConnection {
    pub server_id: String,
    pub server_type: McpServerType,
    pub host: String,
    pub port: u16,
    pub stream: Option<TcpStream>,
    pub reader: Option<BufReader<tokio::net::tcp::OwnedReadHalf>>,
    pub writer: Option<tokio::net::tcp::OwnedWriteHalf>,
    pub is_connected: bool,
    pub last_heartbeat: DateTime<Utc>,
    pub connection_attempts: u32,
    pub request_id_counter: u64,
    pub pending_requests: HashMap<u64, PendingRequest>,
    pub capabilities: ServerCapabilities,
    pub session_info: Option<SessionInfo>,
}

#[derive(Debug, Clone)]
pub struct PendingRequest {
    pub id: u64,
    pub method: String,
    pub sent_at: DateTime<Utc>,
    pub timeout_ms: u64,
    pub response_handler: String, 
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub supported_tools: Vec<String>,
    pub supported_resources: Vec<String>,
    pub supported_prompts: Vec<String>,
    pub experimental_features: Vec<String>,
    pub sampling_support: bool,
    pub logging_support: bool,
    pub roots_support: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub protocol_version: String,
    pub client_info: ClientInfo,
    pub server_info: ServerInfo,
    pub established_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub capabilities: ServerCapabilities,
}

#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    pub max_connections_per_server: u32,
    pub connection_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub retry_attempts: u32,
    pub retry_delay_ms: u64,
    pub keep_alive: bool,
    pub tcp_nodelay: bool,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_server: 5,
            connection_timeout_ms: 10000,
            heartbeat_interval_ms: 30000,
            retry_attempts: 3,
            retry_delay_ms: 1000,
            keep_alive: true,
            tcp_nodelay: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct MessageRouter {
    pub routes: HashMap<String, RouteConfig>,
    pub load_balancer: LoadBalancerConfig,
    pub retry_policy: RetryPolicy,
}

#[derive(Debug, Clone)]
pub struct RouteConfig {
    pub server_id: String,
    pub method_patterns: Vec<String>,
    pub priority: u32,
    pub load_balance: bool,
}

#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    pub strategy: LoadBalanceStrategy,
    pub health_check_enabled: bool,
    pub failover_enabled: bool,
}

#[derive(Debug, Clone)]
pub enum LoadBalanceStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRandom,
    HealthBased,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f32,
    pub retryable_errors: Vec<i32>,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalanceStrategy::RoundRobin,
            health_check_enabled: true,
            failover_enabled: true,
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            retryable_errors: vec![-32603, -32602], 
        }
    }
}

pub trait McpEventHandler {
    fn handle_connection_established(&self, server_id: &str, server_info: &ServerInfo);
    fn handle_connection_lost(&self, server_id: &str, error: &str);
    fn handle_agent_spawned(&self, server_id: &str, agent: &MultiMcpAgentStatus);
    fn handle_agent_updated(&self, server_id: &str, agent: &MultiMcpAgentStatus);
    fn handle_agent_terminated(&self, server_id: &str, agent_id: &str);
    fn handle_swarm_topology_changed(&self, server_id: &str, topology: &SwarmTopologyData);
    fn handle_server_error(&self, server_id: &str, error: &McpError);
}

#[derive(Debug, Default)]
pub struct HealthMonitor {
    pub server_health: HashMap<String, ServerHealth>,
    pub global_health: GlobalHealth,
    pub monitoring_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ServerHealth {
    pub server_id: String,
    pub is_healthy: bool,
    pub response_time_ms: f64,
    pub error_rate: f32,
    pub last_health_check: DateTime<Utc>,
    pub consecutive_failures: u32,
    pub uptime_percentage: f32,
}

#[derive(Debug, Clone, Default)]
pub struct GlobalHealth {
    pub total_servers: u32,
    pub healthy_servers: u32,
    pub average_response_time: f64,
    pub global_error_rate: f32,
    pub last_updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Default)]
pub struct AuthenticationManager {
    pub credentials: HashMap<String, ServerCredentials>,
    pub auth_cache: HashMap<String, AuthToken>,
    pub auth_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ServerCredentials {
    pub server_id: String,
    pub auth_type: AuthenticationType,
    pub username: Option<String>,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub token: Option<String>,
    pub certificate_path: Option<String>,
}

#[derive(Debug, Clone)]
pub enum AuthenticationType {
    None,
    Basic,
    Bearer,
    ApiKey,
    Certificate,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct AuthToken {
    pub token: String,
    pub expires_at: DateTime<Utc>,
    pub refresh_token: Option<String>,
}

#[derive(Debug, Default)]
pub struct RequestTracker {
    pub active_requests: HashMap<u64, TrackedRequest>,
    pub request_history: Vec<crate::types::mcp_responses::RequestHistoryEntry>,
    pub max_history_size: usize,
}

#[derive(Debug, Clone)]
pub struct TrackedRequest {
    pub id: u64,
    pub server_id: String,
    pub method: String,
    pub started_at: DateTime<Utc>,
    pub timeout_at: DateTime<Utc>,
    pub retry_count: u32,
}

#[derive(Debug, Default, Clone)]
pub struct IntegrationStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub requests_per_second: f64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub active_connections: u32,
    pub total_connections_created: u64,
    pub connection_errors: u64,
    pub uptime_seconds: u64,
    pub last_reset: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpMessage {
    Request(McpRequest),
    Response(McpResponse<Value>),
    Notification(McpNotification),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpNotification {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
}

impl Default for RealMcpIntegrationBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl RealMcpIntegrationBridge {
    
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            pool_config: ConnectionPoolConfig::default(),
            message_router: Arc::new(RwLock::new(MessageRouter::default())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
            health_monitor: Arc::new(Mutex::new(HealthMonitor::default())),
            auth_manager: Arc::new(RwLock::new(AuthenticationManager::default())),
            request_tracker: Arc::new(RwLock::new(RequestTracker {
                max_history_size: 1000,
                ..Default::default()
            })),
            stats: Arc::new(RwLock::new(IntegrationStats {
                last_reset: time::now(),
                ..Default::default()
            })),
        }
    }

    
    pub async fn initialize(&mut self, config: BridgeConfiguration) -> Result<(), String> {
        info!("Initializing Real MCP Integration Bridge");

        self.pool_config = config.connection_pool;

        
        if config.authentication_enabled {
            let mut auth_manager = self.auth_manager.write().await;
            auth_manager.auth_enabled = true;
            for cred in config.server_credentials {
                auth_manager
                    .credentials
                    .insert(cred.server_id.clone(), cred);
            }
        }

        
        let mut router = self.message_router.write().await;
        router.routes = config.routes;
        router.load_balancer = config.load_balancer;
        router.retry_policy = config.retry_policy;

        
        let mut health_monitor = self.health_monitor.lock().await;
        health_monitor.monitoring_enabled = config.health_monitoring_enabled;

        
        self.start_background_tasks().await;

        info!("Real MCP Integration Bridge initialized successfully");
        Ok(())
    }

    
    pub async fn connect_to_server(
        &self,
        server_id: String,
        server_type: McpServerType,
        host: String,
        port: u16,
    ) -> Result<(), String> {
        info!(
            "Connecting to MCP server: {} at {}:{}",
            server_id, host, port
        );

        let mut connections = self.connections.write().await;

        
        if let Some(existing) = connections.get(&server_id) {
            if existing.is_connected {
                warn!("Already connected to server: {}", server_id);
                return Ok(());
            }
        }

        
        let mut connection = McpConnection {
            server_id: server_id.clone(),
            server_type,
            host: host.clone(),
            port,
            stream: None,
            reader: None,
            writer: None,
            is_connected: false,
            last_heartbeat: time::now(),
            connection_attempts: 0,
            request_id_counter: 0,
            pending_requests: HashMap::new(),
            capabilities: ServerCapabilities::default(),
            session_info: None,
        };

        
        match self.establish_tcp_connection(&mut connection).await {
            Ok(_) => {
                
                match self.initialize_mcp_session(&mut connection).await {
                    Ok(session_info) => {
                        connection.session_info = Some(session_info.clone());
                        connection.is_connected = true;
                        connection.last_heartbeat = time::now();

                        connections.insert(server_id.clone(), connection);

                        
                        let mut stats = self.stats.write().await;
                        stats.active_connections += 1;
                        stats.total_connections_created += 1;

                        
                        self.notify_connection_established(&server_id, &session_info.server_info)
                            .await;

                        info!("Successfully connected to MCP server: {}", server_id);
                        Ok(())
                    }
                    Err(e) => {
                        error!("Failed to initialize MCP session with {}: {}", server_id, e);
                        Err(format!("Session initialization failed: {}", e))
                    }
                }
            }
            Err(e) => {
                error!("Failed to establish TCP connection to {}: {}", server_id, e);

                
                let mut stats = self.stats.write().await;
                stats.connection_errors += 1;

                Err(format!("TCP connection failed: {}", e))
            }
        }
    }

    
    pub async fn disconnect_from_server(&self, server_id: &str) -> Result<(), String> {
        info!("Disconnecting from MCP server: {}", server_id);

        let mut connections = self.connections.write().await;

        if let Some(mut connection) = connections.remove(server_id) {
            connection.is_connected = false;

            
            if let Some(mut writer) = connection.writer.take() {
                let _ = writer.shutdown().await;
            }

            
            let mut stats = self.stats.write().await;
            if stats.active_connections > 0 {
                stats.active_connections -= 1;
            }

            
            self.notify_connection_lost(server_id, "Disconnected by client")
                .await;

            info!("Disconnected from MCP server: {}", server_id);
            Ok(())
        } else {
            warn!("Server not connected: {}", server_id);
            Err(format!("Server not connected: {}", server_id))
        }
    }

    
    pub async fn send_request(
        &self,
        server_id: &str,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, String> {
        debug!("Sending MCP request: {} to server: {}", method, server_id);

        let mut connections = self.connections.write().await;
        let connection = connections
            .get_mut(server_id)
            .ok_or_else(|| format!("Server not connected: {}", server_id))?;

        if !connection.is_connected {
            return Err(format!("Server not connected: {}", server_id));
        }

        
        connection.request_id_counter += 1;
        let request_id = connection.request_id_counter;

        
        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Value::Number(serde_json::Number::from(request_id)),
            method: method.to_string(),
            params,
        };

        
        let pending_request = PendingRequest {
            id: request_id,
            method: method.to_string(),
            sent_at: time::now(),
            timeout_ms: self.pool_config.connection_timeout_ms,
            response_handler: "default".to_string(),
        };

        connection
            .pending_requests
            .insert(request_id, pending_request.clone());

        
        let mut tracker = self.request_tracker.write().await;
        tracker.active_requests.insert(
            request_id,
            TrackedRequest {
                id: request_id,
                server_id: server_id.to_string(),
                method: method.to_string(),
                started_at: time::now(),
                timeout_at: time::now()
                    + chrono::Duration::milliseconds(self.pool_config.connection_timeout_ms as i64),
                retry_count: 0,
            },
        );

        
        let request_json = to_json(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let message = format!("{}\n", request_json);

        if let Some(ref mut writer) = connection.writer {
            match writer.write_all(message.as_bytes()).await {
                Ok(_) => {
                    match writer.flush().await {
                        Ok(_) => {
                            
                            let mut stats = self.stats.write().await;
                            stats.total_requests += 1;
                            stats.bytes_sent += message.len() as u64;

                            debug!("Sent MCP request: {} (ID: {})", method, request_id);

                            
                            self.wait_for_response(server_id, request_id).await
                        }
                        Err(e) => {
                            connection.pending_requests.remove(&request_id);
                            tracker.active_requests.remove(&request_id);
                            Err(format!("Failed to flush request: {}", e))
                        }
                    }
                }
                Err(e) => {
                    connection.pending_requests.remove(&request_id);
                    tracker.active_requests.remove(&request_id);
                    Err(format!("Failed to send request: {}", e))
                }
            }
        } else {
            connection.pending_requests.remove(&request_id);
            tracker.active_requests.remove(&request_id);
            Err("No active writer for connection".to_string())
        }
    }

    
    pub async fn query_agent_list(
        &self,
        server_id: &str,
    ) -> Result<Vec<MultiMcpAgentStatus>, String> {
        debug!("Querying agent list from server: {}", server_id);

        let response = self.send_request(server_id, "agent_list", None).await?;

        
        if let Some(agents_array) = response.as_array() {
            let mut agents = Vec::new();

            for agent_value in agents_array {
                match serde_json::from_value::<MultiMcpAgentStatus>(agent_value.clone()) {
                    Ok(agent) => agents.push(agent),
                    Err(e) => warn!("Failed to parse agent data: {}", e),
                }
            }

            info!(
                "Retrieved {} agents from server: {}",
                agents.len(),
                server_id
            );
            Ok(agents)
        } else {
            warn!(
                "Invalid agent list response format from server: {}",
                server_id
            );
            Ok(Vec::new())
        }
    }

    
    pub async fn query_server_info(&self, server_id: &str) -> Result<McpServerInfo, String> {
        debug!("Querying server info from: {}", server_id);

        let response = self.send_request(server_id, "server_info", None).await?;

        match serde_json::from_value::<McpServerInfo>(response) {
            Ok(server_info) => {
                debug!("Retrieved server info from: {}", server_id);
                Ok(server_info)
            }
            Err(e) => {
                error!("Failed to parse server info from {}: {}", server_id, e);
                Err(format!("Failed to parse server info: {}", e))
            }
        }
    }

    
    pub async fn query_swarm_status(&self, server_id: &str) -> Result<SwarmTopologyData, String> {
        debug!("Querying swarm status from: {}", server_id);

        let response = self.send_request(server_id, "swarm_status", None).await?;

        match serde_json::from_value::<SwarmTopologyData>(response) {
            Ok(topology) => {
                debug!("Retrieved swarm topology from: {}", server_id);
                Ok(topology)
            }
            Err(e) => {
                error!("Failed to parse swarm topology from {}: {}", server_id, e);
                Err(format!("Failed to parse swarm topology: {}", e))
            }
        }
    }

    
    pub async fn execute_agent_task(
        &self,
        server_id: &str,
        agent_id: &str,
        task: &str,
        parameters: Option<Value>,
    ) -> Result<Value, String> {
        info!(
            "Executing agent task on {}: agent={}, task={}",
            server_id, agent_id, task
        );

        let params = json!({
            "agent_id": agent_id,
            "task": task,
            "parameters": parameters
        });

        let response = self
            .send_request(server_id, "agent_task", Some(params))
            .await?;

        info!("Agent task executed successfully on {}", server_id);
        Ok(response)
    }

    
    pub async fn spawn_agent(
        &self,
        server_id: &str,
        agent_type: &str,
        configuration: Option<Value>,
    ) -> Result<MultiMcpAgentStatus, String> {
        info!("Spawning agent on {}: type={}", server_id, agent_type);

        let params = json!({
            "agent_type": agent_type,
            "configuration": configuration
        });

        let response = self
            .send_request(server_id, "agent_spawn", Some(params))
            .await?;

        match serde_json::from_value::<MultiMcpAgentStatus>(response) {
            Ok(agent) => {
                info!(
                    "Agent spawned successfully on {}: {}",
                    server_id, agent.agent_id
                );

                
                self.notify_agent_spawned(server_id, &agent).await;

                Ok(agent)
            }
            Err(e) => {
                error!("Failed to parse spawned agent data: {}", e);
                Err(format!("Failed to parse agent data: {}", e))
            }
        }
    }

    
    pub async fn get_active_connections(&self) -> Vec<String> {
        let connections = self.connections.read().await;
        connections
            .iter()
            .filter(|(_, conn)| conn.is_connected)
            .map(|(id, _)| id.clone())
            .collect()
    }

    
    pub async fn get_connection_status(&self, server_id: &str) -> Option<ConnectionStatus> {
        let connections = self.connections.read().await;
        connections.get(server_id).map(|conn| ConnectionStatus {
            server_id: conn.server_id.clone(),
            server_type: conn.server_type.clone(),
            is_connected: conn.is_connected,
            last_heartbeat: conn.last_heartbeat,
            pending_requests: conn.pending_requests.len() as u32,
            capabilities: conn.capabilities.clone(),
        })
    }

    
    pub async fn get_statistics(&self) -> IntegrationStats {
        self.stats.read().await.clone()
    }

    
    pub async fn add_event_handler(&self, handler: Box<dyn McpEventHandler + Send + Sync>) {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
    }

    

    async fn establish_tcp_connection(&self, connection: &mut McpConnection) -> Result<(), String> {
        let address = format!("{}:{}", connection.host, connection.port);

        match tokio::time::timeout(
            tokio::time::Duration::from_millis(self.pool_config.connection_timeout_ms),
            TcpStream::connect(&address),
        )
        .await
        {
            Ok(Ok(stream)) => {
                
                if let Err(e) = stream.set_nodelay(self.pool_config.tcp_nodelay) {
                    warn!("Failed to set TCP_NODELAY: {}", e);
                }

                
                let (read_half, write_half) = stream.into_split();
                let reader = BufReader::new(read_half);

                connection.stream = None; 
                connection.reader = Some(reader);
                connection.writer = Some(write_half);
                connection.connection_attempts += 1;

                Ok(())
            }
            Ok(Err(e)) => Err(format!("TCP connection error: {}", e)),
            Err(_) => Err("TCP connection timeout".to_string()),
        }
    }

    async fn initialize_mcp_session(
        &self,
        connection: &mut McpConnection,
    ) -> Result<SessionInfo, String> {
        
        let init_params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "experimental": {},
                "sampling": {},
                "roots": {
                    "listChanged": true
                }
            },
            "clientInfo": {
                "name": "VisionFlow",
                "version": "1.0.0"
            }
        });

        
        

        connection.request_id_counter += 1;
        let request_id = connection.request_id_counter;

        let request = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Value::Number(serde_json::Number::from(request_id)),
            method: "initialize".to_string(),
            params: Some(init_params),
        };

        let request_json = to_json(&request)
            .map_err(|e| format!("Failed to serialize initialize request: {}", e))?;

        let message = format!("{}\n", request_json);

        if let Some(ref mut writer) = connection.writer {
            writer
                .write_all(message.as_bytes())
                .await
                .map_err(|e| format!("Failed to send initialize request: {}", e))?;

            writer
                .flush()
                .await
                .map_err(|e| format!("Failed to flush initialize request: {}", e))?;

            
            if let Some(ref mut reader) = connection.reader {
                let mut response_line = String::new();
                reader
                    .read_line(&mut response_line)
                    .await
                    .map_err(|e| format!("Failed to read initialize response: {}", e))?;

                let response: McpResponse<Value> = from_json(&response_line)
                    .map_err(|e| format!("Failed to parse initialize response: {}", e))?;

                match response {
                    McpResponse::Success(success_response) => {
                        
                        let server_capabilities = success_response
                            .result
                            .get("capabilities")
                            .and_then(|caps| {
                                serde_json::from_value::<ServerCapabilities>(caps.clone()).ok()
                            })
                            .unwrap_or_default();

                        let server_info = ServerInfo {
                            name: success_response
                                .result
                                .get("serverInfo")
                                .and_then(|info| info.get("name"))
                                .and_then(|name| name.as_str())
                                .unwrap_or("Unknown")
                                .to_string(),
                            version: success_response
                                .result
                                .get("serverInfo")
                                .and_then(|info| info.get("version"))
                                .and_then(|version| version.as_str())
                                .unwrap_or("Unknown")
                                .to_string(),
                            capabilities: server_capabilities.clone(),
                        };

                        connection.capabilities = server_capabilities;

                        let session_info = SessionInfo {
                            session_id: format!("session_{}", request_id),
                            protocol_version: "2024-11-05".to_string(),
                            client_info: ClientInfo {
                                name: "VisionFlow".to_string(),
                                version: "1.0.0".to_string(),
                            },
                            server_info,
                            established_at: time::now(),
                        };

                        
                        let initialized_notification = McpNotification {
                            jsonrpc: "2.0".to_string(),
                            method: "notifications/initialized".to_string(),
                            params: None,
                        };

                        let notification_json = to_json(&initialized_notification)
                            .map_err(|e| {
                                format!("Failed to serialize initialized notification: {}", e)
                            })?;

                        let notification_message = format!("{}\n", notification_json);

                        writer
                            .write_all(notification_message.as_bytes())
                            .await
                            .map_err(|e| {
                                format!("Failed to send initialized notification: {}", e)
                            })?;

                        writer.flush().await.map_err(|e| {
                            format!("Failed to flush initialized notification: {}", e)
                        })?;

                        Ok(session_info)
                    }
                    McpResponse::Error(error_response) => Err(format!(
                        "Initialize request failed: {} - {}",
                        error_response.error.code, error_response.error.message
                    )),
                }
            } else {
                Err("No reader available for connection".to_string())
            }
        } else {
            Err("No writer available for connection".to_string())
        }
    }

    async fn wait_for_response(&self, server_id: &str, request_id: u64) -> Result<Value, String> {
        debug!(
            "Waiting for response to request {} from server {}",
            request_id, server_id
        );

        let timeout_duration =
            tokio::time::Duration::from_millis(self.pool_config.connection_timeout_ms);
        let start_time = tokio::time::Instant::now();

        
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<Value, String>>(1);

        
        let connections = self.connections.clone();
        let server_id_clone = server_id.to_string();
        let stats = self.stats.clone();
        let request_tracker = self.request_tracker.clone();

        tokio::spawn(async move {
            let mut response_received = false;

            while start_time.elapsed() < timeout_duration && !response_received {
                let mut connections_guard = connections.write().await;

                if let Some(connection) = connections_guard.get_mut(&server_id_clone) {
                    if let Some(ref mut reader) = connection.reader {
                        
                        let mut response_line = String::new();

                        match tokio::time::timeout(
                            tokio::time::Duration::from_millis(100),
                            reader.read_line(&mut response_line),
                        )
                        .await
                        {
                            Ok(Ok(bytes_read)) => {
                                if bytes_read > 0 {
                                    debug!("Received response line: {}", response_line.trim());

                                    
                                    match serde_json::from_str::<McpResponse<Value>>(&response_line)
                                    {
                                        Ok(response) => {
                                            match response {
                                                McpResponse::Success(success_response) => {
                                                    
                                                    if let Some(id_val) = &success_response.id {
                                                        if let Some(id_num) = id_val.as_u64() {
                                                            if id_num == request_id {
                                                                
                                                                connection
                                                                    .pending_requests
                                                                    .remove(&request_id);

                                                                
                                                                let mut stats_guard =
                                                                    stats.write().await;
                                                                stats_guard.successful_requests +=
                                                                    1;
                                                                stats_guard.bytes_received +=
                                                                    response_line.len() as u64;

                                                                
                                                                let mut tracker =
                                                                    request_tracker.write().await;
                                                                if let Some(tracked_request) =
                                                                    tracker
                                                                        .active_requests
                                                                        .remove(&request_id)
                                                                {
                                                                    let duration = time::now()
                                                                        .signed_duration_since(
                                                                            tracked_request
                                                                                .started_at,
                                                                        );
                                                                    let history_entry = crate::types::mcp_responses::RequestHistoryEntry {
                                                                        id: request_id,
                                                                        server_id: server_id_clone.clone(),
                                                                        method: tracked_request.method,
                                                                        started_at: tracked_request.started_at,
                                                                        completed_at: Some(time::now()),
                                                                        duration_ms: Some(duration.num_milliseconds() as u64),
                                                                        success: true,
                                                                        error_message: None,
                                                                    };
                                                                    tracker
                                                                        .request_history
                                                                        .push(history_entry);

                                                                    
                                                                    if tracker.request_history.len()
                                                                        > tracker.max_history_size
                                                                    {
                                                                        tracker
                                                                            .request_history
                                                                            .remove(0);
                                                                    }
                                                                }

                                                                info!("Successfully received response for request {}", request_id);
                                                                let _ = tx
                                                                    .send(Ok(
                                                                        success_response.result
                                                                    ))
                                                                    .await;
                                                                response_received = true;
                                                            }
                                                        }
                                                    }
                                                }
                                                McpResponse::Error(error_response) => {
                                                    
                                                    if let Some(id_val) = &error_response.id {
                                                        if let Some(id_num) = id_val.as_u64() {
                                                            if id_num == request_id {
                                                                
                                                                connection
                                                                    .pending_requests
                                                                    .remove(&request_id);

                                                                
                                                                let mut stats_guard =
                                                                    stats.write().await;
                                                                stats_guard.failed_requests += 1;

                                                                
                                                                let mut tracker =
                                                                    request_tracker.write().await;
                                                                if let Some(tracked_request) =
                                                                    tracker
                                                                        .active_requests
                                                                        .remove(&request_id)
                                                                {
                                                                    let duration = time::now()
                                                                        .signed_duration_since(
                                                                            tracked_request
                                                                                .started_at,
                                                                        );
                                                                    let history_entry = crate::types::mcp_responses::RequestHistoryEntry {
                                                                        id: request_id,
                                                                        server_id: server_id_clone.clone(),
                                                                        method: tracked_request.method,
                                                                        started_at: tracked_request.started_at,
                                                                        completed_at: Some(time::now()),
                                                                        duration_ms: Some(duration.num_milliseconds() as u64),
                                                                        success: false,
                                                                        error_message: Some(error_response.error.message.clone()),
                                                                    };
                                                                    tracker
                                                                        .request_history
                                                                        .push(history_entry);
                                                                }

                                                                error!("Received error response for request {}: {}", request_id, error_response.error.message);
                                                                let _ = tx
                                                                    .send(Err(format!(
                                                                        "MCP Error {}: {}",
                                                                        error_response.error.code,
                                                                        error_response
                                                                            .error
                                                                            .message
                                                                    )))
                                                                    .await;
                                                                response_received = true;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            warn!(
                                                "Failed to parse JSON-RPC response: {} - Raw: {}",
                                                e,
                                                response_line.trim()
                                            );
                                        }
                                    }
                                }
                            }
                            Ok(Err(e)) => {
                                error!("Error reading from connection: {}", e);
                                break;
                            }
                            Err(_) => {
                                
                            }
                        }
                    } else {
                        error!("No reader available for connection to {}", server_id_clone);
                        break;
                    }
                } else {
                    error!("Connection not found for server {}", server_id_clone);
                    break;
                }

                drop(connections_guard);

                
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }

            
            if !response_received {
                warn!(
                    "Request {} to {} timed out after {:?}",
                    request_id, server_id_clone, timeout_duration
                );

                
                let mut connections_guard = connections.write().await;
                if let Some(connection) = connections_guard.get_mut(&server_id_clone) {
                    connection.pending_requests.remove(&request_id);
                }

                let mut stats_guard = stats.write().await;
                stats_guard.failed_requests += 1;

                let mut tracker = request_tracker.write().await;
                if let Some(tracked_request) = tracker.active_requests.remove(&request_id) {
                    let history_entry = crate::types::mcp_responses::RequestHistoryEntry {
                        id: request_id,
                        server_id: server_id_clone,
                        method: tracked_request.method,
                        started_at: tracked_request.started_at,
                        completed_at: Some(time::now()),
                        duration_ms: Some(timeout_duration.as_millis() as u64),
                        success: false,
                        error_message: Some("Request timeout".to_string()),
                    };
                    tracker.request_history.push(history_entry);
                }

                let _ = tx
                    .send(Err(format!("Request timeout after {:?}", timeout_duration)))
                    .await;
            }
        });

        
        match rx.recv().await {
            Some(result) => result,
            None => Err("Internal error: response channel closed".to_string()),
        }
    }

    async fn start_background_tasks(&self) {
        
        let connections = self.connections.clone();
        let pool_config = self.pool_config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(
                pool_config.heartbeat_interval_ms,
            ));

            loop {
                interval.tick().await;

                let connections_guard = connections.read().await;
                for (server_id, connection) in connections_guard.iter() {
                    if connection.is_connected {
                        
                        debug!("Sending heartbeat to server: {}", server_id);
                        
                    }
                }
            }
        });

        
        let request_tracker = self.request_tracker.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

            loop {
                interval.tick().await;

                let mut tracker = request_tracker.write().await;
                let now = time::now();

                
                tracker.active_requests.retain(|_, request| {
                    if now > request.timeout_at {
                        warn!("Request {} to {} timed out", request.id, request.server_id);
                        false
                    } else {
                        true
                    }
                });
            }
        });

        info!("Background tasks started for MCP integration bridge");
    }

    async fn notify_connection_established(&self, server_id: &str, server_info: &ServerInfo) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.handle_connection_established(server_id, server_info);
        }
    }

    async fn notify_connection_lost(&self, server_id: &str, error: &str) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.handle_connection_lost(server_id, error);
        }
    }

    async fn notify_agent_spawned(&self, server_id: &str, agent: &MultiMcpAgentStatus) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.handle_agent_spawned(server_id, agent);
        }
    }
}

#[derive(Debug, Clone)]
pub struct BridgeConfiguration {
    pub connection_pool: ConnectionPoolConfig,
    pub authentication_enabled: bool,
    pub server_credentials: Vec<ServerCredentials>,
    pub routes: HashMap<String, RouteConfig>,
    pub load_balancer: LoadBalancerConfig,
    pub retry_policy: RetryPolicy,
    pub health_monitoring_enabled: bool,
}

impl Default for BridgeConfiguration {
    fn default() -> Self {
        Self {
            connection_pool: ConnectionPoolConfig::default(),
            authentication_enabled: false,
            server_credentials: Vec::new(),
            routes: HashMap::new(),
            load_balancer: LoadBalancerConfig {
                strategy: LoadBalanceStrategy::RoundRobin,
                health_check_enabled: true,
                failover_enabled: true,
            },
            retry_policy: RetryPolicy {
                max_retries: 3,
                base_delay_ms: 1000,
                max_delay_ms: 10000,
                backoff_multiplier: 2.0,
                retryable_errors: vec![-32603, -32000], 
            },
            health_monitoring_enabled: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStatus {
    pub server_id: String,
    pub server_type: McpServerType,
    pub is_connected: bool,
    pub last_heartbeat: DateTime<Utc>,
    pub pending_requests: u32,
    pub capabilities: ServerCapabilities,
}
