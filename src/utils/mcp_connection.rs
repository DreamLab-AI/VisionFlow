use log::{debug, error, info, warn};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Persistent MCP Connection that maintains the stream
pub struct PersistentMCPConnection {
    stream: Arc<Mutex<TcpStream>>,
    session_id: String,
    initialized: bool,
}

impl PersistentMCPConnection {
    /// Create and initialize a new MCP connection
    pub async fn new(
        host: &str,
        port: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let addr = format!("{}:{}", host, port);
        info!("Connecting to MCP server at {}", addr);

        let mut stream = TcpStream::connect(&addr).await?;
        info!("TCP connection established to MCP server");

        let session_id = Uuid::new_v4().to_string();

        // Send initialization request
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": session_id.clone(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "VisionFlow-BotsClient",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": {
                        "listChanged": true
                    }
                }
            }
        });

        let msg = format!("{}\n", init_request.to_string());
        debug!("Sending MCP init: {}", msg.trim());
        stream.write_all(msg.as_bytes()).await?;
        stream.flush().await?;

        // Read response directly without BufReader
        let mut buffer = Vec::new();
        let mut byte = [0u8; 1];

        // Read until we get our initialization response
        loop {
            buffer.clear();

            // Read line byte by byte
            loop {
                match tokio::time::timeout(Duration::from_secs(5), stream.read_exact(&mut byte))
                    .await
                {
                    Ok(Ok(_)) => {
                        if byte[0] == b'\n' {
                            break;
                        }
                        buffer.push(byte[0]);
                    }
                    Ok(Err(e)) => {
                        error!("Error reading from stream: {}", e);
                        return Err(Box::new(e));
                    }
                    Err(_) => {
                        error!("Timeout reading MCP initialization response");
                        return Err("MCP initialization timeout".into());
                    }
                }
            }

            let response_line = String::from_utf8_lossy(&buffer);
            debug!("MCP response: {}", response_line.trim());

            // Skip server.initialized messages
            if response_line.contains("server.initialized") {
                continue;
            }

            // Parse actual response
            if let Ok(response) = serde_json::from_str::<Value>(&response_line) {
                if response.get("id").and_then(|id| id.as_str()) == Some(&session_id) {
                    if response.get("result").is_some() {
                        info!("MCP session initialized: {}", session_id);

                        return Ok(PersistentMCPConnection {
                            stream: Arc::new(Mutex::new(stream)),
                            session_id,
                            initialized: true,
                        });
                    } else if let Some(error) = response.get("error") {
                        error!("MCP init error: {:?}", error);
                        return Err(format!("MCP initialization failed: {:?}", error).into());
                    }
                }
            }
        }
    }

    /// Execute a command on the persistent connection
    pub async fn execute_command(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Err("Connection not initialized".into());
        }

        let request_id = Uuid::new_v4().to_string();
        let request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        });

        let msg = format!("{}\n", request.to_string());
        debug!("Sending MCP command: {}", msg.trim());

        // Lock the stream and send command
        let mut stream = self.stream.lock().await;
        stream.write_all(msg.as_bytes()).await?;
        stream.flush().await?;

        // Read response
        let mut buffer = Vec::new();
        let mut byte = [0u8; 1];

        // Read until we get our response
        loop {
            buffer.clear();

            // Read line byte by byte
            loop {
                match tokio::time::timeout(Duration::from_secs(10), stream.read_exact(&mut byte))
                    .await
                {
                    Ok(Ok(_)) => {
                        if byte[0] == b'\n' {
                            break;
                        }
                        buffer.push(byte[0]);
                    }
                    Ok(Err(e)) => {
                        error!("Error reading from stream: {}", e);
                        return Err(Box::new(e));
                    }
                    Err(_) => {
                        error!("Timeout reading MCP response");
                        return Err("MCP response timeout".into());
                    }
                }
            }

            let response_line = String::from_utf8_lossy(&buffer);
            let trimmed = response_line.trim();

            if trimmed.is_empty() {
                continue;
            }

            debug!("MCP response: {}", trimmed);

            // Skip notifications
            if trimmed.contains("server.initialized") {
                continue;
            }

            // Parse response
            if let Ok(response) = serde_json::from_str::<Value>(trimmed) {
                // Check if this is our response
                if response.get("id").and_then(|id| id.as_str()) == Some(&request_id) {
                    if let Some(result) = response.get("result") {
                        info!("MCP command '{}' executed successfully", method);
                        return Ok(result.clone());
                    } else if let Some(error) = response.get("error") {
                        error!("MCP command error: {:?}", error);
                        return Err(format!("MCP error: {:?}", error).into());
                    }
                } else if response.get("method").is_some() {
                    // This is a notification, skip it
                    continue;
                }
            }
        }
    }
}

/// MCP Connection Pool for managing multiple persistent connections
#[derive(Clone)]
pub struct MCPConnectionPool {
    connections: Arc<RwLock<HashMap<String, Arc<PersistentMCPConnection>>>>,
    host: String,
    port: String,
    max_retries: u32,
    retry_delay: Duration,
}

impl MCPConnectionPool {
    pub fn new(host: String, port: String) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            host,
            port,
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
        }
    }

    /// Get or create a connection for a specific purpose
    pub async fn get_connection(
        &self,
        purpose: &str,
    ) -> Result<Arc<PersistentMCPConnection>, Box<dyn std::error::Error + Send + Sync>> {
        // Check if we have an existing connection
        {
            let connections = self.connections.read().await;
            if let Some(conn) = connections.get(purpose) {
                debug!("Reusing existing MCP connection for {}", purpose);
                return Ok(Arc::clone(conn));
            }
        }

        // Create new connection
        info!("Creating new MCP connection for {}", purpose);

        for attempt in 1..=self.max_retries {
            info!("Connection attempt {}/{}", attempt, self.max_retries);

            match PersistentMCPConnection::new(&self.host, &self.port).await {
                Ok(conn) => {
                    let conn = Arc::new(conn);

                    // Store in pool
                    let mut connections = self.connections.write().await;
                    connections.insert(purpose.to_string(), Arc::clone(&conn));

                    info!("MCP connection established for {}", purpose);
                    return Ok(conn);
                }
                Err(e) => {
                    warn!("Failed to create connection (attempt {}): {}", attempt, e);
                    if attempt < self.max_retries {
                        tokio::time::sleep(self.retry_delay).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err("Failed to establish MCP connection after all retries".into())
    }

    /// Execute a command using the connection pool
    pub async fn execute_command(
        &self,
        purpose: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.get_connection(purpose).await?;
        conn.execute_command(method, params).await
    }

    /// Remove a connection from the pool
    pub async fn remove_connection(&self, purpose: &str) {
        let mut connections = self.connections.write().await;
        if connections.remove(purpose).is_some() {
            info!("Removed MCP connection for {}", purpose);
        }
    }
}

/// Simplified function to call swarm_init
pub async fn call_swarm_init(
    host: &str,
    port: &str,
    topology: &str,
    max_agents: u32,
    strategy: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());

    let params = json!({
        "name": "swarm_init",
        "arguments": {
            "topology": topology,
            "maxAgents": max_agents,
            "strategy": strategy
        }
    });

    pool.execute_command("swarm_init", "tools/call", params)
        .await
}

/// Simplified function to list agents
pub async fn call_agent_list(
    host: &str,
    port: &str,
    filter: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());

    let params = json!({
        "name": "agent_list",
        "arguments": {
            "filter": filter
        }
    });

    pool.execute_command("agent_list", "tools/call", params)
        .await
}

/// Simplified function to destroy a swarm
pub async fn call_swarm_destroy(
    host: &str,
    port: &str,
    swarm_id: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());

    info!("Calling swarm_destroy for swarm_id: {}", swarm_id);

    let params = json!({
        "name": "swarm_destroy",
        "arguments": {
            "swarmId": swarm_id
        }
    });

    pool.execute_command("swarm_destroy", "tools/call", params)
        .await
}

/// Simplified function to spawn an agent
pub async fn call_agent_spawn(
    host: &str,
    port: &str,
    agent_type: &str,
    swarm_id: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());

    info!(
        "Spawning agent of type: {} in swarm: {}",
        agent_type, swarm_id
    );

    let params = json!({
        "name": "agent_spawn",
        "arguments": {
            "type": agent_type,
            "swarmId": swarm_id
        }
    });

    pool.execute_command("agent_spawn", "tools/call", params)
        .await
}

/// Simplified function to orchestrate a task
pub async fn call_task_orchestrate(
    host: &str,
    port: &str,
    task: &str,
    priority: Option<&str>,
    strategy: Option<&str>,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());

    info!("Orchestrating task: {}", task);

    let mut args = json!({
        "task": task
    });

    if let Some(p) = priority {
        args["priority"] = json!(p);
    }

    if let Some(s) = strategy {
        args["strategy"] = json!(s);
    }

    let params = json!({
        "name": "task_orchestrate",
        "arguments": args
    });

    pool.execute_command("task_orchestrate", "tools/call", params)
        .await
}

/// Hybrid task orchestration supporting both Docker exec and MCP fallback
#[derive(Debug, Clone)]
pub enum TaskMethod {
    Docker, // Primary method via Docker exec
    MCP,    // Fallback via TCP/MCP
    Hybrid, // Try Docker first, fallback to MCP
}

// DEPRECATED: Legacy Docker orchestration functions removed
// Use TaskOrchestratorActor with Management API instead

/*
/// Docker-based task orchestration using hive-mind
/// This is the PRIMARY method for task creation, replacing TCP MCP spawning
pub async fn call_task_orchestrate_docker(
    task: &str,
    priority: Option<&str>,
    strategy: Option<&str>,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    use crate::utils::docker_hive_mind::{create_docker_hive_mind, SwarmConfig, SwarmPriority, SwarmStrategy};

    info!("Orchestrating task via Docker hive-mind: {}", task);

    // Create hive mind instance
    let hive_mind = create_docker_hive_mind();

    // Convert priority
    let swarm_priority = match priority {
        Some("high") | Some("critical") => SwarmPriority::High,
        Some("medium") => SwarmPriority::Medium,
        Some("low") => SwarmPriority::Low,
        _ => SwarmPriority::Medium,
    };

    // Convert strategy
    let swarm_strategy = match strategy {
        Some("tactical") => SwarmStrategy::Tactical,
        Some("strategic") => SwarmStrategy::Strategic,
        Some("adaptive") => SwarmStrategy::Adaptive,
        _ => SwarmStrategy::HiveMind,
    };

    // Build config
    let config = SwarmConfig {
        priority: swarm_priority,
        strategy: swarm_strategy,
        auto_scale: true,
        monitor: true,
        verbose: false,
        ..Default::default()
    };

    // Spawn swarm
    match hive_mind.spawn_swarm(task, config).await {
        Ok(session_id) => {
            info!("Swarm spawned successfully with ID: {}", session_id);

            Ok(json!({
                "success": true,
                "taskId": session_id,
                "swarmId": session_id,
                "objective": task,
                "strategy": strategy.unwrap_or("hive-mind"),
                "priority": priority.unwrap_or("medium"),
                "status": "spawning",
                "method": "docker-exec",
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        },
        Err(e) => {
            error!("Failed to spawn swarm via Docker: {}", e);
            Err(format!("Docker swarm spawn failed: {}", e).into())
        }
    }
}

/// Hybrid task orchestration - tries Docker first, falls back to MCP
pub async fn call_task_orchestrate_hybrid(
    task: &str,
    priority: Option<&str>,
    strategy: Option<&str>,
    host: &str,
    port: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    info!("Attempting hybrid task orchestration for: {}", task);

    // Try Docker first
    match call_task_orchestrate_docker(task, priority, strategy).await {
        Ok(result) => {
            info!("Docker orchestration successful");
            Ok(result)
        },
        Err(docker_err) => {
            warn!("Docker orchestration failed: {}, trying MCP fallback", docker_err);

            // Fallback to MCP
            match call_task_orchestrate(host, port, task, priority, strategy).await {
                Ok(mut result) => {
                    // Mark as MCP fallback
                    if let Some(obj) = result.as_object_mut() {
                        obj.insert("method".to_string(), json!("mcp-fallback"));
                        obj.insert("docker_error".to_string(), json!(docker_err.to_string()));
                    }
                    info!("MCP fallback successful");
                    Ok(result)
                },
                Err(mcp_err) => {
                    error!("Both Docker and MCP failed. Docker: {}, MCP: {}", docker_err, mcp_err);
                    Err(format!("Hybrid orchestration failed - Docker: {}, MCP: {}", docker_err, mcp_err).into())
                }
            }
        }
    }
}

/// Get Docker swarm status with MCP fallback for telemetry
pub async fn get_swarm_status_hybrid(
    swarm_id: &str,
    host: &str,
    port: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    use crate::utils::docker_hive_mind::create_docker_hive_mind;

    let hive_mind = create_docker_hive_mind();

    // Get primary status from Docker
    let docker_status = match hive_mind.get_swarm_status(swarm_id).await {
        Ok(status) => Some(status),
        Err(e) => {
            warn!("Failed to get Docker swarm status: {}", e);
            None
        }
    };

    // Get telemetry from MCP (non-blocking)
    let mcp_telemetry = match call_task_status(host, port, Some(swarm_id)).await {
        Ok(telemetry) => Some(telemetry),
        Err(e) => {
            debug!("MCP telemetry unavailable: {}", e);
            None
        }
    };

    // Combine results
    let mut result = json!({
        "swarmId": swarm_id,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "sources": {
            "docker": docker_status.is_some(),
            "mcp": mcp_telemetry.is_some()
        }
    });

    if let Some(ref status) = docker_status {
        result["status"] = json!(format!("{:?}", status));
        result["primary_source"] = json!("docker");
    }

    if let Some(ref telemetry) = mcp_telemetry {
        result["telemetry"] = telemetry.clone();
        result["telemetry_source"] = json!("mcp");
    }

    if docker_status.is_none() && mcp_telemetry.is_none() {
        result["status"] = json!("unknown");
        result["error"] = json!("No data sources available");
    }

    Ok(result)
}

/// Get all swarms from both Docker and MCP sources
pub async fn get_all_swarms_hybrid(
    host: &str,
    port: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    use crate::utils::docker_hive_mind::create_docker_hive_mind;

    let hive_mind = create_docker_hive_mind();

    // Get Docker sessions
    let docker_sessions = match hive_mind.get_sessions().await {
        Ok(sessions) => sessions,
        Err(e) => {
            warn!("Failed to get Docker sessions: {}", e);
            Vec::new()
        }
    };

    // Get MCP agent list (for telemetry context)
    let mcp_agents = match call_agent_list(host, port, "active").await {
        Ok(agents) => Some(agents),
        Err(e) => {
            debug!("MCP agent list unavailable: {}", e);
            None
        }
    };

    // Combine and format response
    Ok(json!({
        "swarms": docker_sessions,
        "mcp_agents": mcp_agents,
        "total_swarms": docker_sessions.len(),
        "source": "hybrid",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}
*/

/// Simplified function to get task status
pub async fn call_task_status(
    host: &str,
    port: &str,
    task_id: Option<&str>,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());

    info!("Getting task status for: {:?}", task_id);

    let mut args = json!({});

    if let Some(id) = task_id {
        args["taskId"] = json!(id);
    }

    let params = json!({
        "name": "task_status",
        "arguments": args
    });

    pool.execute_command("task_status", "tools/call", params)
        .await
}
