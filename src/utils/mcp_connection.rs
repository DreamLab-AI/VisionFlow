use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use serde_json::{json, Value};
use uuid::Uuid;
use log::{info, warn, error, debug};
use std::time::Duration;
use std::collections::HashMap;

/// MCP Connection Pool for stable connections
pub struct MCPConnectionPool {
    connections: Arc<RwLock<HashMap<String, MCPConnection>>>,
    host: String,
    port: String,
    max_retries: u32,
    retry_delay: Duration,
}

struct MCPConnection {
    stream: Option<TcpStream>,
    session_id: String,
    initialized: bool,
    last_used: std::time::Instant,
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
    pub async fn get_connection(&self, purpose: &str) -> Result<TcpStream, Box<dyn std::error::Error>> {
        let mut connections = self.connections.write().await;
        
        // Check if we have an existing connection
        if let Some(conn) = connections.get_mut(purpose) {
            if conn.initialized && conn.stream.is_some() {
                debug!("Reusing existing MCP connection for {}", purpose);
                conn.last_used = std::time::Instant::now();
                
                // Clone the stream for use
                if let Some(stream) = &conn.stream {
                    // For now, create a new connection since TcpStream can't be cloned
                    // In production, we'd use a connection pool with Arc<Mutex<>>
                    return self.create_new_connection(purpose).await;
                }
            }
        }
        
        // Create new connection
        self.create_new_connection(purpose).await
    }

    async fn create_new_connection(&self, purpose: &str) -> Result<TcpStream, Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.host, self.port);
        
        for attempt in 1..=self.max_retries {
            info!("MCP connection attempt {}/{} to {} for {}", attempt, self.max_retries, addr, purpose);
            
            match TcpStream::connect(&addr).await {
                Ok(mut stream) => {
                    info!("TCP connection established to MCP server at {}", addr);
                    
                    // Initialize MCP session
                    match self.initialize_mcp_session(&mut stream).await {
                        Ok(session_id) => {
                            info!("MCP session initialized successfully: {}", session_id);
                            return Ok(stream);
                        }
                        Err(e) => {
                            warn!("Failed to initialize MCP session (attempt {}): {}", attempt, e);
                            if attempt < self.max_retries {
                                tokio::time::sleep(self.retry_delay).await;
                                continue;
                            }
                            return Err(e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to connect to MCP server (attempt {}): {}", attempt, e);
                    if attempt < self.max_retries {
                        tokio::time::sleep(self.retry_delay).await;
                        continue;
                    }
                    return Err(Box::new(e));
                }
            }
        }
        
        Err("Failed to establish MCP connection after all retries".into())
    }

    async fn initialize_mcp_session(&self, stream: &mut TcpStream) -> Result<String, Box<dyn std::error::Error>> {
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
        
        // Read response with timeout
        let mut reader = BufReader::new(stream);
        let mut response_line = String::new();
        
        // Read and skip server.initialized notifications
        loop {
            response_line.clear();
            match tokio::time::timeout(
                Duration::from_secs(5),
                reader.read_line(&mut response_line)
            ).await {
                Ok(Ok(n)) if n > 0 => {
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
                                return Ok(session_id);
                            } else if let Some(error) = response.get("error") {
                                error!("MCP init error: {:?}", error);
                                return Err(format!("MCP initialization failed: {:?}", error).into());
                            }
                        }
                    }
                }
                Ok(Ok(_)) => {
                    warn!("Empty response from MCP server");
                    return Err("Empty response from MCP server".into());
                }
                Ok(Err(e)) => {
                    error!("Error reading MCP response: {}", e);
                    return Err(Box::new(e));
                }
                Err(_) => {
                    error!("Timeout waiting for MCP initialization response");
                    return Err("MCP initialization timeout".into());
                }
            }
        }
    }

    /// Execute an MCP command with retry logic
    pub async fn execute_command(
        &self,
        purpose: &str,
        method: &str,
        params: Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.host, self.port);
        
        for attempt in 1..=self.max_retries {
            info!("Executing MCP command '{}' (attempt {}/{})", method, attempt, self.max_retries);
            
            // Get fresh connection for each attempt
            match TcpStream::connect(&addr).await {
                Ok(mut stream) => {
                    // Initialize session
                    if let Err(e) = self.initialize_mcp_session(&mut stream).await {
                        warn!("Failed to initialize MCP session: {}", e);
                        if attempt < self.max_retries {
                            tokio::time::sleep(self.retry_delay).await;
                            continue;
                        }
                        return Err(e);
                    }
                    
                    // Send command
                    let request_id = Uuid::new_v4().to_string();
                    let request = json!({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params
                    });
                    
                    let msg = format!("{}\n", request.to_string());
                    debug!("Sending MCP command: {}", msg.trim());
                    
                    if let Err(e) = stream.write_all(msg.as_bytes()).await {
                        warn!("Failed to send command: {}", e);
                        if attempt < self.max_retries {
                            tokio::time::sleep(self.retry_delay).await;
                            continue;
                        }
                        return Err(Box::new(e));
                    }
                    
                    stream.flush().await?;
                    
                    // Read response
                    let mut reader = BufReader::new(stream);
                    let mut response_line = String::new();
                    
                    // Read with timeout
                    match tokio::time::timeout(
                        Duration::from_secs(10),
                        Self::read_response(&mut reader, &request_id)
                    ).await {
                        Ok(Ok(result)) => {
                            info!("MCP command '{}' executed successfully", method);
                            return Ok(result);
                        }
                        Ok(Err(e)) => {
                            warn!("Failed to read response: {}", e);
                            if attempt < self.max_retries {
                                tokio::time::sleep(self.retry_delay).await;
                                continue;
                            }
                            return Err(e);
                        }
                        Err(_) => {
                            warn!("Timeout waiting for response");
                            if attempt < self.max_retries {
                                tokio::time::sleep(self.retry_delay).await;
                                continue;
                            }
                            return Err("Command execution timeout".into());
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to connect for command execution: {}", e);
                    if attempt < self.max_retries {
                        tokio::time::sleep(self.retry_delay).await;
                        continue;
                    }
                    return Err(Box::new(e));
                }
            }
        }
        
        Err("Failed to execute command after all retries".into())
    }

    async fn read_response(
        reader: &mut BufReader<TcpStream>,
        request_id: &str,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let mut response_line = String::new();
        
        // Read until we get our response (skip notifications)
        loop {
            response_line.clear();
            let n = reader.read_line(&mut response_line).await?;
            
            if n == 0 {
                return Err("Connection closed while reading response".into());
            }
            
            let trimmed = response_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            
            debug!("MCP response line: {}", trimmed);
            
            // Skip notifications
            if trimmed.contains("server.initialized") {
                continue;
            }
            
            // Parse response
            if let Ok(response) = serde_json::from_str::<Value>(trimmed) {
                // Check if this is our response
                if response.get("id").and_then(|id| id.as_str()) == Some(request_id) {
                    if let Some(result) = response.get("result") {
                        return Ok(result.clone());
                    } else if let Some(error) = response.get("error") {
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

/// Simplified function to call swarm_init
pub async fn call_swarm_init(
    host: &str,
    port: &str,
    topology: &str,
    max_agents: u32,
    strategy: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());
    
    let params = json!({
        "name": "swarm_init",
        "arguments": {
            "topology": topology,
            "maxAgents": max_agents,
            "strategy": strategy
        }
    });
    
    pool.execute_command("swarm_init", "tools/call", params).await
}

/// Simplified function to destroy a swarm
pub async fn call_swarm_destroy(
    host: &str,
    port: &str,
    swarm_id: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());
    
    info!("Calling swarm_destroy for swarm_id: {}", swarm_id);
    
    let params = json!({
        "name": "swarm_destroy", 
        "arguments": {
            "swarmId": swarm_id
        }
    });
    
    pool.execute_command("swarm_destroy", "tools/call", params).await
}

/// Simplified function to spawn an agent
pub async fn call_agent_spawn(
    host: &str,
    port: &str,
    agent_type: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());
    
    info!("Spawning agent of type: {}", agent_type);
    
    let params = json!({
        "name": "agent_spawn",
        "arguments": {
            "type": agent_type
        }
    });
    
    pool.execute_command("agent_spawn", "tools/call", params).await
}

/// Simplified function to list agents
pub async fn call_agent_list(
    host: &str,
    port: &str,
    filter: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());
    
    let params = json!({
        "name": "agent_list",
        "arguments": {
            "filter": filter
        }
    });
    
    pool.execute_command("agent_list", "tools/call", params).await
}