# External System Integration Guide

## Overview

This guide shows how to integrate external systems (particularly Rust-based) with the Multi-Agent Docker container using the hybrid control/data plane architecture.

## Integration Architecture

```
Your Rust System
    │
    ├─→ Docker Exec (Control Plane)
    │   ├─ Create sessions
    │   ├─ Start tasks
    │   ├─ Query status
    │   └─ Get results
    │
    └─→ TCP/MCP (Data Plane)
        ├─ Stream telemetry
        ├─ Monitor agents
        └─ Real-time visualization
```

## Core Integration Pattern

### 1. Task Spawning (Control Plane)

Use `docker exec` commands to manage task lifecycle:

```rust
use tokio::process::Command;

pub struct HiveMindClient {
    container_name: String,
}

impl HiveMindClient {
    pub async fn spawn_task(
        &self,
        task: &str,
        priority: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<SessionHandle, Error> {
        // Create session
        let metadata_json = metadata
            .map(|m| serde_json::to_string(&m).unwrap())
            .unwrap_or_else(|| "null".to_string());

        let create_output = Command::new("docker")
            .args(&[
                "exec",
                &self.container_name,
                "/app/scripts/hive-session-manager.sh",
                "create",
                task,
                priority,
                &metadata_json,
            ])
            .output()
            .await?;

        let uuid = String::from_utf8_lossy(&create_output.stdout)
            .trim()
            .to_string();

        // Start session in background
        Command::new("docker")
            .args(&[
                "exec", "-d",
                &self.container_name,
                "/app/scripts/hive-session-manager.sh",
                "start",
                &uuid,
            ])
            .spawn()?;

        Ok(SessionHandle {
            uuid,
            client: self.clone(),
        })
    }
}
```

### 2. Status Monitoring (Control Plane)

Poll session status via docker exec:

```rust
pub struct SessionHandle {
    uuid: String,
    client: HiveMindClient,
}

impl SessionHandle {
    pub async fn wait_for_completion(&self) -> Result<SessionStatus, Error> {
        let mut delay = Duration::from_secs(1);

        loop {
            let status = self.get_status().await?;

            match status.as_str() {
                "completed" => return Ok(SessionStatus::Completed),
                "failed" => return Ok(SessionStatus::Failed),
                _ => {
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(Duration::from_secs(30));
                }
            }
        }
    }

    pub async fn get_status(&self) -> Result<String, Error> {
        let output = Command::new("docker")
            .args(&[
                "exec",
                &self.client.container_name,
                "/app/scripts/hive-session-manager.sh",
                "status",
                &self.uuid,
            ])
            .output()
            .await?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    pub async fn get_metadata(&self) -> Result<SessionMetadata, Error> {
        let output = Command::new("docker")
            .args(&[
                "exec",
                &self.client.container_name,
                "/app/scripts/hive-session-manager.sh",
                "get",
                &self.uuid,
            ])
            .output()
            .await?;

        let json = String::from_utf8_lossy(&output.stdout);
        Ok(serde_json::from_str(&json)?)
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct SessionMetadata {
    pub session_id: String,
    pub task: String,
    pub status: String,
    pub created: String,
    pub updated: Option<String>,
    pub working_dir: String,
    pub output_dir: String,
    pub database: String,
    pub log_file: String,
    pub metadata: Option<serde_json::Value>,
}
```

### 3. Telemetry Streaming (Data Plane)

Connect to MCP TCP server for real-time metrics:

```rust
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub struct McpTelemetryClient {
    stream: TcpStream,
    reader: BufReader<tokio::io::ReadHalf<TcpStream>>,
    writer: tokio::io::WriteHalf<TcpStream>,
}

impl McpTelemetryClient {
    pub async fn connect(host: &str, port: u16) -> Result<Self, Error> {
        let stream = TcpStream::connect((host, port)).await?;
        let (read_half, write_half) = tokio::io::split(stream);
        let reader = BufReader::new(read_half);

        Ok(Self {
            stream,
            reader,
            writer: write_half,
        })
    }

    pub async fn query_session_metrics(
        &mut self,
        session_id: &str,
    ) -> Result<SessionMetrics, Error> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "session_metrics",
                "arguments": {
                    "session_id": session_id
                }
            }
        });

        // Send request
        let request_str = serde_json::to_string(&request)?;
        self.writer.write_all(request_str.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        // Read response
        let mut line = String::new();
        self.reader.read_line(&mut line).await?;

        let response: serde_json::Value = serde_json::from_str(&line)?;
        Ok(serde_json::from_value(response["result"].clone())?)
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct SessionMetrics {
    pub swarm_id: String,
    pub session_id: String,
    pub agents: Vec<AgentMetric>,
    pub tasks: TaskMetrics,
    pub performance: PerformanceMetrics,
}
```

### 4. Result Retrieval (File System)

Access task outputs from mounted directory:

```rust
use std::path::PathBuf;

impl SessionHandle {
    pub async fn get_output_dir(&self) -> Result<PathBuf, Error> {
        let output = Command::new("docker")
            .args(&[
                "exec",
                &self.client.container_name,
                "/app/scripts/hive-session-manager.sh",
                "output-dir",
                &self.uuid,
            ])
            .output()
            .await?;

        let path = String::from_utf8_lossy(&output.stdout).trim();
        Ok(PathBuf::from(path))
    }

    pub async fn read_output_files(&self) -> Result<Vec<PathBuf>, Error> {
        let output_dir = self.get_output_dir().await?;

        // If ext/ is mounted to host at ./workspace/ext/
        let host_path = format!("./workspace/ext/hive-sessions/{}/", self.uuid);

        let mut files = Vec::new();
        for entry in std::fs::read_dir(host_path)? {
            let entry = entry?;
            files.push(entry.path());
        }

        Ok(files)
    }
}
```

## Complete Integration Example

```rust
use tokio;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize client
    let client = HiveMindClient::new("multi-agent-container");

    // Spawn task with metadata
    let session = client.spawn_task(
        "Build a Rust CLI tool for parsing log files",
        "high",
        Some(json!({
            "project": "log-parser",
            "requester": "voice-api",
            "deadline": "2025-10-06T00:00:00Z"
        }))
    ).await?;

    println!("Task spawned with UUID: {}", session.uuid());

    // Connect telemetry stream
    let mut mcp = McpTelemetryClient::connect("localhost", 9500).await?;

    // Monitor in parallel
    let uuid = session.uuid().to_string();
    let telemetry_task = tokio::spawn(async move {
        loop {
            if let Ok(metrics) = mcp.query_session_metrics(&uuid).await {
                println!("Agents active: {}", metrics.agents.len());
                println!("Tasks completed: {}", metrics.tasks.completed);
            }

            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    // Wait for completion
    let status = session.wait_for_completion().await?;

    println!("Task finished with status: {:?}", status);

    // Get results
    if matches!(status, SessionStatus::Completed) {
        let output_files = session.read_output_files().await?;
        println!("Generated {} output files", output_files.len());

        for file in output_files {
            println!("  - {}", file.display());
        }
    } else {
        // Read logs for debugging
        let log_file = session.get_log_file().await?;
        println!("Check logs at: {}", log_file);
    }

    // Cleanup telemetry
    telemetry_task.abort();

    Ok(())
}
```

## Speech Service Integration

For voice command integration:

```rust
pub async fn handle_voice_command(
    command: VoiceCommand,
) -> Result<VoiceResponse, Error> {
    let client = HiveMindClient::new("multi-agent-container");

    // Parse intent from voice command
    let task = extract_task_description(&command)?;
    let priority = determine_priority(&command)?;

    // Spawn task
    let session = client.spawn_task(&task, priority, Some(json!({
        "source": "voice",
        "user_id": command.user_id,
        "timestamp": Utc::now()
    }))).await?;

    // Return immediately with session ID
    Ok(VoiceResponse {
        message: format!(
            "I've started working on that. Session ID: {}",
            session.uuid()
        ),
        session_id: session.uuid().to_string(),
    })
}

pub async fn check_task_status(
    session_id: &str,
) -> Result<VoiceResponse, Error> {
    let client = HiveMindClient::new("multi-agent-container");
    let session = SessionHandle::from_uuid(session_id, client);

    let status = session.get_status().await?;

    let message = match status.as_str() {
        "running" => "Still working on it...",
        "completed" => "Task completed successfully!",
        "failed" => "Task encountered an error. Check logs.",
        _ => "Task status unknown",
    };

    Ok(VoiceResponse {
        message: message.to_string(),
        session_id: session_id.to_string(),
    })
}
```

## WebSocket Visualization Integration

For real-time spring system updates:

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::StreamExt;

pub async fn stream_to_spring_system(
    session_id: String,
) -> Result<(), Error> {
    let (ws_stream, _) = connect_async("ws://localhost:3002")
        .await?;

    let (mut write, mut read) = ws_stream.split();

    // Subscribe to specific session
    let subscribe = json!({
        "action": "subscribe",
        "session_id": session_id,
        "filters": {
            "include_performance": true,
            "include_topology": true
        }
    });

    write.send(Message::Text(serde_json::to_string(&subscribe)?))
        .await?;

    // Stream to GPU spring system
    while let Some(message) = read.next().await {
        let message = message?;

        if let Message::Text(text) = message {
            let telemetry: TelemetryUpdate = serde_json::from_str(&text)?;

            // Update spring system
            update_spring_system_agents(telemetry.agents).await;
            update_spring_system_connections(telemetry.topology).await;
        }
    }

    Ok(())
}
```

## Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Docker exec failed: {0}")]
    DockerExec(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("MCP connection failed: {0}")]
    McpConnection(String),

    #[error("Timeout waiting for task completion")]
    Timeout,

    #[error("Task failed: {reason}")]
    TaskFailed { reason: String },
}

// Retry logic
pub async fn with_retry<F, T>(
    mut operation: F,
    max_retries: u32,
) -> Result<T, IntegrationError>
where
    F: FnMut() -> Pin<Box<dyn Future<Output = Result<T, IntegrationError>>>>,
{
    let mut last_error = None;

    for attempt in 1..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries {
                    let delay = Duration::from_millis(500 * 2_u64.pow(attempt - 1));
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap())
}
```

## Best Practices

### 1. Session Management
- Store session UUIDs persistently
- Track session metadata locally
- Implement cleanup for completed sessions
- Handle container restarts gracefully

### 2. Error Recovery
- Retry transient docker exec failures
- Implement timeout for long-running tasks
- Log all session operations
- Preserve failed session logs for debugging

### 3. Performance
- Use connection pooling for MCP
- Batch telemetry queries
- Implement backpressure for WebSocket streams
- Cache frequently accessed session metadata

### 4. Monitoring
- Track session creation rate
- Monitor active session count
- Alert on high failure rates
- Dashboard for session lifecycle visibility

## Testing Integration

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_lifecycle() {
        let client = HiveMindClient::new("multi-agent-container");

        // Create session
        let session = client.spawn_task(
            "test task",
            "medium",
            None
        ).await.unwrap();

        // Wait for startup
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Check status
        let status = session.get_status().await.unwrap();
        assert!(matches!(status.as_str(), "starting" | "running"));

        // Wait for completion (with timeout)
        let result = tokio::time::timeout(
            Duration::from_secs(60),
            session.wait_for_completion()
        ).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_mcp_telemetry() {
        let mut mcp = McpTelemetryClient::connect("localhost", 9500)
            .await.unwrap();

        // Query should not panic
        let metrics = mcp.query_session_metrics("test-uuid").await;
        // May error if session doesn't exist, but connection should work
    }
}
```

## Migration from Legacy System

If migrating from direct `claude-flow hive-mind spawn`:

**Before**:
```rust
Command::new("docker")
    .args(&[
        "exec", "-d",
        "multi-agent-container",
        "claude-flow", "hive-mind", "spawn",
        task,
        "--claude"
    ])
    .spawn()?;
```

**After**:
```rust
let client = HiveMindClient::new("multi-agent-container");
let session = client.spawn_task(task, "medium", None).await?;
// Now you have UUID for tracking!
```

**Benefits**:
- UUID tracking
- Database isolation (no crashes)
- Status monitoring
- Output directory access
- Persistent logs
