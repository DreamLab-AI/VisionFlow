use async_trait::async_trait;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::Mutex;
use std::sync::Arc;
use serde_json::{json, Value};
use log::{debug, info, warn, error};
use std::time::Duration;
use tokio::time::{timeout, sleep};

use super::Transport;
use crate::services::claude_flow::error::{ConnectorError, Result};
use crate::services::claude_flow::types::{McpRequest, McpResponse, McpNotification};

/// TCP transport for direct MCP server connection
/// Connects to claude-flow TCP server on port 9500 (or configured port)
pub struct TcpTransport {
    host: String,
    port: u16,
    stream: Option<Arc<Mutex<TcpStream>>>,
    reader: Option<Arc<Mutex<BufReader<TcpStream>>>>,
    writer: Option<Arc<Mutex<BufWriter<TcpStream>>>>,
    reconnect_attempts: u32,
    reconnect_delay: Duration,
    connection_timeout: Duration,
}

impl TcpTransport {
    pub fn new(host: &str, port: u16) -> Self {
        let reconnect_attempts = std::env::var("MCP_RECONNECT_ATTEMPTS")
            .unwrap_or_else(|_| "3".to_string())
            .parse()
            .unwrap_or(3);
            
        let reconnect_delay = std::env::var("MCP_RECONNECT_DELAY")
            .unwrap_or_else(|_| "1000".to_string())
            .parse()
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_secs(1));
            
        let connection_timeout = std::env::var("MCP_CONNECTION_TIMEOUT")
            .unwrap_or_else(|_| "30000".to_string())
            .parse()
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_secs(30));

        Self {
            host: host.to_string(),
            port,
            stream: None,
            reader: None,
            writer: None,
            reconnect_attempts,
            reconnect_delay,
            connection_timeout,
        }
    }

    pub fn new_with_defaults() -> Self {
        let host = std::env::var("CLAUDE_FLOW_HOST")
            .unwrap_or_else(|_| "multi-agent-container".to_string());
        let port = std::env::var("MCP_TCP_PORT")
            .unwrap_or_else(|_| "9500".to_string())
            .parse()
            .unwrap_or(9500);
            
        Self::new(&host, port)
    }

    async fn connect_with_retry(&mut self) -> Result<()> {
        let mut attempt = 0;
        
        loop {
            match self.try_connect().await {
                Ok(_) => {
                    info!("Connected to MCP TCP server at {}:{}", self.host, self.port);
                    return Ok(());
                }
                Err(e) if attempt < self.reconnect_attempts => {
                    attempt += 1;
                    warn!(
                        "TCP connection attempt {} to {}:{} failed: {}. Retrying in {:?}",
                        attempt, self.host, self.port, e, self.reconnect_delay
                    );
                    sleep(self.reconnect_delay * attempt).await;
                }
                Err(e) => {
                    error!(
                        "Failed to connect to MCP TCP server after {} attempts: {}",
                        attempt, e
                    );
                    return Err(e);
                }
            }
        }
    }

    async fn try_connect(&mut self) -> Result<()> {
        debug!("Attempting TCP connection to {}:{}", self.host, self.port);
        
        // Connect with timeout
        let addr = format!("{}:{}", self.host, self.port);
        let stream = match timeout(self.connection_timeout, TcpStream::connect(&addr)).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => {
                return Err(ConnectorError::ConnectionError(format!(
                    "TCP connection failed: {}",
                    e
                )));
            }
            Err(_) => {
                return Err(ConnectorError::Timeout(format!(
                    "TCP connection timeout after {:?}",
                    self.connection_timeout
                )));
            }
        };

        // Set TCP keepalive
        stream.set_nodelay(true).map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to set TCP nodelay: {}", e))
        })?;

        // Clone for reader and writer
        let reader_stream = stream.try_clone().await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to clone stream for reader: {}", e))
        })?;
        
        let writer_stream = stream.try_clone().await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to clone stream for writer: {}", e))
        })?;

        // Create buffered reader and writer
        let reader = BufReader::new(reader_stream);
        let writer = BufWriter::new(writer_stream);

        self.stream = Some(Arc::new(Mutex::new(stream)));
        self.reader = Some(Arc::new(Mutex::new(reader)));
        self.writer = Some(Arc::new(Mutex::new(writer)));

        debug!("TCP connection established successfully");
        Ok(())
    }

    async fn ensure_connected(&mut self) -> Result<()> {
        if self.stream.is_none() {
            self.connect_with_retry().await?;
        }
        Ok(())
    }
}

#[async_trait]
impl Transport for TcpTransport {
    async fn connect(&mut self) -> Result<()> {
        self.connect_with_retry().await
    }

    async fn disconnect(&mut self) -> Result<()> {
        debug!("Disconnecting TCP transport");
        
        if let Some(stream) = self.stream.take() {
            if let Ok(s) = stream.lock().await {
                // Note: TcpStream doesn't have shutdown in async context
                drop(s);
            }
        }
        
        self.reader = None;
        self.writer = None;
        
        info!("TCP transport disconnected");
        Ok(())
    }

    async fn send_request(&mut self, request: McpRequest) -> Result<McpResponse> {
        self.ensure_connected().await?;
        
        let writer = self.writer.as_ref()
            .ok_or_else(|| ConnectorError::NotConnected)?;
        let reader = self.reader.as_ref()
            .ok_or_else(|| ConnectorError::NotConnected)?;
            
        let mut writer = writer.lock().await;
        let mut reader = reader.lock().await;
        
        // Serialize request to JSON
        let json = serde_json::to_string(&request).map_err(|e| {
            ConnectorError::SerializationError(format!("Failed to serialize request: {}", e))
        })?;
        
        debug!("Sending TCP request: {}", json);
        
        // Write JSON line (newline-delimited)
        writer.write_all(json.as_bytes()).await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to write to TCP stream: {}", e))
        })?;
        
        writer.write_all(b"\n").await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to write newline: {}", e))
        })?;
        
        writer.flush().await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to flush TCP stream: {}", e))
        })?;
        
        // Read response line
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to read from TCP stream: {}", e))
        })?;
        
        if bytes_read == 0 {
            return Err(ConnectorError::ConnectionError(
                "TCP connection closed by server".to_string()
            ));
        }
        
        debug!("Received TCP response: {}", line.trim());
        
        // Parse JSON response
        let response: McpResponse = serde_json::from_str(&line).map_err(|e| {
            ConnectorError::SerializationError(format!(
                "Failed to parse response: {}. Raw: {}",
                e,
                line.trim()
            ))
        })?;
        
        Ok(response)
    }

    async fn send_notification(&mut self, notification: McpNotification) -> Result<()> {
        self.ensure_connected().await?;
        
        let writer = self.writer.as_ref()
            .ok_or_else(|| ConnectorError::NotConnected)?;
            
        let mut writer = writer.lock().await;
        
        // Serialize notification to JSON
        let json = serde_json::to_string(&notification).map_err(|e| {
            ConnectorError::SerializationError(format!("Failed to serialize notification: {}", e))
        })?;
        
        debug!("Sending TCP notification: {}", json);
        
        // Write JSON line
        writer.write_all(json.as_bytes()).await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to write notification: {}", e))
        })?;
        
        writer.write_all(b"\n").await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to write newline: {}", e))
        })?;
        
        writer.flush().await.map_err(|e| {
            ConnectorError::ConnectionError(format!("Failed to flush: {}", e))
        })?;
        
        Ok(())
    }

    async fn receive_notification(&mut self) -> Result<Option<McpNotification>> {
        self.ensure_connected().await?;
        
        let reader = self.reader.as_ref()
            .ok_or_else(|| ConnectorError::NotConnected)?;
            
        let mut reader = reader.lock().await;
        
        // Try to read a line with short timeout
        let mut line = String::new();
        match timeout(Duration::from_millis(100), reader.read_line(&mut line)).await {
            Ok(Ok(0)) => {
                // Connection closed
                return Err(ConnectorError::ConnectionError(
                    "TCP connection closed".to_string()
                ));
            }
            Ok(Ok(_)) => {
                // Got data, try to parse as notification
                if let Ok(notif) = serde_json::from_str::<McpNotification>(&line) {
                    return Ok(Some(notif));
                }
                // Not a notification, might be a response - ignore
                Ok(None)
            }
            Ok(Err(e)) => {
                return Err(ConnectorError::ConnectionError(format!(
                    "Failed to read notification: {}",
                    e
                )));
            }
            Err(_) => {
                // Timeout - no notification available
                Ok(None)
            }
        }
    }

    fn is_connected(&self) -> bool {
        self.stream.is_some()
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Attempt to close connection gracefully
        if let Some(stream) = self.stream.take() {
            // Connection will be closed when Arc is dropped
            drop(stream);
        }
    }
}