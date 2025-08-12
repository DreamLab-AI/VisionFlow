use super::client::ClaudeFlowClient;
use super::transport::{Transport, TransportType};
use super::transport::tcp::TcpTransport;
use super::transport::websocket::WebSocketTransport;
use super::transport::http::HttpTransport;
use super::transport::stdio::StdioTransport;
use super::error::Result;
use std::time::Duration;

/// Builder for creating ClaudeFlowClient with various transport options
pub struct ClaudeFlowClientBuilder {
    transport_type: TransportType,
    host: String,
    port: u16,
    auth_token: Option<String>,
    reconnect_attempts: u32,
    reconnect_delay: Duration,
    timeout: Duration,
}

impl ClaudeFlowClientBuilder {
    pub fn new() -> Self {
        Self {
            transport_type: TransportType::from_env(),
            host: std::env::var("CLAUDE_FLOW_HOST")
                .unwrap_or_else(|_| "multi-agent-container".to_string()),
            port: std::env::var("MCP_TCP_PORT")
                .unwrap_or_else(|_| "9500".to_string())
                .parse()
                .unwrap_or(9500),
            auth_token: std::env::var("MCP_AUTH_TOKEN").ok(),
            reconnect_attempts: 3,
            reconnect_delay: Duration::from_secs(1),
            timeout: Duration::from_secs(30),
        }
    }

    /// Use TCP transport (recommended for direct connection)
    pub fn with_tcp(mut self) -> Self {
        self.transport_type = TransportType::Tcp;
        self.port = std::env::var("MCP_TCP_PORT")
            .unwrap_or_else(|_| "9500".to_string())
            .parse()
            .unwrap_or(9500);
        self
    }

    /// Use WebSocket transport (for backward compatibility)
    pub fn with_websocket(mut self) -> Self {
        self.transport_type = TransportType::WebSocket;
        self.port = 3002; // WebSocket bridge port
        self
    }

    /// Use HTTP transport
    pub fn with_http(mut self) -> Self {
        self.transport_type = TransportType::Http;
        self.port = 3000; // HTTP API port
        self
    }

    /// Use stdio transport (for local execution)
    pub fn with_stdio(mut self) -> Self {
        self.transport_type = TransportType::Stdio;
        self
    }

    pub fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn auth_token(mut self, token: Option<String>) -> Self {
        self.auth_token = token;
        self
    }

    pub fn with_retry(mut self, attempts: u32, delay: Duration) -> Self {
        self.reconnect_attempts = attempts;
        self.reconnect_delay = delay;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Build the client with automatic connection
    pub async fn build(self) -> Result<ClaudeFlowClient> {
        let transport: Box<dyn Transport> = match self.transport_type {
            TransportType::Tcp => {
                log::info!("Creating TCP transport to {}:{}", self.host, self.port);
                Box::new(TcpTransport::new(&self.host, self.port))
            }
            TransportType::WebSocket => {
                log::info!("Creating WebSocket transport to {}:{}", self.host, self.port);
                Box::new(WebSocketTransport::new(&self.host, self.port, self.auth_token))
            }
            TransportType::Http => {
                log::info!("Creating HTTP transport to {}:{}", self.host, self.port);
                Box::new(HttpTransport::new(&self.host, self.port))
            }
            TransportType::Stdio => {
                log::info!("Creating stdio transport");
                Box::new(StdioTransport::new())
            }
        };

        let mut client = ClaudeFlowClient::new(transport).await;
        
        // Auto-connect
        client.connect().await?;
        
        // Auto-initialize
        client.initialize().await?;
        
        Ok(client)
    }

    /// Build without connecting (manual connection required)
    pub async fn build_lazy(self) -> ClaudeFlowClient {
        let transport: Box<dyn Transport> = match self.transport_type {
            TransportType::Tcp => Box::new(TcpTransport::new(&self.host, self.port)),
            TransportType::WebSocket => Box::new(WebSocketTransport::new(&self.host, self.port, self.auth_token)),
            TransportType::Http => Box::new(HttpTransport::new(&self.host, self.port)),
            TransportType::Stdio => Box::new(StdioTransport::new()),
        };

        ClaudeFlowClient::new(transport).await
    }
}

impl Default for ClaudeFlowClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}