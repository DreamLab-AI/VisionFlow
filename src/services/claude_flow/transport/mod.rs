pub mod http;
pub mod websocket;
pub mod tcp;  // NEW: Direct TCP transport for optimal performance
// DISABLED: stdio transport incorrectly spawns subprocess instead of connecting to multi-agent-container
// pub mod stdio;

use async_trait::async_trait;
use crate::services::claude_flow::error::Result;
use crate::services::claude_flow::types::{McpRequest, McpResponse, McpNotification};

#[async_trait]
pub trait Transport: Send + Sync {
    async fn connect(&mut self) -> Result<()>;
    async fn disconnect(&mut self) -> Result<()>;
    async fn send_request(&mut self, request: McpRequest) -> Result<McpResponse>;
    async fn send_notification(&mut self, notification: McpNotification) -> Result<()>;
    async fn receive_notification(&mut self) -> Result<Option<McpNotification>>;
    fn is_connected(&self) -> bool;
}

/// Transport type enumeration for easy selection
#[derive(Debug, Clone)]
pub enum TransportType {
    Http,
    WebSocket,
    Tcp,  // NEW: TCP transport option
}

impl TransportType {
    /// Determine transport type from environment or default to TCP
    pub fn from_env() -> Self {
        match std::env::var("MCP_TRANSPORT").as_deref() {
            Ok("http") => TransportType::Http,
            Ok("websocket") | Ok("ws") => TransportType::WebSocket,
            Ok("tcp") | _ => TransportType::Tcp, // Default to TCP for performance
        }
    }
}