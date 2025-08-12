use actix::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use log::{info, warn, error, debug};
use serde_json::{json, Value};

use crate::services::claude_flow::{
    client::ClaudeFlowClient,
    client_builder::ClaudeFlowClientBuilder,
    error::Result as McpResult,
};

/// Enhanced Claude Flow Actor using direct TCP connection
pub struct ClaudeFlowActorTcp {
    client: Option<ClaudeFlowClient>,
    is_connected: bool,
    auto_reconnect: bool,
    reconnect_handle: Option<SpawnHandle>,
    telemetry_handle: Option<SpawnHandle>,
    connection_stats: ConnectionStats,
}

#[derive(Default, Debug)]
struct ConnectionStats {
    connection_attempts: u32,
    successful_connections: u32,
    failed_connections: u32,
    messages_sent: u64,
    messages_received: u64,
    last_error: Option<String>,
}

impl ClaudeFlowActorTcp {
    pub fn new() -> Self {
        Self {
            client: None,
            is_connected: false,
            auto_reconnect: true,
            reconnect_handle: None,
            telemetry_handle: None,
            connection_stats: ConnectionStats::default(),
        }
    }

    async fn connect_tcp(&mut self) -> McpResult<()> {
        self.connection_stats.connection_attempts += 1;
        
        info!("Connecting to MCP via TCP...");
        
        // Build client with TCP transport
        match ClaudeFlowClientBuilder::new()
            .with_tcp()  // Use TCP transport
            .with_retry(3, Duration::from_secs(2))
            .with_timeout(Duration::from_secs(30))
            .build()
            .await
        {
            Ok(client) => {
                self.client = Some(client);
                self.is_connected = true;
                self.connection_stats.successful_connections += 1;
                info!("Successfully connected to MCP TCP server");
                
                // Start telemetry if needed
                self.start_telemetry_stream();
                
                Ok(())
            }
            Err(e) => {
                self.connection_stats.failed_connections += 1;
                self.connection_stats.last_error = Some(format!("{}", e));
                error!("Failed to connect to MCP TCP server: {}", e);
                Err(e)
            }
        }
    }

    fn start_telemetry_stream(&mut self) {
        // Cancel existing telemetry
        if let Some(handle) = self.telemetry_handle.take() {
            handle.cancel();
        }

        // Start new telemetry stream
        let addr = self.address();
        self.telemetry_handle = Some(
            ctx.run_interval(Duration::from_millis(100), move |act, _ctx| {
                if act.is_connected {
                    if let Some(client) = &mut act.client {
                        // Send telemetry
                        let telemetry = json!({
                            "type": "telemetry",
                            "timestamp": chrono::Utc::now().to_rfc3339(),
                            "stats": {
                                "messages_sent": act.connection_stats.messages_sent,
                                "messages_received": act.connection_stats.messages_received,
                            }
                        });
                        
                        // Fire and forget telemetry
                        let _ = client.call_tool("telemetry_update", telemetry);
                    }
                }
            })
        );
    }

    fn schedule_reconnect(&mut self, ctx: &mut Context<Self>) {
        if !self.auto_reconnect {
            return;
        }

        // Cancel existing reconnect
        if let Some(handle) = self.reconnect_handle.take() {
            ctx.cancel_future(handle);
        }

        // Schedule new reconnect
        self.reconnect_handle = Some(
            ctx.run_later(Duration::from_secs(5), |act, ctx| {
                info!("Attempting to reconnect to MCP TCP server...");
                ctx.notify(Connect);
            })
        );
    }
}

impl Actor for ClaudeFlowActorTcp {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Claude Flow TCP Actor started");
        
        // Auto-connect on start
        ctx.notify(Connect);
        
        // Periodic health check
        ctx.run_interval(Duration::from_secs(30), |act, ctx| {
            if !act.is_connected {
                warn!("MCP TCP connection lost, attempting reconnect...");
                ctx.notify(Connect);
            }
        });
    }

    fn stopped(&mut self, _: &mut Self::Context) {
        info!("Claude Flow TCP Actor stopped");
        
        // Clean disconnect
        if let Some(mut client) = self.client.take() {
            let _ = client.disconnect();
        }
    }
}

// Messages

#[derive(Message)]
#[rtype(result = "McpResult<()>")]
struct Connect;

#[derive(Message)]
#[rtype(result = "McpResult<()>")]
struct Disconnect;

#[derive(Message, Serialize, Deserialize)]
#[rtype(result = "McpResult<Value>")]
pub struct CallTool {
    pub name: String,
    pub arguments: Value,
}

#[derive(Message)]
#[rtype(result = "McpResult<Vec<String>>")]
pub struct ListTools;

#[derive(Message)]
#[rtype(result = "ConnectionInfo")]
pub struct GetConnectionInfo;

#[derive(Serialize)]
pub struct ConnectionInfo {
    pub connected: bool,
    pub transport: String,
    pub host: String,
    pub port: u16,
    pub stats: ConnectionStatsPublic,
}

#[derive(Serialize)]
pub struct ConnectionStatsPublic {
    pub connection_attempts: u32,
    pub successful_connections: u32,
    pub failed_connections: u32,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub last_error: Option<String>,
}

// Message Handlers

impl Handler<Connect> for ClaudeFlowActorTcp {
    type Result = ResponseActFuture<Self, McpResult<()>>;

    fn handle(&mut self, _: Connect, _: &mut Context<Self>) -> Self::Result {
        Box::pin(
            async move {
                self.connect_tcp().await
            }
            .into_actor(self)
            .map(|result, act, ctx| {
                if result.is_err() && act.auto_reconnect {
                    act.schedule_reconnect(ctx);
                }
                result
            })
        )
    }
}

impl Handler<Disconnect> for ClaudeFlowActorTcp {
    type Result = ResponseActFuture<Self, McpResult<()>>;

    fn handle(&mut self, _: Disconnect, _: &mut Context<Self>) -> Self::Result {
        Box::pin(
            async move {
                if let Some(mut client) = self.client.take() {
                    client.disconnect().await?;
                }
                self.is_connected = false;
                
                // Cancel telemetry
                if let Some(handle) = self.telemetry_handle.take() {
                    handle.cancel();
                }
                
                Ok(())
            }
            .into_actor(self)
        )
    }
}

impl Handler<CallTool> for ClaudeFlowActorTcp {
    type Result = ResponseActFuture<Self, McpResult<Value>>;

    fn handle(&mut self, msg: CallTool, ctx: &mut Context<Self>) -> Self::Result {
        if !self.is_connected {
            warn!("Not connected, attempting to connect before tool call");
            ctx.notify(Connect);
            
            return Box::pin(
                async move {
                    Err(crate::services::claude_flow::error::ConnectorError::NotConnected.into())
                }
                .into_actor(self)
            );
        }

        let client = self.client.clone();
        
        Box::pin(
            async move {
                if let Some(mut client) = client {
                    client.call_tool(&msg.name, msg.arguments).await
                } else {
                    Err(crate::services::claude_flow::error::ConnectorError::NotConnected.into())
                }
            }
            .into_actor(self)
            .map(|result, act, ctx| {
                match &result {
                    Ok(_) => {
                        act.connection_stats.messages_sent += 1;
                        act.connection_stats.messages_received += 1;
                    }
                    Err(e) => {
                        error!("Tool call failed: {}", e);
                        act.connection_stats.last_error = Some(format!("{}", e));
                        
                        // Reconnect on connection errors
                        if format!("{}", e).contains("connection") {
                            act.is_connected = false;
                            ctx.notify(Connect);
                        }
                    }
                }
                result
            })
        )
    }
}

impl Handler<ListTools> for ClaudeFlowActorTcp {
    type Result = ResponseActFuture<Self, McpResult<Vec<String>>>;

    fn handle(&mut self, _: ListTools, ctx: &mut Context<Self>) -> Self::Result {
        if !self.is_connected {
            ctx.notify(Connect);
            
            return Box::pin(
                async move {
                    Err(crate::services::claude_flow::error::ConnectorError::NotConnected.into())
                }
                .into_actor(self)
            );
        }

        let client = self.client.clone();
        
        Box::pin(
            async move {
                if let Some(mut client) = client {
                    let tools = client.list_tools().await?;
                    Ok(tools.into_iter().map(|t| t.name).collect())
                } else {
                    Err(crate::services::claude_flow::error::ConnectorError::NotConnected.into())
                }
            }
            .into_actor(self)
        )
    }
}

impl Handler<GetConnectionInfo> for ClaudeFlowActorTcp {
    type Result = ConnectionInfo;

    fn handle(&mut self, _: GetConnectionInfo, _: &mut Context<Self>) -> Self::Result {
        ConnectionInfo {
            connected: self.is_connected,
            transport: "TCP".to_string(),
            host: std::env::var("CLAUDE_FLOW_HOST")
                .unwrap_or_else(|_| "multi-agent-container".to_string()),
            port: std::env::var("MCP_TCP_PORT")
                .unwrap_or_else(|_| "9500".to_string())
                .parse()
                .unwrap_or(9500),
            stats: ConnectionStatsPublic {
                connection_attempts: self.connection_stats.connection_attempts,
                successful_connections: self.connection_stats.successful_connections,
                failed_connections: self.connection_stats.failed_connections,
                messages_sent: self.connection_stats.messages_sent,
                messages_received: self.connection_stats.messages_received,
                last_error: self.connection_stats.last_error.clone(),
            },
        }
    }
}