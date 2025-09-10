//! JSON-RPC Client - Handles MCP protocol message correlation and serialization
//! 
//! This component is responsible for:
//! - JSON-RPC message formatting and parsing
//! - Request/response correlation using IDs
//! - MCP protocol initialization
//! - Tool calling abstractions

use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::{RwLock, oneshot};
use std::sync::Arc;
use serde_json::{json, Value};
use uuid::Uuid;
use log::{info, error, debug, warn};
use actix::prelude::*;

use super::tcp_connection_actor::{TcpConnectionActor, TcpConnectionEvent, TcpConnectionEventType, SendJsonMessage, SubscribeToEvents};

/// JSON-RPC Client for MCP protocol handling
pub struct JsonRpcClient {
    /// Connection to TCP actor
    tcp_actor: Option<Addr<TcpConnectionActor>>,
    
    /// Pending requests awaiting responses
    pending_requests: Arc<RwLock<HashMap<String, oneshot::Sender<Value>>>>,
    
    /// MCP session state
    is_initialized: bool,
    session_id: Option<String>,
    
    /// Client capabilities
    client_info: ClientInfo,
    protocol_version: String,
}

#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

impl Default for ClientInfo {
    fn default() -> Self {
        Self {
            name: "visionflow".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

/// Messages for JsonRpcClient
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ConnectToTcpActor {
    pub tcp_actor: Addr<TcpConnectionActor>,
}

#[derive(Message)]
#[rtype(result = "Result<Value, String>")]
pub struct CallTool {
    pub tool_name: String,
    pub params: Value,
    pub timeout: Option<Duration>,
}

#[derive(Message)]
#[rtype(result = "Result<Value, String>")]
pub struct SendRequest {
    pub method: String,
    pub params: Value,
    pub timeout: Option<Duration>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SendNotification {
    pub method: String,
    pub params: Value,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InitializeMcpSession;

#[derive(Message)]
#[rtype(result = "()")]
pub struct ProcessIncomingMessage {
    pub message: Value,
}

impl JsonRpcClient {
    pub fn new() -> Self {
        Self {
            tcp_actor: None,
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            is_initialized: false,
            session_id: None,
            client_info: ClientInfo::default(),
            protocol_version: "1.0.0".to_string(),
        }
    }
    
    pub fn with_client_info(mut self, client_info: ClientInfo) -> Self {
        self.client_info = client_info;
        self
    }
    
    /// Generate a unique request ID
    fn generate_request_id() -> String {
        Uuid::new_v4().to_string()
    }
    
    /// Create a JSON-RPC request message
    fn create_request(id: String, method: String, params: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        })
    }
    
    /// Create a JSON-RPC notification message
    fn create_notification(method: String, params: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        })
    }
    
    /// Create MCP initialize request
    fn create_initialize_request(id: String, client_info: &ClientInfo, protocol_version: &str) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {
                    "roots": true,
                    "sampling": true,
                    "tools": true
                },
                "clientInfo": {
                    "name": client_info.name,
                    "version": client_info.version
                }
            }
        })
    }
    
    /// Create tool call request
    fn create_tool_call_request(id: String, tool_name: String, params: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        })
    }
    
    /// Send a request and wait for response
    async fn send_request_internal(
        tcp_actor: &Addr<TcpConnectionActor>,
        pending_requests: Arc<RwLock<HashMap<String, oneshot::Sender<Value>>>>,
        request: Value,
        request_id: String,
        timeout: Duration,
    ) -> Result<Value, String> {
        // Set up response channel
        let (tx, rx) = oneshot::channel();
        pending_requests.write().await.insert(request_id.clone(), tx);
        
        // Send the request
        match tcp_actor.send(SendJsonMessage { message: request }).await {
            Ok(Ok(())) => {
                // Wait for response with timeout
                match tokio::time::timeout(timeout, rx).await {
                    Ok(Ok(response)) => {
                        Self::validate_response(&response)?;
                        Ok(response)
                    }
                    Ok(Err(_)) => {
                        pending_requests.write().await.remove(&request_id);
                        Err("Response channel closed".to_string())
                    }
                    Err(_) => {
                        pending_requests.write().await.remove(&request_id);
                        Err("Request timeout".to_string())
                    }
                }
            }
            Ok(Err(e)) => {
                pending_requests.write().await.remove(&request_id);
                Err(format!("Failed to send request: {}", e))
            }
            Err(e) => {
                pending_requests.write().await.remove(&request_id);
                Err(format!("Actor communication error: {}", e))
            }
        }
    }
    
    /// Validate JSON-RPC response format
    fn validate_response(response: &Value) -> Result<(), String> {
        if !response.is_object() {
            return Err("Response is not a JSON object".to_string());
        }
        
        if response.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
            return Err("Invalid JSON-RPC version".to_string());
        }
        
        if response.as_object().map_or(true, |obj| !obj.contains_key("id")) {
            return Err("Response missing ID".to_string());
        }
        
        if response.as_object().map_or(false, |obj| obj.contains_key("error")) {
            if let Some(error) = response.get("error") {
                let error_message = error.get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error");
                return Err(format!("JSON-RPC error: {}", error_message));
            }
        }
        
        Ok(())
    }
    
    /// Process an incoming message (response or notification)
    fn process_message(&self, _ctx: &mut Context<Self>, message: Value) {
        debug!("Processing JSON-RPC message: {:?}", message);
        
        // Check if this is a response to a pending request
        if let Some(id) = message.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()) {
            let pending_requests = self.pending_requests.clone();
            let message_clone = message.clone();
            
            tokio::spawn(async move {
                let mut requests = pending_requests.write().await;
                if let Some(sender) = requests.remove(&id) {
                    debug!("Correlating response for request ID: {}", id);
                    let _ = sender.send(message_clone);
                } else {
                    warn!("Received response for unknown request ID: {}", id);
                }
            });
        } else {
            // This might be a notification or event
            if let Some(method) = message.get("method").and_then(|m| m.as_str()) {
                debug!("Received JSON-RPC notification: {}", method);
                // Handle notifications (could be extended for specific MCP events)
            }
        }
    }
}

impl Actor for JsonRpcClient {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("JsonRpcClient started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("JsonRpcClient stopped - cleaning up pending requests");
        
        // Clear pending requests with error responses
        let pending_requests = self.pending_requests.clone();
        tokio::spawn(async move {
            let mut requests = pending_requests.write().await;
            for (id, sender) in requests.drain() {
                let error_response = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -1,
                        "message": "Client shutting down"
                    }
                });
                let _ = sender.send(error_response);
            }
        });
    }
}

impl Handler<ConnectToTcpActor> for JsonRpcClient {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: ConnectToTcpActor, ctx: &mut Self::Context) -> Self::Result {
        info!("Connecting JsonRpcClient to TcpConnectionActor");
        
        self.tcp_actor = Some(msg.tcp_actor.clone());
        
        // Subscribe to TCP connection events
        let subscriber = ctx.address().recipient::<TcpConnectionEvent>();
        msg.tcp_actor.do_send(SubscribeToEvents { subscriber });
        
        Ok(())
    }
}

impl Handler<InitializeMcpSession> for JsonRpcClient {
    type Result = ResponseFuture<Result<(), String>>;
    
    fn handle(&mut self, _: InitializeMcpSession, _ctx: &mut Self::Context) -> Self::Result {
        if self.is_initialized {
            return Box::pin(async move { Ok(()) });
        }
        
        let tcp_actor = match self.tcp_actor.clone() {
            Some(actor) => actor,
            None => return Box::pin(async move { Err("No TCP connection available".to_string()) }),
        };
        
        let pending_requests = self.pending_requests.clone();
        let client_info = self.client_info.clone();
        let protocol_version = self.protocol_version.clone();
        
        Box::pin(async move {
            let request_id = Self::generate_request_id();
            let request = Self::create_initialize_request(request_id.clone(), &client_info, &protocol_version);
            
            match Self::send_request_internal(
                &tcp_actor,
                pending_requests,
                request,
                request_id,
                Duration::from_secs(10),
            ).await {
                Ok(response) => {
                    info!("MCP session initialized successfully: {:?}", response);
                    Ok(())
                }
                Err(e) => {
                    error!("Failed to initialize MCP session: {}", e);
                    Err(e)
                }
            }
        })
    }
}

impl Handler<CallTool> for JsonRpcClient {
    type Result = ResponseFuture<Result<Value, String>>;
    
    fn handle(&mut self, msg: CallTool, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_initialized {
            return Box::pin(async move { 
                Err("MCP session not initialized".to_string()) 
            });
        }
        
        let tcp_actor = match self.tcp_actor.clone() {
            Some(actor) => actor,
            None => return Box::pin(async move { Err("No TCP connection available".to_string()) }),
        };
        
        let pending_requests = self.pending_requests.clone();
        let timeout = msg.timeout.unwrap_or(Duration::from_secs(30));
        
        Box::pin(async move {
            let request_id = Self::generate_request_id();
            let request = Self::create_tool_call_request(request_id.clone(), msg.tool_name.clone(), msg.params);
            
            debug!("Calling tool '{}' with request ID: {}", msg.tool_name, request_id);
            
            Self::send_request_internal(
                &tcp_actor,
                pending_requests,
                request,
                request_id,
                timeout,
            ).await
        })
    }
}

impl Handler<SendRequest> for JsonRpcClient {
    type Result = ResponseFuture<Result<Value, String>>;
    
    fn handle(&mut self, msg: SendRequest, _ctx: &mut Self::Context) -> Self::Result {
        let tcp_actor = match self.tcp_actor.clone() {
            Some(actor) => actor,
            None => return Box::pin(async move { Err("No TCP connection available".to_string()) }),
        };
        
        let pending_requests = self.pending_requests.clone();
        let timeout = msg.timeout.unwrap_or(Duration::from_secs(10));
        
        Box::pin(async move {
            let request_id = Self::generate_request_id();
            let request = Self::create_request(request_id.clone(), msg.method.clone(), msg.params);
            
            debug!("Sending request '{}' with ID: {}", msg.method, request_id);
            
            Self::send_request_internal(
                &tcp_actor,
                pending_requests,
                request,
                request_id,
                timeout,
            ).await
        })
    }
}

impl Handler<SendNotification> for JsonRpcClient {
    type Result = ResponseFuture<Result<(), String>>;
    
    fn handle(&mut self, msg: SendNotification, _ctx: &mut Self::Context) -> Self::Result {
        let tcp_actor = match self.tcp_actor.clone() {
            Some(actor) => actor,
            None => return Box::pin(async move { Err("No TCP connection available".to_string()) }),
        };
        
        Box::pin(async move {
            let notification = Self::create_notification(msg.method.clone(), msg.params);
            
            debug!("Sending notification: {}", msg.method);
            
            match tcp_actor.send(SendJsonMessage { message: notification }).await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(format!("Failed to send notification: {}", e)),
                Err(e) => Err(format!("Actor communication error: {}", e)),
            }
        })
    }
}

impl Handler<TcpConnectionEvent> for JsonRpcClient {
    type Result = ();
    
    fn handle(&mut self, msg: TcpConnectionEvent, ctx: &mut Self::Context) {
        match msg.event_type {
            TcpConnectionEventType::Connected => {
                info!("TCP connection established, ready for MCP initialization");
                // Could automatically initialize MCP session here
            }
            TcpConnectionEventType::Disconnected => {
                warn!("TCP connection lost, marking MCP session as uninitialized");
                self.is_initialized = false;
                self.session_id = None;
            }
            TcpConnectionEventType::MessageReceived => {
                if let Some(message) = msg.data {
                    self.process_message(ctx, message);
                }
            }
            TcpConnectionEventType::MessageSent => {
                debug!("Message sent successfully");
            }
            TcpConnectionEventType::Error(error) => {
                error!("TCP connection error: {}", error);
            }
        }
    }
}

impl Handler<ProcessIncomingMessage> for JsonRpcClient {
    type Result = ();
    
    fn handle(&mut self, msg: ProcessIncomingMessage, ctx: &mut Self::Context) {
        self.process_message(ctx, msg.message);
    }
}