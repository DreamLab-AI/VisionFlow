//! TCP Connection Actor - Handles low-level TCP stream management
//! 
//! This actor is responsible for:
//! - Establishing and maintaining TCP connections
//! - Reading/writing raw TCP data
//! - Connection resilience (reconnection, timeouts)
//! - Connection lifecycle management

use actix::prelude::*;
use std::time::Duration;
use log::{info, error, debug, warn};
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{Utc, DateTime};
use uuid::Uuid;
use serde_json::Value;

use crate::utils::network::{
    NetworkResilienceManager, CircuitBreaker,
    retry_tcp_connection, RetryableError,
    TimeoutConfig, ConnectionPool, ConnectionPoolConfig
};
use crate::utils::resource_monitor::{ResourceMonitor, ResourceLimits};

/// TCP Connection Actor - manages raw TCP streams and connection lifecycle
pub struct TcpConnectionActor {
    /// Connection configuration
    host: String,
    port: u16,
    
    /// Connection state
    is_connected: bool,
    connection_id: Option<String>,
    
    /// TCP stream components
    tcp_writer: Option<Arc<RwLock<BufWriter<tokio::net::tcp::OwnedWriteHalf>>>>,
    tcp_reader: Option<Arc<RwLock<BufReader<tokio::net::tcp::OwnedReadHalf>>>>,
    
    /// Connection statistics
    connection_stats: ConnectionStats,
    
    /// Network resilience components
    resilience_manager: Arc<NetworkResilienceManager>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    timeout_config: TimeoutConfig,
    connection_pool: Option<Arc<tokio::sync::Mutex<ConnectionPool>>>,
    
    /// Resource monitoring
    active_connections: Arc<tokio::sync::RwLock<std::collections::HashMap<String, std::time::Instant>>>,
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Subscribers for connection events
    subscribers: Vec<Recipient<TcpConnectionEvent>>,
}

#[derive(Debug, Default, Clone)]
struct ConnectionStats {
    connected_at: Option<DateTime<Utc>>,
    messages_sent: u64,
    messages_received: u64,
    bytes_sent: u64,
    bytes_received: u64,
    last_message_at: Option<DateTime<Utc>>,
    reconnect_attempts: u32,
}

/// Connection events that subscribers can listen to
#[derive(Message, Clone)]
#[rtype(result = "()")]
pub struct TcpConnectionEvent {
    pub event_type: TcpConnectionEventType,
    pub connection_id: String,
    pub data: Option<Value>,
}

#[derive(Debug, Clone)]
pub enum TcpConnectionEventType {
    Connected,
    Disconnected,
    MessageReceived,
    MessageSent,
    Error(String),
}

/// Messages for TcpConnectionActor
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct EstablishConnection;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct CloseConnection;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SendRawMessage {
    pub data: Vec<u8>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SendJsonMessage {
    pub message: Value,
}

#[derive(Message)]
#[rtype(result = "()")]
pub struct SubscribeToEvents {
    pub subscriber: Recipient<TcpConnectionEvent>,
}

#[derive(Message)]
#[rtype(result = "()")]
struct ConnectionEstablished {
    writer: BufWriter<tokio::net::tcp::OwnedWriteHalf>,
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
}

#[derive(Message)]
#[rtype(result = "()")]
struct ConnectionLost;

// RetryableError implementation for Box<dyn Error> is defined in claude_flow_actor_tcp.rs to avoid conflicts

impl TcpConnectionActor {
    pub fn new(host: String, port: u16) -> Self {
        info!("Creating TcpConnectionActor for {}:{}", host, port);
        
        Self {
            host,
            port,
            is_connected: false,
            connection_id: None,
            tcp_writer: None,
            tcp_reader: None,
            connection_stats: ConnectionStats::default(),
            resilience_manager: Arc::new(NetworkResilienceManager::new()),
            circuit_breaker: None,
            timeout_config: TimeoutConfig::tcp_connection(),
            connection_pool: None,
            active_connections: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            resource_monitor: Arc::new(ResourceMonitor::new(ResourceLimits::default())),
            subscribers: Vec::new(),
        }
    }
    
    /// Initialize connection pool and resilience components
    fn initialize_resilience(&mut self) {
        if self.connection_pool.is_none() {
            let pool_config = ConnectionPoolConfig {
                max_connections_per_endpoint: 2,
                max_total_connections: 5,
                connection_timeout: std::time::Duration::from_secs(10),
                idle_timeout: std::time::Duration::from_secs(60),
                max_connection_lifetime: std::time::Duration::from_secs(300),
                cleanup_interval: std::time::Duration::from_secs(30),
                validate_on_borrow: true,
                validate_while_idle: false,
            };
            let mut pool = ConnectionPool::new(pool_config);
            pool.start_cleanup_task();
            self.connection_pool = Some(Arc::new(tokio::sync::Mutex::new(pool)));
        }
    }
    
    /// Establish TCP connection with resilience patterns
    fn establish_connection(&mut self, ctx: &mut Context<Self>) {
        debug!("Establishing TCP connection to {}:{}", self.host, self.port);
        
        self.initialize_resilience();
        
        let addr = ctx.address();
        let host = self.host.clone();
        let port = self.port;
        let active_connections = self.active_connections.clone();
        
        tokio::spawn(async move {
            let connection_id = Uuid::new_v4().to_string();
            {
                let mut connections = active_connections.write().await;
                connections.insert(connection_id.clone(), std::time::Instant::now());
            }
            
            let connection_operation = || async {
                Self::connect_tcp(&host, port).await
                    .map_err(|e| std::sync::Arc::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
            };
            
            match retry_tcp_connection(connection_operation).await {
                Ok((writer, reader)) => {
                    info!("TCP connection established to {}:{}", host, port);
                    addr.do_send(ConnectionEstablished { writer, reader });
                }
                Err(e) => {
                    error!("Failed to establish TCP connection: {:?}", e);
                    addr.do_send(ConnectionLost);
                }
            }
            
            {
                let mut connections = active_connections.write().await;
                connections.remove(&connection_id);
            }
        });
    }
    
    /// Low-level TCP connection establishment
    async fn connect_tcp(
        host: &str, 
        port: u16
    ) -> Result<(BufWriter<tokio::net::tcp::OwnedWriteHalf>, BufReader<tokio::net::tcp::OwnedReadHalf>), Box<dyn std::error::Error + Send + Sync>> {
        let addr = format!("{}:{}", host, port);
        debug!("Connecting to TCP address: {}", addr);
        
        let stream = TcpStream::connect(&addr).await?;
        stream.set_nodelay(true)?;
        
        let (read_half, write_half) = stream.into_split();
        let reader = BufReader::new(read_half);
        let writer = BufWriter::new(write_half);
        
        Ok((writer, reader))
    }
    
    /// Start reading messages from TCP stream
    fn start_message_reader(&mut self, ctx: &mut Context<Self>) {
        if let Some(reader_arc) = self.tcp_reader.clone() {
            let addr = ctx.address();
            let connection_id = self.connection_id.clone().unwrap_or_default();
            let subscribers = self.subscribers.clone();
            
            tokio::spawn(async move {
                let mut reader = reader_arc.write().await;
                
                loop {
                    match Self::read_line_message(&mut *reader).await {
                        Ok(line) => {
                            debug!("Received TCP message: {}", line);
                            
                            // Try to parse as JSON
                            match serde_json::from_str::<Value>(&line) {
                                Ok(json_message) => {
                                    // Notify subscribers
                                    let event = TcpConnectionEvent {
                                        event_type: TcpConnectionEventType::MessageReceived,
                                        connection_id: connection_id.clone(),
                                        data: Some(json_message),
                                    };
                                    
                                    for subscriber in &subscribers {
                                        let _ = subscriber.do_send(event.clone());
                                    }
                                }
                                Err(e) => {
                                    warn!("Received non-JSON message: {} (parse error: {})", line, e);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Error reading TCP message: {}", e);
                            break;
                        }
                    }
                }
                
                // Connection lost
                addr.do_send(ConnectionLost);
            });
        }
    }
    
    /// Read a line-delimited message from TCP stream
    async fn read_line_message(
        reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        Ok(line.trim().to_string())
    }
    
    /// Send raw bytes over TCP connection
    async fn send_raw_bytes(
        writer: &mut BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        data: &[u8],
        stats: &mut ConnectionStats
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        writer.write_all(data).await?;
        writer.flush().await?;
        
        // Update stats
        stats.messages_sent += 1;
        stats.bytes_sent += data.len() as u64;
        stats.last_message_at = Some(Utc::now());
        
        Ok(())
    }
    
    /// Send JSON message over TCP connection
    async fn send_json_message(
        writer: &mut BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        message: &Value,
        stats: &mut ConnectionStats
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let msg_str = serde_json::to_string(message)?;
        let msg_bytes = format!("{}\n", msg_str);
        
        Self::send_raw_bytes(writer, msg_bytes.as_bytes(), stats).await?;
        debug!("Sent JSON message: {}", msg_str);
        
        Ok(())
    }
    
    /// Notify subscribers of an event
    fn notify_subscribers(&self, event: TcpConnectionEvent) {
        for subscriber in &self.subscribers {
            let _ = subscriber.do_send(event.clone());
        }
    }
}

impl Actor for TcpConnectionActor {
    type Context = Context<Self>;
    
    fn started(&mut self, ctx: &mut Self::Context) {
        info!("TcpConnectionActor started for {}:{}", self.host, self.port);
        
        // Start resource monitoring
        let resource_monitor = self.resource_monitor.clone();
        tokio::spawn(async move {
            if let Err(e) = resource_monitor.start_monitoring(std::time::Duration::from_secs(10)).await {
                error!("Failed to start resource monitoring: {}", e);
            }
        });
    }
    
    fn stopped(&mut self, _: &mut Self::Context) {
        info!("TcpConnectionActor stopping - cleaning up connection to {}:{}", self.host, self.port);
        
        // Cleanup connections
        if let Some(pool) = self.connection_pool.take() {
            tokio::spawn(async move {
                let mut pool_guard = pool.lock().await;
                pool_guard.shutdown().await;
            });
        }
        
        // Notify subscribers of disconnection
        if let Some(connection_id) = &self.connection_id {
            self.notify_subscribers(TcpConnectionEvent {
                event_type: TcpConnectionEventType::Disconnected,
                connection_id: connection_id.clone(),
                data: None,
            });
        }
    }
}

impl Handler<EstablishConnection> for TcpConnectionActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _: EstablishConnection, ctx: &mut Self::Context) -> Self::Result {
        if self.is_connected {
            return Ok(());
        }
        
        self.establish_connection(ctx);
        Ok(())
    }
}

impl Handler<CloseConnection> for TcpConnectionActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _: CloseConnection, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_connected {
            return Ok(());
        }
        
        info!("Closing TCP connection to {}:{}", self.host, self.port);
        
        // Close writer
        if let Some(writer_arc) = self.tcp_writer.take() {
            tokio::spawn(async move {
                if let Ok(mut writer) = writer_arc.try_write() {
                    let _ = writer.shutdown().await;
                }
            });
        }
        
        // Reader will be dropped automatically
        self.tcp_reader.take();
        
        self.is_connected = false;
        self.connection_id = None;
        
        // Notify subscribers
        self.notify_subscribers(TcpConnectionEvent {
            event_type: TcpConnectionEventType::Disconnected,
            connection_id: "manual_close".to_string(),
            data: None,
        });
        
        Ok(())
    }
}

impl Handler<SendRawMessage> for TcpConnectionActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: SendRawMessage, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_connected {
            return Err("Not connected".to_string());
        }
        
        if let Some(writer_arc) = self.tcp_writer.clone() {
            let data = msg.data;
            let mut stats = self.connection_stats.clone();
            
            tokio::spawn(async move {
                let mut writer = writer_arc.write().await;
                if let Err(e) = Self::send_raw_bytes(&mut *writer, &data, &mut stats).await {
                    error!("Failed to send raw message: {}", e);
                }
            });
            
            Ok(())
        } else {
            Err("No TCP writer available".to_string())
        }
    }
}

impl Handler<SendJsonMessage> for TcpConnectionActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: SendJsonMessage, _ctx: &mut Self::Context) -> Self::Result {
        if !self.is_connected {
            return Err("Not connected".to_string());
        }
        
        if let Some(writer_arc) = self.tcp_writer.clone() {
            let message = msg.message;
            let mut stats = self.connection_stats.clone();
            let subscribers = self.subscribers.clone();
            let connection_id = self.connection_id.clone().unwrap_or_default();
            
            tokio::spawn(async move {
                let mut writer = writer_arc.write().await;
                match Self::send_json_message(&mut *writer, &message, &mut stats).await {
                    Ok(_) => {
                        // Notify subscribers of successful send
                        let event = TcpConnectionEvent {
                            event_type: TcpConnectionEventType::MessageSent,
                            connection_id,
                            data: Some(message),
                        };
                        for subscriber in &subscribers {
                            let _ = subscriber.do_send(event.clone());
                        }
                    }
                    Err(e) => {
                        error!("Failed to send JSON message: {}", e);
                        // Notify subscribers of error
                        let event = TcpConnectionEvent {
                            event_type: TcpConnectionEventType::Error(e.to_string()),
                            connection_id,
                            data: Some(message),
                        };
                        for subscriber in &subscribers {
                            let _ = subscriber.do_send(event.clone());
                        }
                    }
                }
            });
            
            Ok(())
        } else {
            Err("No TCP writer available".to_string())
        }
    }
}

impl Handler<SubscribeToEvents> for TcpConnectionActor {
    type Result = ();
    
    fn handle(&mut self, msg: SubscribeToEvents, _ctx: &mut Self::Context) {
        debug!("New subscriber registered for TCP connection events");
        self.subscribers.push(msg.subscriber);
    }
}

impl Handler<ConnectionEstablished> for TcpConnectionActor {
    type Result = ();
    
    fn handle(&mut self, msg: ConnectionEstablished, ctx: &mut Self::Context) {
        info!("TCP connection established successfully");
        
        self.tcp_writer = Some(Arc::new(RwLock::new(msg.writer)));
        self.tcp_reader = Some(Arc::new(RwLock::new(msg.reader)));
        self.is_connected = true;
        self.connection_id = Some(Uuid::new_v4().to_string());
        self.connection_stats.connected_at = Some(Utc::now());
        self.connection_stats.reconnect_attempts = 0;
        
        // Start reading messages
        self.start_message_reader(ctx);
        
        // Notify subscribers
        if let Some(connection_id) = &self.connection_id {
            self.notify_subscribers(TcpConnectionEvent {
                event_type: TcpConnectionEventType::Connected,
                connection_id: connection_id.clone(),
                data: None,
            });
        }
    }
}

impl Handler<ConnectionLost> for TcpConnectionActor {
    type Result = ();
    
    fn handle(&mut self, _: ConnectionLost, ctx: &mut Self::Context) {
        warn!("TCP connection lost to {}:{}", self.host, self.port);
        
        self.is_connected = false;
        let old_connection_id = self.connection_id.take().unwrap_or_default();
        
        // Close existing connections
        self.tcp_writer.take();
        self.tcp_reader.take();
        
        // Update stats
        self.connection_stats.reconnect_attempts += 1;
        
        // Notify subscribers
        self.notify_subscribers(TcpConnectionEvent {
            event_type: TcpConnectionEventType::Disconnected,
            connection_id: old_connection_id,
            data: None,
        });
        
        // Schedule reconnection with exponential backoff
        let backoff_delay = std::cmp::min(
            Duration::from_secs(5 * (1 << self.connection_stats.reconnect_attempts.min(5))),
            Duration::from_secs(300)
        );
        
        info!("Scheduling reconnection attempt {} in {:?}", 
              self.connection_stats.reconnect_attempts, backoff_delay);
        
        ctx.run_later(backoff_delay, |act, ctx| {
            info!("Attempting to reconnect (attempt {})", act.connection_stats.reconnect_attempts);
            act.establish_connection(ctx);
        });
    }
}