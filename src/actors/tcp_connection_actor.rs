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
use socket2::{Socket, Domain, Type, Protocol, SockRef};
use std::net::SocketAddr;
use fastrand;

use crate::utils::network::{
    NetworkResilienceManager, CircuitBreaker,
    retry_tcp_connection, RetryableError,
    TimeoutConfig, ConnectionPool, ConnectionPoolConfig
};
use crate::utils::resource_monitor::{ResourceMonitor, ResourceLimits};

/// TCP connection state tracking
#[derive(Debug, Clone, PartialEq)]
pub enum TcpConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed(String),
}

/// TCP connection configuration for keep-alive and stability
#[derive(Debug, Clone)]
pub struct TcpConnectionConfig {
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub keep_alive_timeout: Duration,
    pub keep_alive_interval: Duration,
    pub keep_alive_retries: u32,
    pub max_retry_attempts: u32,
    pub enable_nodelay: bool,
    pub enable_keep_alive: bool,
}

impl Default for TcpConnectionConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(5),
            keep_alive_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(10),
            keep_alive_retries: 3,
            max_retry_attempts: 3,
            enable_nodelay: true,
            enable_keep_alive: true,
        }
    }
}

/// TCP Connection Actor - manages raw TCP streams and connection lifecycle
pub struct TcpConnectionActor {
    /// Connection configuration
    host: String,
    port: u16,
    
    /// Connection state and configuration
    connection_state: TcpConnectionState,
    connection_id: Option<String>,
    connection_config: TcpConnectionConfig,
    
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
            connection_state: TcpConnectionState::Disconnected,
            connection_id: None,
            connection_config: TcpConnectionConfig::default(),
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

    /// Create a new TCP connection actor with custom configuration
    pub fn with_config(host: String, port: u16, config: TcpConnectionConfig) -> Self {
        info!("Creating TcpConnectionActor for {}:{} with custom config", host, port);

        let mut actor = Self::new(host, port);
        actor.connection_config = config;
        actor
    }

    /// Get current connection state
    pub fn connection_state(&self) -> &TcpConnectionState {
        &self.connection_state
    }

    /// Check if connection is healthy and ready for operations
    pub fn is_connection_healthy(&self) -> bool {
        matches!(self.connection_state, TcpConnectionState::Connected) &&
        self.tcp_writer.is_some() &&
        self.tcp_reader.is_some()
    }

    /// Update connection state and notify subscribers
    fn update_connection_state(&mut self, new_state: TcpConnectionState) {
        let old_state = self.connection_state.clone();
        self.connection_state = new_state.clone();

        debug!("Connection state changed from {:?} to {:?}", old_state, new_state);

        // Notify subscribers of state change
        if let Some(connection_id) = &self.connection_id {
            let event_type = match new_state {
                TcpConnectionState::Connected => TcpConnectionEventType::Connected,
                TcpConnectionState::Disconnected | TcpConnectionState::Failed(_) => TcpConnectionEventType::Disconnected,
                _ => return, // Don't notify for intermediate states
            };

            self.notify_subscribers(TcpConnectionEvent {
                event_type,
                connection_id: connection_id.clone(),
                data: None,
            });
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

        // Update connection state to connecting
        self.update_connection_state(TcpConnectionState::Connecting);

        self.initialize_resilience();

        let addr = ctx.address();
        let host = self.host.clone();
        let port = self.port;
        let active_connections = self.active_connections.clone();
        let config = self.connection_config.clone();

        tokio::spawn(async move {
            let connection_id = Uuid::new_v4().to_string();
            {
                let mut connections = active_connections.write().await;
                connections.insert(connection_id.clone(), std::time::Instant::now());
            }

            let connection_operation = || async {
                Self::connect_tcp_with_keepalive(&host, port, &config).await
                    .map_err(|e| std::sync::Arc::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
            };

            match retry_tcp_connection(connection_operation).await {
                Ok((writer, reader)) => {
                    info!("TCP connection established to {}:{} with keep-alive configured", host, port);
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
    
    /// Low-level TCP connection establishment with keep-alive configuration
    async fn connect_tcp_with_keepalive(
        host: &str,
        port: u16,
        config: &TcpConnectionConfig,
    ) -> Result<(BufWriter<tokio::net::tcp::OwnedWriteHalf>, BufReader<tokio::net::tcp::OwnedReadHalf>), Box<dyn std::error::Error + Send + Sync>> {
        let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
        debug!("Connecting to TCP address: {} with keep-alive configuration", addr);

        // Create socket with socket2 for advanced options
        let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;

        // Configure TCP keep-alive
        if config.enable_keep_alive {
            socket.set_keepalive(true)?;

            #[cfg(target_os = "linux")]
            {
                // Linux-specific keep-alive configuration
                let sock_ref = SockRef::from(&socket);

                // Set keep-alive timeout (time before first probe)
                if let Err(e) = sock_ref.set_tcp_keepalive(&socket2::TcpKeepalive::new()
                    .with_time(config.keep_alive_timeout)
                    .with_interval(config.keep_alive_interval)
                    .with_retries(config.keep_alive_retries)) {
                    warn!("Failed to set TCP keep-alive parameters: {}", e);
                }
            }

            #[cfg(not(target_os = "linux"))]
            {
                // Fallback for other platforms
                if let Err(e) = socket.set_tcp_keepalive(&socket2::TcpKeepalive::new().with_time(config.keep_alive_timeout)) {
                    warn!("Failed to set basic TCP keep-alive: {}", e);
                }
            }

            debug!("TCP keep-alive configured: timeout={}s, interval={}s, retries={}",
                   config.keep_alive_timeout.as_secs(),
                   config.keep_alive_interval.as_secs(),
                   config.keep_alive_retries);
        }

        // Configure TCP_NODELAY for low latency
        if config.enable_nodelay {
            socket.set_nodelay(true)?;
            debug!("TCP_NODELAY enabled for low latency");
        }

        // Set socket timeouts
        socket.set_read_timeout(Some(config.read_timeout))?;
        socket.set_write_timeout(Some(config.write_timeout))?;

        // Connect with timeout - socket2's connect returns Result<(), Error>, not a Future
        // We need to wrap it in an async block to make it awaitable
        let connect_future = async {
            socket.connect(&addr.into())
        };

        let connect_result = tokio::time::timeout(config.connect_timeout, connect_future).await;

        match connect_result {
            Ok(Ok(())) => {
                debug!("TCP socket connected successfully with advanced options");
            }
            Ok(Err(e)) => {
                error!("TCP socket connection failed: {}", e);
                return Err(Box::new(e));
            }
            Err(_) => {
                error!("TCP connection timeout after {:?}", config.connect_timeout);
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Connection timeout"
                )));
            }
        }

        // Convert socket2::Socket to tokio::net::TcpStream
        let std_stream: std::net::TcpStream = socket.into();
        std_stream.set_nonblocking(true)?;
        let stream = TcpStream::from_std(std_stream)?;

        let (read_half, write_half) = stream.into_split();
        let reader = BufReader::new(read_half);
        let writer = BufWriter::new(write_half);

        info!("TCP connection established with keep-alive: host={}, port={}", host, port);
        Ok((writer, reader))
    }

    /// Legacy method for backward compatibility
    async fn connect_tcp(
        host: &str,
        port: u16
    ) -> Result<(BufWriter<tokio::net::tcp::OwnedWriteHalf>, BufReader<tokio::net::tcp::OwnedReadHalf>), Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_tcp_with_keepalive(host, port, &TcpConnectionConfig::default()).await
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

    /// Check if an error indicates the connection is broken and needs reconnection
    fn is_connection_broken_error(error: &Box<dyn std::error::Error + Send + Sync>) -> bool {
        let error_str = error.to_string().to_lowercase();
        error_str.contains("broken pipe") ||
        error_str.contains("connection reset") ||
        error_str.contains("connection aborted") ||
        error_str.contains("connection refused") ||
        error_str.contains("not connected") ||
        error_str.contains("connection lost")
    }

    /// Validate connection health by checking various indicators
    fn validate_connection_health(&self) -> Result<(), String> {
        // Check connection state
        if !matches!(self.connection_state, TcpConnectionState::Connected) {
            return Err(format!("Connection state is {:?}", self.connection_state));
        }

        // Check if writer and reader are available
        if self.tcp_writer.is_none() {
            return Err("TCP writer not available".to_string());
        }

        if self.tcp_reader.is_none() {
            return Err("TCP reader not available".to_string());
        }

        // Check connection age (optional health indicator)
        if let Some(connected_at) = self.connection_stats.connected_at {
            let connection_age = Utc::now() - connected_at;
            if connection_age > chrono::Duration::hours(24) {
                warn!("Connection is {} hours old, consider refresh", connection_age.num_hours());
            }
        }

        Ok(())
    }
}

impl Actor for TcpConnectionActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
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
            // Use actix::spawn instead of tokio::spawn to avoid runtime panic
            actix::spawn(async move {
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
        if matches!(self.connection_state, TcpConnectionState::Connected) {
            debug!("Already connected to {}:{}", self.host, self.port);
            return Ok(());
        }

        if matches!(self.connection_state, TcpConnectionState::Connecting) {
            debug!("Connection already in progress to {}:{}", self.host, self.port);
            return Ok(());
        }

        info!("Establishing connection to {}:{}", self.host, self.port);
        self.establish_connection(ctx);
        Ok(())
    }
}

impl Handler<CloseConnection> for TcpConnectionActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _: CloseConnection, _ctx: &mut Self::Context) -> Self::Result {
        if matches!(self.connection_state, TcpConnectionState::Disconnected) {
            debug!("Already disconnected from {}:{}", self.host, self.port);
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

        // Update connection state
        self.update_connection_state(TcpConnectionState::Disconnected);
        self.connection_id = None;

        Ok(())
    }
}

impl Handler<SendRawMessage> for TcpConnectionActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SendRawMessage, ctx: &mut Self::Context) -> Self::Result {
        // Validate connection health before sending
        if !self.is_connection_healthy() {
            warn!("Connection not healthy for sending raw message to {}:{}", self.host, self.port);

            // Attempt to reconnect if disconnected
            if matches!(self.connection_state, TcpConnectionState::Disconnected | TcpConnectionState::Failed(_)) {
                info!("Attempting to reconnect before sending message");
                self.establish_connection(ctx);
            }

            return Err(format!("Connection not healthy. State: {:?}", self.connection_state));
        }

        if let Some(writer_arc) = self.tcp_writer.clone() {
            let data = msg.data;
            let mut stats = self.connection_stats.clone();
            let host = self.host.clone();
            let port = self.port;
            let addr = ctx.address();

            tokio::spawn(async move {
                let mut writer = writer_arc.write().await;
                if let Err(e) = Self::send_raw_bytes(&mut *writer, &data, &mut stats).await {
                    error!("Failed to send raw message to {}:{}: {}", host, port, e);

                    // Check if error indicates connection is broken
                    if Self::is_connection_broken_error(&e) {
                        warn!("Connection appears broken, triggering reconnection");
                        addr.do_send(ConnectionLost);
                    }
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

    fn handle(&mut self, msg: SendJsonMessage, ctx: &mut Self::Context) -> Self::Result {
        // Validate connection health before sending
        if !self.is_connection_healthy() {
            warn!("Connection not healthy for sending JSON message to {}:{}", self.host, self.port);

            // Attempt to reconnect if disconnected
            if matches!(self.connection_state, TcpConnectionState::Disconnected | TcpConnectionState::Failed(_)) {
                info!("Attempting to reconnect before sending JSON message");
                self.establish_connection(ctx);
            }

            return Err(format!("Connection not healthy. State: {:?}", self.connection_state));
        }

        if let Some(writer_arc) = self.tcp_writer.clone() {
            let message = msg.message;
            let mut stats = self.connection_stats.clone();
            let subscribers = self.subscribers.clone();
            let connection_id = self.connection_id.clone().unwrap_or_default();
            let host = self.host.clone();
            let port = self.port;
            let addr = ctx.address();

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
                        error!("Failed to send JSON message to {}:{}: {}", host, port, e);

                        // Check if error indicates connection is broken
                        if Self::is_connection_broken_error(&e) {
                            warn!("Connection appears broken, triggering reconnection");
                            addr.do_send(ConnectionLost);
                        }

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
        info!("TCP connection established successfully to {}:{}", self.host, self.port);

        self.tcp_writer = Some(Arc::new(RwLock::new(msg.writer)));
        self.tcp_reader = Some(Arc::new(RwLock::new(msg.reader)));
        self.connection_id = Some(Uuid::new_v4().to_string());
        self.connection_stats.connected_at = Some(Utc::now());
        self.connection_stats.reconnect_attempts = 0;

        // Update connection state to connected
        self.update_connection_state(TcpConnectionState::Connected);

        // Start reading messages
        self.start_message_reader(ctx);

        info!("TCP connection fully established and operational: {}:{}", self.host, self.port);
    }
}

impl Handler<ConnectionLost> for TcpConnectionActor {
    type Result = ();

    fn handle(&mut self, _: ConnectionLost, ctx: &mut Self::Context) {
        warn!("TCP connection lost to {}:{}", self.host, self.port);

        let old_connection_id = self.connection_id.take().unwrap_or_default();

        // Close existing connections
        self.tcp_writer.take();
        self.tcp_reader.take();

        // Update stats
        self.connection_stats.reconnect_attempts += 1;

        // Update connection state to reconnecting
        self.update_connection_state(TcpConnectionState::Reconnecting);

        // Calculate exponential backoff with improved algorithm
        let base_delay = Duration::from_millis(500); // Start with 500ms
        let max_delay = Duration::from_secs(60); // Cap at 60 seconds
        let attempts = self.connection_stats.reconnect_attempts.min(10); // Limit exponential growth

        let backoff_delay = std::cmp::min(
            base_delay * 2_u32.saturating_pow(attempts),
            max_delay
        );

        // Add jitter to prevent thundering herd
        let jitter = fastrand::f64() * 0.1; // 10% jitter
        let jittered_delay = Duration::from_millis(
            (backoff_delay.as_millis() as f64 * (1.0 + jitter)) as u64
        );

        info!("Scheduling reconnection attempt {} in {:?} (with jitter)",
              self.connection_stats.reconnect_attempts, jittered_delay);

        // Check if we've exceeded max retry attempts
        if self.connection_stats.reconnect_attempts >= self.connection_config.max_retry_attempts {
            error!("Max reconnection attempts ({}) exceeded for {}:{}",
                   self.connection_config.max_retry_attempts, self.host, self.port);
            self.update_connection_state(TcpConnectionState::Failed(
                format!("Max retries ({}) exceeded", self.connection_config.max_retry_attempts)
            ));
            return;
        }

        ctx.run_later(jittered_delay, |act, ctx| {
            info!("Attempting to reconnect (attempt {}) to {}:{}",
                  act.connection_stats.reconnect_attempts, act.host, act.port);
            act.establish_connection(ctx);
        });
    }
}