//! Client Coordinator Actor - WebSocket Communication Management
//!
//! This actor coordinates all client-related WebSocket communications, handling:
//! - Real-time position updates broadcasting
//! - Client connection state management
//! - Force broadcasts for new clients
//! - Initial client synchronization
//! - Adaptive broadcasting based on graph state
//!
//! ## Key Features
//! - **Time-based Broadcasting**: Prevents spam during stable periods
//! - **Force Broadcast Support**: Immediate updates for new clients
//! - **Efficient Binary Protocol**: Optimized WebSocket data transmission
//! - **Telemetry Integration**: Comprehensive logging and monitoring
//! - **Connection Tracking**: Manages client lifecycle and state
//!
//! ## Broadcasting Strategy
//! - **Active Periods**: 20Hz (50ms intervals) during graph changes
//! - **Stable Periods**: 1Hz (1s intervals) during settled states
//! - **New Client**: Immediate broadcast regardless of graph state
//! - **Binary Protocol**: 28-byte optimized node data for network efficiency

use actix::prelude::*;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// Import required types and messages
use crate::actors::messages::*;
use crate::handlers::socket_flow_handler::SocketFlowServer;
use crate::telemetry::agent_telemetry::{get_telemetry_logger, CorrelationId, Position3D};
use crate::utils::socket_flow_messages::BinaryNodeDataClient;

///
#[derive(Debug, Clone)]
pub struct ClientState {
    pub client_id: usize,
    pub addr: Addr<SocketFlowServer>,
    pub connected_at: Instant,
    pub last_update: Instant,
    pub position_sent: bool,
    pub initial_sync_completed: bool,
}

///
pub struct ClientManager {
    pub clients: HashMap<usize, ClientState>,
    pub next_id: usize,
    pub total_connections: usize,
    pub active_connections: usize,
}


/// Helper to convert RwLock poison errors to ActorError
fn handle_rwlock_error<T>(result: Result<T, std::sync::PoisonError<T>>) -> Result<T, crate::errors::ActorError> {
    result.map_err(|_| crate::errors::ActorError::RuntimeFailure {
        actor_name: "ClientCoordinatorActor".to_string(),
        reason: "RwLock poisoned - a thread panicked while holding the lock".to_string(),
    })
}

impl ClientManager {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            next_id: 1,
            total_connections: 0,
            active_connections: 0,
        }
    }

    pub fn register_client(&mut self, addr: Addr<SocketFlowServer>) -> usize {
        let client_id = self.next_id;
        self.next_id += 1;

        let now = Instant::now();
        let client_state = ClientState {
            client_id,
            addr,
            connected_at: now,
            last_update: now,
            position_sent: false,
            initial_sync_completed: false,
        };

        self.clients.insert(client_id, client_state);
        self.total_connections += 1;
        self.active_connections = self.clients.len();

        debug!(
            "Client {} registered. Total active: {}",
            client_id, self.active_connections
        );
        client_id
    }

    pub fn unregister_client(&mut self, client_id: usize) -> bool {
        if self.clients.remove(&client_id).is_some() {
            self.active_connections = self.clients.len();
            debug!(
                "Client {} unregistered. Total active: {}",
                client_id, self.active_connections
            );
            true
        } else {
            warn!("Attempted to unregister non-existent client {}", client_id);
            false
        }
    }

    pub fn mark_client_synced(&mut self, client_id: usize) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.initial_sync_completed = true;
            client.last_update = Instant::now();
        }
    }

    pub fn update_client_timestamp(&mut self, client_id: usize) {
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.last_update = Instant::now();
        }
    }

    pub fn broadcast_to_all(&self, data: Vec<u8>) -> usize {
        let mut broadcast_count = 0;
        for (_, client_state) in &self.clients {
            client_state.addr.do_send(SendToClientBinary(data.clone()));
            broadcast_count += 1;
        }
        broadcast_count
    }

    pub fn broadcast_message(&self, message: String) -> usize {
        let mut broadcast_count = 0;
        for (_, client_state) in &self.clients {
            client_state.addr.do_send(SendToClientText(message.clone()));
            broadcast_count += 1;
        }
        broadcast_count
    }

    pub fn get_client_count(&self) -> usize {
        self.clients.len()
    }

    pub fn get_unsynced_clients(&self) -> Vec<usize> {
        self.clients
            .values()
            .filter(|client| !client.initial_sync_completed)
            .map(|client| client.client_id)
            .collect()
    }
}

///
pub struct ClientCoordinatorActor {
    
    client_manager: Arc<RwLock<ClientManager>>,

    
    last_broadcast: Instant,

    
    broadcast_interval: Duration,

    
    active_broadcast_interval: Duration,

    
    stable_broadcast_interval: Duration,

    
    initial_positions_sent: bool,


    graph_service_addr: Option<Addr<crate::actors::GraphServiceSupervisor>>,

    
    position_cache: HashMap<u32, BinaryNodeDataClient>,

    
    broadcast_count: u64,
    bytes_sent: u64,

    
    force_broadcast_requests: u32,

    
    connection_stats: ConnectionStats,

    
    bandwidth_limit_bytes_per_sec: usize, 
    bytes_sent_this_second: usize,
    last_bandwidth_check: Instant,

    
    pending_voice_data: Vec<Vec<u8>>,
    voice_data_queued_bytes: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub total_registrations: usize,
    pub total_unregistrations: usize,
    pub current_clients: usize,
    pub peak_clients: usize,
    pub average_session_duration: Duration,
}

impl ClientCoordinatorActor {
    pub fn new() -> Self {
        Self {
            client_manager: Arc::new(RwLock::new(ClientManager::new())),
            last_broadcast: Instant::now(),
            broadcast_interval: Duration::from_millis(50), 
            active_broadcast_interval: Duration::from_millis(50), 
            stable_broadcast_interval: Duration::from_millis(1000), 
            initial_positions_sent: false,
            graph_service_addr: None,
            position_cache: HashMap::new(),
            broadcast_count: 0,
            bytes_sent: 0,
            force_broadcast_requests: 0,
            connection_stats: ConnectionStats::default(),
            bandwidth_limit_bytes_per_sec: 1_000_000, 
            bytes_sent_this_second: 0,
            last_bandwidth_check: Instant::now(),
            pending_voice_data: Vec::new(),
            voice_data_queued_bytes: 0,
        }
    }

    
    pub fn set_bandwidth_limit(&mut self, bytes_per_sec: usize) {
        self.bandwidth_limit_bytes_per_sec = bytes_per_sec;
        info!("Bandwidth limit set to {} bytes/sec", bytes_per_sec);
    }

    
    fn check_bandwidth_available(&mut self, bytes_needed: usize) -> bool {
        if self.bandwidth_limit_bytes_per_sec == 0 {
            return true; 
        }

        
        if self.last_bandwidth_check.elapsed() >= Duration::from_secs(1) {
            self.bytes_sent_this_second = 0;
            self.last_bandwidth_check = Instant::now();
        }

        
        self.bytes_sent_this_second + bytes_needed <= self.bandwidth_limit_bytes_per_sec
    }

    
    fn record_bytes_sent(&mut self, bytes: usize) {
        self.bytes_sent_this_second += bytes;
        self.bytes_sent += bytes as u64;
    }

    
    pub fn queue_voice_data(&mut self, audio: Vec<u8>) {
        let audio_len = audio.len();
        self.voice_data_queued_bytes += audio_len;
        self.pending_voice_data.push(audio);
        debug!(
            "Queued voice data: {} bytes, total queued: {} bytes",
            audio_len, self.voice_data_queued_bytes
        );
    }

    
    fn send_prioritized_broadcasts(&mut self) -> Result<usize, String> {
        use crate::utils::binary_protocol::BinaryProtocol;

        let mut total_sent = 0;

        
        while !self.pending_voice_data.is_empty() {
            
            let voice_data_len = self.pending_voice_data[0].len();
            let encoded = BinaryProtocol::encode_voice_data(&self.pending_voice_data[0]);

            
            if !self.check_bandwidth_available(encoded.len()) {
                debug!(
                    "Bandwidth limit reached, deferring {} voice messages",
                    self.pending_voice_data.len()
                );
                break;
            }

            
            let client_count = {
                let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
                manager.broadcast_to_all(encoded.clone())
            };

            self.record_bytes_sent(encoded.len());
            total_sent += client_count;

            
            self.voice_data_queued_bytes -= voice_data_len;
            self.pending_voice_data.remove(0);

            debug!(
                "Sent voice data: {} bytes to {} clients",
                encoded.len(),
                client_count
            );
        }

        
        if !self.position_cache.is_empty() && self.should_broadcast() {
            
            let mut position_data = Vec::new();
            for (_, node_data) in &self.position_cache {
                position_data.push(*node_data);
            }


            let binary_data = self.serialize_positions(&position_data);


            if self.check_bandwidth_available(binary_data.len()) {

                let client_count = {
                    let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
                    manager.broadcast_to_all(binary_data.clone())
                };

                self.record_bytes_sent(binary_data.len());
                self.broadcast_count += 1;
                self.last_broadcast = Instant::now();
                total_sent += client_count;

                debug!(
                    "Sent graph update: {} nodes, {} bytes to {} clients",
                    position_data.len(),
                    binary_data.len(),
                    client_count
                );
            } else {
                debug!("Bandwidth limit reached, deferring graph update");
            }
        }

        Ok(total_sent)
    }


    pub fn set_graph_service_addr(
        &mut self,
        addr: Addr<crate::actors::GraphServiceSupervisor>,
    ) {
        self.graph_service_addr = Some(addr);
        debug!("Graph service address set in client coordinator");
    }

    
    pub fn update_broadcast_interval(&mut self, is_stable: bool) {
        let new_interval = if is_stable {
            self.stable_broadcast_interval
        } else {
            self.active_broadcast_interval
        };

        if new_interval != self.broadcast_interval {
            self.broadcast_interval = new_interval;
            debug!(
                "Broadcast interval updated: {}ms (stable: {})",
                new_interval.as_millis(),
                is_stable
            );
        }
    }

    
    pub fn should_broadcast(&self) -> bool {
        self.last_broadcast.elapsed() >= self.broadcast_interval
    }

    
    pub fn force_broadcast(&mut self, reason: &str) -> bool {
        info!("Force broadcasting positions: {}", reason);
        self.force_broadcast_requests += 1;

        let client_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return false;
                }
            };
            manager.get_client_count()
        };

        if client_count == 0 {
            debug!("No clients connected for force broadcast");
            return false;
        }

        
        let mut position_data = Vec::new();
        for (_, node_data) in &self.position_cache {
            position_data.push(*node_data);
        }

        if position_data.is_empty() {
            warn!(
                "Force broadcast requested but no position data available (reason: {})",
                reason
            );
            return false;
        }

        
        let binary_data = self.serialize_positions(&position_data);


        let broadcast_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return false;
                }
            };
            manager.broadcast_to_all(binary_data.clone())
        };


        self.broadcast_count += 1;
        self.bytes_sent += binary_data.len() as u64;
        self.last_broadcast = Instant::now();
        self.initial_positions_sent = true;

        
        if let Some(logger) = get_telemetry_logger() {
            let correlation_id = CorrelationId::new();
            logger.log_event(
                crate::telemetry::agent_telemetry::TelemetryEvent::new(
                    correlation_id,
                    crate::telemetry::agent_telemetry::LogLevel::INFO,
                    "client_coordinator",
                    "force_broadcast",
                    &format!(
                        "Force broadcast: {} nodes to {} clients (reason: {})",
                        position_data.len(),
                        broadcast_count,
                        reason
                    ),
                    "client_coordinator_actor",
                )
                .with_metadata("bytes_sent", serde_json::json!(binary_data.len()))
                .with_metadata("client_count", serde_json::json!(broadcast_count))
                .with_metadata("reason", serde_json::json!(reason)),
            );
        }

        info!(
            "Force broadcast complete: {} nodes sent to {} clients (reason: {})",
            position_data.len(),
            broadcast_count,
            reason
        );
        true
    }

    
    
    fn serialize_positions(&self, positions: &[BinaryNodeDataClient]) -> Vec<u8> {
        
        use crate::utils::binary_protocol::{BinaryProtocol, GraphType};

        
        let nodes: Vec<(String, [f32; 6])> = positions
            .iter()
            .map(|pos| {
                (
                    pos.node_id.to_string(),
                    [pos.x, pos.y, pos.z, pos.vx, pos.vy, pos.vz],
                )
            })
            .collect();

        
        
        BinaryProtocol::encode_graph_update(GraphType::KnowledgeGraph, &nodes)
    }

    
    pub fn update_position_cache(&mut self, positions: Vec<(u32, BinaryNodeDataClient)>) {
        for (node_id, node_data) in positions {
            self.position_cache.insert(node_id, node_data);
        }
        debug!(
            "Position cache updated with {} nodes",
            self.position_cache.len()
        );
    }

    
    pub fn broadcast_positions(&mut self, is_stable: bool) -> Result<usize, String> {
        self.update_broadcast_interval(is_stable);

        let client_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
            manager.get_client_count()
        };

        if client_count == 0 {
            return Ok(0);
        }

        
        let force_broadcast = !self.initial_positions_sent;

        if !force_broadcast && !self.should_broadcast() {
            return Ok(0); 
        }

        
        let mut position_data = Vec::new();
        for (_, node_data) in &self.position_cache {
            position_data.push(*node_data);
        }

        if position_data.is_empty() {
            return Err("No position data available for broadcast".to_string());
        }

        
        let binary_data = self.serialize_positions(&position_data);


        let broadcast_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
            manager.broadcast_to_all(binary_data.clone())
        };


        self.broadcast_count += 1;
        self.bytes_sent += binary_data.len() as u64;
        self.last_broadcast = Instant::now();

        if force_broadcast {
            self.initial_positions_sent = true;
            info!(
                "Sent initial positions to clients ({} nodes to {} clients)",
                position_data.len(),
                broadcast_count
            );
        }

        
        if crate::utils::logging::is_debug_enabled() && !force_broadcast {
            debug!(
                "Broadcast positions: {} nodes to {} clients, stable: {}",
                position_data.len(),
                broadcast_count,
                is_stable
            );
        }

        
        if force_broadcast || position_data.len() > 100 {
            if let Some(logger) = get_telemetry_logger() {
                let correlation_id = CorrelationId::new();
                logger.log_event(
                    crate::telemetry::agent_telemetry::TelemetryEvent::new(
                        correlation_id,
                        crate::telemetry::agent_telemetry::LogLevel::DEBUG,
                        "client_coordinator",
                        "position_broadcast",
                        &format!(
                            "Broadcast: {} nodes to {} clients",
                            position_data.len(),
                            broadcast_count
                        ),
                        "client_coordinator_actor",
                    )
                    .with_metadata("bytes_sent", serde_json::json!(binary_data.len()))
                    .with_metadata("client_count", serde_json::json!(broadcast_count))
                    .with_metadata("is_initial", serde_json::json!(force_broadcast))
                    .with_metadata("is_stable", serde_json::json!(is_stable)),
                );
            }
        }

        Ok(broadcast_count)
    }

    
    fn generate_initial_position(&self, client_id: usize) -> Position3D {
        use rand::prelude::*;

        let mut rng = thread_rng();

        
        let radius = rng.gen_range(50.0..200.0);
        let theta = rng.gen_range(0.0..std::f32::consts::PI * 2.0);
        let phi = rng.gen_range(0.0..std::f32::consts::PI);

        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        let position = Position3D::new(x, y, z);

        info!(
            "Generated position for client {}: ({:.2}, {:.2}, {:.2}), magnitude: {:.2}",
            client_id, position.x, position.y, position.z, position.magnitude
        );

        
        if position.is_origin() {
            warn!(
                "ORIGIN POSITION BUG DETECTED: Client {} generated at origin despite parameters",
                client_id
            );
        }

        position
    }

    
    fn update_connection_stats(&mut self) {
        let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return;
                }
            };
        self.connection_stats.current_clients = manager.get_client_count();

        if self.connection_stats.current_clients > self.connection_stats.peak_clients {
            self.connection_stats.peak_clients = self.connection_stats.current_clients;
        }
    }

    
    pub fn get_stats(&self) -> ClientCoordinatorStats {
        let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return ClientCoordinatorStats {
                        active_clients: 0,
                        total_broadcasts: self.broadcast_count,
                        bytes_sent: self.bytes_sent,
                        force_broadcasts: self.force_broadcast_requests,
                        position_cache_size: self.position_cache.len(),
                        initial_positions_sent: self.initial_positions_sent,
                        current_broadcast_interval: self.broadcast_interval,
                        connection_stats: self.connection_stats.clone(),
                    };
                }
            };
        ClientCoordinatorStats {
            active_clients: manager.get_client_count(),
            total_broadcasts: self.broadcast_count,
            bytes_sent: self.bytes_sent,
            force_broadcasts: self.force_broadcast_requests,
            position_cache_size: self.position_cache.len(),
            initial_positions_sent: self.initial_positions_sent,
            current_broadcast_interval: self.broadcast_interval,
            connection_stats: self.connection_stats.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCoordinatorStats {
    pub active_clients: usize,
    pub total_broadcasts: u64,
    pub bytes_sent: u64,
    pub force_broadcasts: u32,
    pub position_cache_size: usize,
    pub initial_positions_sent: bool,
    pub current_broadcast_interval: Duration,
    pub connection_stats: ConnectionStats,
}

impl Actor for ClientCoordinatorActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("ClientCoordinatorActor started - WebSocket communication manager ready");

        
        if let Some(logger) = get_telemetry_logger() {
            let correlation_id = CorrelationId::new();
            logger.log_event(
                crate::telemetry::agent_telemetry::TelemetryEvent::new(
                    correlation_id,
                    crate::telemetry::agent_telemetry::LogLevel::INFO,
                    "actor_lifecycle",
                    "client_coordinator_start",
                    "Client Coordinator Actor started successfully",
                    "client_coordinator_actor",
                )
                .with_metadata(
                    "broadcast_interval_ms",
                    serde_json::json!(self.broadcast_interval.as_millis()),
                )
                .with_metadata(
                    "stable_interval_ms",
                    serde_json::json!(self.stable_broadcast_interval.as_millis()),
                )
                .with_metadata(
                    "active_interval_ms",
                    serde_json::json!(self.active_broadcast_interval.as_millis()),
                ),
            );
        }
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        let stats = self.get_stats();
        info!(
            "ClientCoordinatorActor stopped - {} clients, {} broadcasts, {} bytes sent",
            stats.active_clients, stats.total_broadcasts, stats.bytes_sent
        );

        
        if let Some(logger) = get_telemetry_logger() {
            let correlation_id = CorrelationId::new();
            logger.log_event(
                crate::telemetry::agent_telemetry::TelemetryEvent::new(
                    correlation_id,
                    crate::telemetry::agent_telemetry::LogLevel::INFO,
                    "actor_lifecycle",
                    "client_coordinator_stop",
                    &format!(
                        "Client Coordinator Actor stopped - processed {} clients",
                        stats.active_clients
                    ),
                    "client_coordinator_actor",
                )
                .with_metadata(
                    "final_stats",
                    serde_json::to_value(&stats).unwrap_or_default(),
                ),
            );
        }
    }
}

// ===== MESSAGE HANDLERS =====

///
impl Handler<RegisterClient> for ClientCoordinatorActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, msg: RegisterClient, _ctx: &mut Self::Context) -> Self::Result {
        let client_id = {
            let mut manager = match handle_rwlock_error(self.client_manager.write()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e).into());
                }
            };
            manager.register_client(msg.addr)
        };

        
        let initial_position = self.generate_initial_position(client_id);

        
        self.connection_stats.total_registrations += 1;
        self.update_connection_stats();

        
        if let Some(logger) = get_telemetry_logger() {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("client_id".to_string(), serde_json::json!(client_id));
            metadata.insert(
                "total_clients".to_string(),
                serde_json::json!(self.connection_stats.current_clients),
            );
            metadata.insert(
                "position_generation_method".to_string(),
                serde_json::json!("random_spherical"),
            );

            logger.log_agent_spawn(
                &format!("client_{}", client_id),
                None, 
                initial_position,
                metadata,
            );
        }

        
        if !self.position_cache.is_empty() {
            self.force_broadcast(&format!("new_client_{}", client_id));
        } else {
            debug!("No position data available for new client {} - broadcast will occur when data is available", client_id);
        }

        info!(
            "Client {} registered successfully. Total clients: {}",
            client_id, self.connection_stats.current_clients
        );
        Ok(client_id)
    }
}

///
impl Handler<UnregisterClient> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UnregisterClient, _ctx: &mut Self::Context) -> Self::Result {
        let success = {
            let mut manager = match handle_rwlock_error(self.client_manager.write()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
            manager.unregister_client(msg.client_id)
        };

        if success {
            
            self.connection_stats.total_unregistrations += 1;
            self.update_connection_stats();

            
            if let Some(logger) = get_telemetry_logger() {
                let correlation_id =
                    CorrelationId::from_agent_id(&format!("client_{}", msg.client_id));
                logger.log_event(
                    crate::telemetry::agent_telemetry::TelemetryEvent::new(
                        correlation_id,
                        crate::telemetry::agent_telemetry::LogLevel::INFO,
                        "client_management",
                        "client_disconnect",
                        &format!("Client {} disconnected", msg.client_id),
                        "client_coordinator_actor",
                    )
                    .with_agent_id(&format!("client_{}", msg.client_id))
                    .with_metadata(
                        "remaining_clients",
                        serde_json::json!(self.connection_stats.current_clients),
                    ),
                );
            }

            info!(
                "Client {} unregistered successfully. Total clients: {}",
                msg.client_id, self.connection_stats.current_clients
            );
            Ok(())
        } else {
            let error_msg = format!("Failed to unregister client {}: not found", msg.client_id);
            error!("{}", error_msg);
            Err(error_msg)
        }
    }
}

///
impl Handler<BroadcastNodePositions> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        let client_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
            manager.broadcast_to_all(msg.positions.clone())
        };

        if client_count > 0 {
            
            self.broadcast_count += 1;
            self.bytes_sent += msg.positions.len() as u64;
            self.last_broadcast = Instant::now();

            debug!(
                "Broadcasted {} bytes to {} clients",
                msg.positions.len(),
                client_count
            );

            
            if msg.positions.len() > 1000 || client_count > 10 {
                info!(
                    "Large broadcast: {} bytes to {} clients",
                    msg.positions.len(),
                    client_count
                );

                if let Some(logger) = get_telemetry_logger() {
                    let correlation_id = CorrelationId::new();
                    logger.log_event(
                        crate::telemetry::agent_telemetry::TelemetryEvent::new(
                            correlation_id,
                            crate::telemetry::agent_telemetry::LogLevel::INFO,
                            "client_coordinator",
                            "large_broadcast",
                            &format!(
                                "Large broadcast: {} bytes to {} clients",
                                msg.positions.len(),
                                client_count
                            ),
                            "client_coordinator_actor",
                        )
                        .with_metadata("bytes_sent", serde_json::json!(msg.positions.len()))
                        .with_metadata("client_count", serde_json::json!(client_count)),
                    );
                }
            }
        }

        Ok(())
    }
}

///
impl Handler<BroadcastMessage> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastMessage, _ctx: &mut Self::Context) -> Self::Result {
        let client_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
            manager.broadcast_message(msg.message.clone())
        };

        if client_count > 0 {
            debug!(
                "Broadcasted message to {} clients: {}",
                client_count,
                if msg.message.len() > 100 {
                    format!("{}...", &msg.message[..100])
                } else {
                    msg.message.clone()
                }
            );
        }

        Ok(())
    }
}

///
impl Handler<GetClientCount> for ClientCoordinatorActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, _msg: GetClientCount, _ctx: &mut Self::Context) -> Self::Result {
        let count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
            manager.get_client_count()
        };
        Ok(count)
    }
}

///
impl Handler<ForcePositionBroadcast> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ForcePositionBroadcast, _ctx: &mut Self::Context) -> Self::Result {
        if self.force_broadcast(&msg.reason) {
            Ok(())
        } else {
            let error_msg = format!("Force broadcast failed: {}", msg.reason);
            warn!("{}", error_msg);
            Err(error_msg)
        }
    }
}

///
impl Handler<InitialClientSync> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitialClientSync, _ctx: &mut Self::Context) -> Self::Result {
        info!(
            "Initial client sync requested by {} from {}",
            msg.client_identifier, msg.trigger_source
        );

        
        let broadcast_reason = format!(
            "initial_sync_{}_{}",
            msg.client_identifier, msg.trigger_source
        );

        if self.force_broadcast(&broadcast_reason) {
            
            if let Ok(client_id) = msg.client_identifier.parse::<usize>() {
                let mut manager = match handle_rwlock_error(self.client_manager.write()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
                manager.mark_client_synced(client_id);
            }

            info!(
                "Initial sync broadcast complete for client {} from {}",
                msg.client_identifier, msg.trigger_source
            );
            Ok(())
        } else {
            let error_msg = format!(
                "Initial sync failed for client {} - no position data available",
                msg.client_identifier
            );
            warn!("{}", error_msg);
            Err(error_msg)
        }
    }
}

///
impl Handler<UpdateNodePositions> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        
        let mut client_positions = Vec::new();
        for (node_id, node_data) in msg.positions {
            let client_data = BinaryNodeDataClient {
                node_id: node_data.node_id,
                x: node_data.x,
                y: node_data.y,
                z: node_data.z,
                vx: node_data.vx,
                vy: node_data.vy,
                vz: node_data.vz,
            };
            client_positions.push((node_id, client_data));
        }

        
        self.update_position_cache(client_positions);

        
        let client_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
            manager.get_client_count()
        };

        if client_count > 0 {

            let unsynced_clients = {
                let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error: {}", e);
                    return Err(format!("Failed to acquire client manager lock: {}", e));
                }
            };
                manager.get_unsynced_clients()
            };

            let force_broadcast = !unsynced_clients.is_empty() || !self.initial_positions_sent;

            if force_broadcast {
                self.force_broadcast("position_update_with_unsynced_clients");
            } else {
                
                self.broadcast_positions(false)?; 
            }
        }

        debug!(
            "Updated position cache with {} nodes for {} clients",
            self.position_cache.len(),
            client_count
        );
        Ok(())
    }
}

///
impl Handler<SetGraphServiceAddress> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: SetGraphServiceAddress, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Setting graph service address in client coordinator");
        self.set_graph_service_addr(msg.addr);
    }
}

///
#[derive(Message)]
#[rtype(result = "Result<ClientCoordinatorStats, String>")]
pub struct GetClientCoordinatorStats;

impl Handler<GetClientCoordinatorStats> for ClientCoordinatorActor {
    type Result = Result<ClientCoordinatorStats, String>;

    fn handle(
        &mut self,
        _msg: GetClientCoordinatorStats,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        Ok(self.get_stats())
    }
}

///
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct QueueVoiceData {
    pub audio: Vec<u8>,
}

impl Handler<QueueVoiceData> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: QueueVoiceData, _ctx: &mut Self::Context) -> Self::Result {
        self.queue_voice_data(msg.audio);

        
        match self.send_prioritized_broadcasts() {
            Ok(count) => {
                debug!("Voice data queued and {} broadcasts sent", count);
                Ok(())
            }
            Err(e) => {
                warn!(
                    "Failed to send prioritized broadcasts after queuing voice: {}",
                    e
                );
                Ok(()) 
            }
        }
    }
}

///
#[derive(Message)]
#[rtype(result = "()")]
pub struct SetBandwidthLimit {
    pub bytes_per_sec: usize,
}

impl Handler<SetBandwidthLimit> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: SetBandwidthLimit, _ctx: &mut Self::Context) -> Self::Result {
        self.set_bandwidth_limit(msg.bytes_per_sec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_manager_registration() {
        let mut manager = ClientManager::new();
        assert_eq!(manager.get_client_count(), 0);

        
        
    }

    #[test]
    fn test_position_serialization() {
        let actor = ClientCoordinatorActor::new();
        let positions = vec![BinaryNodeDataClient {
            node_id: 1,
            x: 1.0,
            y: 2.0,
            z: 3.0,
            vx: 0.1,
            vy: 0.2,
            vz: 0.3,
        }];

        let serialized = actor.serialize_positions(&positions);
        assert_eq!(
            serialized.len(),
            std::mem::size_of::<BinaryNodeDataClient>()
        );
    }

    #[test]
    fn test_broadcast_timing() {
        let mut actor = ClientCoordinatorActor::new();

        
        assert!(actor.should_broadcast());

        
        actor.last_broadcast = Instant::now();

        
        assert!(!actor.should_broadcast());
    }
}
