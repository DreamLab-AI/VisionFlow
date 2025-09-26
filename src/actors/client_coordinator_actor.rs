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
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use log::{debug, info, warn, error};
use serde_json;
use serde::{Serialize, Deserialize};

// Import required types and messages
use crate::actors::messages::*;
use crate::handlers::socket_flow_handler::SocketFlowServer;
use crate::utils::socket_flow_messages::BinaryNodeDataClient;
use crate::telemetry::agent_telemetry::{get_telemetry_logger, CorrelationId, Position3D};

/// Client state information for tracking and management
#[derive(Debug, Clone)]
pub struct ClientState {
    pub client_id: usize,
    pub addr: Addr<SocketFlowServer>,
    pub connected_at: Instant,
    pub last_update: Instant,
    pub position_sent: bool,
    pub initial_sync_completed: bool,
}

/// Client Manager wrapper for thread-safe access
pub struct ClientManager {
    pub clients: HashMap<usize, ClientState>,
    pub next_id: usize,
    pub total_connections: usize,
    pub active_connections: usize,
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

        debug!("Client {} registered. Total active: {}", client_id, self.active_connections);
        client_id
    }

    pub fn unregister_client(&mut self, client_id: usize) -> bool {
        if self.clients.remove(&client_id).is_some() {
            self.active_connections = self.clients.len();
            debug!("Client {} unregistered. Total active: {}", client_id, self.active_connections);
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

/// Client Coordinator Actor - Manages WebSocket client communications
pub struct ClientCoordinatorActor {
    /// Thread-safe client manager
    client_manager: Arc<RwLock<ClientManager>>,

    /// Last broadcast timestamp for timing control
    last_broadcast: Instant,

    /// Broadcast interval configuration
    broadcast_interval: Duration,

    /// Active broadcast interval (high frequency)
    active_broadcast_interval: Duration,

    /// Stable broadcast interval (low frequency)
    stable_broadcast_interval: Duration,

    /// Track if initial positions have been sent
    initial_positions_sent: bool,

    /// Graph service actor address for coordination
    graph_service_addr: Option<Addr<crate::actors::graph_actor::GraphServiceActor>>,

    /// Position cache for efficient broadcasting
    position_cache: HashMap<u32, BinaryNodeDataClient>,

    /// Broadcasting statistics
    broadcast_count: u64,
    bytes_sent: u64,

    /// Force broadcast tracking
    force_broadcast_requests: u32,

    /// Client connection statistics
    connection_stats: ConnectionStats,
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
            broadcast_interval: Duration::from_millis(50), // Default to active rate
            active_broadcast_interval: Duration::from_millis(50), // 20Hz for active periods
            stable_broadcast_interval: Duration::from_millis(1000), // 1Hz for stable periods
            initial_positions_sent: false,
            graph_service_addr: None,
            position_cache: HashMap::new(),
            broadcast_count: 0,
            bytes_sent: 0,
            force_broadcast_requests: 0,
            connection_stats: ConnectionStats::default(),
        }
    }

    /// Set the graph service address for coordination
    pub fn set_graph_service_addr(&mut self, addr: Addr<crate::actors::graph_actor::GraphServiceActor>) {
        self.graph_service_addr = Some(addr);
        debug!("Graph service address set in client coordinator");
    }

    /// Update broadcast interval based on graph stability
    pub fn update_broadcast_interval(&mut self, is_stable: bool) {
        let new_interval = if is_stable {
            self.stable_broadcast_interval
        } else {
            self.active_broadcast_interval
        };

        if new_interval != self.broadcast_interval {
            self.broadcast_interval = new_interval;
            debug!("Broadcast interval updated: {}ms (stable: {})",
                  new_interval.as_millis(), is_stable);
        }
    }

    /// Check if it's time for a scheduled broadcast
    pub fn should_broadcast(&self) -> bool {
        self.last_broadcast.elapsed() >= self.broadcast_interval
    }

    /// Force immediate broadcast regardless of timing
    pub fn force_broadcast(&mut self, reason: &str) -> bool {
        info!("Force broadcasting positions: {}", reason);
        self.force_broadcast_requests += 1;

        let client_count = {
            let manager = self.client_manager.read().unwrap();
            manager.get_client_count()
        };

        if client_count == 0 {
            debug!("No clients connected for force broadcast");
            return false;
        }

        // Create position data from cache
        let mut position_data = Vec::new();
        for (_, node_data) in &self.position_cache {
            position_data.push(*node_data);
        }

        if position_data.is_empty() {
            warn!("Force broadcast requested but no position data available (reason: {})", reason);
            return false;
        }

        // Serialize to binary format
        let binary_data = self.serialize_positions(&position_data);

        // Broadcast to all clients
        let broadcast_count = {
            let manager = self.client_manager.read().unwrap();
            manager.broadcast_to_all(binary_data.clone())
        };

        // Update statistics
        self.broadcast_count += 1;
        self.bytes_sent += binary_data.len() as u64;
        self.last_broadcast = Instant::now();
        self.initial_positions_sent = true;

        // Log telemetry
        if let Some(logger) = get_telemetry_logger() {
            let correlation_id = CorrelationId::new();
            logger.log_event(
                crate::telemetry::agent_telemetry::TelemetryEvent::new(
                    correlation_id,
                    crate::telemetry::agent_telemetry::LogLevel::INFO,
                    "client_coordinator",
                    "force_broadcast",
                    &format!("Force broadcast: {} nodes to {} clients (reason: {})",
                            position_data.len(), broadcast_count, reason),
                    "client_coordinator_actor"
                ).with_metadata("bytes_sent", serde_json::json!(binary_data.len()))
                 .with_metadata("client_count", serde_json::json!(broadcast_count))
                 .with_metadata("reason", serde_json::json!(reason))
            );
        }

        info!("Force broadcast complete: {} nodes sent to {} clients (reason: {})",
              position_data.len(), broadcast_count, reason);
        true
    }

    /// Serialize position data to binary WebSocket format
    fn serialize_positions(&self, positions: &[BinaryNodeDataClient]) -> Vec<u8> {
        let mut binary_data = Vec::with_capacity(positions.len() * std::mem::size_of::<BinaryNodeDataClient>());

        for position in positions {
            let bytes = bytemuck::bytes_of(position);
            binary_data.extend_from_slice(bytes);
        }

        binary_data
    }

    /// Update position cache with new node data
    pub fn update_position_cache(&mut self, positions: Vec<(u32, BinaryNodeDataClient)>) {
        for (node_id, node_data) in positions {
            self.position_cache.insert(node_id, node_data);
        }
        debug!("Position cache updated with {} nodes", self.position_cache.len());
    }

    /// Broadcast current positions to all clients
    pub fn broadcast_positions(&mut self, is_stable: bool) -> Result<usize, String> {
        self.update_broadcast_interval(is_stable);

        let client_count = {
            let manager = self.client_manager.read().unwrap();
            manager.get_client_count()
        };

        if client_count == 0 {
            return Ok(0);
        }

        // Check if we should broadcast based on timing
        let force_broadcast = !self.initial_positions_sent;

        if !force_broadcast && !self.should_broadcast() {
            return Ok(0); // Skip broadcast due to timing
        }

        // Create position data from cache
        let mut position_data = Vec::new();
        for (_, node_data) in &self.position_cache {
            position_data.push(*node_data);
        }

        if position_data.is_empty() {
            return Err("No position data available for broadcast".to_string());
        }

        // Serialize to binary format
        let binary_data = self.serialize_positions(&position_data);

        // Broadcast to all clients
        let broadcast_count = {
            let manager = self.client_manager.read().unwrap();
            manager.broadcast_to_all(binary_data.clone())
        };

        // Update statistics
        self.broadcast_count += 1;
        self.bytes_sent += binary_data.len() as u64;
        self.last_broadcast = Instant::now();

        if force_broadcast {
            self.initial_positions_sent = true;
            info!("Sent initial positions to clients ({} nodes to {} clients)",
                  position_data.len(), broadcast_count);
        }

        // Log debug information
        if crate::utils::logging::is_debug_enabled() && !force_broadcast {
            debug!("Broadcast positions: {} nodes to {} clients, stable: {}",
                   position_data.len(), broadcast_count, is_stable);
        }

        // Log telemetry for significant broadcasts
        if force_broadcast || position_data.len() > 100 {
            if let Some(logger) = get_telemetry_logger() {
                let correlation_id = CorrelationId::new();
                logger.log_event(
                    crate::telemetry::agent_telemetry::TelemetryEvent::new(
                        correlation_id,
                        crate::telemetry::agent_telemetry::LogLevel::DEBUG,
                        "client_coordinator",
                        "position_broadcast",
                        &format!("Broadcast: {} nodes to {} clients", position_data.len(), broadcast_count),
                        "client_coordinator_actor"
                    ).with_metadata("bytes_sent", serde_json::json!(binary_data.len()))
                     .with_metadata("client_count", serde_json::json!(broadcast_count))
                     .with_metadata("is_initial", serde_json::json!(force_broadcast))
                     .with_metadata("is_stable", serde_json::json!(is_stable))
                );
            }
        }

        Ok(broadcast_count)
    }

    /// Generate initial position for a new client
    fn generate_initial_position(&self, client_id: usize) -> Position3D {
        use rand::prelude::*;

        let mut rng = thread_rng();

        // Generate random position in spherical coordinates to avoid clustering
        let radius = rng.gen_range(50.0..200.0);
        let theta = rng.gen_range(0.0..std::f32::consts::PI * 2.0);
        let phi = rng.gen_range(0.0..std::f32::consts::PI);

        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        let position = Position3D::new(x, y, z);

        info!("Generated position for client {}: ({:.2}, {:.2}, {:.2}), magnitude: {:.2}",
              client_id, position.x, position.y, position.z, position.magnitude);

        // Check for origin position bug
        if position.is_origin() {
            warn!("ORIGIN POSITION BUG DETECTED: Client {} generated at origin despite parameters", client_id);
        }

        position
    }

    /// Update connection statistics
    fn update_connection_stats(&mut self) {
        let manager = self.client_manager.read().unwrap();
        self.connection_stats.current_clients = manager.get_client_count();

        if self.connection_stats.current_clients > self.connection_stats.peak_clients {
            self.connection_stats.peak_clients = self.connection_stats.current_clients;
        }
    }

    /// Get comprehensive client coordinator statistics
    pub fn get_stats(&self) -> ClientCoordinatorStats {
        let manager = self.client_manager.read().unwrap();
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

        // Log startup telemetry
        if let Some(logger) = get_telemetry_logger() {
            let correlation_id = CorrelationId::new();
            logger.log_event(
                crate::telemetry::agent_telemetry::TelemetryEvent::new(
                    correlation_id,
                    crate::telemetry::agent_telemetry::LogLevel::INFO,
                    "actor_lifecycle",
                    "client_coordinator_start",
                    "Client Coordinator Actor started successfully",
                    "client_coordinator_actor"
                ).with_metadata("broadcast_interval_ms", serde_json::json!(self.broadcast_interval.as_millis()))
                 .with_metadata("stable_interval_ms", serde_json::json!(self.stable_broadcast_interval.as_millis()))
                 .with_metadata("active_interval_ms", serde_json::json!(self.active_broadcast_interval.as_millis()))
            );
        }
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        let stats = self.get_stats();
        info!("ClientCoordinatorActor stopped - {} clients, {} broadcasts, {} bytes sent",
              stats.active_clients, stats.total_broadcasts, stats.bytes_sent);

        // Log shutdown telemetry with final statistics
        if let Some(logger) = get_telemetry_logger() {
            let correlation_id = CorrelationId::new();
            logger.log_event(
                crate::telemetry::agent_telemetry::TelemetryEvent::new(
                    correlation_id,
                    crate::telemetry::agent_telemetry::LogLevel::INFO,
                    "actor_lifecycle",
                    "client_coordinator_stop",
                    &format!("Client Coordinator Actor stopped - processed {} clients", stats.active_clients),
                    "client_coordinator_actor"
                ).with_metadata("final_stats", serde_json::to_value(&stats).unwrap_or_default())
            );
        }
    }
}

// ===== MESSAGE HANDLERS =====

/// Handle client registration
impl Handler<RegisterClient> for ClientCoordinatorActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, msg: RegisterClient, _ctx: &mut Self::Context) -> Self::Result {
        let client_id = {
            let mut manager = self.client_manager.write().unwrap();
            manager.register_client(msg.addr)
        };

        // Generate initial position for telemetry
        let initial_position = self.generate_initial_position(client_id);

        // Update connection statistics
        self.connection_stats.total_registrations += 1;
        self.update_connection_stats();

        // Log enhanced telemetry for client registration
        if let Some(logger) = get_telemetry_logger() {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("client_id".to_string(), serde_json::json!(client_id));
            metadata.insert("total_clients".to_string(), serde_json::json!(self.connection_stats.current_clients));
            metadata.insert("position_generation_method".to_string(), serde_json::json!("random_spherical"));

            logger.log_agent_spawn(
                &format!("client_{}", client_id),
                initial_position,
                metadata
            );
        }

        // Trigger force broadcast for new client if we have position data
        if !self.position_cache.is_empty() {
            self.force_broadcast(&format!("new_client_{}", client_id));
        } else {
            debug!("No position data available for new client {} - broadcast will occur when data is available", client_id);
        }

        info!("Client {} registered successfully. Total clients: {}", client_id, self.connection_stats.current_clients);
        Ok(client_id)
    }
}

/// Handle client unregistration
impl Handler<UnregisterClient> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UnregisterClient, _ctx: &mut Self::Context) -> Self::Result {
        let success = {
            let mut manager = self.client_manager.write().unwrap();
            manager.unregister_client(msg.client_id)
        };

        if success {
            // Update connection statistics
            self.connection_stats.total_unregistrations += 1;
            self.update_connection_stats();

            // Log telemetry for client disconnection
            if let Some(logger) = get_telemetry_logger() {
                let correlation_id = CorrelationId::from_agent_id(&format!("client_{}", msg.client_id));
                logger.log_event(
                    crate::telemetry::agent_telemetry::TelemetryEvent::new(
                        correlation_id,
                        crate::telemetry::agent_telemetry::LogLevel::INFO,
                        "client_management",
                        "client_disconnect",
                        &format!("Client {} disconnected", msg.client_id),
                        "client_coordinator_actor"
                    ).with_agent_id(&format!("client_{}", msg.client_id))
                     .with_metadata("remaining_clients", serde_json::json!(self.connection_stats.current_clients))
                );
            }

            info!("Client {} unregistered successfully. Total clients: {}", msg.client_id, self.connection_stats.current_clients);
            Ok(())
        } else {
            let error_msg = format!("Failed to unregister client {}: not found", msg.client_id);
            error!("{}", error_msg);
            Err(error_msg)
        }
    }
}

/// Handle broadcasting node positions to all clients
impl Handler<BroadcastNodePositions> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        let client_count = {
            let manager = self.client_manager.read().unwrap();
            manager.broadcast_to_all(msg.positions.clone())
        };

        if client_count > 0 {
            // Update statistics
            self.broadcast_count += 1;
            self.bytes_sent += msg.positions.len() as u64;
            self.last_broadcast = Instant::now();

            debug!("Broadcasted {} bytes to {} clients", msg.positions.len(), client_count);

            // Log significant broadcasts
            if msg.positions.len() > 1000 || client_count > 10 {
                info!("Large broadcast: {} bytes to {} clients", msg.positions.len(), client_count);

                if let Some(logger) = get_telemetry_logger() {
                    let correlation_id = CorrelationId::new();
                    logger.log_event(
                        crate::telemetry::agent_telemetry::TelemetryEvent::new(
                            correlation_id,
                            crate::telemetry::agent_telemetry::LogLevel::INFO,
                            "client_coordinator",
                            "large_broadcast",
                            &format!("Large broadcast: {} bytes to {} clients", msg.positions.len(), client_count),
                            "client_coordinator_actor"
                        ).with_metadata("bytes_sent", serde_json::json!(msg.positions.len()))
                         .with_metadata("client_count", serde_json::json!(client_count))
                    );
                }
            }
        }

        Ok(())
    }
}

/// Handle broadcasting text messages to all clients
impl Handler<BroadcastMessage> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastMessage, _ctx: &mut Self::Context) -> Self::Result {
        let client_count = {
            let manager = self.client_manager.read().unwrap();
            manager.broadcast_message(msg.message.clone())
        };

        if client_count > 0 {
            debug!("Broadcasted message to {} clients: {}", client_count,
                  if msg.message.len() > 100 {
                      format!("{}...", &msg.message[..100])
                  } else {
                      msg.message.clone()
                  });
        }

        Ok(())
    }
}

/// Handle getting current client count
impl Handler<GetClientCount> for ClientCoordinatorActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, _msg: GetClientCount, _ctx: &mut Self::Context) -> Self::Result {
        let count = {
            let manager = self.client_manager.read().unwrap();
            manager.get_client_count()
        };
        Ok(count)
    }
}

/// Handle force position broadcast requests
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

/// Handle initial client synchronization
impl Handler<InitialClientSync> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitialClientSync, _ctx: &mut Self::Context) -> Self::Result {
        info!("Initial client sync requested by {} from {}", msg.client_identifier, msg.trigger_source);

        // Force broadcast current positions to ensure client synchronization
        let broadcast_reason = format!("initial_sync_{}_{}", msg.client_identifier, msg.trigger_source);

        if self.force_broadcast(&broadcast_reason) {
            // Mark client as synced if we can identify them
            if let Ok(client_id) = msg.client_identifier.parse::<usize>() {
                let mut manager = self.client_manager.write().unwrap();
                manager.mark_client_synced(client_id);
            }

            info!("Initial sync broadcast complete for client {} from {}",
                  msg.client_identifier, msg.trigger_source);
            Ok(())
        } else {
            let error_msg = format!("Initial sync failed for client {} - no position data available", msg.client_identifier);
            warn!("{}", error_msg);
            Err(error_msg)
        }
    }
}

/// Handle updating node positions (from graph actor)
impl Handler<UpdateNodePositions> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        // Convert BinaryNodeData to BinaryNodeDataClient and update cache
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

        // Update position cache
        self.update_position_cache(client_positions);

        // Trigger broadcast if we have clients waiting
        let client_count = {
            let manager = self.client_manager.read().unwrap();
            manager.get_client_count()
        };

        if client_count > 0 {
            // Check if there are unsynced clients who need immediate updates
            let unsynced_clients = {
                let manager = self.client_manager.read().unwrap();
                manager.get_unsynced_clients()
            };

            let force_broadcast = !unsynced_clients.is_empty() || !self.initial_positions_sent;

            if force_broadcast {
                self.force_broadcast("position_update_with_unsynced_clients");
            } else {
                // Normal adaptive broadcasting
                self.broadcast_positions(false)?; // Assume active state for position updates
            }
        }

        debug!("Updated position cache with {} nodes for {} clients", self.position_cache.len(), client_count);
        Ok(())
    }
}

/// Handle setting graph service address for coordination
impl Handler<SetGraphServiceAddress> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: SetGraphServiceAddress, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Setting graph service address in client coordinator");
        self.set_graph_service_addr(msg.addr);
    }
}

/// Custom message to get client coordinator statistics
#[derive(Message)]
#[rtype(result = "Result<ClientCoordinatorStats, String>")]
pub struct GetClientCoordinatorStats;

impl Handler<GetClientCoordinatorStats> for ClientCoordinatorActor {
    type Result = Result<ClientCoordinatorStats, String>;

    fn handle(&mut self, _msg: GetClientCoordinatorStats, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_manager_registration() {
        let mut manager = ClientManager::new();
        assert_eq!(manager.get_client_count(), 0);

        // Note: Can't easily test with real Addr in unit tests
        // This would require integration tests with actual actor system
    }

    #[test]
    fn test_position_serialization() {
        let actor = ClientCoordinatorActor::new();
        let positions = vec![
            BinaryNodeDataClient {
                node_id: 1,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }
        ];

        let serialized = actor.serialize_positions(&positions);
        assert_eq!(serialized.len(), std::mem::size_of::<BinaryNodeDataClient>());
    }

    #[test]
    fn test_broadcast_timing() {
        let mut actor = ClientCoordinatorActor::new();

        // Should broadcast initially
        assert!(actor.should_broadcast());

        // Update last broadcast time
        actor.last_broadcast = Instant::now();

        // Should not broadcast immediately after
        assert!(!actor.should_broadcast());
    }
}