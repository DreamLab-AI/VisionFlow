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

#[derive(Debug, Clone)]
pub struct ClientState {
    pub client_id: usize,
    pub addr: Addr<SocketFlowServer>,
    pub connected_at: Instant,
    pub last_update: Instant,
    pub position_sent: bool,
    pub initial_sync_completed: bool,
    pub pubkey: Option<String>,
    pub is_power_user: bool,
    pub filter: ClientFilter,
}

/// Per-client filter settings for graph visibility
#[derive(Debug, Clone)]
pub struct ClientFilter {
    pub enabled: bool,
    pub quality_threshold: f64,
    pub authority_threshold: f64,
    pub filter_by_quality: bool,
    pub filter_by_authority: bool,
    pub filter_mode: FilterMode,
    pub max_nodes: Option<usize>,
    pub filtered_node_ids: std::collections::HashSet<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterMode {
    And,
    Or,
}

impl Default for ClientFilter {
    fn default() -> Self {
        Self {
            // DEFAULT FILTER ENABLED: Fresh clients get quality-filtered sparse dataset
            enabled: true,
            quality_threshold: 0.7,
            authority_threshold: 0.5,
            filter_by_quality: true,
            filter_by_authority: false,
            filter_mode: FilterMode::Or,
            max_nodes: Some(10000),
            filtered_node_ids: std::collections::HashSet::new(),
        }
    }
}

impl std::str::FromStr for FilterMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "and" => Ok(FilterMode::And),
            "or" => Ok(FilterMode::Or),
            _ => Err(format!("Invalid filter mode: {}", s)),
        }
    }
}

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
            pubkey: None,
            is_power_user: false,
            filter: ClientFilter::default(),
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

    pub fn get_client_mut(&mut self, client_id: usize) -> Option<&mut ClientState> {
        self.clients.get_mut(&client_id)
    }

    pub fn get_client(&self, client_id: usize) -> Option<&ClientState> {
        self.clients.get(&client_id)
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

    /// Broadcast with per-client filtering, including node type flags in binary protocol
    pub fn broadcast_with_filter(
        &self,
        positions: &[BinaryNodeDataClient],
        node_type_arrays: &crate::actors::messages::NodeTypeArrays,
    ) -> usize {
        let mut broadcast_count = 0;
        for (_, client_state) in &self.clients {
            let filtered_positions = if client_state.filter.enabled {
                positions.iter()
                    .filter(|pos| client_state.filter.filtered_node_ids.contains(&pos.node_id))
                    .copied()
                    .collect::<Vec<_>>()
            } else {
                positions.to_vec()
            };

            if !filtered_positions.is_empty() {
                let binary_data = self.serialize_positions(&filtered_positions, node_type_arrays);
                client_state.addr.do_send(SendToClientBinary(binary_data));
                broadcast_count += 1;
            }
        }
        broadcast_count
    }

    fn serialize_positions(
        &self,
        positions: &[BinaryNodeDataClient],
        nta: &crate::actors::messages::NodeTypeArrays,
    ) -> Vec<u8> {
        use crate::utils::binary_protocol::encode_node_data_extended;
        use crate::utils::socket_flow_messages::BinaryNodeData;
        // Convert to (u32, BinaryNodeData) format for V3 protocol encoding
        let nodes: Vec<(u32, BinaryNodeData)> = positions
            .iter()
            .map(|pos| (pos.node_id, *pos))
            .collect();
        encode_node_data_extended(&nodes, &nta.agent_ids, &nta.knowledge_ids, &nta.ontology_class_ids, &nta.ontology_individual_ids, &nta.ontology_property_ids)
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

pub struct ClientCoordinatorActor {
    
    client_manager: Arc<RwLock<ClientManager>>,

    
    last_broadcast: Instant,

    
    broadcast_interval: Duration,

    
    active_broadcast_interval: Duration,

    
    stable_broadcast_interval: Duration,

    
    initial_positions_sent: bool,


    graph_service_addr: Option<Addr<crate::actors::GraphServiceSupervisor>>,

    /// GPU compute actor address for backpressure acknowledgements
    gpu_compute_addr: Option<Addr<crate::actors::gpu::force_compute_actor::ForceComputeActor>>,

    /// Broadcast sequence counter for acknowledgement correlation
    broadcast_sequence: u64,

    // Neo4j settings repository for loading/saving user filters
    neo4j_settings_repository: Option<Arc<crate::adapters::neo4j_settings_repository::Neo4jSettingsRepository>>,


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

    /// Cached node type arrays from GraphStateActor for binary protocol flags
    node_type_arrays: crate::actors::messages::NodeTypeArrays,
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
            gpu_compute_addr: None,
            broadcast_sequence: 0,
            neo4j_settings_repository: None,
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
            node_type_arrays: crate::actors::messages::NodeTypeArrays::default(),
        }
    }

    /// Set the GPU compute actor address for backpressure acknowledgements
    pub fn set_gpu_compute_addr(&mut self, addr: Addr<crate::actors::gpu::force_compute_actor::ForceComputeActor>) {
        self.gpu_compute_addr = Some(addr);
        info!("GPU compute address configured for ClientCoordinatorActor backpressure acks");
    }


    pub fn set_bandwidth_limit(&mut self, bytes_per_sec: usize) {
        self.bandwidth_limit_bytes_per_sec = bytes_per_sec;
        info!("Bandwidth limit set to {} bytes/sec", bytes_per_sec);
    }

    /// Set the Neo4j settings repository for loading user filters
    pub fn set_neo4j_repository(&mut self, repo: Arc<crate::adapters::neo4j_settings_repository::Neo4jSettingsRepository>) {
        self.neo4j_settings_repository = Some(repo);
        info!("Neo4j settings repository configured for ClientCoordinatorActor");
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
        use crate::utils::binary_protocol::encode_node_data_extended;
        use crate::utils::socket_flow_messages::BinaryNodeData;
        // Convert to (u32, BinaryNodeData) format for V3 protocol encoding
        let nodes: Vec<(u32, BinaryNodeData)> = positions
            .iter()
            .map(|pos| (pos.node_id, *pos))
            .collect();
        let nta = &self.node_type_arrays;
        encode_node_data_extended(&nodes, &nta.agent_ids, &nta.knowledge_ids, &nta.ontology_class_ids, &nta.ontology_individual_ids, &nta.ontology_property_ids)
    }

    /// Update cached node type arrays from GraphStateActor
    pub fn update_node_type_arrays(&mut self, arrays: crate::actors::messages::NodeTypeArrays) {
        info!(
            "Node type arrays updated: knowledge={}, agent={}, owl_class={}, owl_individual={}, owl_property={}",
            arrays.knowledge_ids.len(), arrays.agent_ids.len(), arrays.ontology_class_ids.len(),
            arrays.ontology_individual_ids.len(), arrays.ontology_property_ids.len()
        );
        self.node_type_arrays = arrays;
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

/// Handler for BroadcastPositions - modern position broadcasting with backpressure ack
impl Handler<BroadcastPositions> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: BroadcastPositions, _ctx: &mut Self::Context) -> Self::Result {
        let client_count = {
            let manager = match handle_rwlock_error(self.client_manager.read()) {
                Ok(manager) => manager,
                Err(e) => {
                    error!("RwLock error in BroadcastPositions: {}", e);
                    return;
                }
            };
            manager.broadcast_with_filter(&msg.positions, &self.node_type_arrays)
        };

        if client_count > 0 {
            self.broadcast_count += 1;
            self.broadcast_sequence += 1;
            // Approximate byte size (28 bytes per node in binary format)
            let approx_bytes = msg.positions.len() * 28;
            self.bytes_sent += approx_bytes as u64;
            self.last_broadcast = Instant::now();

            // Send acknowledgement to GPU actor for backpressure flow control
            if let Some(ref gpu_addr) = self.gpu_compute_addr {
                gpu_addr.do_send(PositionBroadcastAck {
                    correlation_id: self.broadcast_sequence,
                    clients_delivered: client_count as u32,
                });
            }

            debug!(
                "Broadcasted {} node positions to {} clients (~{} bytes), seq: {}",
                msg.positions.len(),
                client_count,
                approx_bytes,
                self.broadcast_sequence
            );
        }
    }
}

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

impl Handler<SetGraphServiceAddress> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: SetGraphServiceAddress, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Setting graph service address in client coordinator");
        self.set_graph_service_addr(msg.addr);
    }
}

/// Handler for UpdateNodeTypeArrays - caches node type classification for binary protocol flags
impl Handler<UpdateNodeTypeArrays> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateNodeTypeArrays, _ctx: &mut Self::Context) -> Self::Result {
        self.update_node_type_arrays(msg.arrays);
    }
}

/// Handler for SetGpuComputeAddress - enables backpressure acknowledgements
impl Handler<SetGpuComputeAddress> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: SetGpuComputeAddress, _ctx: &mut Self::Context) -> Self::Result {
        self.set_gpu_compute_addr(msg.addr);
    }
}

/// Handler for ClientBroadcastAck - true end-to-end backpressure flow control
/// Forwards client ACKs to GPU actor to replenish tokens based on actual client receipt
/// This replaces queue-only ACKs with application-level acknowledgements
impl Handler<ClientBroadcastAck> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: ClientBroadcastAck, _ctx: &mut Self::Context) -> Self::Result {
        // Forward the acknowledgement to GPU actor for backpressure token restoration
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            gpu_addr.do_send(PositionBroadcastAck {
                correlation_id: msg.sequence_id,
                clients_delivered: 1, // Each client ACK counts as 1 delivery
            });

            // Log at trace level to avoid spam (every 100th ACK at debug)
            if msg.sequence_id % 100 == 0 {
                debug!(
                    "ClientBroadcastAck: seq={}, nodes={}, client_timestamp={}ms, client_id={:?}",
                    msg.sequence_id, msg.nodes_received, msg.timestamp, msg.client_id
                );
            }
        } else {
            // GPU address not set, log warning once per 1000 ACKs
            if msg.sequence_id % 1000 == 0 {
                warn!("ClientBroadcastAck: GPU compute address not set, cannot forward ACK");
            }
        }
    }
}

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

/// Handle client authentication
impl Handler<AuthenticateClient> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AuthenticateClient, ctx: &mut Self::Context) -> Self::Result {
        let mut manager = match handle_rwlock_error(self.client_manager.write()) {
            Ok(manager) => manager,
            Err(e) => {
                error!("RwLock error: {}", e);
                return Err(format!("Failed to acquire client manager lock: {}", e));
            }
        };

        if let Some(client) = manager.get_client_mut(msg.client_id) {
            client.pubkey = Some(msg.pubkey.clone());
            client.is_power_user = msg.is_power_user;
            info!(
                "Client {} authenticated as pubkey {} (power_user: {})",
                msg.client_id, msg.pubkey, msg.is_power_user
            );

            // Load saved filter from Neo4j if repository is available
            if let Some(neo4j_repo) = self.neo4j_settings_repository.clone() {
                let pubkey_clone = msg.pubkey.clone();
                let client_id = msg.client_id;
                let manager_arc = self.client_manager.clone();
                let graph_addr = self.graph_service_addr.clone();

                ctx.spawn(actix::fut::wrap_future::<_, Self>(async move {
                    match neo4j_repo.get_user_filter(&pubkey_clone).await {
                        Ok(Some(user_filter)) => {
                            info!("✅ Loaded filter from Neo4j for pubkey {}: enabled={}, quality_threshold={}",
                                  pubkey_clone, user_filter.enabled, user_filter.quality_threshold);

                            // Update client filter with loaded settings
                            if let Ok(mut manager) = manager_arc.write() {
                                if let Some(client) = manager.get_client_mut(client_id) {
                                    client.filter.enabled = user_filter.enabled;
                                    client.filter.quality_threshold = user_filter.quality_threshold;
                                    client.filter.authority_threshold = user_filter.authority_threshold;
                                    client.filter.filter_by_quality = user_filter.filter_by_quality;
                                    client.filter.filter_by_authority = user_filter.filter_by_authority;
                                    client.filter.filter_mode = match user_filter.filter_mode.as_str() {
                                        "and" => FilterMode::And,
                                        _ => FilterMode::Or,
                                    };
                                    client.filter.max_nodes = user_filter.max_nodes.map(|n| n as usize);

                                    // Recompute filtered nodes with loaded settings
                                    if client.filter.enabled {
                                        if let Some(graph_addr) = graph_addr {
                                            use crate::actors::messages::GetGraphData;
                                            match graph_addr.send(GetGraphData).await {
                                                Ok(Ok(graph_data)) => {
                                                    crate::actors::client_filter::recompute_filtered_nodes(
                                                        &mut client.filter,
                                                        &graph_data
                                                    );
                                                    info!("Recomputed filter for authenticated client {}: {} nodes visible",
                                                          client_id, client.filter.filtered_node_ids.len());
                                                }
                                                _ => warn!("Failed to fetch graph data for filter recomputation"),
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Ok(None) => {
                            info!("No saved filter found for pubkey {}, using defaults", pubkey_clone);
                        }
                        Err(e) => {
                            error!("Failed to load filter from Neo4j: {}", e);
                        }
                    }
                }).map(|_, _, _| ()));
            } else {
                // Fallback: No Neo4j repo configured, use default filter behavior
                warn!("Neo4j repository not configured, using default filter for client {}", msg.client_id);
            }

            // Original behavior: recompute if filter is enabled (kept for non-Neo4j fallback)
            if client.filter.enabled && self.neo4j_settings_repository.is_none() {
                if let Some(graph_addr) = self.graph_service_addr.clone() {
                    let client_id = msg.client_id;
                    let manager_arc = self.client_manager.clone();

                    ctx.spawn(actix::fut::wrap_future::<_, Self>(async move {
                        use crate::actors::messages::GetGraphData;
                        match graph_addr.send(GetGraphData).await {
                            Ok(Ok(graph_data)) => {
                                if let Ok(mut manager) = manager_arc.write() {
                                    if let Some(client) = manager.get_client_mut(client_id) {
                                        crate::actors::client_filter::recompute_filtered_nodes(
                                            &mut client.filter,
                                            &graph_data
                                        );
                                        info!("Recomputed filter for authenticated client {}: {} nodes visible",
                                              client_id, client.filter.filtered_node_ids.len());
                                    }
                                }
                            }
                            Err(e) => warn!("Failed to fetch graph data for filter recomputation: {}", e),
                            Ok(Err(e)) => warn!("Graph data fetch error: {}", e),
                        }
                    }).map(|_, _, _| ()));
                }
            }

            Ok(())
        } else {
            Err(format!("Client {} not found", msg.client_id))
        }
    }
}

/// Handle filter updates from client
impl Handler<UpdateClientFilter> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateClientFilter, ctx: &mut Self::Context) -> Self::Result {
        let mut manager = match handle_rwlock_error(self.client_manager.write()) {
            Ok(manager) => manager,
            Err(e) => {
                error!("RwLock error: {}", e);
                return Err(format!("Failed to acquire client manager lock: {}", e));
            }
        };

        if let Some(client) = manager.get_client_mut(msg.client_id) {
            let filter_mode = msg.filter_mode.parse::<FilterMode>()
                .map_err(|e| format!("Invalid filter mode: {}", e))?;

            client.filter.enabled = msg.enabled;
            client.filter.quality_threshold = msg.quality_threshold;
            client.filter.authority_threshold = msg.authority_threshold;
            client.filter.filter_by_quality = msg.filter_by_quality;
            client.filter.filter_by_authority = msg.filter_by_authority;
            client.filter.filter_mode = filter_mode;
            client.filter.max_nodes = msg.max_nodes.map(|n| n as usize);

            info!(
                "Updated filter for client {}: enabled={}, quality_threshold={}, max_nodes={:?}",
                msg.client_id, msg.enabled, msg.quality_threshold, msg.max_nodes
            );

            // Recompute filtered nodes with updated criteria and send filtered graph to client
            if let Some(graph_addr) = self.graph_service_addr.clone() {
                let client_id = msg.client_id;
                let manager_arc = self.client_manager.clone();

                ctx.spawn(actix::fut::wrap_future::<_, Self>(async move {
                    use crate::actors::messages::GetGraphData;
                    use crate::utils::socket_flow_messages::{InitialNodeData, InitialEdgeData};

                    match graph_addr.send(GetGraphData).await {
                        Ok(Ok(graph_data)) => {
                            if let Ok(mut manager) = manager_arc.write() {
                                if let Some(client) = manager.get_client_mut(client_id) {
                                    // Recompute which nodes pass the filter
                                    crate::actors::client_filter::recompute_filtered_nodes(
                                        &mut client.filter,
                                        &graph_data
                                    );

                                    let filtered_count = client.filter.filtered_node_ids.len();
                                    info!("Recomputed filter for client {}: {} nodes visible",
                                          client_id, filtered_count);

                                    // Build filtered node data
                                    let filtered_nodes: Vec<InitialNodeData> = graph_data
                                        .nodes
                                        .iter()
                                        .filter(|n| client.filter.filtered_node_ids.contains(&n.id))
                                        .map(|node| InitialNodeData {
                                            id: node.id,
                                            metadata_id: node.metadata_id.clone(),
                                            label: node.label.clone(),
                                            x: node.data.x,
                                            y: node.data.y,
                                            z: node.data.z,
                                            vx: node.data.vx,
                                            vy: node.data.vy,
                                            vz: node.data.vz,
                                            owl_class_iri: node.owl_class_iri.clone(),
                                            node_type: node.node_type.clone(),
                                        })
                                        .collect();

                                    // Build filtered edge data (only edges where both endpoints pass filter)
                                    let filtered_edges: Vec<InitialEdgeData> = graph_data
                                        .edges
                                        .iter()
                                        .filter(|e| {
                                            client.filter.filtered_node_ids.contains(&e.source) &&
                                            client.filter.filtered_node_ids.contains(&e.target)
                                        })
                                        .map(|edge| InitialEdgeData {
                                            id: edge.id.clone(),
                                            source_id: edge.source,
                                            target_id: edge.target,
                                            weight: Some(edge.weight),
                                            edge_type: edge.edge_type.clone(),
                                        })
                                        .collect();

                                    info!("Sending filtered graph to client {}: {} nodes, {} edges",
                                          client_id, filtered_nodes.len(), filtered_edges.len());

                                    // Send filtered graph data to this specific client
                                    client.addr.do_send(SendInitialGraphLoad {
                                        nodes: filtered_nodes,
                                        edges: filtered_edges,
                                    });
                                }
                            }
                        }
                        Err(e) => warn!("Failed to fetch graph data for filter recomputation: {}", e),
                        Ok(Err(e)) => warn!("Graph data fetch error: {}", e),
                    }
                }).map(|_, _, _| ()));
            } else {
                // Clear filtered_node_ids if we can't fetch graph data
                client.filter.filtered_node_ids.clear();
            }

            Ok(())
        } else {
            Err(format!("Client {} not found", msg.client_id))
        }
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
        // V3 protocol: 1 header byte + 48 bytes per node
        assert_eq!(serialized.len(), 1 + 48);
    }

    #[test]
    fn test_broadcast_timing() {
        let mut actor = ClientCoordinatorActor::new();

        // Immediately after creation, last_broadcast is set to now, so should NOT broadcast
        assert!(!actor.should_broadcast());

        // Set last_broadcast to the past (beyond broadcast_interval of 50ms)
        actor.last_broadcast = Instant::now() - Duration::from_millis(100);

        // Now should broadcast since elapsed > broadcast_interval
        assert!(actor.should_broadcast());
    }
}
