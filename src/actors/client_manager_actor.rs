//! Client Manager Actor to replace static APP_CLIENT_MANAGER singleton

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::actors::messages::*;
use crate::handlers::socket_flow_handler::SocketFlowServer;
use crate::telemetry::agent_telemetry::{get_telemetry_logger, CorrelationId, Position3D};
// WsMessage is no longer needed here as we use custom messages
use log::{debug, warn, info};
use serde_json;
use std::collections::HashMap as TelemetryHashMap;

pub struct ClientManagerActor {
    clients: HashMap<usize, Addr<SocketFlowServer>>,
    next_id: AtomicUsize,
    graph_service_addr: Option<Addr<crate::actors::graph_actor::GraphServiceActor>>,
}

impl ClientManagerActor {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            next_id: AtomicUsize::new(1),
            graph_service_addr: None,
        }
    }

    // WEBSOCKET SETTLING FIX: Method to set graph service address after creation
    pub fn set_graph_service_addr(&mut self, addr: Addr<crate::actors::graph_actor::GraphServiceActor>) {
        self.graph_service_addr = Some(addr);
    }

    pub fn register_client(&mut self, addr: Addr<SocketFlowServer>) -> usize {
        let client_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.clients.insert(client_id, addr);

        // Generate initial position for this agent (DEBUG: Check for origin position bug)
        let initial_position = self.generate_initial_position(client_id);

        debug!("Client {} registered. Total clients: {}", client_id, self.clients.len());

        // Enhanced telemetry logging for agent spawn
        if let Some(logger) = get_telemetry_logger() {
            let mut metadata = TelemetryHashMap::new();
            metadata.insert("client_id".to_string(), serde_json::json!(client_id));
            metadata.insert("total_clients".to_string(), serde_json::json!(self.clients.len()));
            metadata.insert("position_generation_method".to_string(), serde_json::json!("random_spherical"));

            logger.log_agent_spawn(
                &format!("client_{}", client_id),
                initial_position.clone(),
                metadata
            );
        }

        // WEBSOCKET SETTLING FIX: Trigger immediate position broadcast for new client
        // This ensures new clients get graph data immediately, even if the graph is settled
        if let Some(ref graph_addr) = self.graph_service_addr {
            debug!("Triggering force broadcast for new client {}", client_id);
            graph_addr.do_send(crate::actors::messages::ForcePositionBroadcast {
                reason: format!("new_client_{}", client_id),
            });

            // Log position broadcast trigger
            if let Some(logger) = get_telemetry_logger() {
                let correlation_id = CorrelationId::from_agent_id(&format!("client_{}", client_id));
                logger.log_event(
                    crate::telemetry::agent_telemetry::TelemetryEvent::new(
                        correlation_id,
                        crate::telemetry::agent_telemetry::LogLevel::DEBUG,
                        "agent_lifecycle",
                        "position_broadcast_trigger",
                        &format!("Triggered position broadcast for new client {}", client_id),
                        "client_manager_actor"
                    ).with_agent_id(&format!("client_{}", client_id))
                     .with_position(initial_position)
                );
            }
        } else {
            warn!("Cannot trigger force broadcast for new client {} - no graph service address", client_id);

            // Log the missing graph service issue
            if let Some(logger) = get_telemetry_logger() {
                let correlation_id = CorrelationId::from_agent_id(&format!("client_{}", client_id));
                logger.log_event(
                    crate::telemetry::agent_telemetry::TelemetryEvent::new(
                        correlation_id,
                        crate::telemetry::agent_telemetry::LogLevel::WARN,
                        "agent_lifecycle",
                        "missing_graph_service",
                        &format!("Cannot trigger broadcast for client {} - graph service not available", client_id),
                        "client_manager_actor"
                    ).with_agent_id(&format!("client_{}", client_id))
                     .with_metadata("issue", serde_json::json!("graph_service_addr_not_set"))
                );
            }
        }

        client_id
    }

    pub fn unregister_client(&mut self, client_id: usize) {
        if self.clients.remove(&client_id).is_some() {
            debug!("Client {} unregistered. Total clients: {}", client_id, self.clients.len());
        } else {
            warn!("Attempted to unregister non-existent client {}", client_id);
        }
    }

    pub fn broadcast_to_all(&self, data: Vec<u8>) {
        if self.clients.is_empty() {
            return;
        }

        debug!("Broadcasting {} bytes to {} clients", data.len(), self.clients.len());
        
        for (_client_id, addr) in &self.clients {
            addr.do_send(SendToClientBinary(data.clone()));
        }
    }

    pub fn broadcast_message(&self, message: String) {
        if self.clients.is_empty() {
            return;
        }

        debug!("Broadcasting message to {} clients", self.clients.len());
        
        for (_client_id, addr) in &self.clients {
            addr.do_send(SendToClientText(message.clone()));
        }
    }

    pub fn get_client_count(&self) -> usize {
        self.clients.len()
    }

    /// Generate initial position for agent (DEBUG: Position generation debugging)
    fn generate_initial_position(&self, client_id: usize) -> Position3D {
        use rand::prelude::*;

        // ORIGIN POSITION BUG DEBUG: Detailed position generation with logging
        let mut rng = thread_rng();

        // Generate random position in spherical coordinates to avoid clustering
        let radius = rng.gen_range(50.0..200.0); // Distance from center
        let theta = rng.gen_range(0.0..std::f32::consts::PI * 2.0); // Azimuthal angle
        let phi = rng.gen_range(0.0..std::f32::consts::PI); // Polar angle

        let x = radius * phi.sin() * theta.cos();
        let y = radius * phi.sin() * theta.sin();
        let z = radius * phi.cos();

        let position = Position3D::new(x, y, z);

        // DEBUG: Log position generation details
        info!("Generated position for client {}: ({:.2}, {:.2}, {:.2}), magnitude: {:.2}, radius: {:.2}",
              client_id, position.x, position.y, position.z, position.magnitude, radius);

        // Check for potential origin bug
        if position.is_origin() {
            warn!("ORIGIN POSITION BUG DETECTED: Client {} generated at origin despite non-zero parameters (r={:.2}, θ={:.2}, φ={:.2})",
                  client_id, radius, theta, phi);
        }

        position
    }
}

impl Actor for ClientManagerActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        debug!("ClientManagerActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        debug!("ClientManagerActor stopped with {} clients", self.clients.len());
    }
}

impl Handler<RegisterClient> for ClientManagerActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, msg: RegisterClient, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.register_client(msg.addr))
    }
}

impl Handler<UnregisterClient> for ClientManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UnregisterClient, _ctx: &mut Self::Context) -> Self::Result {
        self.unregister_client(msg.client_id);
        Ok(())
    }
}

impl Handler<BroadcastNodePositions> for ClientManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        self.broadcast_to_all(msg.positions);
        Ok(())
    }
}

impl Handler<BroadcastMessage> for ClientManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.broadcast_message(msg.message);
        Ok(())
    }
}

impl Handler<GetClientCount> for ClientManagerActor {
    type Result = Result<usize, String>;

    fn handle(&mut self, _msg: GetClientCount, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_client_count())
    }
}

// WEBSOCKET SETTLING FIX: Handler to set graph service address
impl Handler<crate::actors::messages::SetGraphServiceAddress> for ClientManagerActor {
    type Result = ();

    fn handle(&mut self, msg: crate::actors::messages::SetGraphServiceAddress, _ctx: &mut Self::Context) -> Self::Result {
        debug!("Setting graph service address in client manager");
        self.graph_service_addr = Some(msg.addr);
    }
}