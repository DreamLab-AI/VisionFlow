// src/actors/lifecycle.rs
//! Actor Lifecycle Management
//!
//! Manages the lifecycle of Actix actors including startup, shutdown,
//! health monitoring, and supervision strategies.

use actix::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::actors::physics_orchestrator_actor::PhysicsOrchestratorActor;
use crate::actors::semantic_processor_actor::SemanticProcessorActor;

/// Actor system lifecycle manager
pub struct ActorLifecycleManager {
    physics_actor: Option<Addr<PhysicsOrchestratorActor>>,
    semantic_actor: Option<Addr<SemanticProcessorActor>>,
    health_check_interval: Duration,
}

impl Default for ActorLifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ActorLifecycleManager {
    /// Create new actor lifecycle manager
    pub fn new() -> Self {
        Self {
            physics_actor: None,
            semantic_actor: None,
            health_check_interval: Duration::from_secs(30),
        }
    }

    /// Initialize all actors
    pub async fn initialize(&mut self) -> Result<(), ActorLifecycleError> {
        info!("Initializing actor system");

        // Start physics actor
        self.start_physics_actor().await?;

        // Start semantic actor
        self.start_semantic_actor().await?;

        // Start health monitoring
        self.start_health_monitoring();

        info!("Actor system initialized successfully");
        Ok(())
    }

    /// Start physics actor
    async fn start_physics_actor(&mut self) -> Result<(), ActorLifecycleError> {
        info!("Starting PhysicsOrchestratorActor");

        let actor = PhysicsOrchestratorActor::default();
        let addr = actor.start();

        self.physics_actor = Some(addr);
        info!("PhysicsOrchestratorActor started successfully");

        Ok(())
    }

    /// Start semantic actor
    async fn start_semantic_actor(&mut self) -> Result<(), ActorLifecycleError> {
        info!("Starting SemanticProcessorActor");

        let actor = SemanticProcessorActor::default();
        let addr = actor.start();

        self.semantic_actor = Some(addr);
        info!("SemanticProcessorActor started successfully");

        Ok(())
    }

    /// Start health monitoring
    fn start_health_monitoring(&self) {
        let physics_actor = self.physics_actor.clone();
        let semantic_actor = self.semantic_actor.clone();
        let interval = self.health_check_interval;

        actix::spawn(async move {
            let mut timer = actix::clock::interval(interval);

            loop {
                timer.tick().await;

                // Check physics actor health
                if let Some(addr) = &physics_actor {
                    if addr.connected() {
                        info!("PhysicsActor health check: OK");
                    } else {
                        warn!("PhysicsActor health check: DISCONNECTED");
                    }
                }

                // Check semantic actor health
                if let Some(addr) = &semantic_actor {
                    if addr.connected() {
                        info!("SemanticActor health check: OK");
                    } else {
                        warn!("SemanticActor health check: DISCONNECTED");
                    }
                }
            }
        });
    }

    /// Graceful shutdown of all actors
    pub async fn shutdown(&mut self) -> Result<(), ActorLifecycleError> {
        info!("Starting graceful actor shutdown");

        // Stop physics actor
        if let Some(_addr) = self.physics_actor.take() {
            info!("Stopping PhysicsOrchestratorActor");
            // Actor will be dropped and stopped when addr is dropped
        }

        // Stop semantic actor
        if let Some(_addr) = self.semantic_actor.take() {
            info!("Stopping SemanticProcessorActor");
            // Actor will be dropped and stopped when addr is dropped
        }

        // Wait for actors to stop
        tokio::time::sleep(Duration::from_secs(2)).await;

        info!("Actor system shutdown complete");
        Ok(())
    }

    /// Restart physics actor
    pub async fn restart_physics_actor(&mut self) -> Result<(), ActorLifecycleError> {
        warn!("Restarting PhysicsOrchestratorActor");

        // Stop existing actor by dropping it
        if let Some(_addr) = self.physics_actor.take() {
            // Actor will be stopped when addr is dropped
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Start new actor
        self.start_physics_actor().await?;

        info!("PhysicsOrchestratorActor restarted successfully");
        Ok(())
    }

    /// Restart semantic actor
    pub async fn restart_semantic_actor(&mut self) -> Result<(), ActorLifecycleError> {
        warn!("Restarting SemanticProcessorActor");

        // Stop existing actor by dropping it
        if let Some(_addr) = self.semantic_actor.take() {
            // Actor will be stopped when addr is dropped
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Start new actor
        self.start_semantic_actor().await?;

        info!("SemanticProcessorActor restarted successfully");
        Ok(())
    }

    /// Get physics actor address
    pub fn get_physics_actor(&self) -> Option<&Addr<PhysicsOrchestratorActor>> {
        self.physics_actor.as_ref()
    }

    /// Get semantic actor address
    pub fn get_semantic_actor(&self) -> Option<&Addr<SemanticProcessorActor>> {
        self.semantic_actor.as_ref()
    }

    /// Check if all actors are running
    pub fn is_healthy(&self) -> bool {
        self.physics_actor.as_ref().map_or(false, |a| a.connected())
            && self
                .semantic_actor
                .as_ref()
                .map_or(false, |a| a.connected())
    }

    /// Set health check interval
    pub fn set_health_check_interval(&mut self, interval: Duration) {
        self.health_check_interval = interval;
    }
}

/// Actor lifecycle errors
#[derive(Debug, thiserror::Error)]
pub enum ActorLifecycleError {
    #[error("Actor initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Actor not running")]
    ActorNotRunning,

    #[error("Actor communication error: {0}")]
    CommunicationError(String),

    #[error("Shutdown timeout")]
    ShutdownTimeout,
}

/// Supervision strategy for actor failures
pub struct SupervisionStrategy {
    max_restarts: usize,
    restart_window: Duration,
}

impl Default for SupervisionStrategy {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

impl SupervisionStrategy {
    /// Create new supervision strategy
    pub fn new(max_restarts: usize, restart_window: Duration) -> Self {
        Self {
            max_restarts,
            restart_window,
        }
    }

    /// Handle actor failure
    pub async fn handle_failure(
        &self,
        actor_name: &str,
        restart_count: usize,
    ) -> SupervisionDecision {
        if restart_count >= self.max_restarts {
            error!(
                "Actor {} exceeded max restarts ({}), giving up",
                actor_name, self.max_restarts
            );
            SupervisionDecision::Stop
        } else {
            warn!(
                "Actor {} failed, restarting (attempt {}/{})",
                actor_name,
                restart_count + 1,
                self.max_restarts
            );
            SupervisionDecision::Restart
        }
    }
}

/// Supervision decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupervisionDecision {
    Restart,
    Stop,
}

/// Global actor system manager
pub static ACTOR_SYSTEM: once_cell::sync::Lazy<Arc<RwLock<ActorLifecycleManager>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(ActorLifecycleManager::new())));

/// Initialize global actor system
pub async fn initialize_actor_system() -> Result<(), ActorLifecycleError> {
    let mut system = ACTOR_SYSTEM.write().await;
    system.initialize().await
}

/// Shutdown global actor system
pub async fn shutdown_actor_system() -> Result<(), ActorLifecycleError> {
    let mut system = ACTOR_SYSTEM.write().await;
    system.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lifecycle_manager_creation() {
        let manager = ActorLifecycleManager::new();
        assert!(!manager.is_healthy());
    }

    #[tokio::test]
    async fn test_supervision_strategy() {
        let strategy = SupervisionStrategy::default();

        let decision = strategy.handle_failure("test_actor", 0).await;
        assert_eq!(decision, SupervisionDecision::Restart);

        let decision = strategy.handle_failure("test_actor", 3).await;
        assert_eq!(decision, SupervisionDecision::Stop);
    }

    #[test]
    fn test_supervision_strategy_custom() {
        let strategy = SupervisionStrategy::new(5, Duration::from_secs(120));
        assert_eq!(strategy.max_restarts, 5);
        assert_eq!(strategy.restart_window, Duration::from_secs(120));
    }
}
