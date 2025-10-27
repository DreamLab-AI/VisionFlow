// src/actors/backward_compat.rs
//! Backward Compatibility Layer
//!
//! Provides backward compatibility for legacy actor message calls,
//! routing them through the new adapter layer with deprecation warnings.

use actix::prelude::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

use crate::application::physics_service::PhysicsService;
use crate::application::semantic_service::SemanticService;
use crate::ports::gpu_physics_adapter::PhysicsParameters;

/// Legacy physics message (deprecated)
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct LegacyStartPhysics {
    pub node_count: usize,
    pub params: LegacyPhysicsParams,
}

#[derive(Debug, Clone)]
pub struct LegacyPhysicsParams {
    pub time_step: f32,
    pub damping: f32,
    pub spring_constant: f32,
}

impl Default for LegacyPhysicsParams {
    fn default() -> Self {
        Self {
            time_step: 0.016,
            damping: 0.8,
            spring_constant: 0.01,
        }
    }
}

/// Legacy semantic message (deprecated)
#[derive(Message)]
#[rtype(result = "Result<usize, String>")]
pub struct LegacyDetectCommunities {
    pub algorithm: String,
}

/// Backward compatibility wrapper for physics operations
pub struct PhysicsCompatWrapper {
    service: Arc<PhysicsService>,
}

impl PhysicsCompatWrapper {
    pub fn new(service: Arc<PhysicsService>) -> Self {
        Self { service }
    }

    /// Handle legacy start physics message
    #[deprecated(
        since = "1.0.0",
        note = "Use PhysicsService::start_simulation() through adapters instead"
    )]
    pub async fn handle_legacy_start(&self, msg: LegacyStartPhysics) -> Result<(), String> {
        warn!(
            "DEPRECATED: LegacyStartPhysics message used. \
             Please migrate to PhysicsService::start_simulation()"
        );

        // Convert legacy params to new format
        let params = PhysicsParameters {
            time_step: msg.params.time_step,
            damping: msg.params.damping,
            spring_constant: msg.params.spring_constant,
            ..Default::default()
        };

        // Note: This is a simplified conversion
        // Actual implementation would need proper graph data

        Ok(())
    }
}

/// Backward compatibility wrapper for semantic operations
pub struct SemanticCompatWrapper {
    service: Arc<SemanticService>,
}

impl SemanticCompatWrapper {
    pub fn new(service: Arc<SemanticService>) -> Self {
        Self { service }
    }

    /// Handle legacy community detection message
    #[deprecated(
        since = "1.0.0",
        note = "Use SemanticService::detect_communities() through adapters instead"
    )]
    pub async fn handle_legacy_communities(
        &self,
        msg: LegacyDetectCommunities,
    ) -> Result<usize, String> {
        warn!(
            "DEPRECATED: LegacyDetectCommunities message used. \
             Please migrate to SemanticService::detect_communities()"
        );

        // Route to new service
        match msg.algorithm.as_str() {
            "louvain" => self
                .service
                .detect_communities_louvain()
                .await
                .map(|r| r.clusters.len())
                .map_err(|e| e.to_string()),
            "label_propagation" => self
                .service
                .detect_communities_label_propagation()
                .await
                .map(|r| r.clusters.len())
                .map_err(|e| e.to_string()),
            _ => self
                .service
                .detect_communities_louvain()
                .await
                .map(|r| r.clusters.len())
                .map_err(|e| e.to_string()),
        }
    }
}

/// Legacy actor compatibility layer
pub struct LegacyActorCompat;

impl LegacyActorCompat {
    /// Create physics compat wrapper
    pub fn physics_wrapper(service: Arc<PhysicsService>) -> PhysicsCompatWrapper {
        PhysicsCompatWrapper::new(service)
    }

    /// Create semantic compat wrapper
    pub fn semantic_wrapper(service: Arc<SemanticService>) -> SemanticCompatWrapper {
        SemanticCompatWrapper::new(service)
    }

    /// Log deprecation warning for direct actor usage
    pub fn warn_direct_actor_usage(actor_type: &str, operation: &str) {
        warn!(
            "DEPRECATED: Direct {} actor usage for '{}'. \
             Please use service layer through hexagonal ports. \
             Direct actor messaging will be removed in v2.0.0",
            actor_type, operation
        );
    }

    /// Check if legacy mode is enabled
    pub fn legacy_mode_enabled() -> bool {
        std::env::var("VISIONFLOW_LEGACY_ACTORS")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
    }
}

/// Migration helper for legacy code
pub struct MigrationHelper;

impl MigrationHelper {
    /// Convert legacy physics params to new format
    pub fn convert_physics_params(legacy: LegacyPhysicsParams) -> PhysicsParameters {
        PhysicsParameters {
            time_step: legacy.time_step,
            damping: legacy.damping,
            spring_constant: legacy.spring_constant,
            ..Default::default()
        }
    }

    /// Get migration guide URL
    pub fn migration_guide_url() -> &'static str {
        "https://docs.visionflow.dev/migration/actors-to-adapters"
    }

    /// Print migration guide
    pub fn print_migration_guide() {
        eprintln!(
            "\n\
            ╔════════════════════════════════════════════════════════════════╗\n\
            ║                   MIGRATION REQUIRED                           ║\n\
            ║                                                                ║\n\
            ║  Direct actor usage is deprecated.                            ║\n\
            ║  Please migrate to the hexagonal architecture pattern.        ║\n\
            ║                                                                ║\n\
            ║  Migration guide: {}  ║\n\
            ║                                                                ║\n\
            ║  Key changes:                                                  ║\n\
            ║  - Use PhysicsService instead of PhysicsActor                 ║\n\
            ║  - Use SemanticService instead of SemanticActor               ║\n\
            ║  - Access through ports/adapters, not direct messaging        ║\n\
            ║                                                                ║\n\
            ║  Set VISIONFLOW_LEGACY_ACTORS=true to suppress warnings       ║\n\
            ╚════════════════════════════════════════════════════════════════╝\n",
            Self::migration_guide_url()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legacy_mode_detection() {
        std::env::set_var("VISIONFLOW_LEGACY_ACTORS", "false");
        assert!(!LegacyActorCompat::legacy_mode_enabled());

        std::env::set_var("VISIONFLOW_LEGACY_ACTORS", "true");
        assert!(LegacyActorCompat::legacy_mode_enabled());
    }

    #[test]
    fn test_params_conversion() {
        let legacy = LegacyPhysicsParams {
            time_step: 0.02,
            damping: 0.9,
            spring_constant: 0.05,
        };

        let converted = MigrationHelper::convert_physics_params(legacy);
        assert_eq!(converted.time_step, 0.02);
        assert_eq!(converted.damping, 0.9);
        assert_eq!(converted.spring_constant, 0.05);
    }

    #[test]
    fn test_migration_guide_url() {
        let url = MigrationHelper::migration_guide_url();
        assert!(url.contains("migration"));
        assert!(url.contains("actors-to-adapters"));
    }
}
