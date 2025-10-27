// src/application/mod.rs
//! CQRS Application Layer
//!
//! This module implements the application layer using hexser's CQRS patterns.
//! It contains all Directives (write operations) and Queries (read operations)
//! organized by domain.
//!
//! ## Architecture
//!
//! Following hexagonal architecture principles:
//! - **Directives**: Write operations that modify state (commands)
//! - **Queries**: Read operations that retrieve state (queries)
//! - **Handlers**: Process directives and queries using ports (repositories, adapters)
//! - **Services**: High-level orchestration of commands, queries, and events
//!
//! ## Domains
//!
//! - **Settings**: Application configuration and user preferences
//! - **Knowledge Graph**: Main graph structure from local markdown files
//! - **Ontology**: OWL-based ontology graph from GitHub markdown
//! - **Physics**: GPU-accelerated physics simulation

pub mod events;
pub mod graph;
pub mod knowledge_graph;
pub mod ontology;
pub mod physics;
pub mod services;
pub mod settings;

// Re-export settings domain
pub use settings::{
    ClearSettingsCache,
    ClearSettingsCacheHandler,
    DeletePhysicsProfile,
    DeletePhysicsProfileHandler,
    GetPhysicsSettings,
    GetPhysicsSettingsHandler,
    // Queries
    GetSetting,
    GetSettingHandler,
    GetSettingsBatch,
    GetSettingsBatchHandler,
    ListPhysicsProfiles,
    ListPhysicsProfilesHandler,
    LoadAllSettings,
    LoadAllSettingsHandler,
    SaveAllSettings,
    SaveAllSettingsHandler,
    UpdatePhysicsSettings,
    UpdatePhysicsSettingsHandler,
    // Directives
    UpdateSetting,
    UpdateSettingHandler,
    UpdateSettingsBatch,
    UpdateSettingsBatchHandler,
};

// Re-export knowledge graph domain
pub use knowledge_graph::{
    AddEdge,
    AddEdgeHandler,
    // Directives
    AddNode,
    AddNodeHandler,
    BatchUpdatePositions,
    BatchUpdatePositionsHandler,
    GetGraphStatistics,
    GetGraphStatisticsHandler,
    GetNode,
    GetNodeEdges,
    GetNodeEdgesHandler,
    GetNodeHandler,
    GetNodesByMetadataId,
    GetNodesByMetadataIdHandler,
    LoadGraph,
    LoadGraphHandler,
    QueryNodes,
    QueryNodesHandler,
    // Queries
    QueryResult,
    RemoveEdge,
    RemoveEdgeHandler,
    RemoveNode,
    RemoveNodeHandler,
    SaveGraph,
    SaveGraphHandler,
    UpdateEdge,
    UpdateEdgeHandler,
    UpdateNode,
    UpdateNodeHandler,
};

// Re-export ontology domain
pub use ontology::{
    // Directives
    AddAxiom,
    AddAxiomHandler,
    AddOwlClass,
    AddOwlClassHandler,
    AddOwlProperty,
    AddOwlPropertyHandler,
    // Queries
    GetClassAxioms,
    GetClassAxiomsHandler,
    GetInferenceResults,
    GetInferenceResultsHandler,
    GetOntologyMetrics,
    GetOntologyMetricsHandler,
    GetOwlClass,
    GetOwlClassHandler,
    GetOwlProperty,
    GetOwlPropertyHandler,
    ListOwlClasses,
    ListOwlClassesHandler,
    ListOwlProperties,
    ListOwlPropertiesHandler,
    LoadOntologyGraph,
    LoadOntologyGraphHandler,
    QueryOntology,
    QueryOntologyHandler,
    RemoveAxiom,
    RemoveAxiomHandler,
    RemoveOwlClass,
    RemoveOwlClassHandler,
    SaveOntologyGraph,
    SaveOntologyGraphHandler,
    StoreInferenceResults,
    StoreInferenceResultsHandler,
    UpdateOwlClass,
    UpdateOwlClassHandler,
    UpdateOwlProperty,
    UpdateOwlPropertyHandler,
    ValidateOntology,
    ValidateOntologyHandler,
};

// Re-export graph domain
pub use graph::{
    // Queries
    ComputeShortestPaths,
    ComputeShortestPathsHandler,
    GetAutoBalanceNotifications,
    GetAutoBalanceNotificationsHandler,
    GetBotsGraphData,
    GetBotsGraphDataHandler,
    GetConstraints,
    GetConstraintsHandler,
    GetEquilibriumStatus,
    GetEquilibriumStatusHandler,
    GetGraphData,
    GetGraphDataHandler,
    GetNodeMap,
    GetNodeMapHandler,
    GetPhysicsState,
    GetPhysicsStateHandler,
};

// Re-export application services
pub use services::{
    GraphApplicationService, OntologyApplicationService, PhysicsApplicationService,
    SettingsApplicationService,
};

// Phase 5: New hexagonal architecture services
pub mod physics_service;
pub mod semantic_service;

pub use physics_service::PhysicsService;
pub use semantic_service::SemanticService;

// Phase 7: Inference service
pub mod inference_service;

pub use inference_service::{InferenceService, InferenceServiceConfig, InferenceEvent};

// Re-export events
pub use events::DomainEvent;
