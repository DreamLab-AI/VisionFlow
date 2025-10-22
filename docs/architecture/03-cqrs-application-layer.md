# CQRS Application Layer Design

## Overview

This document defines the CQRS (Command Query Responsibility Segregation) application layer using hexser's `Directive` (write) and `Query` (read) patterns. This layer orchestrates business logic between HTTP handlers and the ports/adapters.

## Core Concepts

- **Directives** (Commands): Operations that modify state (create, update, delete)
- **Queries**: Operations that read state without modification
- **Handlers**: Process directives and queries using ports (repositories, adapters)
- **Events**: Optional - emitted after successful directive execution for event sourcing

## Settings Domain

### Directives (Write Operations)

```rust
// src/application/settings/directives.rs

use hexser::{Directive, DirectiveHandler};
use std::collections::HashMap;
use crate::ports::settings_repository::{SettingsRepository, SettingValue};
use crate::config::{AppFullSettings, PhysicsSettings};

// ============================================================================
// UPDATE SETTING
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct UpdateSetting {
    pub key: String,
    pub value: SettingValue,
    pub description: Option<String>,
}

pub struct UpdateSettingHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> UpdateSettingHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> DirectiveHandler<UpdateSetting> for UpdateSettingHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: UpdateSetting) -> Result<Self::Output, Self::Error> {
        log::info!("Executing UpdateSetting directive: key={}", directive.key);

        self.repository
            .set_setting(&directive.key, directive.value, directive.description.as_deref())
            .await?;

        log::info!("Setting '{}' updated successfully", directive.key);
        Ok(())
    }
}

// ============================================================================
// UPDATE SETTINGS BATCH
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct UpdateSettingsBatch {
    pub updates: HashMap<String, SettingValue>,
}

pub struct UpdateSettingsBatchHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> UpdateSettingsBatchHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> DirectiveHandler<UpdateSettingsBatch> for UpdateSettingsBatchHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: UpdateSettingsBatch) -> Result<Self::Output, Self::Error> {
        log::info!("Executing UpdateSettingsBatch directive: {} updates", directive.updates.len());

        self.repository
            .set_settings_batch(directive.updates)
            .await?;

        log::info!("Settings batch updated successfully");
        Ok(())
    }
}

// ============================================================================
// SAVE ALL SETTINGS
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct SaveAllSettings {
    pub settings: AppFullSettings,
}

pub struct SaveAllSettingsHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> SaveAllSettingsHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> DirectiveHandler<SaveAllSettings> for SaveAllSettingsHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: SaveAllSettings) -> Result<Self::Output, Self::Error> {
        log::info!("Executing SaveAllSettings directive");

        self.repository
            .save_all_settings(&directive.settings)
            .await?;

        log::info!("All settings saved successfully");
        Ok(())
    }
}

// ============================================================================
// UPDATE PHYSICS SETTINGS
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct UpdatePhysicsSettings {
    pub profile_name: String,
    pub settings: PhysicsSettings,
}

pub struct UpdatePhysicsSettingsHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> UpdatePhysicsSettingsHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> DirectiveHandler<UpdatePhysicsSettings> for UpdatePhysicsSettingsHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: UpdatePhysicsSettings) -> Result<Self::Output, Self::Error> {
        log::info!("Executing UpdatePhysicsSettings directive: profile={}", directive.profile_name);

        self.repository
            .save_physics_settings(&directive.profile_name, &directive.settings)
            .await?;

        log::info!("Physics settings for profile '{}' updated successfully", directive.profile_name);
        Ok(())
    }
}

// ============================================================================
// DELETE PHYSICS PROFILE
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct DeletePhysicsProfile {
    pub profile_name: String,
}

pub struct DeletePhysicsProfileHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> DeletePhysicsProfileHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> DirectiveHandler<DeletePhysicsProfile> for DeletePhysicsProfileHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: DeletePhysicsProfile) -> Result<Self::Output, Self::Error> {
        log::info!("Executing DeletePhysicsProfile directive: profile={}", directive.profile_name);

        self.repository
            .delete_physics_profile(&directive.profile_name)
            .await?;

        log::info!("Physics profile '{}' deleted successfully", directive.profile_name);
        Ok(())
    }
}

// ============================================================================
// CLEAR SETTINGS CACHE
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct ClearSettingsCache;

pub struct ClearSettingsCacheHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> ClearSettingsCacheHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> DirectiveHandler<ClearSettingsCache> for ClearSettingsCacheHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, _directive: ClearSettingsCache) -> Result<Self::Output, Self::Error> {
        log::info!("Executing ClearSettingsCache directive");

        self.repository.clear_cache().await?;

        log::info!("Settings cache cleared successfully");
        Ok(())
    }
}
```

### Queries (Read Operations)

```rust
// src/application/settings/queries.rs

use hexser::{Query, QueryHandler};
use std::collections::HashMap;
use crate::ports::settings_repository::{SettingsRepository, SettingValue};
use crate::config::{AppFullSettings, PhysicsSettings};

// ============================================================================
// GET SETTING
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct GetSetting {
    pub key: String,
}

pub struct GetSettingHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> GetSettingHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> QueryHandler<GetSetting> for GetSettingHandler<R> {
    type Output = Option<SettingValue>;
    type Error = String;

    async fn handle(&self, query: GetSetting) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing GetSetting query: key={}", query.key);

        self.repository.get_setting(&query.key).await
    }
}

// ============================================================================
// GET SETTINGS BATCH
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct GetSettingsBatch {
    pub keys: Vec<String>,
}

pub struct GetSettingsBatchHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> GetSettingsBatchHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> QueryHandler<GetSettingsBatch> for GetSettingsBatchHandler<R> {
    type Output = HashMap<String, SettingValue>;
    type Error = String;

    async fn handle(&self, query: GetSettingsBatch) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing GetSettingsBatch query: {} keys", query.keys.len());

        self.repository.get_settings_batch(&query.keys).await
    }
}

// ============================================================================
// LOAD ALL SETTINGS
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct LoadAllSettings;

pub struct LoadAllSettingsHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> LoadAllSettingsHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> QueryHandler<LoadAllSettings> for LoadAllSettingsHandler<R> {
    type Output = Option<AppFullSettings>;
    type Error = String;

    async fn handle(&self, _query: LoadAllSettings) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing LoadAllSettings query");

        self.repository.load_all_settings().await
    }
}

// ============================================================================
// GET PHYSICS SETTINGS
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct GetPhysicsSettings {
    pub profile_name: String,
}

pub struct GetPhysicsSettingsHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> GetPhysicsSettingsHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> QueryHandler<GetPhysicsSettings> for GetPhysicsSettingsHandler<R> {
    type Output = PhysicsSettings;
    type Error = String;

    async fn handle(&self, query: GetPhysicsSettings) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing GetPhysicsSettings query: profile={}", query.profile_name);

        self.repository.get_physics_settings(&query.profile_name).await
    }
}

// ============================================================================
// LIST PHYSICS PROFILES
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct ListPhysicsProfiles;

pub struct ListPhysicsProfilesHandler<R: SettingsRepository> {
    repository: R,
}

impl<R: SettingsRepository> ListPhysicsProfilesHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: SettingsRepository> QueryHandler<ListPhysicsProfiles> for ListPhysicsProfilesHandler<R> {
    type Output = Vec<String>;
    type Error = String;

    async fn handle(&self, _query: ListPhysicsProfiles) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing ListPhysicsProfiles query");

        self.repository.list_physics_profiles().await
    }
}
```

## Knowledge Graph Domain

### Directives

```rust
// src/application/knowledge_graph/directives.rs

use hexser::{Directive, DirectiveHandler};
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;

// ============================================================================
// ADD NODE
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct AddNode {
    pub node: Node,
}

pub struct AddNodeHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> AddNodeHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> DirectiveHandler<AddNode> for AddNodeHandler<R> {
    type Output = u32; // Returns assigned node ID
    type Error = String;

    async fn handle(&self, directive: AddNode) -> Result<Self::Output, Self::Error> {
        log::info!("Executing AddNode directive: metadata_id={}", directive.node.metadata_id);

        let node_id = self.repository.add_node(&directive.node).await?;

        log::info!("Node added successfully: id={}", node_id);
        Ok(node_id)
    }
}

// ============================================================================
// UPDATE NODE
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct UpdateNode {
    pub node: Node,
}

pub struct UpdateNodeHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> UpdateNodeHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> DirectiveHandler<UpdateNode> for UpdateNodeHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: UpdateNode) -> Result<Self::Output, Self::Error> {
        log::info!("Executing UpdateNode directive: id={}", directive.node.id);

        self.repository.update_node(&directive.node).await?;

        log::info!("Node updated successfully: id={}", directive.node.id);
        Ok(())
    }
}

// ============================================================================
// REMOVE NODE
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct RemoveNode {
    pub node_id: u32,
}

pub struct RemoveNodeHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> RemoveNodeHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> DirectiveHandler<RemoveNode> for RemoveNodeHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: RemoveNode) -> Result<Self::Output, Self::Error> {
        log::info!("Executing RemoveNode directive: id={}", directive.node_id);

        self.repository.remove_node(directive.node_id).await?;

        log::info!("Node removed successfully: id={}", directive.node_id);
        Ok(())
    }
}

// ============================================================================
// ADD EDGE
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct AddEdge {
    pub edge: Edge,
}

pub struct AddEdgeHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> AddEdgeHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> DirectiveHandler<AddEdge> for AddEdgeHandler<R> {
    type Output = String; // Returns assigned edge ID
    type Error = String;

    async fn handle(&self, directive: AddEdge) -> Result<Self::Output, Self::Error> {
        log::info!("Executing AddEdge directive: source={}, target={}",
                   directive.edge.source, directive.edge.target);

        let edge_id = self.repository.add_edge(&directive.edge).await?;

        log::info!("Edge added successfully: id={}", edge_id);
        Ok(edge_id)
    }
}

// ============================================================================
// REMOVE EDGE
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct RemoveEdge {
    pub edge_id: String,
}

pub struct RemoveEdgeHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> RemoveEdgeHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> DirectiveHandler<RemoveEdge> for RemoveEdgeHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: RemoveEdge) -> Result<Self::Output, Self::Error> {
        log::info!("Executing RemoveEdge directive: id={}", directive.edge_id);

        self.repository.remove_edge(&directive.edge_id).await?;

        log::info!("Edge removed successfully: id={}", directive.edge_id);
        Ok(())
    }
}

// ============================================================================
// SAVE GRAPH
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct SaveGraph {
    pub graph: GraphData,
}

pub struct SaveGraphHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> SaveGraphHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> DirectiveHandler<SaveGraph> for SaveGraphHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: SaveGraph) -> Result<Self::Output, Self::Error> {
        log::info!("Executing SaveGraph directive: {} nodes, {} edges",
                   directive.graph.nodes.len(), directive.graph.edges.len());

        self.repository.save_graph(&directive.graph).await?;

        log::info!("Graph saved successfully");
        Ok(())
    }
}

// ============================================================================
// BATCH UPDATE POSITIONS
// ============================================================================

#[derive(Debug, Clone, Directive)]
pub struct BatchUpdatePositions {
    pub positions: Vec<(u32, f32, f32, f32)>, // (node_id, x, y, z)
}

pub struct BatchUpdatePositionsHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> BatchUpdatePositionsHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> DirectiveHandler<BatchUpdatePositions> for BatchUpdatePositionsHandler<R> {
    type Output = ();
    type Error = String;

    async fn handle(&self, directive: BatchUpdatePositions) -> Result<Self::Output, Self::Error> {
        log::info!("Executing BatchUpdatePositions directive: {} positions", directive.positions.len());

        self.repository.batch_update_positions(directive.positions).await?;

        log::info!("Positions updated successfully");
        Ok(())
    }
}
```

### Queries

```rust
// src/application/knowledge_graph/queries.rs

use hexser::{Query, QueryHandler};
use std::sync::Arc;
use crate::ports::knowledge_graph_repository::{KnowledgeGraphRepository, GraphStatistics};
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;

// ============================================================================
// LOAD GRAPH
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct LoadGraph;

pub struct LoadGraphHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> LoadGraphHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> QueryHandler<LoadGraph> for LoadGraphHandler<R> {
    type Output = Arc<GraphData>;
    type Error = String;

    async fn handle(&self, _query: LoadGraph) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing LoadGraph query");

        self.repository.load_graph().await
    }
}

// ============================================================================
// GET NODE
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct GetNode {
    pub node_id: u32,
}

pub struct GetNodeHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> GetNodeHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> QueryHandler<GetNode> for GetNodeHandler<R> {
    type Output = Option<Node>;
    type Error = String;

    async fn handle(&self, query: GetNode) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing GetNode query: id={}", query.node_id);

        self.repository.get_node(query.node_id).await
    }
}

// ============================================================================
// GET NODES BY METADATA ID
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct GetNodesByMetadataId {
    pub metadata_id: String,
}

pub struct GetNodesByMetadataIdHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> GetNodesByMetadataIdHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> QueryHandler<GetNodesByMetadataId> for GetNodesByMetadataIdHandler<R> {
    type Output = Vec<Node>;
    type Error = String;

    async fn handle(&self, query: GetNodesByMetadataId) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing GetNodesByMetadataId query: metadata_id={}", query.metadata_id);

        self.repository.get_nodes_by_metadata_id(&query.metadata_id).await
    }
}

// ============================================================================
// GET NODE EDGES
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct GetNodeEdges {
    pub node_id: u32,
}

pub struct GetNodeEdgesHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> GetNodeEdgesHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> QueryHandler<GetNodeEdges> for GetNodeEdgesHandler<R> {
    type Output = Vec<Edge>;
    type Error = String;

    async fn handle(&self, query: GetNodeEdges) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing GetNodeEdges query: node_id={}", query.node_id);

        self.repository.get_node_edges(query.node_id).await
    }
}

// ============================================================================
// QUERY NODES
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct QueryNodes {
    pub query_string: String,
}

pub struct QueryNodesHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> QueryNodesHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> QueryHandler<QueryNodes> for QueryNodesHandler<R> {
    type Output = Vec<Node>;
    type Error = String;

    async fn handle(&self, query: QueryNodes) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing QueryNodes query: query_string={}", query.query_string);

        self.repository.query_nodes(&query.query_string).await
    }
}

// ============================================================================
// GET GRAPH STATISTICS
// ============================================================================

#[derive(Debug, Clone, Query)]
pub struct GetGraphStatistics;

pub struct GetGraphStatisticsHandler<R: KnowledgeGraphRepository> {
    repository: R,
}

impl<R: KnowledgeGraphRepository> GetGraphStatisticsHandler<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }
}

#[hexser::async_trait]
impl<R: KnowledgeGraphRepository> QueryHandler<GetGraphStatistics> for GetGraphStatisticsHandler<R> {
    type Output = GraphStatistics;
    type Error = String;

    async fn handle(&self, _query: GetGraphStatistics) -> Result<Self::Output, Self::Error> {
        log::debug!("Executing GetGraphStatistics query");

        self.repository.get_statistics().await
    }
}
```

## Ontology Domain

Due to space, I'll provide the key directives and queries:

```rust
// src/application/ontology/directives.rs

// AddOwlClass, UpdateOwlClass, RemoveOwlClass
// AddOwlProperty, UpdateOwlProperty
// AddAxiom, RemoveAxiom
// StoreInferenceResults
// SaveOntologyGraph
```

```rust
// src/application/ontology/queries.rs

// LoadOntologyGraph
// GetOwlClass, ListOwlClasses
// GetOwlProperty, ListOwlProperties
// GetClassAxioms
// GetInferenceResults
// ValidateOntology
// QueryOntology
// GetOntologyMetrics
```

## Physics Domain

```rust
// src/application/physics/directives.rs

// InitializePhysics
// StartSimulation
// StopSimulation
// UpdateSimulationParameters
// UploadConstraints
// ClearConstraints
// SetNodePosition

// src/application/physics/queries.rs

// GetPhysicsStatistics
// GetNodePositions
// GetDeviceInfo
```

## Semantic Analysis Domain

```rust
// src/application/semantic/directives.rs

// TriggerCommunityDetection
// GenerateSemanticConstraints
// OptimizeLayout

// src/application/semantic/queries.rs

// ComputeShortestPaths
// AnalyzeNodeImportance
// GetSemanticStatistics
```

## Handler Registration

```rust
// src/application/mod.rs

use crate::ports::*;
use crate::adapters::*;

pub struct ApplicationServices {
    // Settings
    pub update_setting: UpdateSettingHandler<SqliteSettingsRepository>,
    pub get_setting: GetSettingHandler<SqliteSettingsRepository>,
    // ... all other handlers

    // Knowledge Graph
    pub add_node: AddNodeHandler<SqliteKnowledgeGraphRepository>,
    pub load_graph: LoadGraphHandler<SqliteKnowledgeGraphRepository>,
    // ... all other handlers

    // Ontology
    pub add_owl_class: AddOwlClassHandler<SqliteOntologyRepository>,
    pub load_ontology_graph: LoadOntologyGraphHandler<SqliteOntologyRepository>,
    // ... all other handlers

    // Physics
    pub start_simulation: StartSimulationHandler<PhysicsOrchestratorAdapter>,
    // ... all other handlers

    // Semantic
    pub detect_communities: DetectCommunitiesHandler<SemanticProcessorAdapter>,
    // ... all other handlers
}

impl ApplicationServices {
    pub fn new(
        settings_repo: SqliteSettingsRepository,
        kg_repo: SqliteKnowledgeGraphRepository,
        ontology_repo: SqliteOntologyRepository,
        physics_adapter: PhysicsOrchestratorAdapter,
        semantic_adapter: SemanticProcessorAdapter,
    ) -> Self {
        Self {
            // Initialize all handlers with their repositories/adapters
            update_setting: UpdateSettingHandler::new(settings_repo.clone()),
            get_setting: GetSettingHandler::new(settings_repo.clone()),
            // ... etc
        }
    }
}
```

## HTTP Handler Integration Example

```rust
// src/handlers/settings_handler.rs

use actix_web::{web, HttpResponse};
use crate::application::settings::*;

pub async fn update_setting_endpoint(
    services: web::Data<ApplicationServices>,
    directive: web::Json<UpdateSetting>,
) -> HttpResponse {
    match services.update_setting.handle(directive.into_inner()).await {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "message": "Setting updated successfully"
        })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "success": false,
            "error": e
        })),
    }
}

pub async fn get_setting_endpoint(
    services: web::Data<ApplicationServices>,
    query: web::Query<GetSetting>,
) -> HttpResponse {
    match services.get_setting.handle(query.into_inner()).await {
        Ok(Some(value)) => HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "value": value
        })),
        Ok(None) => HttpResponse::NotFound().json(serde_json::json!({
            "success": false,
            "error": "Setting not found"
        })),
        Err(e) => HttpResponse::InternalServerError().json(serde_json::json!({
            "success": false,
            "error": e
        })),
    }
}
```

## Summary

This CQRS layer provides:

1. **Clear separation** between read and write operations
2. **Type-safe** directives and queries using hexser
3. **Testable** handlers that depend on ports (interfaces), not implementations
4. **Composable** application services that can be easily extended
5. **HTTP-agnostic** business logic that can be used in any context

All handlers:
- Have explicit error types
- Log important operations
- Return meaningful results
- Are async-first
- Have NO business logic coupling to HTTP, WebSocket, or Actor framework
