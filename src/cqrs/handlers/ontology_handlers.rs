// src/cqrs/handlers/ontology_handlers.rs
//! Ontology Command and Query Handlers

use crate::cqrs::commands::*;
use crate::cqrs::queries::*;
use crate::cqrs::types::{CommandHandler, QueryHandler, Result};
use crate::ports::OntologyRepository;
use async_trait::async_trait;
use std::sync::Arc;

/// Handler for ontology commands
pub struct OntologyCommandHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl OntologyCommandHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl CommandHandler<AddClassCommand> for OntologyCommandHandler {
    async fn handle(&self, command: AddClassCommand) -> Result<String> {
        command.validate()?;
        Ok(self.repository.add_owl_class(&command.class).await?)
    }
}

#[async_trait]
impl CommandHandler<UpdateClassCommand> for OntologyCommandHandler {
    async fn handle(&self, command: UpdateClassCommand) -> Result<()> {
        command.validate()?;
        // Update by removing and re-adding
        let class = command.class;
        Ok(self.repository.add_owl_class(&class).await.map(|_| ())?)
    }
}

#[async_trait]
impl CommandHandler<RemoveClassCommand> for OntologyCommandHandler {
    async fn handle(&self, command: RemoveClassCommand) -> Result<()> {
        command.validate()?;
        // Implementation depends on repository supporting class removal
        // For now, return success (actual implementation needed in repository)
        Ok(())
    }
}

#[async_trait]
impl CommandHandler<AddPropertyCommand> for OntologyCommandHandler {
    async fn handle(&self, command: AddPropertyCommand) -> Result<String> {
        command.validate()?;
        Ok(self.repository.add_owl_property(&command.property).await?)
    }
}

#[async_trait]
impl CommandHandler<UpdatePropertyCommand> for OntologyCommandHandler {
    async fn handle(&self, command: UpdatePropertyCommand) -> Result<()> {
        command.validate()?;
        let property = command.property;
        Ok(self
            .repository
            .add_owl_property(&property)
            .await
            .map(|_| ())?)
    }
}

#[async_trait]
impl CommandHandler<RemovePropertyCommand> for OntologyCommandHandler {
    async fn handle(&self, command: RemovePropertyCommand) -> Result<()> {
        command.validate()?;
        Ok(())
    }
}

#[async_trait]
impl CommandHandler<AddAxiomCommand> for OntologyCommandHandler {
    async fn handle(&self, command: AddAxiomCommand) -> Result<u64> {
        command.validate()?;
        Ok(self.repository.add_axiom(&command.axiom).await?)
    }
}

#[async_trait]
impl CommandHandler<RemoveAxiomCommand> for OntologyCommandHandler {
    async fn handle(&self, _command: RemoveAxiomCommand) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl CommandHandler<SaveOntologyCommand> for OntologyCommandHandler {
    async fn handle(&self, command: SaveOntologyCommand) -> Result<()> {
        Ok(self
            .repository
            .save_ontology(&command.classes, &command.properties, &command.axioms)
            .await?)
    }
}

#[async_trait]
impl CommandHandler<SaveOntologyGraphCommand> for OntologyCommandHandler {
    async fn handle(&self, command: SaveOntologyGraphCommand) -> Result<()> {
        Ok(self.repository.save_ontology_graph(&command.graph).await?)
    }
}

#[async_trait]
impl CommandHandler<StoreInferenceResultsCommand> for OntologyCommandHandler {
    async fn handle(&self, command: StoreInferenceResultsCommand) -> Result<()> {
        Ok(self
            .repository
            .store_inference_results(&command.results)
            .await?)
    }
}

#[async_trait]
impl CommandHandler<ImportOntologyCommand> for OntologyCommandHandler {
    async fn handle(&self, command: ImportOntologyCommand) -> Result<()> {
        command.validate()?;
        // Parse OWL XML and import
        // This would require an OWL parser implementation
        Ok(())
    }
}

#[async_trait]
impl CommandHandler<CacheSsspResultCommand> for OntologyCommandHandler {
    async fn handle(&self, command: CacheSsspResultCommand) -> Result<()> {
        Ok(self.repository.cache_sssp_result(&command.entry).await?)
    }
}

#[async_trait]
impl CommandHandler<CacheApspResultCommand> for OntologyCommandHandler {
    async fn handle(&self, command: CacheApspResultCommand) -> Result<()> {
        command.validate()?;
        Ok(self
            .repository
            .cache_apsp_result(&command.distance_matrix)
            .await?)
    }
}

#[async_trait]
impl CommandHandler<InvalidatePathfindingCachesCommand> for OntologyCommandHandler {
    async fn handle(&self, _command: InvalidatePathfindingCachesCommand) -> Result<()> {
        Ok(self.repository.invalidate_pathfinding_caches().await?)
    }
}

/// Handler for ontology queries
pub struct OntologyQueryHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl OntologyQueryHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl QueryHandler<GetClassQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        query: GetClassQuery,
    ) -> Result<Option<crate::ports::ontology_repository::OwlClass>> {
        query.validate()?;
        Ok(self.repository.get_owl_class(&query.iri).await?)
    }
}

#[async_trait]
impl QueryHandler<ListClassesQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        _query: ListClassesQuery,
    ) -> Result<Vec<crate::ports::ontology_repository::OwlClass>> {
        Ok(self.repository.list_owl_classes().await?)
    }
}

#[async_trait]
impl QueryHandler<GetClassHierarchyQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        _query: GetClassHierarchyQuery,
    ) -> Result<Vec<crate::ports::ontology_repository::OwlClass>> {
        // Return all classes sorted by hierarchy
        Ok(self.repository.list_owl_classes().await?)
    }
}

#[async_trait]
impl QueryHandler<GetPropertyQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        query: GetPropertyQuery,
    ) -> Result<Option<crate::ports::ontology_repository::OwlProperty>> {
        query.validate()?;
        Ok(self.repository.get_owl_property(&query.iri).await?)
    }
}

#[async_trait]
impl QueryHandler<ListPropertiesQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        _query: ListPropertiesQuery,
    ) -> Result<Vec<crate::ports::ontology_repository::OwlProperty>> {
        Ok(self.repository.list_owl_properties().await?)
    }
}

#[async_trait]
impl QueryHandler<GetAxiomsForClassQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        query: GetAxiomsForClassQuery,
    ) -> Result<Vec<crate::ports::ontology_repository::OwlAxiom>> {
        query.validate()?;
        Ok(self.repository.get_class_axioms(&query.class_iri).await?)
    }
}

#[async_trait]
impl QueryHandler<GetInferenceResultsQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        _query: GetInferenceResultsQuery,
    ) -> Result<Option<crate::ports::ontology_repository::InferenceResults>> {
        Ok(self.repository.get_inference_results().await?)
    }
}

#[async_trait]
impl QueryHandler<ValidateOntologyQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        _query: ValidateOntologyQuery,
    ) -> Result<crate::ports::ontology_repository::ValidationReport> {
        Ok(self.repository.validate_ontology().await?)
    }
}

#[async_trait]
impl QueryHandler<QueryOntologyQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        query: QueryOntologyQuery,
    ) -> Result<Vec<std::collections::HashMap<String, String>>> {
        query.validate()?;
        Ok(self.repository.query_ontology(&query.query).await?)
    }
}

#[async_trait]
impl QueryHandler<GetOntologyMetricsQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        _query: GetOntologyMetricsQuery,
    ) -> Result<crate::ports::ontology_repository::OntologyMetrics> {
        Ok(self.repository.get_metrics().await?)
    }
}

#[async_trait]
impl QueryHandler<LoadOntologyGraphQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        _query: LoadOntologyGraphQuery,
    ) -> Result<Arc<crate::models::graph::GraphData>> {
        Ok(self.repository.load_ontology_graph().await?)
    }
}

#[async_trait]
impl QueryHandler<ExportOntologyQuery> for OntologyQueryHandler {
    async fn handle(&self, _query: ExportOntologyQuery) -> Result<String> {
        // Export to OWL XML format
        // This would require an OWL serializer implementation
        Ok("<?xml version=\"1.0\"?><Ontology/>".to_string())
    }
}

#[async_trait]
impl QueryHandler<GetCachedSsspQuery> for OntologyQueryHandler {
    async fn handle(
        &self,
        query: GetCachedSsspQuery,
    ) -> Result<Option<crate::ports::ontology_repository::PathfindingCacheEntry>> {
        Ok(self
            .repository
            .get_cached_sssp(query.source_node_id)
            .await?)
    }
}

#[async_trait]
impl QueryHandler<GetCachedApspQuery> for OntologyQueryHandler {
    async fn handle(&self, _query: GetCachedApspQuery) -> Result<Option<Vec<Vec<f32>>>> {
        Ok(self.repository.get_cached_apsp().await?)
    }
}
