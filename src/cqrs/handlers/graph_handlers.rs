// src/cqrs/handlers/graph_handlers.rs
//! Knowledge Graph Command and Query Handlers
//!
//! Implements command and query handlers for the knowledge graph repository.

use crate::cqrs::commands::*;
use crate::cqrs::queries::*;
use crate::cqrs::types::{Command, CommandHandler, Query, QueryHandler, Result};
use crate::ports::KnowledgeGraphRepository;
use async_trait::async_trait;
use std::sync::Arc;

/// Handler for knowledge graph commands
pub struct GraphCommandHandler {
    repository: Arc<dyn KnowledgeGraphRepository>,
}

impl GraphCommandHandler {
    pub fn new(repository: Arc<dyn KnowledgeGraphRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl CommandHandler<AddNodeCommand> for GraphCommandHandler {
    async fn handle(&self, command: AddNodeCommand) -> Result<u32> {
        command.validate()?;
        Ok(self.repository.add_node(&command.node).await?)
    }
}

#[async_trait]
impl CommandHandler<AddNodesCommand> for GraphCommandHandler {
    async fn handle(&self, command: AddNodesCommand) -> Result<Vec<u32>> {
        command.validate()?;
        Ok(self.repository.batch_add_nodes(command.nodes).await?)
    }
}

#[async_trait]
impl CommandHandler<UpdateNodeCommand> for GraphCommandHandler {
    async fn handle(&self, command: UpdateNodeCommand) -> Result<()> {
        command.validate()?;
        Ok(self.repository.update_node(&command.node).await?)
    }
}

#[async_trait]
impl CommandHandler<UpdateNodesCommand> for GraphCommandHandler {
    async fn handle(&self, command: UpdateNodesCommand) -> Result<()> {
        command.validate()?;
        Ok(self.repository.batch_update_nodes(command.nodes).await?)
    }
}

#[async_trait]
impl CommandHandler<RemoveNodeCommand> for GraphCommandHandler {
    async fn handle(&self, command: RemoveNodeCommand) -> Result<()> {
        Ok(self.repository.remove_node(command.node_id).await?)
    }
}

#[async_trait]
impl CommandHandler<RemoveNodesCommand> for GraphCommandHandler {
    async fn handle(&self, command: RemoveNodesCommand) -> Result<()> {
        command.validate()?;
        Ok(self.repository.batch_remove_nodes(command.node_ids).await?)
    }
}

#[async_trait]
impl CommandHandler<AddEdgeCommand> for GraphCommandHandler {
    async fn handle(&self, command: AddEdgeCommand) -> Result<String> {
        command.validate()?;
        Ok(self.repository.add_edge(&command.edge).await?)
    }
}

#[async_trait]
impl CommandHandler<AddEdgesCommand> for GraphCommandHandler {
    async fn handle(&self, command: AddEdgesCommand) -> Result<Vec<String>> {
        command.validate()?;
        Ok(self.repository.batch_add_edges(command.edges).await?)
    }
}

#[async_trait]
impl CommandHandler<UpdateEdgeCommand> for GraphCommandHandler {
    async fn handle(&self, command: UpdateEdgeCommand) -> Result<()> {
        Ok(self.repository.update_edge(&command.edge).await?)
    }
}

#[async_trait]
impl CommandHandler<RemoveEdgeCommand> for GraphCommandHandler {
    async fn handle(&self, command: RemoveEdgeCommand) -> Result<()> {
        command.validate()?;
        Ok(self.repository.remove_edge(&command.edge_id).await?)
    }
}

#[async_trait]
impl CommandHandler<RemoveEdgesCommand> for GraphCommandHandler {
    async fn handle(&self, command: RemoveEdgesCommand) -> Result<()> {
        command.validate()?;
        Ok(self.repository.batch_remove_edges(command.edge_ids).await?)
    }
}

#[async_trait]
impl CommandHandler<SaveGraphCommand> for GraphCommandHandler {
    async fn handle(&self, command: SaveGraphCommand) -> Result<()> {
        Ok(self.repository.save_graph(&command.graph).await?)
    }
}

#[async_trait]
impl CommandHandler<ClearGraphCommand> for GraphCommandHandler {
    async fn handle(&self, _command: ClearGraphCommand) -> Result<()> {
        Ok(self.repository.clear_graph().await?)
    }
}

#[async_trait]
impl CommandHandler<UpdatePositionsCommand> for GraphCommandHandler {
    async fn handle(&self, command: UpdatePositionsCommand) -> Result<()> {
        command.validate()?;
        Ok(self
            .repository
            .batch_update_positions(command.positions)
            .await?)
    }
}

/// Handler for knowledge graph queries
pub struct GraphQueryHandler {
    repository: Arc<dyn KnowledgeGraphRepository>,
}

impl GraphQueryHandler {
    pub fn new(repository: Arc<dyn KnowledgeGraphRepository>) -> Self {
        Self { repository }
    }
}

#[async_trait]
impl QueryHandler<GetNodeQuery> for GraphQueryHandler {
    async fn handle(&self, query: GetNodeQuery) -> Result<Option<crate::models::node::Node>> {
        Ok(self.repository.get_node(query.node_id).await?)
    }
}

#[async_trait]
impl QueryHandler<GetNodesQuery> for GraphQueryHandler {
    async fn handle(&self, query: GetNodesQuery) -> Result<Vec<crate::models::node::Node>> {
        query.validate()?;
        Ok(self.repository.get_nodes(query.node_ids).await?)
    }
}

#[async_trait]
impl QueryHandler<GetAllNodesQuery> for GraphQueryHandler {
    async fn handle(&self, _query: GetAllNodesQuery) -> Result<Vec<crate::models::node::Node>> {
        let graph = self.repository.load_graph().await?;
        Ok(graph.nodes.clone())
    }
}

#[async_trait]
impl QueryHandler<SearchNodesQuery> for GraphQueryHandler {
    async fn handle(&self, query: SearchNodesQuery) -> Result<Vec<crate::models::node::Node>> {
        query.validate()?;
        Ok(self
            .repository
            .search_nodes_by_label(&query.label_pattern)
            .await?)
    }
}

#[async_trait]
impl QueryHandler<GetNodesByMetadataQuery> for GraphQueryHandler {
    async fn handle(
        &self,
        query: GetNodesByMetadataQuery,
    ) -> Result<Vec<crate::models::node::Node>> {
        query.validate()?;
        Ok(self
            .repository
            .get_nodes_by_metadata_id(&query.metadata_id)
            .await?)
    }
}

#[async_trait]
impl QueryHandler<GetNodeEdgesQuery> for GraphQueryHandler {
    async fn handle(&self, query: GetNodeEdgesQuery) -> Result<Vec<crate::models::edge::Edge>> {
        Ok(self.repository.get_node_edges(query.node_id).await?)
    }
}

#[async_trait]
impl QueryHandler<GetEdgesBetweenQuery> for GraphQueryHandler {
    async fn handle(&self, query: GetEdgesBetweenQuery) -> Result<Vec<crate::models::edge::Edge>> {
        Ok(self
            .repository
            .get_edges_between(query.source_id, query.target_id)
            .await?)
    }
}

#[async_trait]
impl QueryHandler<GetNeighborsQuery> for GraphQueryHandler {
    async fn handle(&self, query: GetNeighborsQuery) -> Result<Vec<crate::models::node::Node>> {
        Ok(self.repository.get_neighbors(query.node_id).await?)
    }
}

#[async_trait]
impl QueryHandler<CountNodesQuery> for GraphQueryHandler {
    async fn handle(&self, _query: CountNodesQuery) -> Result<usize> {
        let stats = self.repository.get_statistics().await?;
        Ok(stats.node_count)
    }
}

#[async_trait]
impl QueryHandler<CountEdgesQuery> for GraphQueryHandler {
    async fn handle(&self, _query: CountEdgesQuery) -> Result<usize> {
        let stats = self.repository.get_statistics().await?;
        Ok(stats.edge_count)
    }
}

#[async_trait]
impl QueryHandler<GetGraphStatsQuery> for GraphQueryHandler {
    async fn handle(
        &self,
        _query: GetGraphStatsQuery,
    ) -> Result<crate::ports::knowledge_graph_repository::GraphStatistics> {
        Ok(self.repository.get_statistics().await?)
    }
}

#[async_trait]
impl QueryHandler<LoadGraphQuery> for GraphQueryHandler {
    async fn handle(&self, _query: LoadGraphQuery) -> Result<Arc<crate::models::graph::GraphData>> {
        Ok(self.repository.load_graph().await?)
    }
}

#[async_trait]
impl QueryHandler<QueryNodesQuery> for GraphQueryHandler {
    async fn handle(&self, query: QueryNodesQuery) -> Result<Vec<crate::models::node::Node>> {
        query.validate()?;
        Ok(self.repository.query_nodes(&query.query).await?)
    }
}

#[async_trait]
impl QueryHandler<GraphHealthCheckQuery> for GraphQueryHandler {
    async fn handle(&self, _query: GraphHealthCheckQuery) -> Result<bool> {
        Ok(self.repository.health_check().await?)
    }
}
