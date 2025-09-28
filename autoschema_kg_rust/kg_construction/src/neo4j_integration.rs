//! Neo4j integration example and convenience functions

use crate::neo4j::{
    Neo4jConnectionManager, Neo4jHooks, HookHelpers, GraphOperationsImpl,
    BatchProcessor, IndexManager, Triple, Node, Relationship, GraphData,
    Neo4jError, Result,
};
use log::{info, warn, error};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// High-level Neo4j service for knowledge graph operations
pub struct Neo4jService {
    connection_manager: Neo4jConnectionManager,
    graph_ops: GraphOperationsImpl,
    batch_processor: BatchProcessor,
    index_manager: IndexManager,
    hooks: Arc<tokio::sync::Mutex<Neo4jHooks>>,
}

impl Neo4jService {
    /// Create a new Neo4j service
    pub async fn new() -> Result<Self> {
        let connection_manager = Neo4jConnectionManager::from_env().await?;
        Self::with_connection_manager(connection_manager).await
    }
    
    /// Create with custom connection manager
    pub async fn with_connection_manager(connection_manager: Neo4jConnectionManager) -> Result<Self> {
        let graph_ops = GraphOperationsImpl::new(connection_manager.clone());
        let batch_processor = BatchProcessor::with_defaults(connection_manager.clone());
        let index_manager = IndexManager::new(connection_manager.clone());
        let hooks = Arc::new(tokio::sync::Mutex::new(Neo4jHooks::new(connection_manager.clone())));
        
        let service = Self {
            connection_manager,
            graph_ops,
            batch_processor,
            index_manager,
            hooks,
        };
        
        // Initialize hooks
        service.initialize_hooks().await?;
        
        Ok(service)
    }
    
    /// Initialize coordination hooks
    async fn initialize_hooks(&self) -> Result<()> {
        let mut hooks = self.hooks.lock().await;
        
        // Check if hooks are available
        if hooks.check_hooks_available().await {
            info!("Coordination hooks initialized successfully");
        } else {
            warn!("Coordination hooks not available - running in standalone mode");
        }
        
        Ok(())
    }
    
    /// Set up the knowledge graph with indexes and constraints
    pub async fn setup_knowledge_graph(&self) -> Result<()> {
        info!("Setting up knowledge graph with indexes and constraints");
        
        // Execute pre-task hook
        {
            let mut hooks = self.hooks.lock().await;
            if let Err(e) = HookHelpers::pre_task(&mut hooks, "Setup Neo4j knowledge graph").await {
                warn!("Pre-task hook failed: {}", e);
            }
        }
        
        // Create indexes and constraints
        match self.index_manager.optimize_for_knowledge_graph().await {
            Ok(_) => info!("Knowledge graph setup completed successfully"),
            Err(e) => {
                error!("Knowledge graph setup failed: {}", e);
                return Err(e);
            }
        }
        
        // Execute post-task hook
        {
            let mut hooks = self.hooks.lock().await;
            if let Err(e) = HookHelpers::post_task(&mut hooks, "kg_setup").await {
                warn!("Post-task hook failed: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Store a single triple with hooks integration
    pub async fn store_triple(&self, triple: &Triple) -> Result<()> {
        // Store operation metadata
        {
            let hooks = self.hooks.lock().await;
            let metadata = json!({
                "operation": "store_triple",
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object
            });
            
            if let Err(e) = hooks.store_operation_metadata("store_triple", metadata).await {
                warn!("Failed to store operation metadata: {}", e);
            }
        }
        
        let start_time = std::time::Instant::now();
        let result = self.graph_ops.store_triple(triple).await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        // Store performance metrics
        {
            let hooks = self.hooks.lock().await;
            if let Err(e) = hooks.store_performance_metrics("store_triple", duration_ms, result.is_ok()).await {
                warn!("Failed to store performance metrics: {}", e);
            }
        }
        
        result
    }
    
    /// Store multiple triples in batch with hooks
    pub async fn store_triples_batch(&self, triples: Vec<Triple>) -> Result<crate::neo4j::operations::BatchResult> {
        info!("Storing {} triples in batch", triples.len());
        
        // Execute pre-task hook
        {
            let mut hooks = self.hooks.lock().await;
            if let Err(e) = HookHelpers::pre_task(&mut hooks, &format!("Store {} triples in batch", triples.len())).await {
                warn!("Pre-task hook failed: {}", e);
            }
        }
        
        let start_time = std::time::Instant::now();
        let result = self.batch_processor.process_triples_batch(triples).await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        // Store performance metrics and notify
        {
            let hooks = self.hooks.lock().await;
            if let Err(e) = hooks.store_performance_metrics("batch_store_triples", duration_ms, result.is_ok()).await {
                warn!("Failed to store performance metrics: {}", e);
            }
        }
        
        match &result {
            Ok(batch_result) => {
                let mut hooks = self.hooks.lock().await;
                let message = format!("Successfully stored {}/{} triples in {}ms", 
                                     batch_result.successful_items, 
                                     batch_result.total_items, 
                                     duration_ms);
                if let Err(e) = HookHelpers::notify(&mut hooks, &message).await {
                    warn!("Notify hook failed: {}", e);
                }
                
                if let Err(e) = HookHelpers::post_task(&mut hooks, "batch_store_triples").await {
                    warn!("Post-task hook failed: {}", e);
                }
            }
            Err(e) => {
                let mut hooks = self.hooks.lock().await;
                let message = format!("Batch triple storage failed: {}", e);
                if let Err(e) = HookHelpers::notify(&mut hooks, &message).await {
                    warn!("Notify hook failed: {}", e);
                }
            }
        }
        
        result
    }
    
    /// Import graph data with comprehensive hooks integration
    pub async fn import_graph_data(&self, data: GraphData) -> Result<crate::neo4j::operations::ImportResult> {
        info!("Importing graph data: {} nodes, {} relationships, {} triples", 
              data.node_count(), data.relationship_count(), data.triple_count());
        
        // Execute pre-task hook
        {
            let mut hooks = self.hooks.lock().await;
            let description = format!("Import graph data: {} nodes, {} relationships, {} triples", 
                                     data.node_count(), data.relationship_count(), data.triple_count());
            if let Err(e) = HookHelpers::pre_task(&mut hooks, &description).await {
                warn!("Pre-task hook failed: {}", e);
            }
        }
        
        let start_time = std::time::Instant::now();
        let result = self.batch_processor.process_graph_data(data).await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        // Store comprehensive metrics and notify
        {
            let hooks = self.hooks.lock().await;
            if let Err(e) = hooks.store_performance_metrics("import_graph_data", duration_ms, result.is_ok()).await {
                warn!("Failed to store performance metrics: {}", e);
            }
        }
        
        match &result {
            Ok(import_result) => {
                let mut hooks = self.hooks.lock().await;
                let message = format!("Graph import completed: {} nodes, {} relationships, {} triples in {}ms", 
                                     import_result.nodes_created,
                                     import_result.relationships_created,
                                     import_result.triples_processed,
                                     duration_ms);
                if let Err(e) = HookHelpers::notify(&mut hooks, &message).await {
                    warn!("Notify hook failed: {}", e);
                }
                
                // Store detailed import metrics
                let metadata = json!({
                    "operation": "import_graph_data",
                    "nodes_created": import_result.nodes_created,
                    "relationships_created": import_result.relationships_created,
                    "triples_processed": import_result.triples_processed,
                    "errors": import_result.errors.len(),
                    "duration_ms": duration_ms
                });
                
                if let Err(e) = hooks.store_operation_metadata("import_graph_data", metadata).await {
                    warn!("Failed to store import metadata: {}", e);
                }
                
                if let Err(e) = HookHelpers::post_task(&mut hooks, "import_graph_data").await {
                    warn!("Post-task hook failed: {}", e);
                }
            }
            Err(e) => {
                let mut hooks = self.hooks.lock().await;
                let message = format!("Graph import failed: {}", e);
                if let Err(e) = HookHelpers::notify(&mut hooks, &message).await {
                    warn!("Notify hook failed: {}", e);
                }
            }
        }
        
        result
    }
    
    /// Search triples with hooks integration
    pub async fn search_triples(&self, subject: Option<&str>, predicate: Option<&str>, object: Option<&str>) -> Result<Vec<Triple>> {
        let search_desc = format!("Search triples: subject={:?}, predicate={:?}, object={:?}", subject, predicate, object);
        
        let start_time = std::time::Instant::now();
        let result = self.graph_ops.search_triples(subject, predicate, object).await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        // Store search metadata
        {
            let hooks = self.hooks.lock().await;
            let metadata = json!({
                "operation": "search_triples",
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "results_count": result.as_ref().map(|r| r.len()).unwrap_or(0),
                "duration_ms": duration_ms
            });
            
            if let Err(e) = hooks.store_operation_metadata("search_triples", metadata).await {
                warn!("Failed to store search metadata: {}", e);
            }
            
            if let Err(e) = hooks.store_performance_metrics("search_triples", duration_ms, result.is_ok()).await {
                warn!("Failed to store performance metrics: {}", e);
            }
        }
        
        if let Ok(ref triples) = result {
            info!("Search completed: found {} triples in {}ms", triples.len(), duration_ms);
        }
        
        result
    }
    
    /// Get graph statistics with hooks
    pub async fn get_graph_statistics(&self) -> Result<crate::neo4j::operations::GraphStatistics> {
        let start_time = std::time::Instant::now();
        let result = self.graph_ops.get_graph_statistics().await;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        if let Ok(ref stats) = result {
            // Store statistics as memory data
            let mut hooks = self.hooks.lock().await;
            let stats_data = json!({
                "node_count": stats.node_count,
                "relationship_count": stats.relationship_count,
                "label_counts": stats.label_counts,
                "relationship_type_counts": stats.relationship_type_counts,
                "timestamp": chrono::Utc::now().to_rfc3339()
            });
            
            if let Err(e) = HookHelpers::store_memory(&mut hooks, "graph_statistics", stats_data).await {
                warn!("Failed to store graph statistics in memory: {}", e);
            }
            
            if let Err(e) = hooks.store_performance_metrics("get_graph_statistics", duration_ms, true).await {
                warn!("Failed to store performance metrics: {}", e);
            }
        }
        
        result
    }
    
    /// Health check for the entire service
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let mut status = HealthStatus::new();
        
        // Check connection
        match self.connection_manager.health_check().await {
            Ok(true) => {
                status.connection_healthy = true;
                info!("Neo4j connection health check passed");
            }
            Ok(false) => {
                status.connection_healthy = false;
                warn!("Neo4j connection health check failed");
            }
            Err(e) => {
                status.connection_healthy = false;
                status.errors.push(format!("Connection health check error: {}", e));
                error!("Neo4j connection health check error: {}", e);
            }
        }
        
        // Check hooks availability
        {
            let hooks = self.hooks.lock().await;
            status.hooks_available = hooks.check_hooks_available().await;
        }
        
        // Get basic statistics
        match self.get_graph_statistics().await {
            Ok(stats) => {
                status.node_count = Some(stats.node_count);
                status.relationship_count = Some(stats.relationship_count);
            }
            Err(e) => {
                status.errors.push(format!("Statistics retrieval failed: {}", e));
            }
        }
        
        status.overall_healthy = status.connection_healthy && status.errors.is_empty();
        
        Ok(status)
    }
    
    /// Close the service and clean up resources
    pub async fn close(&self) -> Result<()> {
        info!("Closing Neo4j service");
        
        // Execute session end hook
        {
            let mut hooks = self.hooks.lock().await;
            if let Err(e) = hooks.execute_hook(crate::neo4j::HookEvent::SessionEnd { export_metrics: true }).await {
                warn!("Session end hook failed: {}", e);
            }
        }
        
        // Close connection pool
        self.connection_manager.close().await;
        
        info!("Neo4j service closed successfully");
        Ok(())
    }
    
    /// Get service statistics
    pub async fn get_service_statistics(&self) -> ServiceStatistics {
        let mut stats = ServiceStatistics::new();
        
        // Connection statistics
        stats.connection_stats = Some(self.connection_manager.get_stats().await);
        
        // Batch processing statistics
        stats.batch_stats = Some(self.batch_processor.get_statistics().await);
        
        // Graph statistics
        if let Ok(graph_stats) = self.get_graph_statistics().await {
            stats.graph_stats = Some(graph_stats);
        }
        
        stats
    }
}

/// Health status of the Neo4j service
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub overall_healthy: bool,
    pub connection_healthy: bool,
    pub hooks_available: bool,
    pub node_count: Option<u64>,
    pub relationship_count: Option<u64>,
    pub errors: Vec<String>,
}

impl HealthStatus {
    pub fn new() -> Self {
        Self {
            overall_healthy: false,
            connection_healthy: false,
            hooks_available: false,
            node_count: None,
            relationship_count: None,
            errors: Vec::new(),
        }
    }
}

/// Comprehensive service statistics
#[derive(Debug, Clone)]
pub struct ServiceStatistics {
    pub connection_stats: Option<crate::neo4j::connection::ConnectionStats>,
    pub batch_stats: Option<crate::neo4j::batch::BatchStatistics>,
    pub graph_stats: Option<crate::neo4j::operations::GraphStatistics>,
}

impl ServiceStatistics {
    pub fn new() -> Self {
        Self {
            connection_stats: None,
            batch_stats: None,
            graph_stats: None,
        }
    }
}

/// Example usage and integration patterns
pub mod examples {
    use super::*;
    
    /// Example: Basic triple storage with hooks
    pub async fn example_store_triples() -> Result<()> {
        let service = Neo4jService::new().await?;
        
        // Set up the knowledge graph
        service.setup_knowledge_graph().await?;
        
        // Create some example triples
        let triples = vec![
            Triple::new("Alice", "knows", "Bob")
                .with_confidence(0.9)
                .with_source("example"),
            Triple::new("Bob", "worksAt", "Company")
                .with_confidence(0.8)
                .with_source("example"),
            Triple::new("Company", "locatedIn", "City")
                .with_confidence(1.0)
                .with_source("example"),
        ];
        
        // Store triples in batch
        let result = service.store_triples_batch(triples).await?;
        println!("Stored {}/{} triples successfully", result.successful_items, result.total_items);
        
        // Search for triples
        let found_triples = service.search_triples(Some("Alice"), None, None).await?;
        println!("Found {} triples for Alice", found_triples.len());
        
        // Get statistics
        let stats = service.get_graph_statistics().await?;
        println!("Graph has {} nodes and {} relationships", stats.node_count, stats.relationship_count);
        
        // Health check
        let health = service.health_check().await?;
        println!("Service healthy: {}", health.overall_healthy);
        
        service.close().await?;
        Ok(())
    }
    
    /// Example: Knowledge graph import
    pub async fn example_import_knowledge_graph() -> Result<()> {
        let service = Neo4jService::new().await?;
        
        // Create graph data
        let mut graph_data = GraphData::new();
        
        // Add some nodes
        graph_data = graph_data
            .add_node(Node::new()
                .with_label("Person")
                .with_property("name", json!("Alice"))
                .with_property("age", json!(30)))
            .add_node(Node::new()
                .with_label("Company")
                .with_property("name", json!("TechCorp"))
                .with_property("industry", json!("Technology")));
        
        // Add some triples
        graph_data = graph_data
            .add_triple(Triple::new("Alice", "worksAt", "TechCorp")
                .with_confidence(0.95)
                .with_source("hr_system"))
            .add_triple(Triple::new("TechCorp", "hasIndustry", "Technology")
                .with_confidence(1.0)
                .with_source("company_data"));
        
        // Import the data
        let import_result = service.import_graph_data(graph_data).await?;
        println!("Import completed: {} nodes, {} relationships, {} triples",
                import_result.nodes_created,
                import_result.relationships_created, 
                import_result.triples_processed);
        
        service.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_health_status_creation() {
        let status = HealthStatus::new();
        assert!(!status.overall_healthy);
        assert!(!status.connection_healthy);
        assert!(!status.hooks_available);
        assert!(status.errors.is_empty());
    }
    
    #[test]
    fn test_service_statistics_creation() {
        let stats = ServiceStatistics::new();
        assert!(stats.connection_stats.is_none());
        assert!(stats.batch_stats.is_none());
        assert!(stats.graph_stats.is_none());
    }
}
