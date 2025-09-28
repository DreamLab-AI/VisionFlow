//! Batch transaction handling for Neo4j operations

use crate::neo4j::{
    connection::Neo4jConnectionManager,
    error::{Neo4jError, Result},
    models::{Node, Relationship, Triple, GraphData},
    query_builder::{CypherQueryBuilder, QueryBuilder},
    operations::{BatchResult, ImportResult},
};
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use neo4rs::{Query, RowStream};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{Semaphore, RwLock};
use log::{debug, info, warn, error};
use std::time::Instant;

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub max_concurrent_batches: usize,
    pub retry_attempts: usize,
    pub retry_delay_ms: u64,
    pub enable_transaction: bool,
    pub timeout_seconds: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_concurrent_batches: 5,
            retry_attempts: 3,
            retry_delay_ms: 1000,
            enable_transaction: true,
            timeout_seconds: 300, // 5 minutes
        }
    }
}

/// Batch operation types
#[derive(Debug, Clone)]
pub enum BatchOperation {
    CreateNode(Node),
    CreateRelationship(Relationship),
    StoreTriple(Triple),
    UpdateNode { id: String, properties: std::collections::HashMap<String, Value> },
    UpdateRelationship { id: String, properties: std::collections::HashMap<String, Value> },
    DeleteNode(String),
    DeleteRelationship(String),
    CustomQuery(Query),
}

/// Result of individual batch operation
#[derive(Debug, Clone)]
pub struct BatchOperationResult {
    pub operation_index: usize,
    pub success: bool,
    pub result: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Statistics for batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchStatistics {
    pub total_operations: usize,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub total_batches: usize,
    pub successful_batches: usize,
    pub failed_batches: usize,
    pub total_duration_ms: u64,
    pub average_batch_time_ms: u64,
    pub throughput_ops_per_second: f64,
}

/// Batch processor for Neo4j operations
pub struct BatchProcessor {
    connection_manager: Arc<Neo4jConnectionManager>,
    config: BatchConfig,
    statistics: Arc<RwLock<BatchStatistics>>,
    semaphore: Arc<Semaphore>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(connection_manager: Neo4jConnectionManager, config: BatchConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_batches));
        
        Self {
            connection_manager: Arc::new(connection_manager),
            config,
            statistics: Arc::new(RwLock::new(BatchStatistics::default())),
            semaphore,
        }
    }
    
    /// Create with default configuration
    pub fn with_defaults(connection_manager: Neo4jConnectionManager) -> Self {
        Self::new(connection_manager, BatchConfig::default())
    }
    
    /// Process a batch of operations
    pub async fn process_batch(&self, operations: Vec<BatchOperation>) -> Result<BatchResult> {
        let start_time = Instant::now();
        info!("Starting batch processing of {} operations", operations.len());
        
        if operations.is_empty() {
            return Ok(BatchResult {
                total_items: 0,
                successful_items: 0,
                failed_items: 0,
                errors: Vec::new(),
                duration_ms: 0,
            });
        }
        
        let mut statistics = self.statistics.write().await;
        statistics.total_operations += operations.len();
        statistics.total_batches += 1;
        drop(statistics);
        
        // Split operations into smaller batches
        let batches: Vec<Vec<BatchOperation>> = operations
            .chunks(self.config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        info!("Split {} operations into {} batches of size {}", 
              operations.len(), batches.len(), self.config.batch_size);
        
        // Process batches concurrently with semaphore limiting
        let batch_results: Vec<Result<Vec<BatchOperationResult>>> = stream::iter(batches.into_iter().enumerate())
            .map(|(batch_index, batch_ops)| {
                let processor = self.clone();
                async move {
                    let _permit = processor.semaphore.acquire().await.map_err(|e| {
                        Neo4jError::query_error(format!("Failed to acquire semaphore: {}", e))
                    })?;
                    
                    processor.process_single_batch(batch_index, batch_ops).await
                }
            })
            .buffer_unordered(self.config.max_concurrent_batches)
            .collect()
            .await;
        
        // Aggregate results
        let mut all_results = Vec::new();
        let mut successful_items = 0;
        let mut failed_items = 0;
        let mut errors = Vec::new();
        let mut successful_batches = 0;
        let mut failed_batches = 0;
        
        for batch_result in batch_results {
            match batch_result {
                Ok(operation_results) => {
                    successful_batches += 1;
                    for op_result in operation_results {
                        if op_result.success {
                            successful_items += 1;
                        } else {
                            failed_items += 1;
                            if let Some(error) = op_result.error {
                                errors.push(error);
                            }
                        }
                        all_results.push(op_result);
                    }
                }
                Err(e) => {
                    failed_batches += 1;
                    errors.push(format!("Batch processing failed: {}", e));
                    // Estimate failed operations in this batch
                    failed_items += self.config.batch_size.min(operations.len());
                }
            }
        }
        
        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;
        
        // Update statistics
        let mut statistics = self.statistics.write().await;
        statistics.successful_operations += successful_items;
        statistics.failed_operations += failed_items;
        statistics.successful_batches += successful_batches;
        statistics.failed_batches += failed_batches;
        statistics.total_duration_ms += duration_ms;
        
        if statistics.total_batches > 0 {
            statistics.average_batch_time_ms = statistics.total_duration_ms / statistics.total_batches as u64;
        }
        
        if duration_ms > 0 {
            statistics.throughput_ops_per_second = (successful_items as f64) / (duration_ms as f64 / 1000.0);
        }
        
        info!("Batch processing completed in {}ms: {}/{} successful operations, {} successful batches, {} errors",
              duration_ms, successful_items, operations.len(), successful_batches, errors.len());
        
        Ok(BatchResult {
            total_items: operations.len(),
            successful_items,
            failed_items,
            errors,
            duration_ms,
        })
    }
    
    /// Process a single batch with retries
    async fn process_single_batch(&self, batch_index: usize, operations: Vec<BatchOperation>) -> Result<Vec<BatchOperationResult>> {
        let mut attempt = 0;
        let mut last_error = None;
        
        while attempt < self.config.retry_attempts {
            match self.execute_batch_operations(batch_index, &operations).await {
                Ok(results) => {
                    if attempt > 0 {
                        info!("Batch {} succeeded on attempt {}", batch_index, attempt + 1);
                    }
                    return Ok(results);
                }
                Err(e) => {
                    warn!("Batch {} failed on attempt {}: {}", batch_index, attempt + 1, e);
                    last_error = Some(e);
                    attempt += 1;
                    
                    if attempt < self.config.retry_attempts {
                        let delay = tokio::time::Duration::from_millis(self.config.retry_delay_ms * attempt as u64);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| Neo4jError::query_error("Batch processing failed after all retries")))
    }
    
    /// Execute operations in a single batch
    async fn execute_batch_operations(&self, batch_index: usize, operations: &[BatchOperation]) -> Result<Vec<BatchOperationResult>> {
        debug!("Executing batch {} with {} operations", batch_index, operations.len());
        let start_time = Instant::now();
        
        let mut results = Vec::new();
        
        if self.config.enable_transaction {
            // Use transaction for consistency
            self.execute_batch_with_transaction(operations, &mut results).await?
        } else {
            // Execute operations individually
            self.execute_batch_individually(operations, &mut results).await;
        }
        
        let duration = start_time.elapsed().as_millis() as u64;
        debug!("Batch {} completed in {}ms", batch_index, duration);
        
        Ok(results)
    }
    
    /// Execute batch operations within a transaction
    async fn execute_batch_with_transaction(&self, operations: &[BatchOperation], results: &mut Vec<BatchOperationResult>) -> Result<()> {
        // Neo4rs doesn't have explicit transaction support in the current version,
        // so we'll simulate it by preparing all queries and executing them together
        debug!("Executing {} operations in transaction-like batch", operations.len());
        
        let mut queries = Vec::new();
        
        // Convert operations to queries
        for (index, operation) in operations.iter().enumerate() {
            match self.operation_to_query(operation) {
                Ok(query) => queries.push((index, query)),
                Err(e) => {
                    results.push(BatchOperationResult {
                        operation_index: index,
                        success: false,
                        result: None,
                        error: Some(format!("Query preparation failed: {}", e)),
                        duration_ms: 0,
                    });
                }
            }
        }
        
        // Execute queries sequentially (simulating transaction)
        for (index, query) in queries {
            let op_start = Instant::now();
            
            match self.connection_manager.execute_query(query).await {
                Ok(mut result_stream) => {
                    let mut result_data = None;
                    if let Ok(Some(row)) = result_stream.next().await {
                        // Try to extract meaningful result
                        if let Ok(id) = row.get::<String>("id") {
                            result_data = Some(id);
                        } else if let Ok(count) = row.get::<i64>("count") {
                            result_data = Some(count.to_string());
                        }
                    }
                    
                    results.push(BatchOperationResult {
                        operation_index: index,
                        success: true,
                        result: result_data,
                        error: None,
                        duration_ms: op_start.elapsed().as_millis() as u64,
                    });
                }
                Err(e) => {
                    results.push(BatchOperationResult {
                        operation_index: index,
                        success: false,
                        result: None,
                        error: Some(format!("Query execution failed: {}", e)),
                        duration_ms: op_start.elapsed().as_millis() as u64,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute batch operations individually
    async fn execute_batch_individually(&self, operations: &[BatchOperation], results: &mut Vec<BatchOperationResult>) {
        debug!("Executing {} operations individually", operations.len());
        
        for (index, operation) in operations.iter().enumerate() {
            let op_start = Instant::now();
            
            match self.execute_single_operation(operation).await {
                Ok(result_data) => {
                    results.push(BatchOperationResult {
                        operation_index: index,
                        success: true,
                        result: result_data,
                        error: None,
                        duration_ms: op_start.elapsed().as_millis() as u64,
                    });
                }
                Err(e) => {
                    results.push(BatchOperationResult {
                        operation_index: index,
                        success: false,
                        result: None,
                        error: Some(e.to_string()),
                        duration_ms: op_start.elapsed().as_millis() as u64,
                    });
                }
            }
        }
    }
    
    /// Execute a single operation
    async fn execute_single_operation(&self, operation: &BatchOperation) -> Result<Option<String>> {
        match operation {
            BatchOperation::CreateNode(node) => {
                let query = self.operation_to_query(operation)?;
                let mut result = self.connection_manager.execute_query(query).await?;
                if let Some(row) = result.next().await? {
                    if let Ok(id) = row.get::<String>("node_id") {
                        return Ok(Some(id));
                    }
                }
                Ok(None)
            }
            BatchOperation::CreateRelationship(relationship) => {
                let query = self.operation_to_query(operation)?;
                let mut result = self.connection_manager.execute_query(query).await?;
                if let Some(row) = result.next().await? {
                    if let Ok(id) = row.get::<i64>("rel_id") {
                        return Ok(Some(id.to_string()));
                    }
                }
                Ok(None)
            }
            BatchOperation::StoreTriple(_) => {
                let query = self.operation_to_query(operation)?;
                let mut _result = self.connection_manager.execute_query(query).await?;
                Ok(Some("triple_stored".to_string()))
            }
            BatchOperation::UpdateNode { .. } | BatchOperation::UpdateRelationship { .. } => {
                let query = self.operation_to_query(operation)?;
                let mut result = self.connection_manager.execute_query(query).await?;
                if let Some(row) = result.next().await? {
                    if let Ok(count) = row.get::<i64>("updated_count") {
                        return Ok(Some(count.to_string()));
                    }
                }
                Ok(None)
            }
            BatchOperation::DeleteNode(_) | BatchOperation::DeleteRelationship(_) => {
                let query = self.operation_to_query(operation)?;
                let mut result = self.connection_manager.execute_query(query).await?;
                if let Some(row) = result.next().await? {
                    if let Ok(count) = row.get::<i64>("deleted_count") {
                        return Ok(Some(count.to_string()));
                    }
                }
                Ok(None)
            }
            BatchOperation::CustomQuery(query) => {
                let mut _result = self.connection_manager.execute_query(query.clone()).await?;
                Ok(Some("custom_query_executed".to_string()))
            }
        }
    }
    
    /// Convert batch operation to query
    fn operation_to_query(&self, operation: &BatchOperation) -> Result<Query> {
        match operation {
            BatchOperation::CreateNode(node) => {
                CypherQueryBuilder::create()
                    .create_node(node, "n")
                    .return_clause("n.id as node_id")
                    .build()
            }
            BatchOperation::CreateRelationship(relationship) => {
                CypherQueryBuilder::create()
                    .add_part("MATCH (start) WHERE start.id = $start_id")
                    .add_parameter("start_id", Value::String(relationship.start_node_id.clone()))
                    .add_part("MATCH (end) WHERE end.id = $end_id")
                    .add_parameter("end_id", Value::String(relationship.end_node_id.clone()))
                    .add_part(&format!("CREATE (start)-[r:{} $rel_props]->(end)", relationship.relationship_type))
                    .add_parameter("rel_props", serde_json::to_value(&relationship.properties)?)
                    .return_clause("id(r) as rel_id")
                    .build()
            }
            BatchOperation::StoreTriple(triple) => {
                CypherQueryBuilder::store_triple(triple).build()
            }
            BatchOperation::UpdateNode { id, properties } => {
                CypherQueryBuilder::update()
                    .add_part("MATCH (n)")
                    .add_part("WHERE n.id = $node_id")
                    .add_parameter("node_id", Value::String(id.clone()))
                    .add_part("SET n += $properties")
                    .add_parameter("properties", serde_json::to_value(properties)?)
                    .return_clause("count(n) as updated_count")
                    .build()
            }
            BatchOperation::UpdateRelationship { id, properties } => {
                CypherQueryBuilder::update()
                    .add_part("MATCH ()-[r]->()")
                    .add_part("WHERE id(r) = $rel_id")
                    .add_parameter("rel_id", Value::String(id.clone()))
                    .add_part("SET r += $properties")
                    .add_parameter("properties", serde_json::to_value(properties)?)
                    .return_clause("count(r) as updated_count")
                    .build()
            }
            BatchOperation::DeleteNode(id) => {
                CypherQueryBuilder::delete()
                    .add_part("MATCH (n)")
                    .add_part("WHERE n.id = $node_id")
                    .add_parameter("node_id", Value::String(id.clone()))
                    .add_part("DETACH DELETE n")
                    .return_clause("count(n) as deleted_count")
                    .build()
            }
            BatchOperation::DeleteRelationship(id) => {
                CypherQueryBuilder::delete()
                    .add_part("MATCH ()-[r]->()")
                    .add_part("WHERE id(r) = $rel_id")
                    .add_parameter("rel_id", Value::String(id.clone()))
                    .add_part("DELETE r")
                    .return_clause("count(r) as deleted_count")
                    .build()
            }
            BatchOperation::CustomQuery(query) => Ok(query.clone()),
        }
    }
    
    /// Get current batch statistics
    pub async fn get_statistics(&self) -> BatchStatistics {
        self.statistics.read().await.clone()
    }
    
    /// Reset statistics
    pub async fn reset_statistics(&self) {
        let mut statistics = self.statistics.write().await;
        *statistics = BatchStatistics::default();
        info!("Batch statistics reset");
    }
    
    /// Process nodes in batch
    pub async fn process_nodes_batch(&self, nodes: Vec<Node>) -> Result<BatchResult> {
        let operations: Vec<BatchOperation> = nodes
            .into_iter()
            .map(BatchOperation::CreateNode)
            .collect();
        
        self.process_batch(operations).await
    }
    
    /// Process relationships in batch
    pub async fn process_relationships_batch(&self, relationships: Vec<Relationship>) -> Result<BatchResult> {
        let operations: Vec<BatchOperation> = relationships
            .into_iter()
            .map(BatchOperation::CreateRelationship)
            .collect();
        
        self.process_batch(operations).await
    }
    
    /// Process triples in batch
    pub async fn process_triples_batch(&self, triples: Vec<Triple>) -> Result<BatchResult> {
        let operations: Vec<BatchOperation> = triples
            .into_iter()
            .map(BatchOperation::StoreTriple)
            .collect();
        
        self.process_batch(operations).await
    }
    
    /// Process graph data in optimized batches
    pub async fn process_graph_data(&self, graph_data: GraphData) -> Result<ImportResult> {
        let start_time = Instant::now();
        info!("Starting optimized graph data processing: {} nodes, {} relationships, {} triples",
              graph_data.node_count(), graph_data.relationship_count(), graph_data.triple_count());
        
        let mut total_errors = Vec::new();
        let mut nodes_created = 0u64;
        let mut relationships_created = 0u64;
        let mut triples_processed = 0u64;
        
        // Process nodes first
        if !graph_data.nodes.is_empty() {
            match self.process_nodes_batch(graph_data.nodes).await {
                Ok(result) => {
                    nodes_created = result.successful_items as u64;
                    total_errors.extend(result.errors);
                }
                Err(e) => {
                    error!("Failed to process nodes batch: {}", e);
                    total_errors.push(format!("Nodes batch failed: {}", e));
                }
            }
        }
        
        // Process relationships second
        if !graph_data.relationships.is_empty() {
            match self.process_relationships_batch(graph_data.relationships).await {
                Ok(result) => {
                    relationships_created = result.successful_items as u64;
                    total_errors.extend(result.errors);
                }
                Err(e) => {
                    error!("Failed to process relationships batch: {}", e);
                    total_errors.push(format!("Relationships batch failed: {}", e));
                }
            }
        }
        
        // Process triples last
        if !graph_data.triples.is_empty() {
            match self.process_triples_batch(graph_data.triples).await {
                Ok(result) => {
                    triples_processed = result.successful_items as u64;
                    total_errors.extend(result.errors);
                }
                Err(e) => {
                    error!("Failed to process triples batch: {}", e);
                    total_errors.push(format!("Triples batch failed: {}", e));
                }
            }
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        info!("Optimized graph processing completed in {}ms: {} nodes, {} relationships, {} triples, {} errors",
              duration_ms, nodes_created, relationships_created, triples_processed, total_errors.len());
        
        Ok(ImportResult {
            nodes_created,
            relationships_created,
            triples_processed,
            errors: total_errors,
            duration_ms,
        })
    }
}

// Implement Clone for BatchProcessor
impl Clone for BatchProcessor {
    fn clone(&self) -> Self {
        Self {
            connection_manager: Arc::clone(&self.connection_manager),
            config: self.config.clone(),
            statistics: Arc::clone(&self.statistics),
            semaphore: Arc::clone(&self.semaphore),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::Node;
    use serde_json::Value;
    
    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.max_concurrent_batches, 5);
        assert_eq!(config.retry_attempts, 3);
    }
    
    #[test]
    fn test_batch_operation_conversion() {
        let node = Node::new()
            .with_label("Test")
            .with_property("name", Value::String("test_node".to_string()));
        
        let operation = BatchOperation::CreateNode(node);
        
        match operation {
            BatchOperation::CreateNode(n) => {
                assert!(n.labels.contains(&"Test".to_string()));
            }
            _ => panic!("Wrong operation type"),
        }
    }
    
    #[test]
    fn test_batch_statistics() {
        let stats = BatchStatistics::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.successful_operations, 0);
        assert_eq!(stats.failed_operations, 0);
    }
}
