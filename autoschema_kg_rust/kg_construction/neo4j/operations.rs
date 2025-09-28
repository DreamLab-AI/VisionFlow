//! Graph operations for nodes and relationships

use crate::neo4j::{
    connection::Neo4jConnectionManager,
    error::{Neo4jError, Result},
    models::{Node, Relationship, Triple, GraphData, QueryParams},
    query_builder::{CypherQueryBuilder, QueryBuilder, TripleQueryBuilder},
};
use async_trait::async_trait;
use futures::StreamExt;
use neo4rs::{Row, RowStream};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;
use log::{debug, info, warn, error};

/// Trait for node operations
#[async_trait]
pub trait NodeOperations {
    async fn create_node(&self, node: &Node) -> Result<String>;
    async fn get_node(&self, id: &str) -> Result<Option<Node>>;
    async fn update_node(&self, id: &str, properties: HashMap<String, Value>) -> Result<bool>;
    async fn delete_node(&self, id: &str) -> Result<bool>;
    async fn find_nodes_by_label(&self, label: &str, params: Option<QueryParams>) -> Result<Vec<Node>>;
    async fn find_nodes_by_properties(&self, properties: HashMap<String, Value>, params: Option<QueryParams>) -> Result<Vec<Node>>;
    async fn count_nodes(&self) -> Result<u64>;
}

/// Trait for relationship operations
#[async_trait]
pub trait RelationshipOperations {
    async fn create_relationship(&self, relationship: &Relationship) -> Result<String>;
    async fn get_relationship(&self, id: &str) -> Result<Option<Relationship>>;
    async fn update_relationship(&self, id: &str, properties: HashMap<String, Value>) -> Result<bool>;
    async fn delete_relationship(&self, id: &str) -> Result<bool>;
    async fn find_relationships_by_type(&self, rel_type: &str, params: Option<QueryParams>) -> Result<Vec<Relationship>>;
    async fn find_relationships_between_nodes(&self, start_node_id: &str, end_node_id: &str) -> Result<Vec<Relationship>>;
    async fn count_relationships(&self) -> Result<u64>;
}

/// Trait for graph-level operations
#[async_trait]
pub trait GraphOperations {
    async fn import_graph_data(&self, data: &GraphData) -> Result<ImportResult>;
    async fn export_graph_data(&self, params: Option<QueryParams>) -> Result<GraphData>;
    async fn store_triple(&self, triple: &Triple) -> Result<()>;
    async fn store_triples_batch(&self, triples: &[Triple]) -> Result<BatchResult>;
    async fn get_triples(&self, params: Option<QueryParams>) -> Result<Vec<Triple>>;
    async fn search_triples(&self, subject: Option<&str>, predicate: Option<&str>, object: Option<&str>) -> Result<Vec<Triple>>;
    async fn delete_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<bool>;
    async fn get_graph_statistics(&self) -> Result<GraphStatistics>;
    async fn clear_graph(&self) -> Result<()>;
}

/// Result of import operation
#[derive(Debug, Clone)]
pub struct ImportResult {
    pub nodes_created: u64,
    pub relationships_created: u64,
    pub triples_processed: u64,
    pub errors: Vec<String>,
    pub duration_ms: u64,
}

/// Result of batch operation
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub total_items: usize,
    pub successful_items: usize,
    pub failed_items: usize,
    pub errors: Vec<String>,
    pub duration_ms: u64,
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub node_count: u64,
    pub relationship_count: u64,
    pub label_counts: HashMap<String, u64>,
    pub relationship_type_counts: HashMap<String, u64>,
}

/// Implementation of graph operations
pub struct GraphOperationsImpl {
    connection_manager: Neo4jConnectionManager,
}

impl GraphOperationsImpl {
    pub fn new(connection_manager: Neo4jConnectionManager) -> Self {
        Self { connection_manager }
    }
    
    /// Helper to convert Neo4j row to Node
    async fn row_to_node(row: &Row) -> Result<Node> {
        let neo4j_node: neo4rs::Node = row.get("n").map_err(|e| {
            Neo4jError::query_error(format!("Failed to get node from row: {}", e))
        })?;
        
        let id = neo4j_node.id().to_string();
        let labels: Vec<String> = neo4j_node.labels().map(|s| s.to_string()).collect();
        let properties: HashMap<String, Value> = neo4j_node.properties()
            .map(|(k, v)| (k.to_string(), Self::neo4j_value_to_json(v)))
            .collect();
        
        Ok(Node {
            id: Some(id),
            labels,
            properties,
            created_at: chrono::Utc::now(), // This should ideally come from the database
            updated_at: None,
        })
    }
    
    /// Helper to convert Neo4j row to Relationship
    async fn row_to_relationship(row: &Row) -> Result<Relationship> {
        let neo4j_rel: neo4rs::Relation = row.get("r").map_err(|e| {
            Neo4jError::query_error(format!("Failed to get relationship from row: {}", e))
        })?;
        
        let id = neo4j_rel.id().to_string();
        let relationship_type = neo4j_rel.typ().to_string();
        let start_node_id = neo4j_rel.start_node_id().to_string();
        let end_node_id = neo4j_rel.end_node_id().to_string();
        let properties: HashMap<String, Value> = neo4j_rel.properties()
            .map(|(k, v)| (k.to_string(), Self::neo4j_value_to_json(v)))
            .collect();
        
        Ok(Relationship {
            id: Some(id),
            relationship_type,
            start_node_id,
            end_node_id,
            properties,
            created_at: chrono::Utc::now(), // This should ideally come from the database
            updated_at: None,
        })
    }
    
    /// Helper to convert Neo4j row to Triple
    async fn row_to_triple(row: &Row) -> Result<Triple> {
        let subject: String = row.get("subject").map_err(|e| {
            Neo4jError::query_error(format!("Failed to get subject from row: {}", e))
        })?;
        
        let predicate: String = row.get("predicate").map_err(|e| {
            Neo4jError::query_error(format!("Failed to get predicate from row: {}", e))
        })?;
        
        let object: String = row.get("object").map_err(|e| {
            Neo4jError::query_error(format!("Failed to get object from row: {}", e))
        })?;
        
        let neo4j_rel: neo4rs::Relation = row.get("relationship").map_err(|e| {
            Neo4jError::query_error(format!("Failed to get relationship from row: {}", e))
        })?;
        
        let properties: HashMap<String, Value> = neo4j_rel.properties()
            .map(|(k, v)| (k.to_string(), Self::neo4j_value_to_json(v)))
            .collect();
        
        let confidence = properties.get("confidence")
            .and_then(|v| v.as_f64());
        
        let source = properties.get("source")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let created_at = properties.get("created_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(chrono::Utc::now);
        
        let mut metadata = properties.clone();
        metadata.remove("confidence");
        metadata.remove("source");
        metadata.remove("created_at");
        
        Ok(Triple {
            subject,
            predicate,
            object,
            confidence,
            source,
            created_at,
            metadata,
        })
    }
    
    /// Helper to convert Neo4j value to JSON value
    fn neo4j_value_to_json(value: &neo4rs::BoltType) -> Value {
        match value {
            neo4rs::BoltType::String(s) => Value::String(s.value().to_string()),
            neo4rs::BoltType::Integer(i) => Value::Number(serde_json::Number::from(i.value())),
            neo4rs::BoltType::Float(f) => {
                if let Some(num) = serde_json::Number::from_f64(f.value()) {
                    Value::Number(num)
                } else {
                    Value::Null
                }
            }
            neo4rs::BoltType::Boolean(b) => Value::Bool(b.value()),
            neo4rs::BoltType::Null(_) => Value::Null,
            neo4rs::BoltType::List(list) => {
                Value::Array(list.value().iter().map(Self::neo4j_value_to_json).collect())
            }
            neo4rs::BoltType::Map(map) => {
                let mut obj = serde_json::Map::new();
                for (k, v) in map.value() {
                    obj.insert(k.to_string(), Self::neo4j_value_to_json(v));
                }
                Value::Object(obj)
            }
            _ => Value::Null, // For other types we don't handle
        }
    }
}

#[async_trait]
impl NodeOperations for GraphOperationsImpl {
    async fn create_node(&self, node: &Node) -> Result<String> {
        debug!("Creating node with labels: {:?}", node.labels);
        
        let node_id = node.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let mut node_with_id = node.clone();
        node_with_id.id = Some(node_id.clone());
        node_with_id.properties.insert("id".to_string(), Value::String(node_id.clone()));
        
        let query = CypherQueryBuilder::create()
            .create_node(&node_with_id, "n")
            .return_clause("n.id as node_id")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let created_id: String = row.get("node_id").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get created node id: {}", e))
            })?;
            info!("Successfully created node with id: {}", created_id);
            Ok(created_id)
        } else {
            Err(Neo4jError::query_error("Failed to create node"))
        }
    }
    
    async fn get_node(&self, id: &str) -> Result<Option<Node>> {
        debug!("Getting node with id: {}", id);
        
        let query = CypherQueryBuilder::match_query()
            .add_part("MATCH (n)")
            .add_part("WHERE n.id = $node_id")
            .add_parameter("node_id", Value::String(id.to_string()))
            .return_clause("n")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let node = Self::row_to_node(&row).await?;
            Ok(Some(node))
        } else {
            Ok(None)
        }
    }
    
    async fn update_node(&self, id: &str, properties: HashMap<String, Value>) -> Result<bool> {
        debug!("Updating node {} with properties: {:?}", id, properties);
        
        let query = CypherQueryBuilder::update()
            .add_part("MATCH (n)")
            .add_part("WHERE n.id = $node_id")
            .add_parameter("node_id", Value::String(id.to_string()))
            .add_part("SET n += $properties")
            .add_parameter("properties", serde_json::to_value(properties)?)
            .return_clause("count(n) as updated_count")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("updated_count").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get update count: {}", e))
            })?;
            Ok(count > 0)
        } else {
            Ok(false)
        }
    }
    
    async fn delete_node(&self, id: &str) -> Result<bool> {
        debug!("Deleting node with id: {}", id);
        
        let query = CypherQueryBuilder::delete()
            .add_part("MATCH (n)")
            .add_part("WHERE n.id = $node_id")
            .add_parameter("node_id", Value::String(id.to_string()))
            .add_part("DETACH DELETE n")
            .add_part("RETURN count(n) as deleted_count")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("deleted_count").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get delete count: {}", e))
            })?;
            Ok(count > 0)
        } else {
            Ok(false)
        }
    }
    
    async fn find_nodes_by_label(&self, label: &str, params: Option<QueryParams>) -> Result<Vec<Node>> {
        debug!("Finding nodes with label: {}", label);
        
        let mut query_builder = CypherQueryBuilder::match_query()
            .add_part(&format!("MATCH (n:{})", label))
            .return_clause("n");
        
        if let Some(params) = params {
            query_builder = query_builder.apply_params(&params);
        }
        
        let query = query_builder.build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        let mut nodes = Vec::new();
        while let Some(row) = result.next().await? {
            let node = Self::row_to_node(&row).await?;
            nodes.push(node);
        }
        
        info!("Found {} nodes with label: {}", nodes.len(), label);
        Ok(nodes)
    }
    
    async fn find_nodes_by_properties(&self, properties: HashMap<String, Value>, params: Option<QueryParams>) -> Result<Vec<Node>> {
        debug!("Finding nodes with properties: {:?}", properties);
        
        let labels = vec![];
        let mut query_builder = CypherQueryBuilder::match_query()
            .match_node(&labels, &properties, "n")
            .return_clause("n");
        
        if let Some(params) = params {
            query_builder = query_builder.apply_params(&params);
        }
        
        let query = query_builder.build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        let mut nodes = Vec::new();
        while let Some(row) = result.next().await? {
            let node = Self::row_to_node(&row).await?;
            nodes.push(node);
        }
        
        info!("Found {} nodes with specified properties", nodes.len());
        Ok(nodes)
    }
    
    async fn count_nodes(&self) -> Result<u64> {
        let query = CypherQueryBuilder::count_nodes().build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("node_count").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get node count: {}", e))
            })?;
            Ok(count as u64)
        } else {
            Ok(0)
        }
    }
}

#[async_trait]
impl RelationshipOperations for GraphOperationsImpl {
    async fn create_relationship(&self, relationship: &Relationship) -> Result<String> {
        debug!("Creating relationship of type: {}", relationship.relationship_type);
        
        let rel_id = relationship.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let query = CypherQueryBuilder::create()
            .add_part("MATCH (start) WHERE start.id = $start_id")
            .add_parameter("start_id", Value::String(relationship.start_node_id.clone()))
            .add_part("MATCH (end) WHERE end.id = $end_id")
            .add_parameter("end_id", Value::String(relationship.end_node_id.clone()))
            .add_part(&format!("CREATE (start)-[r:{} $rel_props]->(end)", relationship.relationship_type))
            .add_parameter("rel_props", serde_json::to_value(&relationship.properties)?)
            .return_clause("id(r) as rel_id")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let created_id: i64 = row.get("rel_id").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get created relationship id: {}", e))
            })?;
            let id_str = created_id.to_string();
            info!("Successfully created relationship with id: {}", id_str);
            Ok(id_str)
        } else {
            Err(Neo4jError::query_error("Failed to create relationship"))
        }
    }
    
    async fn get_relationship(&self, id: &str) -> Result<Option<Relationship>> {
        debug!("Getting relationship with id: {}", id);
        
        let query = CypherQueryBuilder::match_query()
            .add_part("MATCH ()-[r]->()")
            .add_part("WHERE id(r) = $rel_id")
            .add_parameter("rel_id", Value::String(id.to_string()))
            .return_clause("r")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let relationship = Self::row_to_relationship(&row).await?;
            Ok(Some(relationship))
        } else {
            Ok(None)
        }
    }
    
    async fn update_relationship(&self, id: &str, properties: HashMap<String, Value>) -> Result<bool> {
        debug!("Updating relationship {} with properties: {:?}", id, properties);
        
        let query = CypherQueryBuilder::update()
            .add_part("MATCH ()-[r]->()")
            .add_part("WHERE id(r) = $rel_id")
            .add_parameter("rel_id", Value::String(id.to_string()))
            .add_part("SET r += $properties")
            .add_parameter("properties", serde_json::to_value(properties)?)
            .return_clause("count(r) as updated_count")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("updated_count").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get update count: {}", e))
            })?;
            Ok(count > 0)
        } else {
            Ok(false)
        }
    }
    
    async fn delete_relationship(&self, id: &str) -> Result<bool> {
        debug!("Deleting relationship with id: {}", id);
        
        let query = CypherQueryBuilder::delete()
            .add_part("MATCH ()-[r]->()")
            .add_part("WHERE id(r) = $rel_id")
            .add_parameter("rel_id", Value::String(id.to_string()))
            .add_part("DELETE r")
            .add_part("RETURN count(r) as deleted_count")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("deleted_count").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get delete count: {}", e))
            })?;
            Ok(count > 0)
        } else {
            Ok(false)
        }
    }
    
    async fn find_relationships_by_type(&self, rel_type: &str, params: Option<QueryParams>) -> Result<Vec<Relationship>> {
        debug!("Finding relationships of type: {}", rel_type);
        
        let mut query_builder = CypherQueryBuilder::match_query()
            .add_part(&format!("MATCH ()-[r:{}]->()", rel_type))
            .return_clause("r");
        
        if let Some(params) = params {
            query_builder = query_builder.apply_params(&params);
        }
        
        let query = query_builder.build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        let mut relationships = Vec::new();
        while let Some(row) = result.next().await? {
            let relationship = Self::row_to_relationship(&row).await?;
            relationships.push(relationship);
        }
        
        info!("Found {} relationships of type: {}", relationships.len(), rel_type);
        Ok(relationships)
    }
    
    async fn find_relationships_between_nodes(&self, start_node_id: &str, end_node_id: &str) -> Result<Vec<Relationship>> {
        debug!("Finding relationships between nodes: {} -> {}", start_node_id, end_node_id);
        
        let query = CypherQueryBuilder::match_query()
            .add_part("MATCH (start)-[r]->(end)")
            .add_part("WHERE start.id = $start_id AND end.id = $end_id")
            .add_parameter("start_id", Value::String(start_node_id.to_string()))
            .add_parameter("end_id", Value::String(end_node_id.to_string()))
            .return_clause("r")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        
        let mut relationships = Vec::new();
        while let Some(row) = result.next().await? {
            let relationship = Self::row_to_relationship(&row).await?;
            relationships.push(relationship);
        }
        
        info!("Found {} relationships between specified nodes", relationships.len());
        Ok(relationships)
    }
    
    async fn count_relationships(&self) -> Result<u64> {
        let query = CypherQueryBuilder::count_relationships().build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        if let Some(row) = result.next().await? {
            let count: i64 = row.get("relationship_count").map_err(|e| {
                Neo4jError::query_error(format!("Failed to get relationship count: {}", e))
            })?;
            Ok(count as u64)
        } else {
            Ok(0)
        }
    }
}

#[async_trait]
impl GraphOperations for GraphOperationsImpl {
    async fn import_graph_data(&self, data: &GraphData) -> Result<ImportResult> {
        let start_time = std::time::Instant::now();
        info!("Starting graph data import: {} nodes, {} relationships, {} triples", 
              data.node_count(), data.relationship_count(), data.triple_count());
        
        let mut nodes_created = 0u64;
        let mut relationships_created = 0u64;
        let mut triples_processed = 0u64;
        let mut errors = Vec::new();
        
        // Import nodes
        for node in &data.nodes {
            match self.create_node(node).await {
                Ok(_) => nodes_created += 1,
                Err(e) => {
                    error!("Failed to create node: {}", e);
                    errors.push(format!("Node creation failed: {}", e));
                }
            }
        }
        
        // Import relationships
        for relationship in &data.relationships {
            match self.create_relationship(relationship).await {
                Ok(_) => relationships_created += 1,
                Err(e) => {
                    error!("Failed to create relationship: {}", e);
                    errors.push(format!("Relationship creation failed: {}", e));
                }
            }
        }
        
        // Import triples
        if !data.triples.is_empty() {
            match self.store_triples_batch(&data.triples).await {
                Ok(batch_result) => {
                    triples_processed = batch_result.successful_items as u64;
                    errors.extend(batch_result.errors);
                }
                Err(e) => {
                    error!("Failed to store triples batch: {}", e);
                    errors.push(format!("Triple batch storage failed: {}", e));
                }
            }
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        info!("Graph import completed in {}ms: {} nodes, {} relationships, {} triples, {} errors",
              duration_ms, nodes_created, relationships_created, triples_processed, errors.len());
        
        Ok(ImportResult {
            nodes_created,
            relationships_created,
            triples_processed,
            errors,
            duration_ms,
        })
    }
    
    async fn export_graph_data(&self, params: Option<QueryParams>) -> Result<GraphData> {
        info!("Starting graph data export");
        
        let mut graph_data = GraphData::new();
        
        // Export nodes
        let nodes = self.find_nodes_by_label("", params.clone()).await.unwrap_or_default();
        graph_data.nodes = nodes;
        
        // Export relationships
        let relationships = self.find_relationships_by_type("", params.clone()).await.unwrap_or_default();
        graph_data.relationships = relationships;
        
        // Export triples
        let triples = self.get_triples(params).await.unwrap_or_default();
        graph_data.triples = triples;
        
        info!("Graph export completed: {} nodes, {} relationships, {} triples",
              graph_data.node_count(), graph_data.relationship_count(), graph_data.triple_count());
        
        Ok(graph_data)
    }
    
    async fn store_triple(&self, triple: &Triple) -> Result<()> {
        debug!("Storing triple: {} -> {} -> {}", triple.subject, triple.predicate, triple.object);
        
        let query = CypherQueryBuilder::store_triple(triple).build()?;
        let mut _result = self.connection_manager.execute_query(query).await?;
        
        info!("Successfully stored triple");
        Ok(())
    }
    
    async fn store_triples_batch(&self, triples: &[Triple]) -> Result<BatchResult> {
        let start_time = std::time::Instant::now();
        info!("Starting batch storage of {} triples", triples.len());
        
        let total_items = triples.len();
        let mut successful_items = 0;
        let mut errors = Vec::new();
        
        // Process in smaller batches to avoid memory issues
        const BATCH_SIZE: usize = 100;
        
        for chunk in triples.chunks(BATCH_SIZE) {
            for triple in chunk {
                match self.store_triple(triple).await {
                    Ok(_) => successful_items += 1,
                    Err(e) => {
                        error!("Failed to store triple {}->{}->{}: {}", 
                               triple.subject, triple.predicate, triple.object, e);
                        errors.push(format!("Triple storage failed: {}", e));
                    }
                }
            }
        }
        
        let failed_items = total_items - successful_items;
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        info!("Batch storage completed in {}ms: {}/{} successful, {} failed",
              duration_ms, successful_items, total_items, failed_items);
        
        Ok(BatchResult {
            total_items,
            successful_items,
            failed_items,
            errors,
            duration_ms,
        })
    }
    
    async fn get_triples(&self, params: Option<QueryParams>) -> Result<Vec<Triple>> {
        debug!("Getting triples with params: {:?}", params);
        
        let mut query_builder = CypherQueryBuilder::get_triples();
        
        if let Some(params) = params {
            query_builder = query_builder.apply_params(&params);
        }
        
        let query = query_builder.build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        let mut triples = Vec::new();
        while let Some(row) = result.next().await? {
            let triple = Self::row_to_triple(&row).await?;
            triples.push(triple);
        }
        
        info!("Retrieved {} triples", triples.len());
        Ok(triples)
    }
    
    async fn search_triples(&self, subject: Option<&str>, predicate: Option<&str>, object: Option<&str>) -> Result<Vec<Triple>> {
        debug!("Searching triples: subject={:?}, predicate={:?}, object={:?}", subject, predicate, object);
        
        let query = TripleQueryBuilder::search_triples(subject, predicate, object).build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        let mut triples = Vec::new();
        while let Some(row) = result.next().await? {
            let triple = Self::row_to_triple(&row).await?;
            triples.push(triple);
        }
        
        info!("Found {} matching triples", triples.len());
        Ok(triples)
    }
    
    async fn delete_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        debug!("Deleting triple: {} -> {} -> {}", subject, predicate, object);
        
        let query = CypherQueryBuilder::delete_triple(subject, predicate, object).build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        // Check if any rows were affected
        let deleted = result.next().await?.is_some();
        
        if deleted {
            info!("Successfully deleted triple");
        } else {
            warn!("No triple found to delete");
        }
        
        Ok(deleted)
    }
    
    async fn get_graph_statistics(&self) -> Result<GraphStatistics> {
        debug!("Getting graph statistics");
        
        let query = CypherQueryBuilder::get_graph_stats().build()?;
        let mut result = self.connection_manager.execute_query(query).await?;
        
        let (node_count, relationship_count) = if let Some(row) = result.next().await? {
            let nodes: i64 = row.get("nodes").unwrap_or(0);
            let relationships: i64 = row.get("relationships").unwrap_or(0);
            (nodes as u64, relationships as u64)
        } else {
            (0, 0)
        };
        
        // Get label counts
        let label_query = CypherQueryBuilder::new()
            .add_part("MATCH (n)")
            .add_part("UNWIND labels(n) as label")
            .add_part("RETURN label, count(*) as count")
            .build()?;
        
        let mut label_result = self.connection_manager.execute_query(label_query).await?;
        let mut label_counts = HashMap::new();
        
        while let Some(row) = label_result.next().await? {
            let label: String = row.get("label").unwrap_or_default();
            let count: i64 = row.get("count").unwrap_or(0);
            label_counts.insert(label, count as u64);
        }
        
        // Get relationship type counts
        let rel_type_query = CypherQueryBuilder::new()
            .add_part("MATCH ()-[r]->()")
            .add_part("RETURN type(r) as rel_type, count(*) as count")
            .build()?;
        
        let mut rel_type_result = self.connection_manager.execute_query(rel_type_query).await?;
        let mut relationship_type_counts = HashMap::new();
        
        while let Some(row) = rel_type_result.next().await? {
            let rel_type: String = row.get("rel_type").unwrap_or_default();
            let count: i64 = row.get("count").unwrap_or(0);
            relationship_type_counts.insert(rel_type, count as u64);
        }
        
        let stats = GraphStatistics {
            node_count,
            relationship_count,
            label_counts,
            relationship_type_counts,
        };
        
        info!("Graph statistics: {} nodes, {} relationships", node_count, relationship_count);
        Ok(stats)
    }
    
    async fn clear_graph(&self) -> Result<()> {
        warn!("Clearing all graph data - this operation cannot be undone!");
        
        let query = CypherQueryBuilder::clear_all().build()?;
        let mut _result = self.connection_manager.execute_query(query).await?;
        
        info!("Graph cleared successfully");
        Ok(())
    }
}
