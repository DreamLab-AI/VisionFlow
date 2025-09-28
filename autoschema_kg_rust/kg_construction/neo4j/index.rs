//! Index creation and management for Neo4j

use crate::neo4j::{
    connection::Neo4jConnectionManager,
    error::{Neo4jError, Result},
    query_builder::{CypherQueryBuilder, QueryBuilder},
};
use async_trait::async_trait;
use futures::StreamExt;
use neo4rs::Row;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use log::{debug, info, warn, error};
use std::time::Instant;

/// Types of indexes supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    /// Standard B-tree index
    BTree,
    /// Text search index
    Text,
    /// Full-text search index  
    FullText,
    /// Range index for numeric/temporal data
    Range,
    /// Point index for spatial data
    Point,
    /// Composite index on multiple properties
    Composite,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexType::BTree => write!(f, "BTREE"),
            IndexType::Text => write!(f, "TEXT"),
            IndexType::FullText => write!(f, "FULLTEXT"),
            IndexType::Range => write!(f, "RANGE"),
            IndexType::Point => write!(f, "POINT"),
            IndexType::Composite => write!(f, "BTREE"), // Composite uses BTREE type
        }
    }
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub name: String,
    pub index_type: IndexType,
    pub labels: Vec<String>,
    pub properties: Vec<String>,
    pub options: HashMap<String, String>,
}

impl IndexConfig {
    pub fn new<S: Into<String>>(name: S, index_type: IndexType) -> Self {
        Self {
            name: name.into(),
            index_type,
            labels: Vec::new(),
            properties: Vec::new(),
            options: HashMap::new(),
        }
    }
    
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.labels.push(label.into());
        self
    }
    
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }
    
    pub fn with_property<S: Into<String>>(mut self, property: S) -> Self {
        self.properties.push(property.into());
        self
    }
    
    pub fn with_properties(mut self, properties: Vec<String>) -> Self {
        self.properties = properties;
        self
    }
    
    pub fn with_option<S: Into<String>>(mut self, key: S, value: S) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
    
    /// Validate the index configuration
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Neo4jError::ConfigError("Index name cannot be empty".to_string()));
        }
        
        if self.labels.is_empty() {
            return Err(Neo4jError::ConfigError("At least one label is required for index".to_string()));
        }
        
        if self.properties.is_empty() {
            return Err(Neo4jError::ConfigError("At least one property is required for index".to_string()));
        }
        
        // Validate index type constraints
        match self.index_type {
            IndexType::FullText => {
                if self.properties.len() > 1 {
                    return Err(Neo4jError::ConfigError(
                        "Full-text indexes support multiple properties but require special syntax".to_string()
                    ));
                }
            }
            IndexType::Composite => {
                if self.properties.len() < 2 {
                    return Err(Neo4jError::ConfigError(
                        "Composite indexes require at least 2 properties".to_string()
                    ));
                }
            }
            _ => {}
        }
        
        Ok(())
    }
}

/// Information about an existing index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub name: String,
    pub index_type: String,
    pub entity_type: String, // NODE or RELATIONSHIP
    pub labels_or_types: Vec<String>,
    pub properties: Vec<String>,
    pub state: String, // ONLINE, POPULATING, FAILED
    pub provider: String,
    pub options: HashMap<String, String>,
}

/// Index usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    pub name: String,
    pub unique_values: Option<u64>,
    pub updates: u64,
    pub size: u64,
    pub estimated_selectivity: Option<f64>,
    pub last_read: Option<String>,
    pub last_write: Option<String>,
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstraintType {
    Unique,
    NodeKey,
    Exists,
}

impl std::fmt::Display for ConstraintType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintType::Unique => write!(f, "UNIQUE"),
            ConstraintType::NodeKey => write!(f, "NODE KEY"),
            ConstraintType::Exists => write!(f, "EXISTS"),
        }
    }
}

/// Constraint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintConfig {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub label: String,
    pub properties: Vec<String>,
}

impl ConstraintConfig {
    pub fn new<S: Into<String>>(name: S, constraint_type: ConstraintType, label: S) -> Self {
        Self {
            name: name.into(),
            constraint_type,
            label: label.into(),
            properties: Vec::new(),
        }
    }
    
    pub fn with_property<S: Into<String>>(mut self, property: S) -> Self {
        self.properties.push(property.into());
        self
    }
    
    pub fn with_properties(mut self, properties: Vec<String>) -> Self {
        self.properties = properties;
        self
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Neo4jError::ConfigError("Constraint name cannot be empty".to_string()));
        }
        
        if self.label.is_empty() {
            return Err(Neo4jError::ConfigError("Label is required for constraint".to_string()));
        }
        
        if self.properties.is_empty() {
            return Err(Neo4jError::ConfigError("At least one property is required for constraint".to_string()));
        }
        
        Ok(())
    }
}

/// Trait for index management operations
#[async_trait]
pub trait IndexManagement {
    async fn create_index(&self, config: &IndexConfig) -> Result<()>;
    async fn drop_index(&self, name: &str) -> Result<bool>;
    async fn list_indexes(&self) -> Result<Vec<IndexInfo>>;
    async fn get_index_info(&self, name: &str) -> Result<Option<IndexInfo>>;
    async fn get_index_statistics(&self, name: &str) -> Result<Option<IndexStatistics>>;
    async fn rebuild_index(&self, name: &str) -> Result<()>;
    async fn create_constraint(&self, config: &ConstraintConfig) -> Result<()>;
    async fn drop_constraint(&self, name: &str) -> Result<bool>;
    async fn list_constraints(&self) -> Result<Vec<ConstraintConfig>>;
}

/// Index manager implementation
pub struct IndexManager {
    connection_manager: Neo4jConnectionManager,
}

impl IndexManager {
    pub fn new(connection_manager: Neo4jConnectionManager) -> Self {
        Self { connection_manager }
    }
    
    /// Create standard indexes for knowledge graph entities
    pub async fn create_knowledge_graph_indexes(&self) -> Result<()> {
        info!("Creating standard knowledge graph indexes");
        
        let indexes = vec![
            // Entity name index
            IndexConfig::new("entity_name_index", IndexType::BTree)
                .with_label("Entity")
                .with_property("name"),
            
            // Entity ID index
            IndexConfig::new("entity_id_index", IndexType::BTree)
                .with_label("Entity")
                .with_property("id"),
            
            // Full-text search on entity names
            IndexConfig::new("entity_fulltext_index", IndexType::FullText)
                .with_label("Entity")
                .with_property("name"),
            
            // Triple source index
            IndexConfig::new("triple_source_index", IndexType::BTree)
                .with_label("Entity")
                .with_property("source"),
            
            // Confidence index for relationships
            IndexConfig::new("confidence_index", IndexType::Range)
                .with_label("Entity")
                .with_property("confidence"),
            
            // Created timestamp index
            IndexConfig::new("created_at_index", IndexType::Range)
                .with_label("Entity")
                .with_property("created_at"),
        ];
        
        let mut created_count = 0;
        let mut errors = Vec::new();
        
        for index_config in indexes {
            match self.create_index(&index_config).await {
                Ok(_) => {
                    created_count += 1;
                    info!("Created index: {}", index_config.name);
                }
                Err(e) => {
                    warn!("Failed to create index {}: {}", index_config.name, e);
                    errors.push(format!("Index {}: {}", index_config.name, e));
                }
            }
        }
        
        if !errors.is_empty() {
            warn!("Some indexes failed to create: {:?}", errors);
        }
        
        info!("Created {} knowledge graph indexes", created_count);
        Ok(())
    }
    
    /// Create standard constraints for knowledge graph
    pub async fn create_knowledge_graph_constraints(&self) -> Result<()> {
        info!("Creating standard knowledge graph constraints");
        
        let constraints = vec![
            // Unique entity names
            ConstraintConfig::new("entity_name_unique", ConstraintType::Unique, "Entity")
                .with_property("name"),
            
            // Unique entity IDs
            ConstraintConfig::new("entity_id_unique", ConstraintType::Unique, "Entity")
                .with_property("id"),
        ];
        
        let mut created_count = 0;
        let mut errors = Vec::new();
        
        for constraint_config in constraints {
            match self.create_constraint(&constraint_config).await {
                Ok(_) => {
                    created_count += 1;
                    info!("Created constraint: {}", constraint_config.name);
                }
                Err(e) => {
                    warn!("Failed to create constraint {}: {}", constraint_config.name, e);
                    errors.push(format!("Constraint {}: {}", constraint_config.name, e));
                }
            }
        }
        
        if !errors.is_empty() {
            warn!("Some constraints failed to create: {:?}", errors);
        }
        
        info!("Created {} knowledge graph constraints", created_count);
        Ok(())
    }
    
    /// Optimize indexes for knowledge graph queries
    pub async fn optimize_for_knowledge_graph(&self) -> Result<()> {
        info!("Optimizing indexes for knowledge graph queries");
        
        // Create indexes and constraints
        self.create_knowledge_graph_indexes().await?;
        self.create_knowledge_graph_constraints().await?;
        
        info!("Knowledge graph optimization completed");
        Ok(())
    }
    
    /// Parse index information from Neo4j row
    fn parse_index_info(row: &Row) -> Result<IndexInfo> {
        let name: String = row.get("name").map_err(|e| {
            Neo4jError::query_error(format!("Failed to get index name: {}", e))
        })?;
        
        let index_type: String = row.get("type").unwrap_or_default();
        let entity_type: String = row.get("entityType").unwrap_or_default();
        let state: String = row.get("state").unwrap_or_default();
        let provider: String = row.get("provider").unwrap_or_default();
        
        // Parse labels/types and properties - these might be arrays
        let labels_or_types = if let Ok(labels) = row.get::<Vec<String>>("labelsOrTypes") {
            labels
        } else {
            Vec::new()
        };
        
        let properties = if let Ok(props) = row.get::<Vec<String>>("properties") {
            props
        } else {
            Vec::new()
        };
        
        Ok(IndexInfo {
            name,
            index_type,
            entity_type,
            labels_or_types,
            properties,
            state,
            provider,
            options: HashMap::new(), // Neo4j doesn't expose options in SHOW INDEXES
        })
    }
}

#[async_trait]
impl IndexManagement for IndexManager {
    async fn create_index(&self, config: &IndexConfig) -> Result<()> {
        config.validate()?;
        
        debug!("Creating index: {} of type {:?}", config.name, config.index_type);
        
        let query = match config.index_type {
            IndexType::FullText => {
                // Full-text indexes have different syntax
                let labels = config.labels.join(", ");
                let properties = config.properties.join(", ");
                CypherQueryBuilder::new()
                    .add_part(&format!(
                        "CREATE FULLTEXT INDEX {} FOR (n:{}) ON EACH [{}]",
                        config.name, labels, properties
                    ))
                    .build()?
            }
            IndexType::Composite => {
                // Composite index on multiple properties
                let label = config.labels.first().ok_or_else(|| {
                    Neo4jError::ConfigError("At least one label required".to_string())
                })?;
                let properties: Vec<String> = config.properties.iter()
                    .map(|p| format!("n.{}", p))
                    .collect();
                
                CypherQueryBuilder::new()
                    .add_part(&format!(
                        "CREATE INDEX {} FOR (n:{}) ON ({})",
                        config.name, label, properties.join(", ")
                    ))
                    .build()?
            }
            _ => {
                // Standard single-property indexes
                let label = config.labels.first().ok_or_else(|| {
                    Neo4jError::ConfigError("At least one label required".to_string())
                })?;
                let property = config.properties.first().ok_or_else(|| {
                    Neo4jError::ConfigError("At least one property required".to_string())
                })?;
                
                CypherQueryBuilder::new()
                    .add_part(&format!(
                        "CREATE INDEX {} FOR (n:{}) ON (n.{})",
                        config.name, label, property
                    ))
                    .build()?
            }
        };
        
        let mut _result = self.connection_manager.execute_query(query).await?;
        info!("Successfully created index: {}", config.name);
        Ok(())
    }
    
    async fn drop_index(&self, name: &str) -> Result<bool> {
        debug!("Dropping index: {}", name);
        
        let query = CypherQueryBuilder::new()
            .add_part(&format!("DROP INDEX {} IF EXISTS", name))
            .build()?;
        
        let mut _result = self.connection_manager.execute_query(query).await?;
        info!("Successfully dropped index: {}", name);
        Ok(true)
    }
    
    async fn list_indexes(&self) -> Result<Vec<IndexInfo>> {
        debug!("Listing all indexes");
        
        let query = CypherQueryBuilder::new()
            .add_part("SHOW INDEXES")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        let mut indexes = Vec::new();
        
        while let Some(row) = result.next().await? {
            match Self::parse_index_info(&row) {
                Ok(index_info) => indexes.push(index_info),
                Err(e) => warn!("Failed to parse index info: {}", e),
            }
        }
        
        info!("Found {} indexes", indexes.len());
        Ok(indexes)
    }
    
    async fn get_index_info(&self, name: &str) -> Result<Option<IndexInfo>> {
        debug!("Getting info for index: {}", name);
        
        let indexes = self.list_indexes().await?;
        Ok(indexes.into_iter().find(|idx| idx.name == name))
    }
    
    async fn get_index_statistics(&self, name: &str) -> Result<Option<IndexStatistics>> {
        debug!("Getting statistics for index: {}", name);
        
        // Neo4j doesn't expose detailed index statistics via Cypher in community edition
        // This would require calling procedures that might not be available
        warn!("Index statistics not fully implemented - requires Neo4j Enterprise features");
        
        Ok(Some(IndexStatistics {
            name: name.to_string(),
            unique_values: None,
            updates: 0,
            size: 0,
            estimated_selectivity: None,
            last_read: None,
            last_write: None,
        }))
    }
    
    async fn rebuild_index(&self, name: &str) -> Result<()> {
        warn!("Index rebuild not supported in Neo4j Community Edition");
        
        // In enterprise edition, this would be:
        // CALL db.index.fulltext.drop(name)
        // followed by recreating the index
        
        Err(Neo4jError::IndexError(
            "Index rebuild requires Neo4j Enterprise Edition".to_string()
        ))
    }
    
    async fn create_constraint(&self, config: &ConstraintConfig) -> Result<()> {
        config.validate()?;
        
        debug!("Creating constraint: {} of type {:?}", config.name, config.constraint_type);
        
        let properties = if config.properties.len() == 1 {
            format!("n.{}", config.properties[0])
        } else {
            let props: Vec<String> = config.properties.iter()
                .map(|p| format!("n.{}", p))
                .collect();
            format!("({})", props.join(", "))
        };
        
        let query = match config.constraint_type {
            ConstraintType::Unique => {
                CypherQueryBuilder::new()
                    .add_part(&format!(
                        "CREATE CONSTRAINT {} FOR (n:{}) REQUIRE {} IS UNIQUE",
                        config.name, config.label, properties
                    ))
                    .build()?
            }
            ConstraintType::NodeKey => {
                CypherQueryBuilder::new()
                    .add_part(&format!(
                        "CREATE CONSTRAINT {} FOR (n:{}) REQUIRE {} IS NODE KEY",
                        config.name, config.label, properties
                    ))
                    .build()?
            }
            ConstraintType::Exists => {
                CypherQueryBuilder::new()
                    .add_part(&format!(
                        "CREATE CONSTRAINT {} FOR (n:{}) REQUIRE {} IS NOT NULL",
                        config.name, config.label, properties
                    ))
                    .build()?
            }
        };
        
        let mut _result = self.connection_manager.execute_query(query).await?;
        info!("Successfully created constraint: {}", config.name);
        Ok(())
    }
    
    async fn drop_constraint(&self, name: &str) -> Result<bool> {
        debug!("Dropping constraint: {}", name);
        
        let query = CypherQueryBuilder::new()
            .add_part(&format!("DROP CONSTRAINT {} IF EXISTS", name))
            .build()?;
        
        let mut _result = self.connection_manager.execute_query(query).await?;
        info!("Successfully dropped constraint: {}", name);
        Ok(true)
    }
    
    async fn list_constraints(&self) -> Result<Vec<ConstraintConfig>> {
        debug!("Listing all constraints");
        
        let query = CypherQueryBuilder::new()
            .add_part("SHOW CONSTRAINTS")
            .build()?;
        
        let mut result = self.connection_manager.execute_query(query).await?;
        let mut constraints = Vec::new();
        
        while let Some(row) = result.next().await? {
            // Parse constraint information from row
            if let Ok(name) = row.get::<String>("name") {
                let constraint_type = row.get::<String>("type").unwrap_or_default();
                let entity_type = row.get::<String>("entityType").unwrap_or_default();
                
                // Map string type to enum
                let constraint_type_enum = match constraint_type.as_str() {
                    "UNIQUENESS" => ConstraintType::Unique,
                    "NODE_KEY" => ConstraintType::NodeKey,
                    "NODE_PROPERTY_EXISTENCE" => ConstraintType::Exists,
                    _ => continue, // Skip unknown types
                };
                
                // Get label and properties (might need to parse from other fields)
                let label = entity_type;
                let properties = vec![]; // Would need to parse from constraint details
                
                constraints.push(ConstraintConfig {
                    name,
                    constraint_type: constraint_type_enum,
                    label,
                    properties,
                });
            }
        }
        
        info!("Found {} constraints", constraints.len());
        Ok(constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_index_config_creation() {
        let config = IndexConfig::new("test_index", IndexType::BTree)
            .with_label("Person")
            .with_property("name");
        
        assert_eq!(config.name, "test_index");
        assert_eq!(config.index_type, IndexType::BTree);
        assert!(config.labels.contains(&"Person".to_string()));
        assert!(config.properties.contains(&"name".to_string()));
    }
    
    #[test]
    fn test_index_config_validation() {
        let config = IndexConfig::new("", IndexType::BTree);
        assert!(config.validate().is_err());
        
        let config = IndexConfig::new("test", IndexType::BTree)
            .with_label("Person");
        assert!(config.validate().is_err()); // No properties
        
        let config = IndexConfig::new("test", IndexType::BTree)
            .with_label("Person")
            .with_property("name");
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_constraint_config() {
        let config = ConstraintConfig::new("unique_name", ConstraintType::Unique, "Person")
            .with_property("name");
        
        assert_eq!(config.name, "unique_name");
        assert_eq!(config.constraint_type, ConstraintType::Unique);
        assert_eq!(config.label, "Person");
        assert!(config.properties.contains(&"name".to_string()));
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_index_type_display() {
        assert_eq!(IndexType::BTree.to_string(), "BTREE");
        assert_eq!(IndexType::FullText.to_string(), "FULLTEXT");
        assert_eq!(IndexType::Range.to_string(), "RANGE");
    }
    
    #[test]
    fn test_constraint_type_display() {
        assert_eq!(ConstraintType::Unique.to_string(), "UNIQUE");
        assert_eq!(ConstraintType::NodeKey.to_string(), "NODE KEY");
        assert_eq!(ConstraintType::Exists.to_string(), "EXISTS");
    }
}
