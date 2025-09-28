//! Cypher query builder for Neo4j operations

use crate::neo4j::models::{Node, Relationship, Triple, QueryParams};
use crate::neo4j::error::{Neo4jError, Result};
use std::collections::HashMap;
use std::fmt;
use serde_json::Value;

/// Trait for building Cypher queries
pub trait QueryBuilder {
    fn build(&self) -> Result<neo4rs::Query>;
    fn to_cypher(&self) -> String;
}

/// Cypher query builder for knowledge graph operations
#[derive(Debug, Clone)]
pub struct CypherQueryBuilder {
    query_parts: Vec<String>,
    parameters: HashMap<String, Value>,
    query_type: QueryType,
}

#[derive(Debug, Clone, PartialEq)]
enum QueryType {
    Create,
    Match,
    Merge,
    Delete,
    Update,
    Custom,
}

impl CypherQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            query_parts: Vec::new(),
            parameters: HashMap::new(),
            query_type: QueryType::Custom,
        }
    }
    
    /// Create a CREATE query
    pub fn create() -> Self {
        let mut builder = Self::new();
        builder.query_type = QueryType::Create;
        builder
    }
    
    /// Create a MATCH query
    pub fn match_query() -> Self {
        let mut builder = Self::new();
        builder.query_type = QueryType::Match;
        builder
    }
    
    /// Create a MERGE query
    pub fn merge() -> Self {
        let mut builder = Self::new();
        builder.query_type = QueryType::Merge;
        builder
    }
    
    /// Create a DELETE query
    pub fn delete() -> Self {
        let mut builder = Self::new();
        builder.query_type = QueryType::Delete;
        builder
    }
    
    /// Create an UPDATE query
    pub fn update() -> Self {
        let mut builder = Self::new();
        builder.query_type = QueryType::Update;
        builder
    }
    
    /// Add a custom query part
    pub fn add_part<S: Into<String>>(mut self, part: S) -> Self {
        self.query_parts.push(part.into());
        self
    }
    
    /// Add a parameter
    pub fn add_parameter<S: Into<String>>(mut self, key: S, value: Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }
    
    /// Create a node in the graph
    pub fn create_node(mut self, node: &Node, var_name: &str) -> Self {
        let labels = if node.labels.is_empty() {
            String::new()
        } else {
            format!(":{}", node.labels.join(":"))
        };
        
        let param_name = format!("{}_props", var_name);
        self = self.add_parameter(param_name.clone(), serde_json::to_value(&node.properties).unwrap());
        
        let query_part = format!("CREATE ({}{}{{{}}})", var_name, labels, format!("${}", param_name));
        self.query_parts.push(query_part);
        self.query_type = QueryType::Create;
        self
    }
    
    /// Create a relationship in the graph
    pub fn create_relationship(mut self, rel: &Relationship, start_var: &str, end_var: &str, rel_var: &str) -> Self {
        let param_name = format!("{}_props", rel_var);
        self = self.add_parameter(param_name.clone(), serde_json::to_value(&rel.properties).unwrap());
        
        let query_part = format!(
            "CREATE ({})-[{}:{}{{{}}}]->({})",
            start_var,
            rel_var,
            rel.relationship_type,
            format!("${}", param_name),
            end_var
        );
        self.query_parts.push(query_part);
        self.query_type = QueryType::Create;
        self
    }
    
    /// Match nodes by labels and properties
    pub fn match_node(mut self, labels: &[String], properties: &HashMap<String, Value>, var_name: &str) -> Self {
        let labels_str = if labels.is_empty() {
            String::new()
        } else {
            format!(":{}", labels.join(":"))
        };
        
        let mut query_part = format!("MATCH ({}{})", var_name, labels_str);
        
        if !properties.is_empty() {
            let mut conditions = Vec::new();
            for (key, value) in properties {
                let param_name = format!("{}_{}_{}", var_name, key, self.parameters.len());
                self = self.add_parameter(param_name.clone(), value.clone());
                conditions.push(format!("{}.{} = ${}", var_name, key, param_name));
            }
            query_part = format!("{} WHERE {}", query_part, conditions.join(" AND "));
        }
        
        self.query_parts.push(query_part);
        self.query_type = QueryType::Match;
        self
    }
    
    /// Match relationships
    pub fn match_relationship(mut self, rel_type: &str, start_var: &str, end_var: &str, rel_var: &str) -> Self {
        let query_part = format!("MATCH ({})-[{}:{}]->({}) ", start_var, rel_var, rel_type, end_var);
        self.query_parts.push(query_part);
        self.query_type = QueryType::Match;
        self
    }
    
    /// Add WHERE clause
    pub fn where_clause<S: Into<String>>(mut self, condition: S) -> Self {
        self.query_parts.push(format!("WHERE {}", condition.into()));
        self
    }
    
    /// Add RETURN clause
    pub fn return_clause<S: Into<String>>(mut self, return_items: S) -> Self {
        self.query_parts.push(format!("RETURN {}", return_items.into()));
        self
    }
    
    /// Add ORDER BY clause
    pub fn order_by<S: Into<String>>(mut self, order_by: S) -> Self {
        self.query_parts.push(format!("ORDER BY {}", order_by.into()));
        self
    }
    
    /// Add LIMIT clause
    pub fn limit(mut self, limit: usize) -> Self {
        self.query_parts.push(format!("LIMIT {}", limit));
        self
    }
    
    /// Add SKIP clause
    pub fn skip(mut self, skip: usize) -> Self {
        self.query_parts.push(format!("SKIP {}", skip));
        self
    }
    
    /// Apply query parameters
    pub fn apply_params(mut self, params: &QueryParams) -> Self {
        if let Some(limit) = params.limit {
            self = self.limit(limit);
        }
        
        if let Some(offset) = params.offset {
            self = self.skip(offset);
        }
        
        if let Some(ref order_by) = params.order_by {
            self = self.order_by(order_by);
        }
        
        // Apply filters as WHERE conditions
        if !params.filters.is_empty() {
            let mut conditions = Vec::new();
            for (key, value) in &params.filters {
                let param_name = format!("filter_{}_{}", key, self.parameters.len());
                self = self.add_parameter(param_name.clone(), value.clone());
                conditions.push(format!("{} = ${}", key, param_name));
            }
            self = self.where_clause(conditions.join(" AND "));
        }
        
        self
    }
    
    /// Build query for triple storage
    pub fn store_triple(triple: &Triple) -> Self {
        let mut builder = Self::new();
        
        // Create subject node
        builder = builder
            .add_part("MERGE (s:Entity {name: $subject_name})")
            .add_parameter("subject_name", Value::String(triple.subject.clone()));
        
        // Create object node
        builder = builder
            .add_part("MERGE (o:Entity {name: $object_name})")
            .add_parameter("object_name", Value::String(triple.object.clone()));
        
        // Create relationship
        let mut rel_props = HashMap::new();
        rel_props.insert("created_at".to_string(), Value::String(triple.created_at.to_rfc3339()));
        
        if let Some(confidence) = triple.confidence {
            rel_props.insert("confidence".to_string(), 
                Value::Number(serde_json::Number::from_f64(confidence).unwrap()));
        }
        
        if let Some(ref source) = triple.source {
            rel_props.insert("source".to_string(), Value::String(source.clone()));
        }
        
        for (key, value) in &triple.metadata {
            rel_props.insert(key.clone(), value.clone());
        }
        
        builder = builder
            .add_part(&format!("MERGE (s)-[r:{}]->(o)", triple.predicate))
            .add_part("SET r += $rel_props")
            .add_parameter("rel_props", serde_json::to_value(rel_props).unwrap());
        
        builder.query_type = QueryType::Merge;
        builder
    }
    
    /// Build query to retrieve triples
    pub fn get_triples() -> Self {
        Self::new()
            .add_part("MATCH (s:Entity)-[r]->(o:Entity)")
            .add_part("RETURN s.name as subject, type(r) as predicate, o.name as object, r as relationship")
    }
    
    /// Build query to get triples by subject
    pub fn get_triples_by_subject(subject: &str) -> Self {
        Self::new()
            .add_part("MATCH (s:Entity)-[r]->(o:Entity)")
            .add_part("WHERE s.name = $subject")
            .add_parameter("subject", Value::String(subject.to_string()))
            .add_part("RETURN s.name as subject, type(r) as predicate, o.name as object, r as relationship")
    }
    
    /// Build query to get triples by predicate
    pub fn get_triples_by_predicate(predicate: &str) -> Self {
        Self::new()
            .add_part(&format!("MATCH (s:Entity)-[r:{}]->(o:Entity)", predicate))
            .add_part("RETURN s.name as subject, type(r) as predicate, o.name as object, r as relationship")
    }
    
    /// Build query to delete triple
    pub fn delete_triple(subject: &str, predicate: &str, object: &str) -> Self {
        Self::new()
            .add_part(&format!("MATCH (s:Entity {{name: $subject}})-[r:{}]->(o:Entity {{name: $object}})", predicate))
            .add_parameter("subject", Value::String(subject.to_string()))
            .add_parameter("object", Value::String(object.to_string()))
            .add_part("DELETE r")
    }
    
    /// Build query to count nodes
    pub fn count_nodes() -> Self {
        Self::new()
            .add_part("MATCH (n)")
            .add_part("RETURN count(n) as node_count")
    }
    
    /// Build query to count relationships
    pub fn count_relationships() -> Self {
        Self::new()
            .add_part("MATCH ()-[r]->()")
            .add_part("RETURN count(r) as relationship_count")
    }
    
    /// Build query to get graph statistics
    pub fn get_graph_stats() -> Self {
        Self::new()
            .add_part("MATCH (n)")
            .add_part("OPTIONAL MATCH ()-[r]->()")
            .add_part("RETURN count(DISTINCT n) as nodes, count(r) as relationships")
    }
    
    /// Build query to clear all data (use with caution)
    pub fn clear_all() -> Self {
        Self::new()
            .add_part("MATCH (n)")
            .add_part("DETACH DELETE n")
    }
}

impl QueryBuilder for CypherQueryBuilder {
    fn build(&self) -> Result<neo4rs::Query> {
        let cypher = self.to_cypher();
        
        if cypher.trim().is_empty() {
            return Err(Neo4jError::query_error("Empty query"));
        }
        
        let mut query = neo4rs::query(&cypher);
        
        // Add parameters
        for (key, value) in &self.parameters {
            match value {
                Value::String(s) => query = query.param(key, s.clone()),
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        query = query.param(key, i);
                    } else if let Some(f) = n.as_f64() {
                        query = query.param(key, f);
                    }
                }
                Value::Bool(b) => query = query.param(key, *b),
                Value::Array(arr) => {
                    // Convert array to string representation
                    query = query.param(key, serde_json::to_string(arr).unwrap());
                }
                Value::Object(obj) => {
                    // Convert object to string representation
                    query = query.param(key, serde_json::to_string(obj).unwrap());
                }
                Value::Null => query = query.param(key, Option::<String>::None),
            }
        }
        
        Ok(query)
    }
    
    fn to_cypher(&self) -> String {
        self.query_parts.join(" ")
    }
}

impl Default for CypherQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CypherQueryBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_cypher())
    }
}

/// Specialized query builders for common operations
pub struct TripleQueryBuilder;

impl TripleQueryBuilder {
    /// Create batch insert query for triples
    pub fn batch_insert_triples(triples: &[Triple]) -> CypherQueryBuilder {
        let mut builder = CypherQueryBuilder::new();
        
        builder = builder.add_part("UNWIND $triples as triple");
        builder = builder.add_part("MERGE (s:Entity {name: triple.subject})");
        builder = builder.add_part("MERGE (o:Entity {name: triple.object})");
        builder = builder.add_part("WITH s, o, triple");
        builder = builder.add_part("CALL apoc.create.relationship(s, triple.predicate, triple.properties, o) YIELD rel");
        builder = builder.add_part("RETURN count(rel) as created_relationships");
        
        // Convert triples to parameter format
        let triple_data: Vec<Value> = triples
            .iter()
            .map(|t| {
                let mut props = t.metadata.clone();
                props.insert("created_at".to_string(), Value::String(t.created_at.to_rfc3339()));
                
                if let Some(confidence) = t.confidence {
                    props.insert("confidence".to_string(), 
                        Value::Number(serde_json::Number::from_f64(confidence).unwrap()));
                }
                
                if let Some(ref source) = t.source {
                    props.insert("source".to_string(), Value::String(source.clone()));
                }
                
                serde_json::json!({
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "properties": props
                })
            })
            .collect();
        
        builder.add_parameter("triples", Value::Array(triple_data))
    }
    
    /// Search triples by pattern
    pub fn search_triples(subject: Option<&str>, predicate: Option<&str>, object: Option<&str>) -> CypherQueryBuilder {
        let mut builder = CypherQueryBuilder::new();
        let mut conditions = Vec::new();
        
        builder = builder.add_part("MATCH (s:Entity)-[r]->(o:Entity)");
        
        if let Some(subj) = subject {
            conditions.push("s.name = $subject");
            builder = builder.add_parameter("subject", Value::String(subj.to_string()));
        }
        
        if let Some(pred) = predicate {
            builder = CypherQueryBuilder::new()
                .add_part(&format!("MATCH (s:Entity)-[r:{}]->(o:Entity)", pred));
        }
        
        if let Some(obj) = object {
            conditions.push("o.name = $object");
            builder = builder.add_parameter("object", Value::String(obj.to_string()));
        }
        
        if !conditions.is_empty() {
            builder = builder.add_part(&format!("WHERE {}", conditions.join(" AND ")));
        }
        
        builder.add_part("RETURN s.name as subject, type(r) as predicate, o.name as object, r as relationship")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    #[test]
    fn test_create_node_query() {
        let node = Node::new()
            .with_label("Person")
            .with_property("name", Value::String("Alice".to_string()))
            .with_property("age", Value::Number(serde_json::Number::from(30)));
        
        let builder = CypherQueryBuilder::create()
            .create_node(&node, "n");
        
        let cypher = builder.to_cypher();
        assert!(cypher.contains("CREATE (n:Person"));
        assert!(cypher.contains("$n_props"));
    }
    
    #[test]
    fn test_triple_storage_query() {
        let triple = Triple::new("Alice", "knows", "Bob")
            .with_confidence(0.9)
            .with_source("test");
        
        let builder = CypherQueryBuilder::store_triple(&triple);
        let cypher = builder.to_cypher();
        
        assert!(cypher.contains("MERGE (s:Entity"));
        assert!(cypher.contains("MERGE (o:Entity"));
        assert!(cypher.contains("MERGE (s)-[r:knows]->(o)"));
    }
    
    #[test]
    fn test_query_builder_build() {
        let builder = CypherQueryBuilder::new()
            .add_part("MATCH (n)")
            .add_part("RETURN n")
            .add_parameter("test", Value::String("value".to_string()));
        
        let query = builder.build().unwrap();
        // Neo4rs query doesn't expose internal structure for testing,
        // but we can verify it builds without error
    }
}
