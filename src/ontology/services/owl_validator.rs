// src/ontology/services/owl_validator.rs

//! Core service for OWL/RDF validation, reasoning, and graph mapping.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

// Re-export types from services module
pub use crate::services::owl_validator::{
    GraphEdge, GraphNode, PropertyGraph, RdfTriple, Severity, ValidationConfig, ValidationError,
    ValidationReport, Violation,
};

/// Mapping configuration loaded from mapping.toml
#[derive(Debug, Clone, Deserialize)]
pub struct MappingConfig {
    pub metadata: MappingMetadata,
    pub global: GlobalConfig,
    pub defaults: DefaultsConfig,
    pub namespaces: HashMap<String, String>,
    pub class_mappings: HashMap<String, ClassMapping>,
    pub object_property_mappings: HashMap<String, ObjectPropertyMapping>,
    pub data_property_mappings: HashMap<String, DataPropertyMapping>,
    pub iri_templates: IriTemplates,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MappingMetadata {
    pub title: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub created: String,
    pub last_modified: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlobalConfig {
    pub base_iri: String,
    pub default_vocabulary: String,
    pub version_iri: String,
    pub default_language: String,
    pub strict_mode: bool,
    pub auto_generate_inverses: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DefaultsConfig {
    pub default_node_class: String,
    pub default_edge_property: String,
    pub default_datatype: String,
    pub fallback_namespace: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClassMapping {
    pub owl_class: String,
    pub rdfs_label: String,
    pub rdfs_comment: String,
    #[serde(default)]
    pub rdfs_subclass_of: Vec<String>,
    #[serde(default)]
    pub equivalent_classes: Vec<String>,
    #[serde(default)]
    pub disjoint_with: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObjectPropertyMapping {
    pub owl_property: String,
    pub rdfs_label: String,
    pub rdfs_comment: String,
    pub rdfs_domain: PropertyDomain,
    pub rdfs_range: PropertyRange,
    #[serde(default)]
    pub owl_inverse_of: Option<String>,
    pub property_type: String,
    #[serde(default)]
    pub characteristics: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataPropertyMapping {
    pub owl_property: String,
    pub rdfs_label: String,
    pub rdfs_comment: String,
    pub rdfs_domain: PropertyDomain,
    pub rdfs_range: String,
    pub property_type: String,
    #[serde(default)]
    pub characteristics: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PropertyDomain {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PropertyRange {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct IriTemplates {
    pub nodes: HashMap<String, String>,
    pub edges: HashMap<String, String>,
    pub metadata: MetadataTemplates,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetadataTemplates {
    pub property: String,
    pub class: String,
}

/// The main service for ontology validation.
pub struct OwlValidatorService {
    mapping_config: Arc<MappingConfig>,
}

impl OwlValidatorService {
    /// Creates a new instance of the validation service.
    pub fn new() -> Result<Self> {
        let mapping_toml = std::fs::read_to_string("ontology/mapping.toml")
            .context("Failed to read ontology/mapping.toml")?;

        let mapping_config: MappingConfig =
            toml::from_str(&mapping_toml).context("Failed to parse mapping.toml")?;

        Ok(Self {
            mapping_config: Arc::new(mapping_config),
        })
    }

    /// Creates a new instance with a custom mapping configuration
    pub fn with_config(mapping_config: MappingConfig) -> Self {
        Self {
            mapping_config: Arc::new(mapping_config),
        }
    }

    /// Maps the property graph to RDF triples based on `mapping.toml`.
    pub fn map_graph_to_rdf(&self, graph: &PropertyGraph) -> Result<Vec<RdfTriple>> {
        let mut triples = Vec::new();

        // Process nodes
        for node in &graph.nodes {
            triples.extend(self.map_node_to_triples(node)?);
        }

        // Process edges
        for edge in &graph.edges {
            triples.extend(self.map_edge_to_triples(edge)?);
        }

        Ok(triples)
    }

    /// Maps a single node to RDF triples
    fn map_node_to_triples(&self, node: &GraphNode) -> Result<Vec<RdfTriple>> {
        let mut triples = Vec::new();

        // Generate node IRI using templates
        let node_iri = self.generate_node_iri(node)?;

        // Map node labels to OWL classes
        for label in &node.labels {
            if let Some(class_mapping) = self.mapping_config.class_mappings.get(label) {
                // Add rdf:type triple
                let owl_class_iri = self.expand_prefixed_iri(&class_mapping.owl_class)?;
                triples.push(RdfTriple {
                    subject: node_iri.clone(),
                    predicate: self.expand_prefixed_iri("rdf:type")?,
                    object: owl_class_iri,
                    is_literal: false,
                    datatype: None,
                    language: None,
                });
            } else {
                // Use default class if no mapping found
                triples.push(RdfTriple {
                    subject: node_iri.clone(),
                    predicate: self.expand_prefixed_iri("rdf:type")?,
                    object: self
                        .expand_prefixed_iri(&self.mapping_config.defaults.default_node_class)?,
                    is_literal: false,
                    datatype: None,
                    language: None,
                });
            }
        }

        // Map node properties to data properties
        for (prop_name, prop_value) in &node.properties {
            if let Some(data_prop_mapping) =
                self.mapping_config.data_property_mappings.get(prop_name)
            {
                let prop_iri = self.expand_prefixed_iri(&data_prop_mapping.owl_property)?;

                // Handle multi-valued properties
                let values = if prop_value.is_array() {
                    prop_value.as_array().unwrap().iter().collect()
                } else {
                    vec![prop_value]
                };

                for value in values {
                    let (object_str, datatype) =
                        self.serialize_literal_value(value, &data_prop_mapping.rdfs_range)?;

                    triples.push(RdfTriple {
                        subject: node_iri.clone(),
                        predicate: prop_iri.clone(),
                        object: object_str,
                        is_literal: true,
                        datatype: Some(datatype),
                        language: None,
                    });
                }
            }
        }

        Ok(triples)
    }

    /// Maps a single edge to RDF triples
    fn map_edge_to_triples(&self, edge: &GraphEdge) -> Result<Vec<RdfTriple>> {
        let mut triples = Vec::new();

        let source_iri = self.generate_node_iri_from_id(&edge.source)?;
        let target_iri = self.generate_node_iri_from_id(&edge.target)?;

        // Map edge relationship to object property
        if let Some(obj_prop_mapping) = self
            .mapping_config
            .object_property_mappings
            .get(&edge.relationship_type)
        {
            let prop_iri = self.expand_prefixed_iri(&obj_prop_mapping.owl_property)?;

            triples.push(RdfTriple {
                subject: source_iri.clone(),
                predicate: prop_iri,
                object: target_iri.clone(),
                is_literal: false,
                datatype: None,
                language: None,
            });

            // Handle inverse properties if auto-generation is enabled
            if self.mapping_config.global.auto_generate_inverses {
                if let Some(inverse_prop) = &obj_prop_mapping.owl_inverse_of {
                    let inverse_iri = self.expand_prefixed_iri(inverse_prop)?;
                    triples.push(RdfTriple {
                        subject: target_iri,
                        predicate: inverse_iri,
                        object: source_iri,
                        is_literal: false,
                        datatype: None,
                        language: None,
                    });
                }
            }
        } else {
            // Use default edge property
            let default_prop =
                self.expand_prefixed_iri(&self.mapping_config.defaults.default_edge_property)?;
            triples.push(RdfTriple {
                subject: source_iri,
                predicate: default_prop,
                object: target_iri,
                is_literal: false,
                datatype: None,
                language: None,
            });
        }

        // Map edge properties
        for (prop_name, prop_value) in &edge.properties {
            if let Some(data_prop_mapping) =
                self.mapping_config.data_property_mappings.get(prop_name)
            {
                let edge_iri = self.generate_edge_iri(edge)?;
                let prop_iri = self.expand_prefixed_iri(&data_prop_mapping.owl_property)?;

                let (object_str, datatype) =
                    self.serialize_literal_value(prop_value, &data_prop_mapping.rdfs_range)?;

                triples.push(RdfTriple {
                    subject: edge_iri,
                    predicate: prop_iri,
                    object: object_str,
                    is_literal: true,
                    datatype: Some(datatype),
                    language: None,
                });
            }
        }

        Ok(triples)
    }

    /// Generates IRI for a node using templates from mapping.toml
    fn generate_node_iri(&self, node: &GraphNode) -> Result<String> {
        // Try to find a template for the first label
        if let Some(label) = node.labels.first() {
            let label_lower = label.to_lowercase();
            if let Some(template) = self.mapping_config.iri_templates.nodes.get(&label_lower) {
                return self.apply_template(template, &node.id, node);
            }
        }

        // Fallback to base IRI + node ID
        Ok(format!(
            "{}{}",
            self.mapping_config.global.base_iri, node.id
        ))
    }

    /// Generates IRI for a node from just its ID
    fn generate_node_iri_from_id(&self, node_id: &str) -> Result<String> {
        Ok(format!(
            "{}{}",
            self.mapping_config.global.base_iri, node_id
        ))
    }

    /// Generates IRI for an edge using templates
    fn generate_edge_iri(&self, edge: &GraphEdge) -> Result<String> {
        let rel_type_lower = edge.relationship_type.to_lowercase();
        if let Some(template) = self.mapping_config.iri_templates.edges.get(&rel_type_lower) {
            let template_str = template
                .replace("{base_iri}", &self.mapping_config.global.base_iri)
                .replace("{source_id}", &edge.source)
                .replace("{target_id}", &edge.target);
            return Ok(template_str);
        }

        // Fallback
        Ok(format!(
            "{}edge/{}",
            self.mapping_config.global.base_iri, edge.id
        ))
    }

    /// Applies a template string with variables
    fn apply_template(&self, template: &str, node_id: &str, node: &GraphNode) -> Result<String> {
        let mut result = template.to_string();

        result = result.replace("{base_iri}", &self.mapping_config.global.base_iri);
        result = result.replace("{id}", node_id);

        // Handle hash generation for special templates
        if result.contains("{hash}") || result.contains("{path_hash}") {
            let hash = self.calculate_hash(node_id);
            result = result.replace("{hash}", &hash);
            result = result.replace("{path_hash}", &hash);
        }

        Ok(result)
    }

    /// Calculates a simple hash for IRI generation
    fn calculate_hash(&self, input: &str) -> String {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(input.as_bytes());
        let hash = hasher.finalize();
        hash.to_hex()[..16].to_string() // Use first 16 chars
    }

    /// Expands a prefixed IRI (e.g., "foaf:Person") to full IRI
    fn expand_prefixed_iri(&self, prefixed: &str) -> Result<String> {
        if prefixed.contains("://") {
            // Already a full IRI
            return Ok(prefixed.to_string());
        }

        if let Some(colon_pos) = prefixed.find(':') {
            let (prefix, local) = prefixed.split_at(colon_pos);
            let local = &local[1..]; // Remove colon

            if let Some(namespace) = self.mapping_config.namespaces.get(prefix) {
                return Ok(format!("{}{}", namespace, local));
            } else {
                bail!("Unknown namespace prefix: {}", prefix);
            }
        }

        // No prefix, use default vocabulary
        Ok(format!(
            "{}{}",
            self.mapping_config.global.default_vocabulary, prefixed
        ))
    }

    /// Serializes a JSON value to a literal with appropriate datatype
    fn serialize_literal_value(
        &self,
        value: &serde_json::Value,
        expected_range: &str,
    ) -> Result<(String, String)> {
        let full_range_iri = self.expand_prefixed_iri(expected_range)?;

        match value {
            serde_json::Value::String(s) => {
                // Check if it's a URL/URI
                if s.starts_with("http://") || s.starts_with("https://") {
                    return Ok((s.clone(), full_range_iri));
                }

                // Detect dateTime patterns
                if s.contains('T') && (s.contains('Z') || s.contains('+') || s.contains('-')) {
                    if expected_range == "xsd:dateTime" {
                        return Ok((s.clone(), self.expand_prefixed_iri("xsd:dateTime")?));
                    }
                }

                Ok((s.clone(), full_range_iri))
            }
            serde_json::Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    let datatype = if expected_range == "xsd:nonNegativeInteger" {
                        self.expand_prefixed_iri("xsd:nonNegativeInteger")?
                    } else {
                        self.expand_prefixed_iri("xsd:integer")?
                    };
                    Ok((n.to_string(), datatype))
                } else {
                    Ok((n.to_string(), self.expand_prefixed_iri("xsd:double")?))
                }
            }
            serde_json::Value::Bool(b) => {
                Ok((b.to_string(), self.expand_prefixed_iri("xsd:boolean")?))
            }
            _ => {
                // Default to string representation
                Ok((value.to_string(), full_range_iri))
            }
        }
    }

    /// Runs consistency checks on the loaded ontology and data.
    pub fn run_consistency_checks(&self) -> Result<()> {
        // TODO: Implement consistency checks using whelk-rs
        Ok(())
    }

    /// Performs inference to discover new relationships.
    pub fn perform_inference(&self) -> Result<()> {
        // TODO: Implement inference logic using whelk-rs
        Ok(())
    }
}

impl Default for OwlValidatorService {
    fn default() -> Self {
        Self::new().expect("Failed to load default mapping configuration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_service() -> OwlValidatorService {
        // Load from actual mapping.toml for tests
        OwlValidatorService::new().expect("Failed to create test service")
    }

    #[test]
    fn test_service_creation() {
        let service = create_test_service();
        assert!(!service.mapping_config.namespaces.is_empty());
    }

    #[test]
    fn test_expand_prefixed_iri() {
        let service = create_test_service();

        let expanded = service.expand_prefixed_iri("foaf:Person").unwrap();
        assert_eq!(expanded, "http://xmlns.com/foaf/0.1/Person");

        let expanded = service.expand_prefixed_iri("rdf:type").unwrap();
        assert_eq!(expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    }

    #[test]
    fn test_map_simple_node() {
        let service = create_test_service();

        let node = GraphNode {
            id: "person1".to_string(),
            labels: vec!["Person".to_string()],
            properties: {
                let mut props = HashMap::new();
                props.insert("name".to_string(), json!("John Doe"));
                props.insert("age".to_string(), json!(30));
                props
            },
        };

        let triples = service.map_node_to_triples(&node).unwrap();

        // Should have at least rdf:type triple
        assert!(triples
            .iter()
            .any(|t| t.predicate.contains("rdf-syntax-ns#type") && t.object.contains("Person")));

        // Should have name property
        assert!(triples
            .iter()
            .any(|t| t.predicate.contains("foaf") && t.object == "John Doe"));
    }

    #[test]
    fn test_map_graph_to_rdf() {
        let service = create_test_service();

        let graph = PropertyGraph {
            nodes: vec![
                GraphNode {
                    id: "person1".to_string(),
                    labels: vec!["Person".to_string()],
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("name".to_string(), json!("Alice"));
                        props.insert("email".to_string(), json!("alice@example.com"));
                        props
                    },
                },
                GraphNode {
                    id: "company1".to_string(),
                    labels: vec!["Company".to_string()],
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("name".to_string(), json!("ACME Corp"));
                        props
                    },
                },
            ],
            edges: vec![GraphEdge {
                id: "edge1".to_string(),
                source: "person1".to_string(),
                target: "company1".to_string(),
                relationship_type: "employedBy".to_string(),
                properties: HashMap::new(),
            }],
            metadata: HashMap::new(),
        };

        let triples = service.map_graph_to_rdf(&graph).unwrap();

        assert!(!triples.is_empty());

        // Verify we have type triples
        let type_triples: Vec<_> = triples
            .iter()
            .filter(|t| t.predicate.contains("rdf-syntax-ns#type"))
            .collect();
        assert!(!type_triples.is_empty());

        // Verify we have the employedBy relationship
        assert!(triples.iter().any(|t| t.predicate.contains("employedBy")));
    }
}
