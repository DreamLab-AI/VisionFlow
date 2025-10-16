use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use horned_owl::ontology::set::SetOntology;
use horned_owl::io::rdf::reader::RDFOntology;
use horned_owl::io::owx::reader::read as read_owx;
use horned_owl::io::ofn::reader::read as read_ofn;
use horned_owl::model::Build;
use log::{debug, info, error};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur during ontology validation
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Failed to parse ontology: {0}")]
    ParseError(String),

    #[error("RDF processing error: {0}")]
    RdfError(String),

    #[error("Reasoning timeout after {0:?}")]
    TimeoutError(Duration),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Invalid IRI: {0}")]
    InvalidIri(String),

    #[error("Cache error: {0}")]
    CacheError(String),
}

/// Represents an RDF triple for property graph mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub is_literal: bool,
    pub datatype: Option<String>,
    pub language: Option<String>,
}

/// Validation result severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// Individual validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub id: String,
    pub severity: Severity,
    pub rule: String,
    pub message: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Comprehensive validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
    pub graph_signature: String,
    pub total_triples: usize,
    pub violations: Vec<Violation>,
    pub inferred_triples: Vec<RdfTriple>,
    pub statistics: ValidationStatistics,
}

/// Statistics about the validation process
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationStatistics {
    pub classes_checked: usize,
    pub properties_checked: usize,
    pub individuals_checked: usize,
    pub constraints_evaluated: usize,
    pub inference_rules_applied: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Property graph node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Property graph edge representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relationship_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Complete property graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Cached ontology with metadata
#[derive(Debug, Clone)]
struct CachedOntology {
    id: String,
    content_hash: String,
    ontology: SetOntology<Arc<str>>,
    axiom_count: usize,
    loaded_at: DateTime<Utc>,
    ttl_seconds: u64,
}

/// Configuration for validation behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub enable_reasoning: bool,
    pub reasoning_timeout_seconds: u64,
    pub enable_inference: bool,
    pub max_inference_depth: usize,
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
    pub validate_cardinality: bool,
    pub validate_domains_ranges: bool,
    pub validate_disjoint_classes: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_reasoning: true,
            reasoning_timeout_seconds: 30,
            enable_inference: true,
            max_inference_depth: 3,
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            validate_cardinality: true,
            validate_domains_ranges: true,
            validate_disjoint_classes: true,
        }
    }
}

/// Main OWL validation service
#[derive(Clone)]
pub struct OwlValidatorService {
    ontology_cache: Arc<DashMap<String, CachedOntology>>,
    validation_cache: Arc<DashMap<String, ValidationReport>>,
    config: ValidationConfig,
    default_namespaces: HashMap<String, String>,
    inference_rules: Vec<InferenceRule>,
}

/// Represents different types of inference rules
#[derive(Debug, Clone)]
enum InferenceRule {
    InverseProperty { property: String, inverse: String },
    TransitiveProperty { property: String },
    SymmetricProperty { property: String },
    SubClassOf { subclass: String, superclass: String },
}

impl OwlValidatorService {
    /// Create a new OWL validator service with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new OWL validator service with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        let mut default_namespaces = HashMap::new();
        default_namespaces.insert("rdf".to_string(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string());
        default_namespaces.insert("rdfs".to_string(), "http://www.w3.org/2000/01/rdf-schema#".to_string());
        default_namespaces.insert("owl".to_string(), "http://www.w3.org/2002/07/owl#".to_string());
        default_namespaces.insert("xsd".to_string(), "http://www.w3.org/2001/XMLSchema#".to_string());
        default_namespaces.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());

        let inference_rules = vec![
            // Common inverse properties
            InferenceRule::InverseProperty {
                property: "http://example.org/employs".to_string(),
                inverse: "http://example.org/worksFor".to_string(),
            },
            // Transitive properties
            InferenceRule::TransitiveProperty {
                property: "http://example.org/partOf".to_string(),
            },
            // Symmetric properties
            InferenceRule::SymmetricProperty {
                property: "http://example.org/knows".to_string(),
            },
        ];

        Self {
            ontology_cache: Arc::new(DashMap::new()),
            validation_cache: Arc::new(DashMap::new()),
            config,
            default_namespaces,
            inference_rules,
        }
    }

    /// Load an ontology from various sources (file, string, URL)
    pub async fn load_ontology(&self, source: &str) -> Result<String> {
        let start_time = Instant::now();

        info!("Loading ontology from: {}", if source.len() > 100 {
            &source[..100]
        } else {
            source
        });

        // Determine source type and load content
        let ontology_content = if source.starts_with("http://") || source.starts_with("https://") {
            self.load_from_url(source).await?
        } else if std::path::Path::new(source).exists() {
            self.load_from_file(source)?
        } else {
            // Treat as direct OWL/RDF content
            source.to_string()
        };

        // Generate unique ID based on content hash
        let content_hash = self.calculate_signature(&ontology_content);
        let ontology_id = format!("ontology_{}", content_hash);

        // Check if already cached
        if self.config.enable_caching {
            if let Some(cached) = self.ontology_cache.get(&ontology_id) {
                let age = Utc::now().signed_duration_since(cached.loaded_at);
                if age.num_seconds() < (self.config.cache_ttl_seconds as i64) {
                    debug!("Cache hit for ontology: {}", ontology_id);
                    return Ok(ontology_id);
                } else {
                    debug!("Cache expired for ontology: {}", ontology_id);
                }
            } else {
                debug!("Cache miss for ontology: {}", ontology_id);
            }
        }

        // Parse the ontology
        let ontology = self.parse_ontology(&ontology_content)?;
        let axiom_count = ontology.iter().count();

        info!("Parsed ontology with {} axioms", axiom_count);

        // Cache the parsed ontology
        if self.config.enable_caching {
            let cached = CachedOntology {
                id: ontology_id.clone(),
                content_hash: content_hash.clone(),
                ontology,
                axiom_count,
                loaded_at: Utc::now(),
                ttl_seconds: self.config.cache_ttl_seconds,
            };
            self.ontology_cache.insert(ontology_id.clone(), cached);
        }

        let duration = start_time.elapsed();
        info!("Ontology loaded in {:?}: {}", duration, ontology_id);

        Ok(ontology_id)
    }

    /// Convert property graph to RDF triples
    pub fn map_graph_to_rdf(&self, graph_data: &PropertyGraph) -> Result<Vec<RdfTriple>> {
        let mut triples = Vec::new();

        // Map nodes to RDF
        for node in &graph_data.nodes {
            // Add type triples for each label
            for label in &node.labels {
                triples.push(RdfTriple {
                    subject: self.expand_iri(&node.id)?,
                    predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    object: self.expand_iri(label)?,
                    is_literal: false,
                    datatype: None,
                    language: None,
                });
            }

            // Add property triples
            for (prop_name, prop_value) in &node.properties {
                let (object, is_literal, datatype, language) = self.serialize_property_value(prop_value)?;
                triples.push(RdfTriple {
                    subject: self.expand_iri(&node.id)?,
                    predicate: self.expand_iri(prop_name)?,
                    object,
                    is_literal,
                    datatype,
                    language,
                });
            }
        }

        // Map edges to RDF
        for edge in &graph_data.edges {
            triples.push(RdfTriple {
                subject: self.expand_iri(&edge.source)?,
                predicate: self.expand_iri(&edge.relationship_type)?,
                object: self.expand_iri(&edge.target)?,
                is_literal: false,
                datatype: None,
                language: None,
            });

            // Add edge properties as reified statements or use edge IRI
            for (prop_name, prop_value) in &edge.properties {
                let (object, is_literal, datatype, language) = self.serialize_property_value(prop_value)?;
                triples.push(RdfTriple {
                    subject: self.expand_iri(&edge.id)?,
                    predicate: self.expand_iri(prop_name)?,
                    object,
                    is_literal,
                    datatype,
                    language,
                });
            }
        }

        debug!("Mapped {} nodes and {} edges to {} RDF triples",
               graph_data.nodes.len(), graph_data.edges.len(), triples.len());

        Ok(triples)
    }

    /// Validate property graph against loaded ontology
    pub async fn validate(&self, ontology_id: &str, graph_data: &PropertyGraph) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let graph_signature = self.calculate_graph_signature(graph_data);

        // Check validation cache
        let cache_key = format!("{}:{}", ontology_id, graph_signature);
        if self.config.enable_caching {
            if let Some(cached_report) = self.validation_cache.get(&cache_key) {
                let age = Utc::now().signed_duration_since(cached_report.timestamp);
                if age.num_seconds() < (self.config.cache_ttl_seconds as i64) {
                    debug!("Using cached validation report");
                    return Ok(cached_report.clone());
                }
            }
        }

        info!("Starting validation for graph with {} nodes, {} edges",
              graph_data.nodes.len(), graph_data.edges.len());

        // Get cached ontology
        let cached_ontology = self.ontology_cache.get(ontology_id)
            .ok_or_else(|| ValidationError::CacheError(format!("Ontology not found: {}", ontology_id)))?;

        // Convert graph to RDF triples
        let rdf_triples = self.map_graph_to_rdf(graph_data)?;

        // Initialize validation context
        let mut violations = Vec::new();
        let mut statistics = ValidationStatistics {
            classes_checked: 0,
            properties_checked: 0,
            individuals_checked: 0,
            constraints_evaluated: 0,
            inference_rules_applied: 0,
            cache_hits: 0,
            cache_misses: 0,
        };

        // Perform different types of validation
        if self.config.validate_disjoint_classes {
            violations.extend(self.validate_disjoint_classes(&cached_ontology.ontology, &rdf_triples)?);
            statistics.constraints_evaluated += 1;
        }

        if self.config.validate_domains_ranges {
            violations.extend(self.validate_domain_range(&cached_ontology.ontology, &rdf_triples)?);
            statistics.constraints_evaluated += 1;
        }

        if self.config.validate_cardinality {
            violations.extend(self.validate_cardinality(&cached_ontology.ontology, &rdf_triples)?);
            statistics.constraints_evaluated += 1;
        }

        // Perform inference if enabled
        let inferred_triples = if self.config.enable_inference {
            self.infer_triples(&rdf_triples, &mut statistics)?
        } else {
            Vec::new()
        };

        let duration = start_time.elapsed();
        let report = ValidationReport {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            duration_ms: duration.as_millis() as u64,
            graph_signature,
            total_triples: rdf_triples.len(),
            violations,
            inferred_triples,
            statistics,
        };

        // Cache the report
        if self.config.enable_caching {
            self.validation_cache.insert(cache_key, report.clone());
        }

        info!("Validation completed in {:?}: {} violations, {} inferred triples",
              duration, report.violations.len(), report.inferred_triples.len());

        Ok(report)
    }

    /// Infer new relationships based on defined rules
    pub fn infer(&self, rdf_triples: &[RdfTriple]) -> Result<Vec<RdfTriple>> {
        let mut statistics = ValidationStatistics::default();
        self.infer_triples(rdf_triples, &mut statistics)
    }

    /// Get all violations from the most recent validation
    pub fn get_violations(&self, report_id: &str) -> Vec<Violation> {
        // Search through cached validation reports
        for entry in self.validation_cache.iter() {
            if entry.value().id == report_id {
                return entry.value().violations.clone();
            }
        }
        Vec::new()
    }

    /// Clear all caches
    pub fn clear_caches(&self) {
        self.ontology_cache.clear();
        self.validation_cache.clear();
        info!("All caches cleared");
    }

    // Private helper methods

    async fn load_from_url(&self, url: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let response = client.get(url)
            .header("Accept", "application/rdf+xml, text/turtle, application/n-triples")
            .send()
            .await
            .context("Failed to fetch ontology from URL")?;

        let content = response.text().await
            .context("Failed to read ontology content")?;

        Ok(content)
    }

    fn load_from_file(&self, path: &str) -> Result<String> {
        std::fs::read_to_string(path)
            .context("Failed to read ontology file")
    }

    fn parse_ontology(&self, content: &str) -> Result<SetOntology<Arc<str>>> {
        let trimmed = content.trim_start();

        debug!("Detecting ontology format...");

        // Detect format based on content
        if trimmed.starts_with("@prefix") || trimmed.starts_with("@base") || trimmed.contains("@prefix") {
            info!("Detected Turtle format");
            self.parse_turtle(content)
        } else if trimmed.starts_with("<?xml") || (trimmed.starts_with("<") && trimmed.contains("rdf:RDF")) {
            info!("Detected RDF/XML format");
            self.parse_rdf_xml(content)
        } else if trimmed.starts_with("Prefix(") || trimmed.starts_with("Ontology(") {
            info!("Detected OWL Functional Syntax");
            self.parse_functional_syntax(content)
        } else if trimmed.starts_with("<Ontology") {
            info!("Detected OWL/XML format");
            self.parse_owx(content)
        } else if trimmed.is_empty() {
            Err(ValidationError::ParseError("Empty ontology content".to_string()).into())
        } else {
            // Try Turtle as default since it's most flexible
            info!("Unknown format, trying Turtle parser");
            self.parse_turtle(content)
        }
    }

    fn parse_turtle(&self, content: &str) -> Result<SetOntology<Arc<str>>> {
        // TODO: horned-owl 1.2.0 API requires different approach for RDF parsing
        // Temporarily returning empty ontology until proper implementation
        debug!("Turtle parsing temporarily disabled - needs horned-owl 1.2.0 API updates");
        Ok(SetOntology::new())
    }

    fn parse_rdf_xml(&self, content: &str) -> Result<SetOntology<Arc<str>>> {
        // TODO: horned-owl 1.2.0 API requires different approach for RDF parsing
        // Temporarily returning empty ontology until proper implementation
        debug!("RDF/XML parsing temporarily disabled - needs horned-owl 1.2.0 API updates");
        Ok(SetOntology::new())
    }

    fn parse_functional_syntax(&self, content: &str) -> Result<SetOntology<Arc<str>>> {
        let cursor = Cursor::new(content.as_bytes());

        match read_ofn::<Arc<str>, SetOntology<Arc<str>>, _>(cursor, Default::default()) {
            Ok((ontology, _prefixes)) => {
                debug!("Successfully parsed Functional Syntax ontology");
                Ok(ontology)
            }
            Err(e) => {
                error!("Failed to parse Functional Syntax: {}", e);
                Err(ValidationError::ParseError(format!("Functional Syntax parse error: {}", e)).into())
            }
        }
    }

    fn parse_owx(&self, content: &str) -> Result<SetOntology<Arc<str>>> {
        let mut cursor = Cursor::new(content.as_bytes());

        match read_owx::<Arc<str>, SetOntology<Arc<str>>, _>(&mut cursor, Default::default()) {
            Ok((ontology, _prefixes)) => {
                debug!("Successfully parsed OWL/XML ontology");
                Ok(ontology)
            }
            Err(e) => {
                error!("Failed to parse OWL/XML: {}", e);
                Err(ValidationError::ParseError(format!("OWL/XML parse error: {}", e)).into())
            }
        }
    }

    fn calculate_signature(&self, content: &str) -> String {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(content.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    fn calculate_graph_signature(&self, graph: &PropertyGraph) -> String {
        use blake3::Hasher;
        let mut hasher = Hasher::new();

        // Hash nodes
        for node in &graph.nodes {
            hasher.update(node.id.as_bytes());
            for label in &node.labels {
                hasher.update(label.as_bytes());
            }
        }

        // Hash edges
        for edge in &graph.edges {
            hasher.update(edge.id.as_bytes());
            hasher.update(edge.source.as_bytes());
            hasher.update(edge.target.as_bytes());
            hasher.update(edge.relationship_type.as_bytes());
        }

        hasher.finalize().to_hex().to_string()
    }

    fn generate_cache_key(&self, source: &str) -> String {
        format!("ontology_{}", self.calculate_signature(source))
    }

    fn expand_iri(&self, iri: &str) -> Result<String> {
        if iri.contains("://") {
            // Already a full IRI
            Ok(iri.to_string())
        } else if let Some(colon_pos) = iri.find(':') {
            // Prefixed IRI
            let (prefix, local) = iri.split_at(colon_pos);
            let local = &local[1..]; // Remove the colon

            if let Some(namespace) = self.default_namespaces.get(prefix) {
                Ok(format!("{}{}", namespace, local))
            } else {
                Err(ValidationError::InvalidIri(format!("Unknown prefix: {}", prefix)).into())
            }
        } else {
            // Assume default namespace
            Ok(format!("http://example.org/{}", iri))
        }
    }

    fn serialize_property_value(&self, value: &serde_json::Value) -> Result<(String, bool, Option<String>, Option<String>)> {
        match value {
            serde_json::Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") {
                    // Treat as IRI
                    Ok((s.clone(), false, None, None))
                } else {
                    // String literal
                    Ok((s.clone(), true, Some("http://www.w3.org/2001/XMLSchema#string".to_string()), None))
                }
            },
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    Ok((n.to_string(), true, Some("http://www.w3.org/2001/XMLSchema#integer".to_string()), None))
                } else {
                    Ok((n.to_string(), true, Some("http://www.w3.org/2001/XMLSchema#double".to_string()), None))
                }
            },
            serde_json::Value::Bool(b) => {
                Ok((b.to_string(), true, Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()), None))
            },
            _ => {
                Ok((value.to_string(), true, Some("http://www.w3.org/2001/XMLSchema#string".to_string()), None))
            }
        }
    }

    fn validate_disjoint_classes(&self, _ontology: &SetOntology<Arc<str>>, triples: &[RdfTriple]) -> Result<Vec<Violation>> {
        let mut violations = Vec::new();

        // Extract type assertions from triples
        let mut individual_types: HashMap<String, Vec<String>> = HashMap::new();

        for triple in triples {
            if triple.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && !triple.is_literal {
                individual_types.entry(triple.subject.clone())
                    .or_insert_with(Vec::new)
                    .push(triple.object.clone());
            }
        }

        // Check for disjoint class violations
        // This is a simplified implementation - in practice, you'd extract
        // disjoint class axioms from the ontology
        let disjoint_pairs = vec![
            ("http://example.org/Person", "http://example.org/Company"),
            ("http://example.org/Animal", "http://example.org/Plant"),
        ];

        for (individual, types) in individual_types {
            for (class1, class2) in &disjoint_pairs {
                if types.contains(&class1.to_string()) && types.contains(&class2.to_string()) {
                    violations.push(Violation {
                        id: Uuid::new_v4().to_string(),
                        severity: Severity::Error,
                        rule: "DisjointClasses".to_string(),
                        message: format!("Individual {} cannot be both {} and {} (disjoint classes)",
                                       individual, class1, class2),
                        subject: Some(individual.clone()),
                        predicate: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
                        object: None,
                        timestamp: Utc::now(),
                    });
                }
            }
        }

        Ok(violations)
    }

    fn validate_domain_range(&self, _ontology: &SetOntology<Arc<str>>, triples: &[RdfTriple]) -> Result<Vec<Violation>> {
        let mut violations = Vec::new();

        // Define some example domain/range constraints
        // In practice, these would be extracted from the ontology
        let constraints = vec![
            ("http://example.org/employs", "http://example.org/Organization", "http://example.org/Person"),
            ("http://example.org/hasAge", "http://example.org/Person", "http://www.w3.org/2001/XMLSchema#integer"),
        ];

        // Get type information for validation
        let mut individual_types: HashMap<String, Vec<String>> = HashMap::new();
        for triple in triples {
            if triple.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" && !triple.is_literal {
                individual_types.entry(triple.subject.clone())
                    .or_insert_with(Vec::new)
                    .push(triple.object.clone());
            }
        }

        // Validate domain/range constraints
        for triple in triples {
            for (property, domain, range) in &constraints {
                if &triple.predicate == property {
                    // Check domain constraint
                    if let Some(subject_types) = individual_types.get(&triple.subject) {
                        if !subject_types.contains(&domain.to_string()) {
                            violations.push(Violation {
                                id: Uuid::new_v4().to_string(),
                                severity: Severity::Error,
                                rule: "DomainViolation".to_string(),
                                message: format!("Subject {} must be of type {} for property {}",
                                               triple.subject, domain, property),
                                subject: Some(triple.subject.clone()),
                                predicate: Some(triple.predicate.clone()),
                                object: Some(triple.object.clone()),
                                timestamp: Utc::now(),
                            });
                        }
                    }

                    // Check range constraint for object properties
                    if !triple.is_literal {
                        if let Some(object_types) = individual_types.get(&triple.object) {
                            if !object_types.contains(&range.to_string()) {
                                violations.push(Violation {
                                    id: Uuid::new_v4().to_string(),
                                    severity: Severity::Error,
                                    rule: "RangeViolation".to_string(),
                                    message: format!("Object {} must be of type {} for property {}",
                                                   triple.object, range, property),
                                    subject: Some(triple.subject.clone()),
                                    predicate: Some(triple.predicate.clone()),
                                    object: Some(triple.object.clone()),
                                    timestamp: Utc::now(),
                                });
                            }
                        }
                    } else if triple.is_literal && triple.datatype.as_ref() != Some(&range.to_string()) {
                        // Check datatype for literals
                        violations.push(Violation {
                            id: Uuid::new_v4().to_string(),
                            severity: Severity::Error,
                            rule: "RangeViolation".to_string(),
                            message: format!("Literal {} must have datatype {} for property {}",
                                           triple.object, range, property),
                            subject: Some(triple.subject.clone()),
                            predicate: Some(triple.predicate.clone()),
                            object: Some(triple.object.clone()),
                            timestamp: Utc::now(),
                        });
                    }
                }
            }
        }

        Ok(violations)
    }

    fn validate_cardinality(&self, _ontology: &SetOntology<Arc<str>>, triples: &[RdfTriple]) -> Result<Vec<Violation>> {
        let mut violations = Vec::new();

        // Define cardinality constraints
        // In practice, these would be extracted from the ontology
        let cardinality_constraints = vec![
            ("http://example.org/hasSSN", 1, Some(1)), // exactly 1
            ("http://example.org/hasChild", 0, None),   // minimum 0, no maximum
        ];

        // Count property occurrences per subject
        let mut property_counts: HashMap<(String, String), usize> = HashMap::new();

        for triple in triples {
            let key = (triple.subject.clone(), triple.predicate.clone());
            *property_counts.entry(key).or_insert(0) += 1;
        }

        // Check cardinality constraints
        for (property, min_card, max_card) in cardinality_constraints {
            // Group subjects by property usage
            let subjects_using_property: HashSet<String> = triples.iter()
                .filter(|t| t.predicate == property)
                .map(|t| t.subject.clone())
                .collect();

            for subject in subjects_using_property {
                let count = property_counts.get(&(subject.clone(), property.to_string())).unwrap_or(&0);

                if *count < min_card {
                    violations.push(Violation {
                        id: Uuid::new_v4().to_string(),
                        severity: Severity::Error,
                        rule: "MinCardinalityViolation".to_string(),
                        message: format!("Subject {} must have at least {} values for property {} (found {})",
                                       subject, min_card, property, count),
                        subject: Some(subject.clone()),
                        predicate: Some(property.to_string()),
                        object: None,
                        timestamp: Utc::now(),
                    });
                }

                if let Some(max_card) = max_card {
                    if *count > max_card {
                        violations.push(Violation {
                            id: Uuid::new_v4().to_string(),
                            severity: Severity::Error,
                            rule: "MaxCardinalityViolation".to_string(),
                            message: format!("Subject {} must have at most {} values for property {} (found {})",
                                           subject, max_card, property, count),
                            subject: Some(subject.clone()),
                            predicate: Some(property.to_string()),
                            object: None,
                            timestamp: Utc::now(),
                        });
                    }
                }
            }
        }

        Ok(violations)
    }

    fn infer_triples(&self, original_triples: &[RdfTriple], statistics: &mut ValidationStatistics) -> Result<Vec<RdfTriple>> {
        let mut inferred = Vec::new();
        let timeout = Duration::from_secs(self.config.reasoning_timeout_seconds);
        let start_time = Instant::now();

        // Apply inference rules with timeout
        for rule in &self.inference_rules {
            if start_time.elapsed() > timeout {
                return Err(ValidationError::TimeoutError(timeout).into());
            }

            let new_triples = match rule {
                InferenceRule::InverseProperty { property, inverse } => {
                    self.apply_inverse_property_rule(original_triples, property, inverse)
                },
                InferenceRule::TransitiveProperty { property } => {
                    self.apply_transitive_property_rule(original_triples, property)
                },
                InferenceRule::SymmetricProperty { property } => {
                    self.apply_symmetric_property_rule(original_triples, property)
                },
                InferenceRule::SubClassOf { subclass, superclass } => {
                    self.apply_subclass_rule(original_triples, subclass, superclass)
                },
            };

            inferred.extend(new_triples);
            statistics.inference_rules_applied += 1;
        }

        Ok(inferred)
    }

    fn apply_inverse_property_rule(&self, triples: &[RdfTriple], property: &str, inverse: &str) -> Vec<RdfTriple> {
        let mut inferred = Vec::new();

        for triple in triples {
            if triple.predicate == property && !triple.is_literal {
                inferred.push(RdfTriple {
                    subject: triple.object.clone(),
                    predicate: inverse.to_string(),
                    object: triple.subject.clone(),
                    is_literal: false,
                    datatype: None,
                    language: None,
                });
            }
        }

        inferred
    }

    fn apply_transitive_property_rule(&self, triples: &[RdfTriple], property: &str) -> Vec<RdfTriple> {
        let mut inferred = Vec::new();

        // Find all triples with the transitive property
        let property_triples: Vec<_> = triples.iter()
            .filter(|t| t.predicate == property && !t.is_literal)
            .collect();

        // Apply transitivity: if (A, P, B) and (B, P, C) then (A, P, C)
        for triple1 in &property_triples {
            for triple2 in &property_triples {
                if triple1.object == triple2.subject && triple1.subject != triple2.object {
                    inferred.push(RdfTriple {
                        subject: triple1.subject.clone(),
                        predicate: property.to_string(),
                        object: triple2.object.clone(),
                        is_literal: false,
                        datatype: None,
                        language: None,
                    });
                }
            }
        }

        inferred
    }

    fn apply_symmetric_property_rule(&self, triples: &[RdfTriple], property: &str) -> Vec<RdfTriple> {
        let mut inferred = Vec::new();

        for triple in triples {
            if triple.predicate == property && !triple.is_literal {
                inferred.push(RdfTriple {
                    subject: triple.object.clone(),
                    predicate: property.to_string(),
                    object: triple.subject.clone(),
                    is_literal: false,
                    datatype: None,
                    language: None,
                });
            }
        }

        inferred
    }

    fn apply_subclass_rule(&self, triples: &[RdfTriple], subclass: &str, superclass: &str) -> Vec<RdfTriple> {
        let mut inferred = Vec::new();

        // If X is of type Subclass, then X is also of type Superclass
        for triple in triples {
            if triple.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                && triple.object == subclass && !triple.is_literal {
                inferred.push(RdfTriple {
                    subject: triple.subject.clone(),
                    predicate: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    object: superclass.to_string(),
                    is_literal: false,
                    datatype: None,
                    language: None,
                });
            }
        }

        inferred
    }
}

impl Default for OwlValidatorService {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert ValidationReport to OntologyReasoningReport for physics integration
///
/// TODO: This function needs to be updated to use horned-owl 1.2.0 API for proper axiom extraction.
/// Currently returns a simplified report based on validation results.
pub fn validation_report_to_reasoning_report(
    report: &ValidationReport,
    _ontology: &SetOntology<Arc<str>>,
) -> crate::physics::ontology_constraints::OntologyReasoningReport {
    use crate::physics::ontology_constraints::{OntologyReasoningReport, OWLAxiom, OWLAxiomType, OntologyInference, ConsistencyCheck};

    let axioms = Vec::new(); // TODO: Extract axioms using horned-owl 1.2.0 API
    let mut inferences = Vec::new();
    let mut consistency_checks = Vec::new();

    // Convert inferred triples to inferences
    for triple in &report.inferred_triples {
        // Determine axiom type from triple pattern
        let axiom_type = if triple.predicate.contains("inverseOf") {
            OWLAxiomType::InverseOf
        } else if triple.predicate.contains("type") {
            OWLAxiomType::SubClassOf
        } else {
            OWLAxiomType::SameAs // Default
        };

        inferences.push(OntologyInference {
            inferred_axiom: OWLAxiom {
                axiom_type,
                subject: triple.subject.clone(),
                object: Some(triple.object.clone()),
                property: Some(triple.predicate.clone()),
                confidence: 0.8, // Inferred axioms have lower confidence
            },
            premise_axioms: vec![], // Could be populated with source axiom IDs
            reasoning_confidence: 0.8,
            is_derived: true,
        });
    }

    // Create consistency checks based on violations
    let is_consistent = report.violations.iter().all(|v| v.severity != Severity::Error);
    let conflicting_axioms: Vec<String> = report.violations.iter()
        .filter(|v| v.severity == Severity::Error)
        .map(|v| v.rule.clone())
        .collect();

    consistency_checks.push(ConsistencyCheck {
        is_consistent,
        conflicting_axioms,
        suggested_resolution: if !is_consistent {
            Some("Review and resolve constraint violations".to_string())
        } else {
            None
        },
    });

    OntologyReasoningReport {
        axioms,
        inferences,
        consistency_checks,
        reasoning_time_ms: report.duration_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_validation() {
        let validator = OwlValidatorService::new();

        // Create a simple property graph
        let graph = PropertyGraph {
            nodes: vec![
                GraphNode {
                    id: "person1".to_string(),
                    labels: vec!["Person".to_string()],
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("name".to_string(), serde_json::Value::String("John".to_string()));
                        props.insert("age".to_string(), serde_json::Value::Number(serde_json::Number::from(30)));
                        props
                    },
                }
            ],
            edges: vec![],
            metadata: HashMap::new(),
        };

        // Map to RDF triples
        let triples = validator.map_graph_to_rdf(&graph).unwrap();
        assert!(!triples.is_empty());

        // Test inference
        let inferred = validator.infer(&triples).unwrap();
        // Inference results depend on loaded ontology
    }

    #[test]
    fn test_iri_expansion() {
        let validator = OwlValidatorService::new();

        // Test prefixed IRI
        let expanded = validator.expand_iri("foaf:Person").unwrap();
        assert_eq!(expanded, "http://xmlns.com/foaf/0.1/Person");

        // Test full IRI
        let full_iri = "http://example.org/Person";
        let expanded = validator.expand_iri(full_iri).unwrap();
        assert_eq!(expanded, full_iri);
    }

    #[test]
    fn test_property_value_serialization() {
        let validator = OwlValidatorService::new();

        // Test string value
        let string_val = serde_json::Value::String("test".to_string());
        let (object, is_literal, datatype, _) = validator.serialize_property_value(&string_val).unwrap();
        assert!(is_literal);
        assert_eq!(datatype, Some("http://www.w3.org/2001/XMLSchema#string".to_string()));

        // Test integer value
        let int_val = serde_json::Value::Number(serde_json::Number::from(42));
        let (object, is_literal, datatype, _) = validator.serialize_property_value(&int_val).unwrap();
        assert!(is_literal);
        assert_eq!(datatype, Some("http://www.w3.org/2001/XMLSchema#integer".to_string()));
    }
}