use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use horned_owl::io::ofn::reader::read as read_ofn;
use horned_owl::io::owx::reader::read as read_owx;
use horned_owl::ontology::set::SetOntology;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use uuid::Uuid;

///
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

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub is_literal: bool,
    pub datatype: Option<String>,
    pub language: Option<String>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

///
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

///
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

///
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

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: HashMap<String, serde_json::Value>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relationship_type: String,
    pub properties: HashMap<String, serde_json::Value>,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub metadata: HashMap<String, serde_json::Value>,
}

///
#[derive(Debug, Clone)]
struct CachedOntology {
    id: String,
    content_hash: String,
    ontology: SetOntology<Arc<str>>,
    axiom_count: usize,
    loaded_at: DateTime<Utc>,
    ttl_seconds: u64,
}

///
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
            cache_ttl_seconds: 3600, 
            validate_cardinality: true,
            validate_domains_ranges: true,
            validate_disjoint_classes: true,
        }
    }
}

///
#[derive(Clone)]
pub struct OwlValidatorService {
    ontology_cache: Arc<DashMap<String, CachedOntology>>,
    validation_cache: Arc<DashMap<String, ValidationReport>>,
    config: ValidationConfig,
    default_namespaces: HashMap<String, String>,
    inference_rules: Vec<InferenceRule>,
}

///
#[derive(Debug, Clone)]
enum InferenceRule {
    InverseProperty {
        property: String,
        inverse: String,
    },
    TransitiveProperty {
        property: String,
    },
    SymmetricProperty {
        property: String,
    },
    SubClassOf {
        subclass: String,
        superclass: String,
    },
}

impl OwlValidatorService {
    
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    
    pub fn with_config(config: ValidationConfig) -> Self {
        let mut default_namespaces = HashMap::new();
        default_namespaces.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        default_namespaces.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        default_namespaces.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        default_namespaces.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        default_namespaces.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());

        let inference_rules = vec![
            
            InferenceRule::InverseProperty {
                property: "http://example.org/employs".to_string(),
                inverse: "http://example.org/worksFor".to_string(),
            },
            
            InferenceRule::TransitiveProperty {
                property: "http://example.org/partOf".to_string(),
            },
            
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

    
    pub async fn load_ontology(&self, source: &str) -> Result<String> {
        let start_time = Instant::now();

        info!(
            "Loading ontology from: {}",
            if source.len() > 100 {
                &source[..100]
            } else {
                source
            }
        );

        
        let ontology_content = if source.starts_with("http://") || source.starts_with("https://") {
            self.load_from_url(source).await?
        } else if std::path::Path::new(source).exists() {
            self.load_from_file(source)?
        } else {
            
            source.to_string()
        };

        
        let content_hash = self.calculate_signature(&ontology_content);
        let ontology_id = format!("ontology_{}", content_hash);

        
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

        
        let ontology = self.parse_ontology(&ontology_content)?;
        let axiom_count = ontology.iter().count();

        info!("Parsed ontology with {} axioms", axiom_count);

        
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

    
    pub fn map_graph_to_rdf(&self, graph_data: &PropertyGraph) -> Result<Vec<RdfTriple>> {
        let mut triples = Vec::new();

        
        for node in &graph_data.nodes {
            
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

            
            for (prop_name, prop_value) in &node.properties {
                let (object, is_literal, datatype, language) =
                    self.serialize_property_value(prop_value)?;
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

        
        for edge in &graph_data.edges {
            triples.push(RdfTriple {
                subject: self.expand_iri(&edge.source)?,
                predicate: self.expand_iri(&edge.relationship_type)?,
                object: self.expand_iri(&edge.target)?,
                is_literal: false,
                datatype: None,
                language: None,
            });

            
            for (prop_name, prop_value) in &edge.properties {
                let (object, is_literal, datatype, language) =
                    self.serialize_property_value(prop_value)?;
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

        debug!(
            "Mapped {} nodes and {} edges to {} RDF triples",
            graph_data.nodes.len(),
            graph_data.edges.len(),
            triples.len()
        );

        Ok(triples)
    }

    
    pub async fn validate(
        &self,
        ontology_id: &str,
        graph_data: &PropertyGraph,
    ) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let graph_signature = self.calculate_graph_signature(graph_data);

        
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

        info!(
            "Starting validation for graph with {} nodes, {} edges",
            graph_data.nodes.len(),
            graph_data.edges.len()
        );

        
        let cached_ontology = self.ontology_cache.get(ontology_id).ok_or_else(|| {
            ValidationError::CacheError(format!("Ontology not found: {}", ontology_id))
        })?;

        
        let rdf_triples = self.map_graph_to_rdf(graph_data)?;

        
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

        
        if self.config.validate_disjoint_classes {
            violations
                .extend(self.validate_disjoint_classes(&cached_ontology.ontology, &rdf_triples)?);
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

        
        if self.config.enable_caching {
            self.validation_cache.insert(cache_key, report.clone());
        }

        info!(
            "Validation completed in {:?}: {} violations, {} inferred triples",
            duration,
            report.violations.len(),
            report.inferred_triples.len()
        );

        Ok(report)
    }

    
    pub fn infer(&self, rdf_triples: &[RdfTriple]) -> Result<Vec<RdfTriple>> {
        let mut statistics = ValidationStatistics::default();
        self.infer_triples(rdf_triples, &mut statistics)
    }

    
    pub fn get_violations(&self, report_id: &str) -> Vec<Violation> {
        
        for entry in self.validation_cache.iter() {
            if entry.value().id == report_id {
                return entry.value().violations.clone();
            }
        }
        Vec::new()
    }

    
    pub fn clear_caches(&self) {
        self.ontology_cache.clear();
        self.validation_cache.clear();
        info!("All caches cleared");
    }

    

    async fn load_from_url(&self, url: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let response = client
            .get(url)
            .header(
                "Accept",
                "application/rdf+xml, text/turtle, application/n-triples",
            )
            .send()
            .await
            .context("Failed to fetch ontology from URL")?;

        let content = response
            .text()
            .await
            .context("Failed to read ontology content")?;

        Ok(content)
    }

    fn load_from_file(&self, path: &str) -> Result<String> {
        std::fs::read_to_string(path).context("Failed to read ontology file")
    }

    fn parse_ontology(&self, content: &str) -> Result<SetOntology<Arc<str>>> {
        let trimmed = content.trim_start();

        debug!("Detecting ontology format...");

        
        if trimmed.starts_with("@prefix")
            || trimmed.starts_with("@base")
            || trimmed.contains("@prefix")
        {
            error!("Turtle format not supported. Please use OWL Functional Syntax or OWL/XML.");
            Err(ValidationError::ParseError("Turtle format not supported. Use OWL Functional Syntax (starts with 'Prefix(' or 'Ontology(') or OWL/XML.".to_string()).into())
        } else if trimmed.starts_with("<?xml")
            || (trimmed.starts_with("<") && trimmed.contains("rdf:RDF"))
        {
            error!("RDF/XML format not supported. Please use OWL Functional Syntax or OWL/XML.");
            Err(ValidationError::ParseError("RDF/XML format not supported. Use OWL Functional Syntax (starts with 'Prefix(' or 'Ontology(') or OWL/XML.".to_string()).into())
        } else if trimmed.starts_with("Prefix(") || trimmed.starts_with("Ontology(") {
            info!("Detected OWL Functional Syntax");
            self.parse_functional_syntax(content)
        } else if trimmed.starts_with("<Ontology") {
            info!("Detected OWL/XML format");
            self.parse_owx(content)
        } else if trimmed.is_empty() {
            Err(ValidationError::ParseError("Empty ontology content".to_string()).into())
        } else {
            error!("Unsupported or unrecognized ontology format");
            Err(ValidationError::ParseError("Unsupported format. Please use OWL Functional Syntax (starts with 'Prefix(' or 'Ontology(') or OWL/XML (starts with '<Ontology').".to_string()).into())
        }
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
                Err(
                    ValidationError::ParseError(format!("Functional Syntax parse error: {}", e))
                        .into(),
                )
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

        
        for node in &graph.nodes {
            hasher.update(node.id.as_bytes());
            for label in &node.labels {
                hasher.update(label.as_bytes());
            }
        }

        
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
            
            Ok(iri.to_string())
        } else if let Some(colon_pos) = iri.find(':') {
            
            let (prefix, local) = iri.split_at(colon_pos);
            let local = &local[1..]; 

            if let Some(namespace) = self.default_namespaces.get(prefix) {
                Ok(format!("{}{}", namespace, local))
            } else {
                Err(ValidationError::InvalidIri(format!("Unknown prefix: {}", prefix)).into())
            }
        } else {
            
            Ok(format!("http://example.org/{}", iri))
        }
    }

    fn serialize_property_value(
        &self,
        value: &serde_json::Value,
    ) -> Result<(String, bool, Option<String>, Option<String>)> {
        match value {
            serde_json::Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") {
                    
                    Ok((s.clone(), false, None, None))
                } else {
                    
                    Ok((
                        s.clone(),
                        true,
                        Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                        None,
                    ))
                }
            }
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    Ok((
                        n.to_string(),
                        true,
                        Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                        None,
                    ))
                } else {
                    Ok((
                        n.to_string(),
                        true,
                        Some("http://www.w3.org/2001/XMLSchema#double".to_string()),
                        None,
                    ))
                }
            }
            serde_json::Value::Bool(b) => Ok((
                b.to_string(),
                true,
                Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
                None,
            )),
            _ => Ok((
                value.to_string(),
                true,
                Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                None,
            )),
        }
    }

    fn validate_disjoint_classes(
        &self,
        _ontology: &SetOntology<Arc<str>>,
        triples: &[RdfTriple],
    ) -> Result<Vec<Violation>> {
        let mut violations = Vec::new();

        
        let mut individual_types: HashMap<String, Vec<String>> = HashMap::new();

        for triple in triples {
            if triple.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                && !triple.is_literal
            {
                individual_types
                    .entry(triple.subject.clone())
                    .or_insert_with(Vec::new)
                    .push(triple.object.clone());
            }
        }

        
        
        
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
                        message: format!(
                            "Individual {} cannot be both {} and {} (disjoint classes)",
                            individual, class1, class2
                        ),
                        subject: Some(individual.clone()),
                        predicate: Some(
                            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                        ),
                        object: None,
                        timestamp: Utc::now(),
                    });
                }
            }
        }

        Ok(violations)
    }

    fn validate_domain_range(
        &self,
        _ontology: &SetOntology<Arc<str>>,
        triples: &[RdfTriple],
    ) -> Result<Vec<Violation>> {
        let mut violations = Vec::new();

        
        
        let constraints = vec![
            (
                "http://example.org/employs",
                "http://example.org/Organization",
                "http://example.org/Person",
            ),
            (
                "http://example.org/hasAge",
                "http://example.org/Person",
                "http://www.w3.org/2001/XMLSchema#integer",
            ),
        ];

        
        let mut individual_types: HashMap<String, Vec<String>> = HashMap::new();
        for triple in triples {
            if triple.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                && !triple.is_literal
            {
                individual_types
                    .entry(triple.subject.clone())
                    .or_insert_with(Vec::new)
                    .push(triple.object.clone());
            }
        }

        
        for triple in triples {
            for (property, domain, range) in &constraints {
                if &triple.predicate == property {
                    
                    if let Some(subject_types) = individual_types.get(&triple.subject) {
                        if !subject_types.contains(&domain.to_string()) {
                            violations.push(Violation {
                                id: Uuid::new_v4().to_string(),
                                severity: Severity::Error,
                                rule: "DomainViolation".to_string(),
                                message: format!(
                                    "Subject {} must be of type {} for property {}",
                                    triple.subject, domain, property
                                ),
                                subject: Some(triple.subject.clone()),
                                predicate: Some(triple.predicate.clone()),
                                object: Some(triple.object.clone()),
                                timestamp: Utc::now(),
                            });
                        }
                    }

                    
                    if !triple.is_literal {
                        if let Some(object_types) = individual_types.get(&triple.object) {
                            if !object_types.contains(&range.to_string()) {
                                violations.push(Violation {
                                    id: Uuid::new_v4().to_string(),
                                    severity: Severity::Error,
                                    rule: "RangeViolation".to_string(),
                                    message: format!(
                                        "Object {} must be of type {} for property {}",
                                        triple.object, range, property
                                    ),
                                    subject: Some(triple.subject.clone()),
                                    predicate: Some(triple.predicate.clone()),
                                    object: Some(triple.object.clone()),
                                    timestamp: Utc::now(),
                                });
                            }
                        }
                    } else if triple.is_literal
                        && triple.datatype.as_ref() != Some(&range.to_string())
                    {
                        
                        violations.push(Violation {
                            id: Uuid::new_v4().to_string(),
                            severity: Severity::Error,
                            rule: "RangeViolation".to_string(),
                            message: format!(
                                "Literal {} must have datatype {} for property {}",
                                triple.object, range, property
                            ),
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

    fn validate_cardinality(
        &self,
        _ontology: &SetOntology<Arc<str>>,
        triples: &[RdfTriple],
    ) -> Result<Vec<Violation>> {
        let mut violations = Vec::new();

        
        
        let cardinality_constraints = vec![
            ("http://example.org/hasSSN", 1, Some(1)), 
            ("http://example.org/hasChild", 0, None),  
        ];

        
        let mut property_counts: HashMap<(String, String), usize> = HashMap::new();

        for triple in triples {
            let key = (triple.subject.clone(), triple.predicate.clone());
            *property_counts.entry(key).or_insert(0) += 1;
        }

        
        for (property, min_card, max_card) in cardinality_constraints {
            
            let subjects_using_property: HashSet<String> = triples
                .iter()
                .filter(|t| t.predicate == property)
                .map(|t| t.subject.clone())
                .collect();

            for subject in subjects_using_property {
                let count = property_counts
                    .get(&(subject.clone(), property.to_string()))
                    .unwrap_or(&0);

                if *count < min_card {
                    violations.push(Violation {
                        id: Uuid::new_v4().to_string(),
                        severity: Severity::Error,
                        rule: "MinCardinalityViolation".to_string(),
                        message: format!(
                            "Subject {} must have at least {} values for property {} (found {})",
                            subject, min_card, property, count
                        ),
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
                            message: format!(
                                "Subject {} must have at most {} values for property {} (found {})",
                                subject, max_card, property, count
                            ),
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

    fn infer_triples(
        &self,
        original_triples: &[RdfTriple],
        statistics: &mut ValidationStatistics,
    ) -> Result<Vec<RdfTriple>> {
        let mut inferred = Vec::new();
        let timeout = Duration::from_secs(self.config.reasoning_timeout_seconds);
        let start_time = Instant::now();

        
        for rule in &self.inference_rules {
            if start_time.elapsed() > timeout {
                return Err(ValidationError::TimeoutError(timeout).into());
            }

            let new_triples = match rule {
                InferenceRule::InverseProperty { property, inverse } => {
                    self.apply_inverse_property_rule(original_triples, property, inverse)
                }
                InferenceRule::TransitiveProperty { property } => {
                    self.apply_transitive_property_rule(original_triples, property)
                }
                InferenceRule::SymmetricProperty { property } => {
                    self.apply_symmetric_property_rule(original_triples, property)
                }
                InferenceRule::SubClassOf {
                    subclass,
                    superclass,
                } => self.apply_subclass_rule(original_triples, subclass, superclass),
            };

            inferred.extend(new_triples);
            statistics.inference_rules_applied += 1;
        }

        Ok(inferred)
    }

    fn apply_inverse_property_rule(
        &self,
        triples: &[RdfTriple],
        property: &str,
        inverse: &str,
    ) -> Vec<RdfTriple> {
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

    fn apply_transitive_property_rule(
        &self,
        triples: &[RdfTriple],
        property: &str,
    ) -> Vec<RdfTriple> {
        let mut inferred = Vec::new();

        
        let property_triples: Vec<_> = triples
            .iter()
            .filter(|t| t.predicate == property && !t.is_literal)
            .collect();

        
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

    fn apply_symmetric_property_rule(
        &self,
        triples: &[RdfTriple],
        property: &str,
    ) -> Vec<RdfTriple> {
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

    fn apply_subclass_rule(
        &self,
        triples: &[RdfTriple],
        subclass: &str,
        superclass: &str,
    ) -> Vec<RdfTriple> {
        let mut inferred = Vec::new();

        
        for triple in triples {
            if triple.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                && triple.object == subclass
                && !triple.is_literal
            {
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

///
///
///
///
pub fn validation_report_to_reasoning_report(
    report: &ValidationReport,
    _ontology: &SetOntology<Arc<str>>,
) -> crate::physics::ontology_constraints::OntologyReasoningReport {
    use crate::physics::ontology_constraints::{
        ConsistencyCheck, OWLAxiom, OWLAxiomType, OntologyInference, OntologyReasoningReport,
    };

    let axioms = Vec::new(); 
    let mut inferences = Vec::new();
    let mut consistency_checks = Vec::new();

    
    for triple in &report.inferred_triples {
        
        let axiom_type = if triple.predicate.contains("inverseOf") {
            OWLAxiomType::InverseOf
        } else if triple.predicate.contains("type") {
            OWLAxiomType::SubClassOf
        } else {
            OWLAxiomType::SameAs 
        };

        inferences.push(OntologyInference {
            inferred_axiom: OWLAxiom {
                axiom_type,
                subject: triple.subject.clone(),
                object: Some(triple.object.clone()),
                property: Some(triple.predicate.clone()),
                confidence: 0.8, 
            },
            premise_axioms: vec![], 
            reasoning_confidence: 0.8,
            is_derived: true,
        });
    }

    
    let is_consistent = report
        .violations
        .iter()
        .all(|v| v.severity != Severity::Error);
    let conflicting_axioms: Vec<String> = report
        .violations
        .iter()
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

        
        let graph = PropertyGraph {
            nodes: vec![GraphNode {
                id: "person1".to_string(),
                labels: vec!["Person".to_string()],
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "name".to_string(),
                        serde_json::Value::String("John".to_string()),
                    );
                    props.insert(
                        "age".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(30)),
                    );
                    props
                },
            }],
            edges: vec![],
            metadata: HashMap::new(),
        };

        
        let triples = validator.map_graph_to_rdf(&graph).unwrap();
        assert!(!triples.is_empty());

        
        let inferred = validator.infer(&triples).unwrap();
        
    }

    #[test]
    fn test_iri_expansion() {
        let validator = OwlValidatorService::new();

        
        let expanded = validator.expand_iri("foaf:Person").unwrap();
        assert_eq!(expanded, "http://xmlns.com/foaf/0.1/Person");

        
        let full_iri = "http://example.org/Person";
        let expanded = validator.expand_iri(full_iri).unwrap();
        assert_eq!(expanded, full_iri);
    }

    #[test]
    fn test_property_value_serialization() {
        let validator = OwlValidatorService::new();

        
        let string_val = serde_json::Value::String("test".to_string());
        let (object, is_literal, datatype, _) =
            validator.serialize_property_value(&string_val).unwrap();
        assert!(is_literal);
        assert_eq!(
            datatype,
            Some("http://www.w3.org/2001/XMLSchema#string".to_string())
        );

        
        let int_val = serde_json::Value::Number(serde_json::Number::from(42));
        let (object, is_literal, datatype, _) =
            validator.serialize_property_value(&int_val).unwrap();
        assert!(is_literal);
        assert_eq!(
            datatype,
            Some("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );
    }
}
