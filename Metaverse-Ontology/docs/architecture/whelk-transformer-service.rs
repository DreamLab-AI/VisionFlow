// WhelkTransformerService - Reference Implementation
// This service transforms AnnotatedOntology to whelk-rs compatible format

use std::sync::Arc;
use std::collections::HashMap;
use crate::models::{AnnotatedOntology, Class, Property, Individual};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransformError {
    #[error("Invalid IRI format: {0}")]
    InvalidIRI(String),

    #[error("Unsupported axiom type: {0}")]
    UnsupportedAxiom(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Whelk conversion error: {0}")]
    WhelkError(String),
}

/// Configuration for Whelk transformation
#[derive(Clone, Debug)]
pub struct WhelkTransformerConfig {
    /// Include annotations in transformation
    pub include_annotations: bool,

    /// Strict mode (fail on unsupported features)
    pub strict_mode: bool,

    /// Simplify complex axioms
    pub simplify_axioms: bool,

    /// Maximum transformation time (seconds)
    pub timeout_seconds: u64,
}

impl Default for WhelkTransformerConfig {
    fn default() -> Self {
        Self {
            include_annotations: true,
            strict_mode: false,
            simplify_axioms: true,
            timeout_seconds: 300, // 5 minutes
        }
    }
}

/// Service for transforming ontologies to Whelk format
pub struct WhelkTransformerService {
    config: WhelkTransformerConfig,
    /// Cache for IRI mappings
    iri_cache: Arc<tokio::sync::RwLock<HashMap<String, String>>>,
}

impl WhelkTransformerService {
    pub fn new(config: WhelkTransformerConfig) -> Self {
        Self {
            config,
            iri_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Transform AnnotatedOntology to Whelk-compatible format
    pub async fn transform_for_reasoning(
        &self,
        ontology: &AnnotatedOntology,
    ) -> Result<WhelkOntology, TransformError> {
        // Start transformation timer
        let start = std::time::Instant::now();

        let mut whelk_ontology = WhelkOntology::new();

        // Transform classes to Whelk concepts
        for class in &ontology.classes {
            let concept = self.transform_class(class).await?;
            whelk_ontology.add_concept(concept);

            // Check timeout
            if start.elapsed().as_secs() > self.config.timeout_seconds {
                return Err(TransformError::WhelkError(
                    "Transformation timeout exceeded".to_string()
                ));
            }
        }

        // Transform properties to Whelk roles
        for property in &ontology.properties {
            let role = self.transform_property(property).await?;
            whelk_ontology.add_role(role);
        }

        // Transform individuals to Whelk instances
        for individual in &ontology.individuals {
            let instance = self.transform_individual(individual).await?;
            whelk_ontology.add_instance(instance);
        }

        // Transform axioms to Whelk constraints
        for axiom in &ontology.axioms {
            let constraint = self.transform_axiom(axiom).await?;
            whelk_ontology.add_constraint(constraint);
        }

        Ok(whelk_ontology)
    }

    /// Transform a single class to Whelk concept
    async fn transform_class(&self, class: &Class) -> Result<WhelkConcept, TransformError> {
        let iri = self.normalize_iri(&class.iri).await?;

        let mut concept = WhelkConcept::new(iri);

        // Add superclasses
        for parent_iri in &class.super_classes {
            let normalized = self.normalize_iri(parent_iri).await?;
            concept.add_parent(normalized);
        }

        // Add equivalent classes
        for equiv_iri in &class.equivalent_classes {
            let normalized = self.normalize_iri(equiv_iri).await?;
            concept.add_equivalent(normalized);
        }

        // Add disjoint classes
        for disjoint_iri in &class.disjoint_classes {
            let normalized = self.normalize_iri(disjoint_iri).await?;
            concept.add_disjoint(normalized);
        }

        // Handle annotations if enabled
        if self.config.include_annotations {
            for annotation in &class.annotations {
                concept.add_annotation(
                    annotation.property.clone(),
                    annotation.value.clone(),
                );
            }
        }

        Ok(concept)
    }

    /// Transform a property to Whelk role
    async fn transform_property(&self, property: &Property) -> Result<WhelkRole, TransformError> {
        let iri = self.normalize_iri(&property.iri).await?;

        let mut role = match property.property_type {
            PropertyType::Object => WhelkRole::new_object_property(iri),
            PropertyType::Data => WhelkRole::new_data_property(iri),
            PropertyType::Annotation => WhelkRole::new_annotation_property(iri),
        };

        // Add domain
        if let Some(domain_iri) = &property.domain {
            let normalized = self.normalize_iri(domain_iri).await?;
            role.set_domain(normalized);
        }

        // Add range
        if let Some(range_iri) = &property.range {
            let normalized = self.normalize_iri(range_iri).await?;
            role.set_range(normalized);
        }

        // Add characteristics
        role.set_functional(property.is_functional);
        role.set_inverse_functional(property.is_inverse_functional);
        role.set_transitive(property.is_transitive);
        role.set_symmetric(property.is_symmetric);
        role.set_asymmetric(property.is_asymmetric);
        role.set_reflexive(property.is_reflexive);
        role.set_irreflexive(property.is_irreflexive);

        Ok(role)
    }

    /// Transform an individual to Whelk instance
    async fn transform_individual(
        &self,
        individual: &Individual,
    ) -> Result<WhelkInstance, TransformError> {
        let iri = self.normalize_iri(&individual.iri).await?;

        let mut instance = WhelkInstance::new(iri);

        // Add types (classes this individual belongs to)
        for class_iri in &individual.types {
            let normalized = self.normalize_iri(class_iri).await?;
            instance.add_type(normalized);
        }

        // Add property assertions
        for assertion in &individual.property_assertions {
            let prop_iri = self.normalize_iri(&assertion.property).await?;

            match &assertion.value {
                AssertionValue::Object(object_iri) => {
                    let obj_iri = self.normalize_iri(object_iri).await?;
                    instance.add_object_property_value(prop_iri, obj_iri);
                }
                AssertionValue::Data(literal) => {
                    instance.add_data_property_value(prop_iri, literal.clone());
                }
            }
        }

        Ok(instance)
    }

    /// Transform an axiom to Whelk constraint
    async fn transform_axiom(&self, axiom: &Axiom) -> Result<WhelkConstraint, TransformError> {
        match axiom {
            Axiom::SubClassOf { subclass, superclass } => {
                let sub = self.normalize_iri(subclass).await?;
                let sup = self.normalize_iri(superclass).await?;
                Ok(WhelkConstraint::SubClassOf { subclass: sub, superclass: sup })
            }

            Axiom::EquivalentClasses { classes } => {
                let mut normalized = Vec::new();
                for class in classes {
                    normalized.push(self.normalize_iri(class).await?);
                }
                Ok(WhelkConstraint::EquivalentClasses { classes: normalized })
            }

            Axiom::DisjointClasses { classes } => {
                let mut normalized = Vec::new();
                for class in classes {
                    normalized.push(self.normalize_iri(class).await?);
                }
                Ok(WhelkConstraint::DisjointClasses { classes: normalized })
            }

            // Add more axiom transformations as needed
            _ => {
                if self.config.strict_mode {
                    Err(TransformError::UnsupportedAxiom(format!("{:?}", axiom)))
                } else {
                    // Skip unsupported axioms in non-strict mode
                    Ok(WhelkConstraint::Noop)
                }
            }
        }
    }

    /// Normalize IRI to Whelk-compatible format
    async fn normalize_iri(&self, iri: &str) -> Result<String, TransformError> {
        // Check cache first
        {
            let cache = self.iri_cache.read().await;
            if let Some(normalized) = cache.get(iri) {
                return Ok(normalized.clone());
            }
        }

        // Perform normalization
        let normalized = self.normalize_iri_uncached(iri)?;

        // Cache result
        {
            let mut cache = self.iri_cache.write().await;
            cache.insert(iri.to_string(), normalized.clone());
        }

        Ok(normalized)
    }

    fn normalize_iri_uncached(&self, iri: &str) -> Result<String, TransformError> {
        // Basic IRI validation
        if iri.is_empty() {
            return Err(TransformError::InvalidIRI("Empty IRI".to_string()));
        }

        // Remove angle brackets if present
        let trimmed = iri.trim_matches(|c| c == '<' || c == '>');

        // Validate IRI format
        if !trimmed.contains("://") && !trimmed.contains(':') {
            return Err(TransformError::InvalidIRI(format!(
                "Invalid IRI format: {}",
                iri
            )));
        }

        Ok(trimmed.to_string())
    }

    /// Clear IRI cache
    pub async fn clear_cache(&self) {
        let mut cache = self.iri_cache.write().await;
        cache.clear();
    }

    /// Get transformation statistics
    pub async fn get_stats(&self) -> TransformStats {
        let cache = self.iri_cache.read().await;
        TransformStats {
            cached_iris: cache.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformStats {
    pub cached_iris: usize,
}

// Placeholder types (to be replaced with actual whelk-rs types)
pub struct WhelkOntology {
    concepts: Vec<WhelkConcept>,
    roles: Vec<WhelkRole>,
    instances: Vec<WhelkInstance>,
    constraints: Vec<WhelkConstraint>,
}

impl WhelkOntology {
    pub fn new() -> Self {
        Self {
            concepts: Vec::new(),
            roles: Vec::new(),
            instances: Vec::new(),
            constraints: Vec::new(),
        }
    }

    pub fn add_concept(&mut self, concept: WhelkConcept) {
        self.concepts.push(concept);
    }

    pub fn add_role(&mut self, role: WhelkRole) {
        self.roles.push(role);
    }

    pub fn add_instance(&mut self, instance: WhelkInstance) {
        self.instances.push(instance);
    }

    pub fn add_constraint(&mut self, constraint: WhelkConstraint) {
        self.constraints.push(constraint);
    }
}

#[derive(Debug, Clone)]
pub struct WhelkConcept {
    iri: String,
    parents: Vec<String>,
    equivalents: Vec<String>,
    disjoints: Vec<String>,
    annotations: HashMap<String, String>,
}

impl WhelkConcept {
    pub fn new(iri: String) -> Self {
        Self {
            iri,
            parents: Vec::new(),
            equivalents: Vec::new(),
            disjoints: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    pub fn add_parent(&mut self, parent: String) {
        self.parents.push(parent);
    }

    pub fn add_equivalent(&mut self, equiv: String) {
        self.equivalents.push(equiv);
    }

    pub fn add_disjoint(&mut self, disjoint: String) {
        self.disjoints.push(disjoint);
    }

    pub fn add_annotation(&mut self, property: String, value: String) {
        self.annotations.insert(property, value);
    }
}

#[derive(Debug, Clone)]
pub struct WhelkRole {
    iri: String,
    role_type: RoleType,
    domain: Option<String>,
    range: Option<String>,
    characteristics: RoleCharacteristics,
}

#[derive(Debug, Clone)]
pub enum RoleType {
    ObjectProperty,
    DataProperty,
    AnnotationProperty,
}

#[derive(Debug, Clone, Default)]
pub struct RoleCharacteristics {
    pub functional: bool,
    pub inverse_functional: bool,
    pub transitive: bool,
    pub symmetric: bool,
    pub asymmetric: bool,
    pub reflexive: bool,
    pub irreflexive: bool,
}

impl WhelkRole {
    pub fn new_object_property(iri: String) -> Self {
        Self {
            iri,
            role_type: RoleType::ObjectProperty,
            domain: None,
            range: None,
            characteristics: RoleCharacteristics::default(),
        }
    }

    pub fn new_data_property(iri: String) -> Self {
        Self {
            iri,
            role_type: RoleType::DataProperty,
            domain: None,
            range: None,
            characteristics: RoleCharacteristics::default(),
        }
    }

    pub fn new_annotation_property(iri: String) -> Self {
        Self {
            iri,
            role_type: RoleType::AnnotationProperty,
            domain: None,
            range: None,
            characteristics: RoleCharacteristics::default(),
        }
    }

    pub fn set_domain(&mut self, domain: String) {
        self.domain = Some(domain);
    }

    pub fn set_range(&mut self, range: String) {
        self.range = Some(range);
    }

    pub fn set_functional(&mut self, value: bool) {
        self.characteristics.functional = value;
    }

    pub fn set_inverse_functional(&mut self, value: bool) {
        self.characteristics.inverse_functional = value;
    }

    pub fn set_transitive(&mut self, value: bool) {
        self.characteristics.transitive = value;
    }

    pub fn set_symmetric(&mut self, value: bool) {
        self.characteristics.symmetric = value;
    }

    pub fn set_asymmetric(&mut self, value: bool) {
        self.characteristics.asymmetric = value;
    }

    pub fn set_reflexive(&mut self, value: bool) {
        self.characteristics.reflexive = value;
    }

    pub fn set_irreflexive(&mut self, value: bool) {
        self.characteristics.irreflexive = value;
    }
}

#[derive(Debug, Clone)]
pub struct WhelkInstance {
    iri: String,
    types: Vec<String>,
    object_properties: HashMap<String, Vec<String>>,
    data_properties: HashMap<String, Vec<Literal>>,
}

impl WhelkInstance {
    pub fn new(iri: String) -> Self {
        Self {
            iri,
            types: Vec::new(),
            object_properties: HashMap::new(),
            data_properties: HashMap::new(),
        }
    }

    pub fn add_type(&mut self, class: String) {
        self.types.push(class);
    }

    pub fn add_object_property_value(&mut self, property: String, value: String) {
        self.object_properties
            .entry(property)
            .or_insert_with(Vec::new)
            .push(value);
    }

    pub fn add_data_property_value(&mut self, property: String, value: Literal) {
        self.data_properties
            .entry(property)
            .or_insert_with(Vec::new)
            .push(value);
    }
}

#[derive(Debug, Clone)]
pub enum WhelkConstraint {
    SubClassOf {
        subclass: String,
        superclass: String,
    },
    EquivalentClasses {
        classes: Vec<String>,
    },
    DisjointClasses {
        classes: Vec<String>,
    },
    Noop, // Used for unsupported axioms in non-strict mode
}

// Placeholder types (to be replaced with actual model types)
#[derive(Debug, Clone)]
pub enum PropertyType {
    Object,
    Data,
    Annotation,
}

#[derive(Debug, Clone)]
pub enum AssertionValue {
    Object(String),
    Data(Literal),
}

#[derive(Debug, Clone)]
pub struct Literal {
    pub value: String,
    pub datatype: Option<String>,
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Axiom {
    SubClassOf {
        subclass: String,
        superclass: String,
    },
    EquivalentClasses {
        classes: Vec<String>,
    },
    DisjointClasses {
        classes: Vec<String>,
    },
    // Add more axiom types as needed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_normalize_iri() {
        let service = WhelkTransformerService::new(WhelkTransformerConfig::default());

        // Test basic IRI
        let result = service.normalize_iri("http://example.org/Class1").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://example.org/Class1");

        // Test IRI with angle brackets
        let result = service.normalize_iri("<http://example.org/Class2>").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://example.org/Class2");

        // Test invalid IRI
        let result = service.normalize_iri("invalid_iri").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cache() {
        let service = WhelkTransformerService::new(WhelkTransformerConfig::default());

        // First call should cache
        service.normalize_iri("http://example.org/Class1").await.unwrap();

        let stats = service.get_stats().await;
        assert_eq!(stats.cached_iris, 1);

        // Second call should use cache
        service.normalize_iri("http://example.org/Class1").await.unwrap();

        let stats = service.get_stats().await;
        assert_eq!(stats.cached_iris, 1); // Still 1, used cache

        // Clear cache
        service.clear_cache().await;
        let stats = service.get_stats().await;
        assert_eq!(stats.cached_iris, 0);
    }
}
