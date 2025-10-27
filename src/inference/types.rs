// src/inference/types.rs
//! Inference Result Types
//!
//! Domain types for representing inference results, explanations, and validation outcomes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::ports::ontology_repository::OwlAxiom;

/// Type of inference performed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceType {
    /// Class assertion (instance is of type class)
    ClassAssertion,

    /// Subclass relationship (child is subclass of parent)
    SubClassOf,

    /// Equivalent classes (two classes are equivalent)
    EquivalentClass,

    /// Disjoint classes (classes cannot have common instances)
    DisjointClasses,

    /// Property assertion (individual has property value)
    PropertyAssertion,

    /// Property domain inference
    PropertyDomain,

    /// Property range inference
    PropertyRange,

    /// Inverse property inference
    InverseProperty,

    /// Transitive property inference
    TransitiveProperty,
}

impl std::fmt::Display for InferenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClassAssertion => write!(f, "Class Assertion"),
            Self::SubClassOf => write!(f, "SubClass Of"),
            Self::EquivalentClass => write!(f, "Equivalent Class"),
            Self::DisjointClasses => write!(f, "Disjoint Classes"),
            Self::PropertyAssertion => write!(f, "Property Assertion"),
            Self::PropertyDomain => write!(f, "Property Domain"),
            Self::PropertyRange => write!(f, "Property Range"),
            Self::InverseProperty => write!(f, "Inverse Property"),
            Self::TransitiveProperty => write!(f, "Transitive Property"),
        }
    }
}

/// An inferred fact with confidence and explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inference {
    /// Unique identifier for this inference
    pub id: Option<String>,

    /// Type of inference
    pub inference_type: InferenceType,

    /// Subject IRI (class, individual, or property)
    pub subject: String,

    /// Predicate/relationship type
    pub predicate: String,

    /// Object IRI (class, individual, or property)
    pub object: String,

    /// Confidence score (0.0 - 1.0), 1.0 for deterministic inferences
    pub confidence: f32,

    /// Axioms that support this inference
    pub explanation: Vec<OwlAxiom>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,

    /// Timestamp when inference was computed
    pub computed_at: DateTime<Utc>,
}

impl Inference {
    /// Create a new inference with default confidence
    pub fn new(
        inference_type: InferenceType,
        subject: String,
        predicate: String,
        object: String,
    ) -> Self {
        Self {
            id: None,
            inference_type,
            subject,
            predicate,
            object,
            confidence: 1.0, // Deterministic by default
            explanation: Vec::new(),
            metadata: HashMap::new(),
            computed_at: Utc::now(),
        }
    }

    /// Add explanation axioms
    pub fn with_explanation(mut self, axioms: Vec<OwlAxiom>) -> Self {
        self.explanation = axioms;
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Explanation for an inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceExplanation {
    /// The axiom being explained
    pub axiom: OwlAxiom,

    /// Chain of axioms that lead to this conclusion
    pub axiom_chain: Vec<OwlAxiom>,

    /// Natural language explanation
    pub description: String,

    /// Confidence in this explanation
    pub confidence: f32,
}

/// Result of ontology validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the ontology is consistent
    pub consistent: bool,

    /// Unsatisfiable classes (classes equivalent to owl:Nothing)
    pub unsatisfiable: Vec<UnsatisfiableClass>,

    /// Validation warnings
    pub warnings: Vec<String>,

    /// Validation errors
    pub errors: Vec<String>,

    /// Time taken for validation (milliseconds)
    pub validation_time_ms: u64,
}

impl ValidationResult {
    /// Create a consistent validation result
    pub fn consistent() -> Self {
        Self {
            consistent: true,
            unsatisfiable: Vec::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
            validation_time_ms: 0,
        }
    }

    /// Create an inconsistent validation result
    pub fn inconsistent(unsatisfiable: Vec<UnsatisfiableClass>) -> Self {
        Self {
            consistent: false,
            unsatisfiable,
            warnings: Vec::new(),
            errors: Vec::new(),
            validation_time_ms: 0,
        }
    }

    /// Check if validation has warnings or errors
    pub fn has_issues(&self) -> bool {
        !self.warnings.is_empty() || !self.errors.is_empty() || !self.consistent
    }
}

/// Information about an unsatisfiable class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsatisfiableClass {
    /// Class IRI
    pub class_iri: String,

    /// Reason why the class is unsatisfiable
    pub reason: String,

    /// Axioms that cause unsatisfiability
    pub conflicting_axioms: Vec<OwlAxiom>,
}

/// Result of classifying an ontology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Classified hierarchy as (child, parent) pairs
    pub hierarchy: Vec<(String, String)>,

    /// Equivalent class groups
    pub equivalent_classes: Vec<Vec<String>>,

    /// Time taken for classification (milliseconds)
    pub classification_time_ms: u64,

    /// Number of inferred subsumptions
    pub inferred_count: usize,
}

/// Detailed consistency report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    /// Overall consistency status
    pub is_consistent: bool,

    /// Unsatisfiable classes
    pub unsatisfiable_classes: Vec<UnsatisfiableClass>,

    /// Classes checked
    pub classes_checked: usize,

    /// Axioms checked
    pub axioms_checked: usize,

    /// Time taken (milliseconds)
    pub check_time_ms: u64,

    /// Reasoner version
    pub reasoner_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_creation() {
        let inf = Inference::new(
            InferenceType::SubClassOf,
            "ex:Dog".to_string(),
            "rdfs:subClassOf".to_string(),
            "ex:Animal".to_string(),
        );

        assert_eq!(inf.inference_type, InferenceType::SubClassOf);
        assert_eq!(inf.confidence, 1.0);
        assert!(inf.explanation.is_empty());
    }

    #[test]
    fn test_inference_builder() {
        let inf = Inference::new(
            InferenceType::ClassAssertion,
            "ex:fido".to_string(),
            "rdf:type".to_string(),
            "ex:Dog".to_string(),
        )
        .with_confidence(0.95)
        .with_metadata("source".to_string(), "ml_classifier".to_string());

        assert_eq!(inf.confidence, 0.95);
        assert_eq!(inf.metadata.get("source").unwrap(), "ml_classifier");
    }

    #[test]
    fn test_validation_result_consistent() {
        let result = ValidationResult::consistent();
        assert!(result.consistent);
        assert!(!result.has_issues());
    }

    #[test]
    fn test_validation_result_inconsistent() {
        let unsat = UnsatisfiableClass {
            class_iri: "ex:Square  Circle".to_string(),
            reason: "Conflicting axioms".to_string(),
            conflicting_axioms: Vec::new(),
        };

        let result = ValidationResult::inconsistent(vec![unsat]);
        assert!(!result.consistent);
        assert!(result.has_issues());
        assert_eq!(result.unsatisfiable.len(), 1);
    }
}
