// src/application/ontology/mod.rs
//! Ontology Domain Application Layer
//!
//! Contains all directives (write operations) and queries (read operations)
//! for ontology management following CQRS patterns.

pub mod directives;
pub mod queries;

// Re-export directives
pub use directives::{
    AddAxiom, AddAxiomHandler, AddOwlClass, AddOwlClassHandler, AddOwlProperty,
    AddOwlPropertyHandler, RemoveAxiom, RemoveAxiomHandler, RemoveOwlClass, RemoveOwlClassHandler,
    SaveOntologyGraph, SaveOntologyGraphHandler, StoreInferenceResults,
    StoreInferenceResultsHandler, UpdateOwlClass, UpdateOwlClassHandler, UpdateOwlProperty,
    UpdateOwlPropertyHandler,
};

// Re-export queries
pub use queries::{
    GetClassAxioms, GetClassAxiomsHandler, GetInferenceResults, GetInferenceResultsHandler,
    GetOntologyMetrics, GetOntologyMetricsHandler, GetOwlClass, GetOwlClassHandler, GetOwlProperty,
    GetOwlPropertyHandler, ListOwlClasses, ListOwlClassesHandler, ListOwlProperties,
    ListOwlPropertiesHandler, LoadOntologyGraph, LoadOntologyGraphHandler, QueryOntology,
    QueryOntologyHandler, ValidateOntology, ValidateOntologyHandler,
};
