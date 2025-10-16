// src/ontology/mod.rs

//! The ontology module provides services for OWL/RDF validation, reasoning,
//! and translation of semantic axioms into physics constraints.

pub mod actors;
pub mod physics;

#[cfg(feature = "ontology")]
pub mod services;

#[cfg(feature = "ontology")]
pub mod parser;
