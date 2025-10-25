// src/services/parsers/mod.rs
//! Parser modules for ingesting data from GitHub markdown files

pub mod knowledge_graph_parser;
pub mod ontology_parser;

pub use knowledge_graph_parser::KnowledgeGraphParser;
pub use ontology_parser::OntologyParser;
