pub mod knowledge_graph_parser;
pub mod ontology_parser;

pub use knowledge_graph_parser::KnowledgeGraphParser;
pub use ontology_parser::OntologyParser;

/// Structured data extracted from OWL/RDF ontology files
///
/// This structure represents the parsed output from ontology files,
/// containing classes, properties, and logical axioms that define
/// the ontology's semantic structure.
#[derive(Debug, Clone)]
pub struct OntologyData {
    /// OWL classes defined in the ontology
    pub classes: Vec<crate::ports::ontology_repository::OwlClass>,

    /// OWL properties (object, data, and annotation properties)
    pub properties: Vec<crate::ports::ontology_repository::OwlProperty>,

    /// OWL axioms (logical statements and constraints)
    pub axioms: Vec<crate::ports::ontology_repository::OwlAxiom>,
}

impl OntologyData {
    /// Creates a new empty OntologyData instance
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            properties: Vec::new(),
            axioms: Vec::new(),
        }
    }

    /// Creates an OntologyData instance with pre-allocated capacity
    pub fn with_capacity(classes: usize, properties: usize, axioms: usize) -> Self {
        Self {
            classes: Vec::with_capacity(classes),
            properties: Vec::with_capacity(properties),
            axioms: Vec::with_capacity(axioms),
        }
    }

    /// Returns true if the ontology data is empty
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.properties.is_empty() && self.axioms.is_empty()
    }

    /// Returns the total number of elements in the ontology
    pub fn total_elements(&self) -> usize {
        self.classes.len() + self.properties.len() + self.axioms.len()
    }
}

impl Default for OntologyData {
    fn default() -> Self {
        Self::new()
    }
}
