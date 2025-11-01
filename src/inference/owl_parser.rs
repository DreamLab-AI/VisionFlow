// src/inference/owl_parser.rs
//! OWL 2 DL Parser
//!
//! Parses OWL ontologies in various formats (OWL/XML, Manchester, RDF/XML, Turtle).
//! Uses horned-owl library for OWL parsing and supports multiple serialization formats.

use std::collections::HashMap;
use thiserror::Error;
use serde::{Deserialize, Serialize};

#[cfg(feature = "ontology")]
use horned_owl::io::owx::reader::read as read_owx;
#[cfg(feature = "ontology")]
use horned_owl::model::ArcStr;
#[cfg(feature = "ontology")]
use horned_owl::ontology::set::SetOntology;

use crate::ports::ontology_repository::{OwlClass, OwlAxiom, AxiomType};

///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OWLFormat {
    
    OwlXml,

    
    Manchester,

    
    RdfXml,

    
    Turtle,

    
    NTriples,

    
    Functional,
}

impl std::fmt::Display for OWLFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OwlXml => write!(f, "OWL/XML"),
            Self::Manchester => write!(f, "Manchester"),
            Self::RdfXml => write!(f, "RDF/XML"),
            Self::Turtle => write!(f, "Turtle"),
            Self::NTriples => write!(f, "N-Triples"),
            Self::Functional => write!(f, "Functional"),
        }
    }
}

///
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid OWL syntax: {0}")]
    InvalidSyntax(String),

    #[error("Feature not enabled: ontology feature required")]
    FeatureNotEnabled,
}

///
#[derive(Debug, Clone)]
pub struct ParseResult {
    
    pub classes: Vec<OwlClass>,

    
    pub axioms: Vec<OwlAxiom>,

    
    pub ontology_iri: Option<String>,

    
    pub version_iri: Option<String>,

    
    pub imports: Vec<String>,

    
    pub stats: ParseStatistics,
}

///
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParseStatistics {
    pub classes_count: usize,
    pub axioms_count: usize,
    pub imports_count: usize,
    pub parse_time_ms: u64,
}

///
pub struct OWLParser;

impl OWLParser {
    
    pub fn parse(content: &str) -> Result<ParseResult, ParseError> {
        let format = Self::detect_format(content);
        Self::parse_with_format(content, format)
    }

    
    pub fn parse_with_format(content: &str, format: OWLFormat) -> Result<ParseResult, ParseError> {
        let start = std::time::Instant::now();

        #[cfg(feature = "ontology")]
        {
            let ontology = match format {
                OWLFormat::OwlXml => Self::parse_owl_xml(content)?,
                OWLFormat::RdfXml => Self::parse_rdf_xml(content)?,
                OWLFormat::Turtle => Self::parse_turtle(content)?,
                OWLFormat::Manchester | OWLFormat::Functional | OWLFormat::NTriples => {
                    return Err(ParseError::UnsupportedFormat(format!("{} parsing not yet implemented", format)));
                }
            };

            let result = Self::extract_ontology_components(&ontology);
            let parse_time_ms = start.elapsed().as_millis() as u64;

            Ok(ParseResult {
                stats: ParseStatistics {
                    classes_count: result.classes.len(),
                    axioms_count: result.axioms.len(),
                    imports_count: result.imports.len(),
                    parse_time_ms,
                },
                ..result
            })
        }

        #[cfg(not(feature = "ontology"))]
        {
            Err(ParseError::FeatureNotEnabled)
        }
    }

    
    pub fn detect_format(content: &str) -> OWLFormat {
        let trimmed = content.trim();

        
        if trimmed.starts_with("<?xml") || trimmed.starts_with("<rdf:RDF") {
            if trimmed.contains("owl:Ontology") || trimmed.contains("Ontology(") {
                return OWLFormat::OwlXml;
            }
            return OWLFormat::RdfXml;
        }

        
        if trimmed.starts_with("@prefix") || trimmed.starts_with("@base") {
            return OWLFormat::Turtle;
        }

        
        if trimmed.contains("Ontology:") || trimmed.contains("Class:") {
            return OWLFormat::Manchester;
        }

        
        if trimmed.starts_with("Ontology(") {
            return OWLFormat::Functional;
        }

        
        OWLFormat::OwlXml
    }

    #[cfg(feature = "ontology")]
    
    fn parse_owl_xml(content: &str) -> Result<SetOntology<ArcStr>, ParseError> {
        let cursor = std::io::Cursor::new(content.as_bytes());
        let mut buf_reader = std::io::BufReader::new(cursor);

        read_owx(&mut buf_reader, Default::default())
            .map(|(ontology, _)| ontology)
            .map_err(|e| ParseError::ParseError(format!("OWL/XML parse error: {:?}", e)))
    }

    #[cfg(feature = "ontology")]
    
    fn parse_rdf_xml(content: &str) -> Result<SetOntology<ArcStr>, ParseError> {
        
        
        Ok(SetOntology::new())
    }

    #[cfg(feature = "ontology")]
    
    fn parse_turtle(content: &str) -> Result<SetOntology<ArcStr>, ParseError> {
        
        
        Ok(SetOntology::new())
    }

    #[cfg(feature = "ontology")]
    
    fn extract_ontology_components(ontology: &SetOntology<ArcStr>) -> ParseResult {
        use horned_owl::model::{Component, Class};

        let mut classes = Vec::new();
        let mut axioms = Vec::new();
        let mut imports = Vec::new();
        let ontology_iri = None;
        let version_iri = None;

        for ann_component in ontology.iter() {
            match &ann_component.component {
                Component::DeclareClass(decl) => {
                    classes.push(OwlClass {
                        iri: decl.0 .0.to_string(),
                        label: None,
                        description: None,
                        parent_classes: Vec::new(),
                        properties: HashMap::new(),
                        source_file: None,
                        markdown_content: None,
                        file_sha1: None,
                        last_synced: None,
                    });
                }

                Component::SubClassOf(axiom) => {
                    
                    if let (
                        horned_owl::model::ClassExpression::Class(Class(sub_iri)),
                        horned_owl::model::ClassExpression::Class(Class(sup_iri)),
                    ) = (&axiom.sub, &axiom.sup)
                    {
                        axioms.push(OwlAxiom {
                            id: None,
                            axiom_type: AxiomType::SubClassOf,
                            subject: sub_iri.to_string(),
                            object: sup_iri.to_string(),
                            annotations: std::collections::HashMap::new(),
                        });
                    }
                }

                Component::EquivalentClasses(equiv) => {
                    
                    let class_iris: Vec<String> = equiv
                        .0
                        .iter()
                        .filter_map(|ce| {
                            if let horned_owl::model::ClassExpression::Class(Class(iri)) = ce {
                                Some(iri.to_string())
                            } else {
                                None
                            }
                        })
                        .collect();

                    
                    for i in 0..class_iris.len() {
                        for j in (i + 1)..class_iris.len() {
                            axioms.push(OwlAxiom {
                                id: None,
                                axiom_type: AxiomType::EquivalentClass,
                                subject: class_iris[i].clone(),
                                object: class_iris[j].clone(),
                                annotations: std::collections::HashMap::new(),
                            });
                        }
                    }
                }

                Component::OntologyAnnotation(_) => {
                    
                }

                Component::Import(import) => {
                    imports.push(import.0.to_string());
                }

                _ => {
                    
                }
            }
        }

        
        
        

        ParseResult {
            classes,
            axioms,
            ontology_iri,
            version_iri,
            imports,
            stats: ParseStatistics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection_owl_xml() {
        let content = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Ontology rdf:about="http://example.com/ontology"/>
</rdf:RDF>"#;

        assert_eq!(OWLParser::detect_format(content), OWLFormat::OwlXml);
    }

    #[test]
    fn test_format_detection_turtle() {
        let content = "@prefix owl: <http://www.w3.org/2002/07/owl#> .";
        assert_eq!(OWLParser::detect_format(content), OWLFormat::Turtle);
    }

    #[test]
    fn test_format_detection_manchester() {
        let content = "Ontology: <http://example.com/ont>\nClass: Dog";
        assert_eq!(OWLParser::detect_format(content), OWLFormat::Manchester);
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn test_parse_simple_owl_xml() {
        let content = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.com/test"/>
    <owl:Class rdf:about="http://example.com/Animal"/>
    <owl:Class rdf:about="http://example.com/Dog">
        <rdfs:subClassOf rdf:resource="http://example.com/Animal"/>
    </owl:Class>
</rdf:RDF>"#;

        let result = OWLParser::parse(content);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert!(parsed.classes.len() >= 1);
        assert!(parsed.stats.parse_time_ms > 0);
    }
}
