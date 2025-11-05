// src/services/parsers/ontology_parser.rs
//! Ontology Parser
//!
//! Parses markdown files containing `- ### OntologyBlock` to extract:
//! - OWL Classes
//! - Object/Data Properties
//! - Axioms (SubClassOf, DisjointWith, etc.)
//! - Class Hierarchies

use crate::ports::ontology_repository::{AxiomType, OwlAxiom, OwlClass, OwlProperty, PropertyType};
use log::{debug, info};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

// Compile regex patterns once at startup for performance and safety
static CLASS_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"owl:?_?class::\s*([a-zA-Z0-9_:/-]+(\([^)]+\))?)").expect("Invalid CLASS_PATTERN regex")
});

static OBJ_PROP_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"objectProperty::\s*([a-zA-Z0-9_:/-]+)").expect("Invalid OBJ_PROP_PATTERN regex")
});

static DATA_PROP_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"dataProperty::\s*([a-zA-Z0-9_:/-]+)").expect("Invalid DATA_PROP_PATTERN regex")
});

static SUBCLASS_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"subClassOf::\s*([a-zA-Z0-9_:/-]+)").expect("Invalid SUBCLASS_PATTERN regex")
});

static OWL_CLASS_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"owl_class::\s*([a-zA-Z0-9_:/-]+)").expect("Invalid OWL_CLASS_PATTERN regex")
});

pub struct OntologyParser;

#[derive(Debug)]
pub struct OntologyData {
    pub classes: Vec<OwlClass>,
    pub properties: Vec<OwlProperty>,
    pub axioms: Vec<OwlAxiom>,
    pub class_hierarchy: Vec<(String, String)>, 
}

impl OntologyParser {
    pub fn new() -> Self {
        Self
    }

    
    pub fn parse(&self, content: &str, filename: &str) -> Result<OntologyData, String> {
        info!("Parsing ontology file: {}", filename);

        
        let ontology_section = self.extract_ontology_section(content)?;

        
        let classes = self.extract_classes(&ontology_section, filename);

        
        let properties = self.extract_properties(&ontology_section);

        
        let axioms = self.extract_axioms(&ontology_section);

        
        let class_hierarchy = self.extract_class_hierarchy(&ontology_section);

        debug!(
            "Parsed {}: {} classes, {} properties, {} axioms, {} hierarchies",
            filename,
            classes.len(),
            properties.len(),
            axioms.len(),
            class_hierarchy.len()
        );

        Ok(OntologyData {
            classes,
            properties,
            axioms,
            class_hierarchy,
        })
    }

    
    fn extract_ontology_section(&self, content: &str) -> Result<String, String> {
        
        let lines: Vec<&str> = content.lines().collect();
        let mut section_start = None;

        for (i, line) in lines.iter().enumerate() {
            if line.contains("### OntologyBlock") {
                section_start = Some(i);
                break;
            }
        }

        let start = section_start.ok_or_else(|| "No OntologyBlock found in file".to_string())?;

        
        let section: Vec<&str> = lines[start..].iter().copied().collect();

        Ok(section.join("\n"))
    }

    
    fn extract_classes(&self, section: &str, filename: &str) -> Vec<OwlClass> {
        let mut classes = Vec::new();

        
        

        for cap in CLASS_PATTERN.captures_iter(section) {
            if let Some(class_match) = cap.get(1) {
                let class_name = class_match.as_str().trim();

                
                let label = self.find_property_value(section, class_name, "label");
                let description = self.find_property_value(section, class_name, "description");

                
                let parent_classes = self.find_parent_classes(section, class_name);

                
                let mut properties = HashMap::new();
                properties.insert("source_file".to_string(), filename.to_string());

                classes.push(OwlClass {
                    iri: class_name.to_string(),
                    label,
                    description,
                    parent_classes,
                    properties,
                    source_file: Some(filename.to_string()),
                    markdown_content: None,
                    file_sha1: None,
                    last_synced: None,
                });
            }
        }

        classes
    }

    
    fn extract_properties(&self, section: &str) -> Vec<OwlProperty> {
        let mut properties = Vec::new();

        
        
        

        
        for cap in OBJ_PROP_PATTERN.captures_iter(section) {
            if let Some(prop_match) = cap.get(1) {
                let prop_name = prop_match.as_str().trim();
                let label = self.find_property_value(section, prop_name, "label");
                let domain = self.find_property_list(section, prop_name, "domain");
                let range = self.find_property_list(section, prop_name, "range");

                properties.push(OwlProperty {
                    iri: prop_name.to_string(),
                    label,
                    property_type: PropertyType::ObjectProperty,
                    domain,
                    range,
                });
            }
        }

        
        for cap in DATA_PROP_PATTERN.captures_iter(section) {
            if let Some(prop_match) = cap.get(1) {
                let prop_name = prop_match.as_str().trim();
                let label = self.find_property_value(section, prop_name, "label");
                let domain = self.find_property_list(section, prop_name, "domain");
                let range = self.find_property_list(section, prop_name, "range");

                properties.push(OwlProperty {
                    iri: prop_name.to_string(),
                    label,
                    property_type: PropertyType::DataProperty,
                    domain,
                    range,
                });
            }
        }

        properties
    }

    
    fn extract_axioms(&self, section: &str) -> Vec<OwlAxiom> {
        let mut axioms = Vec::new();

        
        

        
        

        let lines: Vec<&str> = section.lines().collect();
        let mut current_class: Option<String> = None;

        for line in lines {
            
            if let Some(cap) = CLASS_PATTERN.captures(line) {
                if let Some(class_match) = cap.get(1) {
                    current_class = Some(class_match.as_str().to_string());
                }
            }

            
            if let Some(cap) = subCLASS_PATTERN.captures(line) {
                if let (Some(class), Some(parent)) = (&current_class, cap.get(1)) {
                    axioms.push(OwlAxiom {
                        id: None,
                        axiom_type: AxiomType::SubClassOf,
                        subject: class.clone(),
                        object: parent.as_str().to_string(),
                        annotations: HashMap::new(),
                    });
                }
            }
        }

        axioms
    }

    
    fn extract_class_hierarchy(&self, section: &str) -> Vec<(String, String)> {
        let mut hierarchy = Vec::new();

        
        

        let lines: Vec<&str> = section.lines().collect();
        let mut current_class: Option<String> = None;

        for line in lines {
            if let Some(cap) = CLASS_PATTERN.captures(line) {
                if let Some(class_match) = cap.get(1) {
                    current_class = Some(class_match.as_str().to_string());
                }
            }

            if let Some(cap) = SUBCLASS_PATTERN.captures(line) {
                if let (Some(child), Some(parent)) = (&current_class, cap.get(1)) {
                    hierarchy.push((child.clone(), parent.as_str().to_string()));
                }
            }
        }

        hierarchy
    }

    
    fn find_property_value(&self, section: &str, entity: &str, property: &str) -> Option<String> {
        
        let lines: Vec<&str> = section.lines().collect();
        let mut found_entity = false;

        for line in lines {
            if line.contains(entity) {
                found_entity = true;
                continue;
            }

            if found_entity {
                
                if line.contains("::") && !line.trim().starts_with("-") {
                    break;
                }

                
                if line.contains(&format!("{}::", property)) {
                    let parts: Vec<&str> = line.split("::").collect();
                    if parts.len() > 1 {
                        return Some(parts[1].trim().to_string());
                    }
                }
            }
        }

        None
    }

    
    fn find_parent_classes(&self, section: &str, class_name: &str) -> Vec<String> {
        let mut parents = Vec::new();
        let lines: Vec<&str> = section.lines().collect();
        let mut found_class = false;

        for line in lines {
            if line.contains(class_name) {
                found_class = true;
                continue;
            }

            if found_class {
                
                if line.contains("owl_class::") {
                    break;
                }

                
                if line.contains("subClassOf::") {
                    let parts: Vec<&str> = line.split("::").collect();
                    if parts.len() > 1 {
                        parents.push(parts[1].trim().to_string());
                    }
                }
            }
        }

        parents
    }

    
    fn find_property_list(&self, section: &str, entity: &str, property: &str) -> Vec<String> {
        if let Some(value) = self.find_property_value(section, entity, property) {
            
            value
                .split(&[',', ';'][..])
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for OntologyParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_owl_class() {
        let parser = OntologyParser::new();
        let content = r#"
# Test Document

- ### OntologyBlock
  - owl_class:: Person
    - label:: Human Person
    - description:: A human being
"#;

        let result = parser.parse(content, "test.md").unwrap();

        assert_eq!(result.classes.len(), 1);
        assert_eq!(result.classes[0].iri, "Person");
        assert_eq!(result.classes[0].label, Some("Human Person".to_string()));
        assert_eq!(
            result.classes[0].description,
            Some("A human being".to_string())
        );
        assert_eq!(result.classes[0].source_file, Some("test.md".to_string()));
    }

    #[test]
    fn test_parse_class_hierarchy() {
        let parser = OntologyParser::new();
        let content = r#"
- ### OntologyBlock
  - owl_class:: Student
    - label:: Student
    - subClassOf:: Person
  - owl_class:: Person
    - label:: Person
"#;

        let result = parser.parse(content, "test.md").unwrap();

        assert_eq!(result.classes.len(), 2);
        assert_eq!(result.class_hierarchy.len(), 1);
        assert_eq!(
            result.class_hierarchy[0],
            ("Student".to_string(), "Person".to_string())
        );

        let student = result.classes.iter().find(|c| c.iri == "Student").unwrap();
        assert_eq!(student.parent_classes, vec!["Person".to_string()]);
    }

    #[test]
    fn test_parse_object_property() {
        let parser = OntologyParser::new();
        let content = r#"
- ### OntologyBlock
  - objectProperty:: hasParent
    - label:: has parent
    - domain:: Person
    - range:: Person
"#;

        let result = parser.parse(content, "test.md").unwrap();

        assert_eq!(result.properties.len(), 1);
        assert_eq!(result.properties[0].iri, "hasParent");
        assert_eq!(result.properties[0].label, Some("has parent".to_string()));
        assert_eq!(
            result.properties[0].property_type,
            PropertyType::ObjectProperty
        );
        assert_eq!(result.properties[0].domain, vec!["Person".to_string()]);
        assert_eq!(result.properties[0].range, vec!["Person".to_string()]);
    }

    #[test]
    fn test_parse_data_property() {
        let parser = OntologyParser::new();
        let content = r#"
- ### OntologyBlock
  - dataProperty:: hasAge
    - label:: has age
    - domain:: Person
    - range:: xsd:integer
"#;

        let result = parser.parse(content, "test.md").unwrap();

        assert_eq!(result.properties.len(), 1);
        assert_eq!(result.properties[0].iri, "hasAge");
        assert_eq!(
            result.properties[0].property_type,
            PropertyType::DataProperty
        );
    }

    #[test]
    fn test_parse_axioms() {
        let parser = OntologyParser::new();
        let content = r#"
- ### OntologyBlock
  - owl_class:: Student
    - subClassOf:: Person
  - owl_class:: Teacher
    - subClassOf:: Person
"#;

        let result = parser.parse(content, "test.md").unwrap();

        assert_eq!(result.axioms.len(), 2);

        let student_axiom = result
            .axioms
            .iter()
            .find(|a| a.subject == "Student")
            .unwrap();
        assert_eq!(student_axiom.axiom_type, AxiomType::SubClassOf);
        assert_eq!(student_axiom.object, "Person");
    }

    #[test]
    fn test_no_ontology_block() {
        let parser = OntologyParser::new();
        let content = r#"
# Just a regular document
No ontology here!
"#;

        let result = parser.parse(content, "test.md");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No OntologyBlock found"));
    }

    #[test]
    fn test_parse_iri_formats() {
        let parser = OntologyParser::new();
        let content = r#"
- ### OntologyBlock
  - owl_class:: http://example.org/ontology#Person
    - label:: Person
  - owl_class:: ex:Student
    - subClassOf:: http://example.org/ontology#Person
"#;

        let result = parser.parse(content, "test.md").unwrap();

        assert_eq!(result.classes.len(), 2);

        let person = result
            .classes
            .iter()
            .find(|c| c.iri == "http://example.org/ontology#Person")
            .unwrap();
        assert_eq!(person.label, Some("Person".to_string()));

        let student = result
            .classes
            .iter()
            .find(|c| c.iri == "ex:Student")
            .unwrap();
        assert_eq!(
            student.parent_classes,
            vec!["http://example.org/ontology#Person".to_string()]
        );
    }
}
