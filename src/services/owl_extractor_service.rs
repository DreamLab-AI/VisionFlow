// src/services/owl_extractor_service.rs
//! OWL Extractor Service
//!
//! Extracts and parses OWL Functional Syntax blocks from markdown content
//! stored in the database using horned-owl for complete semantic preservation.
//!
//! This service reads raw markdown from the database and builds complete
//! OWL ontologies with all restrictions, axioms, and complex semantics.

#[cfg(feature = "ontology")]
use horned_owl::io::owx::reader::read as read_owx;
#[cfg(feature = "ontology")]
use horned_owl::model::*;
#[cfg(feature = "ontology")]
use horned_functional::io::reader::read as read_functional;

use crate::ports::ontology_repository::{OntologyRepository, OwlClass};
use log::{debug, info, warn};
use regex::Regex;
use std::sync::Arc;

///
pub struct OwlExtractorService<R: OntologyRepository> {
    repo: Arc<R>,
}

impl<R: OntologyRepository> OwlExtractorService<R> {
    
    pub fn new(repo: Arc<R>) -> Self {
        Self { repo }
    }

    
    pub async fn extract_owl_from_class(&self, class_iri: &str) -> Result<ExtractedOwl, String> {
        
        let class = self
            .repo
            .get_owl_class(class_iri)
            .await
            .map_err(|e| format!("Failed to fetch class: {}", e))?
            .ok_or_else(|| format!("Class not found: {}", class_iri))?;

        let markdown_content = class
            .markdown_content
            .as_ref()
            .ok_or_else(|| format!("No markdown content for class: {}", class_iri))?;

        self.parse_owl_blocks(markdown_content, class_iri)
    }

    
    pub async fn extract_all_owl(&self) -> Result<Vec<ExtractedOwl>, String> {
        info!("Extracting OWL from all classes in database...");

        let classes = self
            .repo
            .list_owl_classes()
            .await
            .map_err(|e| format!("Failed to list classes: {}", e))?;

        let mut extracted = Vec::new();
        let mut success_count = 0;
        let mut skip_count = 0;
        let mut error_count = 0;

        for class in classes {
            if let Some(markdown_content) = &class.markdown_content {
                match self.parse_owl_blocks(markdown_content, &class.iri) {
                    Ok(owl) => {
                        extracted.push(owl);
                        success_count += 1;
                    }
                    Err(e) => {
                        warn!("Failed to parse OWL for {}: {}", class.iri, e);
                        error_count += 1;
                    }
                }
            } else {
                skip_count += 1;
            }
        }

        info!(
            "OWL extraction complete: {} successful, {} skipped (no markdown), {} errors",
            success_count, skip_count, error_count
        );

        Ok(extracted)
    }

    
    fn parse_owl_blocks(&self, markdown: &str, class_iri: &str) -> Result<ExtractedOwl, String> {
        
        let code_block_pattern = Regex::new(r"```(?:clojure|owl-functional)\s*\n([\s\S]*?)```")
            .map_err(|e| format!("Regex error: {}", e))?;

        let mut owl_blocks = Vec::new();

        for cap in code_block_pattern.captures_iter(markdown) {
            if let Some(block_match) = cap.get(1) {
                let owl_text = block_match.as_str().trim();

                
                if owl_text.contains("Declaration")
                    || owl_text.contains("SubClassOf")
                    || owl_text.contains("ObjectSomeValuesFrom")
                {
                    owl_blocks.push(owl_text.to_string());
                }
            }
        }

        if owl_blocks.is_empty() {
            return Err(format!("No OWL blocks found for class: {}", class_iri));
        }

        debug!(
            "Found {} OWL blocks for class {}",
            owl_blocks.len(),
            class_iri
        );

        Ok(ExtractedOwl {
            class_iri: class_iri.to_string(),
            owl_blocks,
            axiom_count: self.count_axioms(&owl_blocks),
        })
    }

    
    fn count_axioms(&self, blocks: &[String]) -> usize {
        let axiom_patterns = [
            "Declaration",
            "SubClassOf",
            "EquivalentClass",
            "DisjointWith",
            "ObjectSomeValuesFrom",
            "DataPropertyAssertion",
            "ObjectPropertyAssertion",
            "AnnotationAssertion",
        ];

        blocks
            .iter()
            .flat_map(|block| {
                axiom_patterns
                    .iter()
                    .map(|pattern| block.matches(pattern).count())
                    .sum::<usize>()
            })
            .sum()
    }

    
    #[cfg(feature = "ontology")]
    pub fn parse_with_horned_owl(&self, owl_text: &str) -> Result<AnnotatedOntology, String> {
        use std::io::Cursor;

        let cursor = Cursor::new(owl_text.as_bytes());

        read_functional(cursor, Default::default())
            .map_err(|e| format!("Failed to parse OWL with horned-owl: {}", e))
    }

    
    #[cfg(feature = "ontology")]
    pub async fn build_complete_ontology(&self) -> Result<AnnotatedOntology, String> {
        info!("Building complete ontology from database with horned-owl...");

        let extracted = self.extract_all_owl().await?;

        
        let mut combined_ontology = AnnotatedOntology::default();

        for ext in extracted {
            for block in ext.owl_blocks {
                match self.parse_with_horned_owl(&block) {
                    Ok(onto) => {
                        
                        for axiom in onto.axiom() {
                            combined_ontology.insert(axiom.clone());
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to parse OWL block for {}: {}",
                            ext.class_iri, e
                        );
                    }
                }
            }
        }

        info!(
            "Complete ontology built: {} axioms",
            combined_ontology.axiom().len()
        );

        Ok(combined_ontology)
    }
}

///
#[derive(Debug, Clone)]
pub struct ExtractedOwl {
    pub class_iri: String,
    pub owl_blocks: Vec<String>,
    pub axiom_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_owl_blocks() {
        let markdown = r#"
# Test Class

Some description

## OWL Formal Semantics

```clojure
(Declaration (Class :TestClass))
(AnnotationAssertion rdfs:label :TestClass "Test Class"@en)
(SubClassOf :TestClass :ParentClass)
(SubClassOf :TestClass
  (ObjectSomeValuesFrom :hasProperty :SomeValue))
```

More content
"#;

        
        
        let pattern = Regex::new(r"```(?:clojure|owl-functional)\s*\n([\s\S]*?)```").unwrap();
        let captures: Vec<_> = pattern.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 1);
        assert!(captures[0].get(1).unwrap().as_str().contains("Declaration"));
    }
}
