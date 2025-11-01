// src/cqrs/commands/ontology_commands.rs
//! Ontology Commands
//!
//! Write operations for ontology repository.

use crate::cqrs::types::{Command, Result};
use crate::models::graph::GraphData;
use crate::ports::ontology_repository::{
    InferenceResults, OwlAxiom, OwlClass, OwlProperty, PathfindingCacheEntry,
};

/
#[derive(Debug, Clone)]
pub struct AddClassCommand {
    pub class: OwlClass,
}

impl Command for AddClassCommand {
    type Result = String; 

    fn name(&self) -> &'static str {
        "AddClass"
    }

    fn validate(&self) -> Result<()> {
        if self.class.iri.is_empty() {
            return Err(anyhow::anyhow!("Class IRI cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct UpdateClassCommand {
    pub class: OwlClass,
}

impl Command for UpdateClassCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateClass"
    }

    fn validate(&self) -> Result<()> {
        if self.class.iri.is_empty() {
            return Err(anyhow::anyhow!("Class IRI cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct RemoveClassCommand {
    pub iri: String,
}

impl Command for RemoveClassCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "RemoveClass"
    }

    fn validate(&self) -> Result<()> {
        if self.iri.is_empty() {
            return Err(anyhow::anyhow!("Class IRI cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct AddPropertyCommand {
    pub property: OwlProperty,
}

impl Command for AddPropertyCommand {
    type Result = String; 

    fn name(&self) -> &'static str {
        "AddProperty"
    }

    fn validate(&self) -> Result<()> {
        if self.property.iri.is_empty() {
            return Err(anyhow::anyhow!("Property IRI cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct UpdatePropertyCommand {
    pub property: OwlProperty,
}

impl Command for UpdatePropertyCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateProperty"
    }

    fn validate(&self) -> Result<()> {
        if self.property.iri.is_empty() {
            return Err(anyhow::anyhow!("Property IRI cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct RemovePropertyCommand {
    pub iri: String,
}

impl Command for RemovePropertyCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "RemoveProperty"
    }

    fn validate(&self) -> Result<()> {
        if self.iri.is_empty() {
            return Err(anyhow::anyhow!("Property IRI cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct AddAxiomCommand {
    pub axiom: OwlAxiom,
}

impl Command for AddAxiomCommand {
    type Result = u64; 

    fn name(&self) -> &'static str {
        "AddAxiom"
    }

    fn validate(&self) -> Result<()> {
        if self.axiom.subject.is_empty() {
            return Err(anyhow::anyhow!("Axiom subject cannot be empty"));
        }
        if self.axiom.object.is_empty() {
            return Err(anyhow::anyhow!("Axiom object cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct RemoveAxiomCommand {
    pub axiom_id: u64,
}

impl Command for RemoveAxiomCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "RemoveAxiom"
    }
}

/
#[derive(Debug, Clone)]
pub struct SaveOntologyCommand {
    pub classes: Vec<OwlClass>,
    pub properties: Vec<OwlProperty>,
    pub axioms: Vec<OwlAxiom>,
}

impl Command for SaveOntologyCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "SaveOntology"
    }
}

/
#[derive(Debug, Clone)]
pub struct SaveOntologyGraphCommand {
    pub graph: GraphData,
}

impl Command for SaveOntologyGraphCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "SaveOntologyGraph"
    }
}

/
#[derive(Debug, Clone)]
pub struct StoreInferenceResultsCommand {
    pub results: InferenceResults,
}

impl Command for StoreInferenceResultsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "StoreInferenceResults"
    }
}

/
#[derive(Debug, Clone)]
pub struct ImportOntologyCommand {
    pub owl_xml: String,
}

impl Command for ImportOntologyCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "ImportOntology"
    }

    fn validate(&self) -> Result<()> {
        if self.owl_xml.is_empty() {
            return Err(anyhow::anyhow!("OWL XML cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct CacheSsspResultCommand {
    pub entry: PathfindingCacheEntry,
}

impl Command for CacheSsspResultCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "CacheSsspResult"
    }
}

/
#[derive(Debug, Clone)]
pub struct CacheApspResultCommand {
    pub distance_matrix: Vec<Vec<f32>>,
}

impl Command for CacheApspResultCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "CacheApspResult"
    }

    fn validate(&self) -> Result<()> {
        if self.distance_matrix.is_empty() {
            return Err(anyhow::anyhow!("Distance matrix cannot be empty"));
        }
        Ok(())
    }
}

/
#[derive(Debug, Clone)]
pub struct InvalidatePathfindingCachesCommand;

impl Command for InvalidatePathfindingCachesCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "InvalidatePathfindingCaches"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::ontology_repository::PropertyType;

    #[test]
    fn test_add_class_validation() {
        let class = OwlClass {
            iri: "http://example.org/Class1".to_string(),
            label: Some("Class 1".to_string()),
            description: None,
            parent_classes: vec![],
            properties: Default::default(),
            source_file: None,
        };
        let cmd = AddClassCommand { class };
        assert!(cmd.validate().is_ok());

        let class = OwlClass {
            iri: "".to_string(),
            label: None,
            description: None,
            parent_classes: vec![],
            properties: Default::default(),
            source_file: None,
        };
        let cmd = AddClassCommand { class };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_add_property_validation() {
        let property = OwlProperty {
            iri: "http://example.org/hasProperty".to_string(),
            label: Some("Has Property".to_string()),
            property_type: PropertyType::ObjectProperty,
            domain: vec![],
            range: vec![],
        };
        let cmd = AddPropertyCommand { property };
        assert!(cmd.validate().is_ok());
    }

    #[test]
    fn test_add_axiom_validation() {
        use crate::ports::ontology_repository::AxiomType;

        let axiom = OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "Class1".to_string(),
            object: "Class2".to_string(),
            annotations: Default::default(),
        };
        let cmd = AddAxiomCommand { axiom };
        assert!(cmd.validate().is_ok());
    }
}
