//! Triple representation and entity/relation types

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use utils::{AutoSchemaError, Result};

/// A semantic triple representing a fact in the knowledge graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triple {
    pub id: String,
    pub subject: Entity,
    pub predicate: Relation,
    pub object: Entity,
    pub confidence: f32,
    pub source: String,
    pub metadata: std::collections::HashMap<String, String>,
}

/// An entity in the knowledge graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub label: String,
    pub entity_type: EntityType,
    pub properties: std::collections::HashMap<String, String>,
    pub aliases: Vec<String>,
}

/// A relation between two entities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    pub id: String,
    pub label: String,
    pub relation_type: RelationType,
    pub properties: std::collections::HashMap<String, String>,
}

/// Types of entities that can be recognized
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Event,
    Concept,
    Product,
    Unknown,
}

/// Types of relations between entities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationType {
    IsA,
    PartOf,
    LocatedIn,
    WorksFor,
    CreatedBy,
    RelatedTo,
    Causes,
    Before,
    After,
    Similar,
    Unknown,
}

impl Triple {
    /// Create a new triple with generated ID
    pub fn new(
        subject: Entity,
        predicate: Relation,
        object: Entity,
        confidence: f32,
        source: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            subject,
            predicate,
            object,
            confidence,
            source,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Validate the triple
    pub fn validate(&self) -> Result<()> {
        if self.subject.label.trim().is_empty() {
            return Err(AutoSchemaError::validation("subject", "Subject label cannot be empty"));
        }

        if self.object.label.trim().is_empty() {
            return Err(AutoSchemaError::validation("object", "Object label cannot be empty"));
        }

        if self.predicate.label.trim().is_empty() {
            return Err(AutoSchemaError::validation("predicate", "Predicate label cannot be empty"));
        }

        if self.confidence < 0.0 || self.confidence > 1.0 {
            return Err(AutoSchemaError::validation("confidence", "Confidence must be between 0.0 and 1.0"));
        }

        Ok(())
    }

    /// Get a canonical string representation
    pub fn to_canonical_string(&self) -> String {
        format!("({}, {}, {})", self.subject.label, self.predicate.label, self.object.label)
    }
}

impl Entity {
    /// Create a new entity with generated ID
    pub fn new(label: String, entity_type: EntityType) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            label,
            entity_type,
            properties: std::collections::HashMap::new(),
            aliases: Vec::new(),
        }
    }

    /// Add an alias to the entity
    pub fn add_alias<S: Into<String>>(&mut self, alias: S) {
        let alias = alias.into();
        if !self.aliases.contains(&alias) {
            self.aliases.push(alias);
        }
    }

    /// Add a property to the entity
    pub fn add_property<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.properties.insert(key.into(), value.into());
    }
}

impl Relation {
    /// Create a new relation with generated ID
    pub fn new(label: String, relation_type: RelationType) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            label,
            relation_type,
            properties: std::collections::HashMap::new(),
        }
    }

    /// Add a property to the relation
    pub fn add_property<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.properties.insert(key.into(), value.into());
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EntityType::Person => "Person",
            EntityType::Organization => "Organization",
            EntityType::Location => "Location",
            EntityType::Event => "Event",
            EntityType::Concept => "Concept",
            EntityType::Product => "Product",
            EntityType::Unknown => "Unknown",
        };
        write!(f, "{}", s)
    }
}

impl std::fmt::Display for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            RelationType::IsA => "isA",
            RelationType::PartOf => "partOf",
            RelationType::LocatedIn => "locatedIn",
            RelationType::WorksFor => "worksFor",
            RelationType::CreatedBy => "createdBy",
            RelationType::RelatedTo => "relatedTo",
            RelationType::Causes => "causes",
            RelationType::Before => "before",
            RelationType::After => "after",
            RelationType::Similar => "similar",
            RelationType::Unknown => "unknown",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_creation() {
        let subject = Entity::new("Alice".to_string(), EntityType::Person);
        let predicate = Relation::new("worksFor".to_string(), RelationType::WorksFor);
        let object = Entity::new("Company".to_string(), EntityType::Organization);

        let triple = Triple::new(subject, predicate, object, 0.9, "test".to_string());
        assert!(triple.validate().is_ok());
        assert_eq!(triple.confidence, 0.9);
    }

    #[test]
    fn test_entity_aliases() {
        let mut entity = Entity::new("Alice".to_string(), EntityType::Person);
        entity.add_alias("Alice Smith");
        entity.add_alias("A. Smith");

        assert_eq!(entity.aliases.len(), 2);
        assert!(entity.aliases.contains(&"Alice Smith".to_string()));
    }

    #[test]
    fn test_triple_validation() {
        let subject = Entity::new("".to_string(), EntityType::Person);
        let predicate = Relation::new("worksFor".to_string(), RelationType::WorksFor);
        let object = Entity::new("Company".to_string(), EntityType::Organization);

        let triple = Triple::new(subject, predicate, object, 0.9, "test".to_string());
        assert!(triple.validate().is_err());
    }
}