//! Concept generation and semantic clustering

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utils::{Result, UtilsError};

/// A concept represents a semantic cluster of related entities/terms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: String,
    pub label: String,
    pub description: String,
    pub keywords: Vec<String>,
    pub related_entities: Vec<String>,
    pub confidence: f32,
    pub category: ConceptCategory,
}

/// Categories of concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptCategory {
    Domain,      // Domain-specific concept
    Abstract,    // Abstract concept
    Concrete,    // Concrete concept
    Process,     // Process or action
    Attribute,   // Property or attribute
    Unknown,
}

/// Generates concepts from extracted knowledge
pub struct ConceptGenerator {
    similarity_threshold: f32,
    min_cluster_size: usize,
}

impl ConceptGenerator {
    /// Create a new concept generator
    pub fn new(similarity_threshold: f32, min_cluster_size: usize) -> Self {
        Self {
            similarity_threshold,
            min_cluster_size,
        }
    }

    /// Generate concepts from a knowledge graph
    pub fn generate_concepts(&self, graph: &crate::graph::KnowledgeGraph) -> Result<Vec<Concept>> {
        // Simple concept generation based on entity clustering
        let mut concepts = Vec::new();

        // Group entities by type
        let mut entity_groups: HashMap<String, Vec<&crate::triple::Entity>> = HashMap::new();

        for entity in graph.entities.values() {
            let type_key = format!("{}", entity.entity_type);
            entity_groups.entry(type_key).or_default().push(entity);
        }

        // Generate concepts for each entity group
        for (entity_type, entities) in entity_groups {
            if entities.len() >= self.min_cluster_size {
                let concept = self.create_concept_from_entities(&entity_type, &entities)?;
                concepts.push(concept);
            }
        }

        Ok(concepts)
    }

    /// Create a concept from a group of entities
    fn create_concept_from_entities(
        &self,
        entity_type: &str,
        entities: &[&crate::triple::Entity],
    ) -> Result<Concept> {
        let concept_id = uuid::Uuid::new_v4().to_string();
        let label = format!("{} Concept", entity_type);
        let description = format!("Concept representing {} entities", entity_type);

        // Extract keywords from entity labels
        let mut keywords = Vec::new();
        for entity in entities {
            let words: Vec<&str> = entity.label.split_whitespace().collect();
            for word in words {
                if word.len() > 2 && !keywords.contains(&word.to_lowercase()) {
                    keywords.push(word.to_lowercase());
                }
            }
        }

        // Get entity IDs
        let related_entities: Vec<String> = entities.iter().map(|e| e.id.clone()).collect();

        // Calculate confidence based on entity count and keyword coherence
        let confidence = (entities.len() as f32 / 10.0).min(1.0) * 0.8;

        let category = self.categorize_concept(entity_type);

        Ok(Concept {
            id: concept_id,
            label,
            description,
            keywords,
            related_entities,
            confidence,
            category,
        })
    }

    /// Categorize a concept based on entity type
    fn categorize_concept(&self, entity_type: &str) -> ConceptCategory {
        match entity_type.to_lowercase().as_str() {
            "person" | "organization" => ConceptCategory::Concrete,
            "location" => ConceptCategory::Concrete,
            "event" => ConceptCategory::Process,
            "concept" => ConceptCategory::Abstract,
            "product" => ConceptCategory::Concrete,
            _ => ConceptCategory::Unknown,
        }
    }
}

impl Default for ConceptGenerator {
    fn default() -> Self {
        Self::new(0.7, 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::KnowledgeGraph;
    use crate::triple::{Entity, EntityType};

    #[test]
    fn test_concept_generation() {
        let generator = ConceptGenerator::default();
        let mut graph = KnowledgeGraph::new();

        // Add some entities
        let person1 = Entity::new("Alice".to_string(), EntityType::Person);
        let person2 = Entity::new("Bob".to_string(), EntityType::Person);
        let person3 = Entity::new("Charlie".to_string(), EntityType::Person);

        graph.entities.insert(person1.id.clone(), person1);
        graph.entities.insert(person2.id.clone(), person2);
        graph.entities.insert(person3.id.clone(), person3);

        let concepts = generator.generate_concepts(&graph).unwrap();
        assert_eq!(concepts.len(), 1); // Should create one Person concept
        assert_eq!(concepts[0].related_entities.len(), 3);
    }

    #[test]
    fn test_concept_categorization() {
        let generator = ConceptGenerator::default();

        assert!(matches!(generator.categorize_concept("Person"), ConceptCategory::Concrete));
        assert!(matches!(generator.categorize_concept("Event"), ConceptCategory::Process));
        assert!(matches!(generator.categorize_concept("Concept"), ConceptCategory::Abstract));
    }
}