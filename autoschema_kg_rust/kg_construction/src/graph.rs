//! Knowledge graph data structure and operations

use crate::triple::{Triple, Entity, Relation};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use utils::{Result, UtilsError};

/// A knowledge graph containing entities, relations, and triples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub entities: HashMap<String, Entity>,
    pub relations: HashMap<String, Relation>,
    pub triples: Vec<Triple>,
    pub metadata: HashMap<String, String>,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            relations: HashMap::new(),
            triples: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a triple to the knowledge graph
    pub fn add_triple(&mut self, triple: Triple) -> Result<()> {
        // Validate the triple
        triple.validate()?;

        // Add entities and relations if they don't exist
        self.entities.entry(triple.subject.id.clone())
            .or_insert_with(|| triple.subject.clone());

        self.entities.entry(triple.object.id.clone())
            .or_insert_with(|| triple.object.clone());

        self.relations.entry(triple.predicate.id.clone())
            .or_insert_with(|| triple.predicate.clone());

        // Add the triple
        self.triples.push(triple);

        Ok(())
    }

    /// Add multiple triples to the knowledge graph
    pub fn add_triples(&mut self, triples: Vec<Triple>) -> Result<()> {
        for triple in triples {
            self.add_triple(triple)?;
        }
        Ok(())
    }

    /// Get all triples involving a specific entity
    pub fn get_triples_for_entity(&self, entity_id: &str) -> Vec<&Triple> {
        self.triples
            .iter()
            .filter(|t| t.subject.id == entity_id || t.object.id == entity_id)
            .collect()
    }

    /// Get all entities connected to a specific entity
    pub fn get_connected_entities(&self, entity_id: &str) -> HashSet<&Entity> {
        let mut connected = HashSet::new();

        for triple in &self.triples {
            if triple.subject.id == entity_id {
                if let Some(entity) = self.entities.get(&triple.object.id) {
                    connected.insert(entity);
                }
            } else if triple.object.id == entity_id {
                if let Some(entity) = self.entities.get(&triple.subject.id) {
                    connected.insert(entity);
                }
            }
        }

        connected
    }

    /// Find entities by label (case-insensitive)
    pub fn find_entities_by_label(&self, label: &str) -> Vec<&Entity> {
        let label_lower = label.to_lowercase();
        self.entities
            .values()
            .filter(|e| e.label.to_lowercase().contains(&label_lower))
            .collect()
    }

    /// Get entity and relation statistics
    pub fn get_statistics(&self) -> GraphStatistics {
        let entity_types = self.entities
            .values()
            .map(|e| format!("{}", e.entity_type))
            .fold(HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });

        let relation_types = self.relations
            .values()
            .map(|r| format!("{}", r.relation_type))
            .fold(HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });

        let avg_confidence = if self.triples.is_empty() {
            0.0
        } else {
            self.triples.iter().map(|t| t.confidence).sum::<f32>() / self.triples.len() as f32
        };

        GraphStatistics {
            entity_count: self.entities.len(),
            relation_count: self.relations.len(),
            triple_count: self.triples.len(),
            entity_types,
            relation_types,
            average_confidence: avg_confidence,
        }
    }

    /// Merge another knowledge graph into this one
    pub fn merge(&mut self, other: KnowledgeGraph) -> Result<()> {
        // Merge entities (avoid duplicates by ID)
        for (id, entity) in other.entities {
            self.entities.entry(id).or_insert(entity);
        }

        // Merge relations (avoid duplicates by ID)
        for (id, relation) in other.relations {
            self.relations.entry(id).or_insert(relation);
        }

        // Add triples (check for duplicates based on canonical string)
        let existing_triples: HashSet<String> = self.triples
            .iter()
            .map(|t| t.to_canonical_string())
            .collect();

        for triple in other.triples {
            let canonical = triple.to_canonical_string();
            if !existing_triples.contains(&canonical) {
                self.triples.push(triple);
            }
        }

        // Merge metadata
        for (key, value) in other.metadata {
            self.metadata.entry(key).or_insert(value);
        }

        Ok(())
    }

    /// Export to Neo4j Cypher format
    pub fn to_cypher(&self) -> String {
        let mut cypher = String::new();

        // Create nodes for entities
        for entity in self.entities.values() {
            cypher.push_str(&format!(
                "CREATE (e_{}: {} {{label: '{}', id: '{}'}})\n",
                entity.id.replace("-", "_"),
                entity.entity_type,
                entity.label.replace("'", "\\'"),
                entity.id
            ));
        }

        cypher.push('\n');

        // Create relationships
        for triple in &self.triples {
            cypher.push_str(&format!(
                "CREATE (e_{})-[:{}]->(e_{})\n",
                triple.subject.id.replace("-", "_"),
                triple.predicate.relation_type,
                triple.object.id.replace("-", "_")
            ));
        }

        cypher
    }

    /// Count methods
    pub fn entity_count(&self) -> usize { self.entities.len() }
    pub fn relation_count(&self) -> usize { self.relations.len() }
    pub fn triple_count(&self) -> usize { self.triples.len() }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub entity_count: usize,
    pub relation_count: usize,
    pub triple_count: usize,
    pub entity_types: HashMap<String, usize>,
    pub relation_types: HashMap<String, usize>,
    pub average_confidence: f32,
}

/// Builder for constructing knowledge graphs
pub struct GraphBuilder {
    graph: KnowledgeGraph,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            graph: KnowledgeGraph::new(),
        }
    }

    /// Add triples to the graph being built
    pub fn add_triples(&mut self, triples: Vec<Triple>) -> Result<()> {
        self.graph.add_triples(triples)
    }

    /// Link entities with similar names (simple entity linking)
    pub fn link_entities(&mut self) -> Result<()> {
        // Simple entity linking based on label similarity
        let entities: Vec<Entity> = self.graph.entities.values().cloned().collect();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];

                if entities_are_similar(entity1, entity2) {
                    // Create a "similar" relationship
                    if let (Some(e1), Some(e2)) = (
                        self.graph.entities.get(&entity1.id),
                        self.graph.entities.get(&entity2.id)
                    ) {
                        let similar_relation = Relation::new(
                            "similar".to_string(),
                            crate::triple::RelationType::Similar,
                        );

                        let triple = Triple::new(
                            e1.clone(),
                            similar_relation,
                            e2.clone(),
                            0.8, // Default similarity confidence
                            "entity_linking".to_string(),
                        );

                        self.graph.add_triple(triple)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Build the final knowledge graph
    pub fn build(self) -> Result<KnowledgeGraph> {
        Ok(self.graph)
    }

    /// Reset the builder for reuse
    pub fn reset(&mut self) {
        self.graph = KnowledgeGraph::new();
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if two entities are similar (simple implementation)
fn entities_are_similar(entity1: &Entity, entity2: &Entity) -> bool {
    if entity1.entity_type != entity2.entity_type {
        return false;
    }

    // Check if labels are very similar
    let label1 = entity1.label.to_lowercase();
    let label2 = entity2.label.to_lowercase();

    // Simple edit distance check (Levenshtein distance)
    let similarity = string_similarity(&label1, &label2);
    similarity > 0.8
}

/// Calculate simple string similarity (normalized Levenshtein distance)
fn string_similarity(s1: &str, s2: &str) -> f32 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }

    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let max_len = len1.max(len2);
    let distance = levenshtein_distance(s1, s2);

    1.0 - (distance as f32 / max_len as f32)
}

/// Calculate Levenshtein distance between two strings
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();
    let len1 = chars1.len();
    let len2 = chars2.len();

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }

    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }

    matrix[len1][len2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triple::{EntityType, RelationType};

    #[test]
    fn test_knowledge_graph_creation() {
        let graph = KnowledgeGraph::new();
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.triple_count(), 0);
    }

    #[test]
    fn test_add_triple() {
        let mut graph = KnowledgeGraph::new();
        let triple = create_test_triple();

        graph.add_triple(triple).unwrap();
        assert_eq!(graph.entity_count(), 2); // subject and object
        assert_eq!(graph.relation_count(), 1);
        assert_eq!(graph.triple_count(), 1);
    }

    #[test]
    fn test_graph_statistics() {
        let mut graph = KnowledgeGraph::new();
        graph.add_triple(create_test_triple()).unwrap();

        let stats = graph.get_statistics();
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.triple_count, 1);
        assert!(stats.average_confidence > 0.0);
    }

    #[test]
    fn test_string_similarity() {
        assert!(string_similarity("hello", "hello") > 0.9);
        assert!(string_similarity("hello", "helo") > 0.7);
        assert!(string_similarity("hello", "world") < 0.5);
    }

    fn create_test_triple() -> Triple {
        let subject = Entity::new("Alice".to_string(), EntityType::Person);
        let predicate = Relation::new("worksFor".to_string(), RelationType::WorksFor);
        let object = Entity::new("Company".to_string(), EntityType::Organization);

        Triple::new(subject, predicate, object, 0.9, "test".to_string())
    }
}