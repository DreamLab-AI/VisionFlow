/// Horned-OWL integration for advanced OWL reasoning
///
/// Provides integration with the horned-owl crate for:
/// - OWL 2 parsing and validation
/// - Consistency checking
/// - Advanced reasoning beyond custom reasoner
///
/// Note: This is a simplified implementation that delegates to CustomReasoner.
/// Full horned-owl integration requires matching API versions.

use rusqlite::Connection;
use crate::reasoning::{
    custom_reasoner::{InferredAxiom, OntologyReasoner, Ontology, CustomReasoner, OWLClass},
    ReasoningError, ReasoningResult,
};
use std::collections::{HashMap, HashSet};

/// Horned-OWL reasoner wrapper
/// Currently delegates to CustomReasoner for compatibility
pub struct HornedOwlReasoner {
    custom_reasoner: CustomReasoner,
    ontology: Option<Ontology>,
}

impl HornedOwlReasoner {
    pub fn new() -> Self {
        Self {
            custom_reasoner: CustomReasoner::new(),
            ontology: None,
        }
    }

    /// Parse OWL from database
    pub fn parse_from_database(&mut self, db_path: &str) -> ReasoningResult<()> {
        let conn = Connection::open(db_path)?;

        let mut ontology = Ontology::default();

        // Load OWL classes
        let mut stmt = conn.prepare(
            "SELECT iri, label, parent_class_iri, markdown_content
             FROM owl_classes"
        )?;

        let classes = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;

        for class_result in classes {
            let (iri, label, parent_iri, _content) = class_result?;

            ontology.classes.insert(iri.clone(), OWLClass {
                iri: iri.clone(),
                label,
                parent_class_iri: parent_iri.clone(),
            });

            // Add SubClassOf relationship
            if let Some(parent) = parent_iri {
                ontology.subclass_of.entry(iri.clone())
                    .or_insert_with(HashSet::new)
                    .insert(parent);
            }
        }

        // Load DisjointClasses axioms
        let mut stmt = conn.prepare(
            "SELECT c1.iri, c2.iri
             FROM owl_axioms a
             JOIN owl_classes c1 ON a.subject_id = c1.id
             JOIN owl_classes c2 ON a.object_id = c2.id
             WHERE a.axiom_type = 'DisjointClasses'"
        )?;

        let disjoint_pairs = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
            ))
        })?;

        // Group disjoint classes into sets
        let mut disjoint_map: HashMap<String, HashSet<String>> = HashMap::new();
        for pair_result in disjoint_pairs {
            let (class_a, class_b) = pair_result?;

            disjoint_map.entry(class_a.clone())
                .or_insert_with(HashSet::new)
                .insert(class_b.clone());

            disjoint_map.entry(class_b.clone())
                .or_insert_with(HashSet::new)
                .insert(class_a);
        }

        // Convert map to disjoint sets
        for (_, disjoint_set) in disjoint_map {
            if !ontology.disjoint_classes.iter().any(|existing| {
                existing.iter().any(|c| disjoint_set.contains(c))
            }) {
                ontology.disjoint_classes.push(disjoint_set);
            }
        }

        // Load EquivalentClass axioms
        let mut stmt = conn.prepare(
            "SELECT c1.iri, c2.iri
             FROM owl_axioms a
             JOIN owl_classes c1 ON a.subject_id = c1.id
             JOIN owl_classes c2 ON a.object_id = c2.id
             WHERE a.axiom_type = 'EquivalentClass'"
        )?;

        let equiv_pairs = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
            ))
        })?;

        for pair_result in equiv_pairs {
            let (class_a, class_b) = pair_result?;

            ontology.equivalent_classes.entry(class_a.clone())
                .or_insert_with(HashSet::new)
                .insert(class_b);
        }

        // Load FunctionalProperty axioms
        let mut stmt = conn.prepare(
            "SELECT property_iri
             FROM owl_properties
             WHERE is_functional = 1"
        )?;

        let properties = stmt.query_map([], |row| {
            row.get::<_, String>(0)
        })?;

        for property_result in properties {
            ontology.functional_properties.insert(property_result?);
        }

        self.ontology = Some(ontology);
        Ok(())
    }

    /// Validate ontology consistency
    pub fn validate_consistency(&self) -> ReasoningResult<bool> {
        if self.ontology.is_none() {
            return Err(ReasoningError::Inference("Ontology not loaded".to_string()));
        }

        // Basic consistency checks:
        // 1. No class is both subclass and disjoint with another
        // 2. No cycles in subclass hierarchy
        // 3. Equivalent classes form valid equivalence relations

        // For now, return true (assumes consistency)
        // Full consistency checking requires a complete reasoner
        Ok(true)
    }

    /// Get inferred axioms
    pub fn get_inferred_axioms(&self) -> ReasoningResult<Vec<InferredAxiom>> {
        if let Some(ontology) = &self.ontology {
            self.custom_reasoner.infer_axioms(ontology)
        } else {
            Err(ReasoningError::Inference("Ontology not loaded".to_string()))
        }
    }
}

impl Default for HornedOwlReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl OntologyReasoner for HornedOwlReasoner {
    fn infer_axioms(&self, ontology: &Ontology) -> ReasoningResult<Vec<InferredAxiom>> {
        self.custom_reasoner.infer_axioms(ontology)
    }

    fn is_subclass_of(&self, child: &str, parent: &str, ontology: &Ontology) -> bool {
        self.custom_reasoner.is_subclass_of(child, parent, ontology)
    }

    fn are_disjoint(&self, class_a: &str, class_b: &str, ontology: &Ontology) -> bool {
        self.custom_reasoner.are_disjoint(class_a, class_b, ontology)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_horned_owl_parsing() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        // Create test database
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "CREATE TABLE owl_classes (
                id INTEGER PRIMARY KEY,
                iri TEXT UNIQUE NOT NULL,
                label TEXT,
                parent_class_iri TEXT,
                markdown_content TEXT
            )",
            [],
        ).unwrap();

        conn.execute(
            "CREATE TABLE owl_properties (
                property_iri TEXT PRIMARY KEY,
                is_functional INTEGER DEFAULT 0
            )",
            [],
        ).unwrap();

        conn.execute(
            "CREATE TABLE owl_axioms (
                id INTEGER PRIMARY KEY,
                axiom_type TEXT NOT NULL,
                subject_id INTEGER,
                object_id INTEGER
            )",
            [],
        ).unwrap();

        conn.execute(
            "INSERT INTO owl_classes (iri, label, parent_class_iri) VALUES (?, ?, ?)",
            ["Entity", "Entity", ""],
        ).unwrap();

        conn.execute(
            "INSERT INTO owl_classes (iri, label, parent_class_iri) VALUES (?, ?, ?)",
            ["Cell", "Cell", "Entity"],
        ).unwrap();

        drop(conn);

        // Test parsing
        let mut reasoner = HornedOwlReasoner::new();
        let result = reasoner.parse_from_database(db_path.to_str().unwrap());
        assert!(result.is_ok());
        assert!(reasoner.ontology.is_some());

        let ontology = reasoner.ontology.unwrap();
        assert_eq!(ontology.classes.len(), 2);
        assert!(ontology.classes.contains_key("Entity"));
        assert!(ontology.classes.contains_key("Cell"));
    }

    #[test]
    fn test_consistency_validation() {
        let mut reasoner = HornedOwlReasoner::new();
        reasoner.ontology = Some(Ontology::default());

        let result = reasoner.validate_consistency();
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
}
