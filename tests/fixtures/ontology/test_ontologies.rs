// Test Ontology Fixtures
// Provides sample ontologies with known hierarchies for testing

use std::collections::{HashMap, HashSet};
use webxr::reasoning::custom_reasoner::{Ontology, OWLClass};

/// Create a simple ontology with basic hierarchy
/// Entity -> MaterialEntity -> Cell -> {Neuron, Astrocyte}
/// Neuron and Astrocyte are disjoint
pub fn create_simple_hierarchy() -> Ontology {
    let mut ontology = Ontology::default();

    // Classes
    ontology.classes.insert("Entity".to_string(), OWLClass {
        iri: "http://example.org/Entity".to_string(),
        label: Some("Entity".to_string()),
        parent_class_iri: None,
    });

    ontology.classes.insert("MaterialEntity".to_string(), OWLClass {
        iri: "http://example.org/MaterialEntity".to_string(),
        label: Some("Material Entity".to_string()),
        parent_class_iri: Some("Entity".to_string()),
    });

    ontology.classes.insert("Cell".to_string(), OWLClass {
        iri: "http://example.org/Cell".to_string(),
        label: Some("Cell".to_string()),
        parent_class_iri: Some("MaterialEntity".to_string()),
    });

    ontology.classes.insert("Neuron".to_string(), OWLClass {
        iri: "http://example.org/Neuron".to_string(),
        label: Some("Neuron".to_string()),
        parent_class_iri: Some("Cell".to_string()),
    });

    ontology.classes.insert("Astrocyte".to_string(), OWLClass {
        iri: "http://example.org/Astrocyte".to_string(),
        label: Some("Astrocyte".to_string()),
        parent_class_iri: Some("Cell".to_string()),
    });

    // SubClass relationships
    ontology.subclass_of.insert(
        "MaterialEntity".to_string(),
        vec!["Entity".to_string()].into_iter().collect(),
    );
    ontology.subclass_of.insert(
        "Cell".to_string(),
        vec!["MaterialEntity".to_string()].into_iter().collect(),
    );
    ontology.subclass_of.insert(
        "Neuron".to_string(),
        vec!["Cell".to_string()].into_iter().collect(),
    );
    ontology.subclass_of.insert(
        "Astrocyte".to_string(),
        vec!["Cell".to_string()].into_iter().collect(),
    );

    // Disjoint classes
    ontology.disjoint_classes.push(
        vec!["Neuron".to_string(), "Astrocyte".to_string()]
            .into_iter()
            .collect(),
    );

    ontology
}

/// Create an ontology with complex transitive relationships
pub fn create_deep_hierarchy() -> Ontology {
    let mut ontology = Ontology::default();

    // 5-level hierarchy
    let levels = vec![
        ("Level0", None),
        ("Level1", Some("Level0")),
        ("Level2", Some("Level1")),
        ("Level3", Some("Level2")),
        ("Level4", Some("Level3")),
    ];

    for (class_name, parent) in levels {
        ontology.classes.insert(class_name.to_string(), OWLClass {
            iri: format!("http://example.org/{}", class_name),
            label: Some(class_name.to_string()),
            parent_class_iri: parent.map(|s| s.to_string()),
        });

        if let Some(parent_name) = parent {
            ontology.subclass_of.insert(
                class_name.to_string(),
                vec![parent_name.to_string()].into_iter().collect(),
            );
        }
    }

    ontology
}

/// Create an ontology with multiple disjoint sets
pub fn create_multiple_disjoint() -> Ontology {
    let mut ontology = Ontology::default();

    // Base class
    ontology.classes.insert("Thing".to_string(), OWLClass {
        iri: "http://example.org/Thing".to_string(),
        label: Some("Thing".to_string()),
        parent_class_iri: None,
    });

    // Set 1: Colors
    let colors = vec!["Red", "Green", "Blue"];
    for color in &colors {
        ontology.classes.insert(color.to_string(), OWLClass {
            iri: format!("http://example.org/{}", color),
            label: Some(color.to_string()),
            parent_class_iri: Some("Thing".to_string()),
        });
        ontology.subclass_of.insert(
            color.to_string(),
            vec!["Thing".to_string()].into_iter().collect(),
        );
    }
    ontology.disjoint_classes.push(
        colors.iter().map(|s| s.to_string()).collect(),
    );

    // Set 2: Shapes
    let shapes = vec!["Circle", "Square", "Triangle"];
    for shape in &shapes {
        ontology.classes.insert(shape.to_string(), OWLClass {
            iri: format!("http://example.org/{}", shape),
            label: Some(shape.to_string()),
            parent_class_iri: Some("Thing".to_string()),
        });
        ontology.subclass_of.insert(
            shape.to_string(),
            vec!["Thing".to_string()].into_iter().collect(),
        );
    }
    ontology.disjoint_classes.push(
        shapes.iter().map(|s| s.to_string()).collect(),
    );

    ontology
}

/// Create an ontology with equivalent classes
pub fn create_equivalent_classes() -> Ontology {
    let mut ontology = Ontology::default();

    let classes = vec!["Person", "Human", "Individual"];
    for class_name in &classes {
        ontology.classes.insert(class_name.to_string(), OWLClass {
            iri: format!("http://example.org/{}", class_name),
            label: Some(class_name.to_string()),
            parent_class_iri: None,
        });
    }

    // Person ≡ Human
    ontology.equivalent_classes.insert(
        "Person".to_string(),
        vec!["Human".to_string()].into_iter().collect(),
    );

    // Human ≡ Individual
    ontology.equivalent_classes.insert(
        "Human".to_string(),
        vec!["Individual".to_string()].into_iter().collect(),
    );

    ontology
}

/// Create an ontology with diamond pattern (multiple inheritance paths)
pub fn create_diamond_pattern() -> Ontology {
    let mut ontology = Ontology::default();

    //       Top
    //      /   \
    //   Left   Right
    //      \   /
    //      Bottom

    ontology.classes.insert("Top".to_string(), OWLClass {
        iri: "http://example.org/Top".to_string(),
        label: Some("Top".to_string()),
        parent_class_iri: None,
    });

    ontology.classes.insert("Left".to_string(), OWLClass {
        iri: "http://example.org/Left".to_string(),
        label: Some("Left".to_string()),
        parent_class_iri: Some("Top".to_string()),
    });

    ontology.classes.insert("Right".to_string(), OWLClass {
        iri: "http://example.org/Right".to_string(),
        label: Some("Right".to_string()),
        parent_class_iri: Some("Top".to_string()),
    });

    ontology.classes.insert("Bottom".to_string(), OWLClass {
        iri: "http://example.org/Bottom".to_string(),
        label: Some("Bottom".to_string()),
        parent_class_iri: None, // Has multiple parents
    });

    // SubClass relationships
    ontology.subclass_of.insert(
        "Left".to_string(),
        vec!["Top".to_string()].into_iter().collect(),
    );
    ontology.subclass_of.insert(
        "Right".to_string(),
        vec!["Top".to_string()].into_iter().collect(),
    );
    ontology.subclass_of.insert(
        "Bottom".to_string(),
        vec!["Left".to_string(), "Right".to_string()].into_iter().collect(),
    );

    ontology
}

/// Create an ontology with functional properties
pub fn create_functional_properties() -> Ontology {
    let mut ontology = Ontology::default();

    ontology.classes.insert("Person".to_string(), OWLClass {
        iri: "http://example.org/Person".to_string(),
        label: Some("Person".to_string()),
        parent_class_iri: None,
    });

    // hasAge is functional (each person has exactly one age)
    ontology.functional_properties.insert("hasAge".to_string());
    // hasBirthDate is functional
    ontology.functional_properties.insert("hasBirthDate".to_string());

    ontology
}

/// Create an empty ontology for edge case testing
pub fn create_empty_ontology() -> Ontology {
    Ontology::default()
}

/// Create a large ontology for performance testing
pub fn create_large_ontology(num_classes: usize) -> Ontology {
    let mut ontology = Ontology::default();

    // Root class
    ontology.classes.insert("Root".to_string(), OWLClass {
        iri: "http://example.org/Root".to_string(),
        label: Some("Root".to_string()),
        parent_class_iri: None,
    });

    // Generate classes in a tree structure
    for i in 0..num_classes {
        let class_name = format!("Class{}", i);
        let parent = if i == 0 {
            "Root".to_string()
        } else {
            format!("Class{}", i / 2)
        };

        ontology.classes.insert(class_name.clone(), OWLClass {
            iri: format!("http://example.org/{}", class_name),
            label: Some(class_name.clone()),
            parent_class_iri: Some(parent.clone()),
        });

        ontology.subclass_of.insert(
            class_name,
            vec![parent].into_iter().collect(),
        );
    }

    ontology
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_hierarchy() {
        let ontology = create_simple_hierarchy();
        assert_eq!(ontology.classes.len(), 5);
        assert_eq!(ontology.subclass_of.len(), 4);
        assert_eq!(ontology.disjoint_classes.len(), 1);
    }

    #[test]
    fn test_deep_hierarchy() {
        let ontology = create_deep_hierarchy();
        assert_eq!(ontology.classes.len(), 5);
        assert_eq!(ontology.subclass_of.len(), 4);
    }

    #[test]
    fn test_multiple_disjoint() {
        let ontology = create_multiple_disjoint();
        assert_eq!(ontology.disjoint_classes.len(), 2);
    }

    #[test]
    fn test_equivalent_classes() {
        let ontology = create_equivalent_classes();
        assert_eq!(ontology.equivalent_classes.len(), 2);
    }

    #[test]
    fn test_diamond_pattern() {
        let ontology = create_diamond_pattern();
        let bottom_parents = ontology.subclass_of.get("Bottom").unwrap();
        assert_eq!(bottom_parents.len(), 2);
    }

    #[test]
    fn test_large_ontology() {
        let ontology = create_large_ontology(100);
        assert!(ontology.classes.len() > 100);
    }
}
