# Horned-OWL Comprehensive Research Guide

**Version**: 1.2.0 (Latest Stable)
**License**: LGPL-3.0
**Repository**: https://github.com/phillord/horned-owl
**Documentation**: https://docs.rs/horned-owl/latest/
**Research Date**: 2025-10-22

---

## Table of Contents

1. [Overview](#overview)
2. [Performance Characteristics](#performance-characteristics)
3. [Architecture & Design](#architecture--design)
4. [Core Modules](#core-modules)
5. [Type System](#type-system)
6. [Parsing API](#parsing-api)
7. [Ontology Building](#ontology-building)
8. [Querying Ontologies](#querying-ontologies)
9. [Integration Patterns](#integration-patterns)
10. [Database Schema Integration](#database-schema-integration)
11. [Whelk-rs Reasoning Integration](#whelk-rs-reasoning-integration)
12. [Code Examples](#code-examples)
13. [Ecosystem Components](#ecosystem-components)

---

## Overview

Horned-OWL is a high-performance Rust library for parsing, generating, and manipulating OWL (Web Ontology Language) ontologies. It provides a complete implementation of OWL 2 with support for multiple serialization formats.

### Key Features

- **OWL 2 Compliance**: Full W3C OWL 2 Recommendation support
- **SWRL Support**: Semantic Web Rule Language implementation
- **Multiple Formats**: RDF/XML, OWL/XML, Functional Syntax, Turtle
- **High Performance**: 20x-40x faster than OWL API for validation tasks
- **Scalability**: Designed for ontologies with millions of terms
- **Rust Edition 2024**: Latest language features and optimizations
- **Punning Support**: Complete entity punning across all syntaxes (v1.2.0)

### Design Philosophy

Horned-OWL prioritizes:
- **Performance**: Leveraging Rust's zero-cost abstractions
- **Pluggability**: Trait-based architecture for extensibility
- **Memory Efficiency**: Generic IRI types with reference counting
- **Large-Scale Processing**: Bulk ontology manipulation optimized

---

## Performance Characteristics

### Benchmarks

| Operation | horned-owl | OWL API | Speedup |
|-----------|------------|---------|---------|
| Gene Ontology Validation | ~1-2s | ~40-80s | 20x-40x |
| Large Ontology Parsing | Fast | Slow | 1-2 orders of magnitude |
| Memory Usage | Low | High | Significant reduction |

### Scalability

- Tested with **10 million classes** on standard desktop hardware
- Efficient memory management through `Rc<String>` IRI sharing
- Optimized for bulk manipulation operations
- Linear search trade-offs with `SetOntology` for fast insertion

---

## Architecture & Design

### Module Structure

```
horned-owl/
├── model/          # Core OWL2 data structures
├── ontology/       # Ontology implementations and traits
├── io/             # Parsers and renderers
├── error/          # Error handling hierarchy
├── adaptor/        # Data structure conversions
├── normalize/      # Ontology normalization
├── visitor/        # Visitor pattern support
└── vocab/          # RDF vocabularies for OWL2
```

### Core Design Patterns

1. **Trait-Based Architecture** (v0.7.0+)
   - Pluggable `Ontology` implementations
   - Composable `OntologyIndex` designs
   - Generic over IRI types for multi-threading

2. **Visitor Pattern** (v0.12.0+)
   - Navigate ontology structures systematically
   - Visitor library for traversal operations

3. **Immutable by Default**
   - `MutableOntology` trait for modifications
   - Reference-counted data sharing
   - Thread-safe designs

4. **Error Handling** (v0.12.0+)
   - Unified error hierarchy
   - Better location reporting for parse errors
   - Type-safe error propagation

---

## Core Modules

### 1. Model Module (`horned_owl::model`)

Implements fundamental OWL2 data structures.

#### Core Entity Types

```rust
// Class - represents groups of individuals
pub struct Class(pub IRI);

// ObjectProperty - relationships between individuals
pub struct ObjectProperty(pub IRI);

// DataProperty - relationships to concrete data
pub struct DataProperty(pub IRI);

// Individuals - named or anonymous instances
pub enum Individual {
    Named(NamedIndividual),
    Anonymous(AnonymousIndividual),
}

pub struct NamedIndividual(pub IRI);
```

#### IRI Handling

```rust
// Reference-counted IRI for efficient sharing
pub struct IRI(Rc<String>);

// Generic IRI for multi-threading support (v0.12.0+)
// Enables different IRI implementations per use case
```

#### Build Struct (Factory Pattern)

```rust
use horned_owl::model::Build;

// Create IRI and NamedEntity instances
let build = Build::new_rc();

// Method chaining for construction
let object_prop = build.object_property("http://example.org/hasParent");
let class = build.class("http://example.org/Person");
```

#### Axiom Types

Complete axiom support including:

**Class Axioms:**
- `SubClassOf` - class hierarchy
- `EquivalentClasses` - class equivalence
- `DisjointClasses` - disjoint class declarations
- `DisjointUnion` - partitioning

**Property Axioms:**
- `SubObjectPropertyOf` - property hierarchy
- `EquivalentObjectProperties` - property equivalence
- `InverseObjectProperties` - inverse relationships
- `FunctionalObjectProperty` - functional constraints
- `TransitiveObjectProperty` - transitivity
- `SymmetricObjectProperty` - symmetry
- `AsymmetricObjectProperty` - asymmetry
- `ReflexiveObjectProperty` - reflexivity
- `IrreflexiveObjectProperty` - irreflexivity

**Individual Axioms:**
- `ClassAssertion` - type assertions
- `ObjectPropertyAssertion` - relationship assertions
- `DataPropertyAssertion` - data value assertions
- `SameIndividual` - identity
- `DifferentIndividuals` - distinctness

**Annotations:**
- `AnnotationAssertion` - metadata attachment
- Full annotation support on all axioms

### 2. Ontology Module (`horned_owl::ontology`)

Provides ontology storage and query capabilities.

#### Core Traits

```rust
// Mutable ontology operations
pub trait MutableOntology {
    fn insert(&mut self, axiom: AnnotatedComponent);
    fn remove(&mut self, axiom: &AnnotatedComponent);
    // ... more methods
}

// Composable searching strategy
pub trait OntologyIndex {
    fn indexed_axioms(&self) -> &[AnnotatedComponent];
    // ... query methods
}
```

#### SetOntology Implementation

```rust
use horned_owl::ontology::set::SetOntology;

// Simplest implementation using HashSet
// Optimized for fast insertion, linear search
let mut onto = SetOntology::new();

// Insert axioms
onto.insert(axiom);

// Query
let axioms = onto.axioms();
```

#### Specialized Indices

1. **component_mapped** - Retrieve by axiom kind
2. **declaration_mapped** - Fast declared IRI lookup
3. **iri_mapped** - Components by IRI reference
4. **logically_equal** - Logical equivalence queries
5. **indexed** - General composable searching (slower insert)

**Trade-offs:**
- `SetOntology`: Fast insertion, linear search
- Indexed variants: Slower insertion, fast queries

### 3. IO Module (`horned_owl::io`)

Parsers and renderers for multiple formats.

#### Supported Formats

| Format | Module | Read | Write | Status |
|--------|--------|------|-------|--------|
| Functional Syntax | `ofn` | ✓ | ✓ | Complete (v1.0.0) |
| OWL/XML | `owx` | ✓ | ✓ | Complete |
| RDF/XML | `rdf` | ✓ | ✓ | Complete (v0.10.0) |
| Turtle | `rdf` | ✓ | ✓ | Via RIO |
| Manchester | TBD | ✗ | ✗ | In Progress |

#### Parser Configuration

```rust
use horned_owl::io::{ParserConfiguration, OWXParserConfiguration, RDFParserConfiguration};

// General configuration
let config = ParserConfiguration::default();

// XML-specific options
let owx_config = OWXParserConfiguration {
    // ... options
};

// RDF-specific options
let rdf_config = RDFParserConfiguration {
    // ... options
};
```

#### Output Types

```rust
// Parsing results
pub enum ParserOutput {
    OWLOntology(SetOntology),
    // ... other variants
}

// Resource identification
pub enum ResourceType {
    Class,
    ObjectProperty,
    DataProperty,
    Individual,
    // ... more types
}
```

---

## Type System

### Naming Conventions

Horned-OWL follows explicit naming rules for predictability:

1. **Tuple Structs** - Semantically equivalent types
   ```rust
   struct Class(IRI);
   struct ObjectProperty(IRI);
   ```

2. **Named Structs** - Non-equivalent entities
   ```rust
   struct ClassAssertion {
       class: Class,
       individual: Individual,
   }
   ```

3. **Vec Collections** - Unbounded sets
   ```rust
   struct EquivalentClasses(Vec<Class>);
   ```

4. **Enums** - Type variants
   ```rust
   enum Individual {
       Named(NamedIndividual),
       Anonymous(AnonymousIndividual),
   }
   ```

### Generic IRI Design (v0.12.0+)

```rust
// IRI is generic for multi-threading support
pub struct IRI<A: ForIRI> {
    // Implementation details
}

// Allows different backing stores:
// - Rc<String> for single-threaded
// - Arc<String> for multi-threaded
// - Custom implementations
```

---

## Parsing API

### 1. Functional Syntax (horned-functional)

```rust
use horned_functional::{from_str, from_file, from_reader};
use horned_owl::ontology::set::SetOntology;

// Parse from string
let (ontology, prefixes) = from_str::<SetOntology>("
    Prefix(:=<http://example.org/>)
    Ontology(<http://example.org/onto>
        Declaration(Class(:Person))
        Declaration(Class(:Student))
        SubClassOf(:Student :Person)
    )
")?;

// Parse from file
let (ontology, prefixes) = from_file::<SetOntology>("ontology.ofn")?;

// Parse from reader
use std::fs::File;
let file = File::open("ontology.ofn")?;
let (ontology, prefixes) = from_reader::<SetOntology, _>(file)?;
```

### 2. Parse Individual Axioms

```rust
use horned_owl::model::Axiom;
use horned_functional::FromFunctional;

// Parse single axiom
let axiom = Axiom::from_ofn(
    "Declaration(Class(<http://purl.obolibrary.org/obo/MS_1000031>))"
)?;

// Parse with prefix context
use std::collections::HashMap;
let mut prefixes = HashMap::new();
prefixes.insert("obo".to_string(), "http://purl.obolibrary.org/obo/".to_string());

let axiom = Axiom::from_ofn_ctx(
    "Declaration(Class(obo:MS_1000031))",
    &prefixes
)?;
```

### 3. RDF/XML Parsing

```rust
use horned_owl::io::rdf::reader::read as read_rdf;
use std::fs::File;
use std::io::BufReader;

// Parse RDF/XML file
let file = File::open("ontology.owl")?;
let reader = BufReader::new(file);

let (ontology, incomplete) = read_rdf(reader, &Default::default())?;

if !incomplete.is_empty() {
    eprintln!("Warning: Incomplete parsing for {:?}", incomplete);
}
```

### 4. OWL/XML Parsing

```rust
use horned_owl::io::owx::reader::read as read_owx;
use std::fs::File;
use std::io::BufReader;

// Parse OWL/XML file
let file = File::open("ontology.owx")?;
let reader = BufReader::new(file);

let (ontology, incomplete) = read_owx(reader, &Default::default())?;
```

### 5. Remote Import Resolution (RDF/XML)

Enable the `remote` feature in Cargo.toml:

```toml
[dependencies]
horned-owl = { version = "1.2.0", features = ["remote"] }
```

```rust
use horned_owl::io::rdf::reader::read_with_imports;

// Automatically resolves and parses imported ontologies
let (ontology, import_closure) = read_with_imports(reader)?;
```

### 6. Non-UTF-8 Encoding Support

Enable the `encoding` feature:

```toml
[dependencies]
horned-owl = { version = "1.2.0", features = ["encoding"] }
```

Allows parsing documents with non-UTF-8 character encodings.

---

## Ontology Building

### 1. Using Build Struct (Factory Pattern)

```rust
use horned_owl::model::{Build, SubClassOf, Class, Axiom, AnnotatedAxiom};
use horned_owl::ontology::set::SetOntology;

// Create factory
let build = Build::new_rc();

// Build classes
let person = build.class("http://example.org/Person");
let student = build.class("http://example.org/Student");

// Build properties
let has_parent = build.object_property("http://example.org/hasParent");
let has_age = build.data_property("http://example.org/hasAge");

// Build individuals
let john = build.named_individual("http://example.org/John");

// Create axioms
let axiom = SubClassOf {
    sub: student.clone().into(),
    sup: person.clone().into(),
};

// Wrap in annotated axiom
let annotated = AnnotatedAxiom {
    axiom: Axiom::SubClassOf(axiom),
    ann: Default::default(),
};

// Add to ontology
let mut ontology = SetOntology::new();
ontology.insert(annotated.into());
```

### 2. Creating Complex Axioms

```rust
use horned_owl::model::*;

let build = Build::new_rc();

// Equivalent classes
let equivalent = EquivalentClasses(vec![
    build.class("http://example.org/Person"),
    build.class("http://example.org/Human"),
]);

// Disjoint classes
let disjoint = DisjointClasses(vec![
    build.class("http://example.org/Male"),
    build.class("http://example.org/Female"),
]);

// Object property domain/range
let domain = ObjectPropertyDomain {
    ope: build.object_property("http://example.org/hasParent").into(),
    ce: build.class("http://example.org/Person").into(),
};

let range = ObjectPropertyRange {
    ope: build.object_property("http://example.org/hasParent").into(),
    ce: build.class("http://example.org/Person").into(),
};
```

### 3. Class Assertions and Property Assertions

```rust
use horned_owl::model::*;

let build = Build::new_rc();

// Class assertion (type)
let class_assertion = ClassAssertion {
    class: build.class("http://example.org/Person"),
    i: Individual::Named(build.named_individual("http://example.org/John")),
};

// Object property assertion
let obj_prop_assertion = ObjectPropertyAssertion {
    ope: build.object_property("http://example.org/hasParent").into(),
    from: Individual::Named(build.named_individual("http://example.org/John")),
    to: Individual::Named(build.named_individual("http://example.org/Mary")),
};

// Data property assertion
let data_prop_assertion = DataPropertyAssertion {
    dp: build.data_property("http://example.org/hasAge"),
    from: Individual::Named(build.named_individual("http://example.org/John")),
    to: Literal::Integer(30),
};
```

### 4. Annotations

```rust
use horned_owl::model::*;

let build = Build::new_rc();

// Annotation property
let label = build.annotation_property("http://www.w3.org/2000/01/rdf-schema#label");

// Annotation assertion
let annotation = AnnotationAssertion {
    ap: label,
    subject: build.class("http://example.org/Person").into(),
    ann: Annotation {
        ap: label.clone(),
        av: AnnotationValue::Literal(Literal::Simple("Person".to_string())),
    },
};

// Add annotations to axioms
let annotated_axiom = AnnotatedAxiom {
    axiom: Axiom::SubClassOf(/* ... */),
    ann: vec![Annotation {
        ap: label,
        av: AnnotationValue::Literal(Literal::Simple("Core hierarchy".to_string())),
    }],
};
```

---

## Querying Ontologies

### 1. Basic Queries (SetOntology)

```rust
use horned_owl::ontology::set::SetOntology;

let ontology: SetOntology = /* ... */;

// Get all axioms
for axiom in ontology.iter() {
    println!("{:?}", axiom);
}

// Count axioms
let count = ontology.len();

// Check if empty
if ontology.is_empty() {
    println!("Ontology has no axioms");
}
```

### 2. Get Classes, Properties, Individuals

```rust
// This requires traversing axioms and extracting entities
use horned_owl::model::*;
use std::collections::HashSet;

fn extract_classes(ontology: &SetOntology) -> HashSet<Class> {
    let mut classes = HashSet::new();

    for component in ontology.iter() {
        match &component.axiom {
            Axiom::DeclareClass(DeclareClass(class)) => {
                classes.insert(class.clone());
            }
            Axiom::SubClassOf(SubClassOf { sub, sup }) => {
                // Extract classes from class expressions
                // This requires more complex traversal
            }
            // Handle other axiom types...
            _ => {}
        }
    }

    classes
}
```

### 3. Query by IRI (Using IRI-Mapped Index)

```rust
use horned_owl::ontology::indexed::ForIndex;
use horned_owl::ontology::iri_mapped::IRIMappedIndex;

// Build indexed ontology
let indexed = ontology.index::<IRIMappedIndex>();

// Query all axioms referencing an IRI
let iri = build.iri("http://example.org/Person");
let axioms = indexed.axioms_for_iri(&iri);

for axiom in axioms {
    println!("{:?}", axiom);
}
```

### 4. Declaration-Mapped Queries

```rust
use horned_owl::ontology::declaration_mapped::DeclarationMappedIndex;

let indexed = ontology.index::<DeclarationMappedIndex>();

// Fast lookup of declared entity types
let iri = build.iri("http://example.org/Person");

if indexed.is_class(&iri) {
    println!("{} is declared as a class", iri);
}

if indexed.is_object_property(&iri) {
    println!("{} is declared as an object property", iri);
}
```

### 5. Component-Mapped Queries

```rust
use horned_owl::ontology::component_mapped::ComponentMappedIndex;
use horned_owl::model::AxiomKind;

let indexed = ontology.index::<ComponentMappedIndex>();

// Get all SubClassOf axioms
let subclass_axioms = indexed.axioms_of_kind(AxiomKind::SubClassOf);

for axiom in subclass_axioms {
    println!("{:?}", axiom);
}
```

### 6. Visitor Pattern Traversal

```rust
use horned_owl::visitor::Walk;

// Implement custom visitor
struct MyVisitor {
    class_count: usize,
}

impl horned_owl::visitor::Visitor for MyVisitor {
    fn visit_class(&mut self, class: &Class) {
        self.class_count += 1;
        println!("Found class: {:?}", class);
    }

    // Implement other visit methods...
}

// Walk ontology
let mut visitor = MyVisitor { class_count: 0 };
ontology.walk(&mut visitor);

println!("Total classes visited: {}", visitor.class_count);
```

---

## Integration Patterns

### 1. Converting to Database Schema

Strategy for extracting ontology data into a relational database:

```rust
use horned_owl::ontology::set::SetOntology;
use horned_owl::model::*;

struct OntologyExtractor {
    classes: Vec<(String, Option<String>)>,        // (iri, label)
    properties: Vec<(String, String, Option<String>)>, // (iri, type, label)
    individuals: Vec<(String, Option<String>)>,     // (iri, label)
    axioms: Vec<(String, String, String)>,         // (subject, predicate, object)
}

impl OntologyExtractor {
    fn extract(ontology: &SetOntology) -> Self {
        let mut extractor = Self {
            classes: Vec::new(),
            properties: Vec::new(),
            individuals: Vec::new(),
            axioms: Vec::new(),
        };

        for component in ontology.iter() {
            match &component.axiom {
                Axiom::DeclareClass(DeclareClass(class)) => {
                    let label = Self::extract_label(&component.ann);
                    extractor.classes.push((class.0.to_string(), label));
                }

                Axiom::DeclareObjectProperty(DeclareObjectProperty(prop)) => {
                    let label = Self::extract_label(&component.ann);
                    extractor.properties.push((
                        prop.0.to_string(),
                        "ObjectProperty".to_string(),
                        label
                    ));
                }

                Axiom::DeclareDataProperty(DeclareDataProperty(prop)) => {
                    let label = Self::extract_label(&component.ann);
                    extractor.properties.push((
                        prop.0.to_string(),
                        "DataProperty".to_string(),
                        label
                    ));
                }

                Axiom::SubClassOf(SubClassOf { sub, sup }) => {
                    // Extract as triple
                    if let (ClassExpression::Class(sub_class), ClassExpression::Class(sup_class)) = (sub, sup) {
                        extractor.axioms.push((
                            sub_class.0.to_string(),
                            "subClassOf".to_string(),
                            sup_class.0.to_string(),
                        ));
                    }
                }

                Axiom::ClassAssertion(ClassAssertion { class, i }) => {
                    if let Individual::Named(individual) = i {
                        extractor.axioms.push((
                            individual.0.to_string(),
                            "type".to_string(),
                            class.0.to_string(),
                        ));
                    }
                }

                // Handle other axiom types...
                _ => {}
            }
        }

        extractor
    }

    fn extract_label(annotations: &[Annotation]) -> Option<String> {
        for ann in annotations {
            if ann.ap.0.to_string().ends_with("label") {
                if let AnnotationValue::Literal(Literal::Simple(s)) = &ann.av {
                    return Some(s.clone());
                }
            }
        }
        None
    }
}

// Usage
let extractor = OntologyExtractor::extract(&ontology);

// Insert into database
for (iri, label) in extractor.classes {
    // INSERT INTO classes (iri, label) VALUES (?, ?)
}

for (iri, prop_type, label) in extractor.properties {
    // INSERT INTO properties (iri, type, label) VALUES (?, ?, ?)
}

for (subject, predicate, object) in extractor.axioms {
    // INSERT INTO axioms (subject, predicate, object) VALUES (?, ?, ?)
}
```

### 2. PostgreSQL Schema Example

```sql
-- Classes table
CREATE TABLE classes (
    id SERIAL PRIMARY KEY,
    iri TEXT UNIQUE NOT NULL,
    label TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Properties table
CREATE TABLE properties (
    id SERIAL PRIMARY KEY,
    iri TEXT UNIQUE NOT NULL,
    property_type VARCHAR(50) NOT NULL, -- 'ObjectProperty', 'DataProperty', 'AnnotationProperty'
    label TEXT,
    description TEXT,
    domain_iri TEXT,
    range_iri TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Individuals table
CREATE TABLE individuals (
    id SERIAL PRIMARY KEY,
    iri TEXT UNIQUE NOT NULL,
    label TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Axioms table (subject-predicate-object triples)
CREATE TABLE axioms (
    id SERIAL PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    axiom_type VARCHAR(50) NOT NULL, -- 'SubClassOf', 'EquivalentClasses', etc.
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(subject, predicate, object)
);

-- Annotations table
CREATE TABLE annotations (
    id SERIAL PRIMARY KEY,
    entity_iri TEXT NOT NULL,
    property_iri TEXT NOT NULL,
    value TEXT NOT NULL,
    language VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_axioms_subject ON axioms(subject);
CREATE INDEX idx_axioms_predicate ON axioms(predicate);
CREATE INDEX idx_axioms_object ON axioms(object);
CREATE INDEX idx_annotations_entity ON annotations(entity_iri);
CREATE INDEX idx_classes_iri ON classes(iri);
CREATE INDEX idx_properties_iri ON properties(iri);
```

### 3. Batch Insertion for Performance

```rust
use postgres::{Client, NoTls, Error};

fn insert_ontology_batch(
    client: &mut Client,
    extractor: &OntologyExtractor,
) -> Result<(), Error> {
    // Begin transaction
    let mut transaction = client.transaction()?;

    // Prepare statements
    let class_stmt = transaction.prepare(
        "INSERT INTO classes (iri, label) VALUES ($1, $2) ON CONFLICT (iri) DO NOTHING"
    )?;

    let prop_stmt = transaction.prepare(
        "INSERT INTO properties (iri, property_type, label) VALUES ($1, $2, $3) ON CONFLICT (iri) DO NOTHING"
    )?;

    let axiom_stmt = transaction.prepare(
        "INSERT INTO axioms (subject, predicate, object, axiom_type) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING"
    )?;

    // Batch insert classes
    for (iri, label) in &extractor.classes {
        transaction.execute(&class_stmt, &[iri, label])?;
    }

    // Batch insert properties
    for (iri, prop_type, label) in &extractor.properties {
        transaction.execute(&prop_stmt, &[iri, prop_type, label])?;
    }

    // Batch insert axioms
    for (subject, predicate, object) in &extractor.axioms {
        transaction.execute(&axiom_stmt, &[subject, predicate, object, &"SubClassOf".to_string()])?;
    }

    // Commit transaction
    transaction.commit()?;

    Ok(())
}
```

---

## Whelk-rs Reasoning Integration

### Overview

**whelk-rs** is an experimental Rust implementation of the Whelk OWL EL reasoner, designed to work with horned-owl ontologies.

- **Repository**: https://github.com/INCATools/whelk-rs
- **License**: MIT
- **Language**: 100% Rust
- **Status**: Experimental

### Building whelk-rs

```bash
git clone https://github.com/INCATools/whelk-rs.git
cd whelk-rs
cargo build --release
```

### Command-Line Usage

```bash
# Run reasoner on OWL file
./target/release/whelk -i ontology.owl

# Run tests
cargo test --release -- --nocapture
```

### Integration Strategy

Since whelk-rs is experimental, here's a conceptual integration pattern:

```rust
// Conceptual integration (actual API may differ)
use horned_owl::ontology::set::SetOntology;
use horned_functional::from_file;

// 1. Parse ontology with horned-owl
let (ontology, _) = from_file::<SetOntology>("ontology.ofn")?;

// 2. Convert to whelk-rs reasoner input
// (This step depends on whelk-rs API)
let reasoner_input = convert_to_whelk_format(&ontology);

// 3. Run reasoning
// let reasoner = WhelkReasoner::new(reasoner_input);
// let inferred = reasoner.classify();

// 4. Extract inferred axioms
// let inferred_axioms = extract_inferred_axioms(&inferred);

// 5. Merge back into ontology or database
// for axiom in inferred_axioms {
//     ontology.insert(axiom);
// }
```

### Alternative: Using reasonable Rust Reasoner

**reasonable** is another Rust OWL reasoner option:

- **Repository**: https://github.com/gtfierro/reasonable
- **Performance**: 7x faster than Allegro, 38x faster than OWLRL
- **Implements**: OWL 2 RL profile on RDF graphs

```toml
[dependencies]
reasonable = "0.3"
```

```rust
// Example usage (conceptual)
use reasonable::Reasoner;

// Create reasoner from RDF graph
let reasoner = Reasoner::from_rdf_graph(rdf_graph)?;

// Materialize inferences
let materialized = reasoner.materialize()?;

// Query results
let results = reasoner.query("SELECT ?s WHERE { ?s a :Person }")?;
```

---

## Code Examples

### Complete Ontology Building Example

```rust
use horned_owl::model::*;
use horned_owl::ontology::set::SetOntology;
use horned_functional::to_functional;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create factory
    let build = Build::new_rc();

    // Create ontology
    let mut ontology = SetOntology::new();

    // Define namespace
    let ns = "http://example.org/university#";

    // Create classes
    let person = build.class(format!("{}Person", ns));
    let student = build.class(format!("{}Student", ns));
    let professor = build.class(format!("{}Professor", ns));
    let course = build.class(format!("{}Course", ns));

    // Declare classes
    ontology.insert(DeclareClass(person.clone()).into());
    ontology.insert(DeclareClass(student.clone()).into());
    ontology.insert(DeclareClass(professor.clone()).into());
    ontology.insert(DeclareClass(course.clone()).into());

    // Class hierarchy
    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::SubClassOf(SubClassOf {
            sub: student.clone().into(),
            sup: person.clone().into(),
        }),
        ann: vec![],
    }.into());

    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::SubClassOf(SubClassOf {
            sub: professor.clone().into(),
            sup: person.clone().into(),
        }),
        ann: vec![],
    }.into());

    // Disjoint classes
    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::DisjointClasses(DisjointClasses(vec![
            student.clone(),
            professor.clone(),
        ])),
        ann: vec![],
    }.into());

    // Object properties
    let teaches = build.object_property(format!("{}teaches", ns));
    let enrolledIn = build.object_property(format!("{}enrolledIn", ns));

    ontology.insert(DeclareObjectProperty(teaches.clone()).into());
    ontology.insert(DeclareObjectProperty(enrolledIn.clone()).into());

    // Property domain/range
    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::ObjectPropertyDomain(ObjectPropertyDomain {
            ope: teaches.clone().into(),
            ce: professor.clone().into(),
        }),
        ann: vec![],
    }.into());

    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::ObjectPropertyRange(ObjectPropertyRange {
            ope: teaches.clone().into(),
            ce: course.clone().into(),
        }),
        ann: vec![],
    }.into());

    // Individuals
    let john = build.named_individual(format!("{}John", ns));
    let cs101 = build.named_individual(format!("{}CS101", ns));

    ontology.insert(DeclareNamedIndividual(john.clone()).into());
    ontology.insert(DeclareNamedIndividual(cs101.clone()).into());

    // Type assertions
    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::ClassAssertion(ClassAssertion {
            class: professor.clone(),
            i: Individual::Named(john.clone()),
        }),
        ann: vec![],
    }.into());

    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::ClassAssertion(ClassAssertion {
            class: course.clone(),
            i: Individual::Named(cs101.clone()),
        }),
        ann: vec![],
    }.into());

    // Property assertion
    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::ObjectPropertyAssertion(ObjectPropertyAssertion {
            ope: teaches.clone().into(),
            from: Individual::Named(john.clone()),
            to: Individual::Named(cs101.clone()),
        }),
        ann: vec![],
    }.into());

    // Add annotations
    let label = build.annotation_property("http://www.w3.org/2000/01/rdf-schema#label");

    ontology.insert(AnnotatedAxiom {
        axiom: Axiom::AnnotationAssertion(AnnotationAssertion {
            subject: person.0.clone().into(),
            ann: Annotation {
                ap: label.clone(),
                av: AnnotationValue::Literal(Literal::Simple("Person".to_string())),
            },
        }),
        ann: vec![],
    }.into());

    // Render to functional syntax
    let functional_syntax = to_functional(&ontology)?;
    println!("{}", functional_syntax);

    println!("Ontology created with {} axioms", ontology.len());

    Ok(())
}
```

### File Format Conversion Example

```rust
use horned_owl::ontology::set::SetOntology;
use horned_functional::{from_file, to_functional};
use horned_owl::io::rdf::writer::write as write_rdf;
use std::fs::File;
use std::io::BufWriter;

fn convert_ofn_to_rdf(
    input: &str,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read functional syntax
    let (ontology, prefixes) = from_file::<SetOntology>(input)?;

    println!("Loaded {} axioms", ontology.len());

    // Write as RDF/XML
    let output_file = File::create(output)?;
    let writer = BufWriter::new(output_file);

    write_rdf(writer, &ontology, Some(&prefixes))?;

    println!("Converted to RDF/XML: {}", output);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    convert_ofn_to_rdf("input.ofn", "output.owl")?;
    Ok(())
}
```

### Ontology Statistics Example

```rust
use horned_owl::ontology::set::SetOntology;
use horned_owl::model::*;
use std::collections::HashSet;

struct OntologyStats {
    class_count: usize,
    object_property_count: usize,
    data_property_count: usize,
    individual_count: usize,
    axiom_count: usize,
    subclass_axioms: usize,
    equivalent_class_axioms: usize,
    disjoint_class_axioms: usize,
}

fn compute_stats(ontology: &SetOntology) -> OntologyStats {
    let mut classes = HashSet::new();
    let mut object_props = HashSet::new();
    let mut data_props = HashSet::new();
    let mut individuals = HashSet::new();

    let mut subclass = 0;
    let mut equivalent = 0;
    let mut disjoint = 0;

    for component in ontology.iter() {
        match &component.axiom {
            Axiom::DeclareClass(DeclareClass(c)) => {
                classes.insert(c.clone());
            }
            Axiom::DeclareObjectProperty(DeclareObjectProperty(p)) => {
                object_props.insert(p.clone());
            }
            Axiom::DeclareDataProperty(DeclareDataProperty(p)) => {
                data_props.insert(p.clone());
            }
            Axiom::DeclareNamedIndividual(DeclareNamedIndividual(i)) => {
                individuals.insert(i.clone());
            }
            Axiom::SubClassOf(_) => {
                subclass += 1;
            }
            Axiom::EquivalentClasses(_) => {
                equivalent += 1;
            }
            Axiom::DisjointClasses(_) => {
                disjoint += 1;
            }
            _ => {}
        }
    }

    OntologyStats {
        class_count: classes.len(),
        object_property_count: object_props.len(),
        data_property_count: data_props.len(),
        individual_count: individuals.len(),
        axiom_count: ontology.len(),
        subclass_axioms: subclass,
        equivalent_class_axioms: equivalent,
        disjoint_class_axioms: disjoint,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use horned_functional::from_file;

    let (ontology, _) = from_file::<SetOntology>("ontology.ofn")?;
    let stats = compute_stats(&ontology);

    println!("Ontology Statistics:");
    println!("  Classes: {}", stats.class_count);
    println!("  Object Properties: {}", stats.object_property_count);
    println!("  Data Properties: {}", stats.data_property_count);
    println!("  Individuals: {}", stats.individual_count);
    println!("  Total Axioms: {}", stats.axiom_count);
    println!("  SubClassOf: {}", stats.subclass_axioms);
    println!("  EquivalentClasses: {}", stats.equivalent_class_axioms);
    println!("  DisjointClasses: {}", stats.disjoint_class_axioms);

    Ok(())
}
```

---

## Ecosystem Components

### 1. horned-functional

- **Version**: 0.4.0
- **Repository**: https://github.com/fastobo/horned-functional
- **Purpose**: OWL2 Functional Syntax parser and serializer
- **Features**:
  - `FromFunctional` trait for deserialization
  - `AsFunctional` trait for serialization
  - Parser functions: `from_str`, `from_file`, `from_reader`
  - Prefix mapping support

```toml
[dependencies]
horned-functional = "0.4.0"
```

### 2. py-horned-owl

- **Version**: Latest on PyPI
- **Repository**: https://github.com/ontology-tools/py-horned-owl
- **Purpose**: Python bindings via PyO3
- **Features**:
  - `open_ontology()` for file loading
  - `get_classes()`, `get_axioms()` for queries
  - Pythonic API for ontology manipulation

```bash
pip install py-horned-owl
```

### 3. horned-bin

Command-line tools for ontology manipulation:

```bash
# Install
cargo install horned-bin

# Validate ontology
horned-validate ontology.owl

# Convert formats
horned-convert input.ofn output.owl
```

### 4. whelk-rs

- **Repository**: https://github.com/INCATools/whelk-rs
- **Purpose**: OWL EL reasoner in Rust
- **Status**: Experimental
- **Integration**: Works with horned-owl ontologies

### 5. reasonable

- **Repository**: https://github.com/gtfierro/reasonable
- **Purpose**: OWL 2 RL reasoner on RDF graphs
- **Performance**: High-performance Datalog implementation
- **Use Case**: Efficient materialization of inferences

---

## Database Schema Integration

### Complete Integration Example

```rust
use horned_owl::ontology::set::SetOntology;
use horned_owl::model::*;
use horned_functional::from_file;
use postgres::{Client, NoTls, Error};

struct DatabaseIntegrator {
    client: Client,
}

impl DatabaseIntegrator {
    fn new(connection_string: &str) -> Result<Self, Error> {
        let client = Client::connect(connection_string, NoTls)?;
        Ok(Self { client })
    }

    fn create_schema(&mut self) -> Result<(), Error> {
        self.client.batch_execute("
            CREATE TABLE IF NOT EXISTS classes (
                id SERIAL PRIMARY KEY,
                iri TEXT UNIQUE NOT NULL,
                label TEXT,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS properties (
                id SERIAL PRIMARY KEY,
                iri TEXT UNIQUE NOT NULL,
                property_type VARCHAR(50) NOT NULL,
                label TEXT,
                domain_iri TEXT,
                range_iri TEXT
            );

            CREATE TABLE IF NOT EXISTS individuals (
                id SERIAL PRIMARY KEY,
                iri TEXT UNIQUE NOT NULL,
                label TEXT
            );

            CREATE TABLE IF NOT EXISTS axioms (
                id SERIAL PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                axiom_type VARCHAR(50) NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_axioms_subject ON axioms(subject);
            CREATE INDEX IF NOT EXISTS idx_axioms_predicate ON axioms(predicate);
            CREATE INDEX IF NOT EXISTS idx_axioms_object ON axioms(object);
        ")?;

        Ok(())
    }

    fn import_ontology(&mut self, ontology: &SetOntology) -> Result<(), Error> {
        let transaction = self.client.transaction()?;

        for component in ontology.iter() {
            match &component.axiom {
                Axiom::DeclareClass(DeclareClass(class)) => {
                    let label = Self::extract_label(&component.ann);
                    transaction.execute(
                        "INSERT INTO classes (iri, label) VALUES ($1, $2) ON CONFLICT (iri) DO NOTHING",
                        &[&class.0.to_string(), &label],
                    )?;
                }

                Axiom::DeclareObjectProperty(DeclareObjectProperty(prop)) => {
                    let label = Self::extract_label(&component.ann);
                    transaction.execute(
                        "INSERT INTO properties (iri, property_type, label) VALUES ($1, $2, $3) ON CONFLICT (iri) DO NOTHING",
                        &[&prop.0.to_string(), &"ObjectProperty", &label],
                    )?;
                }

                Axiom::SubClassOf(SubClassOf { sub, sup }) => {
                    if let (ClassExpression::Class(sub_class), ClassExpression::Class(sup_class)) = (sub, sup) {
                        transaction.execute(
                            "INSERT INTO axioms (subject, predicate, object, axiom_type) VALUES ($1, $2, $3, $4)",
                            &[&sub_class.0.to_string(), &"subClassOf", &sup_class.0.to_string(), &"SubClassOf"],
                        )?;
                    }
                }

                Axiom::ClassAssertion(ClassAssertion { class, i }) => {
                    if let Individual::Named(individual) = i {
                        transaction.execute(
                            "INSERT INTO axioms (subject, predicate, object, axiom_type) VALUES ($1, $2, $3, $4)",
                            &[&individual.0.to_string(), &"type", &class.0.to_string(), &"ClassAssertion"],
                        )?;
                    }
                }

                _ => {}
            }
        }

        transaction.commit()?;
        Ok(())
    }

    fn extract_label(annotations: &[Annotation]) -> Option<String> {
        for ann in annotations {
            if ann.ap.0.to_string().ends_with("label") {
                if let AnnotationValue::Literal(Literal::Simple(s)) = &ann.av {
                    return Some(s.clone());
                }
            }
        }
        None
    }

    fn query_subclasses(&mut self, class_iri: &str) -> Result<Vec<String>, Error> {
        let rows = self.client.query(
            "SELECT subject FROM axioms WHERE predicate = 'subClassOf' AND object = $1",
            &[&class_iri],
        )?;

        Ok(rows.iter().map(|row| row.get(0)).collect())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse ontology
    let (ontology, _) = from_file::<SetOntology>("university.ofn")?;

    // Connect to database
    let mut integrator = DatabaseIntegrator::new("postgresql://user:pass@localhost/ontodb")?;

    // Create schema
    integrator.create_schema()?;

    // Import ontology
    integrator.import_ontology(&ontology)?;

    println!("Ontology imported successfully!");

    // Query example
    let subclasses = integrator.query_subclasses("http://example.org/Person")?;
    println!("Subclasses of Person: {:?}", subclasses);

    Ok(())
}
```

---

## Best Practices

### 1. Choose the Right Ontology Implementation

- **SetOntology**: Fast insertion, linear search. Good for building ontologies.
- **Indexed Ontology**: Slower insertion, fast queries. Good for read-heavy workloads.
- **Custom Index**: Implement `OntologyIndex` for specialized query patterns.

### 2. Use Build Factory Pattern

```rust
// Good: Reuse Build instance
let build = Build::new_rc();
let class1 = build.class("http://example.org/Class1");
let class2 = build.class("http://example.org/Class2");

// Avoid: Creating new Build for each entity
let class1 = Build::new_rc().class("http://example.org/Class1");
let class2 = Build::new_rc().class("http://example.org/Class2");
```

### 3. Handle Parsing Errors Gracefully

```rust
use horned_functional::from_file;
use horned_owl::ontology::set::SetOntology;

match from_file::<SetOntology>("ontology.ofn") {
    Ok((ontology, prefixes)) => {
        println!("Parsed {} axioms", ontology.len());
    }
    Err(e) => {
        eprintln!("Failed to parse ontology: {}", e);
        // Handle error appropriately
    }
}
```

### 4. Batch Database Operations

Use transactions for large imports to improve performance and ensure atomicity.

### 5. Extract and Cache Common Queries

For frequently accessed data, consider extracting and caching in appropriate data structures or database views.

### 6. Use Feature Flags Appropriately

```toml
[dependencies]
horned-owl = { version = "1.2.0", features = ["remote", "encoding"] }
```

Only enable features you need to minimize dependencies.

---

## Troubleshooting

### Common Issues

1. **Parse Errors**
   - Check file encoding (use `encoding` feature for non-UTF-8)
   - Validate syntax with external tools
   - Check for import resolution issues (use `remote` feature)

2. **Performance Issues**
   - Use indexed ontologies for query-heavy workloads
   - Batch database operations in transactions
   - Consider streaming large files

3. **Memory Usage**
   - Monitor IRI sharing efficiency
   - Use appropriate index types
   - Consider incremental processing for very large ontologies

4. **Integration Issues**
   - Ensure whelk-rs compatibility
   - Check database connection strings
   - Validate schema creation

---

## Future Directions

### Planned Features

- **Manchester Syntax**: Parser/renderer in development
- **Incremental Reasoning**: Integration with reasoners
- **Performance Optimizations**: Continued benchmarking and tuning
- **Extended SWRL Support**: More rule types

### Community Resources

- **GitHub Issues**: https://github.com/phillord/horned-owl/issues
- **Documentation**: https://docs.rs/horned-owl/
- **Crates.io**: https://crates.io/crates/horned-owl

---

## Summary

Horned-OWL provides a complete, high-performance Rust implementation of OWL 2 with excellent performance characteristics and a clean API. Key takeaways:

1. **Use horned-functional** for functional syntax parsing
2. **Build instances** with the `Build` factory pattern
3. **Choose appropriate ontology types** based on workload
4. **Integrate with databases** using extraction patterns
5. **Consider whelk-rs** for reasoning (experimental)
6. **Leverage visitor patterns** for complex traversals
7. **Batch operations** for optimal performance

This guide provides everything needed to integrate horned-owl into an OWL ontology processing pipeline, including database integration and reasoning support.

---

**Research Completed**: 2025-10-22
**Next Steps**: Implement parser integration with existing codebase
