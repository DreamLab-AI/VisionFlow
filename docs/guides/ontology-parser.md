# Guide: Ontology Parser

**Version:** 1.0
**Date:** 2025-10-27

---

## 1. Overview

The `OntologyParser` module is a crucial component for semantic understanding within the VisionFlow system. It is designed to parse markdown files that contain ontology definitions written in a Logseq-style format. The parser extracts OWL (Web Ontology Language) structures, including classes, properties, and axioms, which are then used to build the knowledge graph's semantic layer.

This guide provides developers with the necessary information to use the parser, understand its syntax, and integrate it into their workflows.

## 2. Core Features

-   **Markdown Ontology Block Detection:** Identifies `### OntologyBlock` markers in markdown files to isolate ontology definitions.
-   **OWL Class Extraction:** Parses class definitions including IRI, label, description, and parent classes (`subClassOf`).
-   **OWL Property Extraction:** Supports `objectProperty` (relations between entities) and `dataProperty` (relations to literal values), including their domain and range.
-   **Axiom Extraction:** Captures logical statements like `subClassOf` to define class hierarchies.
-   **Source File Tracking:** Automatically records the source file for each parsed class, aiding in traceability.

## 3. Syntax Reference

The parser uses a simple, indented syntax within a designated `OntologyBlock`.

### 3.1. Ontology Block Marker

All ontology definitions must be placed within a block marked as follows:

```markdown
- ### OntologyBlock
  ... ontology definitions go here ...
```

### 3.2. OWL Class Definition

```markdown
- owl_class:: ClassName
  - label:: Human Readable Name
  - description:: A brief description of the class.
  - subClassOf:: ParentClassName
```

**Multiple Parents:**

```markdown
- owl_class:: ChildClass
  - subClassOf:: Parent1
  - subClassOf:: Parent2
```

### 3.3. Property Definitions

**Object Property:**

```markdown
- objectProperty:: propertyName
  - label:: A human-readable label for the property
  - domain:: SourceClass
  - range:: TargetClass
```

**Data Property:**

```markdown
- dataProperty:: propertyName
  - label:: A human-readable label
  - domain:: SourceClass
  - range:: xsd:datatype  # e.g., xsd:string, xsd:integer
```

**Multiple Domains/Ranges:**

```markdown
- objectProperty:: hasRelative
  - domain:: Person, Animal
  - range:: Person, Animal
```

### 3.4. IRI Formats

The parser supports multiple IRI formats for flexibility:

```markdown
- owl_class:: SimpleName
- owl_class:: prefix:Name
- owl_class:: http://example.org/ontology#Name
```

## 4. Usage Example

The following Rust code demonstrates how to use the `OntologyParser`.

```rust
use webxr::services::parsers::OntologyParser;
use webxr::ports::ontology_repository::OntologyRepository;

async fn parse_and_store_ontology(
    repo: &impl OntologyRepository,
    markdown_content: &str,
    source_filename: &str
) -> Result<(), String> {
    // 1. Create a new parser instance
    let parser = OntologyParser::new();

    // 2. Parse the markdown content
    let ontology_data = parser.parse(markdown_content, source_filename)?;

    // 3. (Optional) Print the parsed data
    println!("Parsed {} classes, {} properties, and {} axioms.",
        ontology_data.classes.len(),
        ontology_data.properties.len(),
        ontology_data.axioms.len()
    );

    // 4. Store the parsed data in a repository
    repo.save_ontology(
        &ontology_data.classes,
        &ontology_data.properties,
        &ontology_data.axioms
    ).await.map_err(|e| e.to_string())?;

    Ok(())
}
```

## 5. Data Structures

The parser returns an `OntologyData` struct, which contains vectors of the core OWL structures.

```rust
pub struct OntologyData {
    pub classes: Vec<OwlClass>,
    pub properties: Vec<OwlProperty>,
    pub axioms: Vec<OwlAxiom>,
    pub class_hierarchy: Vec<(String, String)>, // (child_iri, parent_iri)
}

pub struct OwlClass {
    pub iri: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub parent_classes: Vec<String>,
    pub source_file: Option<String>,
    // ... other fields
}

pub struct OwlProperty {
    pub iri: String,
    pub label: Option<String>,
    pub property_type: PropertyType, // ObjectProperty, DataProperty
    pub domain: Vec<String>,
    pub range: Vec<String>,
}

pub struct OwlAxiom {
    pub axiom_type: AxiomType, // e.g., SubClassOf
    pub subject: String,
    pub object: String,
    // ... other fields
}
```

## 6. Error Handling

The `parse` method returns a `Result<OntologyData, String>`. If parsing fails, it will return an `Err` with a descriptive error message, such as "No OntologyBlock found in file".

## 7. Testing

The `OntologyParser` module includes a comprehensive suite of unit tests to ensure its correctness. You can run these tests with:

```bash
cargo test --lib parsers::ontology_parser::tests
```

The test suite covers:
-   Basic class and property parsing
-   Class hierarchy and axiom extraction
-   Handling of multiple IRI formats
-   Error handling for missing `OntologyBlock`

This guide provides a solid foundation for working with the `OntologyParser`. For more advanced use cases or to extend its functionality, refer to the source code at `src/services/parsers/ontology_parser.rs`.