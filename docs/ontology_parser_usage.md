# OntologyParser Module Documentation

## Overview

The `OntologyParser` module parses markdown files containing Logseq-style ontology definitions and extracts OWL (Web Ontology Language) structures including classes, properties, and axioms.

## Location

`/home/devuser/workspace/project/src/services/parsers/ontology_parser.rs`

## Features

### 1. Markdown Ontology Block Detection
- Identifies `### OntologyBlock` marker in markdown files
- Extracts ontology definitions from the marked section

### 2. OWL Class Extraction
Parses OWL class definitions with:
- **IRI (Internationalized Resource Identifier)**: Full URI or short name
- **Label**: Human-readable name
- **Description**: Class description text
- **Parent Classes**: Hierarchical relationships via `subClassOf`
- **Source File**: Tracks which file defined the class

### 3. OWL Property Extraction
Supports two property types:
- **Object Properties**: Relations between entities
- **Data Properties**: Relations to literal values

Each property includes:
- IRI
- Label
- Property type
- Domain (classes this property applies to)
- Range (valid target classes/datatypes)

### 4. Axiom Extraction
Captures logical statements:
- **SubClassOf**: Class hierarchy relationships
- **EquivalentClass**: Class equivalence (planned)
- **DisjointWith**: Disjointness constraints (planned)

### 5. Class Hierarchy Tracking
Maintains parent-child relationships separately for efficient querying.

## Syntax

### Basic OWL Class

```markdown
- ### OntologyBlock
  - owl_class:: Person
    - label:: Human Person
    - description:: A human being in the system
```

### Class with Parent (SubClassOf)

```markdown
- ### OntologyBlock
  - owl_class:: Student
    - label:: Student
    - description:: A person enrolled in courses
    - subClassOf:: Person
```

### Multiple Parent Classes

```markdown
- ### OntologyBlock
  - owl_class:: TeachingAssistant
    - label:: Teaching Assistant
    - subClassOf:: Student
    - subClassOf:: Teacher
```

### Object Property

```markdown
- ### OntologyBlock
  - objectProperty:: enrolledIn
    - label:: enrolled in
    - domain:: Student
    - range:: Course
```

### Data Property

```markdown
- ### OntologyBlock
  - dataProperty:: hasAge
    - label:: has age
    - domain:: Person
    - range:: xsd:integer
```

### Multiple Domains/Ranges

```markdown
- ### OntologyBlock
  - objectProperty:: hasRelative
    - label:: has relative
    - domain:: Person, Animal
    - range:: Person, Animal
```

### Full URI Format

```markdown
- ### OntologyBlock
  - owl_class:: http://example.org/ontology#Person
    - label:: Person
  - owl_class:: ex:Student
    - subClassOf:: http://example.org/ontology#Person
```

## Usage Example

```rust
use webxr::services::parsers::OntologyParser;

// Create parser instance
let parser = OntologyParser::new();

// Load markdown content
let content = std::fs::read_to_string("ontology.md").unwrap();

// Parse the ontology
let result = parser.parse(&content, "ontology.md").unwrap();

// Access extracted data
println!("Classes: {}", result.classes.len());
println!("Properties: {}", result.properties.len());
println!("Axioms: {}", result.axioms.len());

// Iterate through classes
for class in &result.classes {
    println!("Class: {} ({})", class.label.as_deref().unwrap_or("unnamed"), class.iri);
    for parent in &class.parent_classes {
        println!("  - subClassOf: {}", parent);
    }
}

// Iterate through properties
for prop in &result.properties {
    println!("Property: {} ({})",
        prop.label.as_deref().unwrap_or("unnamed"),
        prop.iri
    );
    println!("  Type: {:?}", prop.property_type);
    println!("  Domain: {:?}", prop.domain);
    println!("  Range: {:?}", prop.range);
}

// Access class hierarchy
for (child, parent) in &result.class_hierarchy {
    println!("{} -> {}", child, parent);
}
```

## Data Structures

### OntologyData

The main result structure returned by the parser:

```rust
pub struct OntologyData {
    pub classes: Vec<OwlClass>,
    pub properties: Vec<OwlProperty>,
    pub axioms: Vec<OwlAxiom>,
    pub class_hierarchy: Vec<(String, String)>, // (child_iri, parent_iri)
}
```

### OwlClass

```rust
pub struct OwlClass {
    pub iri: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub parent_classes: Vec<String>,
    pub properties: HashMap<String, String>,
    pub source_file: Option<String>,
}
```

### OwlProperty

```rust
pub struct OwlProperty {
    pub iri: String,
    pub label: Option<String>,
    pub property_type: PropertyType, // ObjectProperty, DataProperty, AnnotationProperty
    pub domain: Vec<String>,
    pub range: Vec<String>,
}
```

### OwlAxiom

```rust
pub struct OwlAxiom {
    pub id: Option<u64>,
    pub axiom_type: AxiomType, // SubClassOf, EquivalentClass, DisjointWith, etc.
    pub subject: String,
    pub object: String,
    pub annotations: HashMap<String, String>,
}
```

## Supported Patterns

### Markers
- `- ### OntologyBlock` - Section marker
- `owl_class::` - Class definition
- `objectProperty::` - Object property definition
- `dataProperty::` - Data property definition

### Metadata
- `label::` - Human-readable label
- `description::` - Textual description
- `subClassOf::` - Parent class relationship
- `domain::` - Property domain (source)
- `range::` - Property range (target)

## Error Handling

The parser returns `Result<OntologyData, String>`:

```rust
match parser.parse(&content, filename) {
    Ok(data) => {
        // Process ontology data
    },
    Err(e) => {
        eprintln!("Parse error: {}", e);
        // Common errors:
        // - "No OntologyBlock found in file"
        // - Invalid regex patterns
    }
}
```

## Integration with Repository

The parsed data can be stored using the `OntologyRepository` trait:

```rust
use webxr::ports::ontology_repository::OntologyRepository;

async fn import_ontology(
    parser: &OntologyParser,
    repo: &dyn OntologyRepository,
    content: &str,
    filename: &str
) -> Result<(), String> {
    // Parse
    let data = parser.parse(content, filename)?;

    // Store in repository (single transaction)
    repo.save_ontology(
        &data.classes,
        &data.properties,
        &data.axioms
    ).await.map_err(|e| e.to_string())?;

    Ok(())
}
```

## Complete Example: University Ontology

```markdown
# University Domain Ontology

This ontology models the university domain.

- ### OntologyBlock

  ## Classes

  - owl_class:: Person
    - label:: Person
    - description:: A human being

  - owl_class:: Student
    - label:: Student
    - description:: A person enrolled in courses
    - subClassOf:: Person

  - owl_class:: Teacher
    - label:: Teacher
    - description:: A person who teaches courses
    - subClassOf:: Person

  - owl_class:: TeachingAssistant
    - label:: Teaching Assistant
    - description:: A student who assists with teaching
    - subClassOf:: Student
    - subClassOf:: Teacher

  - owl_class:: Course
    - label:: Course
    - description:: An educational course

  - owl_class:: Department
    - label:: Department
    - description:: Academic department

  ## Object Properties

  - objectProperty:: enrolledIn
    - label:: enrolled in
    - domain:: Student
    - range:: Course

  - objectProperty:: teaches
    - label:: teaches
    - domain:: Teacher
    - range:: Course

  - objectProperty:: memberOf
    - label:: member of
    - domain:: Person
    - range:: Department

  ## Data Properties

  - dataProperty:: hasStudentID
    - label:: has student ID
    - domain:: Student
    - range:: xsd:string

  - dataProperty:: hasAge
    - label:: has age
    - domain:: Person
    - range:: xsd:integer

  - dataProperty:: courseName
    - label:: course name
    - domain:: Course
    - range:: xsd:string
```

This would parse into:
- **4 classes**: Person, Student, Teacher, TeachingAssistant, Course, Department (6 total)
- **3 object properties**: enrolledIn, teaches, memberOf
- **3 data properties**: hasStudentID, hasAge, courseName
- **3 axioms**: Student subClassOf Person, Teacher subClassOf Person, TeachingAssistant subClassOf Student/Teacher
- **3 hierarchy relationships**

## Testing

The module includes comprehensive unit tests:

```bash
# Run ontology parser tests
cargo test --lib parsers::ontology_parser::tests
```

Test coverage includes:
- ✅ Basic class parsing
- ✅ Class hierarchy extraction
- ✅ Object property parsing
- ✅ Data property parsing
- ✅ Axiom extraction
- ✅ Error handling (missing OntologyBlock)
- ✅ IRI format support (URIs and prefixed names)
- ✅ Multiple parent classes
- ✅ Multiple domain/range values

## Performance Considerations

- Uses `regex` crate for pattern matching (compiled at runtime)
- Linear scan through content (O(n) where n = content length)
- Suitable for markdown files up to several MB
- For large ontology files, consider streaming parsing

## Future Enhancements

1. **Additional Axiom Types**:
   - EquivalentClass
   - DisjointWith
   - ObjectPropertyAssertion
   - DataPropertyAssertion

2. **Annotation Properties**:
   Support for `annotationProperty::` definitions

3. **RDFS Syntax Support**:
   Currently uses `subClassOf::`, could also support `rdfs:subClassOf::`

4. **Validation**:
   Check for circular hierarchies, undefined references

5. **Performance**:
   Pre-compile regex patterns for better performance

## Related Modules

- `/home/devuser/workspace/project/src/ports/ontology_repository.rs` - Repository trait
- `/home/devuser/workspace/project/src/services/parsers/knowledge_graph_parser.rs` - Graph parser
- `/home/devuser/workspace/project/src/adapters/ontology_repository_adapter.rs` - SQLite implementation

## References

- [OWL Web Ontology Language](https://www.w3.org/OWL/)
- [Logseq Ontology Format](https://logseq.github.io/)
- [RDF Schema (RDFS)](https://www.w3.org/TR/rdf-schema/)
