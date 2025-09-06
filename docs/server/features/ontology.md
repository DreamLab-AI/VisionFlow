# Ontology Validation

## Introduction

The ontology system provides a formal validation and logical inference layer for the knowledge graph, acting as a "truth engine." While the base graph is a flexible property graph, the ontology system maps it to a strict OWL/RDF structure to perform powerful consistency checks and infer new knowledge.

This ensures that the knowledge graph remains logically sound and allows for the discovery of implicit relationships that are not explicitly defined.

## Hybrid Model: Property Graph + OWL/RDF

The system uses a hybrid model that combines the flexibility of a property graph with the formal rigidity of an OWL/RDF graph.

```mermaid
graph TD
    subgraph "VisionFlow Property Graph"
        A[Node A<br/>type: Person] -->|knows| B[Node B<br/>type: Person]
        C[Node C<br/>type: Company] -->|employs| A
    end

    subgraph "Ontology Mapping & Validation"
        direction LR
        M{{Mapping Service}}
        V[OwlValidatorService<br/>(horned-owl + whelk-rs)]
        R[Reasoner]
        
        M -- "Maps to RDF Triples" --> V
        V -- "Consistency Checks" --> R
        R -- "Inference" --> V
    end

    subgraph "Formal OWL/RDF Graph"
        direction LR
        T1["(A, rdf:type, :Person)"]
        T2["(C, rdf:type, :Company)"]
        T3["(C, :employs, A)"]
        I1["(A, :worksFor, C)"]
    end
    
    A --> M
    B --> M
    C --> M
    
    V -- "ValidationReport" --> A

    style I1 fill:#D69E2E,stroke:#333,stroke-width:2px,color:#fff
```

This hybrid approach allows developers and users to interact with a simple, flexible graph structure while leveraging the power of formal semantics for validation and advanced reasoning in the background.

## Architecture

### `OwlValidatorService`
-   **Location**: `owl_validator.rs` (planned)
-   **Description**: This is the core service responsible for the validation logic. It uses the `horned-owl` crate to parse OWL/RDF ontologies and the `whelk-rs` crate as its reasoner. The reasoner checks for logical inconsistencies (e.g., a node being an instance of two disjoint classes) and performs inference to derive new facts from the existing data and ontology axioms.

### `OntologyActor`
-   **Location**: `ontology_actor.rs` (planned)
-   **Description**: This actor provides an asynchronous, message-based API for the validation service. It is designed to handle potentially long-running reasoning tasks without blocking the main server threads.
-   **Message API**: It receives a `ValidateGraph` message, which contains the graph data to be validated. It then uses the `OwlValidatorService` to perform the validation and returns a `ValidationReport` to the caller.

## Workflow

The validation process follows a clear, step-by-step workflow:

1.  **Mapping to RDF**: The property graph is translated into a set of RDF triples. Node properties become data properties, and edges become object properties.
2.  **Consistency Check**: The `whelk-rs` reasoner loads the RDF triples and the domain ontology. It then checks for any logical contradictions based on the axioms defined in the ontology (e.g., `disjointWith`, cardinality restrictions).
3.  **Inference**: If the graph is consistent, the reasoner performs inference to generate new, implicit triples. For example, if the ontology states that `:employs` is the inverse property of `:worksFor`, the reasoner will infer that `(A, :worksFor, C)` is true if `(C, :employs, A)` exists.
4.  **Feedback Loop**: The results, including any inconsistencies or newly inferred relationships, are compiled into a `ValidationReport` and sent back to the `GraphServiceActor`. This report can be used to correct the graph or enrich it with the inferred knowledge.

## API Endpoint

-   **Endpoint**: `POST /api/analytics/validate` (planned)
-   **Description**: This endpoint will expose the ontology validation functionality over the REST API, allowing clients to request a formal validation of the current graph state.

## Use Cases

### Semantically-Aware Physics

The primary use case for the ontology system is to enable "Semantically-Aware Physics." The logical axioms in the ontology can be used to generate "hard" constraints for the physics engine.

For example:
-   An axiom like `Person disjointWith Company` can be translated into a strong repulsive force between all nodes of type `Person` and all nodes of type `Company`.
-   An axiom defining a parent-child relationship can be used to create a stronger attractive force between the parent and its children.

This allows the final graph layout to reflect the logical structure of the data, creating a more intuitive and insightful visualization.