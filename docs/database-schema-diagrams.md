# VisionFlow Database Schema Diagrams

## Current Schema (As Implemented)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED DATABASE (unified.db)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        ONTOLOGY TABLES                              │    │
│  │                                                                      │    │
│  │  ┌─────────────────────┐                                            │    │
│  │  │   owl_classes       │                                            │    │
│  │  ├─────────────────────┤                                            │    │
│  │  │ iri (PK) ◄──────────┼────┐ (Referenced by graph_nodes)          │    │
│  │  │ label               │    │                                       │    │
│  │  │ description         │    │                                       │    │
│  │  │ file_sha1           │    │                                       │    │
│  │  │ markdown_content    │    │                                       │    │
│  │  │ last_synced         │    │                                       │    │
│  │  └─────────────────────┘    │                                       │    │
│  │          ▲                   │                                       │    │
│  │          │                   │                                       │    │
│  │          │                   │                                       │    │
│  │  ┌───────┴─────────────┐    │                                       │    │
│  │  │ owl_class_hierarchy │    │                                       │    │
│  │  ├─────────────────────┤    │                                       │    │
│  │  │ class_iri (FK)      │    │                                       │    │
│  │  │ parent_iri (FK)     │    │                                       │    │
│  │  └─────────────────────┘    │                                       │    │
│  │                               │                                       │    │
│  │  ┌─────────────────────┐    │                                       │    │
│  │  │   owl_properties    │    │                                       │    │
│  │  ├─────────────────────┤    │                                       │    │
│  │  │ iri (PK)            │    │                                       │    │
│  │  │ property_type       │    │                                       │    │
│  │  │ domain              │    │                                       │    │
│  │  │ range               │    │                                       │    │
│  │  └─────────────────────┘    │                                       │    │
│  │                               │                                       │    │
│  │  ┌─────────────────────┐    │                                       │    │
│  │  │   owl_axioms        │    │                                       │    │
│  │  ├─────────────────────┤    │                                       │    │
│  │  │ axiom_type          │    │                                       │    │
│  │  │ subject             │    │                                       │    │
│  │  │ object              │    │                                       │    │
│  │  │ annotations         │    │                                       │    │
│  │  └─────────────────────┘    │                                       │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        GRAPH TABLES                                 │    │
│  │                                                                      │    │
│  │  ┌─────────────────────┐                                            │    │
│  │  │   graph_nodes       │                                            │    │
│  │  ├─────────────────────┤                                            │    │
│  │  │ id (PK)             │                                            │    │
│  │  │ metadata_id (UNIQUE)│                                            │    │
│  │  │ label               │                                            │    │
│  │  │                     │                                            │    │
│  │  │ -- Physics State    │                                            │    │
│  │  │ x, y, z             │  (3D position)                             │    │
│  │  │ vx, vy, vz          │  (velocity)                                │    │
│  │  │ mass, charge        │  (physics properties)                      │    │
│  │  │                     │                                            │    │
│  │  │ -- Ontology Link    │                                            │    │
│  │  │ owl_class_iri (FK) ─┼────┘ ⚠️  CRITICAL: NEVER POPULATED!       │    │
│  │  │                     │                                            │    │
│  │  │ -- Visualization    │                                            │    │
│  │  │ color, size         │                                            │    │
│  │  │ node_type           │                                            │    │
│  │  │                     │                                            │    │
│  │  │ -- Metadata         │                                            │    │
│  │  │ metadata (JSON)     │                                            │    │
│  │  └─────────────────────┘                                            │    │
│  │          │                                                           │    │
│  │          │ (Referenced by edges)                                    │    │
│  │          │                                                           │    │
│  │  ┌───────┴─────────────┐                                            │    │
│  │  │   graph_edges       │                                            │    │
│  │  ├─────────────────────┤                                            │    │
│  │  │ id (PK)             │                                            │    │
│  │  │ source_id (FK) ─────┼─────► graph_nodes.id                      │    │
│  │  │ target_id (FK) ─────┼─────► graph_nodes.id                      │    │
│  │  │ weight              │                                            │    │
│  │  │ relation_type       │                                            │    │
│  │  │ metadata (JSON)     │                                            │    │
│  │  └─────────────────────┘                                            │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Problem: The Missing Link

```
┌────────────────────────────────────────────────────────────────────┐
│                  INTENDED RELATIONSHIP (BROKEN)                     │
└────────────────────────────────────────────────────────────────────┘

  owl_classes                                graph_nodes
  ┌─────────────────┐                       ┌──────────────────┐
  │ iri             │◄─────────────────────┼│ owl_class_iri    │
  │ label           │     FOREIGN KEY       │ id               │
  │ description     │     (NOT POPULATED!)  │ label            │
  │ markdown_content│                       │ x, y, z          │
  └─────────────────┘                       └──────────────────┘
         ▲                                            │
         │                                            │
         │                                            ▼
         │                                   ┌──────────────────┐
         │                                   │ graph_edges      │
         │                                   ├──────────────────┤
         │                                   │ source_id (FK)   │
         │                                   │ target_id (FK)   │
         │                                   └──────────────────┘
         │
         │
    ┌────┴────────┐
    │ PROBLEM:    │
    │ Parser has  │
    │ NO ACCESS   │
    │ to ontology │
    └─────────────┘
```

## Proposed Schema: Ontology as Primary Source

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               RECOMMENDED ARCHITECTURE (Ontology-First)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  STEP 1: Load Ontology (Primary Source of Truth)                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │   owl_classes (PRIMARY NODE STORAGE)                                │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │ iri (PK)              │ "your_ontology:Person"                      │    │
│  │ label                 │ "Person"                                    │    │
│  │ description           │ "A human individual"                        │    │
│  │ parent_classes (JSON) │ ["your_ontology:Agent"]                    │    │
│  │ properties (JSON)     │ {"name": "...", "age": "..."}              │    │
│  │ markdown_content      │ "# Person\nA class representing..."        │    │
│  │ file_sha1             │ "abc123..."                                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                 │                                             │
│                                 │ (1 to many)                                 │
│                                 ▼                                             │
│  STEP 2: Create Visualization State (Secondary)                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │   graph_nodes (PHYSICS STATE ONLY)                                  │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │ id (PK)               │ 42                                          │    │
│  │ owl_class_iri (FK) ───┼───► REQUIRED! Points to owl_classes.iri    │    │
│  │                       │                                             │    │
│  │ -- Physics State      │                                             │    │
│  │ x, y, z               │ 100.5, 200.3, 50.7                         │    │
│  │ vx, vy, vz            │ 5.2, -3.1, 0.8                             │    │
│  │ mass                  │ 1.0                                         │    │
│  │                       │                                             │    │
│  │ -- Visual Overrides   │                                             │    │
│  │ color                 │ "#FF5733" (override class default)         │    │
│  │ size                  │ 15.0 (override class default)              │    │
│  │ is_pinned             │ true                                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                 │                                             │
│                                 │                                             │
│                                 ▼                                             │
│  STEP 3: Semantic Edges (Property-Based)                                     │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │   graph_edges                                                        │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │ source_iri (FK) ──────┼───► owl_classes.iri (not graph_nodes.id!)  │    │
│  │ target_iri (FK) ──────┼───► owl_classes.iri                         │    │
│  │ property_iri (FK) ────┼───► owl_properties.iri                      │    │
│  │ weight                │ 1.0                                         │    │
│  │ rest_length           │ 50.0                                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow: How It Should Work

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ONTOLOGY-FIRST WORKFLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Parse Ontology Markdown
┌─────────────────────────────────┐
│ ontology/Person.md              │
│                                 │
│ # Person                        │
│ A class representing humans     │
│                                 │
│ SubClassOf: Agent               │
│ Properties:                     │
│   - name (String)               │
│   - age (Integer)               │
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ OntologyParser                  │
│ (NEW or ENHANCED)               │
└─────────────────────────────────┘
            │
            ▼
INSERT INTO owl_classes (iri, label, description, markdown_content, ...)
VALUES ('your_ontology:Person', 'Person', 'A class...', '# Person\n...', ...)
            │
            ▼
┌─────────────────────────────────┐
│ owl_classes table               │
│ ✅ Populated                    │
└─────────────────────────────────┘


STEP 2: Parse Knowledge Graph with Classification
┌─────────────────────────────────┐
│ pages/Tim_Cook.md               │
│                                 │
│ # Tim Cook                      │
│ CEO of [[Apple Inc.]]           │
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ KnowledgeGraphParser            │
│ (ENHANCED with ontology)        │
├─────────────────────────────────┤
│ 1. Parse markdown               │
│ 2. Extract page name: "Tim Cook"│
│ 3. CLASSIFY:                    │
│    - Query: SELECT iri FROM     │
│      owl_classes WHERE          │
│      label = 'Person'           │
│    - Result: your_ontology:Person│
│ 4. Create Node with:            │
│    owl_class_iri = result       │
└─────────────────────────────────┘
            │
            ▼
INSERT INTO graph_nodes (metadata_id, label, owl_class_iri, ...)
VALUES ('Tim_Cook', 'Tim Cook', 'your_ontology:Person', ...)
            │
            ▼
┌─────────────────────────────────┐
│ graph_nodes table               │
│ ✅ owl_class_iri populated!     │
└─────────────────────────────────┘


STEP 3: Physics Simulation with Semantic Rules
┌─────────────────────────────────┐
│ force_compute_actor.rs          │
│ (ENHANCED with ontology)        │
├─────────────────────────────────┤
│ Query:                          │
│ SELECT n.*, c.label, c.properties│
│ FROM graph_nodes n              │
│ JOIN owl_classes c              │
│   ON n.owl_class_iri = c.iri    │
├─────────────────────────────────┤
│ Apply class-specific rules:     │
│ - Person: cluster=high          │
│ - Company: gravity=strong       │
│ - Concept: floating             │
└─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Rich 3D Visualization           │
│ ✅ Semantic clustering          │
│ ✅ Type-based colors            │
│ ✅ Ontology-driven layout       │
└─────────────────────────────────┘
```

## Index Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CRITICAL INDEXES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  PRIMARY LOOKUPS (High Priority)                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ CREATE INDEX idx_owl_classes_iri ON owl_classes(iri);              │    │
│  │ CREATE INDEX idx_owl_classes_label ON owl_classes(label);          │    │
│  │ CREATE INDEX idx_graph_nodes_owl_class ON graph_nodes(owl_class_iri);│  │
│  │ CREATE INDEX idx_graph_nodes_metadata_id ON graph_nodes(metadata_id);│  │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  JOIN OPTIMIZATION (Medium Priority)                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ CREATE INDEX idx_nodes_class_join                                   │    │
│  │   ON graph_nodes(owl_class_iri, id);                                │    │
│  │                                                                      │    │
│  │ CREATE INDEX idx_hierarchy_covering                                 │    │
│  │   ON owl_class_hierarchy(subclass_iri, superclass_iri, distance);   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  PHYSICS QUERIES (Low Priority)                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ CREATE INDEX idx_nodes_spatial_3d ON graph_nodes(x, y, z);         │    │
│  │ CREATE INDEX idx_edges_source_target ON graph_edges(source_id, target_id);│
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Migration Path

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MIGRATION STRATEGY                                   │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: Initialize Schema (IMMEDIATE)
┌─────────────────────────────────┐
│ $ sqlite3 data/unified.db <     │
│   migration/unified_schema.sql  │
│                                 │
│ Result: Tables created          │
└─────────────────────────────────┘


PHASE 2: Load Ontology (HIGH PRIORITY)
┌─────────────────────────────────┐
│ $ cargo run --bin load_ontology │
│                                 │
│ Reads: ontology/*.md            │
│ Writes: owl_classes,            │
│         owl_properties,         │
│         owl_axioms              │
└─────────────────────────────────┘


PHASE 3: Backfill Existing Nodes (HIGH PRIORITY)
┌─────────────────────────────────┐
│ UPDATE graph_nodes              │
│ SET owl_class_iri = (           │
│   SELECT iri FROM owl_classes   │
│   WHERE LOWER(label) =          │
│         LOWER(graph_nodes.label)│
│   LIMIT 1                       │
│ )                               │
│ WHERE owl_class_iri IS NULL;    │
│                                 │
│ Result: Links existing nodes    │
└─────────────────────────────────┘


PHASE 4: Modify Parser (MEDIUM PRIORITY)
┌─────────────────────────────────┐
│ KnowledgeGraphParser::new(      │
│   ontology_repo                 │
│ )                               │
│                                 │
│ classify_node() method added    │
│                                 │
│ Result: Future nodes classified │
└─────────────────────────────────┘


PHASE 5: Validate (LOW PRIORITY)
┌─────────────────────────────────┐
│ SELECT COUNT(*) FROM graph_nodes│
│ WHERE owl_class_iri IS NULL;    │
│ -- Expected: 0                  │
│                                 │
│ PRAGMA foreign_key_check;       │
│ -- Expected: (empty)            │
└─────────────────────────────────┘
```

## Foreign Key Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     REFERENTIAL INTEGRITY MAP                                │
└─────────────────────────────────────────────────────────────────────────────┘

owl_classes.iri ◄──────────────────────┐
                                        │
                                        │ FK (CASCADE DELETE)
                                        │
graph_nodes.owl_class_iri ──────────────┘
    │
    │ FK (CASCADE DELETE)
    ▼
graph_edges.source_id ──────────► graph_nodes.id
graph_edges.target_id ──────────► graph_nodes.id


owl_classes.iri ◄──────────────────────┐
                                        │
                                        │ FK (CASCADE DELETE)
                                        │
owl_class_hierarchy.class_iri ──────────┘
owl_class_hierarchy.parent_iri ─────────┘


owl_properties.domain_class_iri ────► owl_classes.iri (SET NULL)
owl_properties.range_class_iri ─────► owl_classes.iri (SET NULL)
```

## Summary

This schema provides:

✅ **Ontology as Source of Truth**: owl_classes is primary node storage
✅ **Foreign Key Integrity**: All relationships enforced
✅ **Efficient Queries**: Proper indexes for joins
✅ **Migration Path**: Clear steps from current to target state
✅ **Validation**: Queries to verify data quality

The critical missing piece is populating `owl_class_iri` during parsing.
