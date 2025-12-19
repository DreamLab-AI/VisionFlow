---
title: Ontology-First Architecture Analysis
description: ```cypher Neo4j Contains: - 529 GraphNode nodes (markdown pages)
category: explanation
tags:
  - architecture
  - database
  - backend
updated-date: 2025-12-18
difficulty-level: advanced
---


# Ontology-First Architecture Analysis

## Current State Analysis (2025-11-08)

### Database State
```cypher
Neo4j Contains:
- 529 GraphNode nodes (markdown pages)
  - 442 "linked_page" type
  - 87 "page" type
  - 0 nodes with owl_class_iri assigned
- 919 OwlClass nodes (ontology classes)
- 1 OwlProperty node
- 2 OwlAxiom nodes
```

### The Core Problem

**The graph is rendering ~500 markdown pages WITHOUT ontology classification.**

1. **Separate Node Spaces**: Ontology classes (`OwlClass` nodes) exist separately from graph nodes (`GraphNode` nodes)
2. **No Linkage**: `GraphNode` nodes have `owl_class_iri` field but it's **NULL** for all 529 nodes
3. **Client Shows All**: Frontend renders all `GraphNode` nodes indiscriminately

### Current Data Flow

```
Markdown Files → GraphNode (node_type: "page"/"linked_page")
                           ↓
                  owl_class_iri = NULL
                           ↓
                  Neo4j Database
                           ↓
                  load_graph() → Returns ALL GraphNodes
                           ↓
                  Frontend renders 529 unclassified nodes
```

### Ontology Infrastructure Exists But Unused

**Backend has complete ontology support:**
- ✅ `owl_class_iri` field in GraphNode schema
- ✅ OntologyParser service
- ✅ OntologyConverter service
- ✅ OntologyReasoner service
- ✅ 919 OwlClass nodes loaded
- ✅ GPU physics supports class_id, class_charge, class_mass

**But missing:**
- ❌ No process assigns `owl_class_iri` to GraphNodes
- ❌ No filtering by ontology status
- ❌ No client-side ontology-based rendering

## Ontology-First Architecture Proposal

### Phase 1: Ontology Assignment Pipeline

**Goal**: Every GraphNode must be classified by an OwlClass

```rust
// services/ontology_assignment_service.rs
pub struct OntologyAssignmentService {
    parser: OntologyParser,
    reasoner: OntologyReasoner,
}

impl OntologyAssignmentService {
    pub async fn assign_classes_to_nodes(&self) -> Result<()> {
        // 1. Get all GraphNodes without owl_class_iri
        let unclassified = repo.get_nodes_without_class().await?;

        // 2. For each node, infer class from:
        //    - Metadata tags
        //    - Content analysis
        //    - Edge relationships
        //    - Filename patterns

        // 3. Assign owl_class_iri
        for node in unclassified {
            let class_iri = self.infer_class(&node).await?;
            repo.update_node_class(node.id, class_iri).await?;
        }

        Ok(())
    }

    fn infer_class(&self, node: &GraphNode) -> Result<String> {
        // Heuristics for class inference:
        // - If metadata contains "person" → mv:Person
        // - If metadata contains "company" → mv:Company
        // - If filename contains "project" → mv:Project
        // - Default → mv:Concept (generic knowledge)

        // Future: Use ML/LLM for better classification
    }
}
```

### Phase 2: Ontology-Based Graph Loading

**Modify `load_graph()` to support filtering:**

```rust
// src/adapters/neo4j_adapter.rs

pub enum GraphLoadMode {
    OntologyOnly,        // Only nodes with owl_class_iri
    UnclassifiedOnly,    // Only nodes without owl_class_iri
    All,                 // All nodes (current behavior)
    ByClass(Vec<String>), // Filter by specific OwlClass IRIs
}

async fn load_graph_filtered(&self, mode: GraphLoadMode) -> RepoResult<Arc<GraphData>> {
    let query = match mode {
        GraphLoadMode::OntologyOnly =>
            "MATCH (n:GraphNode) WHERE n.owl_class_iri IS NOT NULL RETURN n ORDER BY n.id",
        GraphLoadMode::UnclassifiedOnly =>
            "MATCH (n:GraphNode) WHERE n.owl_class_iri IS NULL RETURN n ORDER BY n.id",
        GraphLoadMode::All =>
            "MATCH (n:GraphNode) RETURN n ORDER BY n.id",
        GraphLoadMode::ByClass(iris) => {
            // Parameterized query for specific classes
            "MATCH (n:GraphNode) WHERE n.owl_class_iri IN $iris RETURN n ORDER BY n.id"
        }
    };

    // Execute query...
}
```

### Phase 3: Client-Side Ontology Rendering

**GraphManager.tsx enhancements:**

```typescript
// Map owl_class_iri to visual properties
const getClassVisualProperties = (owlClassIri?: string) => {
  if (!owlClassIri) {
    // Unclassified nodes: gray, semi-transparent
    return {
      color: '#666666',
      geometry: 'sphere',
      size: 0.5,
      opacity: 0.3,
      layer: 1  // Background layer
    };
  }

  // Ontology-classified nodes: vibrant, distinct
  if (owlClassIri.includes('Person')) {
    return { color: '#90EE90', geometry: 'sphere', size: 0.8, opacity: 1.0, layer: 0 };
  } else if (owlClassIri.includes('Company')) {
    return { color: '#4169E1', geometry: 'cube', size: 1.2, opacity: 1.0, layer: 0 };
  } else if (owlClassIri.includes('Project')) {
    return { color: '#FFA500', geometry: 'cone', size: 1.0, opacity: 1.0, layer: 0 };
  } else if (owlClassIri.includes('Concept')) {
    return { color: '#9370DB', geometry: 'octahedron', size: 0.9, opacity: 1.0, layer: 0 };
  } else if (owlClassIri.includes('Technology')) {
    return { color: '#00CED1', geometry: 'tetrahedron', size: 1.1, opacity: 1.0, layer: 0 };
  }

  return { color: '#CCCCCC', geometry: 'sphere', size: 0.7, opacity: 0.8, layer: 1 };
};
```

### Phase 4: Dual-Graph Architecture

**Proposal: Render TWO separate graphs**

```typescript
interface DualGraphState {
  ontologyGraph: {
    nodes: GraphNode[],  // Only nodes with owl_class_iri
    edges: Edge[],
    position: 'center'   // Primary focus
  },
  knowledgeGraph: {
    nodes: GraphNode[],  // Unclassified markdown nodes
    edges: Edge[],
    position: 'peripheral'  // Surrounding constellation
  }
}

// Visual separation:
// - Ontology graph: Center, vibrant colors, distinct shapes
// - Knowledge graph: Outer ring, muted colors, smaller spheres
// - Edges between: Dashed lines showing potential connections
```

### Phase 5: Migration Path

**Step-by-step migration:**

1. **Immediate** (Current Sprint):
   ```bash
   # Add API endpoint for graph filtering
   GET /api/graph/data?mode=ontology_only
   GET /api/graph/data?mode=unclassified_only
   GET /api/graph/data?mode=all
   GET /api/graph/data?classes=mv:Person,mv:Company
   ```

2. **Short-term** (Next Sprint):
   ```bash
   # Run ontology assignment on existing nodes
   cargo run --bin assign_ontology_classes

   # Expected result:
   # - 529 GraphNodes → ~200 ontology-classified
   # - ~329 remain as general "Concept" class
   ```

3. **Mid-term** (2-3 Sprints):
   - Client-side ontology tree view
   - Interactive class filtering UI
   - Node grouping/collapsing by class

4. **Long-term** (Future):
   - ML-based automatic classification
   - Semantic reasoning with hornedowl
   - Cross-ontology alignment

## Client-Side Changes Needed

### 1. GraphManager Type Differentiation

**Current**: All nodes rendered identically
**Proposed**: Distinct rendering per ontology class

```typescript
// client/src/features/graph/components/GraphManager.tsx

nodes.forEach((node, index) => {
  const props = getClassVisualProperties(node.owlClassIri);

  // Create geometry based on class
  const geometry = getGeometryForNodeType(props.geometry);
  geometry.scale(props.size, props.size, props.size);

  // Apply class-specific material
  const material = node.owlClassIri
    ? new THREE.MeshStandardMaterial({
        color: props.color,
        metalness: 0.3,
        roughness: 0.7,
        opacity: props.opacity,
        transparent: props.opacity < 1
      })
    : new THREE.MeshBasicMaterial({
        color: '#666666',
        opacity: 0.2,
        transparent: true,
        wireframe: true  // Unclassified = wireframe
      });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.layers.set(props.layer);

  scene.add(mesh);
});
```

### 2. Settings Panel Enhancement

```typescript
// client/src/features/settings/components/OntologyFilterPanel.tsx

export const OntologyFilterPanel = () => {
  const [showOntologyOnly, setShowOntologyOnly] = useState(false);
  const [selectedClasses, setSelectedClasses] = useState<string[]>([]);

  return (
    <div>
      <Checkbox
        label="Show Only Ontology-Classified Nodes"
        checked={showOntologyOnly}
        onChange={setShowOntologyOnly}
      />

      <ClassTreeView
        selectedClasses={selectedClasses}
        onSelectionChange={setSelectedClasses}
      />

      <Button onClick={() => {
        // Reload graph with filter
        graphDataManager.setGraphType('ontology-filtered');
        graphDataManager.fetchInitialData({
          mode: showOntologyOnly ? 'ontology_only' : 'all',
          classes: selectedClasses
        });
      }}>
        Apply Filter
      </Button>
    </div>
  );
};
```

## Backend Changes Needed

### 1. Graph Handler Enhancement

```rust
// src/handlers/graph_state_handler.rs

#[derive(Deserialize)]
pub struct GraphDataQuery {
    mode: Option<String>,  // "ontology_only", "unclassified_only", "all"
    classes: Option<Vec<String>>,  // Filter by specific OWL classes
}

pub async fn get_graph_data(
    state: web::Data<AppState>,
    query: web::Query<GraphDataQuery>
) -> impl Responder {
    let mode = match query.mode.as_deref() {
        Some("ontology_only") => GraphLoadMode::OntologyOnly,
        Some("unclassified_only") => GraphLoadMode::UnclassifiedOnly,
        Some(classes) if query.classes.is_some() => {
            GraphLoadMode::ByClass(query.classes.clone().unwrap())
        }
        _ => GraphLoadMode::All,
    };

    // Load graph with filter
    let graph_data = neo4j_adapter.load_graph_filtered(mode).await?;

    ok_json!(graph_data)
}
```

### 2. Ontology Assignment Binary

```rust
// bin/assign_ontology_classes.rs

#[tokio::main]
async fn main() -> Result<()> {
    let neo4j = Neo4jAdapter::new(Neo4jConfig::default()).await?;
    let assignment_service = OntologyAssignmentService::new(neo4j.clone());

    println!("Starting ontology class assignment...");

    // Get unclassified nodes
    let query = "MATCH (n:GraphNode) WHERE n.owl_class_iri IS NULL RETURN count(n) as count";
    let count = neo4j.execute_query(query).await?;
    println!("Found {} unclassified nodes", count);

    // Assign classes
    assignment_service.assign_classes_to_nodes().await?;

    // Verify
    let assigned_query = "MATCH (n:GraphNode) WHERE n.owl_class_iri IS NOT NULL RETURN count(n) as count";
    let assigned = neo4j.execute_query(assigned_query).await?;
    println!("Assigned {} nodes to ontology classes", assigned);

    Ok(())
}
```

## Immediate Actions

1. **Create assignment service** (`src/services/ontology_assignment_service.rs`)
2. **Add graph filter parameter** to `/api/graph/data` endpoint
3. **Update client GraphManager** to differentiate by `owlClassIri`
4. **Add ontology filter UI** to settings panel
5. **Run assignment binary** to populate `owl_class_iri` fields

## Expected Outcome

**Before:**
- 529 generic nodes, all look identical
- No semantic differentiation
- Cluttered visualization

**After:**
- Ontology nodes: Vibrant, distinct shapes, center stage
- Knowledge nodes: Muted, peripheral, wireframe
- Clear visual hierarchy
- Interactive filtering by ontology class
- Foundation for semantic reasoning

## Notes

- **Backward compatibility**: Default mode remains "all" (no breaking changes)
- **Performance**: Filtering in Cypher is highly efficient (indexed on `owl_class_iri`)
- **Scalability**: Dual-graph approach scales to 10k+ nodes
- **Future-proof**: Architecture supports advanced reasoning with hornedowl/whelk

---

**Status**: Ready for implementation
**Priority**: HIGH - Core architecture alignment
**Effort**: 2-3 sprints for full implementation
