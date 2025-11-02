# Ontology-Based Graph Rendering

## Overview

VisionFlow now supports ontology-based semantic visualization using OWL class IRIs. Nodes can be rendered with class-specific visual properties (colors, shapes, sizes) based on their ontology classification.

## Backend Support (✅ COMPLETE)

### Phase 1-4 Implementation Complete:
- ✅ Database schema with `owl_class_iri` foreign key
- ✅ OntologyConverter service populates owl_class_iri
- ✅ GPU metadata buffers (`class_id`, `class_charge`, `class_mass`)
- ✅ WebSocket protocol sends `owl_class_iri` in InitialNodeData
- ✅ TypeScript types updated (GraphNode interface)

## Client-Side Implementation (TODO)

### 1. Class-Based Rendering
**File**: `client/src/features/graph/components/GraphManager.tsx`
**Implementation Needed**:
```typescript
// Map OWL class IRI to visual properties
const getClassVisualProperties = (owlClassIri?: string) => {
  if (!owlClassIri) return { color: '#CCCCCC', geometry: 'sphere', size: 1.0 };

  if (owlClassIri.includes('Person') || owlClassIri.includes('Individual')) {
    return { color: '#90EE90', geometry: 'sphere', size: 0.8 };
  } else if (owlClassIri.includes('Company') || owlClassIri.includes('Organization')) {
    return { color: '#4169E1', geometry: 'cube', size: 1.2 };
  } else if (owlClassIri.includes('Project')) {
    return { color: '#FFA500', geometry: 'cone', size: 1.0 };
  } else if (owlClassIri.includes('Concept')) {
    return { color: '#9370DB', geometry: 'octahedron', size: 0.9 };
  } else if (owlClassIri.includes('Technology')) {
    return { color: '#00CED1', geometry: 'tetrahedron', size: 1.1 };
  }

  return { color: '#CCCCCC', geometry: 'sphere', size: 1.0 };
};

// Apply when creating Three.js meshes
nodes.forEach(node => {
  const props = getClassVisualProperties(node.owlClassIri);
  const geometry = createGeometry(props.geometry, props.size);
  const material = new THREE.MeshStandardMaterial({ color: props.color });
  const mesh = new THREE.Mesh(geometry, material);
  // ... add to scene
});
```

### 2. Ontology Tree View Component
**File**: `client/src/features/ontology/components/OntologyTreeView.tsx`
**Purpose**: Hierarchical view of ontology classes with filtering
**Features Needed**:
- Tree view of owl_classes from backend API
- Click to filter graph by class
- Show subclass relationships
- Display class metadata (label, description)

**API Endpoint Needed** (backend):
```rust
// GET /api/ontology/classes
// Returns: Vec<OwlClass> with hierarchy info
```

### 3. Class-Based Filtering
**File**: `client/src/features/graph/services/graphFiltering.ts`
**Features**:
```typescript
interface ClassFilter {
  includedClasses: string[]; // OWL class IRIs to show
  excludedClasses: string[]; // OWL class IRIs to hide
  showUnclassified: boolean; // Show nodes without owl_class_iri
}

const filterNodesByClass = (nodes: GraphNode[], filter: ClassFilter) => {
  return nodes.filter(node => {
    if (!node.owlClassIri) return filter.showUnclassified;
    if (filter.excludedClasses.includes(node.owlClassIri)) return false;
    if (filter.includedClasses.length === 0) return true;
    return filter.includedClasses.includes(node.owlClassIri);
  });
};
```

### 4. Node Collapsing/Grouping (Future)
**Feature**: Collapse nodes of same class into a "super-node"
**Use Case**: When many Person nodes exist, collapse into single "Person Group" node
**Implementation**:
- Detect when >N nodes share same owl_class_iri
- Create virtual "group" node with aggregated properties
- Expand on click to show individual nodes

## Data Flow

```
GitHub Markdown → OntologyParser → OwlClass (IRI) ┐
                                                   ↓
                     OntologyConverter ← UnifiedOntologyRepository
                            ↓
                   Node (owl_class_iri populated)
                            ↓
                UnifiedGraphRepository → Database
                            ↓
                  WebSocket (InitialNodeData)
                            ↓
                Client TypeScript (GraphNode)
                            ↓
            [TODO] Class-Based Rendering Logic
                            ↓
                Three.js Mesh (Color/Shape/Size)
```

## Ontology Class Examples

From `bin/load_ontology.rs`:
- `mv:Person` → Green sphere, small
- `mv:Company` → Blue cube, large
- `mv:Project` → Orange cone, medium
- `mv:Concept` → Purple octahedron, small-medium
- `mv:Technology` → Dark turquoise tetrahedron, medium-large

## Integration with Existing Features

### GPU Physics
The GPU already has:
- `class_id` buffer (integer class IDs)
- `class_charge` buffer (class-specific charge modifiers)
- `class_mass` buffer (class-specific mass modifiers)

**TODO**: Populate these from owl_class_iri mappings in GraphManager

### Advanced Features (Using hornedowl/whelk-rs)
Future reasoning capabilities:
- Infer class memberships from properties
- Detect ontology violations
- Suggest class assignments for new nodes
- Semantic search by class hierarchy

## Testing Client Implementation

1. Run backend: `cargo run --bin load_ontology`
2. Start server: `cargo run --release`
3. Open client: `npm start`
4. Check browser console for `owl_class_iri` in initial graph load
5. Verify nodes have `owlClassIri` property
6. Implement rendering logic and test visual differentiation

## Next Steps (Priority Order)

1. ✅ Update GraphNode TypeScript type (DONE)
2. 🔲 Implement `getClassVisualProperties()` in GraphManager
3. 🔲 Apply class-based rendering in Three.js scene
4. 🔲 Create OntologyTreeView component
5. 🔲 Add class filtering UI
6. 🔲 Backend API endpoint for ontology classes
7. 🔲 Node collapsing/grouping feature
8. 🔲 Integrate with hornedowl reasoning

## Notes

- Client-side collapsing logic: NOT YET IMPLEMENTED (user requested checking)
- Hive mind approach recommended for complex features
- Use docker skill to test in host environment
- MCP devtool available for client debugging
