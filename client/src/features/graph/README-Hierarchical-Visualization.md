# Hierarchical Graph Visualization with Semantic Zoom

## Overview

This implementation provides hierarchical graph visualization with semantic zoom capabilities for ontology-based knowledge graphs. It allows users to navigate class hierarchies by collapsing/expanding groups and zooming between detail levels.

## Components Created

### 1. Ontology Store (`useOntologyStore.ts`)
**Location:** `/client/src/features/ontology/store/useOntologyStore.ts`

**Features:**
- Class hierarchy management with Map-based storage
- Expansion/collapse state tracking
- Semantic zoom levels (0-5)
- Class visibility filtering
- Computed visibility based on hierarchy depth

**Key State:**
```typescript
interface OntologyState {
  hierarchy: ClassHierarchy | null;
  expandedClasses: Set<string>;
  collapsedClasses: Set<string>;
  semanticZoomLevel: number; // 0-5
  visibleClasses: Set<string>;
  highlightedClass: string | null;
}
```

**Usage:**
```typescript
const {
  hierarchy,
  semanticZoomLevel,
  toggleClass,
  setZoomLevel
} = useOntologyStore();
```

### 2. Semantic Zoom Controls (`SemanticZoomControls.tsx`)
**Location:** `/client/src/features/visualisation/components/ControlPanel/SemanticZoomControls.tsx`

**Features:**
- Zoom level slider (0-5) with labels
- Expand/Collapse all buttons
- Auto-zoom toggle (camera distance-based)
- Class filter checkboxes
- Real-time statistics display

**Zoom Levels:**
- **Level 0:** All Instances (show everything)
- **Level 1:** Detailed (all classes visible)
- **Level 2:** Standard (reduce depth by 1)
- **Level 3:** Grouped (show class groups)
- **Level 4:** High-Level (only upper hierarchy)
- **Level 5:** Top Classes (roots only)

### 3. Hierarchical Renderer Utilities (`hierarchicalRenderer.ts`)
**Location:** `/client/src/features/graph/utils/hierarchicalRenderer.ts`

**Functions:**
- `groupNodesByClass()` - Group instances by class
- `getColorForDepth()` - Depth-based coloring
- `calculateTransitionState()` - Smooth animation states
- `filterNodesByZoomLevel()` - Semantic zoom filtering
- `highlightSameClass()` - Class-based selection

### 4. Hierarchical Graph Renderer (`HierarchicalGraphRenderer.tsx`)
**Location:** `/client/src/features/graph/components/HierarchicalGraphRenderer.tsx`

**Features:**
- Dual rendering mode (individual vs grouped)
- Class group spheres for collapsed classes
- Click to expand/collapse
- Double-click to highlight same class
- Billboard labels with instance counts

**Rendering Logic:**
```typescript
const renderMode = semanticZoomLevel >= 3 ? 'grouped' : 'individual';
```

### 5. Class Group Tooltip (`ClassGroupTooltip.tsx`)
**Location:** `/client/src/features/visualisation/components/ClassGroupTooltip.tsx`

**Features:**
- Hover tooltips for class groups
- Shows IRI, depth, parent, children
- Instance count badge
- Interaction hints

### 6. Hierarchical Animation Hook (`useHierarchicalAnimation.ts`)
**Location:** `/client/src/features/graph/hooks/useHierarchicalAnimation.ts`

**Features:**
- Smooth expand/collapse animations
- Position interpolation
- Scale transitions
- Ease-in-out cubic easing
- 800ms default duration

## Integration with GraphManager

### Adding to GraphManager.tsx

```typescript
import { useOntologyStore } from '../../ontology/store/useOntologyStore';
import { HierarchicalGraphRenderer } from './HierarchicalGraphRenderer';

// Inside component:
const { semanticZoomLevel } = useOntologyStore();
const useHierarchicalMode = semanticZoomLevel >= 3;

// In render:
{useHierarchicalMode ? (
  <HierarchicalGraphRenderer
    nodes={visibleNodes}
    edges={graphData.edges}
    nodePositions={nodePositionsRef.current}
    onNodeClick={(nodeId, event) => {
      const nodeIndex = visibleNodes.findIndex(n => n.id === nodeId);
      if (nodeIndex !== -1) {
        handlePointerDown({ ...event, instanceId: nodeIndex });
      }
    }}
    settings={settings}
  />
) : (
  // Existing instancedMesh rendering
)}
```

### Adding Controls to UI

```typescript
import { SemanticZoomControls } from '../../visualisation/components/ControlPanel/SemanticZoomControls';

// In your control panel component:
<SemanticZoomControls className="absolute top-4 right-4" />
```

## Interaction Patterns

### Click Interactions
1. **Click on collapsed class group** → Expands to show instances
2. **Click on expanded instance** → Standard node selection
3. **Double-click on instance** → Highlights all instances of same class

### Hover Interactions
1. **Hover on class group** → Shows tooltip with details
2. **Hover on instance** → Shows node metadata (existing behavior)

### Zoom Interactions
1. **Slider change** → Adjusts visible hierarchy depth
2. **Auto-zoom enabled** → Adjusts based on camera distance

## Animation System

### Expand Animation
```typescript
const { startExpandAnimation } = useHierarchicalAnimation();

// When expanding a class:
startExpandAnimation(
  collapsedCenterPosition,
  expandedNodePositions,
  nodeIds
);
```

### Collapse Animation
```typescript
const { startCollapseAnimation } = useHierarchicalAnimation();

// When collapsing a class:
startCollapseAnimation(
  expandedNodePositions,
  collapsedCenterPosition,
  nodeIds
);
```

## API Requirements

### Backend Endpoint
The ontology store fetches hierarchy from:
```
GET /api/ontology/hierarchy
```

**Expected Response:**
```json
{
  "classes": [
    {
      "iri": "http://example.org/Class1",
      "label": "Class 1",
      "parentIri": null,
      "childIris": ["http://example.org/SubClass1"],
      "instanceCount": 42,
      "depth": 0,
      "description": "Top-level class"
    }
  ]
}
```

### Graph Node Metadata
Nodes should include class information:
```typescript
interface GraphNode {
  id: string;
  label: string;
  position: { x: number; y: number; z: number };
  metadata?: {
    classIri?: string;
    type?: string;
    // ... other metadata
  };
}
```

## Performance Considerations

### Instancing
- Collapsed class groups use single meshes
- Individual nodes use instanced rendering
- Smooth transitions between modes

### LOD (Level of Detail)
- Semantic zoom reduces visible nodes
- Physics simulation runs on all nodes
- Rendering filters based on hierarchy

### Memory
- Map-based storage for O(1) lookups
- Set-based expansion state
- Efficient hierarchy traversal

## Styling

### Colors by Depth
- Depth 0: Red (`#FF6B6B`)
- Depth 1: Cyan (`#4ECDC4`)
- Depth 2: Yellow (`#FFD93D`)
- Depth 3: Light Cyan (`#95E1D3`)
- Depth 4: Purple (`#AA96DA`)
- Depth 5+: Pink (`#F38181`)

### Scale Factors
- Collapsed groups: `1 + log(instanceCount + 1)` (max 5)
- Highlighted: 1.3x base size
- Animated: Smooth interpolation

## Testing

### Manual Testing Checklist
- [ ] Load ontology hierarchy successfully
- [ ] Zoom slider changes visible nodes
- [ ] Click on class group expands it
- [ ] Expand/Collapse all buttons work
- [ ] Smooth animations between states
- [ ] Tooltips show correct information
- [ ] Double-click highlights same class
- [ ] Auto-zoom responds to camera distance

### Performance Testing
- [ ] 1000+ nodes render smoothly
- [ ] Animations maintain 60 FPS
- [ ] Memory usage stays stable
- [ ] Physics simulation unaffected

## Future Enhancements

1. **Auto-Zoom Implementation**
   - Adjust zoom based on camera distance
   - Smooth transitions between levels

2. **Advanced Filtering**
   - Property-based filters
   - Relationship type filters
   - Custom predicates

3. **Minimap**
   - Overview of full hierarchy
   - Navigation aid

4. **Search Integration**
   - Find and highlight classes
   - Breadcrumb navigation

5. **Export/Import**
   - Save view states
   - Share configurations

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `useOntologyStore.ts` | 285 | State management for hierarchy |
| `SemanticZoomControls.tsx` | 250 | UI controls for zoom/expand |
| `hierarchicalRenderer.ts` | 200 | Rendering utilities |
| `HierarchicalGraphRenderer.tsx` | 220 | Main rendering component |
| `ClassGroupTooltip.tsx` | 180 | Tooltip display |
| `useHierarchicalAnimation.ts` | 190 | Animation system |

**Total:** ~1,325 lines of production code

## Dependencies

- `zustand` - State management
- `@react-three/fiber` - React Three.js rendering
- `@react-three/drei` - Three.js helpers (Billboard, Text, Html)
- `three` - 3D rendering library

## Support

For issues or questions, refer to:
- GraphManager.tsx implementation
- Existing expansion state hooks
- Logger utilities for debugging
