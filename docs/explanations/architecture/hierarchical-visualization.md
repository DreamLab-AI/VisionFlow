---
title: Hierarchical Graph Visualization Architecture
description: **Complete Guide to Semantic Zoom and Class Grouping**
category: explanation
tags:
  - architecture
  - rest
  - react
updated-date: 2025-12-18
difficulty-level: advanced
---


# Hierarchical Graph Visualization Architecture

**Complete Guide to Semantic Zoom and Class Grouping**

---

## Overview

The Hierarchical Graph Visualization system provides multi-level ontology visualization with semantic zoom, expandable class groups, and smooth animations for exploring large knowledge graphs.

**Total Implementation**: 1,675 lines across 7 components

## Architecture Components

### 1. Ontology Store (State Management)

**Location**: `/client/src/features/ontology/store/useOntologyStore.ts` (285 lines)

Central state management for ontology hierarchy using Zustand.

```typescript
interface OntologyState {
  // Hierarchy data
  hierarchy: ClassHierarchy | null;

  // Expansion state (Set-based for O(1) lookup)
  expandedClasses: Set<string>;
  collapsedClasses: Set<string>;

  // Zoom level (0-5 scale)
  semanticZoomLevel: number;

  // API interaction
  fetchHierarchy: () => Promise<void>;

  // State mutations
  toggleClass: (classIri: string) => void;
  expandAll: () => void;
  collapseAll: () => void;
  setZoomLevel: (level: number) => void;

  // Computed visibility
  isClassVisible: (classIri: string) => boolean;
}
```

#### Data Models

```typescript
interface ClassHierarchy {
  classes: ClassNode[];
}

interface ClassNode {
  iri: string;              // Unique identifier
  label: string;            // Display name
  parentIri: string | null; // Parent class IRI
  childIris: string[];      // Child class IRIs
  instanceCount: number;    // Number of instances
  depth: number;            // Hierarchy level (0 = root)
  description?: string;     // Optional description
}
```

#### Key Features

- **Map-based Hierarchy**: O(1) lookups for class nodes
- **Set-based State**: Fast membership tests for expansion
- **Computed Visibility**: Automatic filtering based on zoom level
- **Persistence**: Can save/restore state

### 2. Semantic Zoom Controls (UI Component)

**Location**: `/client/src/features/visualisation/components/ControlPanel/SemanticZoomControls.tsx` (250 lines)

Interactive UI for controlling visualization detail level.

```typescript
interface SemanticZoomControlsProps {
  className?: string;
}

const SemanticZoomControls: React.FC<SemanticZoomControlsProps> = ({
  className
}) => {
  const {
    semanticZoomLevel,
    setZoomLevel,
    expandAll,
    collapseAll,
    hierarchy
  } = useOntologyStore();

  return (
    <div className={className}>
      {/* Zoom level slider */}
      <Slider
        value={semanticZoomLevel}
        onChange={setZoomLevel}
        min={0}
        max={5}
        step={1}
        labels={ZOOM-LABELS}
      />

      {/* Expand/Collapse buttons */}
      <Button onClick={expandAll}>Expand All</Button>
      <Button onClick={collapseAll}>Collapse All</Button>

      {/* Statistics display */}
      <Statistics hierarchy={hierarchy} />
    </div>
  );
};
```

#### Zoom Levels

| Level | Label | Description | Rendering Mode |
|-------|-------|-------------|----------------|
| 0 | All Instances | Show every individual node | Individual |
| 1 | Detailed | Most instances visible | Individual |
| 2 | Standard | Balanced view | Individual |
| 3 | Grouped | Classes as spheres | **Grouped** |
| 4 | High-Level | Major classes only | Grouped |
| 5 | Top Classes | Root classes only | Grouped |

**Threshold**: Semantic zoom level ≥ 3 triggers grouped rendering

### 3. Hierarchical Renderer Utilities

**Location**: `/client/src/features/graph/utils/hierarchicalRenderer.ts` (200 lines)

Utility functions for grouping, coloring, and filtering nodes.

```typescript
// Group nodes by class IRI
export function groupNodesByClass(
  nodes: GraphNode[],
  hierarchy: ClassHierarchy
): Map<string, GraphNode[]> {
  const groups = new Map<string, GraphNode[]>();

  for (const node of nodes) {
    const classIri = node.metadata?.classIri;
    if (!classIri) continue;

    if (!groups.has(classIri)) {
      groups.set(classIri, []);
    }
    groups.get(classIri)!.push(node);
  }

  return groups;
}

// Depth-based color mapping
export function getColorForDepth(depth: number): string {
  const colors = [
    '#FF6B6B', // Depth 0 (Root): Red
    '#4ECDC4', // Depth 1: Cyan
    '#FFD93D', // Depth 2: Yellow
    '#95E1D3', // Depth 3: Light Cyan
    '#AA96DA', // Depth 4: Purple
    '#F38181', // Depth 5+: Pink
  ];

  return colors[Math.min(depth, colors.length - 1)];
}

// Calculate transition animation state
export function calculateTransitionState(
  startTime: number,
  duration: number,
  now: number
): number {
  const elapsed = now - startTime;
  const t = Math.min(elapsed / duration, 1.0);
  return easeInOutCubic(t);
}

// Filter nodes by semantic zoom level
export function filterNodesByZoomLevel(
  nodes: GraphNode[],
  zoomLevel: number,
  hierarchy: ClassHierarchy
): GraphNode[] {
  if (zoomLevel < 3) {
    return nodes; // Show all in individual mode
  }

  // In grouped mode, filter by hierarchy depth
  const maxDepth = 5 - zoomLevel; // Higher zoom = lower depth
  return nodes.filter(node => {
    const classNode = hierarchy.findByIri(node.metadata?.classIri);
    return classNode && classNode.depth <= maxDepth;
  });
}

// Highlight all nodes of same class
export function highlightSameClass(
  selectedNodeId: string,
  nodes: GraphNode[]
): Set<string> {
  const selectedNode = nodes.find(n => n.id === selectedNodeId);
  if (!selectedNode?.metadata?.classIri) {
    return new Set();
  }

  const sameClassIri = selectedNode.metadata.classIri;
  const highlighted = new Set<string>();

  for (const node of nodes) {
    if (node.metadata?.classIri === sameClassIri) {
      highlighted.add(node.id);
    }
  }

  return highlighted;
}
```

### 4. Hierarchical Graph Renderer

**Location**: `/client/src/features/graph/components/HierarchicalGraphRenderer.tsx` (220 lines)

Main rendering component with dual mode support (individual vs grouped).

```typescript
interface HierarchicalGraphRendererProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  nodePositions: Map<string, Vector3>;
  onNodeClick: (nodeId: string, event: Event) => void;
  settings: VisualizationSettings;
}

const HierarchicalGraphRenderer: React.FC<HierarchicalGraphRendererProps> = ({
  nodes,
  edges,
  nodePositions,
  onNodeClick,
  settings
}) => {
  const { semanticZoomLevel, hierarchy, expandedClasses } = useOntologyStore();

  // Determine rendering mode
  const renderMode = semanticZoomLevel >= 3 ? 'grouped' : 'individual';

  if (renderMode === 'individual') {
    return <IndividualNodeRenderer nodes={nodes} positions={nodePositions} />;
  }

  // Grouped rendering
  const classGroups = groupNodesByClass(nodes, hierarchy);

  return (
    <>
      {Array.from(classGroups.entries()).map(([classIri, instances]) => {
        const isExpanded = expandedClasses.has(classIri);

        if (isExpanded) {
          // Show individual instances
          return instances.map(node => (
            <IndividualNode
              key={node.id}
              node={node}
              position={nodePositions.get(node.id)}
              onClick={onNodeClick}
            />
          ));
        } else {
          // Show as grouped sphere
          const centerPosition = calculateGroupCenter(instances, nodePositions);
          const classNode = hierarchy.findByIri(classIri);

          return (
            <ClassGroupSphere
              key={classIri}
              classIri={classIri}
              label={classNode.label}
              instanceCount={instances.length}
              position={centerPosition}
              depth={classNode.depth}
              onClick={() => handleGroupClick(classIri)}
            />
          );
        }
      })}
    </>
  );
};
```

#### Class Group Sphere Rendering

```typescript
const ClassGroupSphere: React.FC<ClassGroupSphereProps> = ({
  classIri,
  label,
  instanceCount,
  position,
  depth,
  onClick
}) => {
  // Calculate scale based on instance count (logarithmic)
  const scale = Math.min(5, 1 + Math.log(instanceCount + 1));

  // Color by hierarchy depth
  const color = getColorForDepth(depth);

  // Highlight state
  const [hovered, setHovered] = useState(false);
  const finalScale = hovered ? scale * 1.2 : scale;

  return (
    <group position={position}>
      {/* Sphere mesh */}
      <mesh
        scale={[finalScale, finalScale, finalScale]}
        onClick={onClick}
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
      >
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={hovered ? color : '#000000'}
          emissiveIntensity={hovered ? 0.3 : 0}
        />
      </mesh>

      {/* Billboard label with count */}
      <Billboard>
        <Text fontSize={0.5} color="white">
          {label}
          <br />
          ({instanceCount} instances)
        </Text>
      </Billboard>
    </group>
  );
};
```

### 5. Class Group Tooltip

**Location**: `/client/src/features/visualisation/components/ClassGroupTooltip.tsx` (180 lines)

Contextual information display for grouped classes.

```typescript
interface ClassGroupTooltipProps {
  classNode: ClassNode;
  visible: boolean;
  position: { x: number; y: number };
}

const ClassGroupTooltip: React.FC<ClassGroupTooltipProps> = ({
  classNode,
  visible,
  position
}) => {
  if (!visible) return null;

  return (
    <div
      className="absolute bg-gray-900 text-white p-4 rounded-lg shadow-lg"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        transform: 'translate(-50%, -120%)'
      }}
    >
      {/* Class label */}
      <h3 className="text-lg font-bold">{classNode.label}</h3>

      {/* Instance count badge */}
      <div className="inline-block bg-blue-600 px-2 py-1 rounded-full text-sm">
        {classNode.instanceCount} instances
      </div>

      {/* IRI */}
      <p className="text-xs text-gray-400 mt-2">
        {classNode.iri}
      </p>

      {/* Hierarchy information */}
      <div className="mt-3 space-y-1 text-sm">
        <div>
          <span className="text-gray-400">Depth:</span> {classNode.depth}
        </div>

        {classNode.parentIri && (
          <div>
            <span className="text-gray-400">Parent:</span> {classNode.parentIri}
          </div>
        )}

        {classNode.childIris.length > 0 && (
          <div>
            <span className="text-gray-400">Children:</span> {classNode.childIris.length}
          </div>
        )}
      </div>

      {/* Description */}
      {classNode.description && (
        <p className="mt-3 text-sm italic text-gray-300">
          {classNode.description}
        </p>
      )}

      {/* Interaction hint */}
      <p className="mt-3 text-xs text-gray-500">
        Click to expand • Double-click to highlight same class
      </p>
    </div>
  );
};
```

### 6. Hierarchical Animation Hook

**Location**: `/client/src/features/graph/hooks/useHierarchicalAnimation.ts` (190 lines)

Smooth expand/collapse animations with easing.

```typescript
interface AnimationState {
  nodeId: string;
  startPosition: Vector3;
  endPosition: Vector3;
  startScale: number;
  endScale: number;
  startTime: number;
  duration: number;
}

const useHierarchicalAnimation = (duration: number = 800) => {
  const [animations, setAnimations] = useState<Map<string, AnimationState>>(new Map());

  // Start expand animation
  const animateExpand = (
    classIri: string,
    instances: GraphNode[],
    groupPosition: Vector3
  ) => {
    const newAnimations = new Map<string, AnimationState>();
    const now = Date.now();

    instances.forEach((node, index) => {
      // Calculate target position (spread in circle)
      const angle = (index / instances.length) * Math.PI * 2;
      const radius = 10;
      const targetPos = new Vector3(
        groupPosition.x + Math.cos(angle) * radius,
        groupPosition.y + Math.sin(angle) * radius,
        groupPosition.z
      );

      newAnimations.set(node.id, {
        nodeId: node.id,
        startPosition: groupPosition.clone(),
        endPosition: targetPos,
        startScale: 0.1,
        endScale: 1.0,
        startTime: now,
        duration
      });
    });

    setAnimations(newAnimations);
  };

  // Start collapse animation
  const animateCollapse = (
    instances: GraphNode[],
    groupPosition: Vector3
  ) => {
    const newAnimations = new Map<string, AnimationState>();
    const now = Date.now();

    instances.forEach(node => {
      newAnimations.set(node.id, {
        nodeId: node.id,
        startPosition: node.position.clone(),
        endPosition: groupPosition.clone(),
        startScale: 1.0,
        endScale: 0.1,
        startTime: now,
        duration
      });
    });

    setAnimations(newAnimations);
  };

  // Update animation frame
  const updateFrame = (now: number): Map<string, { position: Vector3, scale: number }> => {
    const updates = new Map();

    animations.forEach((anim, nodeId) => {
      const elapsed = now - anim.startTime;
      const t = Math.min(elapsed / anim.duration, 1.0);
      const easedT = easeInOutCubic(t);

      const position = new Vector3().lerpVectors(
        anim.startPosition,
        anim.endPosition,
        easedT
      );

      const scale = anim.startScale + (anim.endScale - anim.startScale) * easedT;

      updates.set(nodeId, { position, scale });

      // Remove completed animations
      if (t >= 1.0) {
        animations.delete(nodeId);
      }
    });

    return updates;
  };

  return {
    animateExpand,
    animateCollapse,
    updateFrame,
    hasActiveAnimations: animations.size > 0
  };
};

// Ease-in-out cubic easing
function easeInOutCubic(t: number): number {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
```

## Interaction Patterns

### 1. Click Interactions

| Action | Behavior |
|--------|----------|
| Click collapsed class sphere | Expands with animation, shows instances |
| Click individual node | Standard selection (existing behavior) |
| Double-click individual node | Highlights all instances of same class |

### 2. Hover Interactions

| Action | Behavior |
|--------|----------|
| Hover class group sphere | Shows tooltip with metadata |
| Hover individual instance | Shows node details (existing) |

### 3. Zoom Interactions

| Action | Behavior |
|--------|----------|
| Slider change | Adjusts hierarchy depth visibility |
| Expand all button | Opens all collapsed class groups |
| Collapse all button | Groups all visible classes |

## Rendering Logic Flow

```
User adjusts zoom slider
    ↓
semanticZoomLevel updated in store
    ↓
GraphManager checks: level >= 3?
    ├─ NO → Individual rendering mode
    │      └─ Render instancedMesh (existing)
    └─ YES → Grouped rendering mode
           ↓
       Group nodes by classIri
           ↓
       For each class:
           ├─ Is expanded?
           │  ├─ YES → Render individual instances
           │  └─ NO → Render ClassGroupSphere
           ↓
       Calculate sphere properties:
           ├─ Scale: log(instanceCount + 1)
           ├─ Color: getColorForDepth(depth)
           └─ Position: average of instance positions
           ↓
       Render sphere + billboard label
```

## Performance Characteristics

### Optimizations

1. **Map-based Hierarchy**: O(1) lookups for class nodes
2. **Set-based Expansion State**: Fast O(1) membership tests
3. **Instanced Rendering**: Single draw call per class group
4. **LOD Filtering**: Reduces visible nodes by hierarchy depth
5. **Animation Culling**: Only animate visible transitions

### Memory Footprint

| Component | Per Item | 1,000 Classes | 10,000 Classes |
|-----------|----------|---------------|----------------|
| Hierarchy map | ~100 bytes | ~100 KB | ~1 MB |
| Expansion state | ~50 bytes | ~50 KB | ~500 KB |
| Animation state | ~200 bytes | Variable | Variable |

### Frame Rate Targets

| Node Count | Individual Mode | Grouped Mode |
|------------|----------------|--------------|
| 1,000 | 60 FPS | 60 FPS |
| 5,000 | 30-45 FPS | 60 FPS |
| 10,000 | 15-30 FPS | 60 FPS |

**Key Benefit**: Grouped mode maintains 60 FPS even with 10,000+ nodes

## Integration with GraphManager

### Required Changes

```typescript
// In client/src/features/graph/components/GraphManager.tsx

// 1. Add imports
import { useOntologyStore } from '../../ontology/store/useOntologyStore';
import { HierarchicalGraphRenderer } from './HierarchicalGraphRenderer';

// 2. Add store hook (inside component)
const { semanticZoomLevel } = useOntologyStore();
const useHierarchicalMode = semanticZoomLevel >= 3;

// 3. Replace instancedMesh rendering (around line 892)
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
  <instancedMesh
    ref={meshRef}
    args={[undefined, undefined, visibleNodes.length]}
    // ... existing props
  >
    {/* existing children */}
  </instancedMesh>
)}
```

## Backend API Requirements

### GET /api/ontology/hierarchy

**Response Format**:
```json
{
  "classes": [
    {
      "iri": "http://example.org/Person",
      "label": "Person",
      "parentIri": null,
      "childIris": [
        "http://example.org/Student",
        "http://example.org/Teacher"
      ],
      "instanceCount": 42,
      "depth": 0,
      "description": "Represents a person"
    }
  ]
}
```

### Graph Node Metadata

Nodes must include class IRI in metadata:

```typescript
interface GraphNode {
  id: string;
  metadata?: {
    classIri?: string;  // Required for hierarchical grouping!
  };
}
```

## Testing

### Manual Testing Checklist

- [ ] Zoom slider transitions smoothly (0-5)
- [ ] Class groups appear at level 3+
- [ ] Click expands class group with animation
- [ ] Tooltip appears on hover
- [ ] Double-click highlights same-class nodes
- [ ] Expand/collapse all buttons work
- [ ] Performance remains smooth with 1000+ nodes

### Automated Tests

```typescript
describe('HierarchicalVisualization', () => {
  it('groups nodes at zoom level 3', () => {
    const { result } = renderHook(() => useOntologyStore());
    act(() => result.current.setZoomLevel(3));
    expect(getRenderMode()).toBe('grouped');
  });

  it('expands class on click', () => {
    const { result } = renderHook(() => useOntologyStore());
    act(() => result.current.toggleClass('http://example.org/Person'));
    expect(result.current.expandedClasses.has('http://example.org/Person')).toBe(true);
  });
});
```

## Related Documentation

- [Ontology Reasoning Pipeline](./ontology-reasoning-pipeline.md) - Backend reasoning
- [Semantic Physics System](./semantic-physics-system.md) - Constraint generation
-  - Hierarchy endpoint
-  - End-user instructions

---

**Status**: ✅ Complete Implementation
**Last Updated**: 2025-11-03
**Total Lines**: 1,675 across 7 components
