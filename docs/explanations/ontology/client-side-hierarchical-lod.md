---
layout: default
title: "Client-Side Hierarchical LOD (Level of Detail)"
parent: Ontology
grand_parent: Explanations
nav_order: 99
---

# Client-Side Hierarchical LOD (Level of Detail)

**Date:** November 2, 2025
**Task:** Task 2.2 - Hierarchical Expansion/Collapse (Client-Side Only)
**Status:** Planning

## Architecture Decision

### Core Principle: Client-Side Filtering with Full Physics
- ✅ **ALL nodes transmitted to ALL clients** with complete spatial data
- ✅ **ALL nodes participate in physics simulation** at all times
- ✅ **Collapse/expand is per-client UI state** (not persisted server-side)
- ✅ **Higher initial data transmission** (acceptable trade-off for simplicity)
- ❌ **NO database schema changes needed**
- ❌ **NO server-side APIs for expansion state**

---

## Key Architecture Questions

### 1. How is Hierarchy Currently Determined?

**Node Structure (client/src/features/graph/managers/graphWorkerProxy.ts:10-19):**
```typescript
export interface Node {
  id: string;              // e.g., "pages/foo/bar.md"
  label: string;           // Display name
  position: { x, y, z };   // Spatial position (from physics)
  metadata?: Record<string, any>;  // Could contain hierarchy hints
}
```

**Potential Hierarchy Sources:**
1. **Node ID Path Structure** - `pages/foo/bar.md` implies parent `pages/foo/`
2. **Metadata Fields** - `metadata.parent`, `metadata.group`, `metadata.path`
3. **Edge Relationships** - Parent-child edges in the graph
4. **Spatial Clustering** - Nodes that cluster naturally in physics

**Question for Implementation:** Which method should we use to detect hierarchy?

---

### 2. Where Does Physics Stop and Visual Tidying Begin?

**Two Potential Approaches:**

#### Approach A: Hierarchical Physics (Nested Positions)
- **Physics Level 1:** Root nodes participate in force simulation
- **Physics Level 2:** Child nodes have positions *relative to parent*
- **Visual Rendering:** Client chooses whether to render children
- **Benefit:** Natural clustering of children around parents
- **Challenge:** Need to define parent-child relationships in physics

#### Approach B: Flat Physics with Visual Grouping
- **Physics:** ALL nodes participate equally in global force simulation
- **Visual Rendering:** Client applies LOD filter based on:
  - Hierarchy depth detection
  - Camera distance (true LOD)
  - User expansion preferences
- **Benefit:** Simpler physics, no hierarchy in simulation
- **Challenge:** Collapsed nodes still render unless filtered

**Question for Implementation:** Which physics model should we use?

---

## Proposed Implementation (Client-Side Only)

### Phase 1: Hierarchy Detection

**File:** `client/src/features/graph/utils/hierarchyDetector.ts` (new)

```typescript
export interface HierarchyNode extends Node {
  parentId?: string;
  depth: number;
  childIds: string[];
}

export function detectHierarchy(nodes: Node[]): Map<string, HierarchyNode> {
  const hierarchyMap = new Map<string, HierarchyNode>();

  // Option 1: Detect from ID path structure
  nodes.forEach(node => {
    const pathParts = node.id.split('/');
    const parentPath = pathParts.slice(0, -1).join('/');
    const parentId = parentPath || undefined;
    const depth = pathParts.length - 1;

    hierarchyMap.set(node.id, {
      ...node,
      parentId,
      depth,
      childIds: []
    });
  });

  // Build parent-child relationships
  hierarchyMap.forEach((node, id) => {
    if (node.parentId) {
      const parent = hierarchyMap.get(node.parentId);
      if (parent) {
        parent.childIds.push(id);
      }
    }
  });

  return hierarchyMap;
}
```

---

### Phase 2: Client-Side Expansion State

**File:** `client/src/features/graph/hooks/useExpansionState.ts` (new)

```typescript
import { useState, useCallback } from 'react';

export interface ExpansionState {
  expandedNodes: Set<string>;
  toggleExpansion: (nodeId: string) => void;
  isExpanded: (nodeId: string) => boolean;
  isVisible: (nodeId: string, parentId?: string) => boolean;
}

export function useExpansionState(defaultExpanded: boolean = true): ExpansionState {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  const toggleExpansion = useCallback((nodeId: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);

  const isExpanded = useCallback((nodeId: string) => {
    // Default to expanded if not explicitly collapsed
    return defaultExpanded ? !expandedNodes.has(nodeId) : expandedNodes.has(nodeId);
  }, [expandedNodes, defaultExpanded]);

  const isVisible = useCallback((nodeId: string, parentId?: string) => {
    // Root nodes (no parent) always visible
    if (!parentId) return true;

    // Child nodes visible only if parent is expanded
    return isExpanded(parentId);
  }, [isExpanded]);

  return { expandedNodes, toggleExpansion, isExpanded, isVisible };
}
```

---

### Phase 3: LOD Rendering Filter

**File:** `client/src/features/graph/components/GraphManager.tsx`

**Key Changes:**

```typescript
import { useExpansionState } from '../hooks/useExpansionState';
import { detectHierarchy } from '../utils/hierarchyDetector';

// Inside GraphManager component:
const { expandedNodes, toggleExpansion, isVisible } = useExpansionState(true);

// Build hierarchy map
const hierarchyMap = useMemo(() => {
  return detectHierarchy(graphData.nodes);
}, [graphData.nodes]);

// Filter visible nodes for RENDERING ONLY
const visibleNodes = useMemo(() => {
  return graphData.nodes.filter(node => {
    const hierarchyNode = hierarchyMap.get(node.id);
    return isVisible(node.id, hierarchyNode?.parentId);
  });
}, [graphData.nodes, hierarchyMap, isVisible]);

// CRITICAL: Physics uses ALL nodes
// Rendering uses ONLY visibleNodes

// In useFrame (physics loop):
updatePhysics(graphData.nodes); // ALL nodes

// In rendering:
<instancedMesh
  ref={meshRef}
  args={[geometry, material, visibleNodes.length]} // FILTERED
  onClick={(e) => {
    const nodeId = getNodeIdFromInstance(e.instanceId);
    toggleExpansion(nodeId);
  }}
/>
```

---

## Physics vs Visual Boundary Decision

**Recommended Approach:**

1. **Physics Layer:** ALL nodes participate in global force-directed layout
   - Uses: `graphData.nodes` (complete dataset)
   - No hierarchy-aware forces (keeps it simple)
   - All nodes have real spatial positions

2. **Visual Layer:** Client filters based on expansion state
   - Uses: `visibleNodes` (filtered subset)
   - `InstancedMesh` count = `visibleNodes.length`
   - Hidden nodes still in memory, just not rendered

3. **Optional Enhancement:** Add "collapse radius" for visual clustering
   - When parent collapsed, position children at parent's position
   - When parent expanded, interpolate children to their physics positions
   - Smooth 1000ms animation

---

## Implementation Steps (Priority Order)

1. ✅ **Revert Database Changes** (expanded, parent-id columns not needed)
2. ⏭️ **Create Hierarchy Detector** (`utils/hierarchyDetector.ts`)
3. ⏭️ **Create Expansion Hook** (`hooks/useExpansionState.ts`)
4. ⏭️ **Modify GraphManager** to use LOD filtering
5. ⏭️ **Add Click Handler** for node expansion toggle
6. ⏭️ **Test with Real Data** (50+ nodes, verify physics continues)
7. ⏭️ **Optional: Add Collapse Animation** (interpolate positions)

---

## Success Criteria

| Criteria | Status |
|----------|--------|
| All nodes transmitted to clients | ✅ Default |
| All nodes participate in physics | 🔄 Verify |
| Collapse/expand is client-side only | 🔄 Implement |
| No database schema changes | 🔄 Revert |
| Rendering filters to visible nodes | 🔄 Implement |
| Click toggles expansion per-client | 🔄 Implement |
| Performance: 100+ nodes @ 60 FPS | 🔄 Test |

---

## Next Actions

1. **FIRST:** Revert `src/models/node.rs` and `src/repositories/unified-graph-repository.rs`
2. **THEN:** Implement client-side hierarchy detection
3. **THEN:** Implement LOD rendering filter in GraphManager
4. **FINALLY:** Test with multiple clients (verify independent expansion state)

---

**Last Updated:** November 2, 2025
**Architecture:** Client-side LOD, server-agnostic, full data transmission
