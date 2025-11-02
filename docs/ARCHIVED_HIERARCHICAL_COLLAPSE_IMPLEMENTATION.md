# Hierarchical Node Expansion/Collapse - Implementation Progress

**Date:** November 2, 2025
**Task:** Task 2.2 - Hierarchical Expansion/Collapse UI
**Status:** In Progress (Backend foundation complete, Frontend pending)

## Architecture Design

### Core Principle: Physics-Driven LOD (Level of Detail)
**ALL nodes participate in physics simulation regardless of expansion state.**
**ONLY visible nodes (expanded or root) are rendered visually.**

This approach ensures:
- Consistent physics behavior across the entire graph
- Smooth animations when expanding/collapsing
- Optimal rendering performance (fewer draw calls)
- Natural clustering of collapsed hierarchies

---

## ✅ Completed Changes

### 1. Database Schema Enhancement (`src/repositories/unified_graph_repository.rs`)

**Added columns to `graph_nodes` table:**
```sql
expanded BOOLEAN DEFAULT 1,           -- True = children visible, False = children hidden
parent_id INTEGER,                    -- Parent node ID for hierarchy
FOREIGN KEY (parent_id) REFERENCES graph_nodes(id) ON DELETE SET NULL
```

**Location:** `src/repositories/unified_graph_repository.rs:92-121`

**Key Features:**
- `expanded` defaults to `true` (all nodes start expanded)
- `parent_id` establishes parent-child relationships
- Foreign key constraint with `SET NULL` on parent deletion (prevents orphans)

---

### 2. Rust Node Model Extension (`src/models/node.rs`)

**Added fields to Node struct:**
```rust
// Hierarchical expansion/collapse
#[serde(skip_serializing_if = "Option::is_none")]
pub expanded: Option<bool>,
#[serde(skip_serializing_if = "Option::is_none")]
pub parent_id: Option<u32>,
```

**Location:** `src/models/node.rs:60-64`

**Added helper methods:**
```rust
pub fn with_expanded(mut self, expanded: bool) -> Self {
    self.expanded = Some(expanded);
    self
}

pub fn with_parent_id(mut self, parent_id: u32) -> Self {
    self.parent_id = Some(parent_id);
    self
}
```

**Location:** `src/models/node.rs:130-137`

**Updated constructors:**
- `new()`, `new_with_id()`, and `new_with_stored_id()` all default `expanded` to `Some(true)`
- Locations: Lines 125-127, 254, 270-271

---

## 🔄 In Progress / Pending

### 3. Backend API Endpoint (PENDING)

**Required:** `POST /api/graph/nodes/:nodeId/expand`

**Request Body:**
```json
{
  "expanded": true  // or false to collapse
}
```

**Response:**
```json
{
  "success": true,
  "node_id": 42,
  "expanded": true
}
```

**Implementation Location:** `src/handlers/api_handler/graph/mod.rs`

**Database Query Required:**
```sql
UPDATE graph_nodes
SET expanded = ?expanded, updated_at = CURRENT_TIMESTAMP
WHERE id = ?node_id
```

---

### 4. TypeScript Graph Node Interface (PENDING)

**File:** `client/src/features/graph/managers/graphDataManager.ts`

**Add fields to Node interface:**
```typescript
export interface Node {
  id: number;
  metadata_id: string;
  label: string;
  position?: { x: number; y: number; z: number };
  velocity?: { x: number; y: number; z: number };
  metadata?: Record<string, string>;
  type?: string;
  size?: number;
  color?: string;
  weight?: number;
  group?: string;

  // NEW: Hierarchical expansion
  expanded?: boolean;
  parent_id?: number;
}
```

---

### 5. GraphManager LOD Rendering Filter (PENDING)

**File:** `client/src/features/graph/components/GraphManager.tsx`

**Rendering Logic:**

```typescript
// In GraphManager.tsx useEffect or useMemo
const visibleNodes = useMemo(() => {
  // Build parent expansion map
  const expansionMap = new Map<number, boolean>();
  graphData.nodes.forEach(node => {
    expansionMap.set(node.id, node.expanded ?? true);
  });

  // Filter nodes: Show root nodes + children of expanded parents
  return graphData.nodes.filter(node => {
    // Root nodes (no parent) always visible
    if (!node.parent_id) return true;

    // Child nodes visible only if parent is expanded
    return expansionMap.get(node.parent_id) === true;
  });
}, [graphData.nodes]);

// Use visibleNodes for rendering InstancedMesh
// ALL nodes still participate in physics (graphData.nodes)
// ONLY visibleNodes are rendered visually
```

**Key Implementation Points:**
- Keep physics simulation using `graphData.nodes` (ALL nodes)
- Render visuals using `visibleNodes` (filtered subset)
- Update InstancedMesh matrix count based on `visibleNodes.length`

**Location to modify:** Lines 175-180 (graphData state) and rendering loop (lines 400+)

---

### 6. Click Handler for Expansion Toggle (PENDING)

**File:** `client/src/features/graph/components/GraphManager.tsx`

**Implementation:**

```typescript
const handleNodeClick = useCallback(async (event: ThreeEvent<MouseEvent>, nodeId: number) => {
  if (!nodeId) return;

  // Find the clicked node
  const node = graphData.nodes.find(n => n.id === nodeId);
  if (!node) return;

  // Toggle expansion state
  const newExpanded = !(node.expanded ?? true);

  try {
    // Call backend API to persist state
    const response = await fetch(`/api/graph/nodes/${nodeId}/expand`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expanded: newExpanded }),
    });

    if (response.ok) {
      // Update local state optimistically
      setGraphData(prev => ({
        ...prev,
        nodes: prev.nodes.map(n =>
          n.id === nodeId ? { ...n, expanded: newExpanded } : n
        ),
      }));
    }
  } catch (error) {
    console.error('Failed to toggle node expansion:', error);
  }
}, [graphData.nodes]);
```

**Attach to mesh click event:**
```typescript
<instancedMesh
  ref={meshRef}
  args={[geometry, material, visibleNodes.length]}
  onClick={handleNodeClick}
/>
```

**Location to add:** Around line 195 (dragState definition)

---

### 7. Expansion Animation (PENDING)

**File:** `client/src/features/graph/components/GraphManager.tsx`

**Implementation:**

```typescript
// Animation state for expansion
const [expansionAnimation, setExpansionAnimation] = useState<{
  nodeId: number;
  startTime: number;
  fromScale: number;
  toScale: number;
} | null>(null);

// In useFrame hook (animation loop)
useFrame((state, delta) => {
  // ... existing physics updates ...

  // Handle expansion animation
  if (expansionAnimation) {
    const elapsed = Date.now() - expansionAnimation.startTime;
    const duration = 1000; // 1000ms animation

    if (elapsed >= duration) {
      setExpansionAnimation(null); // Animation complete
    } else {
      const progress = elapsed / duration;
      const eased = 1 - Math.pow(1 - progress, 3); // Cubic ease-out

      // Update scale for newly visible nodes
      // (Apply to children of expanded parent)
    }
  }
});
```

**Animation Strategy:**
- When parent expands: Children "emerge" from parent position over 1000ms
- Use cubic easing for smooth motion
- Interpolate from parent position → final position
- Interpolate scale from 0.0 → 1.0

**Location to add:** In useFrame hook around line 220+

---

## 📋 Remaining Tasks (Prioritized)

1. **Backend API Endpoint** (High Priority)
   - Create `POST /api/graph/nodes/:nodeId/expand` in `graph/mod.rs`
   - Add database UPDATE query in `UnifiedGraphRepository`
   - Test endpoint with curl/Postman

2. **TypeScript Interface Update** (Medium Priority)
   - Add `expanded` and `parent_id` to Node interface
   - Update graphDataManager to handle new fields
   - Ensure serialization/deserialization works

3. **LOD Rendering Filter** (High Priority)
   - Implement `visibleNodes` filtering logic
   - Keep physics using ALL nodes
   - Render only visible nodes
   - Test performance with large graphs

4. **Click Handler** (Medium Priority)
   - Add onClick event to InstancedMesh
   - Implement node ID detection from instance
   - Call backend API on click
   - Update local state optimistically

5. **Expansion Animation** (Low Priority - Nice to Have)
   - Add animation state management
   - Implement easing function
   - Interpolate position/scale during expansion
   - Polish timing and feel

6. **End-to-End Testing** (High Priority)
   - Create test hierarchy (parent → children)
   - Test expand/collapse behavior
   - Verify physics continues for collapsed nodes
   - Test with 100+ nodes
   - Performance profiling

---

## 🎯 Success Criteria

| Criteria | Status |
|----------|--------|
| Database schema supports hierarchy | ✅ Complete |
| Node model includes expansion state | ✅ Complete |
| API endpoint toggles expansion | 🔄 Pending |
| Frontend filters rendering by expansion | 🔄 Pending |
| ALL nodes participate in physics | 🔄 Pending (validation needed) |
| ONLY visible nodes render | 🔄 Pending |
| Click toggles expansion | 🔄 Pending |
| Smooth 1000ms animation | 🔄 Pending (optional) |
| End-to-end test passes | ❌ Not started |

---

## 🔧 Technical Architecture Summary

```
User Click
   ↓
GraphManager.handleNodeClick()
   ↓
POST /api/graph/nodes/:nodeId/expand
   ↓
UnifiedGraphRepository.update_node_expansion()
   ↓
Database: UPDATE graph_nodes SET expanded = ?
   ↓
Response: { success: true, expanded: true }
   ↓
GraphManager: Update local graphData state
   ↓
useMemo: Recalculate visibleNodes (filtered)
   ↓
Rendering: InstancedMesh with visibleNodes.length
   ↓
Physics: Continue using ALL graphData.nodes
   ↓
Animation: Smooth emergence over 1000ms (optional)
```

---

## 📝 Notes

- **Database Migration:** If unified.db already exists, it will need to be recreated or migrated to include the new columns
- **Default Behavior:** All nodes default to `expanded = true` (fully visible hierarchy)
- **Physics Continuity:** Collapsed children still participate in physics, ensuring natural clustering
- **Performance:** LOD rendering reduces draw calls significantly for large hierarchies
- **Future Enhancement:** Could add "depth" parameter to collapse multiple levels at once

---

**Last Updated:** November 2, 2025
**Next Action:** Implement backend API endpoint for expansion toggle
