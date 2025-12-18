---
title: Screenshot Capture TODOs
description: Required UI screenshots to complete documentation assets
category: explanation
tags:
  - rest
  - websocket
  - neo4j
updated-date: 2025-12-18
difficulty-level: intermediate
priority: medium
date: 2025-12-02
---


# Screenshot Capture TODOs

**Purpose**: Document UI screenshots needed to complete documentation assets after asset restoration analysis.
**Status**: Pending - requires running application
**Target**: 4 screenshots for user-facing documentation

---

## Prerequisites

Before capturing screenshots:
1. ✅ Start VisionFlow server: `cargo run`
2. ✅ Start client development server: `cd client && npm run dev`
3. ✅ Load sample data or connect to Neo4j database
4. ✅ Ensure browser dev tools are available (for WebSocket inspector)

---

## Screenshot TODO List

### 1. Control Center UI 🔴 High Priority
**File**: `docs/assets/screenshots/control-center-dashboard.png`
**Dimensions**: 1920x1080 (full screen) or 1440x900 (minimum)
**Description**: Main control center dashboard showing panel layout and controls

**Capture Requirements**:
- Open control center (toggle button or default view)
- Show all visible panels:
  - Analytics panel
  - Graph settings panel
  - Filtering controls
  - System health indicator
- Ensure clean UI state (no errors, connected status)

**Usage Locations**:
- `docs/README.md` - Project overview
- `docs/guides/user-guide.md` - User interface section
- `client/README.md` - Client features

**Markdown Reference**:
```markdown
![Control Center Dashboard](../assets/screenshots/control-center-dashboard.png)

*Figure 1: VisionFlow Control Center showing analytics, settings, and filtering panels*
```

---

### 2. Graph Visualization 🟡 Medium Priority
**File**: `docs/assets/screenshots/graph-3d-visualization.png`
**Dimensions**: 1920x1080 (full screen) or 1440x900 (minimum)
**Description**: 3D force-directed graph with nodes and edges rendered

**Capture Requirements**:
- Load sample ontology or knowledge graph data
- Show 3D visualization with:
  - Multiple nodes (50-200 for visual impact)
  - Visible edges connecting nodes
  - Node labels (if visible at zoom level)
  - Camera positioned for good perspective
- Capture during animation (smooth force-directed layout)
- Include control panel in frame (if space permits)

**Usage Locations**:
- `docs/README.md` - Main project showcase
- `docs/ARCHITECTURE_OVERVIEW.md` - Client layer description
- `docs/guides/visualization-guide.md` - Graph rendering section

**Markdown Reference**:
```markdown
![3D Graph Visualization](../assets/screenshots/graph-3d-visualization.png)

*Figure 2: Force-directed 3D graph rendering with 100+ nodes and dynamic physics*
```

---

### 3. Settings Panel 🟡 Medium Priority
**File**: `docs/assets/screenshots/settings-unified-panel.png`
**Dimensions**: 1280x720 (panel focused) or full screen
**Description**: Unified settings tab with filters and controls expanded

**Capture Requirements**:
- Open control center
- Navigate to Settings tab
- Show:
  - Filter threshold sliders
  - Advanced mode toggle
  - Graph rendering options
  - Physics engine controls
- Expand any collapsible sections
- Capture in clean state (no validation errors)

**Usage Locations**:
- `docs/guides/configuration.md` - Settings configuration
- `docs/features/client-side-filtering.md` - Filter controls
- `docs/user-settings-implementation-summary.md` - Settings UI

**Markdown Reference**:
```markdown
![Settings Panel](../assets/screenshots/settings-unified-panel.png)

*Figure 3: Unified settings panel with filter controls and graph configuration options*
```

---

### 4. WebSocket Protocol Inspector 🟢 Low Priority
**File**: `docs/assets/screenshots/websocket-inspector.png`
**Dimensions**: 1920x1080 (full screen) or 1440x900 (minimum)
**Description**: Browser dev tools showing WebSocket messages in Network tab

**Capture Requirements**:
- Open browser DevTools (F12)
- Navigate to Network tab
- Filter to WS (WebSocket)
- Show:
  - Active WebSocket connection to `ws://localhost:4000/ws`
  - Message list with `filter_update`, `graph_update`, etc.
  - Sample message payload (expanded)
  - Binary/JSON message indicator
- Capture during active graph interaction (so messages are flowing)

**Usage Locations**:
- `docs/reference/websocket-protocol.md` - Protocol debugging section
- `docs/guides/developer/05-debugging.md` - WebSocket debugging
- `docs/ARCHITECTURE_OVERVIEW.md` - Binary protocol description

**Markdown Reference**:
```markdown
![WebSocket Inspector](../assets/screenshots/websocket-inspector.png)

*Figure 4: Browser DevTools showing WebSocket protocol messages during graph updates*
```

---

## Screenshot Capture Process

### Recommended Tools
- **Linux**: `gnome-screenshot`, `flameshot`, `shutter`
- **macOS**: `Cmd+Shift+4` (native), `CleanShot X`
- **Windows**: `Win+Shift+S` (Snipping Tool), `Greenshot`

### Capture Steps
1. Launch application and ensure it's in desired state
2. Use screenshot tool to capture area or full screen
3. Save to `/home/devuser/workspace/project/docs/assets/screenshots/`
4. Use descriptive kebab-case filenames
5. Verify image quality and clarity
6. Update documentation with markdown image references

### Image Optimization
```bash
# Optional: Optimize PNG files to reduce size
cd /home/devuser/workspace/project/docs/assets/screenshots/
pngquant --quality=80-95 --ext .png --force *.png

# Or use ImageMagick
mogrify -quality 85 -resize 1920x1080\> *.png
```

---

## Documentation Update Checklist

After capturing screenshots:

### Primary Documentation
- [ ] `docs/README.md` - Add control center and graph visualization
- [ ] `docs/ARCHITECTURE_OVERVIEW.md` - Add client layer screenshot
- [ ] `client/README.md` - Add client features screenshot

### User Guides
- [ ] `docs/guides/user-guide.md` - Add control center UI
- [ ] `docs/guides/configuration.md` - Add settings panel
- [ ] `docs/guides/visualization-guide.md` - Add graph visualization

### Reference Documentation
- [ ] `docs/reference/websocket-protocol.md` - Add WebSocket inspector
- [ ] `docs/guides/developer/05-debugging.md` - Add debugging screenshots

### Feature Documentation
- [ ] `docs/features/client-side-filtering.md` - Add filter UI screenshot
- [ ] `docs/user-settings-implementation-summary.md` - Add settings screenshot

---

## Quality Standards

### Image Requirements
- **Resolution**: Minimum 1280x720, recommended 1920x1080
- **Format**: PNG (lossless) preferred, JPG acceptable for photos
- **File Size**: <2MB per image (optimize if larger)
- **Naming**: Descriptive kebab-case: `control-center-dashboard.png`

### Content Requirements
- Clean UI state (no errors, loading spinners, or debug artifacts)
- Representative data (not empty or placeholder content)
- Clear visibility of relevant UI elements
- Professional appearance (proper zoom, alignment, lighting)

### Markdown Requirements
- Always include alt text: `![Control Center](path/to/image.png)`
- Add caption below: `*Figure N: Description of screenshot*`
- Use relative paths: `../assets/screenshots/filename.png`
- Test links work from document location

---

## Progress Tracking

| Screenshot | Priority | Status | File | Captured By | Date |
|-----------|----------|--------|------|-------------|------|
| Control Center | 🔴 High | 🔲 TODO | `control-center-dashboard.png` | - | - |
| Graph Visualization | 🟡 Medium | 🔲 TODO | `graph-3d-visualization.png` | - | - |
| Settings Panel | 🟡 Medium | 🔲 TODO | `settings-unified-panel.png` | - | - |
| WebSocket Inspector | 🟢 Low | 🔲 TODO | `websocket-inspector.png` | - | - |

**Legend**:
- 🔴 High Priority - User-facing documentation, README visibility
- 🟡 Medium Priority - Feature documentation, guides
- 🟢 Low Priority - Developer/reference documentation
- 🔲 TODO - Not yet captured
- ✅ DONE - Captured and integrated

---

## Related Documents

- `/docs/working/ASSET_RESTORATION.md` - Asset restoration analysis (parent document)
- `/docs/archive/reports/link-audit-summary.md` - Original broken link audit
- `/docs/assets/diagrams/sparc-turboflow-architecture.md` - Mermaid diagram examples

---

**Created**: 2025-12-02
**Last Updated**: 2025-12-02
**Status**: Awaiting application runtime for screenshot capture
