---
name: "FossFLOW Diagram Generator"
description: "Create and validate isometric network diagrams for FossFLOW visualization. Generates JSON diagram files with nodes, connectors, icons, and layouts. Use when building system architecture diagrams, deployment flows, network topologies, or any visual representation suitable for FossFLOW. Includes screenshot verification via virtual display."
---

# FossFLOW Diagram Generator

Generate production-ready isometric diagrams for the FossFLOW visualization tool with automatic screenshot verification.

## Quick Start

Generate a FossFLOW diagram:

```bash
# Create diagram JSON
claude "Create a FossFLOW diagram showing a microservices architecture with API gateway, 3 services, and database"
```

## FossFLOW JSON Schema

### Minimal Diagram Structure

```json
{
  "title": "My Diagram",
  "items": [
    {
      "id": "item-1",
      "name": "Web Client",
      "icon": "isoflow__person",
      "position": { "x": 0, "y": 0 }
    },
    {
      "id": "item-2",
      "name": "API Server",
      "icon": "isoflow__api",
      "position": { "x": 2, "y": 1 }
    }
  ],
  "connectors": [
    {
      "id": "conn-1",
      "from": "item-1",
      "to": "item-2",
      "color": "color-1"
    }
  ],
  "colors": [
    { "id": "color-1", "value": "#4A90D9" }
  ],
  "icons": []
}
```

### Complete Schema Reference

```typescript
interface FossFLOWDiagram {
  title: string;
  version?: string;
  description?: string;
  fitToScreen?: boolean;

  items: Array<{
    id: string;                    // Unique identifier
    name: string;                  // Display label
    description?: string;          // Tooltip text
    icon: string;                  // Icon reference (see Available Icons)
    position: { x: number; y: number };  // Tile coordinates
  }>;

  connectors: Array<{
    id: string;
    from: string;                  // Source item ID
    to: string;                    // Target item ID
    name?: string;                 // Legacy label (use labels[] instead)
    color: string;                 // Color ID reference
    customColor?: string;          // RGB hex override
    width?: number;                // Line thickness (1-5)
    style?: 'SOLID' | 'DOTTED' | 'DASHED';
    lineType?: 'SINGLE' | 'DOUBLE' | 'DOUBLE_WITH_CIRCLE';
    showArrow?: boolean;
    labels?: Array<{
      id: string;
      text: string;
      position: number;            // 0-100 (% along path)
      height?: number;
      line?: '1' | '2';
      showLine?: boolean;
    }>;
    anchors?: Array<{
      id: string;
      ref: { item: string; anchor: string; tile: { x: number; y: number } };
    }>;
  }>;

  colors: Array<{
    id: string;
    value: string;                 // Hex color (#RRGGBB)
  }>;

  icons: Array<{
    id: string;
    name: string;
    url: string;                   // SVG/PNG URL or data URI
    isIsometric: boolean;
    collection?: 'isoflow' | 'imported';
  }>;

  rectangles?: Array<{
    id: string;
    from: { x: number; y: number };
    to: { x: number; y: number };
    color: string;
  }>;

  textBoxes?: Array<{
    id: string;
    text: string;
    position: { x: number; y: number };
  }>;

  views?: Array<{
    id: string;
    name: string;
    items: string[];
    connectors: string[];
    rectangles?: string[];
    textBoxes?: string[];
  }>;
}
```

## Available Icons

### Infrastructure
- `isoflow__server` - Generic server
- `isoflow__database` - Database/storage
- `isoflow__redis` - Redis cache
- `isoflow__backup` - Backup storage

### Services
- `isoflow__api` - API endpoint
- `isoflow__microservice` - Microservice
- `isoflow__authentication` - Auth service
- `isoflow__gateway` - API gateway

### Network
- `isoflow__load_balancer` - Load balancer
- `isoflow__cdn` - CDN node
- `isoflow__queue` - Message queue
- `isoflow__logs` - Logging service

### Users/Devices
- `isoflow__person` - User/person
- `isoflow__mobile` - Mobile device
- `isoflow__web_app` - Web application

### Monitoring
- `isoflow__monitoring` - Monitoring service
- `isoflow__analytics` - Analytics
- `isoflow__notification` - Notifications
- `isoflow__shield` - Security/firewall

## Layout Guidelines

### Tile Positioning
- FossFLOW uses isometric tile coordinates (x, y)
- Each tile is ~100px unprojected
- Positive X moves right-down, positive Y moves left-down
- Keep 1-2 tile spacing between connected items

### Recommended Layouts

**Horizontal Flow (Left to Right)**:
```
User (0,0) → Gateway (2,1) → Service (4,2) → Database (6,3)
```

**Layered Architecture**:
```
              Frontend (2,0)
                   ↓
API Layer: Gateway (1,2), Auth (3,2)
                   ↓
Services:  Svc1 (0,4), Svc2 (2,4), Svc3 (4,4)
                   ↓
Data:      DB (1,6), Cache (3,6)
```

**Hub and Spoke**:
```
            Service1 (0,0)
                 ↑
Service2 (2,2) ← Hub (2,2) → Service3 (4,0)
                 ↓
            Service4 (2,4)
```

## Color Palette

### Recommended Colors
```json
{
  "colors": [
    { "id": "blue", "value": "#4A90D9" },
    { "id": "green", "value": "#7CB342" },
    { "id": "orange", "value": "#FF9800" },
    { "id": "red", "value": "#E53935" },
    { "id": "purple", "value": "#9C27B0" },
    { "id": "gray", "value": "#78909C" },
    { "id": "teal", "value": "#00ACC1" }
  ]
}
```

## Sub-Skills

### Screenshot Verification

After generating a diagram, verify it renders correctly:

```bash
# Start FossFLOW dev server
cd /home/devuser/workspace/FossFLOW/packages/fossflow-app
npm run dev &

# Take screenshot via virtual display
# Uses playwright skill or VNC screenshot
```

**Verification Steps**:
1. Save diagram JSON to FossFLOW data directory
2. Open in browser via virtual display (VNC :1 or playwright)
3. Capture screenshot
4. Analyze for rendering issues

### Playwright Integration

Use the playwright skill for browser automation:

```markdown
/playwright

Navigate to http://localhost:3000
Load diagram from JSON
Take full-page screenshot
Verify all nodes rendered
Check connector paths visible
```

### VNC Screenshot

For virtual display screenshots:

```bash
# Capture VNC display
DISPLAY=:1 import -window root /tmp/fossflow-screenshot.png

# Or use scrot
DISPLAY=:1 scrot /tmp/fossflow-screenshot.png
```

## Workflow: Generate and Verify

### Step 1: Generate Diagram JSON

```javascript
// Example: Microservices Architecture
const diagram = {
  title: "Microservices Architecture",
  items: [
    { id: "user", name: "User", icon: "isoflow__person", position: { x: 0, y: 0 } },
    { id: "gateway", name: "API Gateway", icon: "isoflow__gateway", position: { x: 2, y: 1 } },
    { id: "auth", name: "Auth Service", icon: "isoflow__authentication", position: { x: 4, y: 0 } },
    { id: "users-svc", name: "Users API", icon: "isoflow__api", position: { x: 4, y: 2 } },
    { id: "orders-svc", name: "Orders API", icon: "isoflow__microservice", position: { x: 6, y: 1 } },
    { id: "db", name: "PostgreSQL", icon: "isoflow__database", position: { x: 8, y: 2 } },
    { id: "cache", name: "Redis", icon: "isoflow__redis", position: { x: 8, y: 0 } }
  ],
  connectors: [
    { id: "c1", from: "user", to: "gateway", color: "blue", showArrow: true },
    { id: "c2", from: "gateway", to: "auth", color: "green", showArrow: true },
    { id: "c3", from: "gateway", to: "users-svc", color: "blue", showArrow: true },
    { id: "c4", from: "gateway", to: "orders-svc", color: "blue", showArrow: true },
    { id: "c5", from: "users-svc", to: "db", color: "orange", showArrow: true },
    { id: "c6", from: "orders-svc", to: "db", color: "orange", showArrow: true },
    { id: "c7", from: "orders-svc", to: "cache", color: "purple", style: "DASHED" }
  ],
  colors: [
    { id: "blue", value: "#4A90D9" },
    { id: "green", value: "#7CB342" },
    { id: "orange", value: "#FF9800" },
    { id: "purple", value: "#9C27B0" }
  ],
  icons: []
};
```

### Step 2: Save to FossFLOW

```bash
# Save diagram JSON
cat > /home/devuser/workspace/FossFLOW/packages/fossflow-app/public/diagrams/my-diagram.json << 'EOF'
{
  "title": "My Architecture",
  ...
}
EOF
```

### Step 3: Verify with Screenshot

```bash
# Option A: Playwright (preferred)
/playwright
# Navigate to localhost:3000, import diagram, screenshot

# Option B: VNC capture
DISPLAY=:1 scrot /tmp/diagram-verify.png
```

## Advanced Patterns

### Multi-Label Connectors

```json
{
  "id": "conn-1",
  "from": "api",
  "to": "db",
  "color": "blue",
  "labels": [
    { "id": "l1", "text": "Query", "position": 25 },
    { "id": "l2", "text": "Result", "position": 75 }
  ]
}
```

### Grouped Sections with Rectangles

```json
{
  "rectangles": [
    {
      "id": "backend-zone",
      "from": { "x": 3, "y": 0 },
      "to": { "x": 8, "y": 4 },
      "color": "gray"
    }
  ]
}
```

### Text Annotations

```json
{
  "textBoxes": [
    {
      "id": "note-1",
      "text": "Production Environment",
      "position": { "x": 5, "y": -1 }
    }
  ]
}
```

## Validation Checklist

Before finalizing a FossFLOW diagram:

- [ ] All items have unique IDs
- [ ] All connector `from`/`to` reference valid item IDs
- [ ] All connector colors reference defined color IDs
- [ ] Icon names use valid `isoflow__*` identifiers
- [ ] Positions avoid overlapping (minimum 1 tile spacing)
- [ ] JSON is valid and parseable
- [ ] Diagram renders without errors in FossFLOW
- [ ] Screenshot verification passed

## Integration with Claude-Flow

Use claude-flow swarm for complex diagram generation:

```bash
# Initialize swarm for diagram generation
npx claude-flow@alpha swarm init --topology mesh

# Spawn diagram specialists
npx claude-flow@alpha agent spawn --type architect --name "Diagram Designer"
npx claude-flow@alpha agent spawn --type coder --name "JSON Generator"
npx claude-flow@alpha agent spawn --type reviewer --name "Layout Validator"

# Orchestrate diagram creation
npx claude-flow@alpha task orchestrate "Create FossFLOW diagram for [description]"
```

## Troubleshooting

### Common Issues

**Connectors not visible**: Ensure `from`/`to` reference existing item IDs

**Icons not rendering**: Use exact icon names from Available Icons list

**Overlapping items**: Increase position spacing (x+2, y+1 pattern)

**Labels clipped**: Adjust label `position` value (0-100)

**Export fails**: Verify JSON structure with schema validation
