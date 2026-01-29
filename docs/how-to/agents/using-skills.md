---
title: Multi-Agent Skills - Natural Language Reference
description: **Last Updated:** November 5, 2025 **Status:** Production **VisionFlow Integration:** Agent Control Interface
category: how-to
tags:
  - tutorial
  - api
  - api
  - docker
  - database
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Multi-Agent Skills - Natural Language Reference

> **Note**: For AI model integrations (DeepSeek, Perplexity, RAGFlow, Z.AI), see the [AI Models & Services Guide](ai-models/README.md).

**Last Updated:** November 5, 2025
**Status:** Production
**VisionFlow Integration:** Agent Control Interface

---

## Overview

VisionFlow integrates with a sophisticated multi-agent Docker environment that provides 13+ specialized Claude skills. These skills can be invoked using natural language through the VisionFlow agent control interface, enabling automated development workflows, strategic analysis, and cross-container orchestration.

**Quick Start:**
```
Use Docker Manager to check VisionFlow status
Create a Wardley map for our ontology architecture
Use Chrome DevTools to debug http://localhost:3001
```

---

## Architecture Integration

### VisionFlow → Multi-Agent Communication

```
┌────────────────────────────────────────┐
│      VisionFlow Main Container         │
│    (Graph Engine + Neo4j + WebXR)      │
│                                        │
│         Port: 9090 (API)               │
│         Port: 3001 (Client)            │
└────────────────────────────────────────┘
                  ↕
        Docker Network: docker_ragflow
                  ↕
┌────────────────────────────────────────┐
│   Agentic Workstation Container        │
│  (Claude Code + 13 Skills + MCP)       │
│                                        │
│  • Docker socket mounted               │
│  • Project workspace shared            │
│  • Natural language interface          │
└────────────────────────────────────────┘
```

**Key Integration Points:**
- **Docker Socket**: `/var/run/docker.sock` mounted for container management
- **Shared Workspace**: Project files accessible from both containers
- **Network Bridge**: `docker_ragflow` network for inter-container communication
- **MCP Protocol**: TCP (port 9500) and WebSocket (port 3002) for tool invocation

---

## Core Skills (Always Available)

### 1. Docker Manager
**Control VisionFlow from within agentic-workstation**

**Natural Language Examples:**
```
Build VisionFlow with no cache
Start VisionFlow in development mode
Restart VisionFlow and rebuild
Check VisionFlow container status
Show me the last 50 lines of VisionFlow logs
Execute "cargo test" in VisionFlow container
```

**Capabilities:**
- Container lifecycle (build, start, stop, restart)
- Log streaming and monitoring
- Direct command execution in VisionFlow
- Health checks and status reporting
- Network discovery on docker_ragflow

**Use When:**
- Testing code changes in VisionFlow
- Debugging container issues
- Automating deployment workflows
- Monitoring application health

**Implementation:** `multi-agent-docker/skills/docker-manager/`

---

### 2. Wardley Mapper
**Strategic mapping and competitive analysis**

**Natural Language Examples:**
```
Create a Wardley map for VisionFlow's ontology architecture
Map our GPU acceleration capabilities against competitors
Analyze the evolution of our graph database infrastructure
Generate a strategic map for the multi-agent system
Map VisionFlow's value chain from user to infrastructure
```

**Capabilities:**
- Transforms ANY input into Wardley maps
- Business model mapping
- Technical architecture visualization
- Competitive landscape analysis
- Evolution assessment (Genesis → Commodity)

**Map Types:**
- **Business Maps**: Value chains, competitive positioning
- **Technical Maps**: Architecture components, dependencies
- **Data Maps**: Information flow, storage evolution
- **Competitive Maps**: Market positioning, strategic gaps

**Use When:**
- Planning architectural decisions
- Analyzing competitive positioning
- Communicating strategy to stakeholders
- Identifying technical debt and evolution opportunities

**Advanced Features:**
- Multi-perspective mapping (business, technical, data, competitive)
- Automatic component extraction from descriptions
- Evolution stage assessment
- Dependency chain analysis
- SVG/PNG export

**Implementation:** `multi-agent-docker/skills/wardley-maps/`

---

### 3. Chrome DevTools
**Debug web applications with full Chrome DevTools**

**Natural Language Examples:**
```
Use Chrome DevTools to debug http://localhost:3001
Capture a performance trace of the graph rendering
Take a screenshot of the 3D visualization
Inspect network requests to the Neo4j API
Check console errors on the VisionFlow client
Analyze DOM structure of the ontology viewer
```

**Capabilities:**
- Performance profiling and tracing
- Network inspection and HAR export
- Console debugging and error tracking
- DOM/CSS inspection and manipulation
- Screenshot capture
- JavaScript debugging

**Use When:**
- Debugging frontend performance issues
- Analyzing network requests and timing
- Inspecting React component hierarchy
- Capturing bugs with screenshots
- Profiling rendering performance

**Implementation:** `multi-agent-docker/skills/chrome-devtools/`

---

## Content & Media Skills

### 4. Blender (3D Modeling & Rendering)
**Programmatic 3D content creation**

**Natural Language Examples:**
```
Create a 3D cube in Blender
Generate a photorealistic render of a molecular structure
Model a network graph in 3D space
Apply physics simulation to graph nodes
Export the scene as GLB for WebXR
```

**Capabilities:**
- 3D modeling via Python API
- Materials and shading
- Physics simulation
- Rendering (Cycles, Eevee)
- Animation and keyframing
- Export to various formats (FBX, GLB, OBJ)

**Use When:**
- Creating 3D visualizations for VisionFlow
- Generating marketing materials
- Prototyping WebXR environments
- Creating training data for ML models

**Requirements:** GUI container running, VNC access available

**Implementation:** `multi-agent-docker/skills/blender/`

---

### 5. ImageMagick (Image Processing)
**Batch image manipulation and conversion**

**Natural Language Examples:**
```
Resize all PNGs in ./assets to 1024x1024
Convert SVG icons to PNG with transparency
Create a montage of graph screenshots
Add watermark to documentation images
Optimize images for web delivery
```

**Capabilities:**
- Image conversion (100+ formats)
- Resizing and cropping
- Compositing and montage
- Color manipulation
- Batch processing

**Use When:**
- Preparing documentation images
- Optimizing assets for deployment
- Creating thumbnails and previews
- Batch processing screenshots

**Implementation:** `multi-agent-docker/skills/imagemagick/`

---

### 6. PBR Rendering (Physically-Based Materials)
**Generate realistic textures and materials**

**Natural Language Examples:**
```
Generate brushed metal PBR textures at 2048x2048
Create a rust material set with albedo, normal, metallic, roughness
Generate wood grain textures for 3D models
Create a concrete material with ambient occlusion
```

**Capabilities:**
- PBR texture generation (Albedo, Normal, Metallic, Roughness, AO)
- Material presets (metal, wood, stone, plastic, concrete)
- Custom resolution support
- Tiling texture creation

**Use When:**
- Creating 3D assets for WebXR
- Generating realistic materials for visualization
- Prototyping visual designs

**Requirements:** GUI container running

**Implementation:** `multi-agent-docker/skills/pbr-rendering/`

---

## Web & Automation Skills

### 7. Playwright (Browser Automation)
**End-to-end web testing and automation**

**Natural Language Examples:**
```
Navigate to localhost:3001 and click the 'Import Ontology' button
Fill in the GitHub sync form and submit
Test the graph physics controls by dragging nodes
Capture a video of the 3D visualization loading
Run accessibility checks on the settings panel
```

**Capabilities:**
- Browser automation (Chromium, Firefox, WebKit)
- Form filling and interaction
- Screenshot and video recording
- Network interception
- Accessibility testing

**Use When:**
- Automated UI testing
- Creating demo videos
- Testing user workflows
- Accessibility validation

**Implementation:** `multi-agent-docker/skills/playwright/`

---

### 8. Web Summary (Intelligent Web Scraping)
**Extract structured data from web pages**

**Natural Language Examples:**
```
Summarize the latest Neo4j release notes
Extract documentation from the Babylon.js API docs
Get code examples from the OWL 2 specification
Download and parse GitHub repository README
```

**Capabilities:**
- Intelligent content extraction
- Markdown conversion
- Multi-page crawling
- Link following
- Content summarization

**Use When:**
- Research and documentation gathering
- Competitive analysis
- Updating documentation from external sources
- Extracting code examples

**Implementation:** `multi-agent-docker/skills/web-summary/`

---

## Data & Analysis Skills

### 9. Import to Ontology (OWL Data Import)
**Convert external data to VisionFlow ontologies**

**Natural Language Examples:**
```
Import JSON schema from ./api-spec.json to OWL ontology
Convert the database ERD to OWL classes and properties
Generate ontology from TypeScript interfaces
Import SKOS vocabulary as OWL concepts
```

**Capabilities:**
- JSON/CSV to OWL conversion
- Schema inference
- Relationship extraction
- Namespace generation
- Axiom creation

**Use When:**
- Migrating legacy data to VisionFlow
- Integrating external vocabularies
- Generating ontologies from schemas
- Bootstrapping knowledge graphs

**⚠️ Warning:** This skill performs database writes. Review generated OWL before import.

**Implementation:** `multi-agent-docker/skills/import-to-ontology/`

---

### 10. QGIS (Geospatial Analysis)
**Geographic information systems and spatial analysis**

**Natural Language Examples:**
```
Load the GeoJSON layer and calculate polygon areas
Create a heatmap of graph node locations
Generate a spatial index for entity coordinates
Export the map as a styled PNG
```

**Capabilities:**
- Vector and raster data processing
- Spatial analysis algorithms
- Map generation and styling
- Coordinate transformation
- Geocoding and reverse geocoding

**Use When:**
- Analyzing geospatially-tagged entities
- Creating location-based visualizations
- Processing geographic datasets
- Generating cartographic outputs

**Requirements:** GUI container running, VNC access available

**Implementation:** `multi-agent-docker/skills/qgis/`

---

## Engineering & Electronics Skills

### 11. KiCad (Electronic Design Automation)
**Circuit design and PCB layout**

**Natural Language Examples:**
```
Create a schematic for a sensor board
Generate a BOM for the circuit
Export the PCB layout as Gerber files
Run electrical rule check on the schematic
```

**Capabilities:**
- Schematic capture
- PCB layout design
- BOM generation
- DRC/ERC checking
- Manufacturing file export (Gerber, drill files)

**Use When:**
- Designing hardware interfaces for sensors
- Creating reference designs
- Generating manufacturing documentation

**Requirements:** GUI container running

**Implementation:** `multi-agent-docker/skills/kicad/`

---

### 12. NGSpice (Circuit Simulation)
**SPICE-based circuit simulation and analysis**

**Natural Language Examples:**
```
Simulate the sensor amplifier circuit
Run DC sweep analysis on the voltage regulator
Perform transient analysis on the filter circuit
Export simulation results as CSV
```

**Capabilities:**
- DC, AC, transient analysis
- Monte Carlo simulation
- Noise analysis
- SPICE netlist parsing
- Result export

**Use When:**
- Validating circuit designs
- Analyzing signal conditioning circuits
- Troubleshooting electronic issues

**Implementation:** `multi-agent-docker/skills/ngspice/`

---

### 13. Logseq Formatted (Structured Knowledge Export)
**Export data in Logseq-compatible markdown format**

**Natural Language Examples:**
```
Export graph documentation to Logseq format
Convert ontology classes to Logseq pages
Generate linked notes from knowledge graph
Create daily notes from commit history
```

**Capabilities:**
- Markdown with metadata
- Bi-directional linking
- Hierarchical organization
- Tag generation
- Timestamp formatting

**Use When:**
- Integrating VisionFlow with personal knowledge management
- Generating documentation for Logseq/Roam Research
- Creating linked note systems

**Implementation:** `multi-agent-docker/skills/logseq-formatted/`

---

## Skill Invocation Patterns

### Direct Natural Language

Most straightforward - just describe what you want:

```
Use Docker Manager to restart VisionFlow
Create a Wardley map for our architecture
Debug the client with Chrome DevTools
```

### Specific Parameters

For more control, include parameters in natural language:

```
Use Docker Manager to build VisionFlow with no cache and force rebuild
Create a Wardley map analyzing the competitive landscape with technical focus
Use Chrome DevTools to capture a performance trace for 10 seconds
```

### Chained Operations

Combine multiple skills in a workflow:

```
1. Use Docker Manager to build and restart VisionFlow
2. Use Playwright to navigate to localhost:3001
3. Use Chrome DevTools to capture a screenshot
4. Use ImageMagick to resize the screenshot to 1200x800
```

### Conditional Execution

Skills can check conditions before executing:

```
If VisionFlow is not running, use Docker Manager to start it
Only rebuild if there are uncommitted changes
Run tests before restarting the container
```

---

## Skill Development

### Creating New Skills

Skills are defined in `/home/devuser/.claude/skills/` with:

1. **SKILL.md** - YAML frontmatter + documentation
2. **Implementation** - Script/binary/MCP server
3. **Test suite** - Validation scripts

**Example SKILL.md:**
```markdown
---
name: my-skill
description: Short description of what this skill does
---

# My Skill

## Capabilities
...

## Natural Language Examples
...
```

### Skill Registration

Skills are automatically discovered by Claude Code when placed in `.claude/skills/`. The MCP server indexes them and makes them available for natural language invocation.

### Testing Skills

```bash
# Inside agentic-workstation container
cd /home/devuser/.claude/skills/my-skill
./test-skill.sh
```

---

## Troubleshooting

### Skill Not Available

```bash
# Check skill installation
docker exec agentic-workstation ls -la /home/devuser/.claude/skills/

# Verify MCP server is running
docker exec agentic-workstation mcp-tcp-status
```

### Docker Socket Issues

```bash
# Verify socket is mounted
docker exec agentic-workstation ls -la /var/run/docker.sock

# Should show: srw-rw---- 1 root docker
```

### GUI Skills Timeout

GUI-dependent skills (Blender, QGIS, KiCad) require the GUI container:

```bash
# Check GUI container status
docker ps | grep gui-tools-container

# Access GUI via VNC
# vnc://localhost:5901 (password: turboflow)
```

### Skill Execution Fails

```bash
# Check skill logs
docker logs agentic-workstation | grep "my-skill"

# Test skill directly
docker exec agentic-workstation /home/devuser/.claude/skills/my-skill/test-skill.sh
```

---

## Related Documentation

- [Multi-Agent Architecture](../../explanations/architecture/multi-agent-system.md)
- 
- 
- [Docker Environment Setup](../guides/docker-environment-setup.md)
- 

---

## Learning Path

### Beginner
1. Use Docker Manager to control VisionFlow
2. Try Chrome DevTools for debugging
3. Create simple Wardley maps

### Intermediate
4. Chain skills together (Docker Manager → Playwright → Chrome DevTools)
5. Use Import to Ontology for data migration
6. Generate PBR textures for WebXR

### Advanced
7. Create custom skills
8. Integrate skills with VisionFlow APIs
9. Build automated workflows with multiple skills

---

**All skills are accessible via natural language through the VisionFlow agent control interface.**

**For detailed implementation guides, see individual skill documentation in:**
`/home/user/VisionFlow/docs/reference/skills/`
