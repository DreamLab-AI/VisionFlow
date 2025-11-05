# Skills and Agents Reference

**Complete reference for all 6 skills and 610 agent templates.**

---

## Claude Code Skills (6 Available)

Skills are located in `~/.claude/skills/` and activate automatically when needed.

### 1. Web Summary

**Location**: `~/.claude/skills/web-summary/`
**Purpose**: Web content and YouTube video summarization with semantic topics

**Capabilities**:
- Extract YouTube video transcripts
- Summarize web pages and articles
- Generate semantic topic links (Logseq format)
- Multiple output formats (markdown, JSON, plain text)
- Uses Z.AI for cost-effective API calls

**Configuration**:
- Requires `ZAI_API_KEY` in `~/.config/zai/api.json`
- Automatically configured by entrypoint

**Usage Examples**:
```bash
# Summarize YouTube video
> Summarize this video: https://www.youtube.com/watch?v=VIDEO_ID

# Summarize article
> Extract key points from https://example.com/long-article

# Create Logseq notes with semantic topics
> Summarize this blog post and generate [[topic]] links for Logseq
```

**Output Format**:
```markdown
# Summary

[Main points extracted]

## Key Topics
- [[artificial-intelligence]]
- [[machine-learning]]
- [[neural-networks]]

## Timestamps (for videos)
- 0:00 - Introduction
- 2:30 - Main concept
- 5:45 - Conclusion
```

---

### 2. Blender

**Location**: `~/.claude/skills/blender/`
**Purpose**: 3D modeling, materials, lighting, and rendering

**Communication**: Socket on port 2800

**Capabilities**:
- 3D mesh creation and modification
- Material and shader setup
- Lighting and camera configuration
- Rendering (Cycles, Eevee)
- Import/export (FBX, OBJ, GLTF, etc.)
- Procedural generation
- Animation (basic)

**Usage Examples**:
```bash
# Create simple scene
> Create a 3D cube with red material and render

# Complex procedural scene
> Create a procedural landscape:
> - Plane mesh 100x100 units
> - Subdivision surface modifier
> - Displacement with noise texture
> - Green grass material with roughness variation
> - Sun lighting at 45-degree angle
> - Render at 1920x1080 with Cycles

# Import and modify
> Import model.obj, apply UV unwrapping, and export as FBX
```

**Features**:
- Python scripting via bpy
- Node-based material editing
- Modifier stack operations
- Camera and lighting automation

---

### 3. QGIS

**Location**: `~/.claude/skills/qgis/`
**Purpose**: Geographic Information System operations

**Capabilities**:
- Load vector data (shapefiles, GeoJSON, KML)
- Load raster data (GeoTIFF, PNG with world file)
- Spatial analysis (buffer, intersection, union)
- Map generation and styling
- Coordinate transformation
- Attribute queries and filtering
- Export maps as images

**Usage Examples**:
```bash
# Basic map
> Load cities.geojson and countries.shp
> Style cities by population
> Generate map and export as PNG

# Spatial analysis
> Load roads.shp and parks.geojson
> Find all parks within 500m of major roads
> Create buffer zones and export results

# Choropleth map
> Load countries.geojson with population data
> Create choropleth map colored by population density
> Add legend and scale bar
> Export at 300 DPI
```

**Formats Supported**:
- Vector: Shapefile, GeoJSON, KML, GPX
- Raster: GeoTIFF, PNG+world file
- Export: PNG, PDF, SVG

---

### 4. KiCAD

**Location**: `~/.claude/skills/kicad/`
**Purpose**: Electronic circuit design and PCB layout

**Capabilities**:
- Schematic design
- Component library access
- PCB layout
- Design rule checking (DRC)
- Electrical rule checking (ERC)
- Gerber file generation
- Bill of Materials (BOM) export

**Usage Examples**:
```bash
# Simple LED circuit
> Create LED circuit schematic:
> - 5V DC power supply
> - 220Ω current-limiting resistor
> - Standard red LED
> - Ground connection
> Generate netlist and create PCB

# Microcontroller project
> Design Arduino-based sensor board:
> - ATmega328P microcontroller
> - 3.3V regulator
> - I2C temperature sensor
> - USB-to-serial interface
> - Create PCB layout with 2-layer board
> - Export Gerber files for manufacturing
```

**Output**:
- Schematic files (.kicad_sch)
- PCB layout (.kicad_pcb)
- Gerber files (manufacturing)
- Bill of Materials (CSV)

---

### 5. ImageMagick

**Location**: `~/.claude/skills/imagemagick/`
**Purpose**: Image processing and manipulation

**Capabilities**:
- Resize, crop, rotate images
- Format conversion
- Filters and effects
- Color adjustments
- Compositing and layering
- Batch processing
- Text and annotation
- Watermarking

**Usage Examples**:
```bash
# Basic operations
> Resize image.jpg to 800x600 maintaining aspect ratio
> Convert image.png to image.jpg with 90% quality
> Rotate photo.jpg by 90 degrees clockwise

# Batch processing
> Resize all JPG images in photos/ to max 1920px width
> Apply subtle sharpening to all images
> Save to processed/ directory

# Advanced effects
> Create thumbnail grid from all images in directory
> Apply gaussian blur to background.jpg
> Add watermark "Copyright 2025" to image.jpg bottom-right
> Composite overlay.png onto base.jpg with 50% opacity
```

**Common Filters**:
- Blur, sharpen, edge detection
- Brightness, contrast, saturation
- Sepia, grayscale, colorize
- Emboss, oil paint, sketch

---

### 6. PBR Rendering

**Location**: `~/.claude/skills/pbr-rendering/`
**Purpose**: Physically-Based Rendering material generation

**Capabilities**:
- PBR material creation
- Texture map generation (albedo, normal, roughness, metallic)
- Material parameter tuning
- Shader graph creation
- Texture baking
- HDR environment setup

**Usage Examples**:
```bash
# Metal material
> Create brushed aluminum PBR material:
> - Metallic: 1.0
> - Roughness: 0.3
> - Anisotropic: 0.8
> - Generate all texture maps at 2048x2048

# Weathered material
> Create weathered concrete material:
> - Base color: light gray
> - Roughness variation (smooth to rough)
> - Normal map with cracks and imperfections
> - Ambient occlusion for crevices

# Complete material library
> Generate PBR material set for game assets:
> - Wood (oak, pine)
> - Metal (steel, copper, brass)
> - Fabric (cotton, leather)
> - Stone (granite, marble)
> All at 1024x1024, game-ready
```

**Output Maps**:
- Albedo/Base Color
- Normal
- Roughness
- Metallic
- Ambient Occlusion
- Height/Displacement

---

## Agent Templates (610 Available)

Agents are markdown files in `~/agents/` providing methodologies and workflows.

### Essential Agents (Load First)

#### doc-planner.md
**Purpose**: SPARC methodology for structured development
**Methodology**: Specification → Pseudocode → Architecture → Refinement → Completion
**Use When**: Starting any significant project
**Example**:
```bash
> cat ~/agents/doc-planner.md
> Using SPARC, design a REST API for a blog platform
```

#### microtask-breakdown.md
**Purpose**: Break features into 10-minute atomic tasks
**Methodology**: London School TDD, atomic commits
**Use When**: Implementation phase
**Example**:
```bash
> cat ~/agents/microtask-breakdown.md
> Break this feature into microtasks: User authentication with JWT
```

### GitHub Integration Agents (13 Available)

Located in `~/agents/` with `github-*` prefix:

| Agent | Purpose |
|-------|---------|
| `github-pr-manager.md` | Pull request creation and management |
| `github-code-reviewer.md` | Multi-perspective code review |
| `github-issue-tracker.md` | Issue workflow automation |
| `github-release-manager.md` | Release planning and execution |
| `github-security-manager.md` | Security scanning and fixes |
| `github-workflow-manager.md` | CI/CD optimization |
| `github-actions-creator.md` | GitHub Actions development |

**Usage**:
```bash
> cat ~/agents/github-pr-manager.md ~/agents/github-code-reviewer.md
> Review this PR: https://github.com/user/repo/pull/123
```

### Development Agents

**Code Quality**:
- `code-reviewer.md` - Comprehensive code review
- `refactoring-agent.md` - Code refactoring suggestions
- `performance-optimizer.md` - Performance analysis
- `test-generator.md` - Test creation (unit, integration, e2e)

**Architecture**:
- `system-architect.md` - System design
- `api-designer.md` - REST/GraphQL API design
- `database-architect.md` - Schema design
- `microservices-architect.md` - Microservices patterns

**Security**:
- `security-manager.md` - Security audit
- `vulnerability-scanner.md` - CVE scanning
- `secure-coding.md` - Security best practices
- `penetration-tester.md` - Security testing

### Language-Specific Agents

**Python**:
- `python-developer.md`
- `python-async-expert.md`
- `django-developer.md`
- `fastapi-developer.md`

**Rust**:
- `rust-developer.md`
- `rust-async-expert.md`
- `rust-systems-programmer.md`

**JavaScript/TypeScript**:
- `javascript-developer.md`
- `typescript-developer.md`
- `react-developer.md`
- `nodejs-developer.md`

**Others**:
- `go-developer.md`
- `cpp-developer.md`
- `java-developer.md`

### Domain-Specific Agents

**DevOps**:
- `docker-specialist.md`
- `kubernetes-expert.md`
- `terraform-specialist.md`
- `ci-cd-engineer.md`

**Data Science**:
- `data-scientist.md`
- `ml-engineer.md`
- `data-analyst.md`

**Web Development**:
- `frontend-developer.md`
- `backend-developer.md`
- `full-stack-developer.md`

---

## Discovering Agents

### Search by Keyword

```bash
# Find all GitHub agents
find ~/agents/ -name "*github*"

# Find security-related agents
find ~/agents/ -name "*security*" -o -name "*vulnerability*"

# Find Python agents
find ~/agents/ -name "*python*"

# Find testing agents
find ~/agents/ -name "*test*"
```

### Browse by Category

```bash
# List all agents
ls ~/agents/*.md

# Count total agents
ls ~/agents/*.md | wc -l  # 610

# Random sample of 10 agents
ls ~/agents/*.md | shuf | head -10

# View agent content
cat ~/agents/agent-name.md | less
```

### Agent File Format

Each agent is a markdown file with:
- Role description
- Methodology/approach
- Best practices
- Common patterns
- Example workflows

---

## Using Multiple Agents

### Load Complementary Agents

```bash
# Full-stack development
> cat ~/agents/doc-planner.md \
      ~/agents/microtask-breakdown.md \
      ~/agents/full-stack-developer.md \
      ~/agents/test-generator.md

# GitHub workflow
> cat ~/agents/github-pr-manager.md \
      ~/agents/github-code-reviewer.md \
      ~/agents/ci-cd-engineer.md
```

### Sequential Agent Usage

```bash
# 1. Planning phase
> cat ~/agents/doc-planner.md
> Design a microservices architecture for e-commerce

# 2. Task breakdown
> cat ~/agents/microtask-breakdown.md
> Break the authentication service into microtasks

# 3. Implementation
> cat ~/agents/python-developer.md
> Implement task 1: Database models

# 4. Testing
> cat ~/agents/test-generator.md
> Generate tests for the authentication service

# 5. Code review
> cat ~/agents/code-reviewer.md
> Review the implementation

# 6. GitHub integration
> cat ~/agents/github-pr-manager.md
> Create a pull request
```

---

## Best Practices

### 1. Always Start with Essential Agents

```bash
> cat ~/agents/doc-planner.md ~/agents/microtask-breakdown.md
```

Ensures methodical, structured development.

### 2. Load Relevant Domain Agents

Match agents to your tech stack:

```bash
# FastAPI project
> cat ~/agents/python-developer.md ~/agents/fastapi-developer.md

# Rust game
> cat ~/agents/rust-developer.md ~/agents/game-developer.md
```

### 3. Use GitHub Agents for Team Work

```bash
> cat ~/agents/github-pr-manager.md ~/agents/github-code-reviewer.md
```

Integrates team workflows into development.

### 4. Combine Security Throughout

```bash
> cat ~/agents/doc-planner.md \
      ~/agents/security-manager.md \
      ~/agents/secure-coding.md
```

Security by design, not as afterthought.

### 5. Test-Driven with Agents

```bash
> cat ~/agents/microtask-breakdown.md ~/agents/test-generator.md
```

Generate tests alongside implementation.

---

## Quick Reference

### Skill Locations

```bash
~/.claude/skills/web-summary/       # Web and YouTube summarization
~/.claude/skills/blender/           # 3D modeling
~/.claude/skills/qgis/              # GIS operations
~/.claude/skills/kicad/             # PCB design
~/.claude/skills/imagemagick/       # Image processing
~/.claude/skills/pbr-rendering/     # PBR materials
```

### Agent Locations

```bash
~/agents/*.md                       # All 610 agents
~/agents/doc-planner.md            # SPARC methodology
~/agents/microtask-breakdown.md    # Task atomization
~/agents/github-*.md               # GitHub integration (13 agents)
```

### Using Skills

Skills activate automatically - just describe what you need:

```bash
> Summarize https://example.com
> Create 3D model of a house in Blender
> Process images in photos/ directory
```

### Using Agents

Load with `cat` command in Claude CLI:

```bash
> cat ~/agents/agent-name.md
> [Describe your task]
```

---

**You have a comprehensive toolkit: 6 powerful skills and 610 specialized agents ready to accelerate your development workflow!**
