# Using Claude Code CLI

**Complete guide to Claude Code, skills, and agent templates in Turbo Flow.**

---

## Starting Claude Code

### From VNC Desktop

```bash
# Open terminal in VNC, you're already logged in as devuser
claude
```

### From SSH

```bash
ssh -p 2222 devuser@localhost
claude
```

### From Docker Exec

```bash
docker exec -u devuser -it turbo-flow-unified claude
```

---

## What You Have Access To

### 1. Claude Code Skills (6 Available)

Skills in `~/.claude/skills/` are **automatically discovered**:

| Skill | Purpose | Usage |
|-------|---------|-------|
| **web-summary** | YouTube transcripts, web summarization | `> Summarize https://...` |
| **blender** | 3D modeling automation | `> Create 3D cube in Blender` |
| **qgis** | GIS operations | `> Load shapefile and analyze` |
| **kicad** | PCB design | `> Create LED circuit schematic` |
| **imagemagick** | Image processing | `> Resize image.jpg to 800x600` |
| **pbr-rendering** | PBR material generation | `> Create metal material` |

**Skills activate automatically** - just describe what you need!

### 2. Agent Templates (610 Available)

Agents in `~/agents/` provide methodologies and workflows:

```bash
# View all agents
ls ~/agents/*.md | wc -l  # 610

# Essential agents (always load these first)
cat ~/agents/doc-planner.md          # SPARC methodology
cat ~/agents/microtask-breakdown.md  # Task decomposition

# Search for specific agents
find ~/agents/ -name "*github*"
find ~/agents/ -name "*security*"
find ~/agents/ -name "*test*"
```

### 3. Environment Variables

Inside the container:

```bash
echo $WORKSPACE              # /home/devuser/workspace
echo $AGENTS_DIR             # /home/devuser/agents
echo $ANTHROPIC_API_KEY      # Your Claude API key
echo $ZAI_API_KEY            # Z.AI API key (web-summary skill)
echo $GITHUB_TOKEN           # GitHub token (if configured)
```

---

## Essential Workflows

### 1. Structured Development with Agents

```bash
claude

# Load essential agents
> cat ~/agents/doc-planner.md ~/agents/microtask-breakdown.md

# Describe your project
> Using doc-planner methodology, help me design a REST API
> with authentication for a todo application
```

**Claude will**:
- Apply SPARC methodology (Specification, Pseudocode, Architecture, Refinement, Completion)
- Break down into 10-minute microtasks
- Create detailed implementation plan

### 2. Using Web Summary Skill

```bash
> Summarize this YouTube video: https://www.youtube.com/watch?v=VIDEO_ID

> Extract key points from this article: https://example.com/long-article

> Create Logseq notes with semantic topic links from this blog post
```

The web-summary skill:
- Extracts YouTube transcripts automatically
- Summarizes web content using Z.AI (cost-effective)
- Generates semantic topic links for knowledge management
- Outputs in markdown, plain text, or JSON

### 3. GitHub Integration

```bash
# Load GitHub agents
> cat ~/agents/github-pr-manager.md ~/agents/github-code-reviewer.md

# Work with PRs
> Review this pull request: https://github.com/user/repo/pull/123

> Create a PR for my current branch with conventional commit messages
```

### 4. Blender 3D Modeling

```bash
> Create a 3D scene with a torus and cube, add materials and lighting, then render

> Generate a procedural terrain mesh with displacement mapping

> Import model.obj, apply UV unwrapping, and export as FBX
```

Blender skill communicates via socket (port 2800).

### 5. Image Processing

```bash
> Resize all JPG images in current directory to 1920x1080

> Convert image.png to image.jpg with 90% quality

> Create a thumbnail grid from photos/ directory

> Apply blur filter to background.jpg
```

---

## Agent Discovery

### Find Relevant Agents

```bash
# Terminal (outside Claude Code)
find ~/agents/ -name "*api*"
find ~/agents/ -name "*docker*"
find ~/agents/ -name "*security*"

# Random sample
ls ~/agents/*.md | shuf | head -10

# Count agents
ls ~/agents/*.md | wc -l  # 610
```

### Load Multiple Agents

```bash
# Inside Claude CLI
> cat ~/agents/doc-planner.md \
      ~/agents/microtask-breakdown.md \
      ~/agents/github-pr-manager.md

# Now describe task - Claude uses all loaded agents' methodologies
```

### Essential Agent Categories

**Development**:
- `doc-planner.md` - SPARC methodology
- `microtask-breakdown.md` - Atomic task breakdown
- `code-reviewer.md` - Code review
- `test-generator.md` - Test creation

**GitHub**:
- `github-pr-manager.md` - Pull request management
- `github-issue-tracker.md` - Issue workflows
- `github-release-manager.md` - Release automation
- `github-security-manager.md` - Security scanning

**Security**:
- `security-manager.md` - Security analysis
- `vulnerability-scanner.md` - CVE scanning
- `secure-coding.md` - Best practices

---

## Advanced Usage

### Working with External Projects

If `PROJECT_DIR` is set in `.env`:

```bash
cd ~/workspace/project  # Your external codebase
claude

> Analyze the architecture of this project and suggest improvements
```

### Multi-User AI Switching

```bash
# Start as devuser (Claude)
claude

# Exit and switch to Gemini
exit
as-gemini
# Now using gemini-user with Gemini credentials

# Switch to OpenAI
as-openai
# Now using openai-user with OpenAI credentials
```

Each user has isolated:
- API credentials (`~/.config/{claude,gemini,openai}/`)
- Workspace (`~/workspace/`)
- Configuration

### Batch Operations

```bash
# Load agent once, use multiple times
> cat ~/agents/security-manager.md

> Scan app.py for vulnerabilities
> Scan database.py for SQL injection
> Scan auth.py for authentication issues
```

---

## Skills Deep Dive

### Web Summary Skill

**Configuration**: `~/.config/zai/api.json` (ZAI_API_KEY)

**Capabilities**:
- YouTube transcript extraction
- Web page summarization
- Semantic topic generation
- Knowledge management integration (Logseq)
- Multiple output formats

**Example**:
```bash
> Summarize this video and create Logseq notes with topic links:
> https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

Output includes:
- Summary with key points
- Timestamps (for videos)
- `[[topic-name]]` links for Logseq
- Related topics suggestions

### Blender Skill

**Communication**: Socket on port 2800

**Capabilities**:
- 3D modeling operations
- Material and lighting setup
- Rendering
- Import/export multiple formats
- Procedural generation

**Example**:
```bash
> Create a procedural landscape:
> - Plane mesh 100x100 units
> - Subdivision surface modifier
> - Displacement with noise texture
> - Green grass material
> - Sun lighting at 45 degrees
> - Render 1920x1080
```

### QGIS Skill

**Capabilities**:
- Load shapefiles, GeoJSON, rasters
- Spatial analysis
- Map generation
- Coordinate transformations
- Attribute queries

**Example**:
```bash
> Load cities.geojson and countries.shp
> Filter cities with population > 1 million
> Generate choropleth map by country
> Export as PNG
```

### KiCAD Skill

**Capabilities**:
- Schematic design
- PCB layout
- Component libraries
- Design rule checking
- Gerber export

**Example**:
```bash
> Create LED circuit schematic:
> - 5V power supply
> - 220Ω current-limiting resistor
> - Standard LED
> - Generate PCB layout
> - Export Gerber files
```

### ImageMagick Skill

**Capabilities**:
- Resize, crop, rotate
- Format conversion
- Filters and effects
- Batch processing
- Composite images

**Example**:
```bash
> Batch process all images in photos/:
> - Resize to max 1920px width (maintain aspect ratio)
> - Apply subtle sharpening
> - Convert to JPG 85% quality
> - Save to processed/ directory
```

### PBR Rendering Skill

**Capabilities**:
- PBR material generation
- Texture baking
- Material parameter tuning
- Physically-based rendering

**Example**:
```bash
> Create brushed metal PBR material:
> - Metallic: 1.0
> - Roughness: 0.3
> - Anisotropic: 0.8
> - Generate albedo, normal, roughness maps
```

---

## Complete Example Workflow

### Building a REST API with Full Methodology

```bash
# 1. Start Claude
claude

# 2. Load essential agents
> cat ~/agents/doc-planner.md
> cat ~/agents/microtask-breakdown.md

# 3. Initial specification
> Using SPARC methodology, help me design and implement a REST API for a todo application with:
> - User authentication (JWT)
> - CRUD operations for todos
> - SQLite database
> - FastAPI framework
> - Comprehensive tests

# Claude will create:
# - Specification document
# - Pseudocode for key algorithms
# - Architecture diagram
# - Breakdown into microtasks
# - Implementation plan

# 4. Iterative development
> Implement task 1: Database schema and models
> Implement task 2: Authentication endpoints
> Implement task 3: Todo CRUD operations
# ... etc

# 5. Testing
> cat ~/agents/test-generator.md
> Generate comprehensive tests for the authentication module

# 6. GitHub integration
> cat ~/agents/github-pr-manager.md
> Create a pull request for this implementation
```

---

## Quick Reference

### Starting Claude

```bash
claude                              # Normal mode
claude --dangerously-skip-permissions  # Skip permission prompts (use with caution)
dsp                                # Alias for dangerous mode
```

### Essential Commands

```bash
# Load agent
> cat ~/agents/agent-name.md

# Exit Claude
> exit

# Check available skills
ls ~/.claude/skills/

# Check available agents
ls ~/agents/*.md | wc -l

# Search agents
find ~/agents/ -name "*keyword*"
```

### Environment Locations

```bash
~/.claude/skills/           # 6 skills
~/agents/                  # 610 agent templates
~/workspace/               # Working directory
~/workspace/project/       # External project (if mounted)
~/.config/claude/          # Claude configuration
~/.config/zai/api.json     # Z.AI API key (web-summary)
```

### User Switching Aliases

```bash
as-gemini                  # Switch to gemini-user
as-openai                  # Switch to openai-user
as-zai                     # Switch to zai-user
```

---

## Tips and Best Practices

### 1. Always Load Essential Agents First

```bash
> cat ~/agents/doc-planner.md ~/agents/microtask-breakdown.md
```

This ensures structured, methodical development.

### 2. Use Skills Naturally

Don't specify "use the web-summary skill" - just ask:

```bash
> Summarize this article: https://example.com
```

Claude Code automatically detects and uses the appropriate skill.

### 3. Break Down Complex Tasks

```bash
> cat ~/agents/microtask-breakdown.md
> Break this feature into 10-minute tasks: [complex feature description]
```

Work through one microtask at a time.

### 4. Iterate and Refine

```bash
> Review the code we just wrote for potential issues
> Refactor the authentication module for better testability
> Add error handling to the database operations
```

### 5. Use GitHub Agents for Workflows

```bash
> cat ~/agents/github-pr-manager.md
> cat ~/agents/github-code-reviewer.md

# Now GitHub operations are integrated into Claude's workflow
```

---

**You have 610 agents and 6 skills ready to accelerate your development!**
