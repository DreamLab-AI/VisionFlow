// Continuing with SKILLS.md creation...

# Claude Code Skills Guide

Complete guide to using and developing custom Claude Code skills in the Turbo Flow Unified environment.

## Overview

Skills are Claude Code's replacement for MCP servers, providing:
- Progressive disclosure (loaded on-demand)
- Better token efficiency
- Simpler architecture (no HTTP overhead)
- Direct integration with Claude Code

## Available Skills

### Web Summary Skill

**Purpose**: Fetch and summarize web content including YouTube videos

**Use Cases**:
- Summarize blog posts and articles
- Extract YouTube video transcripts
- Generate knowledge base entries with semantic topics
- Create Logseq-compatible notes

**Tools**:
- `summarize_url`: Summarize any URL (detects YouTube automatically)
- `youtube_transcript`: Extract transcript from YouTube video
- `generate_topics`: Create semantic topic links from text

**Example**:
```
Use the Web Summary skill to summarize https://www.youtube.com/watch?v=dQw4w9WgXcQ
with medium length and include Logseq-style topic tags.
```

**Technical Details**:
- Uses Z.AI service for cost-effective summarization (internal port 9600)
- Falls back to primary Claude API if Z.AI unavailable
- Supports English and auto-detected languages for YouTube
- Caches summaries to avoid redundant requests

### Blender 3D Skill

**Purpose**: Control Blender for 3D modeling and rendering

**Use Cases**:
- Programmatic 3D model creation
- Automated scene generation
- Batch rendering workflows
- Material application and preview

**Tools**:
- `create_object`: Create 3D objects (cube, sphere, camera, light, etc.)
- `apply_material`: Apply PBR materials to objects
- `execute_script`: Run arbitrary Python code in Blender
- `render_scene`: Render current scene
- `import_model`/`export_model`: Import/export 3D files

**Prerequisites**:
- Blender must be running with socket server plugin on port 2800

**Example**:
```
Use the Blender skill to:
1. Create a cube at the origin
2. Create a camera at [5, -5, 5] looking at origin
3. Apply a metallic gold material (metallic=1.0, roughness=0.2)
4. Render to /workspace/output.png at 1920x1080
```

### QGIS Geographic Skill

**Purpose**: Geographic information system operations

**Use Cases**:
- Load and manipulate GIS projects
- Add vector and raster layers
- Run spatial processing algorithms
- Export maps and spatial data

**Tools**:
- `load_project`: Load QGIS project file
- `add_vector_layer`: Add shapefile, GeoJSON, etc.
- `add_raster_layer`: Add raster data
- `run_processing`: Execute QGIS processing tools
- `execute_python`: Run Python in QGIS context

**Prerequisites**:
- QGIS 3.X with socket server plugin on port 2801

**Example**:
```
Use the QGIS skill to create a new project, add a shapefile from
/data/cities.shp, and export the map view to cities_map.png
```

### KiCAD PCB Skill

**Purpose**: Electronic circuit design and PCB layout

**Use Cases**:
- Create schematics programmatically
- Add components and connections
- Generate netlists and BOMs
- Manage PCB layouts

**Tools**:
- `create_project`: New KiCAD project
- `add_component`: Place component in schematic
- `connect_pins`: Wire components together
- `search_library`: Find components
- `generate_netlist`: Create netlist
- `export_bom`: Generate bill of materials

**Example**:
```
Use the KiCAD skill to create a simple LED circuit:
1. Create new project "led_blinker"
2. Add components: 555 timer, LED, resistor, capacitor
3. Connect according to standard 555 astable circuit
4. Generate BOM
```

### ImageMagick Skill

**Purpose**: Image processing and manipulation

**Use Cases**:
- Format conversion (PNG, JPEG, WebP, etc.)
- Resize, crop, rotate images
- Apply filters and effects
- Batch process multiple images
- Create thumbnails and montages

**Tools**:
- `convert`: Convert format or apply transformations
- `resize`: Resize to specific dimensions
- `crop`: Crop to region
- `composite`: Combine images
- `montage`: Create image grid
- `identify`: Get image metadata

**Example**:
```
Use the ImageMagick skill to:
1. Convert all PNG files in /input/ to JPEG
2. Resize to 800x600 maintaining aspect ratio
3. Save to /output/ with 85% quality
```

### PBR Rendering Skill

**Purpose**: Physically-based rendering material generation

**Use Cases**:
- Generate PBR texture sets
- Create material previews
- Batch material library creation
- Export to various PBR formats

**Tools**:
- `generate_material`: Create PBR material textures
- `preview_material`: Render material preview sphere
- `batch_materials`: Generate multiple materials
- `export_textures`: Export albedo, metallic, roughness, normal maps

**Example**:
```
Use the PBR Rendering skill to generate a worn metal material with:
- Base color: #8c8c8c
- Metallic: 0.9
- Roughness: 0.6 with texture variation
- Include normal map for surface detail
Export all maps to /materials/worn_metal/
```

## Skill Development

### Creating a Custom Skill

1. **Create directory structure**:
```bash
mkdir -p ~/.claude/skills/my-skill/tools
cd ~/.claude/skills/my-skill
```

2. **Create SKILL.md** with YAML frontmatter:
```markdown
---
name: My Custom Skill
description: Brief description that helps Claude know when to use this skill (max 200 chars)
---

# My Custom Skill

## Capabilities

List what this skill can do.

## When to Use This Skill

Describe scenarios where Claude should invoke this skill.

## Instructions

Provide clear step-by-step guidance for Claude.

## Tool Functions

### `tool_name`
Description of what this tool does.

Parameters:
- `param1` (required): Description
- `param2` (optional): Description (default: value)

## Examples

### Example 1: Use Case
```
Example of how to invoke the skill
```
```

3. **Create tool implementation** (Python or Node.js):

**Python (`tools/my_tool.py`)**:
```python
#!/usr/bin/env python3
import sys
import json

def handle_request(request):
    tool = request.get("tool")
    params = request.get("params", {})

    # Implement your tool logic
    result = {
        "success": True,
        "data": "result data"
    }

    return result

def main():
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            result = handle_request(request)
            response = {"result": result}
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            error_response = {"error": str(e)}
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
```

**Node.js (`tools/my_tool.js`)**:
```javascript
#!/usr/bin/env node
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  try {
    const request = JSON.parse(line);
    const tool = request.tool;
    const params = request.params || {};

    // Implement your tool logic
    const result = {
      success: true,
      data: 'result data'
    };

    const response = { result };
    console.log(JSON.stringify(response));
  } catch (error) {
    const errorResponse = { error: error.message };
    console.log(JSON.stringify(errorResponse));
  }
});
```

4. **Make executable**:
```bash
chmod +x tools/my_tool.py
```

5. **Test the skill**:
```bash
# Manual test
echo '{"tool":"my_tool","params":{}}' | ./tools/my_tool.py

# Use in Claude Code
claude
> Use my-skill to do something
```

### Best Practices

#### SKILL.md Guidelines

1. **Name**: Max 64 characters, human-friendly
2. **Description**: Max 200 characters, focus on WHEN to use
3. **Instructions**: Clear, step-by-step, actionable
4. **Examples**: Show concrete use cases
5. **Tool Functions**: Document all parameters with types

#### Tool Implementation

1. **Error Handling**: Always catch exceptions and return structured errors
2. **Input Validation**: Validate all parameters before use
3. **Output Format**: Consistent JSON structure
4. **Timeouts**: Handle long-running operations gracefully
5. **Logging**: Log to stderr, not stdout (stdout is for JSON responses)

#### Progressive Disclosure

1. **SKILL.md**: Keep concise, focus on "when" and "what"
2. **REFERENCE.md** (optional): Detailed API documentation
3. **Tools**: Implement actual functionality
4. **Examples**: Show common patterns

### Skill Structure Reference

```
~/.claude/skills/my-skill/
├── SKILL.md           # Required: Skill definition with YAML frontmatter
├── REFERENCE.md       # Optional: Detailed documentation
├── tools/             # Tool implementations
│   ├── __init__.py    # Python package init (if using Python)
│   ├── tool1.py       # Tool script (Python)
│   ├── tool2.js       # Tool script (Node.js)
│   └── helpers.py     # Shared utilities
├── examples/          # Example usage
│   ├── example1.md
│   └── example2.md
└── tests/             # Optional: Tests
    └── test_tools.py
```

## Migration from MCP

### Key Differences

| Aspect | MCP | Claude Code Skills |
|--------|-----|-------------------|
| Architecture | HTTP server | stdio process |
| Loading | Always loaded | On-demand |
| Communication | JSON over HTTP | JSON over stdin/stdout |
| Resource usage | Persistent server | Spawn when needed |
| Configuration | mcp.json | SKILL.md |
| Discovery | Port-based | Progressive disclosure |

### Migration Steps

1. **Convert MCP server to stdio tool**:
   - Remove HTTP server code
   - Read from stdin, write to stdout
   - Keep core logic intact

2. **Create SKILL.md**:
   - Extract description from MCP server docs
   - Convert tool definitions to SKILL.md format
   - Add examples

3. **Test**:
   - Verify stdio communication
   - Test with Claude Code
   - Validate error handling

### Example Migration

**Before (MCP Server)**:
```python
from mcp import Server

app = Server()

@app.tool("my_tool")
def my_tool(param1: str):
    return {"result": f"processed {param1}"}

app.run(port=9876)
```

**After (Claude Code Skill Tool)**:
```python
#!/usr/bin/env python3
import sys
import json

def my_tool(param1):
    return {"result": f"processed {param1}"}

def handle_request(request):
    tool = request.get("tool")
    if tool == "my_tool":
        return my_tool(request["params"]["param1"])
    return {"error": f"Unknown tool: {tool}"}

for line in sys.stdin:
    request = json.loads(line.strip())
    result = handle_request(request)
    print(json.dumps({"result": result}))
    sys.stdout.flush()
```

## Troubleshooting

### Skill not loading

```bash
# List all skills
skill-list

# Check SKILL.md format
head -n 10 ~/.claude/skills/my-skill/SKILL.md

# Verify YAML frontmatter
grep -A 3 "^---" ~/.claude/skills/my-skill/SKILL.md
```

### Tool not executing

```bash
# Check permissions
ls -l ~/.claude/skills/my-skill/tools/*.py

# Make executable
chmod +x ~/.claude/skills/my-skill/tools/*.py

# Test manually
echo '{"tool":"test","params":{}}' | ~/.claude/skills/my-skill/tools/test.py
```

### JSON parsing errors

```bash
# Test with python
echo '{"tool":"test"}' | python3 ~/.claude/skills/my-skill/tools/test.py

# Check for print statements (should only output JSON)
grep -n "print" ~/.claude/skills/my-skill/tools/test.py
```

### Skill not found by Claude

1. Check SKILL.md location: `~/.claude/skills/<skill-name>/SKILL.md`
2. Verify YAML frontmatter has `name` and `description`
3. Ensure description is clear about when to use (max 200 chars)
4. Restart Claude Code to reload skills

## Advanced Topics

### Multi-File Skills

For complex skills, organize code:

```
tools/
├── __init__.py
├── main.py          # Entry point
├── operations/
│   ├── __init__.py
│   ├── create.py
│   ├── update.py
│   └── delete.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

### Async Operations

For long-running operations:

```python
import asyncio

async def long_operation(param):
    await asyncio.sleep(5)
    return {"result": "done"}

# In main loop
result = asyncio.run(long_operation(params))
```

### External Dependencies

Install in skill directory:

```bash
cd ~/.claude/skills/my-skill
python -m venv venv
source venv/bin/activate
pip install requests beautifulsoup4
```

Update shebang:
```python
#!/home/devuser/.claude/skills/my-skill/venv/bin/python3
```

### Caching Results

```python
import json
import hashlib

CACHE_DIR = "/home/devuser/.cache/my-skill"

def cache_key(params):
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

def get_cached(params):
    key = cache_key(params)
    cache_file = f"{CACHE_DIR}/{key}.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return None

def set_cached(params, result):
    key = cache_key(params)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(f"{CACHE_DIR}/{key}.json", "w") as f:
        json.dump(result, f)
```

## Resources

- [Claude Code Documentation](https://docs.claude.com/claude-code)
- [Anthropic Skills Repository](https://github.com/anthropics/skills)
- [Skill Examples](https://github.com/anthropics/claude-cookbooks/tree/main/skills)

## Contributing Skills

Share your skills with the community:

1. Test thoroughly
2. Document clearly
3. Add examples
4. Submit to [anthropics/skills](https://github.com/anthropics/skills)
