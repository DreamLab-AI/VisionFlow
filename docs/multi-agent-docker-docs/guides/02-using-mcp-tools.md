# Guide: Using MCP Tools

This guide explains how to interact with the Model Context Protocol (MCP) tools from within the `multi-agent-container`. The primary interface for this is the `mcp-helper.sh` script.

### The `mcp-helper.sh` Script

This script is your main entry point for managing and testing MCP tools. It automatically uses the local `./.mcp.json` configuration file, ensuring you're always interacting with the correct set of tools.

**Location**: `/workspace/mcp-helper.sh` (copied by `setup-workspace.sh`)

#### Common Commands

**1. List All Available Tools**
To see a list of all tools registered in your workspace's `.mcp.json` file:
```bash
./mcp-helper.sh list-tools
```

**2. Test a Specific Tool**
To check if a tool is correctly configured and available to the `claude-flow` orchestrator:
```bash
./mcp-helper.sh test-tool imagemagick-mcp
```

**3. Run a Tool Manually**
To execute a tool with a specific JSON payload, use the `run-tool` command. This is excellent for debugging and manual testing.

```bash
# Syntax: ./mcp-helper.sh run-tool <tool-name> '<json-payload>'

# Example: Create a 200x200 gold square using the ImageMagick tool
./mcp-helper.sh run-tool imagemagick-mcp '{
  "method": "create",
  "params": {
    "width": 200,
    "height": 200,
    "color": "gold",
    "output": "gold_square.png"
  }
}'
```

**4. Run Automated Tests**
The helper includes simple, built-in tests for some of the core tools.
```bash
# Run a quick test for the ImageMagick tool
./mcp-helper.sh test-imagemagick

# Run all available automated tests
./mcp-helper.sh test-all
```

### Interacting with Bridge Tools

Bridge tools (`blender-mcp`, `qgis-mcp`, `pbr-generator-mcp`) communicate with the `gui-tools-container` over the network. You use them the same way as direct tools, but the execution happens remotely.

**Example: Create a Cube in Blender**
This command sends Python code to be executed within Blender's context.
```bash
./mcp-helper.sh run-tool blender-mcp '{
  "tool": "execute_code",
  "params": {
    "code": "import bpy; bpy.ops.mesh.primitive_cube_add()"
  }
}'
```
To see the result, connect to the VNC session at `localhost:5901` on your host machine.

**Example: Generate PBR Textures**
This command requests the PBR generator service to create a set of textures.
```bash
./mcp-helper.sh run-tool pbr-generator-mcp '{
  "tool": "generate_material",
  "params": {
    "material": "brushed_metal",
    "resolution": "1024x1024",
    "output": "./pbr_textures"
  }
}'
```
The generated textures will appear in the `/workspace/pbr_textures` directory.

### For AI Agents: Using `claude-flow` Directly

While `mcp-helper.sh` is great for humans, AI agents will typically invoke `claude-flow` directly. It's crucial to always use the `--file ./.mcp.json` flag to ensure the local workspace configuration is used.

```bash
# Correct way for an agent to list tools
./node_modules/.bin/claude-flow mcp tools --file ./.mcp.json

# Correct way for an agent to run a tool
echo '{"method":"create", "params":{...}}' | ./node_modules/.bin/claude-flow mcp tool imagemagick-mcp --file ./.mcp.json
```

### Performance Tips

1. **Batch Operations**: When working with multiple files, batch your operations to reduce overhead.
2. **Use TCP for High-Volume**: For high-frequency tool calls, connect directly to the TCP server on port 9500.
3. **Monitor Resources**: GUI tools consume more resources. Use `multi-agent status` to check health.

### Common Tool Patterns

**Image Processing with ImageMagick**
```bash
# Resize an image
./mcp-helper.sh run-tool imagemagick-mcp '{
  "method": "resize",
  "params": {
    "input": "source.jpg",
    "output": "thumbnail.jpg",
    "width": 150,
    "height": 150
  }
}'

# Convert format
./mcp-helper.sh run-tool imagemagick-mcp '{
  "method": "convert",
  "params": {
    "input": "image.png",
    "output": "image.jpg"
  }
}'
```

**3D Modeling with Blender**
```bash
# Import a model
./mcp-helper.sh run-tool blender-mcp '{
  "tool": "import_model",
  "params": {
    "filepath": "/workspace/model.obj",
    "format": "obj"
  }
}'

# Export the scene
./mcp-helper.sh run-tool blender-mcp '{
  "tool": "export_scene",
  "params": {
    "filepath": "/workspace/exported_scene.gltf",
    "format": "gltf"
  }
}'
```

**Geospatial Analysis with QGIS**
```bash
# Load a shapefile
./mcp-helper.sh run-tool qgis-mcp '{
  "tool": "load_vector",
  "params": {
    "filepath": "/workspace/boundaries.shp",
    "name": "City Boundaries"
  }
}'

# Perform buffer analysis
./mcp-helper.sh run-tool qgis-mcp '{
  "tool": "buffer",
  "params": {
    "layer": "City Boundaries",
    "distance": 1000,
    "units": "meters"
  }
}'
```

### Debugging Tool Failures

If a tool fails:

1. **Check JSON Syntax**: Ensure your JSON payload is valid
2. **Verify Parameters**: Check the tool's documentation for required parameters
3. **Review Logs**: Tool output and errors are logged to `/app/mcp-logs/`
4. **Test Connectivity**: For bridge tools, use `nc -zv gui-tools-service <port>`
5. **Check Service Status**: Use `multi-agent services` to verify all services are running