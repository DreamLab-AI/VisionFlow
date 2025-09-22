# GUI Tools Guide

The GUI container provides powerful visual tools accessible through VNC and MCP protocols.

## Overview

All GUI tools run in a dedicated container with:
- Full desktop environment (XFCE)
- VNC access on port 5901
- GPU acceleration support
- MCP integration for programmatic control

## Accessing GUI Tools

### VNC Connection

```bash
# Using VNC viewer
vncviewer localhost:5901

# Using web browser (if noVNC is set up)
http://localhost:6080

# Password: none required by default
```

### Available Tools

1. **Blender 4.5.1** - 3D modeling and animation
2. **QGIS** - Geographic Information System
3. **PBR Generator** - Material texture generation
4. **Playwright** - Browser automation (visual mode)

## Blender MCP (Port 9876)

### Features
- Create and manipulate 3D objects
- Python scripting integration
- Render scenes programmatically
- Import/export various formats

### Usage Examples

```python
# Through Claude Code
claude
> Use mcp__blender__create_cube to create a 3D cube
> Use mcp__blender__render_scene to render the current scene

# Direct TCP access
echo '{"jsonrpc":"2.0","id":"1","method":"tools/call","params":{"name":"create_sphere","arguments":{"radius":2.0}}}' | nc localhost 9876
```

### Common Operations
- `create_cube`, `create_sphere`, `create_cylinder`
- `set_material`, `add_texture`
- `animate_object`, `keyframe_insert`
- `render_scene`, `export_model`

## QGIS MCP (Port 9877)

### Features
- Load and visualize geographic data
- Perform spatial analysis
- Create professional maps
- Process raster and vector data

### Usage Examples

```python
# Load a shapefile
claude
> Use mcp__qgis__load_layer with path "/workspace/data/cities.shp"

# Perform buffer analysis
> Use mcp__qgis__buffer_analysis with layer "cities" and distance 1000
```

### Common Operations
- `load_layer`, `save_project`
- `buffer_analysis`, `intersection`
- `create_heatmap`, `style_layer`
- `export_map`, `print_layout`

## PBR Generator (Port 9878)

### Features
- Generate physically based rendering textures
- Create seamless materials
- Tessellation patterns
- Normal map generation

### Usage Examples

```bash
# Generate wood texture
claude
> Use mcp__pbr-generator__create_material with type "wood" and size 2048

# Create metal surface
> Use mcp__pbr-generator__create_material with type "metal" roughness 0.3
```

### Material Types
- Wood, Metal, Stone, Fabric
- Concrete, Plastic, Glass
- Custom parametric materials

## Playwright Visual Browser (Port 9879)

### Features
- Real-time browser visualization
- Interactive debugging
- Multi-browser support
- Screenshot and video recording

### Usage Examples

```javascript
// Navigate to page
claude
> Use mcp__playwright__navigate to "https://example.com"

// Click element
> Use mcp__playwright__click on selector "#submit-button"

// Take screenshot
> Use mcp__playwright__screenshot with fullPage true
```

### Visual Benefits
- See exactly what automation is doing
- Debug complex selectors
- Record test execution
- Handle popups and dialogs visually

## Tips for Using GUI Tools

### Performance Optimization

1. **GPU Acceleration**: Enabled by default for better performance
2. **Resolution**: VNC runs at 1920x1080 by default
3. **Resource Limits**: Monitor CPU/memory usage

### File Management

```bash
# Shared directories
/workspace      # Main workspace (shared with main container)
/blender-files  # Blender projects
/workspace/pbr_outputs  # Generated materials
```

### Keyboard Shortcuts

When connected via VNC:
- `Ctrl+Alt+Shift` - Toggle fullscreen
- `F8` - VNC menu
- Standard application shortcuts work normally

## Troubleshooting

### VNC Connection Issues

```bash
# Check if VNC is running
docker exec gui-tools-container ps aux | grep vnc

# View VNC logs
docker logs gui-tools-container | grep vnc
```

### Tool Not Responding

```bash
# Check tool status
nc -zv localhost 9876  # Blender
nc -zv localhost 9877  # QGIS
nc -zv localhost 9878  # PBR
nc -zv localhost 9879  # Playwright

# Restart GUI container
docker-compose restart gui-tools-service
```

### Display Issues

```bash
# Inside GUI container
export DISPLAY=:1
xhost +local:

# Check display
echo $DISPLAY
```

## Advanced Usage

### Custom Scripts

Place scripts in shared directories:
- Blender: `/workspace/blender-scripts/`
- QGIS: `/workspace/qgis-scripts/`

### Automation Workflows

```python
# Example: Automated 3D scene generation
1. Generate textures with PBR Generator
2. Create 3D models in Blender
3. Apply generated materials
4. Render final scene
```

### Integration with Main Container

```bash
# From main container, trigger GUI operations
echo '{"method":"create_scene","params":{"objects":["cube","sphere"]}}' | nc gui-tools-service 9876

# Process results
ls /workspace/renders/
```

## Best Practices

1. **Use VNC for Setup**: Configure tools visually first
2. **Automate Repetitive Tasks**: Use MCP for batch operations
3. **Monitor Resources**: GUI tools can be memory-intensive
4. **Save Frequently**: Use shared directories for persistence
5. **Close Unused Tools**: Free resources when not needed

## Next Steps

- Connect to VNC and explore the desktop
- Try creating a 3D model in Blender
- Load geographic data in QGIS
- Test browser automation with visual feedback